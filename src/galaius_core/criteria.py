"""Shared syntax for model criteria stored by every Galaius consumer."""

from __future__ import annotations

import re
import math
from typing import Literal, Self, get_args

from pydantic import Field, model_validator

from .wire import WireModel


class CriteriaError(ValueError):
    """A criterion cannot be parsed or validated."""


#: The numeric comparators every consumer shares (a model node's constraints, an agent's criteria).
ModelComparator = Literal[">", ">=", "<", "<=", "=", "=="]
COMPARATORS: tuple[str, ...] = get_args(ModelComparator)
_PROVIDER_OPERATORS = ("=", "==", "!=", "~")
#: Every operator a clause may carry: a comparator, a provider mode, `in` for a word list, none.
CriteriaOperator = Literal["", ">", ">=", "<", "<=", "=", "==", "!=", "~", "in"]


class CriteriaClause(WireModel):
    """One clause of the shared criteria grammar, parsed. `membership` (`sovereignty in
    self_hosted|eu_sovereign`) is the one clause over a WORD-valued property: its `values` are the
    accepted words, never numbers."""

    kind: Literal["comparison", "bare", "provider", "membership"]
    name: str = Field(min_length=1, max_length=160, pattern=r"^[\w.-]+$")
    operator: CriteriaOperator = ""
    value: float | None = None
    percentile: bool = False
    values: tuple[str, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        expected = {"comparison": COMPARATORS, "bare": ("",), "provider": _PROVIDER_OPERATORS, "membership": ("in",)}[self.kind]
        if self.operator not in expected:
            raise ValueError(f"{self.kind} clause cannot use {self.operator!r}")
        if (self.kind == "comparison") != (self.value is not None and math.isfinite(self.value)):
            raise ValueError("a comparison, and only a comparison, carries a finite value")
        if (self.kind == "membership") != bool(self.values) or any(not _WORD.fullmatch(word) for word in self.values):
            raise ValueError("a membership clause, and only one, lists accepted words")
        return self

    def __str__(self) -> str:
        if self.kind == "bare":
            return self.name
        if self.kind == "membership":
            return f"{self.name} in {'|'.join(self.values)}"
        if self.kind == "provider":
            return f"provider {self.operator} {self.name}"
        return f"{self.name} {self.operator} {self.value:g}{'%' if self.percentile else ''}"


class CriteriaWeight(WireModel):
    """One `name=weight` ranking clause: a weight ORDERS what already passed, never admits."""

    name: str = Field(min_length=1, max_length=160, pattern=r"^[\w-]+\.[\w.-]+$")
    weight: float = Field(gt=0, allow_inf_nan=False)

    def __str__(self) -> str:
        return f"{self.name}={self.weight:g}"


def format_criteria(clauses: tuple[CriteriaClause, ...]) -> str:
    """The text an agent stores for these clauses: `parse_criteria` read backwards."""
    return " and ".join(str(clause) for clause in clauses)


def format_criteria_weights(weights: tuple[CriteriaWeight, ...]) -> str:
    return ",".join(str(weight) for weight in weights)


_TERM = re.compile(r"^\s*([\w.-]+)\s*(>=|<=|==|=|>|<)\s*(-?\d+(?:\.\d+)?%?)\s*$")
_BARE = re.compile(r"^\s*([\w.-]+)\s*$")
_PROVIDER = re.compile(r"^\s*provider\s*(=|==|!=|~)\s*([a-z][a-z0-9_-]*)\s*$")
_MEMBERSHIP = re.compile(r"^\s*([\w.-]+)\s+in\s+([\w.-]+(?:\s*\|\s*[\w.-]+)*)\s*$")
_WORD = re.compile(r"^[\w.-]{1,80}$")


def parse_criteria(text: str) -> tuple[CriteriaClause, ...]:
    """Parse criterion clause syntax without consulting a model catalog."""
    if not text or not text.strip():
        raise CriteriaError("an empty criterion selects nothing — say what you want")

    clauses: list[CriteriaClause] = []
    for raw in re.split(r"\s+and\s+|,", text):
        clause = raw.strip()
        if not clause:
            continue
        if clause.split(None, 1)[0] == "provider":
            match = _PROVIDER.fullmatch(raw)
            if match is None:
                raise CriteriaError(
                    f"{clause!r} is not a usable provider constraint — write "
                    "'provider = <name>' (REQUIRE), 'provider != <name>' (EXCLUDE), "
                    "or 'provider ~ <name>' (PREFER)"
                )
            clauses.append(CriteriaClause(kind="provider", name=match.group(2), operator=match.group(1)))
        elif match := _MEMBERSHIP.fullmatch(raw):
            clauses.append(CriteriaClause(kind="membership", name=match.group(1), operator="in", values=tuple(word.strip() for word in match.group(2).split("|"))))
        elif match := _TERM.fullmatch(raw):
            name, operator, written = match.groups()
            percentile = written.endswith("%")
            value = float(written[:-1] if percentile else written)
            if percentile and not 0 < value <= 100:
                raise CriteriaError(
                    f"{written!r} is a position in the field, so it must be between 0 and "
                    "100 — '90%' is the top tenth"
                )
            clauses.append(CriteriaClause(
                kind="comparison", name=name, operator=operator,
                value=value, percentile=percentile,
            ))
        elif match := _BARE.fullmatch(raw):
            clauses.append(CriteriaClause(kind="bare", name=match.group(1)))
        else:
            raise CriteriaError(f"{clause!r} is not a criterion — write it as 'name > number'")

    if not clauses:
        raise CriteriaError("an empty criterion selects nothing — say what you want")
    return tuple(clauses)


def parse_criteria_weights(text: str, benchmark_names: set[str] | None = None) -> dict[str, float]:
    """Parse normalized benchmark weights, checking names when catalog is available."""
    if not text.strip():
        return {}
    weights: dict[str, float] = {}
    total = 0.0

    for clause in text.split(","):
        name, separator, raw = clause.partition("=")
        name = name.strip()
        if (
            not separator
            or "." not in name
            or (benchmark_names is not None and name not in benchmark_names)
        ):
            raise CriteriaError(
                f"{name!r} is not a normalized benchmark; raw price, latency, and index units cannot be weighted"
            )
        try:
            value = float(raw)
        except ValueError as error:
            raise CriteriaError(f"weight for {name!r} is not a number") from error
        if not math.isfinite(value) or value < 0:
            raise CriteriaError(f"weight for {name!r} must be finite and non-negative")
        if name in weights:
            raise CriteriaError(f"duplicate weight for {name!r}")
        weights[name] = value
        total += value
    if total <= 0:
        raise CriteriaError("criteria weights must have a positive total")
    return {name: value / total for name, value in weights.items()}
