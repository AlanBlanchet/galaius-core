"""Shared syntax for model criteria stored by every Interact consumer."""

from __future__ import annotations

import re
import math
from typing import Literal

from .wire import WireModel


class CriteriaError(ValueError):
    """A criterion cannot be parsed or validated."""


class CriteriaClause(WireModel):
    kind: Literal["comparison", "bare", "provider"]
    name: str
    operator: str = ""
    value: float | None = None
    percentile: bool = False


_TERM = re.compile(r"^\s*([\w.-]+)\s*(>=|<=|==|=|>|<)\s*(-?\d+(?:\.\d+)?%?)\s*$")
_BARE = re.compile(r"^\s*([\w.-]+)\s*$")
_PROVIDER = re.compile(r"^\s*provider\s*(=|==|!=|~)\s*([a-z][a-z0-9_-]*)\s*$")


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
