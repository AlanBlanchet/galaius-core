"""Validation shared by immutable Interact wire contracts.

Direction decides strictness. A peer is upgraded independently of the one it talks to, so a
model READ from the other side (every response, every record read back) keeps the fields its
reader does not know (`WireModel`, `extra="allow"`): an additive server field never breaks a
released client, and a record a client edits and writes back keeps the fields it could not read.
A model only a client AUTHORS and sends (`WireRequest`) refuses an unknown field: there it is a
typo. A server validates every inbound body with `extra="forbid"`, whatever its model.

A server that still answers clients released before this rule applies the older release's
`ContractView` around such a request, renders each response through `ContractView.render`, and
stores such a client's write through `ContractView.carry`.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel, ConfigDict

M = TypeVar("M")


class ContractDump(dict[str, Any]):
    """A `WireModel.model_dump` taken while a `ContractView` is applied: the same content (what is
    stored or hashed never changes), still tied to the model it came from so a response boundary
    can project it."""

    __slots__ = ("source",)

    def __init__(self, content: dict[str, Any], source: BaseModel) -> None:
        super().__init__(content)
        self.source = source


class ContractView(BaseModel):
    """The fields an older contract release declares, per model class name.

    A class the release lacks passes through whole (its reader cannot parse it anyway); a
    current field the release lacks is dropped unless the current contract requires it — a
    required field that release lacks is a rename, which no projection can repair and dropping
    would break the readers built after it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    release: str
    fields: Mapping[str, frozenset[str]]

    _active: ClassVar[ContextVar["ContractView | None"]] = ContextVar("interact_contract_view", default=None)

    @classmethod
    def active(cls) -> "ContractView | None":
        return cls._active.get()

    @contextmanager
    def applied(self) -> Iterator["ContractView"]:
        token = self._active.set(self)
        try:
            yield self
        finally:
            self._active.reset(token)

    def hidden(self, model: type[BaseModel]) -> frozenset[str]:
        """Field names of `model` this release cannot express."""
        declared = self.fields.get(model.__name__)
        if declared is None:
            return frozenset()
        return frozenset(name for name, field in model.model_fields.items() if name not in declared and not field.is_required())

    def render(self, content: Any) -> Any:
        """A response body as this release's reader expects it: every `ContractDump` inside is
        projected; everything else passes through."""
        if isinstance(content, ContractDump):
            return self.project(content.source, content)
        if isinstance(content, dict):
            return {key: self.render(value) for key, value in content.items()}
        if isinstance(content, (list, tuple)):
            return [self.render(value) for value in content]
        return content

    def project(self, model: BaseModel, data: dict[str, Any]) -> dict[str, Any]:
        """`data` (a dump of `model`) as this release's reader expects it, at every depth."""
        fields = type(model).model_fields
        known = type(model).__name__ in self.fields
        hidden = self.hidden(type(model))
        names = {name: name for name in fields} | {
            key: name for name, field in fields.items() if (key := field.serialization_alias or field.alias)
        }
        projected: dict[str, Any] = {}
        for key, value in data.items():
            name = names.get(key)
            if name is None:
                if not known:
                    projected[key] = value
            elif name not in hidden:
                projected[key] = self._project_value(getattr(model, name), value)
        return model.projected(projected) if isinstance(model, WireModel) else projected

    def carry(self, value: M, parents: Mapping[type[BaseModel], Callable[[Any], BaseModel | None]]) -> tuple[M, tuple[str, ...]]:
        """A write from a sender on this release as the server stores it, and `Class.field` for
        each field that write would still reset.

        The sender cannot express a hidden field, so one it did not send (absent from
        `model_fields_set`) is taken from the record the write replaces: for a revision whose type
        `parents` names, the stored parent revision its resolver returns; below it, at every depth,
        the same field, the sequence item with the same `id` (lacking one, the item this release
        sees as equal), the same mapping key. Inside such a revision a field with no stored
        counterpart keeps its default: a new record or item holds nothing to reset. Outside one
        the field is named: writing it would reset what the server holds."""
        missed: list[str] = []
        return self._carry(value, None, False, parents, missed), tuple(dict.fromkeys(missed))

    def _carry(self, value: Any, stored: Any, covered: bool, parents: Mapping[type[BaseModel], Callable[[Any], BaseModel | None]], missed: list[str]) -> Any:
        if isinstance(value, BaseModel):
            model = type(value)
            if (resolve := parents.get(model)) is not None:
                covered, stored = True, None if getattr(value, "parent_revision", None) is None else resolve(value)
            if type(stored) is not model:
                stored = None
            update: dict[str, Any] = {}
            for name in sorted(self.hidden(model) - value.model_fields_set):
                if stored is not None:
                    update[name] = getattr(stored, name)
                elif not covered:
                    missed.append(f"{model.__name__}.{name}")
            for name in model.model_fields:
                if name not in update and (carried := self._carry(child := getattr(value, name), getattr(stored, name, None), covered, parents, missed)) is not child:
                    update[name] = carried
            return value.model_copy(update=update) if update else value
        if isinstance(value, (tuple, list)):
            pool = stored if isinstance(stored, (tuple, list)) else ()
            items = [self._carry(item, self._counterpart(item, pool), covered, parents, missed) for item in value]
            return value if all(new is old for new, old in zip(items, value)) else type(value)(items)
        if isinstance(value, dict):
            pool = stored if isinstance(stored, dict) else {}
            entries = {key: self._carry(item, pool.get(key), covered, parents, missed) for key, item in value.items()}
            return value if all(entries[key] is item for key, item in value.items()) else entries
        if isinstance(value, (set, frozenset)):  # unordered, no identity to match: only named
            for item in value:
                self._carry(item, None, covered, parents, missed)
        return value

    def _counterpart(self, item: Any, pool: Sequence[Any]) -> Any:
        """The stored item `item` replaces: the one with its `id`, else the one this release sees
        as equal (its hidden fields dropped)."""
        if not isinstance(item, BaseModel):
            return None
        candidates = [stored for stored in pool if type(stored) is type(item)]
        if "id" in type(item).model_fields:
            return next((stored for stored in candidates if stored.id == item.id), None)  # type: ignore[attr-defined]
        seen = self.project(item, item.model_dump(mode="json"))
        return next((stored for stored in candidates if self.project(stored, stored.model_dump(mode="json")) == seen), None)

    def _project_value(self, value: object, data: Any) -> Any:
        if isinstance(value, BaseModel) and isinstance(data, dict):
            return self.project(value, data)
        if isinstance(value, (tuple, list)) and isinstance(data, (tuple, list)) and len(value) == len(data):
            return [self._project_value(item, dumped) for item, dumped in zip(value, data)]
        if isinstance(value, dict) and isinstance(data, dict) and len(value) == len(data):
            return {key: self._project_value(item, dumped) for item, (key, dumped) in zip(value.values(), data.items())}
        return data


class WireModel(BaseModel):
    """A contract read from the other side: unknown fields are kept (see module docstring)."""

    model_config = ConfigDict(extra="allow", frozen=True)

    def model_dump(self, **options: Any) -> dict[str, Any]:
        data = super().model_dump(**options)
        return data if ContractView.active() is None else ContractDump(data, self)

    @classmethod
    def projected(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Content a projection of this model's dump must re-derive — a digest sealing the
        fields it covers. None by default."""
        return data


class WireRequest(WireModel):
    """A contract only a client authors and sends: an unknown field is refused."""

    model_config = ConfigDict(extra="forbid", frozen=True)
