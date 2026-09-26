"""Validation shared by immutable Interact wire contracts.

Direction decides strictness. A peer is upgraded independently of the one it talks to, so a
model READ from the other side (every response, every record read back) keeps the fields its
reader does not know (`WireModel`, `extra="allow"`): an additive server field never breaks a
released client, and a record a client edits and writes back keeps the fields it could not read.
A model only a client AUTHORS and sends (`WireRequest`) refuses an unknown field: there it is a
typo. A server validates every inbound body with `extra="forbid"`, whatever its model.

A server that still answers clients released before this rule applies the older release's
`ContractView` around such a request and renders each response through `ContractView.render`.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict


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

    def unexpressed(self, value: object) -> tuple[str, ...]:
        """`Class.field` for every hidden field reachable in `value`: what a sender on this
        release could not have set, so writing its body over a stored record would reset them."""
        found: list[str] = []
        self._collect(value, found)
        return tuple(dict.fromkeys(found))

    def _collect(self, value: object, found: list[str]) -> None:
        if isinstance(value, BaseModel):
            found.extend(f"{type(value).__name__}.{name}" for name in sorted(self.hidden(type(value))))
            for name in type(value).model_fields:
                self._collect(getattr(value, name), found)
        elif isinstance(value, (tuple, list, set, frozenset)):
            for item in value:
                self._collect(item, found)
        elif isinstance(value, dict):
            for item in value.values():
                self._collect(item, found)

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
