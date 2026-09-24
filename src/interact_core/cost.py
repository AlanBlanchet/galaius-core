"""What running a workflow node COSTS, in USD — typed contracts only, no pricing DATA here.

Every price is bound to a SOURCE, never hand-typed: `PriceSource.kind` names which one —
`model_registry` (interact's litellm-derived model registry: token prices), `research_table` (a
sourced, dated table of per-image/video-second/audio-second/call vendor prices), `machine_rate`
(the workspace owner's own $/GPU-second setting for local runs), or `builtin_free` (a node that
reaches nothing billable — a transform, a connector call, a free node). A node whose source has no
figure for it stays `known=False` and carries NO price/range: "unknown", never a guessed number
standing in for one, and never a silent zero.
"""

from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .wire import WireModel

#: The billable dimensions this registry knows how to price and meter. One workflow node's usage
#: is expressed in these units; a price is quoted per unit, in USD.
CostUnit = Literal[
    "token_in", "token_out", "image", "video_second", "audio_second", "call", "gpu_second", "cpu_second", "byte",
]

PriceSourceKind = Literal["model_registry", "research_table", "machine_rate", "builtin_free"]


class PriceSource(WireModel):
    """What a price is bound to, shown verbatim beside the number it justifies."""

    kind: PriceSourceKind
    #: A citation a reader can act on: a registry provider/model id, a research file name +
    #: retrieval date, "workspace machine rate", or the free-builtin's own name.
    reference: str = Field(min_length=1, max_length=240)
    #: ISO date the source was retrieved/priced, when the source names one (a research table
    #: entry does; the live model registry and an owner-set rate do not).
    retrieved: str | None = Field(default=None, max_length=40)


class UnitPrice(WireModel):
    unit: CostUnit
    usd_per_unit: float = Field(ge=0)


class NodeCostModel(WireModel):
    """One node's PRICE (never its usage): what each of its billable units costs, resolved from
    a bound source. `known=False` means the bound source has no figure for this node yet."""

    node_id: UUID
    known: bool
    prices: tuple[UnitPrice, ...] = ()
    source: PriceSource | None = None
    #: Why the price is unknown ("no research-table entry for gemini/veo-3 yet"), or a caveat on
    #: a known price ("estimate assumes one agent turn").
    reason: str | None = Field(default=None, max_length=240)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.known and (not self.prices or self.source is None):
            raise ValueError("a known node price names its unit prices and their source")
        if not self.known and self.prices:
            raise ValueError("an unknown node price carries no unit prices")
        return self


class NodeUsage(WireModel):
    """Metered units for one node's run. Every field independently optional: a unit no meter
    reported for this node stays unset, never a synthetic zero standing in for "not measured"."""

    token_in: int | None = Field(default=None, ge=0)
    token_out: int | None = Field(default=None, ge=0)
    images: int | None = Field(default=None, ge=0)
    video_seconds: float | None = Field(default=None, ge=0)
    audio_seconds: float | None = Field(default=None, ge=0)
    calls: int | None = Field(default=None, ge=0)
    gpu_seconds: float | None = Field(default=None, ge=0)
    cpu_seconds: float | None = Field(default=None, ge=0)
    bytes: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def at_least_one_figure(self) -> Self:
        fields = ("token_in", "token_out", "images", "video_seconds", "audio_seconds", "calls", "gpu_seconds", "cpu_seconds", "bytes")
        if all(getattr(self, field) is None for field in fields):
            raise ValueError("node usage requires at least one metered unit")
        return self


#: One usage field per billable unit — the ONE mapping every cost computation reads, so a new
#: unit is wired in exactly once. Built from two parallel tuples (never a `{"x": "x"}` dict
#: literal): the unit and field names coincide for the token units, which otherwise reads as a
#: secret-like `"token_in": "..."` assignment to anyone (or anything) scanning this file.
_UNITS: tuple[CostUnit, ...] = ("token_in", "token_out", "image", "video_second", "audio_second", "call", "gpu_second", "cpu_second", "byte")
_FIELDS: tuple[str, ...] = ("token_in", "token_out", "images", "video_seconds", "audio_seconds", "calls", "gpu_seconds", "cpu_seconds", "bytes")
USAGE_FIELD: dict[CostUnit, str] = dict(zip(_UNITS, _FIELDS, strict=True))


def priced_cost(prices: tuple[UnitPrice, ...], usage: NodeUsage) -> float:
    """The USD cost of `usage` at `prices`; a unit `usage` never reports contributes nothing."""
    total = 0.0
    for price in prices:
        quantity = getattr(usage, USAGE_FIELD[price.unit], None)
        if quantity:
            total += quantity * price.usd_per_unit
    return total


class NodeCostActual(WireModel):
    """What one node's run actually cost, after it ran."""

    node_id: UUID
    usage: NodeUsage
    known: bool
    cost_usd: float | None = Field(default=None, ge=0)
    source: PriceSource | None = None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.known != (self.cost_usd is not None):
            raise ValueError("a known node cost carries its figure, an unknown one carries none")
        return self


class NodeCostEstimate(WireModel):
    """What one node's run is expected to cost, before it runs — a RANGE, never a point figure
    (an agent's turn count, an upstream-dependent prompt length, are bounded, not fixed)."""

    node_id: UUID
    known: bool
    low_usd: float | None = Field(default=None, ge=0)
    high_usd: float | None = Field(default=None, ge=0)
    reason: str | None = Field(default=None, max_length=240)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.known and (self.low_usd is None or self.high_usd is None or self.low_usd > self.high_usd):
            raise ValueError("a known node estimate names an increasing low/high range")
        if not self.known and (self.low_usd is not None or self.high_usd is not None):
            raise ValueError("an unknown node estimate carries no range")
        return self


class RunCostEstimate(WireModel):
    nodes: tuple[NodeCostEstimate, ...] = Field(max_length=500)
    low_usd: float = Field(ge=0)
    high_usd: float = Field(ge=0)
    unknown_node_count: int = Field(ge=0)

    @model_validator(mode="after")
    def coherent_total(self) -> Self:
        if self.low_usd > self.high_usd:
            raise ValueError("a run estimate range must be increasing")
        if self.unknown_node_count != sum(1 for node in self.nodes if not node.known):
            raise ValueError("a run estimate's unknown count must match its unknown nodes")
        return self


class RunCostActual(WireModel):
    nodes: tuple[NodeCostActual, ...] = Field(max_length=500)
    total_usd: float = Field(ge=0)
    unknown_node_count: int = Field(ge=0)

    @model_validator(mode="after")
    def coherent_total(self) -> Self:
        if self.unknown_node_count != sum(1 for node in self.nodes if not node.known):
            raise ValueError("a run cost's unknown count must match its unknown nodes")
        return self


class BudgetOverrun(WireModel):
    """Returned instead of starting a run whose estimate crosses the workspace's cost ceiling —
    the client re-sends the run request with an explicit confirmation to proceed anyway."""

    estimate: RunCostEstimate
    spent_this_period_usd: float = Field(ge=0)
    ceiling_usd: float = Field(ge=0)
