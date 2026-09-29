"""On-demand cloud compute: resource requirements, the provisioned-machine wire shapes, and the
pure placement decisions every scheduler (server or test) reuses.

interact-core stays provider-independent and ships NO real-world compute catalog or compliance
data of its own — `CloudInstanceType` is a SHAPE; the actual Scaleway (or other provider) rows
are a CSV-bound registry server-side ("CSV copied verbatim, loader binds to it"),
never hand-copied into this package. `cheapest_fit`/`choose_placement` are pure functions over
whatever catalog/candidates the caller supplies."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from .wire import WireModel, WireRequest
from .workflows import (
    AcceleratorKind, DataSovereigntyTier, MachineAccelerator, MachineRef, MachineResources,
    NodeSovereigntyRecord, ResourceRequirement, Sovereignty, SovereigntyRequired, meets_requirement,
    resources_fit,
)

CloudProviderKind = Literal["scaleway"]
CloudMachineState = Literal["provisioning", "running", "stopping", "stopped", "failed"]

#: Rank for the sovereignty-tier floor check (`cheapest_fit`'s `min_tier`): lower = more
#: sovereign. Mirrors `workflows._SOVEREIGNTY_RANK`'s "lower rank is more sovereign, at-least-as-
#: sovereign-as-the-floor" shape, over the finer per-endpoint `DataSovereigntyTier` axis instead
#: of the coarser routing `Sovereignty` axis that rank covers.
_TIER_RANK: dict[DataSovereigntyTier, int] = {"eu_sovereign": 0, "eu_hosted_foreign_law": 1, "non_eu": 2}


def meets_tier(observed: DataSovereigntyTier, floor: DataSovereigntyTier | None) -> bool:
    """Whether `observed` is at least as sovereign as `floor` — `True` with no floor declared."""
    return floor is None or _TIER_RANK[observed] <= _TIER_RANK[floor]


class CloudInstanceType(WireModel):
    """One instance type a cloud provider sells in one region — the SHAPE a server-side CSV-bound
    registry populates (never a literal ships here); the provider's OWN quoted price/currency, an
    honest pair rather than a silently-converted single number."""

    provider: CloudProviderKind
    name: str = Field(min_length=1, max_length=40)
    region: str = Field(min_length=1, max_length=40)
    cpu_count: int = Field(ge=1, le=256)
    ram_mb: int = Field(ge=1, le=1 << 22)
    disk_gb: int = Field(ge=1, le=1 << 16)
    gpu_kind: AcceleratorKind = "none"
    gpu_count: int = Field(default=0, ge=0, le=16)
    vram_mb: int = Field(default=0, ge=0, le=1 << 20)
    price_per_hour: float = Field(ge=0)
    currency: Literal["EUR", "USD"]
    #: This instance's ACTUAL processing jurisdiction (ISO 3166-1 alpha-2), and the
    #: `DataSovereigntyTier` it lands in — sourced from the same registry the server's provider
    #: sovereignty lookup reads, never a second, independently-typed encoding of the same fact.
    jurisdiction: str = Field(min_length=2, max_length=8)
    sovereignty: DataSovereigntyTier

    def fits(self, requirement: ResourceRequirement) -> bool:
        if requirement.cpu_count is not None and self.cpu_count < requirement.cpu_count:
            return False
        if requirement.ram_mb is not None and self.ram_mb < requirement.ram_mb:
            return False
        if requirement.disk_gb is not None and self.disk_gb < requirement.disk_gb:
            return False
        if requirement.gpu_kind is not None and requirement.gpu_kind != "none":
            if self.gpu_kind != requirement.gpu_kind or self.gpu_count < 1:
                return False
            if requirement.vram_mb is not None and self.vram_mb < requirement.vram_mb:
                return False
        return True


def cheapest_fit(requirement: ResourceRequirement, catalog: tuple[CloudInstanceType, ...], provider: CloudProviderKind = "scaleway", region: str | None = None, min_tier: DataSovereigntyTier | None = None) -> CloudInstanceType | None:
    """The cheapest `catalog` entry meeting `requirement`, or `None` if nothing fits. `region=None`
    searches every region for `provider`. `min_tier` is the HARD sovereignty floor (threat-model
    mitigation #6): a caller with a `min_tier` set must get back `None` rather than a type ranked
    below it — never a soft "prefer sovereign, fall back otherwise". Compares within `currency` —
    ranking across mixed EUR/USD rows on raw `price_per_hour` would be wrong; callers comparing
    across currencies convert first."""
    candidates = [
        entry for entry in catalog
        if entry.provider == provider and (region is None or entry.region == region) and entry.fits(requirement) and meets_tier(entry.sovereignty, min_tier)
    ]
    return min(candidates, key=lambda entry: entry.price_per_hour) if candidates else None


class WorkspaceCloudLimits(WireModel):
    """A workspace's server-side ceiling on cloud spend/fleet size — the blast-radius bound the
    scheduler checks before EVERY cloud launch, never only a client-side flag. Denominated in USD,
    the platform's own normalized accounting currency (matches the cost engine's usage records)."""

    max_machines: int = Field(default=2, ge=0, le=50)
    max_hourly_usd: float = Field(default=5.0, ge=0, le=1000)


class CloudMachine(WireModel):
    id: UUID
    provider: CloudProviderKind
    region: str = Field(min_length=1, max_length=40)
    instance_type: str = Field(min_length=1, max_length=40)
    jurisdiction: str = Field(min_length=2, max_length=8)
    sovereignty: DataSovereigntyTier
    state: CloudMachineState
    #: Normalized to USD at launch time from the catalog entry's own `price_per_hour`/`currency` —
    #: the platform's accounting currency, not necessarily what the provider bills in.
    usd_per_hour: float = Field(ge=0)
    cost_usd_so_far: float = Field(default=0.0, ge=0)
    workflow_id: UUID | None = None
    #: Set once the cloud-init'd runner enrolls over the machine channel (same `MachineRef` every
    #: other machine has); `None` until then.
    machine: MachineRef | None = None
    created_at: datetime
    stopped_at: datetime | None = None
    #: When the machine last had no run placed on it; the idle sweep stops/deletes it after the
    #: workspace's configured idle-minutes once this has stood long enough.
    idle_since: datetime | None = None


class CloudLaunchRequest(WireRequest):
    workflow_id: UUID
    requirement: ResourceRequirement
    provider: CloudProviderKind = "scaleway"
    region: str | None = None
    #: Threat-model mitigation #6: a HARD constraint, never a soft preference — mirrors
    #: `SovereigntyRequired.min_sovereignty`'s exact shape over the finer per-endpoint tier axis.
    #: `None` = unconstrained (today's default). Refuses the launch outright (before any provider
    #: call) when no catalog entry both fits and meets this floor, and re-checks the ACTUAL
    #: provisioning response's landed region against it before recording success.
    min_sovereignty_tier: DataSovereigntyTier | None = None


class PlacementDecision(WireModel):
    """What the scheduler decided for one node: an existing connected machine, or that a cloud
    launch is needed (the caller starts one via the provider, then re-resolves)."""

    machine: MachineRef | None = None
    needs_cloud_launch: bool = False
    reason: str = Field(min_length=1, max_length=200)


def choose_placement(
    requirement: ResourceRequirement,
    candidates: tuple[tuple[MachineRef, MachineResources | None, tuple[MachineAccelerator, ...]], ...],
    sovereignty_by_machine: dict[UUID, Sovereignty] | None = None,
    require_sovereign: SovereigntyRequired | None = None,
) -> PlacementDecision:
    """Pick the first connected machine (in `candidates` order) fitting `requirement` AND, when
    `require_sovereign` is set, meeting its floor (threat-model mitigation #6 — a HARD
    constraint, never a soft preference: refuses rather than silently placing on an unmet
    candidate). `candidates` are already filtered to ONLINE, non-revoked machines by the caller —
    this function knows nothing about connection state. `sovereignty_by_machine` is the caller's
    OWN resolution (from `MachineSovereignty`/`CloudMachine`) per machine id; a machine absent
    from it reads `"unknown"`, never guessed `"self_hosted"` just for being enrolled. No fit
    -> the caller provisions a cloud machine (itself sovereignty-constrained via
    `cheapest_fit`'s `min_tier`, a separate call)."""
    graded = sovereignty_by_machine or {}
    fitting = [item for item in candidates if resources_fit(requirement, item[1], item[2])]
    if require_sovereign is not None:
        fitting = [item for item in fitting if meets_requirement(graded.get(item[0].id, "unknown"), require_sovereign)]
    if not fitting:
        reason = "no connected machine meets the resource requirement" if require_sovereign is None else "no connected machine meets both the resource requirement and the sovereignty floor"
        return PlacementDecision(needs_cloud_launch=True, reason=reason)
    return PlacementDecision(machine=fitting[0][0], reason="fits on a connected machine")


def node_sovereignty_record(node_id: UUID, machine: MachineRef, cloud_machine: "CloudMachine | None" = None, machine_sovereignty: "MachineSovereignty | None" = None) -> NodeSovereigntyRecord:
    """The ACTUAL sovereignty of one executed machine-placed node (threat-model mitigation #6):
    sourced from what really ran, never the requested/predicted placement. A cloud-launched
    machine's OWN `CloudMachine` record wins when present — it reflects where the instance
    ACTUALLY landed, resolved post-provisioning against the provider's real response, never the
    zone that was merely requested. Otherwise the owner's own `MachineSovereignty` declaration.
    Otherwise `"unknown"` — never a guessed `"self_hosted"` just because the node reached SOME
    enrolled machine. `source_id` always cites a real, findable record: the `CloudMachine`'s own id,
    the graded `MachineSovereignty`'s machine id, or — ungraded — the placement's own machine id,
    so an auditor can always locate WHY a run was graded the way it was."""
    if cloud_machine is not None:
        return NodeSovereigntyRecord(node_id=node_id, sovereignty="self_hosted", jurisdiction=cloud_machine.jurisdiction, source="cloud_machine", source_id=cloud_machine.id)
    if machine_sovereignty is not None:
        return NodeSovereigntyRecord(node_id=node_id, sovereignty="self_hosted", jurisdiction=machine_sovereignty.jurisdiction, source="machine_sovereignty", source_id=machine_sovereignty.machine.id)
    return NodeSovereigntyRecord(node_id=node_id, sovereignty="unknown", jurisdiction=None, source="machine_sovereignty", source_id=machine.id)


class MachineSovereignty(WireModel):
    """Where a self-hosted machine's compute physically runs and what tier that counts as for this
    workspace — the OWNER'S OWN declaration (a laptop's IP-geolocation is not proof of where it
    sits), never auto-detected. Mirrors `MachineCostRate`: unset until the owner sets it, never a
    guessed default. A `CloudMachine`'s jurisdiction/sovereignty instead come straight from the
    `CloudInstanceType` it was launched as — provider-determined, not owner-declared."""

    machine: MachineRef
    jurisdiction: str = Field(min_length=2, max_length=8)
    sovereignty: DataSovereigntyTier


class MachineSovereigntyUpdate(WireRequest):
    jurisdiction: str = Field(min_length=2, max_length=8)
    sovereignty: DataSovereigntyTier
