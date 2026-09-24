"""On-demand cloud compute: resource requirements, a static provider catalog, provisioned
machines, and the pure placement decision every scheduler (server or test) reuses.

A node's resource need and a machine's actual resources are matched here so "place this node"
never guesses: fits on a connected machine, or a cloud machine of the cheapest fitting type is
needed. The catalog itself is data (`CLOUD_INSTANCE_CATALOG`), sourced from the provider's own
docs, never invented per call — the same shape `MACHINE_MODELS` already uses for model specs."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from .wire import WireModel
from .workflows import AcceleratorKind, MachineAccelerator, MachineRef, MachineResources

CloudProviderKind = Literal["scaleway"]
CloudMachineState = Literal["provisioning", "running", "stopping", "stopped", "failed"]


class ResourceRequirement(WireModel):
    """What one workflow node needs to run, derived from the model registry / the user's model
    record. Every field optional: unset means "no constraint from this axis", never zero — a node
    with no declared requirement places on any connected machine, today's behaviour unchanged."""

    cpu_count: int | None = Field(default=None, ge=1, le=256)
    ram_mb: int | None = Field(default=None, ge=1, le=1 << 22)
    gpu_kind: AcceleratorKind | None = None
    vram_mb: int | None = Field(default=None, ge=1, le=1 << 20)
    disk_gb: int | None = Field(default=None, ge=1, le=1 << 16)


def resources_fit(requirement: ResourceRequirement, resources: MachineResources | None, accelerators: tuple[MachineAccelerator, ...]) -> bool:
    """Whether a machine reporting `resources`/`accelerators` satisfies `requirement`. A machine
    that never reported `resources` (older runner) only fits a requirement with no CPU/RAM/disk
    axis — never silently assumed to fit an unknown size."""
    if requirement.cpu_count is not None and (resources is None or resources.cpu_count < requirement.cpu_count):
        return False
    if requirement.ram_mb is not None and (resources is None or resources.ram_mb < requirement.ram_mb):
        return False
    if requirement.disk_gb is not None and (resources is None or resources.disk_free_gb < requirement.disk_gb):
        return False
    if requirement.gpu_kind is not None and requirement.gpu_kind != "none":
        matching = [item for item in accelerators if item.kind == requirement.gpu_kind]
        if not matching:
            return False
        if requirement.vram_mb is not None and max(item.memory_mb for item in matching) < requirement.vram_mb:
            return False
    return True


class CloudInstanceType(WireModel):
    """One instance type a cloud provider sells in one region: static catalog data (the provider's
    own commercial type list), refreshed by re-running the provider's research — never queried live
    per launch."""

    provider: CloudProviderKind
    name: str = Field(min_length=1, max_length=40)
    region: str = Field(min_length=1, max_length=40)
    cpu_count: int = Field(ge=1, le=256)
    ram_mb: int = Field(ge=1, le=1 << 22)
    disk_gb: int = Field(ge=1, le=1 << 16)
    gpu_kind: AcceleratorKind = "none"
    gpu_count: int = Field(default=0, ge=0, le=16)
    vram_mb: int = Field(default=0, ge=0, le=1 << 20)
    usd_per_hour: float = Field(ge=0)
    jurisdiction: str = Field(min_length=2, max_length=8)
    sovereign: bool

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


#: Scaleway Instances API commercial types, fr-par-2 (the zone carrying every current GPU type;
#: FR jurisdiction — the fr-par-* zones are the only ones physically in France, per
#: ~/.github/research/scaleway-instances-api-2026-09-24.md §4). Every cpu/ram/disk/price cell is
#: copied verbatim from that file's §2 pricing tables (Scaleway's own docs, fetched 2026-09-24) —
#: never invented. `sovereign=False` on every row: SecNumCloud (ANSSI) is NOT qualified for any
#: Scaleway zone yet (same source §4) — EU/French data residency is real, the certified-sovereign
#: bar is not met, so this never silently reads as sovereign. GPU rows with no local disk in the
#: catalog ("none (Block)") get a 125 GB placeholder — Scaleway's own documented GPU-OS root
#: volume minimum (§1.6) — standing for the Block Storage volume this scheduler would attach.
#: §6 of the source file: the LIVE catalog is `GET .../products/servers`, paginated — this static
#: tuple is a snapshot for placement math, refreshed by re-running the provider's research, never
#: hand-edited from a launch failure.
CLOUD_INSTANCE_CATALOG: tuple[CloudInstanceType, ...] = (
    CloudInstanceType(provider="scaleway", name="DEV1-S", region="fr-par-2", cpu_count=2, ram_mb=2 * 1024, disk_gb=20, usd_per_hour=0.008976, jurisdiction="FR", sovereign=False),
    CloudInstanceType(provider="scaleway", name="DEV1-M", region="fr-par-2", cpu_count=3, ram_mb=4 * 1024, disk_gb=40, usd_per_hour=0.020196, jurisdiction="FR", sovereign=False),
    CloudInstanceType(provider="scaleway", name="DEV1-L", region="fr-par-2", cpu_count=4, ram_mb=8 * 1024, disk_gb=80, usd_per_hour=0.04284, jurisdiction="FR", sovereign=False),
    CloudInstanceType(provider="scaleway", name="POP2-2C-8G", region="fr-par-2", cpu_count=2, ram_mb=8 * 1024, disk_gb=125, usd_per_hour=0.0735, jurisdiction="FR", sovereign=False),
    CloudInstanceType(provider="scaleway", name="POP2-8C-32G", region="fr-par-2", cpu_count=8, ram_mb=32 * 1024, disk_gb=125, usd_per_hour=0.29, jurisdiction="FR", sovereign=False),
    CloudInstanceType(provider="scaleway", name="RENDER-S", region="fr-par-2", cpu_count=10, ram_mb=42 * 1024, disk_gb=400, gpu_kind="cuda", gpu_count=1, vram_mb=16 * 1024, usd_per_hour=1.221, jurisdiction="FR", sovereign=False),
    CloudInstanceType(provider="scaleway", name="L4-1-24G", region="fr-par-2", cpu_count=8, ram_mb=48 * 1024, disk_gb=125, gpu_kind="cuda", gpu_count=1, vram_mb=24 * 1024, usd_per_hour=0.7875, jurisdiction="FR", sovereign=False),
    CloudInstanceType(provider="scaleway", name="L40S-1-48G", region="fr-par-2", cpu_count=8, ram_mb=96 * 1024, disk_gb=1600, gpu_kind="cuda", gpu_count=1, vram_mb=48 * 1024, usd_per_hour=1.469916, jurisdiction="FR", sovereign=False),
    CloudInstanceType(provider="scaleway", name="H100-1-80G", region="fr-par-2", cpu_count=24, ram_mb=240 * 1024, disk_gb=3000, gpu_kind="cuda", gpu_count=1, vram_mb=80 * 1024, usd_per_hour=2.8665, jurisdiction="FR", sovereign=False),
)


def cheapest_fit(requirement: ResourceRequirement, provider: CloudProviderKind = "scaleway", region: str | None = None) -> CloudInstanceType | None:
    """The cheapest catalog entry meeting `requirement`, or `None` if nothing in the catalog fits.
    `region=None` searches every region for `provider`."""
    candidates = [entry for entry in CLOUD_INSTANCE_CATALOG if entry.provider == provider and (region is None or entry.region == region) and entry.fits(requirement)]
    return min(candidates, key=lambda entry: entry.usd_per_hour) if candidates else None


class WorkspaceCloudLimits(WireModel):
    """A workspace's server-side ceiling on cloud spend/fleet size — the blast-radius bound the
    scheduler checks before EVERY cloud launch, never only a client-side flag."""

    max_machines: int = Field(default=2, ge=0, le=50)
    max_hourly_usd: float = Field(default=5.0, ge=0, le=1000)


class CloudMachine(WireModel):
    id: UUID
    provider: CloudProviderKind
    region: str = Field(min_length=1, max_length=40)
    instance_type: str = Field(min_length=1, max_length=40)
    jurisdiction: str = Field(min_length=2, max_length=8)
    sovereign: bool
    state: CloudMachineState
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


class CloudLaunchRequest(WireModel):
    workflow_id: UUID
    requirement: ResourceRequirement
    provider: CloudProviderKind = "scaleway"
    region: str | None = None


class PlacementDecision(WireModel):
    """What the scheduler decided for one node: an existing connected machine, or that a cloud
    launch is needed (the caller starts one via the provider, then re-resolves)."""

    machine: MachineRef | None = None
    needs_cloud_launch: bool = False
    reason: str = Field(min_length=1, max_length=200)


def choose_placement(requirement: ResourceRequirement, candidates: tuple[tuple[MachineRef, MachineResources | None, tuple[MachineAccelerator, ...]], ...]) -> PlacementDecision:
    """Pick the first connected machine (in `candidates` order) fitting `requirement`. `candidates`
    are already filtered to ONLINE, non-revoked machines by the caller — this function knows
    nothing about connection state. No fit -> the caller provisions a cloud machine."""
    fitting = next((item for item in candidates if resources_fit(requirement, item[1], item[2])), None)
    if fitting is None:
        return PlacementDecision(needs_cloud_launch=True, reason="no connected machine meets the resource requirement")
    return PlacementDecision(machine=fitting[0], reason="fits on a connected machine")


class MachineSovereignty(WireModel):
    """Where a self-hosted machine's compute physically runs and whether that counts as sovereign
    for this workspace — the OWNER'S OWN declaration (a laptop's IP-geolocation is not proof of
    where it sits), never auto-detected. Mirrors `MachineCostRate`: unset until the owner sets it,
    never a guessed default. A `CloudMachine`'s jurisdiction/sovereign instead come straight from
    the `CloudInstanceType` it was launched as — provider-determined, not owner-declared."""

    machine: MachineRef
    jurisdiction: str = Field(min_length=2, max_length=8)
    sovereign: bool


class MachineSovereigntyUpdate(WireModel):
    jurisdiction: str = Field(min_length=2, max_length=8)
    sovereign: bool
