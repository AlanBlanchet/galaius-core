"""Contracts for the shared/re-usable compute pool's six must-hold safeguards.

Sourced from `~/.github/research/threat-model-shared-compute-2026-09-24.md` (2026-09-24): a
machine serving more than one workspace crosses a NEW trust boundary ("boundary 4", co-tenant
isolation on one host) that today's workspace-private machine model never had to defend. Every
type here is the wire shape a pooled run's safeguard needs; the enforcement itself (a real
per-run container, a real firewall, a real GPU scrub) lives in the runtime that reads these
(`interact.sandbox`, `interact.gpu_scrub`) — this module carries no side effect.

Workspace-private placement (`decisions.md` 2026-09-24) never constructs any of these; a pooled run
does. `POOL_SHARING_ENABLED` gates the feature end to end — see its own docstring below for the
six adversarial tests and the real dispatch that earned it flipping to `True` on 2026-09-25."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .cloud import ResourceRequirement, resources_fit
from .wire import WireModel
from .workflows import MachineAccelerator, MachineRef, MachineResources

#: gVisor (`runsc`) is the only tier this codebase currently proves closes the container-escape
#: surface for untrusted code (verified 2026-09-24: `docker run --runtime=runsc` on this dev
#: machine; installed from gVisor's own apt repo, the same repo an Ubuntu-based Scaleway image
#: uses — no code difference between "this PC" and a Scaleway GPU instance, only the apt run).
#: Firecracker/Kata were the threat model's other named candidate; not picked because (a) it needs
#: KVM nested-virt, unreliable on a cloud VM without a bare-metal offer, and (b) NVIDIA GPU
#: passthrough into a microVM needs VFIO device passthrough, a much heavier operational lift than
#: gVisor's `nvproxy` (ioctl interposition on the existing NVIDIA driver — UNVERIFIED against
#: gVisor's current docs for the exact GPU/driver combination Scaleway ships; route to
#: `web-researcher` before a pooled GPU run ships). "none" is a workspace-private, owner-trusted
#: run — a pooled/cross-tenant run must never carry it.
SandboxTier = Literal["gvisor", "none"]

#: A GPU reset proof: the driver's own device reset (`nvidia-smi --gpu-reset`, unsupported on
#: GeForce-class consumer cards — no SR-IOV) or this runtime's own whole-device fill+free scrub
#: (`interact.gpu_scrub.scrub_all_free_memory`) when the driver reset is unavailable.
GpuResetKind = Literal["device_reset", "scrubbed"]


class EgressAllowEntry(WireModel):
    """One allowed outbound destination for a pooled run — resolved to IPs by the TRUSTED broker
    outside the sandbox, never by the sandboxed process itself (a sandboxed DNS answer is
    untrusted input)."""

    host: str = Field(min_length=1, max_length=253)
    port: int = Field(ge=1, le=65535)


class EgressPolicy(WireModel):
    """Default-deny egress for one pooled run. `allow` is empty by default: a run with no declared
    network need gets none at all — not even its own sandbox's loopback reaching anywhere beyond
    itself. `BLOCKED_EGRESS_HOSTS` binds unconditionally underneath any policy; no `allow` entry
    can ever satisfy it (enforced in `interact.sandbox`, checked again here defensively)."""

    allow: tuple[EgressAllowEntry, ...] = Field(default=())

    def permits(self, host: str, port: int) -> bool:
        if host in BLOCKED_EGRESS_HOSTS:
            return False
        return any(entry.host == host and entry.port == port for entry in self.allow)


#: Never allow-listable regardless of what a run declares — the classic SSRF-to-cloud-credential
#: path on every major provider's instance metadata service (threat-model threat #2). Scaleway,
#: AWS and GCP all answer the same well-known link-local address; the EC2/GCP alternate hostname
#: is blocked too since a resolver inside a pooled run must never be trusted to answer honestly.
BLOCKED_EGRESS_HOSTS: frozenset[str] = frozenset({"169.254.169.254", "metadata.google.internal", "fd00:ec2::254"})


class GpuScrubRecord(WireModel):
    """Server-side provenance the scheduler checks before handing one physical accelerator to a
    new tenant (threat #1c): this machine's GPU at this index was reset/scrubbed at this time.
    `tenant_before` is kept for audit even after the scrub clears the device."""

    machine: UUID
    accelerator_index: int = Field(ge=0)
    kind: GpuResetKind
    scrubbed_at: datetime
    tenant_before: UUID | None = None


class RunBudgetCheck(WireModel):
    """What the scheduler asks before EVERY provisioning call and EVERY pooled run (threat #4):
    real committed spend plus this run's own estimate against the workspace's ceiling. Both money
    figures are the CALLER's responsibility to source from the provider's own billing/usage data —
    this type and `check_budget` never call out anywhere themselves, so a caller can never claim
    "the cost engine verified it" without actually having sourced real numbers first."""

    workspace_id: UUID
    committed_usd_this_period: float = Field(ge=0)
    estimated_run_usd: float = Field(ge=0)
    ceiling_usd_this_period: float = Field(ge=0)


class RunBudgetDecision(WireModel):
    allowed: bool
    reason: str = Field(min_length=1, max_length=200)
    projected_usd: float = Field(ge=0)


def check_budget(check: RunBudgetCheck) -> RunBudgetDecision:
    """Pure server-side gate: committed + estimate vs ceiling. No I/O, no client-reported spend —
    the caller already resolved `committed_usd_this_period` from the provider's own billing data
    before constructing `check`; this function only enforces the arithmetic, so the same
    projection can never be computed two different ways in two call sites."""
    projected = check.committed_usd_this_period + check.estimated_run_usd
    if projected > check.ceiling_usd_this_period:
        return RunBudgetDecision(allowed=False, projected_usd=projected, reason=f"would bring workspace spend to ${projected:.4f}, over its ${check.ceiling_usd_this_period:.4f} ceiling this period")
    return RunBudgetDecision(allowed=True, projected_usd=projected, reason="within ceiling")


#: A user-supplied model's weight format: `torch.load`/`pickle.load` execute arbitrary code as the
#: model-serving process on deserialization (threat #5) — safetensors is the only format this
#: codebase ever loads for a workspace-supplied checkpoint. `interact.model_safety` enforces this
#: at the byte level (extension AND magic-byte/pickle-opcode detection, never extension alone).
ModelWeightFormat = Literal["safetensors"]


class UnsafeModelWeightsError(Exception):
    """A model artifact was refused before any byte of it was deserialized — not safetensors, or a
    pickle stream (possibly under a misleading extension) detected by its own magic bytes."""


#: Flipped 2026-09-25 once every one of the six threat-model safeguards had a passing adversarial
#: test AND a real pooled dispatch existed to read it:
#:   1 isolation  — interact/tests/test_pool_isolation.py (4 tests, real gVisor containers)
#:   2 egress     — interact/tests/test_pool_egress.py (5 tests, real nft ruleset + listener)
#:   3 GPU scrub  — interact/tests/test_gpu_scrub.py (3 tests, real CUDA) +
#:                  server/tests/server tests.py (freshness gate, real dispatch)
#:   4 budget     — server/tests/server tests.py
#:                  (test_dispatch_pooled_refuses_a_run_that_would_cross_the_workspace_ceiling)
#:   5 sovereignty — server/tests/server tests.py
#:                  (test_dispatch_pooled_with_sovereignty_required_skips_a_non_sovereign_owner)
#:   6 safetensors — interact/tests/test_model_safety.py (5 tests, real pickle RCE payload refused)
#: The real dispatch: `server.machines.channel.MachineChannel.dispatch_pooled`, runner
#: routing in `interact.machines.MachineRunner._execute`/`_run_script_pooled`. Scope note: pooled
#: dispatch today only offers SCRIPT nodes to the pool (the safetensors gate applies to model
#: weight loading, not yet reachable from this specific dispatch path — MODEL/FUNCTION pooled
#: dispatch is refused outright by the runner, not silently run unsandboxed). This flag gates the
#: MECHANISM's own internal logic; the general workflow scheduler does not yet call
#: `dispatch_pooled` automatically for an unplaced node (the same state `cloud.py`'s on-demand
#: launch is in) — that auto-placement wiring is the next integration step.
POOL_SHARING_ENABLED = True


class MachinePoolSettingsUpdate(WireModel):
    """The owner's own opt-in for ONE of their machines: off by default, and switching it on needs
    a price in the SAME request — an owner can never end up sharing at an unset (silently free)
    rate."""

    shared: bool
    price_usd_per_hour: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def priced_when_shared(self) -> Self:
        if self.shared and self.price_usd_per_hour is None:
            raise ValueError("sharing a machine needs a price (set price_usd_per_hour, or 0 for free)")
        return self


class MachinePoolSettings(WireModel):
    machine: MachineRef
    shared: bool = False
    price_usd_per_hour: float | None = Field(default=None, ge=0)


class PooledMachineCandidate(WireModel):
    """One OTHER workspace's machine currently offered to the pool — the cross-workspace read a
    placement decision needs (boundary 4: the first query in this codebase that legitimately
    returns a machine record to a workspace that does not own it). Carries only what placement and
    billing need: no owner identity beyond the workspace id billing must attribute to, no token,
    no working-directory path."""

    machine: MachineRef
    owner_workspace_id: UUID
    price_usd_per_hour: float = Field(ge=0)
    resources: MachineResources | None = None
    accelerators: tuple[MachineAccelerator, ...] = ()


def choose_pooled_placement(requirement: ResourceRequirement, candidates: tuple[PooledMachineCandidate, ...]) -> PooledMachineCandidate | None:
    """The cheapest fitting pooled candidate — never the first, so offering a machine at a lower
    price actually wins it more runs. `candidates` are already filtered to ONLINE, opted-in,
    UNOCCUPIED (one-tenant-at-a-time) machines by the caller; this function knows nothing about
    connection or lock state, mirroring `cloud.cheapest_fit`'s split of concerns."""
    fitting = [candidate for candidate in candidates if resources_fit(requirement, candidate.resources, candidate.accelerators)]
    return min(fitting, key=lambda candidate: candidate.price_usd_per_hour) if fitting else None


class PooledRunBilling(WireModel):
    """What one pooled run cost the TENANT workspace, at the OWNER's own rate — the metering fact
    itself; settlement between workspaces is a separate payments concern this type does not carry."""

    run_id: UUID
    node_id: UUID
    tenant_workspace_id: UUID
    owner_workspace_id: UUID
    machine: MachineRef
    price_usd_per_hour: float = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    cost_usd: float = Field(ge=0)


def pooled_run_cost(price_usd_per_hour: float, elapsed_seconds: float) -> float:
    return round(price_usd_per_hour * elapsed_seconds / 3600, 6)
