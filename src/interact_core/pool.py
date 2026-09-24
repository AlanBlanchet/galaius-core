"""Contracts for the shared/re-usable compute pool's six must-hold safeguards.

Sourced from `~/.github/research/threat-model-shared-compute-2026-09-24.md` (2026-09-24): a
machine serving more than one workspace crosses a NEW trust boundary ("boundary 4", co-tenant
isolation on one host) that today's workspace-private machine model never had to defend. Every
type here is the wire shape a pooled run's safeguard needs; the enforcement itself (a real
per-run container, a real firewall, a real GPU scrub) lives in the runtime that reads these
(`interact.sandbox`, `interact.gpu_scrub`) — this module carries no side effect.

Workspace-private placement (today's default, `decisions.md` 2026-09-24) never constructs any of
these; a pooled run does. `POOL_SHARING_ENABLED` gates the feature end to end: flipped only once
every one of the six adversarial tests this module's docstring enumerates passes AND a real
dispatch exists to read it."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from .wire import WireModel

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


#: Flipped to True only once every one of the six threat-model safeguards has a passing adversarial
#: test recorded (isolation, egress, GPU scrub, budget, sovereignty, safetensors) AND a real pooled
#: dispatch exists to read it. False keeps every machine workspace-private, today's shipped
#: behaviour, unchanged.
POOL_SHARING_ENABLED = False
