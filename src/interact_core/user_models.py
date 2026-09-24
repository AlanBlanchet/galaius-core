"""A workspace's OWN model: a checkpoint the owner supplies rather than a vendor catalog entry —
from a Hugging Face repo, an uploaded weights file, an object-storage/SSH path, or a Docker image.
Registered once (`UserModel`), runnable as a `model` node the same way a vendor's model is
(`ModelImplementation(kind="model", provider="workspace", model=str(id))`, `placement.target ==
"machine"`): the machine runner reads `origin` to fetch/mount the weights, `resources` to refuse
placement on a machine that cannot meet it (`interact_core.cloud.resources_fit`, the same check a
cloud launch uses)."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .cloud import ResourceRequirement
from .pool import ModelWeightFormat, SandboxTier
from .workflows import ConnectionResourceRef, ModelTask
from .wire import WireModel

UserModelOriginKind = Literal["huggingface_repo", "uploaded_weights", "object_storage", "docker_image"]

#: The only weight format this app ever deserializes for a workspace-supplied checkpoint
#: (threat-model `cloud-compute-and-sovereignty-2026-09-24.md` #5: `torch.load`/pickle execute
#: arbitrary code on load). Checked again at the byte level by `interact.model_safety` on the
#: machine (extension AND magic-byte/pickle-opcode detection) — this is the registration-time gate,
#: never the only one.
_SAFE_WEIGHT_SUFFIX = ".safetensors"


def _is_safe_weight_filename(name: str) -> bool:
    return name.endswith(_SAFE_WEIGHT_SUFFIX) and name not in (_SAFE_WEIGHT_SUFFIX, "")


class UserModelOrigin(WireModel):
    """Where the weights physically live — the one fact both machine placement (fetch/mount) and
    sovereignty (what this node's weights-storage location is) read. Exactly one branch is set,
    matched to `kind`; the others stay unset rather than defaulting to something that looks set.

    Threat-model #5 (model-weight poisoning): every weight-bearing branch is safetensors-only —
    `uploaded_weights`/`object_storage` refuse a `path` that is not `.safetensors`;
    `huggingface_repo` pins the EXACT files to fetch (`weight_files`), never "every file in the
    repo" (which could include a pickle-format fallback like `pytorch_model.bin`)."""

    kind: UserModelOriginKind
    #: huggingface_repo: "org/name" on the Hugging Face Hub.
    repo_id: str | None = Field(default=None, min_length=1, max_length=200)
    revision: str | None = Field(default=None, min_length=1, max_length=80)
    #: huggingface_repo only: the exact `.safetensors` file(s) the runner fetches — nothing else in
    #: the repo is ever downloaded, closing the pickle-fallback path structurally, not by trusting
    #: whatever the repo happens to contain.
    weight_files: tuple[str, ...] | None = Field(default=None, max_length=64)
    #: huggingface_repo only, default False (the safe path): whether loading this repo needs its
    #: OWN custom Python (`trust_remote_code`-class execution) beyond deserializing named weight
    #: files. `True` makes `runs_bundled_code`/`sandbox_tier` below route every run of this model
    #: through the SAME isolation a script node gets (`interact_core.pool.SandboxTier.gvisor`),
    #: never the workspace-private `"none"` tier — the threat model's "no trust_remote_code without
    #: the same sandboxing as script nodes."
    trusts_remote_code: bool = False
    #: uploaded_weights / object_storage: the connection the bytes live on, plus their path.
    connection: ConnectionResourceRef | None = None
    path: str | None = Field(default=None, min_length=1, max_length=1024)
    #: uploaded_weights only: the uploaded file's content hash, so a later re-upload under the
    #: same path is refused rather than silently swapping what a running placement already trusts.
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    #: docker_image: the image reference ("registry/name:tag"), optionally through a private
    #: registry connection (`connection`, reused from the object_storage/uploaded_weights branch).
    #: A container is arbitrary code BY DEFINITION — `runs_bundled_code` is always True for this
    #: kind, never an opt-in flag the way `trusts_remote_code` is for a Hub repo.
    image: str | None = Field(default=None, min_length=1, max_length=400)

    @model_validator(mode="after")
    def one_branch(self) -> Self:
        if self.kind == "huggingface_repo" and (self.repo_id is None or self.connection is not None or self.path is not None or self.digest is not None or self.image is not None):
            raise ValueError("a Hugging Face origin names only a repo id, and optionally a revision")
        if self.kind == "huggingface_repo" and (not self.weight_files or any(not _is_safe_weight_filename(name) for name in self.weight_files)):
            raise ValueError("a Hugging Face origin must pin at least one .safetensors weight file by exact name; no other format is fetched")
        if self.kind != "huggingface_repo" and (self.weight_files is not None or self.trusts_remote_code):
            raise ValueError("weight_files/trusts_remote_code apply only to a Hugging Face origin")
        if self.kind == "uploaded_weights" and (self.connection is None or self.path is None or self.digest is None or self.repo_id is not None or self.revision is not None or self.image is not None):
            raise ValueError("uploaded weights need their storage connection, path and content digest")
        if self.kind == "uploaded_weights" and not _is_safe_weight_filename(self.path):
            raise ValueError("uploaded weights must be a .safetensors file; no other format is loaded")
        if self.kind == "object_storage" and (self.connection is None or self.path is None or self.repo_id is not None or self.revision is not None or self.digest is not None or self.image is not None):
            raise ValueError("an object-storage origin needs its connection and path")
        if self.kind == "object_storage" and not _is_safe_weight_filename(self.path):
            raise ValueError("an object-storage weight file must be .safetensors; no other format is loaded")
        if self.kind == "docker_image" and (self.image is None or self.repo_id is not None or self.revision is not None or self.path is not None or self.digest is not None):
            raise ValueError("a Docker origin names only an image reference, optionally through a registry connection")
        return self

    @property
    def runs_bundled_code(self) -> bool:
        """Whether ANY run of this origin executes code beyond typed-weight deserialization — the
        one place every dispatcher reads, so a Docker image's always-arbitrary-code nature and an
        opted-in `trust_remote_code` repo are never two separately-checked booleans that could
        drift apart."""
        return self.kind == "docker_image" or self.trusts_remote_code

    @property
    def sandbox_tier(self) -> SandboxTier:
        """The isolation tier the machine runner MUST dispatch this origin's every run at
        (threat-model #5): `"gvisor"` — the same tier a script node gets — whenever it runs bundled
        code; `"none"` (today's workspace-private, owner-trusted run) for a pure weight-file origin
        that only ever deserializes named safetensors files."""
        return "gvisor" if self.runs_bundled_code else "none"


class UserModelRef(WireModel):
    id: UUID
    revision: UUID


class UserModel(WireModel):
    """One workspace-registered model: runnable as a `model` node on any machine whose reported
    `interact_core.MachineResources`/accelerators satisfy `resources`
    (`interact_core.cloud.resources_fit`). Immutable per revision, like every other resource this
    app saves — editing licence/resources/origin makes a new revision, never an in-place mutation
    a running placement could silently start reading differently."""

    id: UUID
    revision: UUID
    name: str = Field(min_length=1, max_length=120)
    task: ModelTask
    #: SPDX id when known ("apache-2.0", "mit"...) or the exact free-text licence name the source
    #: states — never inferred, since a wrong licence is a compliance defect, not a display detail.
    licence: str = Field(min_length=1, max_length=120)
    origin: UserModelOrigin
    #: The only format `origin`'s weight-bearing branches may declare — `Literal["safetensors"]`
    #: today (`interact_core.pool.ModelWeightFormat`): naming it as a field, not a hardcoded
    #: constant, is where a second proven-safe format would be added, never a silent second path.
    weight_format: ModelWeightFormat = "safetensors"
    resources: ResourceRequirement
    created_at: datetime
