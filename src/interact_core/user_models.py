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
from .workflows import ConnectionResourceRef, ModelTask
from .wire import WireModel

UserModelOriginKind = Literal["huggingface_repo", "uploaded_weights", "object_storage", "docker_image"]


class UserModelOrigin(WireModel):
    """Where the weights physically live — the one fact both machine placement (fetch/mount) and
    sovereignty (what this node's weights-storage location is) read. Exactly one branch is set,
    matched to `kind`; the others stay unset rather than defaulting to something that looks set."""

    kind: UserModelOriginKind
    #: huggingface_repo: "org/name" on the Hugging Face Hub.
    repo_id: str | None = Field(default=None, min_length=1, max_length=200)
    revision: str | None = Field(default=None, min_length=1, max_length=80)
    #: uploaded_weights / object_storage: the connection the bytes live on, plus their path.
    connection: ConnectionResourceRef | None = None
    path: str | None = Field(default=None, min_length=1, max_length=1024)
    #: uploaded_weights only: the uploaded file's content hash, so a later re-upload under the
    #: same path is refused rather than silently swapping what a running placement already trusts.
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    #: docker_image: the image reference ("registry/name:tag"), optionally through a private
    #: registry connection (`connection`, reused from the object_storage/uploaded_weights branch).
    image: str | None = Field(default=None, min_length=1, max_length=400)

    @model_validator(mode="after")
    def one_branch(self) -> Self:
        if self.kind == "huggingface_repo" and (self.repo_id is None or self.connection is not None or self.path is not None or self.digest is not None or self.image is not None):
            raise ValueError("a Hugging Face origin names only a repo id, and optionally a revision")
        if self.kind == "uploaded_weights" and (self.connection is None or self.path is None or self.digest is None or self.repo_id is not None or self.revision is not None or self.image is not None):
            raise ValueError("uploaded weights need their storage connection, path and content digest")
        if self.kind == "object_storage" and (self.connection is None or self.path is None or self.repo_id is not None or self.revision is not None or self.digest is not None or self.image is not None):
            raise ValueError("an object-storage origin needs its connection and path")
        if self.kind == "docker_image" and (self.image is None or self.repo_id is not None or self.revision is not None or self.path is not None or self.digest is not None):
            raise ValueError("a Docker origin names only an image reference, optionally through a registry connection")
        return self


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
    resources: ResourceRequirement
    created_at: datetime
