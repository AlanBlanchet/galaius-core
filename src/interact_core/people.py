"""The people of a company and what each one may do (owner, review 6: « assign permissions to the
different people », not groups). One row per person: their company rights (each with where it comes
from), the PCs they may use, the projects they reach. A change is a DELTA (only the keys it names
move) judged all-or-nothing, previewed first by the same server code that applies it."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from .accounts import EMAIL_PATTERN
from .permissions import Permission, PermissionInfo
from .resources import Verb
from .wire import WireModel, WireRequest

SignInKind = Literal["google", "password", "agent", "other"]
#: Where a held (or refused) right comes from: ticked for this person, given by a seat in the org
#: (group / department, named in `via`), taken away for this person, or not held at all.
PersonRightSource = Literal["self", "inherited", "denied", "none"]
PresetName = Literal["admin", "member", "viewer"]
#: The fixed presets beside « Administrateur » (= every company right the reader holds, computed per
#: reader). A key the running catalog does not hold is left out where the presets are served.
PERSON_PRESETS: dict[Literal["member", "viewer"], tuple[str, ...]] = {
    "member": ("agents.view", "agents.run", "workflows.run", "workflows.edit", "billing.view", "accounts.connect_own"),
    "viewer": ("agents.view", "billing.view"),
}
#: The most a PC's owner hands someone else on it: agents that read (`read_only`) or also write in the
#: folders it opens (`workspace_write`); never `full_access` (the agent runs as the owner's OS user).
PcLevel = Literal["read_only", "workspace_write"]


class PersonRight(WireModel):
    key: Permission
    held: bool
    source: PersonRightSource
    via: str | None = None
    may_change: bool


class PersonPc(WireModel):
    machine_id: UUID
    name: str
    owner_name: str
    #: The person may use this PC (always True on a PC they own).
    granted: bool
    #: The person owns this PC.
    owner: bool
    #: The signed-in actor owns this PC: only a PC's owner hands it to someone.
    may_change: bool


class PersonProject(WireModel):
    project_id: UUID
    name: str
    verb: Verb | None
    may_change: bool


class Person(WireModel):
    account_id: UUID
    name: str
    email: str
    sign_in: SignInKind
    guest: bool
    you: bool
    last_seen_at: datetime | None
    #: Opaque digest of this person's access; `PUT` carries it in `If-Match` (409 when it moved).
    revision: str
    rights: tuple[PersonRight, ...] = ()
    pcs: tuple[PersonPc, ...] = ()
    projects: tuple[PersonProject, ...] = ()
    may_edit: bool
    may_remove: bool


class PendingInvite(WireModel):
    invitation_id: UUID
    email: str
    rights: tuple[Permission, ...]
    invited_by: str
    created_at: datetime


class People(WireModel):
    workspace_id: UUID
    personal: bool
    may_invite: bool
    may_manage: bool
    catalog: tuple[PermissionInfo, ...]
    presets: dict[PresetName, tuple[Permission, ...]]
    people: tuple[Person, ...]
    invitations: tuple[PendingInvite, ...] = ()


class PersonChange(WireRequest):
    """A delta: only the keys present change; an absent right / PC / project is left as it is."""

    rights: dict[Permission, bool] = Field(default_factory=dict)
    pcs: dict[UUID, bool] = Field(default_factory=dict)
    projects: dict[UUID, Verb | None] = Field(default_factory=dict)


class Refusal(WireModel):
    key: str
    reason: str


class PersonEffect(WireModel):
    gains: tuple[str, ...] = ()
    loses: tuple[str, ...] = ()
    #: (before, after) of what the person reaches, counted only over what the actor may see.
    counts: dict[Literal["agents", "pcs", "projects"], tuple[int, int]]
    refused: tuple[Refusal, ...] = ()
    warnings: tuple[str, ...] = ()


class InviteProject(WireRequest):
    project_id: UUID
    verb: Verb


class PersonInvite(WireRequest):
    email: str = Field(pattern=EMAIL_PATTERN, max_length=320)
    rights: tuple[Permission, ...] = ()
    pcs: tuple[UUID, ...] = ()
    projects: tuple[InviteProject, ...] = ()


class PcGrant(WireModel):
    """One person a PC's owner lets use the PC: start agents (at most `level`) in `roots` — empty:
    every folder the PC opens to agents — and run file nodes on it."""

    account_id: UUID
    level: PcLevel = "read_only"
    roots: tuple[str, ...] = Field(default=(), max_length=32)


class PcAccess(WireModel):
    owner: UUID
    grants: tuple[PcGrant, ...] = ()


class PcAccessUpdate(WireRequest):
    """The owner's full list of people who may use the PC (the owner always may)."""

    grants: tuple[PcGrant, ...] = Field(default=(), max_length=64)
