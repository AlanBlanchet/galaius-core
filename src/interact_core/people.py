"""The people of a company and what each one may do (owner, review 6: « assign permissions to the
different people », not groups). One row per person: their company rights (each with where it comes
from), the PCs they may use, the projects they reach. A change is a DELTA (only the keys it names
move) judged all-or-nothing, previewed first by the same server code that applies it."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import EmailStr, Field

from .permissions import Permission, PermissionInfo
from .resources import Verb
from .wire import WireModel

SignInKind = Literal["google", "password", "agent", "other"]
#: Where a held (or refused) right comes from: ticked for this person, given by a seat in the org
#: (group / department, named in `via`), taken away for this person, or not held at all.
PersonRightSource = Literal["self", "inherited", "denied", "none"]
PresetName = Literal["admin", "member", "viewer"]


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


class PersonChange(WireModel):
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


class InviteProject(WireModel):
    project_id: UUID
    verb: Verb


class PersonInvite(WireModel):
    email: EmailStr
    rights: tuple[Permission, ...] = ()
    pcs: tuple[UUID, ...] = ()
    projects: tuple[InviteProject, ...] = ()
