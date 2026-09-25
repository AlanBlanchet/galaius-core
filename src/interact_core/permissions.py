"""Deny-by-default platform permissions: one flat catalog, groups hold a set of permissions,
accounts and workspaces join groups, effective permission = union of every group a member is in.
Two axes stay separate on purpose: ADMIN permissions (`admin.*`) are granted to ACCOUNTS (Alan
signs in with Google, the account is what carries platform-admin rights); PRODUCT permissions
(everything else — the capabilities a subscription unlocks) are granted to WORKSPACES (a
workspace runs workflows, spends credits, launches machines — never an individual account).
`GroupMembership.member_kind` carries this so ONE group/membership model serves both axes instead
of two parallel ones."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator
from pydantic.experimental.missing_sentinel import MISSING

from .wire import WireModel

Permission = Literal[
    "admin.access",
    "admin.groups.manage",
    "admin.users.view",
    "admin.plans.manage",
    "admin.audit.view",
    "workflows.run",
    "models.platform_paid",
    "machines.pool.use",
    "machines.cloud.launch",
    "machines.reserve",
    "credits.auto_topup",
]
"""The complete permission vocabulary — a permission not in this tuple can never be attached to a
group or checked, so a typo is a validation error, never a silently-ignored no-op string."""

MemberKind = Literal["account", "workspace"]
MembershipSource = Literal["manual", "subscription"]


class PermissionInfo(WireModel):
    """One catalog row: the key a group stores, plus the human label + description an admin UI
    renders beside its checkbox — never bare enum values on screen."""

    key: Permission
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=240)


PERMISSION_CATALOG: tuple[PermissionInfo, ...] = (
    PermissionInfo(key="admin.access", label="Admin panel access", description="Sign into the admin panel at all; every other admin permission requires this too."),
    PermissionInfo(key="admin.groups.manage", label="Manage groups", description="Create, rename and delete permission groups; edit which permissions and which accounts/workspaces belong to each."),
    PermissionInfo(key="admin.users.view", label="View accounts & workspaces", description="See every account, every workspace, service totals and run failures."),
    PermissionInfo(key="admin.plans.manage", label="Manage subscription plans", description="Create and edit subscription plans, and assign or change a workspace's plan."),
    PermissionInfo(key="admin.audit.view", label="View audit log", description="See the log of every admin action taken, by whom, on what."),
    PermissionInfo(key="workflows.run", label="Run workflows", description="Execute a saved workflow in this workspace."),
    PermissionInfo(key="models.platform_paid", label="Use platform-paid models", description="Call a model through this platform's own provider keys, billed at cost plus the platform fee."),
    PermissionInfo(key="machines.pool.use", label="Use pooled machines", description="Dispatch a run onto another workspace's shared machine, billed to this workspace's credits."),
    PermissionInfo(key="machines.cloud.launch", label="Launch cloud machines", description="Have the platform provision a cloud instance (e.g. Scaleway) on this workspace's behalf."),
    PermissionInfo(key="machines.reserve", label="Reserve machines", description="Hold a machine (own, pooled or cloud) for a fixed number of hours, pre-paid from credits."),
    PermissionInfo(key="credits.auto_topup", label="Enable auto top-up", description="Let this workspace's saved card be charged automatically when its credit balance runs low."),
)


class PermissionGroup(WireModel):
    id: UUID
    name: str = Field(min_length=1, max_length=120)
    permissions: tuple[Permission, ...] = Field(default=())
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def unique_permissions(self) -> Self:
        if len(set(self.permissions)) != len(self.permissions):
            raise ValueError("a group's permissions must be unique")
        return self


class GroupCreate(WireModel):
    name: str = Field(min_length=1, max_length=120)
    permissions: tuple[Permission, ...] = Field(default=())


class GroupUpdate(WireModel):
    name: str | MISSING = MISSING
    permissions: tuple[Permission, ...] | MISSING = MISSING


class GroupMembership(WireModel):
    group_id: UUID
    member_kind: MemberKind
    member_id: UUID
    source: MembershipSource
    added_at: datetime


class GroupMemberView(WireModel):
    """`GroupMembership` plus the one display fact a picker needs (an account's email, a
    workspace's name) — resolved server-side so the admin UI never fans out N+1 lookups."""

    membership: GroupMembership
    label: str = Field(min_length=1, max_length=320)


class EffectivePermissions(WireModel):
    member_kind: MemberKind
    member_id: UUID
    permissions: tuple[Permission, ...]
    groups: tuple[UUID, ...]
    """Which groups contributed — so a UI can explain "why do I have this" without a second call."""
