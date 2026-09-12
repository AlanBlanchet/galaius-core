"""Provider-independent account and workspace HTTP wire contracts."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, SecretStr
from pydantic.experimental.missing_sentinel import MISSING

from .wire import WireModel

_EMAIL = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"
WorkspaceRole = Literal["owner", "admin", "member", "viewer"]
InvitableWorkspaceRole = Literal["admin", "member", "viewer"]


class Account(WireModel):
    account_id: UUID
    email: str = Field(pattern=_EMAIL, max_length=320)
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    locale: Literal["en", "fr"]
    verified: bool


class AccountUpdate(WireModel):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    locale: Literal["en", "fr"] | MISSING = MISSING


class Bootstrap(WireModel):
    account: Account
    workspaces: tuple["Workspace", ...]
    current_workspace_id: UUID
    csrf_token: str = Field(min_length=1)
    session_expires_at: datetime


class SignupRequest(WireModel):
    email: str = Field(pattern=_EMAIL, max_length=320)
    password: SecretStr = Field(min_length=12, max_length=1024)
    locale: Literal["en", "fr"] = "en"


class LoginRequest(WireModel):
    email: str = Field(pattern=_EMAIL, max_length=320)
    password: SecretStr = Field(min_length=1, max_length=1024)


class TokenRequest(WireModel):
    token: SecretStr = Field(min_length=32, max_length=512)


class RecoveryRequest(WireModel):
    email: str = Field(pattern=_EMAIL, max_length=320)


class PasswordResetRequest(TokenRequest):
    password: SecretStr = Field(min_length=12, max_length=1024)


class Workspace(WireModel):
    workspace_id: UUID
    name: str = Field(min_length=1, max_length=120)
    role: WorkspaceRole


class WorkspaceCreate(WireModel):
    name: str = Field(min_length=1, max_length=120)


class WorkspaceUpdate(WireModel):
    name: str = Field(min_length=1, max_length=120)


class WorkspaceMember(WireModel):
    account: Account
    role: WorkspaceRole


class WorkspaceInvitation(WireModel):
    id: UUID
    email: str = Field(pattern=_EMAIL, max_length=320)
    role: InvitableWorkspaceRole
    status: Literal["pending", "accepted", "cancelled", "expired"]
    created_at: datetime
    expires_at: datetime


class WorkspaceMembership(WireModel):
    members: tuple[WorkspaceMember, ...]
    invitations: tuple[WorkspaceInvitation, ...]


class WorkspaceInvite(WireModel):
    email: str = Field(pattern=_EMAIL, max_length=320)
    role: InvitableWorkspaceRole


class WorkspaceMemberRoleUpdate(WireModel):
    role: WorkspaceRole


class PlatformError(WireModel):
    code: Literal[
        "authentication_failed", "csrf_failed", "invalid_origin", "invalid_request",
        "not_found", "rate_limited", "verification_failed", "recovery_failed", "unavailable",
    ]
