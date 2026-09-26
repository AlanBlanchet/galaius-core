"""Provider-independent account and workspace HTTP wire contracts."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, SecretStr
from pydantic.experimental.missing_sentinel import MISSING

from .permissions import Permission
from .wire import WireModel, WireRequest

WorkspaceRole = Literal["owner", "admin", "member", "viewer"]

_EMAIL = r"^[^\s@]+@[^\s@]+\.[^\s@]+$"


class Account(WireModel):
    account_id: UUID
    email: str = Field(pattern=_EMAIL, max_length=320)
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    locale: Literal["en", "fr"]
    verified: bool


class AccountUpdate(WireRequest):
    display_name: str | None = Field(default=None, min_length=1, max_length=120)
    locale: Literal["en", "fr"] | MISSING = MISSING


class Bootstrap(WireModel):
    account: Account
    workspaces: tuple["Workspace", ...]
    current_workspace_id: UUID
    csrf_token: str = Field(min_length=1)
    session_expires_at: datetime


class SignupRequest(WireRequest):
    email: str = Field(pattern=_EMAIL, max_length=320)
    password: SecretStr = Field(min_length=12, max_length=1024)
    locale: Literal["en", "fr"] = "en"


class LoginRequest(WireRequest):
    email: str = Field(pattern=_EMAIL, max_length=320)
    password: SecretStr = Field(min_length=1, max_length=1024)


class TokenRequest(WireRequest):
    token: SecretStr = Field(min_length=32, max_length=512)


class RecoveryRequest(WireRequest):
    email: str = Field(pattern=_EMAIL, max_length=320)


class PasswordResetRequest(TokenRequest):
    password: SecretStr = Field(min_length=12, max_length=1024)


class Workspace(WireModel):
    workspace_id: UUID
    name: str = Field(min_length=1, max_length=120)
    role: WorkspaceRole
    """A coarse label DERIVED from the member's company permissions (`CompanyAccess.role`), kept
    because released clients require it: rights themselves are `CompanyAccess.permissions`,
    never this label."""


class CompanyAccess(WireModel):
    """What the signed-in account may do in one company: the union of its company groups — the
    SAME set the server enforces, so a client shows only controls that will be accepted."""

    workspace_id: UUID
    permissions: tuple[Permission, ...]
    groups: tuple[str, ...]
    """Names of the company groups the account is in, for display ("Owners", "Billing")."""

    @property
    def role(self) -> WorkspaceRole:
        held = set(self.permissions)
        return "owner" if "company.delete" in held else "admin" if "company.members.manage" in held else "member" if "workflows.edit" in held else "viewer"


class WorkspaceCreate(WireRequest):
    name: str = Field(min_length=1, max_length=120)


class WorkspaceUpdate(WireRequest):
    name: str = Field(min_length=1, max_length=120)


class WorkspaceDeleteRequest(WireRequest):
    """The confirmation an owner-only, irreversible workspace delete requires: the workspace's
    OWN current name, typed back — checked server-side against the real row, never trusted from
    an earlier client read."""

    confirm_name: str = Field(min_length=1, max_length=120)


class CompanyDetails(WireModel):
    """Optional legal identity for an existing workspace, never a membership grant."""

    legal_name: str | None = Field(default=None, min_length=1, max_length=240)
    siret: str | None = Field(default=None, pattern=r"^[0-9]{14}$")
    address: str | None = Field(default=None, min_length=1, max_length=500)
    postal_code: str | None = Field(default=None, min_length=1, max_length=32)
    city: str | None = Field(default=None, min_length=1, max_length=120)
    country_code: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    activity_code: str | None = Field(default=None, min_length=1, max_length=32)


class CompanyProfile(CompanyDetails):
    workspace_id: UUID
    revision: int = Field(ge=0)
    logo_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class CompanyProfileUpdate(CompanyDetails, WireRequest):
    expected_revision: int = Field(ge=0)


class CompanyLogoUpload(WireRequest):
    expected_revision: int = Field(ge=0)
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    data_base64: str = Field(min_length=1, max_length=1400000)


class CompanyLookupResult(WireModel):
    company: CompanyDetails
    retrieved_at: datetime
    source_url: Literal["https://recherche-entreprises.api.gouv.fr"] = "https://recherche-entreprises.api.gouv.fr"


class WorkspaceMember(WireModel):
    account: Account
    group_ids: tuple[UUID, ...]
    """This company's groups the member is in (ids into the company's own group list)."""


class WorkspaceInvitation(WireModel):
    id: UUID
    email: str = Field(pattern=_EMAIL, max_length=320)
    group_ids: tuple[UUID, ...]
    """Groups the invitee joins on acceptance; empty = a read-only member."""
    status: Literal["pending", "accepted", "cancelled", "expired"]
    created_at: datetime
    expires_at: datetime


class WorkspaceMembership(WireModel):
    members: tuple[WorkspaceMember, ...]
    invitations: tuple[WorkspaceInvitation, ...]


class WorkspaceInvite(WireRequest):
    email: str = Field(pattern=_EMAIL, max_length=320)
    group_ids: tuple[UUID, ...] = ()


PlatformErrorCode = Literal[
    "authentication_failed", "csrf_failed", "invalid_origin", "invalid_request",
    "last_sign_in_method", "link_expired", "not_found", "not_linked", "permission_denied", "rate_limited",
    "verification_failed", "recovery_failed", "unavailable", "upgrade_required",
]
"""The wire's complete failure vocabulary, and the single source the server's own raisable set
binds to (`server.errors.ErrorCode`) — a code can never reach a client without being
in the contract that client's types are generated from. `link_expired`: a one-time link the
caller presented is gone (expired, already consumed, or never issued), which no retry of the
same link can fix — distinct from `authentication_failed`, where the CREDENTIAL was wrong and
retrying is exactly the right move. `last_sign_in_method`: disconnecting a linked Google
identity was refused because the account has no password and this is its last one — the only
door out, so it is never removed silently. `not_linked`: the email named in a disconnect
request is not one of the caller's own linked Google identities. `permission_denied`: the caller
is authenticated and (where applicable) a workspace member, but lacks the specific permission the
route requires — distinct from `not_found`, which this codebase uses to obscure a resource's
existence from a caller with no relationship to it at all. `upgrade_required`: a client released
before the contract it writes grew a field sent a body that would reset that field on the stored
record; `detail` names the fields — the client must be upgraded to write it."""


class PlatformError(WireModel):
    code: PlatformErrorCode
    detail: str | MISSING = MISSING
    """Why, in words a person can act on, when the code alone cannot say it (e.g. which
    permission a refused group change would have needed)."""
