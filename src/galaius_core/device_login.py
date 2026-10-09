"""Adding a computer to an account from its terminal: `galaius login` (RFC 8628 device grant shape).

The CLI asks for a sign-in (`DeviceLoginStart`), shows the short `user_code` and opens the server's
/link page; the signed-in owner sees who asks (`DeviceLoginView`) and approves it for one workspace
(`DeviceLoginApproval`); the CLI's next poll (`DeviceTokenRequest`) receives the credentials the
approval issued (`DeviceLoginIssued`): the EXISTING machine token for this PC and an EXISTING
workspace API key for the CLI — no new credential kind."""

import re
import secrets
from datetime import datetime
from typing import Annotated, ClassVar, Literal
from uuid import UUID

from pydantic import AfterValidator, Field, SecretStr, field_validator

from .wire import WireModel, WireRequest
from .workflows import MachineSummary, WorkspaceApiKeyScope

#: A computer's name as the owner reads it on the approval page and in Machines: its hostname,
#: reduced to letters, digits, '.', '_' and '-' (nothing a page could render as markup).
_CLIENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


class UserCode:
    """The code a person compares between their terminal and the approval page: 8 letters from 20
    consonants (no vowels: no words; no look-alikes), shown as XXXX-XXXX, ~2.6e10 values."""

    ALPHABET: ClassVar[str] = "BCDFGHJKLMNPQRSTVWXZ"
    LENGTH: ClassVar[int] = 8
    PATTERN: ClassVar[re.Pattern] = re.compile(rf"^[{ALPHABET}]{{4}}-[{ALPHABET}]{{4}}$")

    @classmethod
    def new(cls) -> str:
        raw = "".join(secrets.choice(cls.ALPHABET) for _ in range(cls.LENGTH))
        return f"{raw[:4]}-{raw[4:]}"

    @classmethod
    def normalize(cls, value: str) -> str:
        """What a person typed (any case, spaces, with or without the dash) as the canonical code;
        ValueError when it cannot be one."""
        letters = re.sub(r"[\s-]", "", value).upper()
        code = f"{letters[:4]}-{letters[4:]}"
        if not cls.PATTERN.fullmatch(code):
            raise ValueError("not a sign-in code")
        return code


#: A sign-in code as a person typed or read it, held in its canonical XXXX-XXXX form.
UserCodeText = Annotated[str, AfterValidator(UserCode.normalize)]

#: pending → approved | denied (on /link) → issuing → consumed (the computer collected) → revoked (logout / machine removed).
DeviceLoginStatus = Literal["pending", "approved", "denied", "issuing", "consumed", "revoked"]


class DeviceLoginStart(WireRequest):
    """What the CLI says about the computer asking to join."""

    client_name: str
    platform: Literal["linux", "macos", "windows"]
    client_version: str = Field(min_length=1, max_length=40, pattern=r"^[0-9A-Za-z.+-]+$")
    #: The CLI may also start workflow runs (`galaius login --allow-runs`); otherwise it only reads.
    runs: bool = False

    @field_validator("client_name", mode="before")
    @classmethod
    def plain_name(cls, value: object) -> object:
        if isinstance(value, str):
            value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")[:63] or "computer"
        return value

    @field_validator("client_name")
    @classmethod
    def checked_name(cls, value: str) -> str:
        if not _CLIENT_NAME.fullmatch(value):
            raise ValueError("computer name must be letters, digits, '.', '_' or '-'")
        return value


class DeviceLoginStarted(WireModel):
    device_code: SecretStr
    user_code: UserCodeText
    verification_uri: str
    verification_uri_complete: str
    expires_in: int = Field(gt=0)
    interval: int = Field(gt=0)
    #: The number the computer shows next to the link, never in it: from another network the person
    #: picks it among `DeviceLoginView.choices` (a forwarded link alone never approves).
    match: str = Field(pattern=r"^\d{2}$")

    def revealed(self) -> dict[str, object]:
        """The JSON body a server sends (the device code written out: only the asking computer gets it)."""
        return {**self.model_dump(mode="json"), "device_code": self.device_code.get_secret_value()}


class DeviceLoginWorkspace(WireModel):
    id: UUID
    name: str


class DeviceLoginView(WireModel):
    """What the signed-in person sees before allowing a computer in."""

    user_code: UserCodeText
    client_name: str
    platform: Literal["linux", "macos", "windows"]
    client_version: str
    requested_at: datetime
    expires_at: datetime
    #: The address the request came from, as this server saw it.
    requested_from: str
    #: The viewer's browser and the computer reached this server from the same address. When not,
    #: the person picks the number the computer shows among `choices` (a forwarded link alone never approves).
    same_network: bool
    #: Three numbers, one of them the computer's (`DeviceLoginStarted.match`); none on the same network.
    choices: tuple[str, ...] = Field(default=(), max_length=3)
    #: The computer asked to start workflow runs too (`galaius login --allow-runs`), not only read.
    runs: bool
    #: Workspaces where the viewer may add a computer (`machines.manage`).
    workspaces: tuple[DeviceLoginWorkspace, ...]


class DeviceLoginApproval(WireRequest):
    workspace_id: UUID
    #: The number the person picked among `DeviceLoginView.choices`: required when `same_network` is
    #: false; a wrong one refuses the request for good (`match_refused`, the computer asks again).
    match: str | None = Field(default=None, pattern=r"^\d{2}$")


class DeviceTokenRequest(WireRequest):
    device_code: SecretStr = Field(min_length=16, max_length=128)


DeviceTokenError = Literal["authorization_pending", "slow_down", "access_denied", "expired_token", "invalid_grant"]


class DeviceTokenRefusal(WireModel):
    """The token endpoint's answer (HTTP 400) while it issues nothing."""

    error: DeviceTokenError


class DeviceLoginKey(WireModel):
    id: UUID
    secret: SecretStr
    scopes: tuple[WorkspaceApiKeyScope, ...]


class DeviceLoginIssued(WireModel):
    """The approval's credentials, handed to the computer that asked (only it holds the device code)."""

    workspace: DeviceLoginWorkspace
    #: The account that allowed it in — the CLI shows it, so a person who never asked notices.
    approved_by: str
    machine: MachineSummary
    machine_token: SecretStr
    api_key: DeviceLoginKey

    def revealed(self) -> dict[str, object]:
        """The JSON body a server sends (secrets written out: this is their one delivery)."""
        body = self.model_dump(mode="json")
        body["machine_token"] = self.machine_token.get_secret_value()
        body["api_key"]["secret"] = self.api_key.secret.get_secret_value()
        return body


class DeviceLogout(WireModel):
    api_key: UUID
    machine: UUID | None
