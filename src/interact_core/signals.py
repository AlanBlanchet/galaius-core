"""Signals: a signed-in user flags something in the app ("this is broken here"), and the app says
where it came from without the user having to.

The browser captures the context (screen, route, the element pointed at, ids in view, recent
console errors and failed requests, build, locale, viewport) and the user adds words and, if they
want, a cropped / blurred screenshot. Everything captured crosses a trust boundary (a member's
browser -> database -> the admin's browser), so this module owns every bound once and REDACTS
secrets on validation: whatever a client sends, a stored signal never holds a token, a key, a
password value or a URL query value, and captured text holds no email address. Request and response
BODIES are never part of the contract, so data contents cannot ride along."""

import base64
import binascii
import re
from datetime import datetime
from typing import Annotated, ClassVar, Literal
from uuid import UUID

from pydantic import AfterValidator, Field, model_validator

from .wire import WireModel

SignalStatus = Literal["new", "triaged", "fixed"]
SignalScreenshotType = Literal["image/png", "image/jpeg", "image/webp"]


class SignalRedaction:
    """The one redaction pass every captured string goes through. `secrets` removes credentials
    and URL query values (kept: the parameter NAMES, which say what failed); `anonymous` also
    removes email addresses. UUIDs and paths survive: they point at the run / workflow at fault."""

    MARK: ClassVar[str] = "[redacted]"
    _SECRETS: ClassVar[tuple[tuple[re.Pattern[str], str], ...]] = (
        (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{6,}"), r"\1 [redacted]"),
        (re.compile(r"\beyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*"), "[redacted]"),
        (re.compile(r"\b(?:sk|pk|rk|iwk|ghp|gho|ghs|github_pat|xox[abprs]|AIza|ya29|glpat)[-_][A-Za-z0-9._-]{6,}"), "[redacted]"),
        (re.compile(r"(?i)\b((?:api[_-]?key|access[_-]?token|refresh[_-]?token|id[_-]?token|client[_-]?secret|token|secret|password|passwd|authorization|cookie)[\"']?\s*[:=]\s*[\"']?)[^\s\"'&,;}#]+"), r"\1[redacted]"),
        (re.compile(r"([?&])([^=&#\s?]{1,64})=[^&#\s]*"), r"\1\2=[redacted]"),
    )
    _EMAIL: ClassVar[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    _BLOB: ClassVar[re.Pattern[str]] = re.compile(r"[A-Za-z0-9+/_=-]{32,}")
    _UUID: ClassVar[re.Pattern[str]] = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

    @classmethod
    def secrets(cls, value: str) -> str:
        for pattern, replacement in cls._SECRETS:
            value = pattern.sub(replacement, value)
        return cls._BLOB.sub(cls._blob, value)

    @classmethod
    def anonymous(cls, value: str) -> str:
        return cls._EMAIL.sub("[email]", cls.secrets(value))

    @classmethod
    def _blob(cls, match: re.Match[str]) -> str:
        """A long opaque run is a token unless it is only path words and UUIDs: each `/`-separated
        piece, UUIDs removed, is dropped when 32+ characters mixing letters and digits remain."""
        pieces = [cls._UUID.sub("", piece) for piece in match.group(0).split("/")]
        opaque = any(len(piece) >= 32 and re.search(r"\d", piece) and re.search(r"[A-Za-z]", piece) for piece in pieces)
        return cls.MARK if opaque else match.group(0)


_LINE = r"^[^\x00-\x1f\x7f]*$"
_TEXT = r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]*$"
Captured = Annotated[str, AfterValidator(SignalRedaction.anonymous)]
"""Text the browser captured on its own (labels, console lines, paths): secrets and emails removed."""
Written = Annotated[str, AfterValidator(SignalRedaction.secrets)]
"""Text the user typed: kept as written, except anything shaped like a credential."""


class SignalBox(WireModel):
    """Where the element sat in the viewport, CSS pixels."""

    x: int = Field(ge=-100_000, le=100_000)
    y: int = Field(ge=-100_000, le=100_000)
    width: int = Field(ge=0, le=100_000)
    height: int = Field(ge=0, le=100_000)


class SignalElement(WireModel):
    """The element the user pointed at: a CSS selector that finds it again, the named area it sits
    in, and the words it shows (its label, never a form value)."""

    selector: Captured = Field(min_length=1, max_length=600, pattern=_LINE)
    area: Captured | None = Field(default=None, max_length=160, pattern=_LINE)
    label: Captured | None = Field(default=None, max_length=160, pattern=_LINE)
    box: SignalBox | None = None


class SignalConsoleLine(WireModel):
    level: Literal["error", "warning"]
    message: Captured = Field(max_length=500, pattern=_TEXT)
    at: datetime


class SignalFailedRequest(WireModel):
    """A request this page made that failed: method, path (query values redacted), status (0 = no
    answer at all). Bodies are never captured."""

    method: str = Field(pattern=r"^[A-Z]{3,7}$")
    path: Captured = Field(min_length=1, max_length=400, pattern=_LINE)
    status: int = Field(ge=0, le=599)
    at: datetime


class SignalRef(WireModel):
    """An id visible on the screen, with what it names (`workflow`, `run`, `agent`, `node`, ...)."""

    kind: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,80}$")


class SignalViewport(WireModel):
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    pixel_ratio: float = Field(gt=0, le=16)


class SignalContext(WireModel):
    """What the browser knew when the user signalled: nothing here is typed by the user."""

    route: Captured = Field(min_length=1, max_length=600, pattern=r"^/([^/\\\x00-\x20\x7f][^\\\x00-\x20\x7f]*)?$")
    """Path + hash of the screen, same origin only (`/#workflows/<id>`): the admin's deep link."""
    screen: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,79}$")
    """The app's view name (`workflows`, `runs`, `admin`, ...)."""
    area: Captured | None = Field(default=None, max_length=160, pattern=_LINE)
    """The named part of the screen: the pointed element's area, else the open panel / dialog."""
    title: Captured | None = Field(default=None, max_length=200, pattern=_LINE)
    """The screen's heading as the user read it."""
    element: SignalElement | None = None
    workspace_id: UUID | None = None
    refs: tuple[SignalRef, ...] = Field(default=(), max_length=40)
    locale: str = Field(pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
    viewport: SignalViewport
    user_agent: Captured = Field(max_length=400, pattern=_LINE)
    client_build: str | None = Field(default=None, max_length=80, pattern=r"^[A-Za-z0-9._+:-]+$")
    """The server build this browser tab loaded with; differs from the stored `server_build` when
    the tab is stale."""
    console: tuple[SignalConsoleLine, ...] = Field(default=(), max_length=20)
    requests: tuple[SignalFailedRequest, ...] = Field(default=(), max_length=20)


class SignalScreenshot(WireModel):
    """The optional picture, already cropped / blurred by the user in the browser."""

    MAX_BYTES: ClassVar[int] = 2 * 1024 * 1024
    _MAGIC: ClassVar[dict[str, bytes]] = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff", "image/webp": b"RIFF"}

    media_type: SignalScreenshotType
    data_base64: str = Field(min_length=16, max_length=(MAX_BYTES * 4) // 3 + 4)

    @model_validator(mode="after")
    def is_that_image(self) -> "SignalScreenshot":
        """The bytes must be the declared type (webp: RIFF....WEBP); anything else is refused."""
        try:
            raw = self.data
        except (binascii.Error, ValueError) as error:
            raise ValueError("screenshot is not base64") from error
        if len(raw) > self.MAX_BYTES or not raw.startswith(self._MAGIC[self.media_type]) or (self.media_type == "image/webp" and raw[8:12] != b"WEBP"):
            raise ValueError("screenshot is not the declared image type")
        return self

    @property
    def data(self) -> bytes:
        return base64.b64decode(self.data_base64, validate=True)


class SignalSubmission(WireModel):
    """`POST /v1/signals`. `id` is chosen by the client and is the idempotency key: resending the
    same submission answers the stored signal, never a second one."""

    id: UUID
    description: Written = Field(min_length=1, max_length=4000, pattern=_TEXT)
    context: SignalContext
    screenshot: SignalScreenshot | None = None


class SignalReceipt(WireModel):
    """201 (`created`) or 200 (the same id was already stored by this account)."""

    id: UUID
    created: bool


class SignalReporter(WireModel):
    id: UUID
    name: str
    email: str


class SignalWorkspace(WireModel):
    id: UUID
    name: str


class Signal(WireModel):
    """One stored signal as Admin -> Signals reads it."""

    id: UUID
    created_at: datetime
    status: SignalStatus
    status_at: datetime | None = None
    description: str
    context: SignalContext
    server_build: str | None = None
    reporter: SignalReporter
    workspace: SignalWorkspace | None = None
    has_screenshot: bool


class SignalFacet(WireModel):
    """One filter value and how many stored signals carry it."""

    value: str
    count: int = Field(ge=0)


class SignalList(WireModel):
    """`GET /v1/operator/signals`: newest first, filtered; the facets count every stored signal
    so choosing a filter never hides the other choices."""

    signals: tuple[Signal, ...]
    matching: int = Field(ge=0)
    total: int = Field(ge=0)
    statuses: tuple[SignalFacet, ...]
    screens: tuple[SignalFacet, ...]
    areas: tuple[SignalFacet, ...]
    builds: tuple[SignalFacet, ...]


class SignalStatusUpdate(WireModel):
    status: SignalStatus
