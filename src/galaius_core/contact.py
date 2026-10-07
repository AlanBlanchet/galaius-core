"""The public site's contact form: what a visitor submits, and what the admin inbox reads.

The submission crosses a trust boundary (anonymous visitor -> database -> admin browser), so the
model owns every size and character bound once; the server stores only a validated
`ContactSubmission` and the admin UI renders its text as text, never markup."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from .wire import WireModel, WireRequest

ContactLocale = Literal["en", "fr"]
ContactKind = Literal["contact", "poc"]
"""What the visitor asks for: a plain message, or a free proof of concept on a subject of theirs."""

_SPOOF = "\u0080-\u009f\u200b-\u200f\u2028-\u202e\u2066-\u2069"
"""C1 controls, zero-width and bidi-override characters: they read as nothing and rewrite what follows."""
_LINE = rf"^[^\x00-\x1f\x7f{_SPOOF}]*$"
"""One line of text: no control character (a name or company never carries a newline)."""
_TEXT = rf"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f{_SPOOF}]*$"
"""Multi-line text: tab, newline and carriage return only."""
_EMAIL = r"^[^\s@\x00-\x1f\x7f?#&%/<>\"']+@[^\s@\x00-\x1f\x7f?#&%/<>\"']+\.[^\s@\x00-\x1f\x7f?#&%/<>\"']+$"
"""An address a `mailto:` can carry as-is: no query / fragment / quoting character (RFC 6068 header injection)."""


class ContactSubmission(WireRequest):
    """`POST /v1/site/contact`'s body. Surrounding whitespace is stripped before the bounds apply,
    so a blank name is `required`, never a one-space name. `website` is the honeypot: the server
    answers a filled one like a stored message and drops it before this model validates it.

    A `poc` request names its `subject` (what the proof of concept is about) and may name a
    `sector`; the data itself never travels here — it is exchanged after the first contact."""

    model_config = ConfigDict(str_strip_whitespace=True)

    kind: ContactKind = "contact"
    name: str = Field(min_length=1, max_length=120, pattern=_LINE)
    email: str = Field(min_length=1, max_length=254, pattern=_EMAIL)
    company: str | None = Field(default=None, max_length=160, pattern=_LINE)
    sector: str | None = Field(default=None, max_length=80, pattern=_LINE)
    subject: str | None = Field(default=None, max_length=200, pattern=_LINE)
    message: str = Field(min_length=1, max_length=4000, pattern=_TEXT)
    locale: ContactLocale
    website: str = Field(default="", max_length=0)

    @model_validator(mode="after")
    def blank_optional_is_none_and_poc_names_its_subject(self) -> Self:
        for field in ("company", "sector", "subject"):
            if getattr(self, field) == "":
                object.__setattr__(self, field, None)
        if self.kind == "poc" and self.subject is None:
            raise ValueError("a POC request names its subject")
        return self


ContactFieldReason = Literal["required", "too_long", "invalid", "unexpected"]


class ContactRejection(WireModel):
    """400: which fields failed and why, as a word the site maps to its own copy. `body` names
    a request that is not a JSON object or is larger than any valid one."""

    error: Literal["invalid_request"] = "invalid_request"
    fields: dict[str, ContactFieldReason]


class ContactRateLimited(WireModel):
    """429: seconds until this sender's next message can be accepted."""

    error: Literal["rate_limited"] = "rate_limited"
    retry_after: int = Field(ge=1)


class ContactAccepted(WireModel):
    """202: stored — or a honeypot submission dropped; a sender cannot tell the two apart."""

    ok: Literal[True] = True


class ContactMessage(WireModel):
    """One stored message as the admin inbox reads it; the sender's hashed address stays server-side."""

    id: UUID
    created_at: datetime
    kind: ContactKind = "contact"
    name: str
    email: str
    company: str | None = None
    sector: str | None = None
    subject: str | None = None
    message: str
    locale: ContactLocale
    read_at: datetime | None = None


class ContactInbox(WireModel):
    messages: tuple[ContactMessage, ...]
    """Newest first, at most the server's inbox page size; `total` counts every stored message."""
    total: int = Field(ge=0)
    unread: int = Field(ge=0)
