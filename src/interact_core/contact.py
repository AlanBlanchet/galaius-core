"""The public site's contact form: what a visitor submits, and what the admin inbox reads.

The submission crosses a trust boundary (anonymous visitor -> database -> admin browser), so the
model owns every size and character bound once; the server stores only a validated
`ContactSubmission` and the admin UI renders its text as text, never markup."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator

from .wire import WireModel

ContactLocale = Literal["en", "fr"]

_LINE = r"^[^\x00-\x1f\x7f]*$"
"""One line of text: no control character (a name or company never carries a newline)."""
_TEXT = r"^[^\x00-\x08\x0b\x0c\x0e-\x1f\x7f]*$"
"""Multi-line text: tab, newline and carriage return only."""
_EMAIL = r"^[^\s@\x00-\x1f\x7f]+@[^\s@\x00-\x1f\x7f]+\.[^\s@\x00-\x1f\x7f]+$"


class ContactSubmission(WireModel):
    """`POST /v1/site/contact`'s body. Surrounding whitespace is stripped before the bounds apply,
    so a blank name is `required`, never a one-space name. `website` is the honeypot: the server
    answers a filled one like a stored message and drops it before this model validates it."""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=120, pattern=_LINE)
    email: str = Field(min_length=1, max_length=254, pattern=_EMAIL)
    company: str | None = Field(default=None, max_length=160, pattern=_LINE)
    message: str = Field(min_length=1, max_length=4000, pattern=_TEXT)
    locale: ContactLocale
    website: str = Field(default="", max_length=0)

    @model_validator(mode="after")
    def blank_company_is_none(self) -> Self:
        if self.company == "":
            object.__setattr__(self, "company", None)
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
    name: str
    email: str
    company: str | None = None
    message: str
    locale: ContactLocale
    read_at: datetime | None = None


class ContactInbox(WireModel):
    messages: tuple[ContactMessage, ...]
    """Newest first, at most the server's inbox page size; `total` counts every stored message."""
    total: int = Field(ge=0)
    unread: int = Field(ge=0)
