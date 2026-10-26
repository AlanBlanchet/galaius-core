"""The community library: prompts, agents and skills every signed-in account can read — published by a person, picked
by Galaius, or imported from an open-licence repository pinned at a commit (its licence and the file's link kept). It
is never part of a workspace's own library: a person copies one in, or an agent starts one as a child it controls.

    GET /v1/community/prompts?q=&kind=&sort=&cursor=   -> CommunityPage
    GET /v1/community/prompts/{id}                     -> CommunityPrompt
    GET /v1/community/sources                          -> CommunitySources

A repository whose files may not be copied (no licence, copyleft, non-commercial, a vendor's own prompt text) is a
`CommunitySource` with its link and no entry: listed, never stored, never started."""

import base64
from datetime import datetime
from typing import ClassVar, Literal
from uuid import UUID

from pydantic import Field, model_validator

from .wire import WireModel, WireRequest

CommunityKind = Literal["prompt", "agent", "skill"]
"""What an entry is: a prompt (instructions for one task), an agent (a role's system prompt), a skill (instructions an
agent loads for one kind of work)."""
CommunitySort = Literal["popular", "recent"]
CommunityLicenceClass = Literal["copy", "reference"]
"""`copy`: its licence lets its files be copied with their notice (MIT, Apache-2.0, CC0, CC-BY); `reference`: link only."""
_COMMIT = r"^[0-9a-f]{40}$"


class CommunityAuthor(WireModel):
    """Who published it, as every reader sees it: a name, never an email or an account id.
    `source`: an open-licence collection imported as a whole (its name)."""

    display_name: str = Field(min_length=1, max_length=120)
    kind: Literal["person", "galaius", "source"]


class CommunityPrompt(WireModel):
    """One community entry. `body` is its text exactly as its workspace stored it at `source_digest` (front-matter
    included, parsed by readers as any workspace prompt), or as its pinned source file reads."""

    id: UUID
    kind: CommunityKind = "prompt"
    reviewed: bool = False
    """Its text was screened when it was imported, at its pinned commit: an agent may be started from it as a community
    agent. An entry whose text changes is screened again before it may."""
    title: str = Field(min_length=1, max_length=80)
    summary: str = Field(min_length=1, max_length=280)
    body: str = Field(min_length=1)
    tags: tuple[str, ...]
    author: CommunityAuthor
    published_at: datetime
    copies_total: int = Field(ge=0)
    """Workspaces that copied it (once each)."""
    uses_total: int = Field(ge=0)
    """Agents, in any workspace, whose current revision uses a copy of it (prompt or paradigm)."""
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    licence: str | None = None
    """An imported entry's licence (SPDX id, e.g. `MIT`, `CC0-1.0`); None: published here."""
    origin_url: str | None = None
    """Where an imported entry comes from (the file at its pinned commit)."""

    @model_validator(mode="after")
    def screened_agents_only(self) -> "CommunityPrompt":
        """Only an agent or a skill is screened: a prompt is never started as an agent."""
        if self.reviewed and self.kind == "prompt":
            raise ValueError("a prompt is never screened to start as an agent")
        return self


class CommunityPage(WireModel):
    entries: tuple[CommunityPrompt, ...]
    next: str | None
    sort: CommunitySort


class CommunityQuery(WireRequest):
    """`q`: words (`words`), each held by every entry found (in its title, summary or text), one holding them in its
    title ranking first. `kind`: that kind only. `cursor`: opaque, the next page's position as `CommunityPage.next`
    returned it."""

    MAX_OFFSET: ClassVar[int] = 100_000
    q: str = Field(default="", max_length=200)
    kind: CommunityKind | None = None
    sort: CommunitySort = "popular"
    cursor: str | None = None

    def words(self) -> tuple[str, ...]:
        """`q`'s distinct words, lowercased; a one-letter word (« a », « y ») is in almost every text and finds nothing."""
        return tuple(dict.fromkeys(word for word in self.q.lower().split() if len(word) > 1))

    def offset(self) -> int:
        if self.cursor is None:
            return 0
        try:
            value = int(base64.urlsafe_b64decode(self.cursor.encode()).decode())
        except (ValueError, UnicodeError) as error:
            raise ValueError("cursor is not one this server issued") from error
        if not 0 <= value <= self.MAX_OFFSET:
            raise ValueError("cursor is out of range")
        return value

    @staticmethod
    def cursor_at(offset: int) -> str:
        return base64.urlsafe_b64encode(str(offset).encode()).decode()


class CommunitySource(WireModel):
    """One public repository the community library indexes, pinned at `commit`. `entries`: how many of its files are
    in the library (0 for a `reference` one, whose `note` says why it may not be copied)."""

    name: str = Field(pattern=r"^[a-z0-9-]+$", max_length=60)
    url: str = Field(pattern=r"^https://github\.com/[\w.-]+/[\w.-]+$", max_length=200)
    commit: str = Field(pattern=_COMMIT)
    licence: str = Field(min_length=1, max_length=60)
    licence_class: CommunityLicenceClass
    entries: int = Field(ge=0)
    note: str = Field(default="", max_length=280)


class CommunitySources(WireModel):
    sources: tuple[CommunitySource, ...]
