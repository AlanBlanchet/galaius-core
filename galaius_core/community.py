"""The community library: prompts, agents and skills every signed-in account can read — published by a person, picked
by Galaius, or imported from an open-licence repository pinned at a commit (its licence and the file's link kept). It
is never part of a workspace's own library: a person copies one in, or an agent starts one as a child it controls.

    GET /v1/community/prompts?q=&kind=&sort=&listed=&cursor=   -> CommunityPage
    GET /v1/community/prompts/{id}                     -> CommunityPrompt
    GET /v1/community/sources                          -> CommunitySources
    GET /v1/community/entries?q=&kind=&sort=&offset=   -> CommunityEntry[]       (every entry, listed ones included)
    GET /v1/community/entries/{id}                     -> CommunityEntryDetail
    POST /v1/workspaces/{w}/community/entries/{id}/launch  CommunityLaunchRequest -> CommunityLaunchResult

A repository whose files may not be copied (no licence, copyleft, non-commercial, a vendor's own prompt text) is a
`CommunitySource` with its link and no entry: listed, never stored, never started. What a public search finds (a
repository shipping agent tooling, an MCP server of a registry, a package) is an entry `listed` by link: its description
and popularity, never its files."""

import base64
import re
import unicodedata
from datetime import datetime
from typing import Annotated, ClassVar, Literal
from uuid import UUID

from pydantic import BeforeValidator, ConfigDict, Field, model_validator

from galaius_core.wire import WireModel, WireRequest

CommunityKind = Literal["prompt", "agent", "skill", "mcp", "pipeline"]
"""What an entry is: a prompt (instructions for one task), an agent (a role's system prompt), a skill (instructions an
agent loads for one kind of work), an MCP server (a program whose tools agents call), a pipeline (agents wired together
by a project). An MCP server and a pipeline are always `listed`: their code is never in the library."""
LISTED_KINDS: frozenset[str] = frozenset({"mcp", "pipeline"})
CommunitySort = Literal["popular", "recent"]
CommunityLicenceClass = Literal["copy", "reference"]
"""`copy`: its licence lets its files be copied with their notice (MIT, Apache-2.0, CC0, CC-BY); `reference`: link only."""
_COMMIT = r"^[0-9a-f]{40}$"


_BREAKING = frozenset({"Cc", "Zl", "Zp"})
"""Unicode categories that break or steer a line: controls (C0, C1: a terminal escape), line and paragraph separators."""


def visible(text: str) -> str:
    """`text` as it reads: compatibility forms folded (a full-width bracket is a bracket), invisible format characters
    (zero-width spaces, direction marks) dropped."""
    return "".join(char for char in unicodedata.normalize("NFKC", text) if unicodedata.category(char) != "Cf")


def plain_line(text: object, limit: int) -> str:
    """A stranger's short text (a title, a name) as it reads (`visible`), on one line: no control character, no bracket,
    cut to `limit`."""
    spaced = "".join(" " if unicodedata.category(char) in _BREAKING else char for char in visible(str(text or "")))
    return " ".join(spaced.split()).replace("[", "(").replace("]", ")")[:limit].strip()


TITLE_MAX, AUTHOR_MAX, LICENCE_MAX = 80, 120, 40
"""Characters of an entry's title, of its author's name, of its licence (an SPDX id)."""


def PlainLine(limit: int) -> object:
    """A stranger's short text as a field: one plain line (`plain_line`) cut to `limit`, "?" when nothing is left."""
    return Annotated[str, BeforeValidator(lambda value: plain_line(value, limit) or "?"), Field(min_length=1, max_length=limit)]


def _licence(value: object) -> str | None:
    """A licence as one plain line, None when there is none or it is longer than an SPDX id (`LICENCE_MAX`): never cut."""
    said = plain_line(value, LICENCE_MAX + 1)
    return said if 0 < len(said) <= LICENCE_MAX else None


def _https(value: object) -> str | None:
    """A link as it may be shown and opened: https, at most 1024 characters, no space, control or invisible character;
    None otherwise."""
    if not isinstance(value, str) or not value.startswith("https://") or len(value) > 1024:
        return None
    return None if any(char.isspace() or unicodedata.category(char) in {*_BREAKING, "Cf"} for char in value) else value


Licence = Annotated[str | None, BeforeValidator(_licence)]
"""An entry's licence (SPDX id) as every reader shows it (`_licence`)."""
HttpsLink = Annotated[str | None, BeforeValidator(_https)]
"""An entry's pinned source as every reader shows and opens it (`_https`)."""


CommunityRunKind = Literal["agent", "skill"]
"""What a run can be started from: an agent or a skill, never a prompt (`CommunityPrompt.reviewed`)."""


class CommunityAuthor(WireModel):
    """Who published it, as every reader sees it: a name, never an email or an account id.
    `source`: an open-licence collection imported as a whole (its name)."""

    display_name: str = Field(min_length=1, max_length=AUTHOR_MAX)
    kind: Literal["person", "galaius", "source"]


class CommunityPrompt(WireModel):
    """One community entry. `body` is its text exactly as its workspace stored it at `source_digest` (front-matter
    included, parsed by readers as any workspace prompt), or as its pinned source file reads."""

    id: UUID
    kind: CommunityKind = "prompt"
    reviewed: bool = False
    """Its text was screened when it was imported, at its pinned commit: an agent may be started from it as a community
    agent. An entry whose text changes is screened again before it may."""
    title: str = Field(min_length=1, max_length=TITLE_MAX)
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
    licence: Licence = None
    """An imported entry's licence (SPDX id, e.g. `MIT`, `CC0-1.0`); None: published here."""
    origin_url: HttpsLink = None
    """Where an imported entry comes from (the file at its pinned commit; a `listed` one: its repository or package page)."""
    listed: bool = False
    """Found by a public search, its files not in the library: `body` is its description; never copied, never started."""
    stars: int | None = Field(default=None, ge=0)
    """Stars of its repository, as last read; None: not a repository anyone can count (yet)."""
    downloads: int | None = Field(default=None, ge=0)
    """Downloads of its package over the last month on its registry; None: no package."""
    source_updated_at: datetime | None = None
    """Its source's last change — its repository's last push, its package's last release — as last read; None: not read."""

    @model_validator(mode="after")
    def screened_agents_only(self) -> "CommunityPrompt":
        """Only an agent or a skill whose text is in the library is screened: a prompt is never started as an agent, a
        `listed` entry has no text to start; an MCP server is always `listed`."""
        if self.reviewed and (self.kind == "prompt" or self.listed):
            raise ValueError("only an agent or a skill whose text is in the library is screened to start as an agent")
        if self.kind in LISTED_KINDS and not self.listed:
            raise ValueError(f"a {self.kind} is listed by link, never stored")
        return self


class CommunityEntryRef(WireModel):
    """The community entry an agent run was started from, as every reader of the run shows it (its PC's run list, the
    server's mirror and ledger, the agents panel): its id, kind, one-line title, author, licence and pinned source — never
    its text, nor any other key a record holds. Strangers' words: `title` and `author` arrive as one plain line, cut; a
    licence too long or a link that is not https is dropped, so an entry never fails to read (a community run that lost
    its mark would pass for one of his own roles)."""

    model_config = ConfigDict(extra="ignore")

    entry: UUID
    kind: CommunityRunKind
    title: PlainLine(TITLE_MAX)  # type: ignore[valid-type]
    author: PlainLine(AUTHOR_MAX)  # type: ignore[valid-type]
    licence: Licence = None
    origin_url: HttpsLink = None

    @classmethod
    def of(cls, value: object) -> "CommunityEntryRef | None":
        """It out of a run record's `community` (which holds more: the entry's text, its folders); None for a run started
        from no entry, or a record naming no entry (no id, or a kind no run starts from). Its other fields never refuse it."""
        try:
            return cls.model_validate(value) if isinstance(value, dict) else None
        except ValueError:
            return None

    def ref(self) -> "CommunityEntryRef":
        """Its own fields alone, whatever subclass holds them (a run's `CommunityAgent`): what crosses to another reader."""
        return CommunityEntryRef.model_validate(self.model_dump(include=set(CommunityEntryRef.model_fields)))


class CommunityPage(WireModel):
    entries: tuple[CommunityPrompt, ...]
    next: str | None
    sort: CommunitySort


class CommunityQuery(WireRequest):
    """`q`: words (`words`), each held by every entry found (in its title, summary or text), one holding them in its
    title ranking first. `kind`: that kind only. `listed`: entries `listed` by link too (every MCP server among them), and,
    when the library holds few for `q`, the public sources searched for it now; a reader released before them never asks,
    so never reads a kind it does not know. `cursor`: opaque, the next page's position as `CommunityPage.next` returned it."""

    MAX_OFFSET: ClassVar[int] = 100_000
    q: str = Field(default="", max_length=200)
    kind: CommunityKind | None = None
    sort: CommunitySort = "popular"
    listed: bool = False
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


CommunityEntrySort = Literal["popularity", "track_record", "recent"]
"""popularity: its repository's stars, then its package's downloads, then copies here; track_record: its runs here that
ended well, then popularity; recent: its source's last change."""
SUMMARY_SHOWN = 120
ACTION_MAX = 80


class CommunitySourceRef(WireModel):
    """Where an entry comes from: its repository's or package's page, the file in it (None: the whole repository), the
    commit it is pinned at (None: not pinned, a listed one), its licence (SPDX id; None: not known)."""

    url: HttpsLink = None
    path: str | None = Field(default=None, max_length=1024)
    commit: str | None = Field(default=None, pattern=_COMMIT)
    licence: Licence = None


class CommunityPopularity(WireModel):
    """How popular it is outside and here, as last read: its repository's stars and forks, its last push (or release),
    the workspaces here that copied it."""

    stars: int | None = Field(default=None, ge=0)
    forks: int | None = Field(default=None, ge=0)
    downloads: int | None = Field(default=None, ge=0)
    """Its package's downloads over the last month; None: no package."""
    last_push: datetime | None = None
    installs_here: int = Field(default=0, ge=0)


class CommunityTrackRecord(WireModel):
    """What it has built: public projects that name it (`built_with`, their links, the most starred first), and its runs
    on this platform the reader may see — how many, and the share of those that ended that finished (None: none ended)."""

    built_with: tuple[str, ...] = ()
    runs_here: int = Field(default=0, ge=0)
    success_rate: float | None = Field(default=None, ge=0, le=1)


class CommunityAction(WireModel):
    """A tool or command its own file declares (front-matter `tools`, `allowed-tools`): its name and one line on it."""

    name: PlainLine(ACTION_MAX)  # type: ignore[valid-type]
    summary: str = Field(default="", max_length=ACTION_MAX)


class CommunityScreen(WireModel):
    """Whether its text was screened for steering beyond its job before an agent may start from it: `safe`, `refused`
    (`reason` says what it does), or `unscreened` (not yet, too long to read whole, or no text in the library)."""

    state: Literal["unscreened", "safe", "refused"]
    reason: str | None = Field(default=None, max_length=280)


class CommunityLaunch(WireModel):
    """Whether it can be started from here and how: `spawn` an agent from its text, `skill-load` a run carrying the skill,
    `mcp` a server added to a PC (not from this page); None when it cannot."""

    runnable: bool
    how: Literal["spawn", "mcp", "skill-load"] | None = None

    @classmethod
    def of(cls, kind: CommunityKind, reviewed: bool, listed: bool) -> "CommunityLaunch":
        """The one rule of what starts: an agent or a skill whose text is in the library, screened safe."""
        runnable = reviewed and not listed and kind in ("agent", "skill")
        return cls(runnable=runnable, how={"agent": "spawn", "skill": "skill-load"}[kind] if runnable else "mcp" if kind == "mcp" else None)


class CommunityEntry(WireModel):
    """One entry of the community library as the platform shows and compares it (`CommunityPrompt`: the same entry as
    the Library copies it and an agent starts from it). Strangers' words: `name`, `summary`, `author`, `actions` are one
    plain line each, cut."""

    id: UUID
    kind: CommunityKind
    name: PlainLine(TITLE_MAX)  # type: ignore[valid-type]
    summary: str = Field(max_length=SUMMARY_SHOWN)
    author: PlainLine(AUTHOR_MAX)  # type: ignore[valid-type]
    listed: bool = False
    source: CommunitySourceRef
    popularity: CommunityPopularity
    track_record: CommunityTrackRecord
    actions: tuple[CommunityAction, ...] = ()
    launch: CommunityLaunch
    screened: CommunityScreen


class CommunityFile(WireModel):
    """One file of an entry as its source holds it, shown as written (markdown read as text: never run, never trusted)."""

    path: str = Field(min_length=1, max_length=1024)
    text: str


class CommunityEntryDetail(WireModel):
    entry: CommunityEntry
    files: tuple[CommunityFile, ...]


class CommunityEntriesQuery(WireRequest):
    """`q`: words, as `CommunityQuery.q`; the library holding few for them asks its public sources now. `offset`: entries
    skipped, a page being `LIMIT` of them, `MAX_OFFSET` at most (past it, words narrow the list)."""

    LIMIT: ClassVar[int] = 60
    MAX_OFFSET: ClassVar[int] = 3_000
    q: str = Field(default="", max_length=200)
    kind: CommunityKind | None = None
    sort: CommunityEntrySort = "popularity"
    offset: int = Field(default=0, ge=0, le=MAX_OFFSET)


class CommunityLaunchRequest(WireRequest):
    """Start entry `{id}` in conversation `conversation_id` (its coordinator's run, `ProjectConversation.run_id`) for `task`;
    `machine_id`: the PC it runs on (`ProjectConversation.machine_id`), else the one the server recorded for it."""

    conversation_id: UUID
    task: str = Field(min_length=1, max_length=2000)
    machine_id: UUID | None = None


class CommunityLaunchResult(WireModel):
    """The start handed to the conversation's coordinator as its message `message_id`: it starts the entry as its own
    fenced child (`agent_spawn(community=…)`); `run_id` None until then — that run carries `community.entry` = the entry."""

    conversation_id: UUID
    message_id: UUID
    run_id: UUID | None = None
