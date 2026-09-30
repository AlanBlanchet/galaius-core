"""MCP servers an agent can be connected to (`McpBinding` on `AgentRevision`).

Three places a server comes from, one shape (`McpServer`):

- `interact`: the tools shipped with interact itself, listed from the package's own snapshot;
- remote: a streamable-HTTP server a company adds once (https only, its token kept server-side);
- PC: a local (stdio) server a PC declares for itself. The web lists it and may switch its tools
  off in a binding, never add or enable one.

A server's tool list is pinned by its digest when someone approves it: a server whose list changed
since (a tool added, removed, or its description or schema edited) is `changed` and serves no tool
until it is approved again."""

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal, Self
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import Field, SecretStr, field_validator, model_validator

from .resources import Audience, PersonRef, Verb
from .wire import WireModel, WireRequest
from .workflows import MCP_SERVER_ID, McpToolName

McpTransport = Literal["remote", "pc"]
#: ready: its approved tools serve; needs_approval: never approved; changed: its list moved since
#: the approval (it serves nothing until approved again); unreachable: the last read of its list failed.
McpServerStatus = Literal["ready", "needs_approval", "changed", "unreachable"]
#: How a remote server is signed in to: nothing, a bearer token (an API key or an access token the
#: vendor issued) in `Authorization`, or a key in a header the vendor names.
McpAuthKind = Literal["none", "bearer", "header"]
MCP_PC_NAME = r"^[A-Za-z0-9_.-]{1,64}$"
DIGEST = r"^[0-9a-f]{64}$"


class McpTool(WireModel):
    """One tool a server lists: what the model is told about it (untrusted text: the server wrote it)."""

    name: McpToolName
    description: str = Field(default="", max_length=4000)
    #: What the tool lets an agent do (see, hear, act on apps, files, web, …): the server's own
    #: `_meta` category when it declares one, else `other`. Display only: never part of the pin.
    category: str = Field(default="other", pattern=r"^[a-z][a-z0-9_]{0,39}$")
    input_schema: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def digest(tools: Iterable["McpTool"]) -> str:
        """The pin an approval records: every tool's name, description and input schema, in name order."""
        canonical = sorted(({"name": tool.name, "description": tool.description, "input_schema": tool.input_schema} for tool in tools), key=lambda item: item["name"])
        return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


class McpServer(WireModel):
    id: str = Field(pattern=MCP_SERVER_ID)
    name: str = Field(min_length=1, max_length=120)
    #: Brand-mark key (the client's mark registry) or a kind glyph key.
    mark: str = Field(max_length=64)
    transport: McpTransport
    #: The tools shipped with interact: always listed, never removed.
    builtin: bool = False
    url: str | None = Field(default=None, max_length=2048)
    machine_id: UUID | None = None
    tools: tuple[McpTool, ...] = Field(default=(), max_length=512)
    tools_digest: str = Field(pattern=DIGEST)
    approved_digest: str | None = Field(default=None, pattern=DIGEST)
    approved: bool
    status: McpServerStatus
    #: Plain words why it is not ready ("" when it is).
    reason: str = Field(default="", max_length=300)
    owner: PersonRef | None = None
    audience: Audience = "company"
    my_verbs: tuple[Verb, ...] = ()


class McpServerCreate(WireRequest):
    """A remote server a company adds. Its secret never leaves the server once saved."""

    name: str = Field(min_length=1, max_length=120)
    url: str = Field(min_length=9, max_length=2048)
    auth: McpAuthKind = "none"
    secret: SecretStr | None = None
    #: The header carrying the secret for `auth="header"` (e.g. `X-API-Key`).
    header: str | None = Field(default=None, pattern=r"^[A-Za-z0-9-]{1,64}$")
    #: The registry entry it was added from, when it was (`McpCatalogEntry.name`).
    catalog_name: str | None = Field(default=None, max_length=200)

    @field_validator("url")
    @classmethod
    def https_only(cls, value: str) -> str:
        parts = urlsplit(value)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password or parts.fragment:
            raise ValueError("an MCP server address is an https:// URL with a host and no credentials in it")
        return value

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.auth == "none") != (self.secret is None):
            raise ValueError("a secret is given exactly when the server signs in with one")
        if (self.auth == "header") != (self.header is not None):
            raise ValueError("a header name is given exactly for header sign-in")
        return self


class McpServerApproval(WireRequest):
    """Approve the tool list whose digest the approver saw (a list that moved since is refused)."""

    digest: str = Field(pattern=DIGEST)


class McpDeclaredServer(WireModel):
    """A local server a PC runs, as the PC declares it (heartbeat): its name and tools."""

    name: str = Field(pattern=MCP_PC_NAME)
    tools: tuple[McpTool, ...] = Field(default=(), max_length=512)


class McpPopularity(WireModel):
    """How popular one catalog entry is, and why the page puts it where it does.

    The registry keeps no popularity of its own, so popularity is the stars of the server's OWN
    GitHub repository — never those of a repository it shares with others, which says nothing
    about it. `stars` is a number exactly when there is a count to show."""

    #: Its own repository's stars (None: there are none to read, or none read yet).
    stars: int | None = Field(default=None, ge=0)
    #: When they were read (None: never).
    checked_at: datetime | None = None
    #: own: it names a GitHub repository of its own, and that is what `stars` counts; shared: it
    #: names one that is not its alone (the registry's own repository, or one several entries
    #: claim) — nobody is counted from it; none: no GitHub repository at all to count.
    repository: Literal["own", "shared", "none"] = "none"
    #: What puts it where it is: official, one of the well-known servers interact names, published
    #: under a namespace whose owner the registry verified; stars, its own count; none, neither.
    ranked_by: Literal["official", "stars", "none"] = "none"


class McpCatalogEntry(WireModel):
    """One server of the official MCP registry, as the picker shows it."""

    #: The registry's reverse-DNS name (`<namespace>/<name>`), the key to add it by.
    name: str = Field(min_length=3, max_length=200)
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=2000)
    #: The publisher's namespace (`com.notion`, `io.github.<owner>`).
    vendor: str = Field(max_length=200)
    version: str = Field(default="", max_length=64)
    #: remote: a hosted server a company can add; pc: a package a PC installs; both.
    transport: Literal["remote", "pc", "both"]
    #: The streamable-HTTP address of its remote form (None: none, or only a deprecated SSE one).
    url: str | None = Field(default=None, max_length=2048)
    #: api_key: its remote form asks for a secret header; none: it asks for no secret; unknown:
    #: the registry does not say (most vendor servers sign in with OAuth).
    auth: Literal["none", "api_key", "unknown"]
    mark: str | None = Field(default=None, max_length=64)
    repository: str | None = Field(default=None, max_length=2048)
    #: How popular it is and why it ranks where it does.
    popularity: McpPopularity = Field(default_factory=McpPopularity)


class McpCatalogPage(WireModel):
    """One registry page in the order the picker shows it: the well-known servers interact names
    first (`popularity.ranked_by == "official"`), then the entries whose own repository's stars are
    known, most first, then the rest in the registry's own order."""

    entries: tuple[McpCatalogEntry, ...] = Field(max_length=100)
    next: str | None = Field(default=None, max_length=512)
    ranked_by: Literal["official_then_stars"] = "official_then_stars"


class McpToolModel(WireModel):
    """Which model one model-using tool of one agent asks for, and why: the agent's own rule on its
    binding, else the account-wide rule for that tool, else none (the tool uses no model)."""

    server_id: str = Field(pattern=MCP_SERVER_ID)
    tool: McpToolName
    source: Literal["agent", "account", "none"]
    criteria: str | None = None
    weights: str = ""
    #: The model today's ranking picks among what this workspace can reach (None: none reachable).
    model: str | None = None
    #: One plain line for the UI ("Uses: <model> (best on <bench>, under <limit>)").
    line: str = Field(max_length=300)
