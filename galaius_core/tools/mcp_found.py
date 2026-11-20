"""What a search for an MCP server shows of one: its package, the hosts it would reach, what it needs, whether its registry
vouches for its publisher, and whether Galaius may start it — read from the server's record, never from a stranger's free text.

Every field is a closed value or a string held to a grammar (`npm_name`, `exact_host` …): a record's words (description,
title) never reach these fields, so a listing built from them (`McpServerFacts.said`) carries nothing a stranger wrote
freely. A server is `blocked` for a reason of THIS table (`BLOCK_TEXT`), never one a record names.

The grammars here are the ones a later step holds the same strings to (fetching a package, approving a host): one place."""

import ipaddress
import re
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from galaius_core.wire import WireModel

McpPackageKind = Literal["npm", "pypi", "nuget", "cargo", "oci", "mcpb", "other"]
FETCHED_KINDS: frozenset[str] = frozenset({"npm", "pypi"})
"""Package kinds fetched without building anything on the PC (an npm tarball, a pypi wheel). A cargo crate builds from source
and an oci image needs a container socket: neither is started."""
McpNeedVia = Literal["env", "header", "argument"]
McpRoute = Literal["package", "remote"]
McpBlock = Literal["needs_input", "no_route", "kind_not_fetched", "not_stdio", "host_refused", "record_unreadable", "program_shipped", "start_not_package"]
BLOCK_TEXT: dict[McpBlock, str] = {
    "needs_input": "needs a key or a setting only its user can give",
    "no_route": "its record names neither a package nor a hosted address",
    "kind_not_fetched": "its package kind is not fetched here (only npm and pypi are, built from nothing)",
    "not_stdio": "its package is not a local program speaking over standard input",
    "host_refused": "its address is not an exact public host name Galaius may open",
    "record_unreadable": "part of its record could not be read",
    "start_not_package": "its start is a program or a script, not a package: Galaius does not run one",
    "program_shipped": "it ships with an installed program, which it would control: found and shown, never started",
}
FOUND_BY = Literal["registry", "github-registry", "link", "program"]
NEEDS_MAX, HOST_MAX = 16, 253

_NPM = re.compile(r"^(?:@[a-z0-9][a-z0-9._-]{0,213}/)?[a-z0-9][a-z0-9._-]{0,213}$")
_PYPI = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,212}[A-Za-z0-9])?$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+){0,3}(?:[-+][0-9A-Za-z.+-]{1,40})?$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9@][A-Za-z0-9@._/:+-]{0,213}$")
_ENV = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
_ARGUMENT = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}$")
_HEADER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,63}$")
_LABEL = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_HOST = re.compile(rf"^(?:{_LABEL}\.)+[a-z][a-z0-9-]{{0,61}}[a-z0-9]$")
_SECRET = re.compile(r"(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|BEARER)", re.IGNORECASE)
LOCAL_SUFFIXES = ("local", "localhost", "internal", "lan", "home", "corp", "intranet", "test", "invalid", "example", "arpa")
"""Names that only mean something inside one network: never an exact public host."""
RESERVED_HOSTS = ("anthropic.com", "claude.ai", "claude.com", "openai.com", "chatgpt.com", "googleapis.com", "mistral.ai", "galaius.ai")
"""Registrable names of the model providers a run's CLI reaches and of Galaius itself: a stranger's server never holds them
(the way out for a key to a model under the server's own account)."""


def npm_name(value: object) -> str | None:
    """`value` as an npm package name (scoped or not, lower case, at most 214 characters); None otherwise: never a URL, a path,
    an option (`--registry=…`) or a spec (`git+ssh://…`, `name@^1`)."""
    return value if isinstance(value, str) and len(value) <= 214 and _NPM.fullmatch(value) else None


def pypi_name(value: object) -> str | None:
    """`value` as a PyPI project name; None otherwise (no extras, no specifier, no URL, no leading dash)."""
    return value if isinstance(value, str) and len(value) <= 214 and _PYPI.fullmatch(value) else None


def exact_version(value: object) -> str:
    """`value` as one exact release number (`1.2.3`, `0.4.0-beta.1`); "" for anything else — a range, a tag (`latest`), a URL."""
    return value if isinstance(value, str) and _VERSION.fullmatch(value) else ""


def exact_host(value: object) -> str | None:
    """`value` as one exact public DNS name, lower case: no wildcard, port, path, address, user, trailing dot, no single label, no
    local-only suffix; None otherwise. A name that resolves to a private address is held at connect time, not here."""
    if not isinstance(value, str) or len(value) > HOST_MAX:
        return None
    host = value.lower()
    if not _HOST.fullmatch(host) or host.rsplit(".", 1)[1] in LOCAL_SUFFIXES:
        return None
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host
    return None


def reserved_host(host: str) -> bool:
    """Whether `host` is, or is under, a model provider's name or Galaius's own."""
    return any(host == name or host.endswith("." + name) for name in RESERVED_HOSTS)


def https_host(url: object) -> str | None:
    """The exact host of an https address with no user, port or space (`exact_host`); None otherwise."""
    match = re.fullmatch(r"https://([^/:@?#\s]+)(?:/[^\s]*)?", url) if isinstance(url, str) and len(url) <= 2048 else None
    return exact_host(match[1]) if match else None


def secret_name(name: str) -> bool:
    """Whether a variable or header name says it carries a secret."""
    return _SECRET.search(name) is not None


class McpNeed(WireModel):
    """One thing a server asks its user for: an environment variable, a header, a required argument — its NAME only."""

    name: str = Field(min_length=1, max_length=64)
    via: McpNeedVia
    secret: bool
    required: bool

    @model_validator(mode="after")
    def named_by_grammar(self) -> "McpNeed":
        if not {"header": _HEADER, "argument": _ARGUMENT}.get(self.via, _ENV).fullmatch(self.name):
            raise ValueError(f"{self.name!r} is not a {self.via} name")
        return self


class McpPackage(WireModel):
    """The package a server is started from. `version`: one exact release, "" when its record states none."""

    kind: McpPackageKind
    identifier: str = Field(min_length=1, max_length=214)
    version: str = Field(default="", max_length=64)
    stdio: bool = True

    @model_validator(mode="after")
    def named_by_grammar(self) -> "McpPackage":
        if not _IDENTIFIER.fullmatch(self.identifier):
            raise ValueError(f"{self.identifier!r} is not a package identifier")
        if self.kind in FETCHED_KINDS and (npm_name if self.kind == "npm" else pypi_name)(self.identifier) is None:
            raise ValueError(f"{self.identifier!r} is not a {self.kind} package name")
        if self.version and not exact_version(self.version):
            raise ValueError("a package version is one exact release or empty")
        return self


class McpServerFacts(WireModel):
    """One MCP server as a search shows it. `name`: the record's name (a registry's reverse-DNS name, a repository's link).
    `route`: how Galaius would start it (the package when it is fetched here, else the hosted address); None when it cannot.
    `hosts`: the exact names it would reach — its hosted address's; none for a package (it gets no network unless approved).
    `vouched`: its registry checked that its publisher owns the namespace it is named under. `blocked`: why it may not be
    started now, each of `BLOCK_TEXT`; empty: it may, once its user approves it."""

    name: str = Field(min_length=1, max_length=300)
    found_by: FOUND_BY
    version: str = Field(default="", max_length=64)
    package: McpPackage | None = None
    remote: str | None = Field(default=None, max_length=2048)
    route: McpRoute | None = None
    hosts: tuple[str, ...] = Field(default=(), max_length=8)
    needs: tuple[McpNeed, ...] = Field(default=(), max_length=NEEDS_MAX)
    vouched: bool = False
    first_published: datetime | None = None
    blocked: tuple[McpBlock, ...] = ()

    @property
    def attachable(self) -> bool:
        return self.route is not None and not self.blocked

    @classmethod
    def made(cls, name: str, found_by: FOUND_BY, *, version: str = "", package: McpPackage | None = None, remote: str | None = None,
             needs: tuple[McpNeed, ...] = (), unread: bool = False, vouched: bool = False, first_published: datetime | None = None,
             shipped: bool = False, unpackaged: bool = False) -> "McpServerFacts":
        """The facts of a server from its parts, its route and what blocks it decided here (one place, whichever registry, link
        or program the parts came from). `unread`: part of its record could not be read; `unpackaged`: its start is a command that
        names no package (a script, a binary), which is never run."""
        host = https_host(remote)
        packaged = package is not None and package.kind in FETCHED_KINDS and package.stdio
        hosted = host is not None and not reserved_host(host)
        route: McpRoute | None = "package" if packaged else "remote" if hosted else None
        blocked: list[McpBlock] = []
        if route is None:
            if package is None and remote is None:
                blocked.append("start_not_package" if unpackaged else "no_route")
            if package is not None and package.kind not in FETCHED_KINDS:
                blocked.append("kind_not_fetched")
            if package is not None and package.kind in FETCHED_KINDS and not package.stdio:
                blocked.append("not_stdio")
            if remote is not None:
                blocked.append("host_refused")
        for_route = {"package": ("env", "argument"), "remote": ("header",)}.get(route or "", ())
        if any(need.required for need in needs if need.via in for_route):
            blocked.append("needs_input")
        if unread:
            blocked.append("record_unreadable")
        if shipped:
            blocked.append("program_shipped")
        return cls(name=" ".join(name.split()) or "?", found_by=found_by, version=exact_version(version), package=package, remote=remote if remote and https_host(remote) else None,
                   route=route, hosts=(host,) if route == "remote" and host else (), needs=needs[:NEEDS_MAX], vouched=vouched,
                   first_published=first_published, blocked=tuple(dict.fromkeys(blocked)))

    def said(self) -> str:
        """It as one plain line from its closed fields: what it is started from, what it would reach, what it needs, whether its
        registry vouches for its publisher, and — when it may not be started — why."""
        if self.route == "package" and self.package is not None:
            start = f"package {self.package.kind} {self.package.identifier}" + (f" {self.package.version}" if self.package.version else " (version not stated)")
        elif self.package is not None:
            start = f"package {self.package.kind} {self.package.identifier}" + (f" {self.package.version}" if self.package.version else "")
        else:
            start = ""
        if self.remote and self.route != "package":
            start = (start + "; " if start else "") + f"hosted at {https_host(self.remote) or 'an address Galaius may not open'}"
        reach = f"can reach {', '.join(self.hosts)}" if self.hosts else "reaches no network host"
        needed = ", ".join(f"{need.name} ({'secret ' if need.secret else ''}{need.via}{'' if need.required else ', optional'})" for need in self.needs)
        parts = [start or "no start named", reach, f"needs {needed}" if needed else "needs nothing declared",
                 "publisher vouched by its registry" if self.vouched else "publisher not vouched",
                 f"first published {self.first_published:%Y-%m-%d}" if self.first_published else ""]
        line = "; ".join(part for part in parts if part)
        if self.blocked:
            line += " — not attachable: " + "; ".join(BLOCK_TEXT[reason] for reason in self.blocked)
        return line


class McpMatched(WireModel):
    """How many servers a registry holds for the same words, and from how many publishers: a hundred near-identical servers
    from forty makers is a different choice from one."""

    servers: int = Field(ge=0)
    publishers: int = Field(ge=0)
    source: Literal["registry"] = "registry"


class McpLinkServer(WireModel):
    """One server a repository's own files describe: `file` the one that did (`.mcp.json`, its README's configuration block,
    `package.json`), `key` the name that file gives it (a plain line), its facts. A start command that is not a package
    (a script, a binary path) is never read as one: such a server has no package and says so (`no_route`)."""

    key: str = Field(min_length=1, max_length=80)
    file: Literal[".mcp.json", "README", "package.json"]
    facts: McpServerFacts


class McpLinkRead(WireModel):
    """What reading a repository's files named: `repository` as `owner/name`; `found` the servers they describe (none: none
    of the files was there, or none held one); `read` the files that were; `note` one fixed sentence when part was not read."""

    repository: str = Field(max_length=200)
    found: tuple[McpLinkServer, ...] = Field(default=(), max_length=16)
    read: tuple[str, ...] = Field(default=(), max_length=8)
    note: str | None = Field(default=None, max_length=300)
