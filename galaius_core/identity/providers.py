"""Every provider a person signs in to in Galaius, as data: `providers.json` beside this module holds one row per
vendor, and adding a vendor is adding a row. A row says how he signs in — a coding-agent CLI's login the platform holds
(`program`), an API key (`api`), or a CLI of its own Galaius does not run yet (`cli`) — and what its terms and limits
are. The sign-in KINDS are the classes below (a device code, a pasted code, an API key); the vendors are data."""

import json
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self, get_args
from urllib.parse import parse_qsl, urlsplit

from pydantic import AfterValidator, Field, field_serializer, model_validator

from galaius_core.wire import WireModel

#: A provider's or a program's id, as every page, the server and a PC write it.
ProviderId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")]
#: The model APIs a `provider_api` connection speaks, each one adapter on the server; `openai_compatible` is any
#: server speaking OpenAI's Chat Completions at its own base URL.
ModelApi = Literal["openai", "anthropic", "gemini", "fal", "replicate", "roboflow", "mistral", "elevenlabs", "self_hosted", "openai_compatible"]
#: The callback of a sign-in opened on the PC itself: it returns to the program there.
LOCAL_CALLBACK = re.compile(r"^http://localhost:[0-9]{1,5}/callback$")
#: An https address.
HttpsUrl = Annotated[str, Field(pattern=r"^https://[^\s]+$", max_length=512)]


class ProgramSignIn(WireModel):
    """How one program's login is signed in from the web: the page a code is typed at, or the PC's address shown."""

    def code_page(self) -> str | None:
        """The vendor's page he types this page's code at; None: he signs in at the address the PC shows."""
        return None

    def shows(self, address: str, paste: bool) -> bool:
        """`address`, reported by the PC, is exactly this vendor's sign-in and may be shown to him."""
        return False


class DeviceCodeSignIn(ProgramSignIn):
    """The program's login signed in by the code this page shows, entered on the vendor's `page`."""

    flow: Literal["device_code"] = "device_code"
    page: HttpsUrl

    def code_page(self) -> str:
        return self.page


class PasteCodeSignIn(ProgramSignIn):
    """The program's login signed in at an address the PC shows, the code the vendor's `callback` page then shows pasted
    back. Only an address that is exactly the vendor's sign-in is ever shown (`shows`): its host, path, client and scopes."""

    flow: Literal["paste_code"] = "paste_code"
    hosts: frozenset[str]
    path: str
    #: Parameters every sign-in carries with this value.
    fixed: dict[str, str]
    #: Parameters every sign-in carries with its own value (its PKCE challenge, its state).
    own: frozenset[str]
    scopes: frozenset[str]
    #: The callback of an address any browser signs in at: the vendor's page showing the code.
    callback: HttpsUrl

    @field_serializer("hosts", "own", "scopes")
    def _sorted(self, values: frozenset[str]) -> list[str]:
        """A set written in one order, whatever this process hashes strings with."""
        return sorted(values)

    def shows(self, address: str, paste: bool) -> bool:
        """Https on one of `hosts` at `path`, each query parameter once, `fixed` ones equal, every scope one of `scopes`,
        ending on its code page (`paste`) or on the PC (`LOCAL_CALLBACK`): anything a compromised PC could plant to have
        him consent elsewhere is refused."""
        parts = urlsplit(address)
        if (parts.scheme != "https" or parts.hostname not in self.hosts or parts.netloc.lower() != parts.hostname or parts.path != self.path or parts.fragment
                or any(character == "\\" or character.isspace() for character in address)):
            return False
        query = parse_qsl(parts.query, keep_blank_values=True)
        values = dict(query)
        callback = values.get("redirect_uri", "")
        return (len(values) == len(query) and values.keys() == {*self.fixed, *self.own, "scope", "redirect_uri"}
                and all(values[name] == value for name, value in self.fixed.items()) and all(values[name] for name in self.own)
                and set(values["scope"].split()) <= self.scopes and (callback == self.callback if paste else LOCAL_CALLBACK.fullmatch(callback) is not None))


#: How a CLI's login is signed in from the web.
SignIn = Annotated[DeviceCodeSignIn | PasteCodeSignIn, Field(discriminator="flow")]


class HeldLoginForm(WireModel):
    """How a program's login is held by the platform (signed in ONCE from the web, put on each of the owner's PCs): the
    CLI command that signs in and makes the credential, and where the CLI reads it back on a PC — an environment
    variable (`variable`) or a file in its login folder (`file`), exactly one. A program without one signs in per PC."""

    #: Its CLI's arguments that sign in once and leave the credential to hold (Claude Code `setup-token`: a 1-year token).
    command: tuple[str, ...] = Field(min_length=1)
    #: The command needs a terminal (Claude Code's sign-in screens draw only on one).
    terminal: bool = False
    #: The variable its CLI reads the credential from (Claude Code `CLAUDE_CODE_OAUTH_TOKEN`).
    variable: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,63}$")
    #: The file its CLI keeps the credential in, inside its login folder (Codex `auth.json`).
    file: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{1,64}$")
    #: The credential may be on several PCs at once (a token that never refreshes). False: its CLI refreshes and rotates
    #: it (Codex: « Do not share the same file across concurrent jobs or multiple machines », learn.chatgpt.com
    #: ci-cd-auth, read 2026-11-05): it lives on ONE PC at a time, put on the PC a run starts on only while no other holds it.
    shared: bool = True

    @model_validator(mode="after")
    def _one_place(self) -> Self:
        if (self.variable is None) == (self.file is None):
            raise ValueError("a held login lives in exactly one of a variable or a file")
        return self


class AgentProgram(WireModel):
    """One coding-agent CLI a PC runs agents through, as the PC, the server and the web all name it: a `Provider`
    row's `program`. Running it on a PC is that PC's adapter for the CLI (`galaius.agents.providers`)."""

    provider: ProviderId
    #: Its product name, on every page listing it (« Claude Code »).
    name: str
    #: Its name inside a sentence (« Claude indisponible jusqu’à 22:01 »).
    short: str
    sign_in: SignIn
    #: What its own terminal session switches login with, None when it has none.
    login_command: str | None = None
    #: How its login is held by the platform (`HeldLoginForm`); None: it signs in on each PC.
    held: HeldLoginForm | None = None


class ApiKeyLogin(WireModel):
    """A provider signed in by an API key he pastes once in Galaius: it becomes a `provider_api` connection of his
    workspace (`connection` at `endpoint`), checked with the vendor before it is kept, then usable from any project."""

    flow: Literal["api_key"] = "api_key"
    #: The model API the key's calls speak (`openai_compatible`: Chat Completions at `endpoint`, its version path included).
    connection: ModelApi
    endpoint: HttpsUrl
    #: The vendor's page where he makes a key.
    console: HttpsUrl
    #: A chat model the connection enables at once, the one its first call answers with; None: the key is checked by
    #: listing the vendor's models, which a server of OpenAI's API may list without one (`openai_compatible` names one).
    model: str | None = Field(default=None, min_length=1, max_length=160)

    @model_validator(mode="after")
    def _checked_by_a_call(self) -> Self:
        if self.connection == "openai_compatible" and self.model is None:
            raise ValueError("an OpenAI-compatible key is checked by one call: name its model")
        return self


#: How much of a provider's allowance a program can read: `readable` (used share and reset, read while it runs: the
#: automatic switch works on it), `partial` (only once reached, in an interactive command, or on the vendor's page),
#: `none` (nothing found).
LimitSignal = Literal["readable", "partial", "none"]
#: What a vendor's terms say of a program driving its CLI with the subscriber's own login.
TermsVerdict = Literal["allowed", "unclear", "forbidden"]


class ProviderCli(WireModel):
    """A provider's own coding-agent CLI on a PC."""

    command: str
    #: What its owner types on the PC to sign it in: a shell command, or the CLI then its own command inside.
    sign_in: str


class TermsRead(WireModel):
    """A vendor's terms as read: the verdict and where."""

    verdict: TermsVerdict
    url: HttpsUrl


class Provider(WireModel):
    """One vendor a person signs in to: its program's held login, its API key, its own CLI, or several of them.
    Rows live in `providers.json` (`registry`), in the order every page lists them. Their API endpoints, key pages and
    check models were read at each vendor's own docs on 2026-11-05 (OpenRouter's model at its public model list), and
    every endpoint answered a wrong key with its refusal that day; a model a vendor drops reads « provider_refused » on
    the next key until its row names another."""

    id: ProviderId
    #: The vendor, as it names itself.
    label: str
    #: Its plans, as the vendor sells them; None: pay-as-you-go only.
    plans: str | None = None
    program: AgentProgram | None = None
    api: ApiKeyLogin | None = None
    #: A CLI of its own a PC could sign in, Galaius not running it yet.
    cli: ProviderCli | None = None
    limit: LimitSignal = "none"
    #: None: its terms not read.
    terms: TermsRead | None = None

    @model_validator(mode="after")
    def _signs_in_somehow(self) -> Self:
        if self.program is None and self.api is None and self.cli is None:
            raise ValueError(f"{self.id}: a provider signs in through a program, an API key or a CLI")
        return self

    @classmethod
    @cache
    def registry(cls) -> tuple["Provider", ...]:
        """Every row of `providers.json`, ids and program ids each once."""
        rows = tuple(cls.model_validate(row) for row in json.loads(Path(__file__).with_name("providers.json").read_text(encoding="utf-8")))
        for name, ids in (("provider", [row.id for row in rows]), ("program", [row.program.provider for row in rows if row.program])):
            if len(ids) != len(set(ids)):
                raise ValueError(f"providers.json lists a {name} id twice")
        return rows

    @classmethod
    def by_id(cls, provider: str) -> "Provider | None":
        return next((row for row in cls.registry() if row.id == provider), None)


#: Every provider, in the order every page lists them.
PROVIDERS: tuple[Provider, ...] = Provider.registry()
#: Each program a PC runs agents through, by its id, in `PROVIDERS` order.
AGENT_PROGRAMS: Mapping[str, AgentProgram] = MappingProxyType({row.program.provider: row.program for row in PROVIDERS if row.program})


def _a_program(value: str) -> str:
    if value not in AGENT_PROGRAMS:
        raise ValueError(f"{value!r} is no agent program (one of {', '.join(AGENT_PROGRAMS)})")
    return value


#: The id of a coding-agent CLI a PC runs agents through: one of `AGENT_PROGRAMS`, whatever `providers.json` lists.
AgentProvider = Annotated[ProviderId, AfterValidator(_a_program)]
#: The program a choice nobody made falls to: the first one.
DEFAULT_PROGRAM: str = next(iter(AGENT_PROGRAMS))
#: The model APIs, as a set to test a connection's provider against.
MODEL_APIS: frozenset[str] = frozenset(get_args(ModelApi))


def program_name(provider: str) -> str:
    """`provider`'s product name (« Claude Code »); a provider no program is (an API vendor), as written."""
    return AGENT_PROGRAMS[provider].name if provider in AGENT_PROGRAMS else provider
