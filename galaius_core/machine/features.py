"""What a PC's runner answers beyond the base protocol: one registry, named in its hello's `features`."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Self, get_args

from galaius_core.machine.need import ScriptLanguage


class PcFeature(StrEnum):
    """A request kind (or part of one) a PC's runner answers, as its hello names it. A runner names every member of the
    galaius-core it ships with, and never one it was released before: the server asks a PC only what it names, adapts
    where an older PC can be served another way, and else says that PC is outdated. A hello is read as `PcFeatures`.
    Members are only ever added: retiring one is a migration of every run cause naming it."""

    #: File queries browse the owner's script roots.
    FILE_QUERY = "file_query"
    #: A Script step runs a file from those roots.
    SCRIPT_FILE = "script_file"
    #: A file is read from those roots (the Data page).
    FILE_READ = "file_read"
    #: Its agents are listed, started and asked from the web (`MachineAgentRequestBase`).
    AGENT_CONTROL = "agent_control"
    #: Its agent settings are read and changed from the web.
    AGENT_SETTINGS = "agent_settings"
    #: The web's own switches (agents on, folders open to them) hold on it.
    WEB_SETTINGS = "web_settings"
    #: Its project folders are listed and created from the web.
    WORKSPACES = "workspaces"
    #: A workflow step on it starts read-only, its owner's permission asked first.
    START_PERMISSION = "start_permission"
    #: A project's secrets reach the agents working in its folder.
    PROJECT_SECRETS = "project_secrets"
    #: Its places (folders opened to agents, at a level) are read and changed from the web.
    PLACES = "places"
    #: A transcript's media (images, files) are read from the web.
    AGENT_MEDIA = "agent_media"
    #: Its `galaius mcp` serves the external tools the server connects its agents to.
    TOOL_GATEWAY = "tool_gateway"
    #: It takes the server's swap of a one-time bootstrap token.
    TOKEN_SWAP = "token_swap"
    #: It sends a folder to another PC and receives one, creates an empty project folder, names a folder's repository,
    #: and says why a project folder failed as a code.
    WORKSPACE_COPY = "workspace_copy"
    #: A level from the web applies at once; folder names in the home folder are listed without its own browse switch.
    PLACES_DIRECT = "places_direct"
    #: Its agent programs (Claude Code, Codex) are listed with their sign-in state.
    AGENT_PROGRAMS = "agent_programs"
    #: A start carries one subscription per provider; a refusal says why, typed (`SubscriptionChoices`).
    SUBSCRIPTIONS = "agent_subscription_choices"
    #: A running conversation moves to another subscription (`RunMove`).
    RUN_MOVES = "agent_subscription_moves"
    #: A `RunMove` that does not resume (`resume` False).
    RUN_MOVE_WAITS = "agent_subscription_moves_wait"
    #: A `RunMove` from or to its own login of the run's CLI (`subscription_id` None).
    RUN_MOVE_LOGIN = "agent_login_moves"
    #: A message asked again under the same id is never queued twice.
    SEND_ONCE = "agent_send_once"
    #: A start asked again under its id answers its one run.
    START_ONCE = "agent_start_once"
    #: A start, a continued conversation and the runs list honour the web's `WebLimited.web_limit`.
    WEB_LIMIT = "web_limit"
    #: A sign-in code pasted on the web reaches its agent program.
    SIGN_IN_PASTE = "sign_in_paste"
    #: A conversation is compacted on request.
    AGENT_COMPACT = "agent_compact"
    #: It holds `PROJECTS_ROOT`, the projects folder (else projects land under an agent root the web adds).
    PROJECTS_ROOT = "projects_root"
    #: A message shows images (`AgentMessageSpec.images`): kept in the run's own folder, handed to the agent.
    AGENT_IMAGES = "agent_images"
    #: A message sets its conversation's effort (`AgentMessageSpec.reasoning`).
    AGENT_REASONING = "agent_reasoning"
    #: The files its file tools kept for its runs are served in parts from any offset (`AgentFile`).
    AGENT_FILES = "agent_file_parts"
    #: It reports and installs `MACHINE_TOOLS`.
    MACHINE_TOOLS = "machine_tools"
    #: Its screen is watched, and driven, from the web (`ScreenDesk`).
    SCREEN = "screen_view"
    #: A project folder's files are shown to the web, read-only (`ProjectFilesBase`).
    PROJECT_FILES = "project_files"
    #: A file not shown as text is read as raw bytes, part by part (`ProjectFileBytesRequest`).
    PROJECT_FILE_BYTES = "project_file_bytes"
    #: A file's text is read from a byte offset, page by page (`ProjectFileRequest.offset`).
    PROJECT_FILE_OFFSET = "project_file_offset"
    #: A subscription's login held by the platform is handed back sealed (`AgentHeldTakeRequest`) and put on this PC
    #: (`AgentHeldPutRequest`), each launch of his here then signed in with it.
    HELD_LOGINS = "held_logins"
    #: A held login is signed in here: its CLI's long-lived token command run under a terminal
    #: (`AgentProgramInstallRequest.held`); a PC that cannot run one still takes held logins (`HELD_LOGINS`).
    HELD_SIGN_IN = "held_sign_in"
    #: A message its person sent is taken back (`AgentRetractRequest`): never read when still waiting, its turn stopped
    #: when read, the run's next turn told.
    MESSAGE_RETRACT = "agent_message_retract"
    #: A conversation started here moves to another of its owner's PCs and stays the same one (`AgentExportRequest`,
    #: `AgentImportRequest`, `AgentReleaseRequest`).
    RUN_CARRY = "agent_run_carry"
    #: Its conversations are held by the server, not by this PC: a copy taken between turns never stops its turns here
    #: (`AgentExportRequest`), one whose CLI session is gone here is packed from its stream, and one arriving without its
    #: session (`AgentImportRequest`) has it written from that stream at its next turn.
    RUN_HOLD = "agent_run_hold"


@dataclass(frozen=True, slots=True)
class PcFeatures:
    """What one runner's hello names, read once there: the `PcFeature` members it answers and the Script languages it
    runs. An entry no member or language holds (a newer runner's, or not a string) is dropped: never asked about."""

    named: frozenset[PcFeature] = frozenset()
    scripts: frozenset[ScriptLanguage] = frozenset()

    #: How a hello names a Script language, past every member (`script:python`).
    _SCRIPT = "script:"

    @classmethod
    def of(cls, said: Iterable[object]) -> Self:
        """A hello's `features` (or a set of members, or another `PcFeatures`: it iterates as its hello), read."""
        entries = [entry for entry in said if isinstance(entry, str)]
        languages = {entry.removeprefix(cls._SCRIPT) for entry in entries if entry.startswith(cls._SCRIPT)}
        return cls(frozenset(PcFeature(entry) for entry in entries if entry in _MEMBERS),
                   frozenset(language for language in get_args(ScriptLanguage) if language in languages))

    def __iter__(self) -> Iterator[str]:
        """What a hello's `features` names for this: every member, then each Script language."""
        yield from sorted(self.named)
        yield from (f"{self._SCRIPT}{language}" for language in sorted(self.scripts))

    def said(self) -> list[str]:
        return list(self)

    def missing(self, *features: PcFeature) -> PcFeature | None:
        """The first of `features` this runner does not name (its update brings all of them), None when it names all."""
        return next((feature for feature in features if feature not in self.named), None)

    def __contains__(self, feature: object) -> bool:
        return feature in self.named

    def __or__(self, more: Iterable[object]) -> Self:
        return self.of([*self, *more])

    def __sub__(self, fewer: Iterable[PcFeature]) -> Self:
        return type(self)(self.named - frozenset(fewer), self.scripts)


#: Every member's value: a hello's entry is one exactly when it is here (a member compares equal to its value).
_MEMBERS: frozenset[str] = frozenset(PcFeature)
#: What a PC's owner runs there to bring its runner up to every `PcFeature` (it also does so by itself), and to start or
#: restart its background connection (`galaius login` set it up). Every page naming them says these.
UPDATE_COMMAND, CONNECT_COMMAND = "galaius upgrade check", "galaius machine service restart"
