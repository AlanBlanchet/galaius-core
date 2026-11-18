"""The calls a runner sends its server on its live channel, and the server's replies: one envelope each way.

A call is authored by a PC (an unknown field is refused). No call names a folder, a path outside a project, a model, an
image or a secret: the server resolves the project to the target's folder and the target's own ranking picks the model."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, model_validator

from galaius_core.machine.peer.fleet import PcFleet
from galaius_core.machine.peer.refusal import PeerRefusal
from galaius_core.machine.peer.rules import PEER_KEY_MAX, PEER_KEY_PATTERN, PEER_MESSAGE_MAX, PEER_TEXT_MAX
from galaius_core.wire import Kinded, WireModel, WireRequest
from galaius_core.workflows.contracts import AgentRole, AgentTouchScope, MachineRunEvents, ReasoningLevel

PeerKey = Annotated[str, Field(min_length=1, max_length=PEER_KEY_MAX, pattern=PEER_KEY_PATTERN)]
ProjectPathBeneath = Annotated[str, Field(max_length=1024)]


class RunRef(WireModel):
    """The run asking, as its PC claims it. The server records it and shows it, `by_verified` or not; it never grants on it."""

    run_id: UUID
    conversation_id: UUID


class PeerHello(WireRequest):
    """Added to a runner's hello: a random id of this process (memory only) and the seal the server gave its last accepted
    hello. A second process holding a copy of the link token presents another instance and an older seal."""

    instance: UUID
    seal: str | None = Field(default=None, min_length=16, max_length=96)
    #: This PC's owner switched peer runs off HERE (`galaius machine peer off`): the server latches its own switch off; only his web session turns it on.
    off: bool = False


class PeerSealed(WireModel):
    """The server's answer to an accepted hello: the seal this PC presents at its next one."""

    seal: Annotated[str, Field(min_length=16, max_length=96)]


class PeerCall(Kinded, WireRequest):
    """What every call carries. `by` None: the person's own terminal (no run); the server then opens a conversation of its own."""

    by: RunRef | None = None
    by_verified: bool = False


class PeerFleetCall(PeerCall):
    kind: Literal["peer_fleet"] = "peer_fleet"


class PeerProjectsCall(PeerCall):
    kind: Literal["peer_projects"] = "peer_projects"


class PeerStart(PeerCall):
    kind: Literal["peer_start"] = "peer_start"
    key: PeerKey
    target: UUID
    project: UUID
    path: ProjectPathBeneath = ""
    role: AgentRole
    text: Annotated[str, Field(min_length=1, max_length=PEER_TEXT_MAX)]
    reasoning: ReasoningLevel | None = None
    #: Asks for less than the PCs allow, never more (`PEER_CEILING`).
    permission: AgentTouchScope | None = None


class PeerMessage(PeerCall):
    kind: Literal["peer_message"] = "peer_message"
    key: PeerKey
    machine: UUID
    run_id: UUID
    text: Annotated[str, Field(min_length=1, max_length=PEER_MESSAGE_MAX)]


class PeerEvents(PeerCall):
    kind: Literal["peer_events"] = "peer_events"
    machine: UUID
    run_id: UUID
    cursor: int | None = Field(default=None, ge=0)
    wait_s: float = Field(default=0.0, ge=0, le=25)


class PeerStop(PeerCall):
    kind: Literal["peer_stop"] = "peer_stop"
    key: PeerKey
    machine: UUID
    run_id: UUID


PeerCallAny = Annotated[PeerFleetCall | PeerProjectsCall | PeerStart | PeerMessage | PeerEvents | PeerStop, Field(discriminator="kind")]


class PeerEnvelope(WireRequest):
    """One call on the channel, answered by the `PeerReply` of the same `request_id`."""

    request_id: UUID
    call: PeerCallAny


class PeerStarted(Kinded):
    kind: Literal["peer_started"] = "peer_started"
    dispatch_id: UUID
    run_id: UUID
    machine_id: UUID
    machine_name: str = Field(max_length=120)
    conversation_id: UUID
    #: The permission the run really got: the least of every limit (`PEER_CEILING`, both PCs, the request).
    permission: AgentTouchScope
    replayed: bool = False


class PeerSent(Kinded):
    kind: Literal["peer_sent"] = "peer_sent"
    message_id: UUID
    replayed: bool = False


class PeerStopped(Kinded):
    kind: Literal["peer_stopped"] = "peer_stopped"
    run_id: UUID
    replayed: bool = False


class PeerProject(WireModel):
    id: UUID
    name: str = Field(max_length=200)


class PeerProjects(Kinded):
    """The projects its owner may use, by name: what `PeerStart.project` is chosen from."""

    kind: Literal["peer_projects"] = "peer_projects"
    projects: Annotated[tuple[PeerProject, ...], Field(max_length=200)] = ()


class PeerWindow(Kinded):
    kind: Literal["peer_window"] = "peer_window"
    events: MachineRunEvents


PeerAnswer = Annotated[PcFleet | PeerProjects | PeerStarted | PeerSent | PeerStopped | PeerWindow, Field(discriminator="kind")]


class PeerReply(WireModel):
    """The server's reply: exactly one of an answer or a refusal, never an answer carrying an error."""

    request_id: UUID
    answer: PeerAnswer | None = None
    refusal: PeerRefusal | None = None

    @model_validator(mode="after")
    def one_of_them(self) -> "PeerReply":
        if (self.answer is None) == (self.refusal is None):
            raise ValueError("a peer reply is an answer or a refusal, never both and never neither")
        return self
