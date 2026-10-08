"""Error reports: something went wrong in the app or on a linked PC, the person was asked « Envoyer le
rapport ? » and said yes; the platform's admins read what was sent.

Two senders, one report. The BROWSER holds what it caught (an uncaught script error, a request the
server answered 500) until the person answers its dialog, then sends `ErrorReportSubmission`. A PC
never sends its log on its own: it ASKS (`MachineErrorAsk`: what went wrong, one line), its owner
answers on the PC's page in the web, and only an accepted ask is uploaded (`MachineErrorUpload`: the
service log's tail, prepared and scrubbed on the PC). Every captured string is scrubbed again here on
arrival (`SignalRedaction.anonymous`: credentials, URL query values, long opaque runs, email
addresses), whatever the sender did."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from .signals import Captured600, SignalContext, SignalRedaction, SignalReporter, SignalStatus, SignalWorkspace
from .wire import WireModel, WireRequest

ErrorReportOrigin = Literal["web", "pc"]
WebErrorKind = Literal["script", "request"]
"""What a browser caught: a script error / unhandled rejection, or an app request answered 500."""
MachineErrorKind = Literal["exception", "run_failed", "crashed", "service_unavailable", "service_stopped", "channel_unreachable"]
"""What a PC met: an unhandled exception in galaius, a run that failed, its connection crashing, or one
of the reasons it could not connect (`MachineProblemCode`)."""
WebReportText = SignalRedaction.lines(16_000)
"""What a browser adds to the error: its stack (the console and failed requests ride in the context)."""
MachineReportText = SignalRedaction.lines(262_144)
"""What a PC adds: the tail of its galaius service log, at most 256 KiB."""


class ErrorReportSubmission(WireRequest):
    """What the browser sends once the person pressed « Envoyer le rapport »; `id` is its own key (a
    retried send lands on the same report)."""

    id: UUID
    kind: WebErrorKind
    message: Captured600 = Field(min_length=1)
    detail: WebReportText = ""
    context: SignalContext


class ErrorReportReceipt(WireModel):
    id: UUID
    created: bool


class MachineErrorAsk(WireRequest):
    """A PC's question to its owner (`POST /v1/machine/error-report`, its own token): what went wrong, in
    one line; nothing of its log yet. `id` is the PC's key for the report it holds."""

    id: UUID
    kind: MachineErrorKind
    message: Captured600 = Field(min_length=1)


class MachineErrorAskView(WireModel):
    """The PC's question as its owner reads it on the PC's page; `accepted_at` once he said yes (the PC
    uploads at its next look, `GET /v1/machine/error-report`)."""

    id: UUID
    kind: MachineErrorKind
    message: str
    asked_at: datetime
    accepted_at: datetime | None = None


class MachineErrorDecision(WireRequest):
    """The owner's answer to the PC's question: send the report, or not (the question goes away)."""

    decision: Literal["send", "decline"]


class MachineErrorUploadRequest(WireModel):
    """What the server asks a PC to upload: the one report its owner accepted."""

    id: UUID


class MachineErrorUpload(WireRequest):
    """The accepted report's body (`PUT /v1/machine/error-report/{id}`), once."""

    detail: MachineReportText = Field(min_length=1)


class ErrorReportMachine(WireModel):
    id: UUID
    name: str


class ErrorReport(WireModel):
    """One report as the admins' list shows it: everything but its trace (`detail_size` says how long it
    is; `ErrorReportDetail` carries it), so a page of reports stays small whatever each one holds."""

    id: UUID
    created_at: datetime
    origin: ErrorReportOrigin
    kind: WebErrorKind | MachineErrorKind
    message: str
    detail_size: int = Field(ge=0)
    context: SignalContext | None = None
    machine: ErrorReportMachine | None = None
    server_build: str | None = None
    status: SignalStatus
    status_at: datetime | None = None
    reporter: SignalReporter
    workspace: SignalWorkspace | None = None


class ErrorReportDetail(ErrorReport):
    """One report whole: its trace too, read when the admin opens it."""

    detail: str


class ErrorReportList(WireModel):
    """Admin › Erreurs: the newest page, how many there are, how many still new."""

    reports: tuple[ErrorReport, ...]
    total: int = Field(ge=0)
    fresh: int = Field(ge=0)
