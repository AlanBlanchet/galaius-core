"""His PCs as the link itself says them: on or off, what each can run now, and what crosses the link."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from galaius_core.machine.need import MachineName
from galaius_core.wire import Kinded, WireModel
from galaius_core.workflows.contracts import AgentTouchScope, WebLoad

PcLinkState = Literal["online", "offline", "away", "stopped"]
#: A PC's problem as a closed code: the PC's own free text is not read here (`MachineProblemCode`, plus the link's own).
PcProblem = Literal["service_unavailable", "service_stopped", "channel_unreachable", "crashed", "channel_taken"]
TrafficClass = Literal["beat", "agent", "stream", "copy", "peer"]


class LinkTraffic(WireModel):
    """What crossed this PC's link in the last hour (counted at the channel) against the ceiling he can see."""

    per_hour_bytes: int = Field(ge=0)
    ceiling_per_hour_bytes: int = Field(ge=1)
    top_class: TrafficClass | None = None


class PcState(WireModel):
    """One of his PCs. A PC that takes no peer starts is read minimally (`runs_agents` onward absent): a source never maps it."""

    id: UUID
    name: MachineName
    this: bool = False
    state: PcLinkState
    last_seen_at: datetime | None = None
    problem: PcProblem | None = None
    dispatchable: bool = False
    runs_agents: bool | None = None
    peer_runs: bool | None = None
    peer_out: bool | None = None
    can_fence: bool | None = None
    permission: AgentTouchScope | None = None
    places_free: int | None = Field(default=None, ge=0)
    places_total: int | None = Field(default=None, ge=0)
    memory_free_mb: int | None = Field(default=None, ge=0)
    live_runs: int | None = Field(default=None, ge=0)
    web_limit: WebLoad | None = None


class PcFleet(Kinded):
    kind: Literal["peer_fleet"] = "peer_fleet"
    pcs: Annotated[tuple[PcState, ...], Field(max_length=64)] = ()
    traffic: LinkTraffic | None = None
