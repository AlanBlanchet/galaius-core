"""Why a cross-PC request made nothing: a typed code and one sentence, never a success carrying an error."""

from typing import Literal
from uuid import UUID

from pydantic import Field

from galaius_core.wire import WireModel

PeerRefusalCode = Literal[
    "invalid_request",
    "not_found",            # a PC, run or project that is not his: never confirms it exists
    "permission_denied",    # his right to run agents is gone, or the verb is not this caller's (events / stop only downward)
    "peer_off",             # the target takes no peer starts: its switch, its runner too old, its agents off
    "peer_out_off",         # the source may not place runs elsewhere (lent PC, cloud PC, not trusted)
    "channel_taken",        # a second process took the PC's channel: cleared only on the web
    "cannot_fence",         # the target cannot isolate the run and its owner did not allow that
    "needs_owner",          # the project's folder there needs an answer only its owner gives
    "no_remote",            # the project has no repository its files can travel by
    "peer_full",            # the target has no free place for a peer run
    "in_progress",          # the same key is still being answered
    "key_reused",           # the same key with another request
    "dispatch_rate",        # too many starts / requests in flight: Retry-After
    "link_ceiling",         # the link's hourly byte ceiling for this class of traffic
    "pc_unreachable",       # the target is not connected: nothing started
    "pc_slow",              # the target did not answer in time: a run may exist, ask again with the same key
]


class PeerRefusal(WireModel):
    code: PeerRefusalCode
    message: str = Field(max_length=500)
    retry_after_s: int | None = Field(default=None, ge=1, le=86400)
    #: The server's record of a start (`pc_slow`, `in_progress`): the one a retry with the same key finds again.
    dispatch_id: UUID | None = None
