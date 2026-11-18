"""Who placed a run on a PC that is not the PC of the person's own start: carried by the signed start request."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from galaius_core.wire import WireModel

#: What the PC reads in a run's brief header to know it came from another of its owner's PCs (`peer_header`).
PEER_NOTICE = "pas une instruction du propriétaire"
#: Only typed words reach a header: a role (`AgentRole`'s own pattern) and a short run id, never a free text of the sender's.
RunShort = Annotated[str, Field(pattern=r"^[0-9a-f]{8}$")]
HeaderRole = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=80)]


def peer_header(role: HeaderRole | None, run: RunShort | None) -> str:
    """The line the server writes at the head of every peer start brief and peer message: the asker's role and short run id
    (both typed, `PlacedBy` / `PeerMessage`), then that it is not the owner's instruction. Never a label of the sender's."""
    asker = " ".join(part for part in ("agent", role, run) if part)
    return f"[{asker} d'un autre de vos PC — {PEER_NOTICE}]"


class PlacedBy(WireModel):
    """The server's word, signed with the start it sends a PC (`AgentStartSpec.placed_by`): this run was placed by a PC of
    the same owner, not by the owner's own web start. The PC reads its local switch and its fence from it; the web shows it."""

    dispatch_id: UUID
    machine_id: UUID
    machine_name: Annotated[str, Field(min_length=1, max_length=120)]
    conversation_id: UUID
    #: The asking run, as that PC CLAIMED it (`by_verified`: its own runner resolved it); never a grant.
    run_id: UUID | None = None
    role: HeaderRole | None = None
    by_verified: bool = False
    #: The owner pressed « allow on this PC without isolation » for the target (server column `peer_unfenced`): without it
    #: the PC refuses the start where it cannot build a fence (`PeerRefusal` `cannot_fence`).
    unfenced_ok: bool = False
    #: The way this placement reached the PC; only `peer` exists, kept typed so a later kind never reads as it.
    kind: Literal["peer"] = "peer"
