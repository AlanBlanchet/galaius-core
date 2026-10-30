"""Why something failed, typed: one alias every failure carries (`FailureCause`) and one mixin holding it (`Caused`).
Its reader's words come from its type and figures; the English sentence beside it is the same fact for a log."""

from .paid_use import PaidUseRefusal, PaidUseRefused
from .wire import WireModel

#: Why a run, a step or a chat turn failed, when the server knows it: one member per `code`.
FailureCause = PaidUseRefusal
#: The family bases its members derive from, for an `isinstance` (`cause_of`): a new family joins both lines.
CAUSES: tuple[type[WireModel], ...] = (PaidUseRefused,)


class Caused(WireModel):
    """A record of something that can fail (a run, a step, a chat, an operator's failure row)."""

    #: Why it failed, typed, when the server knows (a metered turn refused): its reader's words come from it; the English
    #: beside it (`error`, `message`) is the same fact, for a log and the technical details.
    cause: FailureCause | None = None
