"""`Refused[R]`: the one coded-refusal base a PC's runtime and the server share — a kind declares its reason's type alone,
a bare reason is validated into it at the raise, the refusal a machine answer carries is read off any error in one call,
and a refusal survives a copy or a pickle (a worker thread's error crossing back, a logged copy)."""
import copy
import pickle
from typing import Literal

import pytest
from pydantic import ValidationError

from galaius_core import refusal_of, AgentRefusal, Coded, MachineRefusal, Refused, ScreenRefusal, SubscriptionCap, SubscriptionRefusal


class Agent(Refused[AgentRefusal]):
    pass


class Answered(Refused[MachineRefusal | None]):
    pass


class Job(Refused[Literal["install_failed", "download_failed"]]):
    pass


CAP = SubscriptionCap(window="weekly", used_percent=81, cap_percent=80, at_cap="cut")


@pytest.mark.parametrize(("raised", "reason", "answered"), [
    (Agent("web_full", "4 conversations are already working here"), AgentRefusal(reason="web_full"), AgentRefusal(reason="web_full")),
    (Answered(SubscriptionRefusal(reason="capped", cap=CAP), "at 81% of its weekly window"), SubscriptionRefusal(reason="capped", cap=CAP), SubscriptionRefusal(reason="capped", cap=CAP)),
    (Answered(ScreenRefusal(reason="passkey_new"), "a passkey it has not pinned"), ScreenRefusal(reason="passkey_new"), ScreenRefusal(reason="passkey_new")),
    (Answered(None, "refused, no reason given"), None, None),
    (Job("download_failed", "too many redirects"), "download_failed", None),          # never on a machine answer
])
def test_a_refusal_is_typed_at_the_raise_read_in_one_call_and_survives_a_copy(raised: Refused, reason, answered) -> None:
    assert isinstance(raised, PermissionError) and raised.refusal == reason and refusal_of(raised) == answered
    for again in (copy.copy(raised), pickle.loads(pickle.dumps(raised))):
        assert (type(again), again.refusal, str(again)) == (type(raised), raised.refusal, str(raised))
    assert refusal_of(ValueError("not a refusal")) is None
    with pytest.raises(ValidationError):
        type(raised)("no_such_reason", "a reason its kind does not know is refused at the raise")


@pytest.mark.parametrize("bare", [Coded, Refused])
def test_a_kind_with_no_reason_type_is_refused_at_its_raise(bare: type[Coded]) -> None:
    """A bare `Coded` / `Refused` names no reason type: raised, it says so, never an `AttributeError` from deep inside."""
    with pytest.raises(TypeError, match="declares no reason type"):
        bare("web_full", "full")
