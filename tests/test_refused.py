"""`Refused[R]`: the one coded-refusal base a PC's runtime and the server share — a kind declares its reason's type alone,
a bare reason is validated into it at the raise, the refusal a machine answer carries is read off any error in one call,
and a refusal survives a copy or a pickle (a worker thread's error crossing back, a logged copy)."""
import copy
import pickle
from typing import Literal

import pytest
from pydantic import ValidationError

from galaius_core import MACHINE_REFUSAL, refusal_of, AgentDeclined, AgentRefusals, Coded, NoCandidate, WebFull, MachineRefusal, Refused, ScreenRefusal, SubscriptionCap, CapReached, SubscriptionDeclined, UsageLimited


class Agent(Refused[AgentRefusals]):
    pass


class Absent(Refused[AgentRefusals], reason="role_absent"):
    pass


class Answered(Refused[MachineRefusal | None]):
    pass


class Job(Refused[Literal["install_failed", "download_failed"]]):
    pass


CAP = SubscriptionCap(window="weekly", used_percent=81, cap_percent=80, at_cap="cut")


@pytest.mark.parametrize(("raised", "reason", "answered"), [
    (Agent("web_full", "4 conversations are already working here"), WebFull(), WebFull()),
    (Agent("no_candidate", "no CLI here could start it"), NoCandidate(), NoCandidate()),
    (Agent("under_way", "already starting"), AgentDeclined(reason="under_way"), AgentDeclined(reason="under_way")),
    (Absent(sentence="not in this catalog yet"), AgentDeclined(reason="role_absent"), AgentDeclined(reason="role_absent")),
    (Answered(CapReached(cap=CAP), "at 81% of its weekly window"), CapReached(cap=CAP), CapReached(cap=CAP)),
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


def test_a_kind_raised_for_one_reason_names_one_its_type_holds() -> None:
    """A `reason=` its kind's type lacks fails where the kind is written, never at its first raise."""
    with pytest.raises(ValidationError):
        class Typo(Refused[AgentRefusals], reason="role_absnet"):
            pass


@pytest.mark.parametrize("raised", [lambda: Agent(sentence="no reason named"), lambda: Agent("web_full")])
def test_a_refusal_raised_without_its_reason_or_its_sentence_says_which(raised) -> None:
    """A kind raised for no one reason names its reason; every refusal names its English sentence, never an empty log line."""
    with pytest.raises(TypeError, match="no one reason|its sentence"):
        raised()


SKIP = {"provider": "claude", "model": "claude-opus-5-5", "reason": "sandbox_unavailable"}


#: Each as a PC on the release before the leaves sends it: one model dumped whole, every other reason's field beside it.
@pytest.mark.parametrize(("sent", "leaf", "flags"), [
    ({"kind": "agent", "reason": "fence_broken", "skips": [], "web": None}, AgentDeclined, {"fenced"}),
    ({"kind": "agent", "reason": "under_way", "skips": [], "web": None}, AgentDeclined, {"pending"}),
    ({"kind": "agent", "reason": "agents_off", "skips": [], "web": None}, AgentDeclined, set()),
    ({"kind": "agent", "reason": "no_candidate", "skips": [SKIP], "web": None}, NoCandidate, {"fenced"}),
    ({"kind": "agent", "reason": "no_candidate", "skips": [SKIP, {**SKIP, "reason": "quota_exceeded"}], "web": None}, NoCandidate, set()),
    ({"kind": "agent", "reason": "web_full", "skips": [], "web": None}, WebFull, set()),
    ({"kind": "subscription", "reason": "not_signed_in", "cap": None}, SubscriptionDeclined, {"signed_out"}),
    ({"kind": "subscription", "reason": "lent", "cap": None}, SubscriptionDeclined, set()),
    ({"kind": "subscription", "reason": "capped", "cap": CAP.model_dump(mode="json")}, CapReached, {"spent"}),
    ({"kind": "subscription", "reason": "usage_limit", "cap": None}, UsageLimited, {"spent"}),
])
def test_a_refusal_from_a_pc_before_the_leaves_reads_as_its_leaf(sent: dict, leaf: type, flags: set[str]) -> None:
    """Its reason picks its leaf whatever other fields ride beside it, and each flag the server reads off it holds."""
    read = MACHINE_REFUSAL.validate_python(sent)
    assert (type(read), read.reason, {flag for flag in ("fenced", "pending", "signed_out", "spent") if getattr(read, flag, False)}) == (leaf, sent["reason"], flags)
