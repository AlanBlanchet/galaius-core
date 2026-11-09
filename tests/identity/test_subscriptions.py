from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from galaius_core import AGENT_PROGRAMS, PROVIDERS, key_held, Provider, RunMove, program_name, RunSubscription, SubscriptionAllowance, SubscriptionChoice, SubscriptionChoices, SubscriptionLimits, CapReached, SubscriptionUse, UsageWindow

NOW = datetime(2026, 10, 11, 12, tzinfo=UTC)


def usage(**percent: float) -> tuple[UsageWindow, ...]:
    return tuple(UsageWindow(window=window, used_percent=value, observed_at=NOW) for window, value in percent.items())


@pytest.mark.parametrize(("limits", "windows", "allowance", "reached"), [
    (SubscriptionLimits(weekly_percent=80), usage(five_hour=95, weekly=79.9), None, None),  # no cap on five_hour: the vendor's 100 % stays
    (SubscriptionLimits(weekly_percent=80, at_cap="cut"), usage(five_hour=10, weekly=80), None, ("weekly", 80, "cut")),
    (SubscriptionLimits(five_hour_percent=50, weekly_percent=80), usage(five_hour=60, weekly=85), None, ("weekly", 80, "finish")),
    (SubscriptionLimits(weekly_percent=80), usage(weekly=85), ("weekly", 90), None),  # the run's allowance raises the cap
    (SubscriptionLimits(weekly_percent=80), usage(weekly=91), ("weekly", 90), ("weekly", 90, "finish")),
    (SubscriptionLimits(), usage(weekly=100), ("weekly", 90), None),  # an allowance never sets a cap
    (SubscriptionLimits(five_hour_percent=50, weekly_percent=80), usage(five_hour=60, weekly=85), ("weekly", 90), ("five_hour", 50, "finish")),  # nor raises another window's
])
def test_cap_reached_says_window_cap_and_mode(limits, windows, allowance, reached) -> None:
    cap = limits.reached(windows, allowance and SubscriptionAllowance(window=allowance[0], up_to_percent=allowance[1]))
    assert (cap if cap is None else (cap.window, cap.cap_percent, cap.at_cap)) == reached


@pytest.mark.parametrize(("minutes", "window"), [(300, "five_hour"), (10080, "weekly"), (43200, "monthly"), (360, "five_hour")])
def test_vendor_period_maps_to_nearest_window(minutes, window) -> None:
    assert UsageWindow.nearest(minutes) == window


@pytest.mark.parametrize(("model", "fields"), [(SubscriptionUse, {"limits": SubscriptionLimits()}), (SubscriptionUse, {"lent": True, "subscription_id": uuid4()}),
                                             (RunSubscription, {}), (RunSubscription, {"lent": True, "subscription_id": uuid4()}),
                                             (RunSubscription, {"subscription_id": uuid4()}),
                                             (RunSubscription, {"lent": True, "allowance": {"window": "weekly", "up_to_percent": 90}})])
def test_subscription_use_refuses_incoherent_requests(model, fields) -> None:
    with pytest.raises(ValidationError):
        model(**fields)


def _choice(provider: str) -> dict:
    return {"provider": provider, "subscription_id": uuid4(), "limits": SubscriptionLimits()}


TWICE = _choice("claude")


@pytest.mark.parametrize(("model", "fields"), [(SubscriptionChoices, {"subscriptions": (TWICE, TWICE)}),  # each subscription once
                                             (SubscriptionChoices, {"lent": True, "subscriptions": (_choice("codex"),)}),
                                             (CapReached, {})])
def test_choices_and_refusals_refuse_incoherent_shapes(model, fields) -> None:
    with pytest.raises(ValidationError):
        model(**fields)


def test_a_start_offers_each_provider_its_own_subscriptions_in_his_order() -> None:
    codex, first, second = _choice("codex"), _choice("claude"), _choice("claude")
    choices = SubscriptionChoices(subscriptions=(codex, first, second))
    assert (choices.of("claude"), choices.of("codex"), SubscriptionChoices().of("claude")) == ((SubscriptionChoice(**first), SubscriptionChoice(**second)), (SubscriptionChoice(**codex),), ())



@pytest.mark.parametrize(("resume", "on_wire"), [(True, False), (False, True)])
def test_run_move_says_resume_only_when_it_waits(resume: bool, on_wire: bool) -> None:
    """A move that resumes dumps as it always did (an older runner verifies it unchanged); one that waits names it."""
    move = RunMove(run_id=uuid4(), subscription_id=uuid4(), limits=SubscriptionLimits(), resume=resume)
    dumped = move.model_dump(mode="json")
    assert ("resume" in dumped) is on_wire
    assert RunMove.model_validate(dumped).resume is resume


def test_every_provider_is_a_row_and_a_program_is_one_of_them() -> None:
    """2026-11-05 « I want to act generically … you only propose Codex and Cloud Code »: providers are rows of
    `providers.json`, ids once each; a program a PC runs is one row's `program`; a name no program has stays as written."""
    ids = [row.id for row in PROVIDERS]
    assert len(set(ids)) == len(ids) and {"mistral", "gemini", "openrouter", "groq"} <= set(ids)
    assert list(AGENT_PROGRAMS) == [row.agent_program().provider for row in PROVIDERS if row.agent_program()]
    assert [program_name(name) for name in ("claude", "codex", "mistral", "anthropic")] == ["Claude Code", "Codex", "Mistral", "anthropic"]


def test_every_provider_row_whose_login_reaches_a_chat_api_runs_agents() -> None:
    """Owner 2026-11-08 « Why can't I, uh, still add a Mistral subscription »: a row runs agents through its own program,
    or its key through the key harness at its Chat Completions base, held in one variable on any number of PCs. Not
    launchable: only a row with neither (a CLI Galaius does not run, a voice API)."""
    for row in PROVIDERS:
        program = AGENT_PROGRAMS.get(row.agent_program().provider) if row.agent_program() else None
        reaches_chat = row.api is not None and row.api.chat_base() is not None
        assert (program is not None) == (row.program is not None or reaches_chat), row.id
        if row.program is None and program is not None:
            assert program.keyed and program.held == key_held(row.id) and program.name == row.label and program.held.shared
    assert {"mistral", "gemini", "openrouter", "groq", "deepseek"} <= {name for name, program in AGENT_PROGRAMS.items() if program.keyed}
    assert not AGENT_PROGRAMS["claude"].keyed and not AGENT_PROGRAMS["codex"].keyed
    keys = [program.held.variable for program in AGENT_PROGRAMS.values() if program.keyed]
    assert len(set(keys)) == len(keys)  # one program's key never reaches another's API


@pytest.mark.parametrize(("row", "refused"), [
    ({"id": "acme", "label": "Acme", "api": {"connection": "openai_compatible", "endpoint": "https://api.acme.test/v1", "console": "https://acme.test/keys", "model": "acme-1"}}, None),
    ({"id": "acme", "label": "Acme", "api": {"connection": "openai_compatible", "endpoint": "https://api.acme.test/v1", "console": "https://acme.test/keys"}}, "name its model"),
    ({"id": "acme", "label": "Acme", "api": {"connection": "openai_compatible", "endpoint": "http://api.acme.test/v1", "console": "https://acme.test/keys", "model": "a"}}, "pattern"),
    ({"id": "acme", "label": "Acme"}, "signs in through"),
    ({"id": "Acme", "label": "Acme", "cli": {"command": "acme", "sign_in": "acme login"}}, "pattern"),
    ({"id": "acme", "label": "Acme", "api": {"connection": "openai_compatible", "endpoint": "https://api.acme.test/v1", "console": "https://acme.test/keys", "model": "acme-1", "plan": True}}, "needs the row's plans"),
])
def test_a_new_provider_is_one_row(row: dict, refused: str | None) -> None:
    """Adding a vendor is a row: one with a way to sign in parses; a key checked by nothing, a plain-http endpoint, a
    row with no way in or an id outside the id shape is refused."""
    if refused is None:
        assert Provider.model_validate(row).api.model == "acme-1"
        return
    with pytest.raises(ValidationError, match=refused):
        Provider.model_validate(row)


def test_a_run_names_only_a_program_some_row_holds() -> None:
    choice = {"subscription_id": str(uuid4()), "limits": {}}
    assert SubscriptionChoice.model_validate({**choice, "provider": "codex"}).provider == "codex"
    with pytest.raises(ValidationError, match="no agent program"):
        SubscriptionChoice.model_validate({**choice, "provider": "copilot"})
    assert SubscriptionChoice.model_validate({**choice, "provider": "mistral"}).provider == "mistral"
