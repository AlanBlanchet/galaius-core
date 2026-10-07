"""`galaius_core.paid_use` — the deny-by-default paid-API contract.

Written before the module: the edge cases that matter are the REFUSALS (a provider nobody
switched on, a cap nobody set, a cap already crossed), not the happy path.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from galaius_core.paid_use import (
    CHARGE_WORDS,
    DEFAULT_MONTHLY_CAP_USD,
    PaidProviderSwitch,
    PaidProviderUpdate,
    PaidUseCapUpdate,
    PaidUsePolicy,
    PaidUseRefusal,
    PaidUseSpend,
    PaidUseState,
    ProviderChargeState,
    charge_kind,
    charge_word,
    month_start,
)


def _switch(provider: str, cap: float | None = None) -> PaidProviderSwitch:
    return PaidProviderSwitch(provider=provider, enabled_at=datetime(2026, 9, 30, tzinfo=UTC),
                              enabled_by=uuid4(), monthly_cap_usd=cap)


@pytest.mark.parametrize(
    ("route", "provider", "expected"),
    [
        ("cli_session", "anthropic", "subscription"),
        ("cli_session", "openai", "subscription"),
        ("self_hosted", "self_hosted", "local"),
        ("api_key", "self_hosted", "local"),
        ("api_key", "openai_compatible", "local"),
        ("api_key", "ollama", "local"),
        ("api_key", "openai", "metered_api"),
        ("api_key", "anthropic", "metered_api"),
        ("api_key", "gemini", "metered_api"),
        # A route nobody classified must read as the money-spending pole, never as free.
        ("something_new", "openai", "metered_api"),
    ],
)
def test_charge_kind_reads_route_and_provider(route: str, provider: str, expected: str) -> None:
    assert charge_kind(route, provider) == expected


def test_every_charge_kind_has_the_owners_word() -> None:
    assert CHARGE_WORDS == {"subscription": "abonnement", "local": "local", "metered_api": "payant à l'usage"}
    assert charge_word("metered_api") == "payant à l'usage"


def test_a_provider_nobody_switched_on_is_refused() -> None:
    policy = PaidUsePolicy()
    assert policy.providers == ()
    assert policy.allows("openai") is False
    assert policy.allows("anthropic") is False
    assert policy.allows("gemini") is False


def test_a_local_provider_needs_no_switch() -> None:
    assert PaidUsePolicy().allows("self_hosted") is True
    assert PaidUsePolicy().allows("ollama") is True


def test_a_switched_on_provider_is_allowed_and_only_that_one() -> None:
    policy = PaidUsePolicy(providers=(_switch("gemini"),))
    assert policy.allows("gemini") is True
    assert policy.allows("openai") is False


def test_no_cap_set_means_the_default_cap_not_no_cap() -> None:
    policy = PaidUsePolicy()
    assert policy.monthly_cap_usd is None
    assert policy.cap_usd == DEFAULT_MONTHLY_CAP_USD == 10.0


def test_zero_is_an_explicit_no_metered_spend_cap() -> None:
    assert PaidUsePolicy(monthly_cap_usd=0).cap_usd == 0.0


def test_a_provider_cap_never_raises_the_workspace_cap() -> None:
    policy = PaidUsePolicy(monthly_cap_usd=5.0, providers=(_switch("openai", cap=50.0), _switch("gemini", cap=1.0)))
    assert policy.cap_for("openai") == 5.0
    assert policy.cap_for("gemini") == 1.0
    assert policy.cap_for("anthropic") == 5.0


def test_spend_counts_the_fee_against_the_cap() -> None:
    spend = PaidUseSpend(period_start=month_start(datetime(2026, 9, 30, tzinfo=UTC)), spent_usd=9.6, fee_usd=0.5, cap_usd=10.0)
    assert spend.total_usd == pytest.approx(10.1)
    assert spend.over_cap is True
    assert spend.remaining_usd == 0.0
    assert spend.fraction == 1.0


def test_spend_under_the_cap_leaves_a_remainder() -> None:
    spend = PaidUseSpend(period_start=month_start(), spent_usd=2.0, fee_usd=0.1, cap_usd=10.0)
    assert spend.over_cap is False
    assert spend.remaining_usd == pytest.approx(7.9)
    assert spend.fraction == pytest.approx(0.21)


def test_a_zero_cap_reads_full_at_zero_spend() -> None:
    spend = PaidUseSpend(period_start=month_start(), spent_usd=0.0, cap_usd=0.0)
    assert spend.over_cap is True
    assert spend.fraction == 1.0


def test_month_start_is_the_first_instant_of_the_month() -> None:
    assert month_start(datetime(2026, 9, 30, 13, 45, 12, tzinfo=UTC)) == datetime(2026, 9, 1, tzinfo=UTC)


def test_a_cap_refusal_must_carry_the_figures_it_refused_on() -> None:
    with pytest.raises(ValidationError):
        PaidUseRefusal(code="monthly_cap_reached", provider="openai", message="over")
    refusal = PaidUseRefusal(code="monthly_cap_reached", provider="openai", message="over",
                             spend=PaidUseSpend(period_start=month_start(), spent_usd=10.0, cap_usd=10.0))
    assert refusal.spend is not None


def test_a_not_enabled_refusal_needs_no_figures() -> None:
    refusal = PaidUseRefusal(code="paid_use_not_enabled", provider="openai",
                             message="paid use of openai is off for this workspace")
    assert refusal.spend is None


def test_state_carries_a_row_per_provider_with_its_word() -> None:
    state = PaidUseState(
        policy=PaidUsePolicy(providers=(_switch("gemini"),)),
        spend=PaidUseSpend(period_start=month_start(), spent_usd=0.2, fee_usd=0.01, cap_usd=10.0),
        providers=(
            ProviderChargeState(provider="anthropic", charge="subscription", allowed=True),
            ProviderChargeState(provider="self_hosted", charge="local", allowed=True),
            ProviderChargeState(provider="gemini", charge="metered_api", allowed=True, spent_usd=0.2),
            ProviderChargeState(provider="openai", charge="metered_api", allowed=False,
                                refusal="paid use of openai is off for this workspace"),
        ),
    )
    assert [item.word for item in state.providers] == ["abonnement", "local", "payant à l'usage", "payant à l'usage"]
    assert [item.allowed for item in state.providers] == [True, True, True, False]


def test_updates_are_wire_requests_with_bounded_numbers() -> None:
    assert PaidProviderUpdate(enabled=True).monthly_cap_usd is None
    assert PaidUseCapUpdate(monthly_cap_usd=25.0).monthly_cap_usd == 25.0
    with pytest.raises(ValidationError):
        PaidProviderUpdate(enabled=True, monthly_cap_usd=-1)
    with pytest.raises(ValidationError):
        PaidUseCapUpdate(monthly_cap_usd=-0.5)


# ---------------------------------------------------------------------------------------------
# visual-critic r1 FAIL rows that land on this contract
# ---------------------------------------------------------------------------------------------

def test_a_provider_row_says_unknown_rather_than_zero_when_nobody_attributed_its_spend() -> None:
    """visual-critic r1: every metered row showed 0.20794 — the MONTH total, repeated per
    provider. A figure nobody attributed is unknown, never zero and never the total."""
    row = ProviderChargeState(provider="openai", charge="metered_api", allowed=False)
    assert row.spent_usd is None


def test_a_refusal_states_a_fact_and_leaves_the_sentence_to_the_reader_language() -> None:
    """visual-critic r1, HIGH, routed to the backend: the English refusal was printed verbatim in
    the French UI, and it said "switch it on in Billing" while shown inside Billing. The wire
    carries the CODE, the provider and the figures; the words a reader sees are the client's."""
    refusal = PaidUseRefusal(code="paid_use_not_enabled", provider="openai",
                             message="paid use of openai is not enabled for this workspace")
    assert refusal.code == "paid_use_not_enabled"
    for pointer in ("Billing", "switch it on", "raise the ceiling", "Facturation"):
        assert pointer not in refusal.message
