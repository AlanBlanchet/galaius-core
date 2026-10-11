from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from galaius_core import RunSubscription, SubscriptionAllowance, SubscriptionCap, SubscriptionChoice, SubscriptionChoices, SubscriptionLimits, SubscriptionRefusal, SubscriptionUse, UsageWindow

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
                                             (SubscriptionRefusal, {"reason": "capped"}), (SubscriptionRefusal, {"reason": "lent", "cap": SubscriptionCap(
                                                 window="weekly", used_percent=81, cap_percent=80, at_cap="cut")})])
def test_choices_and_refusals_refuse_incoherent_shapes(model, fields) -> None:
    with pytest.raises(ValidationError):
        model(**fields)


def test_a_start_offers_each_provider_its_own_subscriptions_in_his_order() -> None:
    codex, first, second = _choice("codex"), _choice("claude"), _choice("claude")
    choices = SubscriptionChoices(subscriptions=(codex, first, second))
    assert (choices.of("claude"), choices.of("codex"), SubscriptionChoices().of("claude")) == ((SubscriptionChoice(**first), SubscriptionChoice(**second)), (SubscriptionChoice(**codex),), ())

