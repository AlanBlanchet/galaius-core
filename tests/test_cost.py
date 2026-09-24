from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core.cost import (
    PLATFORM_PRICING,
    BudgetOverrun,
    NodeCostActual,
    NodeCostEstimate,
    NodeCostModel,
    NodeUsage,
    PlatformPricing,
    PriceSource,
    RunCostActual,
    RunCostEstimate,
    UnitPrice,
    priced_cost,
)


def _source(kind: str = "model_registry", reference: str = "anthropic/claude-x") -> PriceSource:
    return PriceSource(kind=kind, reference=reference)


def test_node_usage_rejects_an_entirely_unreported_run():
    with pytest.raises(ValidationError):
        NodeUsage()


def test_node_usage_accepts_a_single_reported_unit_leaving_the_rest_unset():
    usage = NodeUsage(images=1)
    assert usage.token_in is None
    assert usage.gpu_seconds is None


def test_priced_cost_sums_only_the_units_usage_actually_reports():
    prices = (UnitPrice(unit="token_in", usd_per_unit=1.0), UnitPrice(unit="token_out", usd_per_unit=2.0), UnitPrice(unit="image", usd_per_unit=5.0))
    usage = NodeUsage(token_in=100, token_out=50)
    assert priced_cost(prices, usage) == pytest.approx(100 * 1.0 + 50 * 2.0)


def test_priced_cost_is_zero_for_a_price_list_whose_unit_usage_never_reports():
    prices = (UnitPrice(unit="gpu_second", usd_per_unit=0.01),)
    usage = NodeUsage(images=3)
    assert priced_cost(prices, usage) == 0.0


def test_known_node_cost_model_requires_prices_and_a_source():
    with pytest.raises(ValidationError):
        NodeCostModel(node_id=uuid4(), known=True)


def test_unknown_node_cost_model_carries_no_prices():
    with pytest.raises(ValidationError):
        NodeCostModel(node_id=uuid4(), known=False, prices=(UnitPrice(unit="call", usd_per_unit=0.0),))


def test_unknown_node_cost_model_is_representable_with_no_price_data():
    model = NodeCostModel(node_id=uuid4(), known=False, reason="no research-table entry yet")
    assert model.prices == ()
    assert model.source is None


def test_known_node_cost_actual_requires_a_figure():
    with pytest.raises(ValidationError):
        NodeCostActual(node_id=uuid4(), usage=NodeUsage(calls=1), known=True, cost_usd=None)


def test_unknown_node_cost_actual_carries_no_figure():
    with pytest.raises(ValidationError):
        NodeCostActual(node_id=uuid4(), usage=NodeUsage(calls=1), known=False, cost_usd=0.01)


def test_known_node_estimate_requires_an_increasing_range():
    with pytest.raises(ValidationError):
        NodeCostEstimate(node_id=uuid4(), known=True, low_usd=1.0, high_usd=0.5)


def test_unknown_node_estimate_carries_no_range():
    with pytest.raises(ValidationError):
        NodeCostEstimate(node_id=uuid4(), known=False, low_usd=0.0, high_usd=1.0)


def test_run_cost_estimate_unknown_count_must_match_its_nodes():
    known = NodeCostEstimate(node_id=uuid4(), known=True, low_usd=0.1, high_usd=0.2, fee_low_usd=0.005, fee_high_usd=0.01)
    unknown = NodeCostEstimate(node_id=uuid4(), known=False, reason="no price yet")
    with pytest.raises(ValidationError):
        RunCostEstimate(nodes=(known, unknown), low_usd=0.1, high_usd=0.2, unknown_node_count=0)
    estimate = RunCostEstimate(nodes=(known, unknown), low_usd=0.1, high_usd=0.2, unknown_node_count=1)
    assert estimate.unknown_node_count == 1


def test_known_node_estimate_requires_its_fee_range():
    with pytest.raises(ValidationError):
        NodeCostEstimate(node_id=uuid4(), known=True, low_usd=0.1, high_usd=0.2)


def test_run_cost_actual_sums_known_nodes_and_counts_unknown_ones():
    known = NodeCostActual(node_id=uuid4(), usage=NodeUsage(token_in=10, token_out=5), known=True, cost_usd=0.03, fee_usd=0.0015, source=_source())
    unknown = NodeCostActual(node_id=uuid4(), usage=NodeUsage(calls=1), known=False)
    actual = RunCostActual(nodes=(known, unknown), total_usd=0.03, fee_usd=0.0015, unknown_node_count=1)
    assert actual.total_usd == pytest.approx(0.03)
    assert actual.fee_usd == pytest.approx(0.0015)
    assert actual.unknown_node_count == 1


def test_known_node_actual_requires_its_fee_figure():
    with pytest.raises(ValidationError):
        NodeCostActual(node_id=uuid4(), usage=NodeUsage(calls=1), known=True, cost_usd=0.03, source=_source())


def test_budget_overrun_carries_the_estimate_that_triggered_it():
    estimate = RunCostEstimate(nodes=(), low_usd=5.0, high_usd=9.0, unknown_node_count=0)
    overrun = BudgetOverrun(estimate=estimate, spent_this_period_usd=95.0, ceiling_usd=100.0)
    assert overrun.ceiling_usd == 100.0


def test_platform_pricing_is_alans_pick_cost_plus_5_percent_15_per_month():
    """SAID: "Cost + 5%, $15/mo (Recommended)" — `.github/memory/decisions.md`, 2026-09-24."""
    assert PLATFORM_PRICING.monthly_subscription_usd == 15.0
    assert PLATFORM_PRICING.fee_rate == pytest.approx(0.05)


def test_node_cost_model_known_price_rejects_a_fee_rate_with_no_price():
    with pytest.raises(ValidationError):
        NodeCostModel(node_id=uuid4(), known=False, fee_rate=0.05)


def test_node_cost_model_carries_its_fee_rate():
    model = NodeCostModel(node_id=uuid4(), known=True, prices=(UnitPrice(unit="call", usd_per_unit=0.01),), source=_source(), fee_rate=PLATFORM_PRICING.fee_rate)
    assert model.fee_rate == pytest.approx(0.05)


def test_platform_pricing_fee_rate_is_bounded_to_a_fraction():
    with pytest.raises(ValidationError):
        PlatformPricing(monthly_subscription_usd=15.0, fee_rate=1.5)
