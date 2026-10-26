"""`TokenCounts`: the one token-count type — one additive shape and one spelling whatever the vendor, summed with `+`
(a figure stays None only when no part reported it, never a synthetic zero), priced at a model's unit prices."""
import pytest

from galaius_core import TokenCounts, UnitPrice

PRICES = (UnitPrice(unit="token_in", usd_per_unit=3e-6), UnitPrice(unit="token_cache_read", usd_per_unit=0.3e-6),
          UnitPrice(unit="token_cache_write", usd_per_unit=3.75e-6), UnitPrice(unit="token_out", usd_per_unit=15e-6))


@pytest.mark.parametrize(("parts", "total", "prompt", "usd"), [
    ([TokenCounts(input=1000, output=200, cache_read=4000, cache_write=500)], TokenCounts(input=1000, output=200, cache_read=4000, cache_write=500),
     5500, 1000 * 3e-6 + 4000 * 0.3e-6 + 500 * 3.75e-6 + 200 * 15e-6),
    ([TokenCounts(input=10, output=5), TokenCounts(input=20, output=7, cache_read=100)], TokenCounts(input=30, output=12, cache_read=100), 130,
     30 * 3e-6 + 100 * 0.3e-6 + 12 * 15e-6),
    ([TokenCounts(output=3, reasoning=2), TokenCounts(input=4)], TokenCounts(input=4, output=3, reasoning=2), 4, 4 * 3e-6 + 3 * 15e-6),
    ([], TokenCounts(), 0, 0.0),
])
def test_token_counts_add_up_figure_by_figure_and_price_each_kind_at_its_own_rate(parts, total, prompt, usd) -> None:
    summed = TokenCounts.sum(parts)
    assert summed == total and summed.prompt == prompt and summed.counted == prompt + (summed.output or 0)
    assert summed.priced(PRICES) == pytest.approx(usd)
    assert (summed.cache_write is None) == all(part.cache_write is None for part in parts)   # never a synthetic zero
    assert TokenCounts.model_validate(summed.model_dump(mode="json")) == summed and not summed.model_dump(mode="json").keys() - {"input", "output", "cache_read", "cache_write", "reasoning"}


@pytest.mark.parametrize(("now", "before", "grown"), [
    (TokenCounts(input=50, output=20, cache_read=900), TokenCounts(input=30, output=5, cache_read=400), TokenCounts(input=20, output=15, cache_read=500)),
    (TokenCounts(input=7, output=2), TokenCounts(input=30, output=5), TokenCounts(input=7, output=2)),   # went backwards: restarted
    (TokenCounts(input=5), TokenCounts(), TokenCounts(input=5)),
])
def test_a_cumulative_counter_grows_by_what_it_holds_beyond_its_last_report(now, before, grown) -> None:
    assert now.since(before) == grown and (now - now) == TokenCounts(input=0, output=None if now.output is None else 0, cache_read=None if now.cache_read is None else 0)
    assert (before - now).covers(TokenCounts()) and now.covers(now)
