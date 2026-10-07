from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from galaius_core.billing import WalletLedgerEntry


def _entry(**overrides) -> WalletLedgerEntry:
    base = dict(
        id=uuid4(), workspace_id=uuid4(), kind="run_debit", amount_usd_micros=-1_050_000,
        balance_after_usd_micros=5_000_000, idempotency_key="k", reference="r",
        description="d", created_at=datetime.now(UTC),
    )
    base.update(overrides)
    return WalletLedgerEntry(**base)


def test_ledger_entry_carries_the_cost_fee_split_for_a_run_debit() -> None:
    """The ledger is the same two-line cost/fee surface the run panel and node card show --
    a run_debit row is not a single pre-summed amount (visual-critic finding, 2026-09-25)."""
    entry = _entry(cost_usd_micros=1_000_000, fee_usd_micros=50_000)
    assert entry.cost_usd_micros == 1_000_000
    assert entry.fee_usd_micros == 50_000


def test_ledger_entry_rejects_a_split_that_does_not_sum_to_the_amount() -> None:
    with pytest.raises(ValidationError):
        _entry(cost_usd_micros=1_000_000, fee_usd_micros=1)


def test_ledger_entry_rejects_cost_without_fee() -> None:
    with pytest.raises(ValidationError):
        _entry(cost_usd_micros=1_000_000, fee_usd_micros=None)


def test_ledger_entry_allows_no_split_for_a_topup() -> None:
    entry = _entry(kind="topup", amount_usd_micros=5_000_000)
    assert entry.cost_usd_micros is None
    assert entry.fee_usd_micros is None
