"""The model table's two wire models: a shut entry (what cannot run, until when) and a table row (what a start walks)."""

import pytest
from pydantic import ValidationError

from galaius_core import ShutEntry, TableRow


def test_a_shut_entry_round_trips_and_names_what_it_shuts() -> None:
    entry = ShutEntry(provider="claude", login="work", family="fable", reason="pool", until=1_800_000_000.0, count=2)
    assert ShutEntry.model_validate_json(entry.model_dump_json()) == entry


@pytest.mark.parametrize("bad", [{"reason": "sleepy"}, {"until": -1}, {"provider": "Claude Code"}, {"count": 0}])
def test_a_shut_entry_refuses_what_it_cannot_say(bad: dict) -> None:
    fields = {"provider": "claude", "reason": "limit", "until": 1.0, **bad}
    with pytest.raises(ValidationError):
        ShutEntry(**fields)


def test_a_table_row_out_carries_its_reason_and_instant() -> None:
    row = TableRow(rank=1, provider="claude", model="opus", state="out", reason="overload", until=5.0)
    assert (row.state, row.reason, row.until) == ("out", "overload", 5.0)
    with pytest.raises(ValidationError):
        TableRow(rank=0, provider="claude", model="opus", state="now")


def test_a_usage_report_carries_what_cannot_run_on_the_pc() -> None:
    from galaius_core import SubscriptionUsageReport

    report = SubscriptionUsageReport(shut=(ShutEntry(provider="claude", login="", reason="login_lost"), ShutEntry(provider="claude", family="fable", reason="pool", until=9.0)))
    assert SubscriptionUsageReport.model_validate_json(report.model_dump_json()).shut == report.shut
    assert SubscriptionUsageReport().shut == ()
