from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from galaius_core import (
    TokenCounts,
    ToolUsageIngestRecord,
    ToolUsageIngestRequest,
    ToolUsageRecord,
    ToolUsageSummary,
    ToolUsageSummaryEntry,
    ToolUsageTotals,
)


def _record(id: str = "call-1", route: str = "api_key", **usage) -> ToolUsageIngestRecord:
    return ToolUsageIngestRecord(
        id=id, provider="anthropic", model_id="claude-x", route=route,
        usage=TokenCounts(**usage), recorded_at=datetime.now(UTC),
    )


def test_an_ingest_record_reports_at_least_one_figure_and_leaves_the_rest_unknown():
    with pytest.raises(ValidationError):
        _record()
    assert _record(input=10).usage == TokenCounts(input=10)


def test_ingest_record_carries_no_account_id_field():
    record = _record(input=1, output=1)
    assert "account_id" not in type(record).model_fields


def test_ingest_batch_rejects_duplicate_ids_within_one_post():
    with pytest.raises(ValidationError):
        ToolUsageIngestRequest(records=(_record("dup", input=1), _record("dup", input=2)))


def test_ingest_batch_accepts_distinct_ids():
    batch = ToolUsageIngestRequest(records=(_record("a", input=1), _record("b", input=1)))
    assert len(batch.records) == 2


def test_stored_record_adds_account_id_over_the_ingest_shape():
    record = ToolUsageRecord(
        id="call-1", account_id=uuid4(), provider="openai", model_id="gpt-x", route="cli_session",
        usage=TokenCounts(input=5, output=2), recorded_at=datetime.now(UTC),
    )
    assert record.route == "cli_session"


def test_summary_groups_entries_by_provider_model_and_route():
    summary = ToolUsageSummary(entries=(
        ToolUsageSummaryEntry(provider="anthropic", model_id="claude-x", route="api_key",
                               totals=ToolUsageTotals(calls=2, usage=TokenCounts(input=20, output=8))),
        ToolUsageSummaryEntry(provider="anthropic", model_id="claude-x", route="cli_session",
                               totals=ToolUsageTotals(calls=1, usage=TokenCounts(input=5, output=2))),
    ))
    assert len(summary.entries) == 2
    assert {entry.route for entry in summary.entries} == {"api_key", "cli_session"}
