from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core import (
    ToolTokenUsage,
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
        usage=ToolTokenUsage(**usage), recorded_at=datetime.now(UTC),
    )


def test_token_usage_rejects_an_entirely_unreported_call():
    with pytest.raises(ValidationError):
        ToolTokenUsage()


def test_token_usage_accepts_a_single_reported_figure_leaving_the_rest_unknown():
    usage = ToolTokenUsage(input_tokens=10)
    assert usage.output_tokens is None
    assert usage.cache_read_input_tokens is None
    assert usage.reasoning_tokens is None


def test_ingest_record_carries_no_account_id_field():
    record = _record(input_tokens=1, output_tokens=1)
    assert "account_id" not in type(record).model_fields


def test_ingest_batch_rejects_duplicate_ids_within_one_post():
    with pytest.raises(ValidationError):
        ToolUsageIngestRequest(records=(_record("dup", input_tokens=1), _record("dup", input_tokens=2)))


def test_ingest_batch_accepts_distinct_ids():
    batch = ToolUsageIngestRequest(records=(_record("a", input_tokens=1), _record("b", input_tokens=1)))
    assert len(batch.records) == 2


def test_stored_record_adds_account_id_over_the_ingest_shape():
    record = ToolUsageRecord(
        id="call-1", account_id=uuid4(), provider="openai", model_id="gpt-x", route="cli_session",
        usage=ToolTokenUsage(input_tokens=5, output_tokens=2), recorded_at=datetime.now(UTC),
    )
    assert record.route == "cli_session"


def test_totals_allow_a_metric_to_stay_unreported_across_every_call():
    totals = ToolUsageTotals(calls=3, input_tokens=30, output_tokens=12)
    assert totals.cache_read_input_tokens is None
    assert totals.reasoning_tokens is None


def test_summary_groups_entries_by_provider_model_and_route():
    summary = ToolUsageSummary(entries=(
        ToolUsageSummaryEntry(provider="anthropic", model_id="claude-x", route="api_key",
                               totals=ToolUsageTotals(calls=2, input_tokens=20, output_tokens=8)),
        ToolUsageSummaryEntry(provider="anthropic", model_id="claude-x", route="cli_session",
                               totals=ToolUsageTotals(calls=1, input_tokens=5, output_tokens=2)),
    ))
    assert len(summary.entries) == 2
    assert {entry.route for entry in summary.entries} == {"api_key", "cli_session"}
