"""What an account's local tool actually spent, by provider, model and ROUTE.

A subscription-authenticated CLI session (Claude Code, Codex) and an API key are billed on
different meters — mixing their token counts into one number answers a question nobody asked.
Every token FIELD is independently optional: a provider that never reports a figure leaves it
None here, never a synthetic 0 standing in for "nobody said".

Field-to-provider mapping, verified against each vendor's own docs
(``~/.github/research/llm-api-token-usage-fields-2026.md``, retrieved 2026-09-21):

- Anthropic Messages ``usage``: ``input_tokens``/``output_tokens`` (post-cache-breakpoint,
  ADDITIVE with the two cache fields below — ``total = input + cache_read + cache_creation +
  output``), ``cache_read_input_tokens``, ``cache_creation_input_tokens``. No ``total_tokens``.
- OpenAI Responses/Chat Completions ``usage``: ``input_tokens``/``prompt_tokens``,
  ``output_tokens``/``completion_tokens``, ``total_tokens``; ``cached_tokens`` and
  ``reasoning_tokens`` are SUBSETS already counted inside input/output, not additive.
- Google ``usageMetadata``: ``promptTokenCount``, ``candidatesTokenCount``, ``totalTokenCount``;
  ``cachedContentTokenCount`` is a subset of the prompt count, ``thoughtsTokenCount`` is additive
  (folded into ``totalTokenCount`` alongside prompt and candidate counts).

Because the accounting relationship differs per provider, this module names the FIELDS without
asserting one cross-provider arithmetic invariant on them — a reader interprets
``cache_read_input_tokens``/``reasoning_tokens`` against the provider named on the same record.
"""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .wire import WireModel

#: How the call reached the model — a saved provider API key, or the vendor CLI's own logged-in
#: subscription session (Claude Code, Codex). Distinct billing meters, never merged.
ToolUsageRoute = Literal["api_key", "cli_session"]

_TOKEN_FIELDS = (
    "input_tokens", "output_tokens", "cache_read_input_tokens",
    "cache_creation_input_tokens", "reasoning_tokens", "total_tokens",
)


class ToolTokenUsage(WireModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def at_least_one_figure(self) -> Self:
        if all(getattr(self, field) is None for field in _TOKEN_FIELDS):
            raise ValueError("token usage requires at least one reported figure")
        return self


class ToolUsageIngestRecord(WireModel):
    """One call, as the local tool reports it. No account id: the authenticated session that
    posts the batch IS the account, so a client-supplied id here could log usage onto another
    account. ``id`` is the tool's own idempotency key for this call — a retried post after a
    dropped response must not double-count it."""

    id: str = Field(min_length=1, max_length=160)
    provider: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_-]*$")
    model_id: str = Field(min_length=1, max_length=160)
    route: ToolUsageRoute
    usage: ToolTokenUsage
    cost_usd: float | None = Field(default=None, ge=0)
    recorded_at: datetime


class ToolUsageIngestRequest(WireModel):
    records: tuple[ToolUsageIngestRecord, ...] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def unique_record_ids(self) -> Self:
        if len({record.id for record in self.records}) != len(self.records):
            raise ValueError("usage records must have unique ids within one batch")
        return self


class ToolUsageRecord(WireModel):
    """A stored record: the ingest record plus the account it was posted under."""

    id: str = Field(min_length=1, max_length=160)
    account_id: UUID
    provider: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=160)
    route: ToolUsageRoute
    usage: ToolTokenUsage
    cost_usd: float | None = Field(default=None, ge=0)
    recorded_at: datetime


class ToolUsageTotals(WireModel):
    """Aggregated over ``calls`` records. A metric is None only when NONE of those calls
    reported it — a sum of the calls that DID report it, never a count padded with zeroes for
    the ones that didn't."""

    calls: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class ToolUsageSummaryEntry(WireModel):
    provider: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=160)
    route: ToolUsageRoute
    totals: ToolUsageTotals


class ToolUsageSummary(WireModel):
    period_start: datetime | None = None
    entries: tuple[ToolUsageSummaryEntry, ...] = Field(default=(), max_length=4096)
