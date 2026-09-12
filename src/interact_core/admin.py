"""Provider-independent operator administration, subscription, and usage contracts."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .wire import WireModel


class SubscriptionLimit(WireModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    value: int = Field(ge=1)


class SubscriptionPlanRef(WireModel):
    id: UUID
    revision: UUID


class SubscriptionPlanDefinition(WireModel):
    id: UUID
    revision: UUID
    parent_revision: UUID | None = None
    name: str = Field(min_length=1, max_length=120)
    entitlements: tuple[str, ...] = Field(default=(), max_length=64)
    limits: tuple[SubscriptionLimit, ...] = Field(default=(), max_length=64)
    created_at: datetime

    @model_validator(mode="after")
    def unique_configuration(self) -> Self:
        if any(not value or len(value) > 80 for value in self.entitlements):
            raise ValueError("entitlements must be bounded non-empty names")
        if len(set(self.entitlements)) != len(self.entitlements):
            raise ValueError("entitlements must be unique")
        if len({value.name for value in self.limits}) != len(self.limits):
            raise ValueError("subscription limits must be unique")
        return self

    def limit(self, name: str) -> int | None:
        value = next((item for item in self.limits if item.name == name), None)
        return None if value is None else value.value


class WorkspaceSubscription(WireModel):
    workspace_id: UUID
    plan: SubscriptionPlanRef | None = None
    status: Literal["unconfigured", "active", "suspended", "cancelled"]
    assigned_at: datetime | None = None
    updated_at: datetime

    @model_validator(mode="after")
    def coherent_assignment(self) -> Self:
        configured = self.status != "unconfigured"
        if any(present != configured for present in (self.plan is not None, self.assigned_at is not None)):
            raise ValueError("subscription assignment requires both plan and time exactly when configured")
        return self


class WorkspaceSubscriptionUpdate(WireModel):
    plan: SubscriptionPlanRef
    status: Literal["active", "suspended", "cancelled"]


class UsageSourceRef(WireModel):
    kind: Literal["conversation", "workflow"]
    id: UUID
    provider_response_id: str | None = Field(default=None, min_length=1, max_length=256)


class UsageRecord(WireModel):
    id: UUID
    workspace_id: UUID
    source: UsageSourceRef
    provenance: Literal["measured", "estimated", "unknown"]
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    recorded_at: datetime

    @model_validator(mode="after")
    def coherent_usage(self) -> Self:
        if self.provenance == "unknown":
            if self.input_tokens is not None or self.output_tokens is not None or self.total_tokens is not None:
                raise ValueError("unknown usage cannot carry synthetic zeroes or estimates")
            return self
        if self.input_tokens is None or self.output_tokens is None or self.total_tokens is None:
            raise ValueError("measured and estimated usage require complete token quantities")
        if self.total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("usage total must equal input plus output tokens")
        if self.provenance == "measured" and self.source.provider_response_id is None:
            raise ValueError("provider-reported usage requires a provider response id")
        return self


class UsageTotals(WireModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    calls: int = Field(ge=0)


class UsageProvenanceSummary(WireModel):
    measured: UsageTotals
    estimated: UsageTotals
    unknown_calls: int = Field(ge=0)
    period_start: datetime


class BudgetDecision(WireModel):
    status: Literal["allowed", "denied", "unknown"]
    measured_total_tokens: int = Field(ge=0)
    requested_output_tokens: int = Field(ge=1)
    monthly_measured_token_limit: int | None = Field(default=None, ge=1)
    reason: Literal["unlimited", "within_limit", "request_limit", "monthly_limit", "unmeasured_usage", "subscription_inactive"]


class OperatorAuthority(WireModel):
    account_id: UUID
    authorized: Literal[True]
    provisioned_at: datetime


class OperatorWorkspaceSummary(WireModel):
    workspace_id: UUID
    name: str = Field(min_length=1, max_length=120)
    membership_count: int = Field(ge=0)
    subscription: WorkspaceSubscription


class OperatorServiceSummary(WireModel):
    account_count: int = Field(ge=0)
    workspace_count: int = Field(ge=0)
    operator_count: int = Field(ge=0)
    failed_run_count: int = Field(ge=0)
    recorded_at: datetime


class OperatorRunFailure(WireModel):
    workspace_id: UUID
    run_id: UUID
    status: Literal["failed", "cancelled", "interrupted"]
    error: str | None = Field(default=None, max_length=400)
    created_at: datetime
    updated_at: datetime


class OperatorAuditEvent(WireModel):
    id: UUID
    actor_account_id: UUID | None = None
    action: Literal[
        "operator.provisioned", "account.listed", "workspace.listed", "service.inspected",
        "run_failures.listed", "usage.inspected", "plan.saved", "subscription.assigned",
        "usage.recorded", "budget.evaluated",
    ]
    target_kind: Literal["account", "workspace", "service", "run", "plan", "subscription", "usage"]
    target_id: UUID | None = None
    created_at: datetime
