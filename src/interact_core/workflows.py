"""Provider-independent immutable workflow and execution wire contracts."""

import hashlib
import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self, get_args
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, FiniteFloat, HttpUrl, SecretStr, field_validator, model_validator

from .prompts import PromptExecutionRef, PromptRevision

from .wire import WireModel

ValueType = Literal["text", "number", "boolean", "json", "artifact", "image", "mask", "mesh", "boxes"]
#: Value types carried as a stored file; `boxes` is structured JSON (label, score, box per item).
FILE_VALUE_TYPES = frozenset({"artifact", "image", "mask", "mesh"})
WorkflowValue = str | float | bool | dict[str, object] | list[object]
WorkspaceApiKeyScope = Literal["read", "write", "execute"]


class WorkflowKey(WireModel):
    id: UUID


class WorkflowRevisionRef(WireModel):
    key: WorkflowKey
    revision: UUID


class MachineRef(WireModel):
    id: UUID


VisionModel = Literal["facebook/detr-resnet-50", "facebook/detr-resnet-50-panoptic"]


class MachineRuntime(WireModel):
    provider: Literal["claude", "codex"]
    version: str | None = Field(default=None, max_length=80)


AcceleratorKind = Literal["cuda", "mps", "rocm", "none"]


class MachineAccelerator(WireModel):
    """One GPU (or `none`) the runner detected, so model steps default to a machine that has one."""

    kind: AcceleratorKind
    name: str = Field(min_length=1, max_length=120)
    memory_mb: int = Field(ge=0, le=1 << 20)


class MachineSummary(WireModel):
    id: UUID
    name: str = Field(min_length=1, max_length=120)
    state: Literal["online", "offline", "revoked"]
    runtimes: tuple[MachineRuntime, ...] = Field(default=(), max_length=16)
    accelerators: tuple[MachineAccelerator, ...] = Field(default=(), max_length=16)
    last_seen_at: datetime | None = None


class MachineCreateRequest(WireModel):
    name: str = Field(min_length=1, max_length=120)


class MachineCreated(WireModel):
    machine: MachineSummary
    token: SecretStr = Field(min_length=32, max_length=256)


class WorkspaceApiKeyCreate(WireModel):
    scopes: tuple[WorkspaceApiKeyScope, ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_scopes(self) -> Self:
        if len(set(self.scopes)) != len(self.scopes):
            raise ValueError("workspace API key scopes must be unique")
        return self


class WorkspaceApiKeySummary(WireModel):
    id: UUID
    prefix: str = Field(min_length=8, max_length=16, pattern=r"^iwk_[A-Za-z0-9_-]+$")
    scopes: tuple[WorkspaceApiKeyScope, ...] = Field(min_length=1, max_length=3)
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


class WorkspaceApiKeyCreated(WireModel):
    key: WorkspaceApiKeySummary
    secret: SecretStr = Field(min_length=32, max_length=256)


class ManualTrigger(WireModel):
    kind: Literal["manual"]


class ScheduleTrigger(WireModel):
    kind: Literal["schedule"]
    cron: str = Field(min_length=9, max_length=120)
    timezone: str = Field(min_length=1, max_length=80)

    @field_validator("cron")
    @classmethod
    def validate_cron(cls, value: str) -> str:
        fields = value.split()
        if len(fields) != 5:
            raise ValueError("schedule cron must contain exactly five fields")
        limits = ((0, 59), (0, 23), (1, 31), (1, 12), (0, 7))
        for field, (minimum, maximum) in zip(fields, limits, strict=True):
            cls._cron_values(field, minimum, maximum)
        return value

    @staticmethod
    def _cron_values(field: str, minimum: int, maximum: int) -> frozenset[int]:
        values: set[int] = set()
        for component in field.split(","):
            expression, separator, raw_step = component.partition("/")
            if separator and (not raw_step.isdigit() or int(raw_step) < 1):
                raise ValueError("schedule cron step is invalid")
            step = int(raw_step) if separator else 1
            if expression == "*":
                start, end = minimum, maximum
            elif "-" in expression:
                raw_start, raw_end = expression.split("-", 1)
                if not raw_start.isdigit() or not raw_end.isdigit():
                    raise ValueError("schedule cron range is invalid")
                start, end = int(raw_start), int(raw_end)
            elif expression.isdigit() and not separator:
                start = end = int(expression)
            else:
                raise ValueError("schedule cron field is invalid")
            if start < minimum or end > maximum or start > end:
                raise ValueError("schedule cron value is out of range")
            values.update(range(start, end + 1, step))
        if not values:
            raise ValueError("schedule cron field is empty")
        return frozenset(values)

    @field_validator("timezone")
    @classmethod
    def validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as error:
            raise ValueError("schedule timezone must be an IANA timezone") from error
        return value

    def next_fire_after(self, instant: datetime) -> datetime:
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("schedule evaluation requires a timezone-aware instant")
        minute_field, hour_field, day_field, month_field, weekday_field = self.cron.split()
        minutes = self._cron_values(minute_field, 0, 59)
        hours = self._cron_values(hour_field, 0, 23)
        days = self._cron_values(day_field, 1, 31)
        months = self._cron_values(month_field, 1, 12)
        weekdays = {0 if value == 7 else value for value in self._cron_values(weekday_field, 0, 7)}
        unrestricted_day = day_field == "*"
        unrestricted_weekday = weekday_field == "*"
        timezone = ZoneInfo(self.timezone)
        first_date = instant.astimezone(timezone).date()
        candidates: list[datetime] = []
        for offset in range(366 * 8):
            local_date = first_date + timedelta(days=offset)
            if local_date.month not in months:
                continue
            day_matches = local_date.day in days
            weekday_matches = (local_date.weekday() + 1) % 7 in weekdays
            if not (
                day_matches and weekday_matches
                if unrestricted_day or unrestricted_weekday
                else day_matches or weekday_matches
            ):
                continue
            for hour in hours:
                for minute in minutes:
                    wall_time = datetime.combine(local_date, time(hour, minute))
                    for fold in (0, 1):
                        local = wall_time.replace(tzinfo=timezone, fold=fold)
                        candidate = local.astimezone(UTC)
                        if candidate <= instant.astimezone(UTC):
                            continue
                        round_trip = candidate.astimezone(timezone)
                        if round_trip.replace(tzinfo=None) == wall_time and round_trip.fold == fold:
                            candidates.append(candidate)
            if candidates:
                return min(candidates)
        raise ValueError("schedule cron has no fire time within eight years")


class WebhookTrigger(WireModel):
    kind: Literal["webhook"]


TriggerConfiguration = Annotated[ManualTrigger | ScheduleTrigger | WebhookTrigger, Field(discriminator="kind")]


class TriggerInputMapping(WireModel):
    source: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    target_kind: Literal["input", "variable"]
    target: str = Field(min_length=1, max_length=80)


class TriggerConstant(WireModel):
    """A value the BINDING carries, not the request.

    A schedule has no payload, so a scheduled trigger could only start a workflow whose inputs
    were all optional — "connect it to activate stuff" stopped at the first required input. A
    constant supplies that input from the binding itself, and is the only way a schedule reaches
    a workflow that needs values.
    """

    target_kind: Literal["input", "variable"]
    target: str = Field(min_length=1, max_length=80)
    value: WorkflowValue


class TriggerInvocation(WireModel):
    values: dict[str, WorkflowValue] = Field(default_factory=dict, max_length=32)

    @field_validator("values")
    @classmethod
    def bounded_names(cls, value: dict[str, WorkflowValue]) -> dict[str, WorkflowValue]:
        if any(not name or len(name) > 80 for name in value):
            raise ValueError("trigger invocation value names must be bounded")
        return value


class TriggerEnableUpdate(WireModel):
    enabled: bool


class TriggerScheduleState(WireModel):
    next_fire_at: datetime | None = None
    lease_owner: str | None = Field(default=None, min_length=1, max_length=120)
    lease_expires_at: datetime | None = None
    last_fire_at: datetime | None = None

    @model_validator(mode="after")
    def coherent_lease(self) -> Self:
        if (self.lease_owner is None) != (self.lease_expires_at is None):
            raise ValueError("schedule lease owner and expiry must be present together")
        return self


class WebhookCredentialSummary(WireModel):
    prefix: str = Field(min_length=8, max_length=16, pattern=r"^iwh_[A-Za-z0-9_-]+$")
    created_at: datetime


class WebhookCredentialCreated(WireModel):
    credential: WebhookCredentialSummary
    secret: SecretStr = Field(min_length=32, max_length=256)


class TriggerBinding(WireModel):
    workflow: WorkflowRevisionRef
    configuration: TriggerConfiguration
    input_mapping: tuple[TriggerInputMapping, ...] = Field(default=(), max_length=32)
    constants: tuple[TriggerConstant, ...] = Field(default=(), max_length=32)

    @model_validator(mode="after")
    def coherent_mapping(self) -> Self:
        if isinstance(self.configuration, ScheduleTrigger) and self.input_mapping:
            raise ValueError("schedule triggers cannot map request values")
        constant_targets = {(value.target_kind, value.target) for value in self.constants}
        if len(constant_targets) != len(self.constants):
            raise ValueError("trigger constant targets must be unique")
        if constant_targets & {(value.target_kind, value.target) for value in self.input_mapping}:
            raise ValueError("a target takes its value from the request or from a constant, never both")
        if len({value.source for value in self.input_mapping}) != len(self.input_mapping):
            raise ValueError("trigger input mapping sources must be unique")
        targets = {(value.target_kind, value.target) for value in self.input_mapping}
        if len(targets) != len(self.input_mapping):
            raise ValueError("trigger input mapping targets must be unique")
        return self


class TriggerConfigurationUpdate(WireModel):
    """Compare the displayed configuration before replacing it; runtime state is separate."""

    expected: TriggerBinding
    replacement: TriggerBinding

    @model_validator(mode="after")
    def keep_trigger_kind(self) -> Self:
        if self.expected.configuration.kind != self.replacement.configuration.kind:
            raise ValueError("add a new trigger block to change its kind")
        return self


class TriggerCreate(TriggerBinding):
    enabled: Literal[False] = False


class TriggerDefinition(TriggerBinding):
    id: UUID
    enabled: bool
    created_at: datetime
    schedule: TriggerScheduleState | None = None
    webhook_credential: WebhookCredentialSummary | None = None

    @model_validator(mode="after")
    def coherent_runtime_state(self) -> Self:
        if (self.schedule is not None) != isinstance(self.configuration, ScheduleTrigger):
            raise ValueError("schedule state is required only for schedule triggers")
        if self.webhook_credential is not None and not isinstance(self.configuration, WebhookTrigger):
            raise ValueError("webhook credentials belong only to webhook triggers")
        return self


class TriggerDispatch(WireModel):
    id: UUID
    trigger_id: UUID
    source: Literal["manual", "schedule", "webhook"]
    idempotency_key: str = Field(min_length=1, max_length=160)
    scheduled_for: datetime | None = None
    request_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    run_id: UUID | None = None
    status: Literal["claimed", "dispatched", "failed"]
    created_at: datetime
    updated_at: datetime
    error: Literal["workflow_unavailable", "execution_failed"] | None = None


class SubscriptionConfiguration(WireModel):
    status: Literal["unconfigured", "trialing", "active", "past_due", "cancelled"]
    plan: str | None = Field(default=None, min_length=1, max_length=120)
    current_period_ends_at: datetime | None = None
    updated_at: datetime

    @model_validator(mode="after")
    def coherent_status(self) -> Self:
        configured = self.status != "unconfigured"
        if configured != (self.plan is not None):
            raise ValueError("configured subscriptions require a plan")
        if not configured and self.current_period_ends_at is not None:
            raise ValueError("unconfigured subscriptions cannot have a billing period")
        return self


class AdminWorkspaceSummary(WireModel):
    workspace_id: UUID
    membership_count: int = Field(ge=0)
    active_api_key_count: int = Field(ge=0)
    trigger_count: int = Field(ge=0)
    enabled_trigger_count: int = Field(ge=0)
    updated_at: datetime


class PortSpec(WireModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    direction: Literal["input", "output"]
    value_type: ValueType
    required: bool = True
    multiple: bool = False

    def accepts(self, value: object):
        values = value if self.multiple and isinstance(value, list) else [value]
        if self.multiple and not isinstance(value, list):
            return False
        return all(
            isinstance(item, str) if self.value_type == "text" else
            isinstance(item, bool) if self.value_type == "boolean" else
            isinstance(item, (int, float)) and not isinstance(item, bool) if self.value_type == "number" else
            isinstance(item, ArtifactRef) if self.value_type in FILE_VALUE_TYPES else
            isinstance(item, (dict, list))
            for item in values
        )


class PortAddress(WireModel):
    node: UUID
    port: str = Field(min_length=1, max_length=80)


class WorkflowEdge(WireModel):
    source: PortAddress
    target: PortAddress


class PortExposure(WireModel):
    name: str = Field(min_length=1, max_length=80)
    target: PortAddress


class VariableSpec(WireModel):
    name: str = Field(min_length=1, max_length=80)
    target: PortAddress
    value_type: ValueType
    required: bool = True


class WorkflowInterface(WireModel):
    inputs: tuple[PortExposure, ...] = ()
    outputs: tuple[PortExposure, ...] = ()
    variables: tuple[VariableSpec, ...] = ()


class ConnectionResourceRef(WireModel):
    id: UUID
    revision: UUID
    capability: Literal["read", "write", "list", "http", "command"]


class ConfiguredModelRef(WireModel):
    connection: ConnectionResourceRef
    id: str = Field(min_length=1, max_length=160)


class CredentialRef(WireModel):
    id: UUID


class ConnectionSecretUpdate(WireModel):
    secret: SecretStr = Field(min_length=1, max_length=16 * 1024)


class AgentRevisionRef(WireModel):
    id: UUID
    revision: UUID


class MachineCommand(WireModel):
    """One owner-scoped workflow step requested from one enrolled machine."""

    id: UUID
    nonce: UUID
    machine: MachineRef
    workspace_id: UUID
    run_id: UUID
    workflow: WorkflowRevisionRef
    node_id: UUID
    action: Literal["agent", "model"] = "agent"
    agent: AgentRevisionRef | None = None
    model: VisionModel | None = None
    image_paths: tuple[str, ...] = Field(default=(), max_length=16)
    score_threshold: FiniteFloat = Field(default=0.5, ge=0, le=1)
    expires_at: datetime
    task: str | None = Field(default=None, min_length=1, max_length=1 << 16)
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def coherent_action(self) -> Self:
        if self.action == "agent" and (self.agent is None or self.task is None or self.model is not None or self.image_paths):
            raise ValueError("agent commands require an agent and task only")
        if self.action == "model" and (self.model is None or not self.image_paths or self.agent is not None or self.task is not None):
            raise ValueError("model commands require a model and image paths only")
        return self


class MachineCommandResult(WireModel):
    command_id: UUID
    nonce: UUID
    status: Literal["succeeded", "failed", "cancelled"]
    result: str | None = Field(default=None, max_length=1 << 20)
    error: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def coherent_result(self) -> Self:
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("failed machine command requires an error")
        return self


class MachineEvent(WireModel):
    command_id: UUID
    sequence: int = Field(ge=1)
    kind: Literal["started", "progress", "result", "error"]
    timestamp: datetime
    payload: dict[str, object] = Field(default_factory=dict)


class ToolInputProperty(WireModel):
    type: Literal["string", "number", "integer", "boolean"]
    description: str | None = Field(default=None, min_length=1, max_length=240)


class ToolInputSchema(WireModel):
    type: Literal["object"] = "object"
    properties: dict[str, ToolInputProperty] = Field(max_length=32)
    required: tuple[str, ...] = Field(max_length=32)
    additional_properties: Literal[False] = False

    @model_validator(mode="after")
    def coherent_properties(self) -> Self:
        if any(not name or len(name) > 80 for name in self.properties):
            raise ValueError("tool input property names must be bounded")
        if len(set(self.required)) != len(self.required) or not set(self.required) <= set(self.properties):
            raise ValueError("tool required inputs must name unique properties")
        return self

    def validate_arguments(self, arguments: dict[str, object]) -> dict[str, object]:
        if set(arguments) - set(self.properties) or not set(self.required) <= set(arguments):
            raise ValueError("tool arguments do not match the declared input")
        for name, value in arguments.items():
            expected = self.properties[name].type
            valid = (
                isinstance(value, str) if expected == "string" else
                isinstance(value, bool) if expected == "boolean" else
                isinstance(value, int) and not isinstance(value, bool) if expected == "integer" else
                isinstance(value, (int, float)) and not isinstance(value, bool)
            )
            if not valid:
                raise ValueError("tool arguments do not match the declared input")
        return arguments


ConnectorKind = Literal["google_drive", "github", "slack", "notion", "gmail", "sharepoint", "onedrive"]
ConnectorActionName = Literal[
    "list_files", "list_repositories", "list_channels", "search_pages", "list_sites",
    "search_metadata", "read_message", "send_message",
]
ConnectorBrowseActionName = Literal["list_files", "list_repositories", "list_channels", "search_pages", "list_sites"]
ConnectorAuthKind = Literal["google_oauth", "microsoft_oauth", "access_token"]
#: How a workflow node's compute is actually reached, right now — never a label. "vendor_api" and
#: "vendor_cli_session" are the two existing `ModelRoute` routes to a vendor-hosted model,
#: generalized onto a node; "self_hosted" is the owner's own machine (a `provider_api` connection
#: whose `provider == "self_hosted"`) — the third pole the owner named and nothing vendor sees.
Sovereignty = Literal["vendor_api", "vendor_cli_session", "self_hosted"]


class ConnectorAction(WireModel):
    connector: ConnectorKind
    name: ConnectorActionName
    method: Literal["GET", "POST"]
    endpoint: HttpUrl
    auth_kind: ConnectorAuthKind
    required_access: tuple[str, ...] = Field(min_length=1, max_length=8)
    input_schema: ToolInputSchema
    item_kind: Literal["file", "repository", "channel", "page", "message", "site"]
    docs_url: HttpUrl


class ConnectorDefinition(WireModel):
    connector: ConnectorKind
    name: str = Field(min_length=1, max_length=120)
    auth_kinds: tuple[ConnectorAuthKind, ...] = Field(min_length=1, max_length=2)
    actions: tuple[ConnectorAction, ...] = Field(min_length=1, max_length=8)
    docs_url: HttpUrl


class ConnectorCatalog(WireModel):
    version: Literal["v1"] = "v1"
    connectors: tuple[ConnectorDefinition, ...] = Field(min_length=1, max_length=8)


class ConnectorCheckRequest(WireModel):
    connector: ConnectorKind
    connection: ConnectionResourceRef | None = None


class ConnectorCheck(WireModel):
    connector: ConnectorKind
    connection: ConnectionResourceRef | None = None
    status: Literal["ready", "unconfigured", "unauthorized", "rate_limited", "unavailable"]
    auth_kind: ConnectorAuthKind
    required_access: tuple[str, ...] = Field(min_length=1, max_length=8)
    credential_expires_at: datetime | None = None


class ConnectorBrowseRequest(WireModel):
    action: ConnectorBrowseActionName
    connection: ConnectionResourceRef | None = None
    query: str | None = Field(default=None, max_length=512)
    cursor: str | None = Field(default=None, min_length=1, max_length=2048)
    limit: int = Field(default=25, ge=1, le=50)


class ConnectorLeaf(WireModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    type: Literal["text", "number", "boolean", "url", "null"]
    value: str | float | bool | None

    @model_validator(mode="after")
    def matching_value(self) -> Self:
        valid = (
            self.value is None if self.type == "null" else
            isinstance(self.value, str) if self.type in {"text", "url"} else
            isinstance(self.value, (int, float)) and not isinstance(self.value, bool) if self.type == "number" else
            isinstance(self.value, bool)
        )
        if not valid:
            raise ValueError("connector leaf value does not match its type")
        return self


class ConnectorItem(WireModel):
    id: str = Field(min_length=1, max_length=512)
    title: str = Field(min_length=1, max_length=512)
    kind: Literal["file", "repository", "channel", "page", "message", "site"]
    url: HttpUrl | None = None
    fields: tuple[ConnectorLeaf, ...] = Field(default=(), max_length=32)


class ConnectorBrowse(WireModel):
    connector: ConnectorKind
    action: ConnectorActionName
    status: Literal["ready"] = "ready"
    items: tuple[ConnectorItem, ...] = Field(max_length=50)
    next_cursor: str | None = Field(default=None, max_length=2048)


class HttpAgentTool(WireModel):
    kind: Literal["http"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    connection: ConnectionResourceRef
    method: Literal["GET", "POST"]
    path: str = Field(min_length=1, max_length=512)
    input_schema: ToolInputSchema
    json_body_from_arguments: bool

    @model_validator(mode="after")
    def coherent_request(self) -> Self:
        path = PurePosixPath(self.path)
        if self.connection.capability != "http" or path.is_absolute() or str(path) != self.path or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("HTTP agent tools require a normalized relative HTTP resource path")
        if self.method == "GET" and self.json_body_from_arguments:
            raise ValueError("GET agent tools cannot map arguments to a JSON body")
        if not self.json_body_from_arguments and self.input_schema.properties:
            raise ValueError("tool inputs require explicit JSON body mapping")
        return self


class DelegatedAgentTool(WireModel):
    kind: Literal["delegate"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    agent: AgentRevisionRef
    input_schema: ToolInputSchema


class WorkflowFunctionTool(WireModel):
    """A 'function' as a tool: a block already in the graph, reused. Not bespoke code — a pinned,
    ALREADY-SAVED workflow (deterministic transform nodes only, enforced at save time) exposed as
    a callable with named inputs and one scalar result. Opens no code-execution boundary."""

    kind: Literal["function"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    workflow: WorkflowRevisionRef
    input_schema: ToolInputSchema


class GmailAgentTool(WireModel):
    kind: Literal["gmail"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    operation: Literal["search_metadata", "read_message", "send_message"]
    input_schema: ToolInputSchema


class ConnectorAgentTool(WireModel):
    kind: Literal["connector"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    connector: ConnectorKind
    action: ConnectorBrowseActionName
    connection: ConnectionResourceRef | None = None
    input_schema: ToolInputSchema


#: Capability an SSH operation needs granted on its `ConnectionResource` — reuses the same
#: read/write/list vocabulary SFTP shares with workspace storage; "command" is the one SSH-only
#: grant, since running an arbitrary remote command is a distinct risk from reading one file.
SshOperationName = Literal["run_command", "list_directory", "read_file", "write_file"]


class SshAgentTool(WireModel):
    """One SSH/SFTP operation bound to a pinned `ssh_server` connection. Running a command and
    writing a file are effects on someone else's machine — the server gates both behind the same
    owner-approval-by-default ledger `GmailAgentTool.send_message` uses, switchable per
    (agent, connection) grant; listing and reading run immediately, like Gmail's reads."""

    kind: Literal["ssh"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    connection: ConnectionResourceRef
    operation: SshOperationName
    input_schema: ToolInputSchema


#: Capability an object-storage operation needs granted on its `ConnectionResource`; "write"
#: (`put_object`) is owner-approval-gated by the same generalized ledger as SSH writes.
ObjectStorageOperationName = Literal["list_objects", "get_object", "put_object"]


class ObjectStorageAgentTool(WireModel):
    """One S3-compatible or Azure Blob operation against a named bucket/container, bound to a
    pinned `object_storage` connection. The connection is provider-agnostic (AWS S3, Scaleway,
    OVH, MinIO, Cloudflare R2, Azure Blob all reach this same tool shape); `bucket` is pinned per
    tool the way `HttpAgentTool.path` is pinned per tool, never left to agent-chosen free text."""

    kind: Literal["object_storage"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    connection: ConnectionResourceRef
    operation: ObjectStorageOperationName
    bucket: str = Field(min_length=1, max_length=255)
    input_schema: ToolInputSchema


AgentCapability = Annotated[HttpAgentTool | DelegatedAgentTool | GmailAgentTool | ConnectorAgentTool | SshAgentTool | ObjectStorageAgentTool | WorkflowFunctionTool, Field(discriminator="kind")]


class AgentRevision(WireModel):
    id: UUID
    revision: UUID
    parent_revision: UUID | None = None
    name: str = Field(min_length=1, max_length=120)
    role_key: str | None = Field(default=None, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=120)
    description: str = Field(default="", max_length=8192)
    scope: str = Field(default="core", pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=120)
    department: str | None = Field(default=None, min_length=1, max_length=120)
    reports_to: UUID | None = None
    reasoning: Literal["minimal", "low", "medium", "high", "xhigh", "max", "ultra"] = "high"
    harness_tools: tuple[str, ...] = Field(default=(), max_length=256)
    prompt: PromptExecutionRef
    paradigms: tuple[PromptExecutionRef, ...] = Field(default=(), max_length=16)
    skill_paradigms: tuple[PromptExecutionRef, ...] = Field(default=(), max_length=64)
    model: ConfiguredModelRef | None = None
    criteria: str | None = Field(default=None, min_length=1, max_length=2048)
    criteria_weights: str = Field(default="", max_length=2048)
    resources: tuple[ConnectionResourceRef, ...] = Field(max_length=32)
    capabilities: tuple[AgentCapability, ...] = Field(default=(), max_length=1000)
    created_at: datetime

    @property
    def execution_model(self) -> ConfiguredModelRef:
        if self.model is None:
            raise ValueError("agent model is not configured")
        return self.model

    @model_validator(mode="after")
    def coherent_capabilities(self) -> Self:
        if len({capability.name for capability in self.capabilities}) != len(self.capabilities):
            raise ValueError("agent capability names must be unique")
        resources = set(self.resources)
        if any(isinstance(capability, HttpAgentTool) and capability.connection not in resources for capability in self.capabilities):
            raise ValueError("HTTP agent tools must use an explicitly connected resource")
        if any(isinstance(capability, ConnectorAgentTool) and capability.connection is not None and capability.connection not in resources for capability in self.capabilities):
            raise ValueError("connector agent tools must use an explicitly connected resource")
        if self.reports_to == self.id:
            raise ValueError("an agent cannot report to itself")
        if len(set(self.harness_tools)) != len(self.harness_tools) or any(not tool or len(tool) > 200 or any(char in tool for char in "\n\r\x00") for tool in self.harness_tools):
            raise ValueError("harness tool names must be unique nonempty single-line names")
        references = (self.prompt, *self.paradigms, *self.skill_paradigms)
        if len(set(references)) != len(references):
            raise ValueError("agent prompt and paradigms must be unique")
        return self


class AgentGraph(WireModel):
    """The current workspace agent heads and the selected assistant root."""

    revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_agent: AgentRevisionRef | None = None
    agents: tuple[AgentRevision, ...] = Field(max_length=1000)

    @model_validator(mode="after")
    def valid_root(self) -> Self:
        heads = {agent.id: agent for agent in self.agents}
        if len(heads) != len(self.agents):
            raise ValueError("agent graph contains duplicate identities")
        if self.root_agent is not None:
            root = heads.get(self.root_agent.id)
            if root is None or root.revision != self.root_agent.revision:
                raise ValueError("agent graph root revision is unavailable")
            if root.reports_to is not None:
                raise ValueError("agent graph root must not report to another agent")
        return self


class AgentGraphUpdate(WireModel):
    """Atomic graph edit. ``agents`` contains changed immutable revisions only."""

    expected_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    root_agent: UUID | None = None
    agents: tuple[AgentRevision, ...] = Field(default=(), max_length=1000)

    @model_validator(mode="after")
    def unique_agents(self) -> Self:
        if len({agent.id for agent in self.agents}) != len(self.agents):
            raise ValueError("agent graph update contains duplicate identities")
        return self


class AgentCatalogSnapshot(WireModel):
    """Complete server records and pinned instruction content for rebuildable clients."""

    agents: tuple[AgentRevision, ...]
    paradigms: tuple[PromptRevision, ...]
    root_agent: AgentRevisionRef | None = None
    prompt_heads: tuple[PromptExecutionRef, ...] = ()
    cursor: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(cls, agents: tuple[AgentRevision, ...], paradigms: tuple[PromptRevision, ...],
               root_agent: AgentRevisionRef | None = None,
               prompt_heads: tuple[PromptExecutionRef, ...] = ()):
        agents = tuple(sorted(agents, key=lambda item: str(item.id)))
        paradigms = tuple(sorted(paradigms, key=lambda item: (item.key.namespace, item.key.slug, item.digest)))
        payload = {"agents": [item.model_dump(mode="json") for item in agents], "paradigms": [item.model_dump(mode="json") for item in paradigms]}
        if root_agent is not None:
            payload["root_agent"] = root_agent.model_dump(mode="json")
        prompt_heads = tuple(sorted(prompt_heads, key=lambda item: (item.key.namespace, item.key.slug)))
        if prompt_heads:
            payload["prompt_heads"] = [item.model_dump(mode="json") for item in prompt_heads]
        cursor = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        return cls(agents=agents, paradigms=paradigms, root_agent=root_agent,
                   prompt_heads=prompt_heads, cursor=cursor)

    @model_validator(mode="after")
    def coherent_snapshot(self) -> Self:
        payload = self.model_dump(mode="json", exclude={"cursor"})
        if self.root_agent is None:
            payload.pop("root_agent")
        if not self.prompt_heads:
            payload.pop("prompt_heads")
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        if digest != self.cursor:
            raise ValueError("agent catalog cursor does not match its content")
        agents = {agent.id: agent for agent in self.agents}
        if self.root_agent is not None:
            root = agents.get(self.root_agent.id)
            if root is None or root.revision != self.root_agent.revision or root.reports_to is not None:
                raise ValueError("agent catalog root revision is unavailable or has a parent")
        roles = [agent.role_key for agent in self.agents if agent.role_key is not None]
        if len(agents) != len(self.agents) or len(roles) != len(set(roles)):
            raise ValueError("agent catalog contains duplicate identities")
        prompts = {(item.key.namespace, item.key.slug, item.digest, item.revision) for item in self.paradigms}
        if len(prompts) != len(self.paradigms):
            raise ValueError("agent catalog contains duplicate instruction revisions")
        if len({head.key for head in self.prompt_heads}) != len(self.prompt_heads):
            raise ValueError("agent catalog contains duplicate prompt heads")
        if any((ref.key.namespace, ref.key.slug, ref.digest, ref.revision) not in prompts
               for ref in self.prompt_heads):
            raise ValueError("agent catalog prompt head is unavailable")
        for agent in self.agents:
            for ref in (agent.prompt, *agent.paradigms, *agent.skill_paradigms):
                if (ref.key.namespace, ref.key.slug, ref.digest, ref.revision) not in prompts:
                    raise ValueError("agent catalog is missing a pinned instruction revision")
            seen = {agent.id}
            parent = agent.reports_to
            while parent is not None:
                if parent not in agents or parent in seen:
                    raise ValueError("agent catalog has an invalid reporting hierarchy")
                seen.add(parent)
                parent = agents[parent].reports_to
        return self


class ArtifactRef(WireModel):
    connection: ConnectionResourceRef
    path: str = Field(min_length=1, max_length=512)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1, max_length=160)
    size: int = Field(ge=0)


class WorkflowNode(WireModel):
    id: UUID
    label: str = Field(min_length=1, max_length=120)
    x: float
    y: float
    ports: tuple[PortSpec, ...] = Field(max_length=64)


class InputNode(WorkflowNode):
    kind: Literal["input"]
    value: WorkflowValue


class ProcessingNode(WorkflowNode):
    kind: Literal["processing"]
    operation: Literal["uppercase", "lowercase", "identity", "http_get"]
    model: ConfiguredModelRef | None = None
    connection: ConnectionResourceRef | None = None


class ResultNode(WorkflowNode):
    kind: Literal["result"]
    connection: ConnectionResourceRef | None = None
    artifact_path: str | None = Field(default=None, min_length=1, max_length=512)


class CompositeNode(WorkflowNode):
    kind: Literal["composite"]
    workflow: WorkflowRevisionRef
    variables: dict[str, WorkflowValue] = Field(default_factory=dict)


class AgentTaskNode(WorkflowNode):
    kind: Literal["agent_task"]
    agent: AgentRevisionRef
    parameters: dict[str, WorkflowValue] = Field(default_factory=dict)
    machine: MachineRef | None = None


class ModelTaskNode(WorkflowNode):
    """Run a cached vision model on one enrolled machine over local image paths."""

    kind: Literal["model"]
    model: VisionModel
    machine: MachineRef
    score_threshold: FiniteFloat = Field(default=0.5, ge=0, le=1)


DirectTool = Annotated[HttpAgentTool | GmailAgentTool | ConnectorAgentTool | SshAgentTool | ObjectStorageAgentTool, Field(discriminator="kind")]


class ToolTaskNode(WorkflowNode):
    """A configured connector or API operation, executable without a model call."""

    kind: Literal["tool_task"]
    tool: DirectTool
    arguments: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=32)


Node = Annotated[InputNode | ProcessingNode | ResultNode | CompositeNode | AgentTaskNode | ModelTaskNode | ToolTaskNode, Field(discriminator="kind")]


class WorkflowBlockAvailability(WireModel):
    kind: Literal["input", "processing", "result", "composite", "agent_task", "model", "tool_task"]
    operation: Literal["uppercase", "lowercase", "identity", "http_get"] | None = None
    workflow: WorkflowRevisionRef | None = None
    agent: AgentRevisionRef | None = None
    tool: DirectTool | None = None
    name: str = Field(min_length=1, max_length=120)
    ports: tuple[PortSpec, ...] = Field(max_length=64)
    readiness: Literal["executable", "config_required", "unavailable"]
    required_config_fields: tuple[str, ...] = Field(default=(), max_length=32)
    reason: str = Field(min_length=1, max_length=400)
    #: Set only for `agent_task` blocks whose agent has a directly configured model (never for a
    #: criteria-routed agent, whose route is resolved per run — that stays `None`, not guessed).
    sovereignty: Sovereignty | None = None

    @classmethod
    def builtins(cls):
        entries = []
        for node_type in get_args(get_args(Node)[0]):
            if node_type is ToolTaskNode:
                continue  # Tool blocks come from configured workspace connections.
            kind = get_args(node_type.model_fields["kind"].annotation)[0]
            operations = get_args(node_type.model_fields["operation"].annotation) if node_type is ProcessingNode else (None,)
            for operation in operations:
                required = ("connection",) if operation == "http_get" else ("workflow",) if node_type is CompositeNode else ("agent",) if node_type is AgentTaskNode else ("machine", "model") if node_type is ModelTaskNode else ()
                ports = (
                    (PortSpec(name="images", direction="input", value_type="text", multiple=True), PortSpec(name="result", direction="output", value_type="json"))
                    if node_type is ModelTaskNode else
                    (() if node_type is InputNode else (PortSpec(name="value", direction="input", value_type="text"),)) + (PortSpec(name="result", direction="output", value_type="text"),)
                )
                entries.append(cls(kind=kind, operation=operation, name=operation.replace("_", " ").title() if operation else kind.replace("_", " ").title(), ports=ports, readiness="config_required" if required else "executable", required_config_fields=required, reason="Select the required configuration before execution." if required else "Runs locally without a model provider."))
        entries.append(cls(kind="result", name="Write artifact", ports=(PortSpec(name="value", direction="input", value_type="text"), PortSpec(name="result", direction="output", value_type="artifact")), readiness="config_required", required_config_fields=("connection", "artifact_path"), reason="Select writable workspace storage and a relative artifact path."))
        return tuple(entries)

    @model_validator(mode="after")
    def coherent_source(self) -> Self:
        if (self.kind == "processing") != (self.operation is not None):
            raise ValueError("processing catalog entries require an operation")
        if self.workflow is not None and self.kind != "composite" or self.agent is not None and self.kind != "agent_task":
            raise ValueError("catalog references must match their node kind")
        if (self.kind == "tool_task") != (self.tool is not None):
            raise ValueError("tool catalog entries require a configured tool")
        if self.sovereignty is not None and self.kind != "agent_task":
            raise ValueError("sovereignty applies only to agent blocks")
        if len({port.name for port in self.ports}) != len(self.ports):
            raise ValueError("catalog ports must have unique names")
        return self


ModelComparator = Literal[">", ">=", "<", "<=", "=", "=="]


class ModelProperty(WireModel):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=400)
    source: str = Field(max_length=160)
    kind: Literal["flag", "number"]
    weightable: bool
    rankable: bool = False
    percentile: bool


class ModelCriteriaCatalog(WireModel):
    properties: tuple[ModelProperty, ...]
    comparators: tuple[ModelComparator, ...]


class ModelEligibilityEvidence(WireModel):
    criterion: str = Field(min_length=1, max_length=2048)
    kind: Literal["benchmark", "capability", "availability"]
    outcome: Literal["satisfied", "unsatisfied", "unknown"]
    actual: FiniteFloat | bool | None = None
    expected: FiniteFloat | bool
    comparator: ModelComparator | None = None
    metric: str | None = Field(default=None, max_length=160)
    unit: str | None = Field(default=None, max_length=80)
    score_range: tuple[FiniteFloat, FiniteFloat] | None = None
    higher_is_better: bool | None = None
    source_url: HttpUrl | None = None
    score_url: HttpUrl | None = None
    methodology_url: HttpUrl | None = None
    retrieved: date | None = None
    freshness: Literal["current", "stale", "unknown"] = "unknown"
    reason: str = Field(min_length=1, max_length=400)

    @field_validator("source_url", "score_url", "methodology_url")
    @classmethod
    def public_link(cls, value: HttpUrl | None):
        if value is not None and (value.username or value.password):
            raise ValueError("benchmark links cannot carry credentials")
        return value

    @model_validator(mode="after")
    def coherent_observation(self) -> Self:
        if self.outcome != "unknown" and self.actual is None:
            raise ValueError("known eligibility requires an observation")
        if self.kind == "benchmark" and self.outcome != "unknown" and self.freshness != "current":
            raise ValueError("stale benchmark evidence cannot qualify a model")
        if self.score_range is not None and self.score_range[0] >= self.score_range[1]:
            raise ValueError("benchmark range must be increasing")
        return self


class ModelEligibility(WireModel):
    model: ConfiguredModelRef
    criteria: str = Field(min_length=1, max_length=2048)
    outcome: Literal["eligible", "ineligible", "unknown"]
    rank: int | None = Field(default=None, ge=1)
    evidence: tuple[ModelEligibilityEvidence, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def coherent_outcome(self) -> Self:
        expected = "ineligible" if any(item.outcome == "unsatisfied" for item in self.evidence) else "unknown" if any(item.outcome == "unknown" for item in self.evidence) else "eligible"
        if self.outcome != expected:
            raise ValueError("model eligibility must agree with all required evidence")
        return self


class WorkflowRevision(WireModel):
    key: WorkflowKey
    revision: UUID
    parent_revision: UUID | None = None
    name: str = Field(min_length=1, max_length=120)
    nodes: tuple[Node, ...] = Field(min_length=1, max_length=500)
    edges: tuple[WorkflowEdge, ...] = Field(max_length=1500)
    interface: WorkflowInterface
    created_at: datetime

    @model_validator(mode="after")
    def unique_nodes(self) -> Self:
        if len({node.id for node in self.nodes}) != len(self.nodes):
            raise ValueError("workflow node identifiers must be unique")
        return self


class ConnectionResource(WireModel):
    id: UUID
    revision: UUID
    kind: Literal["workspace_storage", "http_server", "provider_api", "service_connector", "ssh_server", "object_storage"]
    name: str = Field(min_length=1, max_length=120)
    root: str | None = Field(default=None, max_length=1024)
    endpoint: str | None = Field(default=None, max_length=2048)
    credential: CredentialRef | None = None
    provider: Literal["openai", "anthropic", "gemini", "self_hosted", "google_drive", "github", "slack", "notion", "gmail", "sharepoint", "onedrive", "s3_compatible", "azure_blob"] | None = None
    models: tuple[str, ...] = Field(default=(), max_length=256)
    capabilities: tuple[Literal["read", "write", "list", "http", "command"], ...]
    credential_expires_at: datetime | None = None
    #: SSH login name, or the object-storage access-key-id / storage-account name — the one
    #: identity string every non-OAuth remote credential needs alongside its secret.
    username: str | None = Field(default=None, min_length=1, max_length=256)
    #: SHA256 OpenSSH host-key fingerprint pinned on the connection's first successful `ssh_server`
    #: test (trust-on-first-use), shown to the owner then; every later connect refuses a host that
    #: no longer presents this exact key — never silently re-trusted.
    host_key_fingerprint: str | None = Field(default=None, pattern=r"^SHA256:[A-Za-z0-9+/]{43}$")
    #: Object-storage region (SigV4 signing scope). Optional: most S3-compatible vendors accept a
    #: default; AWS S3 buckets outside it reject the signature, so a real bucket names its own.
    region: str | None = Field(default=None, min_length=1, max_length=64)

    @model_validator(mode="after")
    def kind_fields(self) -> Self:
        if self.kind == "workspace_storage" and (self.root is None or self.endpoint is not None):
            raise ValueError("workspace storage requires only a root")
        if self.kind in {"http_server", "provider_api"} and (self.endpoint is None or self.root is not None):
            raise ValueError("server connection requires only an endpoint")
        if self.kind == "service_connector" and (self.endpoint is not None or self.root is not None or self.provider is None or self.credential is None or "list" not in self.capabilities or "write" in self.capabilities):
            raise ValueError("service connector requires a credential, provider, and list capability only")
        # A self-hosted endpoint is the owner's own machine, typically on a trusted/private
        # network with no vendor credential to hold — unlike openai/anthropic/gemini, which
        # always require one.
        if self.kind == "provider_api" and self.provider != "self_hosted" and self.credential is None:
            raise ValueError("provider connection requires a credential reference")
        if self.kind not in {"provider_api", "service_connector", "object_storage"} and self.provider is not None:
            raise ValueError("provider kind is required only for provider, service and object-storage connections")
        if self.kind == "provider_api" and self.provider not in {"openai", "anthropic", "gemini", "self_hosted"}:
            raise ValueError("provider API kind requires a model provider")
        if self.kind not in {"service_connector", "ssh_server", "object_storage"} and self.credential_expires_at is not None:
            raise ValueError("credential expiry belongs only to connections with vendor-issued expiry")
        if self.kind == "service_connector" and self.provider not in {"google_drive", "github", "slack", "notion", "gmail", "sharepoint", "onedrive"}:
            raise ValueError("service connector provider is unsupported")
        if self.kind != "provider_api" and self.models:
            raise ValueError("only provider connections declare models")
        if len(set(self.models)) != len(self.models) or any(not model or len(model) > 160 for model in self.models):
            raise ValueError("provider models must be unique bounded identifiers")
        if self.kind == "ssh_server" and (self.endpoint is None or not self.endpoint.startswith("ssh://") or self.root is not None or self.credential is None or self.username is None or not self.capabilities or set(self.capabilities) - {"command", "read", "write", "list"}):
            raise ValueError("SSH connection requires an ssh:// endpoint, credential, username and at least one of command/read/write/list")
        if self.kind == "object_storage" and (self.endpoint is None or self.root is not None or self.provider not in {"s3_compatible", "azure_blob"} or self.credential is None or self.username is None or not self.capabilities or set(self.capabilities) - {"read", "write", "list"}):
            raise ValueError("object storage connection requires an endpoint, s3_compatible or azure_blob provider, credential, username and at least one of read/write/list")
        if self.kind != "ssh_server" and self.host_key_fingerprint is not None:
            raise ValueError("host key pinning belongs only to SSH connections")
        if self.kind != "object_storage" and self.region is not None:
            raise ValueError("region belongs only to object storage connections")
        if self.kind not in {"ssh_server", "object_storage"} and self.username is not None:
            raise ValueError("username belongs only to SSH and object storage connections")
        return self


class WorkflowRun(WireModel):
    id: UUID
    workflow: WorkflowRevisionRef
    status: Literal["queued", "running", "cancelling", "succeeded", "failed", "cancelled", "interrupted"]
    idempotency_key: str = Field(min_length=1, max_length=160)
    invocation: TriggerInvocation = Field(default_factory=TriggerInvocation)
    created_at: datetime
    updated_at: datetime
    result: WorkflowValue | ArtifactRef | None = None
    error: str | None = Field(default=None, max_length=400)


class WorkflowEvent(WireModel):
    run_id: UUID
    sequence: int = Field(ge=1)
    node_path: tuple[UUID, ...] = ()
    kind: Literal["queued", "started", "progress", "result", "error", "cancelled", "interrupted"]
    timestamp: datetime
    payload: dict[str, object] = Field(default_factory=dict)


class ProviderUsage(WireModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)


class WorkflowAgentActivity(WireModel):
    id: UUID
    parent_activity_id: UUID | None = None
    type: Literal["agent_completion"]
    node_id: UUID
    agent: AgentRevisionRef
    status: Literal["succeeded"]
    provider_response_id: str = Field(min_length=1, max_length=256)
    usage: ProviderUsage


class WorkflowValueSummary(WireModel):
    kind: Literal["text", "json", "binary"]
    size: int = Field(ge=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class WorkflowCapabilityActivity(WireModel):
    id: UUID
    type: Literal["tool", "delegation"]
    node_id: UUID
    parent_activity_id: UUID | None = None
    call_id: str = Field(min_length=1, max_length=256)
    capability: str = Field(min_length=1, max_length=80)
    status: Literal["succeeded", "failed"]
    child_run_id: UUID | None = None
    child_activity_ids: tuple[UUID, ...] = Field(default=(), max_length=64)
    input_summary: WorkflowValueSummary | None = None
    output_summary: WorkflowValueSummary | None = None
    error: Literal["capability_not_allowed", "invalid_arguments", "execution_failed", "depth_limit", "call_limit"] | None = None

    @model_validator(mode="after")
    def coherent_result(self) -> Self:
        if (self.status == "failed") != (self.error is not None):
            raise ValueError("failed capability activity requires an error")
        if self.status == "succeeded" and (self.input_summary is None or self.output_summary is None):
            raise ValueError("successful capability activity requires input and output summaries")
        if self.type == "tool" and (self.child_run_id is not None or self.child_activity_ids):
            raise ValueError("HTTP tool activity cannot reference a child run")
        if self.type == "delegation" and self.child_run_id is None:
            raise ValueError("delegation activity requires a child run")
        return self


class ConversationMessage(WireModel):
    id: UUID
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=1 << 20)
    created_at: datetime
    provider_response_id: str | None = Field(default=None, max_length=256)
    usage: ProviderUsage | None = None


class ConversationCreateRequest(WireModel):
    model: ConfiguredModelRef | None = None
    agent: AgentRevisionRef | None = None
    prompt: PromptExecutionRef | None = None
    input: str = Field(min_length=1, max_length=1 << 20)
    idempotency_key: str = Field(min_length=1, max_length=160)
    max_output_tokens: int = Field(default=2048, ge=1, le=32768)

    @model_validator(mode="after")
    def execution_target(self) -> Self:
        if self.model is None and self.agent is None:
            raise ValueError("choose a configured model or an agent")
        return self


class Conversation(WireModel):
    id: UUID
    revision: int = Field(default=0, ge=0)
    model: ConfiguredModelRef
    agent: AgentRevisionRef | None = None
    prompt: PromptExecutionRef | None = None
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown_outcome"]
    messages: tuple[ConversationMessage, ...]
    created_at: datetime
    updated_at: datetime
    error: str | None = Field(default=None, max_length=400)


class ConversationAppendRequest(WireModel):
    expected_revision: int = Field(ge=0)
    input: str = Field(min_length=1, max_length=1 << 20)
    idempotency_key: str = Field(min_length=1, max_length=160)
    max_output_tokens: int = Field(default=2048, ge=1, le=32768)


class ConversationSummary(WireModel):
    id: UUID
    revision: int = Field(ge=0)
    title: str = Field(max_length=120)
    preview: str = Field(max_length=240)
    status: Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown_outcome"]
    agent: AgentRevisionRef | None = None
    model: ConfiguredModelRef
    updated_at: datetime


class ConversationPage(WireModel):
    items: tuple[ConversationSummary, ...] = Field(max_length=100)
    next_cursor: str | None = Field(default=None, max_length=256)


class ConversationActivity(WireModel):
    conversation_id: UUID
    sequence: int = Field(ge=1)
    kind: Literal["queued", "started", "provider_dispatch", "agent_completion", "message", "delegation", "tool", "cancelled", "error"]
    timestamp: datetime
    parent_run_id: UUID | None = None
    parent_event_sequence: int | None = Field(default=None, ge=1)
    agent_activity: WorkflowAgentActivity | WorkflowCapabilityActivity | None = None


class BudgetPolicy(WireModel):
    monthly_token_limit: int | None = Field(default=None, ge=1)
    max_output_tokens: int = Field(default=2048, ge=1, le=32768)


class UsageSummary(WireModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    period_start: datetime
