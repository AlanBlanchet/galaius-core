"""Provider-independent immutable workflow and execution wire contracts."""

import base64
import hashlib
import json
import re
from datetime import UTC, date, datetime, time, timedelta
from pathlib import PurePosixPath
from collections.abc import Iterable, Iterator
from typing import Annotated, Any, ClassVar, Literal, Self, get_args
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AfterValidator, BaseModel, BeforeValidator, PlainSerializer, ConfigDict, Field, FiniteFloat, TypeAdapter, HttpUrl, SecretStr, SerializationInfo, SerializerFunctionWrapHandler, ValidationError, field_validator, model_serializer, model_validator

from .cost import NodeCostActual, NodeUsage, RunCostActual
from .criteria import CriteriaClause, CriteriaWeight, ModelComparator, format_criteria, format_criteria_weights
from .prompts import PromptExecutionRef, PromptRevision

from .wire import WireModel, WireRequest

ValueType = Literal["text", "number", "boolean", "json", "artifact", "image", "mask", "mesh", "boxes", "video", "audio", "any"]
WorkflowValue = str | int | float | bool | dict[str, object] | list[object]
WorkspaceApiKeyScope = Literal["read", "write", "execute"]


class ValueTypeSpec(WireModel):
    """One port value type: whether its values are stored files, the WIDER types a value of it may
    flow into (`widens`, followed transitively), and the narrower-looking types its ports also take
    (`accepts_from`: an image port takes a text PATH, as the machine-side models read files). The
    ONE lattice both the server's save validation and the canvas read (generated to TS)."""

    name: ValueType
    file: bool = False
    widens: tuple[ValueType, ...] = ()
    accepts_from: tuple[ValueType, ...] = ()
    #: How a tool-calling agent passes a value of this type (`PortSpec.tool_property`): a JSON
    #: scalar of that type, "json" = JSON text in a string argument, decoded on arrival; None = an
    #: agent cannot produce one (a stored file it never saw).
    argument: Literal["string", "number", "boolean", "json"] | None = None


VALUE_TYPES: tuple[ValueTypeSpec, ...] = (
    ValueTypeSpec(name="text", argument="string"),
    ValueTypeSpec(name="number", argument="number"),
    ValueTypeSpec(name="boolean", argument="boolean"),
    ValueTypeSpec(name="json", argument="json"),
    ValueTypeSpec(name="artifact", file=True),
    ValueTypeSpec(name="image", file=True, widens=("artifact",), accepts_from=("text",), argument="string"),
    ValueTypeSpec(name="mask", file=True, widens=("image", "json")),
    #: Structured JSON (label, score, box per item), not a file.
    ValueTypeSpec(name="boxes", widens=("json",), argument="json"),
    ValueTypeSpec(name="mesh", file=True, widens=("artifact",)),
    ValueTypeSpec(name="video", file=True, widens=("artifact",)),
    ValueTypeSpec(name="audio", file=True, widens=("artifact",)),
    #: Takes a value of every type, never reads it as one: the control inputs (`CONTROL_PORTS`).
    ValueTypeSpec(name="any", accepts_from=tuple(name for name in get_args(ValueType) if name != "any"), argument="json"),
)
_VALUE_TYPE_SPECS = {spec.name: spec for spec in VALUE_TYPES}
#: Value types carried as a stored file (an `ArtifactRef` at run time).
FILE_VALUE_TYPES = frozenset(spec.name for spec in VALUE_TYPES if spec.file)


def value_type_widening(name: str) -> tuple[str, ...]:
    """Every type a value of `name` can be read as: itself, then every wider type, transitively."""
    seen = [name]
    for current in seen:
        for wider in (_VALUE_TYPE_SPECS[current].widens if current in _VALUE_TYPE_SPECS else ()):
            if wider not in seen:
                seen.append(wider)
    return tuple(seen)


def value_type_accepts(source: str, target: str) -> bool:
    """A wire from a `source`-typed output may enter a `target`-typed input."""
    return target in value_type_widening(source) or (target in _VALUE_TYPE_SPECS and source in _VALUE_TYPE_SPECS[target].accepts_from)


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
            True if self.value_type == "any" else
            isinstance(item, str) if self.value_type == "text" else
            isinstance(item, bool) if self.value_type == "boolean" else
            isinstance(item, (int, float)) and not isinstance(item, bool) if self.value_type == "number" else
            isinstance(item, ArtifactRef) if self.value_type in FILE_VALUE_TYPES else
            isinstance(item, (dict, list))
            for item in values
        )

    @property
    def argument(self) -> Literal["string", "number", "boolean", "json"] | None:
        """How an agent passes this port's value when it calls the node as a tool; a list port
        always travels as JSON text."""
        kind = _VALUE_TYPE_SPECS[self.value_type].argument
        return "json" if kind is not None and self.multiple else kind

    def tool_property(self) -> "ToolInputProperty | None":
        """This input as one tool argument, None when an agent cannot supply it."""
        kind = self.argument
        if kind is None:
            return None
        shape = f"{self.value_type} list" if self.multiple else self.value_type
        description = f"{self.name.replace('_', ' ')} ({shape}{', as JSON text' if kind == 'json' else ', a file path' if _VALUE_TYPE_SPECS[self.value_type].file else ''})"
        return ToolInputProperty(type="string" if kind == "json" else kind, description=description[:240])

    def from_argument(self, value: object) -> "WorkflowValue":
        """A tool argument as this port's value: JSON text decoded, then checked against the port
        (a file-typed port takes its path as text, `ValueTypeSpec.accepts_from`)."""
        if self.argument == "json":
            try:
                value = json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError:
                raise ValueError(f"{self.name} must be JSON text") from None
        if not (self.accepts(value) or isinstance(value, str) and value_type_accepts("text", self.value_type)):
            raise ValueError(f"{self.name} does not take that value")
        return value


class PortAddress(WireModel):
    node: UUID
    port: str = Field(min_length=1, max_length=80)


class WorkflowKey(WireModel):
    id: UUID


class WorkflowRevisionRef(WireModel):
    key: WorkflowKey
    revision: UUID


class MachineRef(WireModel):
    id: UUID



#: What a model does, named by the Hugging Face Hub `pipeline_tag` ids (huggingface.js
#: `PIPELINE_DATA`) — an existing cross-vendor vocabulary, so a hosted API model and a local
#: checkpoint doing the same job share one task; the PORTS it exposes may still differ by
#: `Placement.target` (`model_task_ports`'s `placement` argument) — a machine-local detector reads
#: image PATHS already on that machine (never uploaded), a hosted one is handed the real file.
ModelTask = Literal[
    "text-generation", "text-to-image", "text-to-video", "object-detection", "image-segmentation",
    "feature-extraction", "text-to-speech", "automatic-speech-recognition", "image-to-3d", "text-to-3d",
    "depth-estimation", "keypoint-detection", "image-to-text",
]

class MachineModelSpec(WireModel):
    """A checkpoint an enrolled machine can run itself (`MachineCommand.model`): its Hugging Face id,
    the task it performs and its licence. The ONE list the server's catalog, the command contract
    and the machine runner read."""

    id: str = Field(min_length=1, max_length=160)
    task: ModelTask
    license: str = Field(min_length=1, max_length=80)


MACHINE_MODELS: dict[str, MachineModelSpec] = {spec.id: spec for spec in (
    MachineModelSpec(id="facebook/detr-resnet-50", task="object-detection", license="Apache-2.0"),
    MachineModelSpec(id="facebook/detr-resnet-50-panoptic", task="image-segmentation", license="Apache-2.0"),
)}
#: The same registry read as model -> task.
VISION_MODEL_TASKS: dict[str, ModelTask] = {model: spec.task for model, spec in MACHINE_MODELS.items()}


def _port(name: str, direction: Literal["input", "output"], value_type: str, required: bool = True, multiple: bool = False) -> "PortSpec":
    return PortSpec(name=name, direction=direction, value_type=value_type, required=required, multiple=multiple)


def model_task_ports(task: ModelTask, placement: Literal["server", "machine"] = "machine") -> tuple["PortSpec", ...]:
    """The port signature a model performing `task` exposes at this `placement`. A MACHINE-placed
    vision task reads image PATHS already on that machine (images never leave it) and returns the
    runner's JSON manifest (labels, scores, boxes or mask files, overlay preview) — unchanged from
    before `placement` existed, so every existing machine-local node keeps its exact shape. A
    SERVER-placed (hosted-API) detector/segmenter is handed the real uploaded image file and
    answers the task's own typed value (`boxes`, a file `mask`) instead of an opaque manifest —
    the two placements genuinely differ in what they may touch, so they earn different ports
    rather than being forced to share one that fits neither well."""
    prompt = _port("prompt", "input", "text")
    reference = _port("image", "input", "image", required=False)
    image_in = _port("image", "input", "image")
    if placement == "machine":
        detection_ports = (_port("images", "input", "text", multiple=True), _port("result", "output", "json"))
        segmentation_ports = (_port("images", "input", "text", multiple=True), _port("result", "output", "json"))
    else:
        detection_ports = (image_in, _port("result", "output", "boxes"))
        segmentation_ports = (image_in, prompt, _port("result", "output", "mask"))
    return {
        "text-generation": (prompt, _port("text", "output", "text")),
        # The generated image is `result`: `image` is already the optional reference INPUT, and a
        # node's port names are unique across both directions (`validate_workflow`).
        "text-to-image": (prompt, reference, _port("result", "output", "image")),
        "text-to-video": (prompt, reference, _port("video", "output", "video")),
        "object-detection": detection_ports,
        "image-segmentation": segmentation_ports,
        "feature-extraction": (_port("text", "input", "text"), _port("embedding", "output", "json")),
        "text-to-speech": (_port("text", "input", "text"), _port("audio", "output", "audio")),
        "automatic-speech-recognition": (_port("audio", "input", "audio"), _port("text", "output", "text")),
        "image-to-3d": (_port("image", "input", "image"), _port("mesh", "output", "mesh")),
        "text-to-3d": (prompt, _port("mesh", "output", "mesh")),
        "depth-estimation": (image_in, _port("result", "output", "image")),
        # An image, never structured JSON: the verified pose APIs this app calls (fal's DWPose)
        # answer a rendered skeleton overlay only, no numeric keypoint coordinates in the response
        # (confirmed against fal's own OpenAPI schema, 2026-09-24) — the port matches what a real
        # provider can actually deliver, not a wished-for shape.
        "keypoint-detection": (image_in, _port("result", "output", "image")),
        "image-to-text": (image_in, _port("text", "output", "text")),
    }[task]


class MachineRuntime(WireModel):
    provider: Literal["claude", "codex"]
    version: str | None = Field(default=None, max_length=80)


AcceleratorKind = Literal["cuda", "mps", "rocm", "none"]


class MachineAccelerator(WireModel):
    """One GPU (or `none`) the runner detected, so model steps default to a machine that has one."""

    kind: AcceleratorKind
    name: str = Field(min_length=1, max_length=120)
    memory_mb: int = Field(ge=0, le=1 << 20)


class MachineResources(WireModel):
    """CPU/RAM/free-disk a runner reports next to its accelerators at hello/heartbeat — the other
    half of a placement fit check (`resources_fit`, below)."""

    cpu_count: int = Field(ge=1, le=256)
    ram_mb: int = Field(ge=1, le=1 << 22)
    disk_free_gb: int = Field(ge=0, le=1 << 16)


class ResourceRequirement(WireModel):
    """What one workflow node needs to run, derived from the model registry / the user's model
    record. Every field optional: unset means "no constraint from this axis", never zero — a node
    with no declared requirement places on any connected machine, today's behaviour unchanged.

    Defined here (not in `.cloud`, which uses it) because `Placement` (below) carries one for an
    AUTO machine placement, and `Placement` predates `.cloud` in the dependency order — `.cloud`
    imports this and `resources_fit` from here rather than the reverse, so the placement contract
    never needs a cross-module forward reference."""

    cpu_count: int | None = Field(default=None, ge=1, le=256)
    ram_mb: int | None = Field(default=None, ge=1, le=1 << 22)
    gpu_kind: AcceleratorKind | None = None
    vram_mb: int | None = Field(default=None, ge=1, le=1 << 20)
    disk_gb: int | None = Field(default=None, ge=1, le=1 << 16)


def resources_fit(requirement: ResourceRequirement, resources: MachineResources | None, accelerators: tuple[MachineAccelerator, ...]) -> bool:
    """Whether a machine reporting `resources`/`accelerators` satisfies `requirement`. A machine
    that never reported `resources` (older runner) only fits a requirement with no CPU/RAM/disk
    axis — never silently assumed to fit an unknown size."""
    if requirement.cpu_count is not None and (resources is None or resources.cpu_count < requirement.cpu_count):
        return False
    if requirement.ram_mb is not None and (resources is None or resources.ram_mb < requirement.ram_mb):
        return False
    if requirement.disk_gb is not None and (resources is None or resources.disk_free_gb < requirement.disk_gb):
        return False
    if requirement.gpu_kind is not None and requirement.gpu_kind != "none":
        matching = [item for item in accelerators if item.kind == requirement.gpu_kind]
        if not matching:
            return False
        if requirement.vram_mb is not None and max(item.memory_mb for item in matching) < requirement.vram_mb:
            return False
    return True


class MachineFunctionSummary(WireModel):
    """One `@galaius.function`-decorated Python callable or registered shell command a machine
    advertises on connect/heartbeat — typed exactly like a workflow node's own ports, so a
    function node (`FunctionImplementation`) copies `ports` verbatim when it is placed. `version` is a content
    hash (name + description + ports) the runner recomputes locally on every call: a node keeps
    running the version it was wired against, and a machine whose function changed shape since
    then refuses the call instead of silently coercing mismatched arguments."""

    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    version: str = Field(pattern=r"^[0-9a-f]{64}$")
    permission: Literal["read_only", "full_access"]
    ports: tuple[PortSpec, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def coherent_ports(self) -> Self:
        if len({port.name for port in self.ports}) != len(self.ports):
            raise ValueError("machine function ports must have unique names")
        if sum(1 for port in self.ports if port.direction == "output") != 1:
            raise ValueError("machine function must declare exactly one output port")
        return self


def plain_file_roots(value: object) -> bool:
    """`value` is a list of machine file roots as a runner reports them: at most 32 relative,
    '/'-separated folder paths, none empty, absolute or climbing ('..')."""
    return isinstance(value, (list, tuple)) and len(value) <= 32 and all(
        isinstance(root, str) and 0 < len(root) <= 240 and not root.startswith("/") and ".." not in root.split("/") for root in value)


MachineProblemCode = Literal["service_unavailable", "service_stopped", "channel_unreachable", "crashed"]


class MachineProblem(WireModel):
    """Why a computer that should be connected is not, as its own galaius saw it: its background
    service could not be set up (`service_unavailable`), was set up but is not running
    (`service_stopped`), runs but cannot open the machine channel (`channel_unreachable`), or its
    connection crashed (`crashed`). Sent over HTTPS with its machine token (`POST
    /v1/machine/problem`), which still answers when the channel does not; the server stamps `at`
    and keeps the latest one until the channel next opens."""

    code: MachineProblemCode
    #: Its own words (an error, a log line), shown under the reason as they are; never parsed.
    detail: str = Field(default="", max_length=500)
    at: datetime | None = None


class MachineSummary(WireModel):
    id: UUID
    name: str = Field(min_length=1, max_length=120)
    state: Literal["online", "offline", "revoked"]
    runtimes: tuple[MachineRuntime, ...] = Field(default=(), max_length=16)
    accelerators: tuple[MachineAccelerator, ...] = Field(default=(), max_length=16)
    functions: tuple[MachineFunctionSummary, ...] = Field(default=(), max_length=64)
    #: CPU/RAM/disk the runner reported at hello/heartbeat; `None` for an older runner that has
    #: never reported it — a placement check treats that like "unknown", never "enough".
    resources: MachineResources | None = None
    #: The folders (relative to the machine's working directory, '/'-separated) its file nodes may
    #: read and write, as its runner reports them — set by the owner ON the machine (`galaius
    #: machine file-roots`) and only the ones its runner accepts. A workflow path is usable iff it
    #: equals one or lies below one. `None`: the runner does not report them (older runner).
    file_roots: tuple[str, ...] | None = Field(default=None, max_length=32)
    last_seen_at: datetime | None = None
    #: Why it is not connected, as it last said (`MachineProblem`); None once its channel opened.
    problem: MachineProblem | None = None

    @field_validator("file_roots")
    @classmethod
    def plain_roots(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and not plain_file_roots(value):
            raise ValueError("a file root is a relative folder path inside the machine's working directory")
        return value

    @model_validator(mode="after")
    def unique_functions(self) -> Self:
        if len({function.name for function in self.functions}) != len(self.functions):
            raise ValueError("machine function names must be unique")
        return self


class MachineCostRate(WireModel):
    """The workspace owner's own price for running a node on this machine: $/GPU-second and
    $/CPU-second, defaulted to 0 ("your hardware" — the machine's own electricity/depreciation
    cost is the owner's business, not this registry's). Set once per machine; every machine-placed
    node's actual cost is its measured wall-clock seconds at this rate."""

    machine: MachineRef
    usd_per_gpu_second: float = Field(default=0.0, ge=0)
    usd_per_cpu_second: float = Field(default=0.0, ge=0)


class MachineCostRateUpdate(WireRequest):
    usd_per_gpu_second: float = Field(default=0.0, ge=0)
    usd_per_cpu_second: float = Field(default=0.0, ge=0)


class MachineCreateRequest(WireRequest):
    name: str = Field(min_length=1, max_length=120)


class MachineCreated(WireModel):
    machine: MachineSummary
    token: SecretStr = Field(min_length=32, max_length=256)


class WorkspaceApiKeyCreate(WireRequest):
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


class ConnectionResourceRef(WireModel):
    id: UUID
    revision: UUID
    capability: Literal["read", "write", "list", "http", "command"]


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


#: What an inbound message hands the workflow it starts, one run per message (the trigger's
#: `input_mapping` sources). Every field is text; `body` is the plain-text part.
MAIL_TRIGGER_FIELDS: tuple[str, ...] = ("from", "to", "subject", "body", "message_id", "thread", "date")


class MailTrigger(WireModel):
    """A new message in a mailbox starts the workflow. Polled every `every_minutes` (none of the
    three providers pushes to an app without a public callback), deduplicated per message: a
    message starts exactly one run however often it is seen. Reads through a `mail_server`
    connection's IMAP side (`connection`, read grant) or a connected Gmail / Microsoft 365 account
    (`account`, its selector). Only messages that arrive after the trigger is switched on count."""

    kind: Literal["mail"]
    provider: Literal["mail_server", "gmail", "microsoft"]
    connection: ConnectionResourceRef | None = None
    account: str | None = Field(default=None, min_length=1, max_length=320)
    folder: str = Field(default="INBOX", min_length=1, max_length=200)
    #: Plain case-insensitive substrings; empty matches everything.
    from_contains: str = Field(default="", max_length=200)
    subject_contains: str = Field(default="", max_length=200)
    every_minutes: int = Field(default=5, ge=1, le=1440)

    @model_validator(mode="after")
    def credential_source(self) -> Self:
        if self.provider == "mail_server":
            if self.connection is None or self.connection.capability != "read" or self.account is not None:
                raise ValueError("an IMAP mail trigger reads through a mail connection's read grant")
        elif self.connection is not None:
            raise ValueError("a Gmail or Microsoft 365 mail trigger reads a connected account, not a saved connection")
        return self

    def next_fire_after(self, instant: datetime) -> datetime:
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("poll evaluation requires a timezone-aware instant")
        return instant + timedelta(minutes=self.every_minutes)


TriggerConfiguration = Annotated[ManualTrigger | ScheduleTrigger | WebhookTrigger | MailTrigger, Field(discriminator="kind")]
#: Triggers the server fires on its own clock (`TriggerScheduleState`): a schedule, a mailbox poll.
POLLED_TRIGGERS = (ScheduleTrigger, MailTrigger)


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


class TriggerEnableUpdate(WireRequest):
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
        if isinstance(self.configuration, MailTrigger) and any(value.source not in MAIL_TRIGGER_FIELDS for value in self.input_mapping):
            raise ValueError(f"a mail trigger maps only {', '.join(MAIL_TRIGGER_FIELDS)}")
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


class TriggerConfigurationUpdate(WireRequest):
    """Compare the displayed configuration before replacing it; runtime state is separate."""

    expected: TriggerBinding
    replacement: TriggerBinding

    @model_validator(mode="after")
    def keep_trigger_kind(self) -> Self:
        if self.expected.configuration.kind != self.replacement.configuration.kind:
            raise ValueError("add a new trigger block to change its kind")
        return self


class TriggerCreate(TriggerBinding, WireRequest):
    enabled: Literal[False] = False


class TriggerDefinition(TriggerBinding):
    id: UUID
    enabled: bool
    created_at: datetime
    schedule: TriggerScheduleState | None = None
    webhook_credential: WebhookCredentialSummary | None = None

    @model_validator(mode="after")
    def coherent_runtime_state(self) -> Self:
        if (self.schedule is not None) != isinstance(self.configuration, POLLED_TRIGGERS):
            raise ValueError("schedule state is required only for schedule and mail triggers")
        if self.webhook_credential is not None and not isinstance(self.configuration, WebhookTrigger):
            raise ValueError("webhook credentials belong only to webhook triggers")
        return self


class TriggerDispatch(WireModel):
    id: UUID
    trigger_id: UUID
    source: Literal["manual", "schedule", "webhook", "mail"]
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
    #: Threat-model #6's hard placement constraint — `None` (default) leaves placement
    #: unconstrained, today's behaviour unchanged. `SovereigntyRequired` is defined further below
    #: in this module (needs `Sovereignty`, defined after this class); Pydantic resolves the
    #: forward reference at class-creation time via `model_rebuild()` at the bottom of this file.
    require_sovereign: "SovereigntyRequired | None" = None


class ConfiguredModelRef(WireModel):
    connection: ConnectionResourceRef
    id: str = Field(min_length=1, max_length=160)


class CredentialRef(WireModel):
    id: UUID


class ConnectionSecretUpdate(WireRequest):
    secret: SecretStr = Field(min_length=1, max_length=16 * 1024)


class AgentRevisionRef(WireModel):
    id: UUID
    revision: UUID


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


ConnectorKind = Literal["google_drive", "github", "slack", "notion", "gmail", "sharepoint", "onedrive", "discord", "telegram", "whatsapp", "gitlab"]
ConnectorActionName = Literal[
    "list_files", "list_repositories", "list_channels", "search_pages", "list_sites",
    "search_metadata", "read_message", "send_message", "list_messages", "read_file",
]
#: Read/list-shaped actions only, reached through the generic `ConnectorAgentTool` with no owner
#: approval — `send_message` and git's write operations run through their own dedicated,
#: approval-gated tool (`SendMessageTool`, `GitAgentTool`), the same split Gmail's read vs send
#: already draws.
ConnectorBrowseActionName = Literal["list_files", "list_repositories", "list_channels", "search_pages", "list_sites", "list_messages", "read_file"]
ConnectorAuthKind = Literal["google_oauth", "microsoft_oauth", "access_token"]
#: How a workflow node's compute is actually reached, right now — never a label. "vendor_api" and
#: "vendor_cli_session" are the two existing `ModelRoute` routes to a vendor-hosted model,
#: generalized onto a node; "self_hosted" is the owner's own machine (a `provider_api` connection
#: whose `provider == "self_hosted"`, or a node placed on a machine whose OWN sourced grading says
#: so) — the third pole the owner named and nothing vendor sees; "unknown" is a node this caller
#: could not classify at all — a connector/subgraph this app has not modelled, a machine placement
#: with no sourced grading, or a criteria-routed agent's undecidable-ahead-of-run route — never
#: silently counted as sovereign. This is the ROUTING axis; the finer per-endpoint JURISDICTION
#: tier (`DataSovereigntyTier`, below) is a separate, CSV-bound question.
Sovereignty = Literal["vendor_api", "vendor_cli_session", "self_hosted", "unknown"]

#: Rank for the workflow-wide weakest-link reduction (`workflow_sovereignty`): higher = less
#: sovereign / less certain. "unknown" ranks BELOW "vendor_api" — a provider nobody has classified
#: yet is treated as worse than one confirmed non-sovereign, never as good-until-proven-otherwise.
_SOVEREIGNTY_RANK: dict[Sovereignty, int] = {"self_hosted": 0, "vendor_cli_session": 1, "vendor_api": 2, "unknown": 3}


#: The EU Cloud Sovereignty Framework v1.2.1 (European Commission DG DIGIT, Oct. 2025) grades a
#: service SEAL-0..SEAL-4 on 8 weighted objectives (legal exposure, data confinement, supply
#: chain...); galaius_core projects that onto 3 buckets for a provider's DEFAULT endpoint/region
#: (`~/.github/research/cloud-compute-and-sovereignty-2026-09-24.md` §1c, sourced 2026-09-24):
#: "eu_sovereign" (EU-HQ, no non-EU parent, data stored AND processed in the EU, SEAL >= 2);
#: "eu_hosted_foreign_law" (EU-region processing available, but the provider or its parent is
#: reachable under non-EU law, e.g. the US CLOUD Act, 18 U.S.C. § 2713); "non_eu" (processing
#: outside the EU, or no region guarantee). PROPOSED classifications, not an official grading.
DataSovereigntyTier = Literal["eu_sovereign", "eu_hosted_foreign_law", "non_eu"]


class ProviderSovereignty(WireModel):
    """One provider's jurisdiction TIER at one connection's actual endpoint — the wire shape a
    CSV-bound registry populates server-side (bound to a sourced machine-readable table, never
    hand-copied into this package: galaius-core stays provider-independent and ships no
    real-world compliance data of its own). `tier` is a property of (provider, ENDPOINT) — Mistral
    on its EU-default endpoint reads `eu_sovereign`, on its opt-in US regional endpoint `non_eu`;
    OpenAI's default endpoint reads `non_eu`, its `eu.api.openai.com` endpoint
    `eu_hosted_foreign_law`. Every field stays `None` (read as unknown) until the sourced registry
    classifies it; `source` then cites it."""

    provider: str = Field(min_length=1, max_length=40)
    tier: DataSovereigntyTier | None = None
    #: HQ country/bloc (ISO 3166-1 alpha-2, or a short bloc code like "EU"), when sourced.
    jurisdiction: str | None = Field(default=None, min_length=2, max_length=8)
    source: str | None = Field(default=None, max_length=200)


def provider_sovereignty(provider: str | None) -> Sovereignty | None:
    """A node's ROUTING figure from the provider it reaches alone — a purely STRUCTURAL fact, no
    compliance data needed: `None` for no provider (a builtin, a bare connector) or `"self_hosted"`
    (the owner's own endpoint, or an enrolled machine — every OTHER caller already resolves those
    before reaching here); any other named provider is a network call to a remote vendor,
    `"vendor_api"`, whether or not that vendor's JURISDICTION has been sourced yet. The finer
    question — which `DataSovereigntyTier` that vendor's ACTUAL endpoint lands in — is a SEPARATE,
    CSV-bound server-side lookup (keyed by provider AND endpoint), never this function's job."""
    if provider is None or provider == "self_hosted":
        return None
    return "vendor_api"


#: Rank for the DATA-TIER weakest-link reduction (`workflow_data_tier`): higher = less sovereign.
_TIER_RANK: dict[DataSovereigntyTier, int] = {"eu_sovereign": 0, "eu_hosted_foreign_law": 1, "non_eu": 2}


def workflow_data_tier(values):
    """The workflow's DATA jurisdiction: the worst `DataSovereigntyTier` among nodes that RECEIVE
    the workflow's actual data (main, relaying the landed `cloud-compute-and-sovereignty-2026-09-24
    .md` research) — deliberately a SEPARATE reduction from `workflow_sovereignty`'s routing figure
    and from a model's WEIGHT-ORIGIN supply-chain question (a Mistral-on-Scaleway pipeline never
    reads non-sovereign only because its weights file came from huggingface.co — the caller feeds
    this function ONLY data-receiving nodes' tiers, a model's origin is a separate flag entirely).
    `None` — never `eu_sovereign` by default — the moment ANY data-receiving node's tier is not
    fully known (an unsourced provider/endpoint, or a node kind this reduction cannot yet
    classify): an unknown hop must never be optimistically assumed sovereign. An empty sequence (no
    data-receiving node at all) reads `"eu_sovereign"`, the identity element."""
    worst: "DataSovereigntyTier | None" = "eu_sovereign"
    for value in values:
        if value is None:
            return None
        if _TIER_RANK[value] > _TIER_RANK[worst]:
            worst = value
    return worst


def workflow_sovereignty(values) -> Sovereignty:
    """The weakest-link (logical AND) reduction shared by BOTH a pre-run PREDICTION (declared
    impl/placement, e.g. `WorkflowRepository.revision_sovereignty`) and a post-run ACTUAL figure
    (`actual_workflow_sovereignty`, below) — one piece of math, never duplicated per caller.
    `None` (a criteria-routed agent, resolved per run; a node this caller could not classify) ranks
    as `"unknown"` — never assumed sovereign because nobody could prove otherwise. A workflow with
    no externally-reaching node at all (every node `self_hosted` or with no provider) is fully
    sovereign: `"self_hosted"` is the identity element, returned for an empty sequence too."""
    worst: Sovereignty = "self_hosted"
    for value in values:
        candidate: Sovereignty = "unknown" if value is None else value
        if _SOVEREIGNTY_RANK[candidate] > _SOVEREIGNTY_RANK[worst]:
            worst = candidate
    return worst


# -- Threat-model #6 (cloud-compute-and-sovereignty-2026-09-24.md): sovereignty from ACTUAL
# execution, never from the requested/declared placement alone, plus a HARD placement constraint --

class NodeSovereigntyRecord(WireModel):
    """The sovereignty ACTUALLY observed for one executed node of one run — sourced from what
    really ran (the cloud provisioning response's region, or which machine/connection really
    served the call), never assumed from the node's static `Placement`. A scheduler that falls
    back to a different region under capacity pressure changes what this record says; it can never
    change a PREDICTED figure computed before the run started. Nodes on an untaken branch never get
    a record — `actual_workflow_sovereignty` reduces over exactly the nodes that ran."""

    node_id: UUID
    sovereignty: Sovereignty
    jurisdiction: str | None = None
    #: Where this record's figure came from — a citation, not free text: a `MachineRef` (joined
    #: through `MachineSovereignty` for a self-hosted machine, or `CloudMachine` for a cloud-
    #: launched one), or a `ConnectionResourceRef` for a hosted-API call. Read by an auditor to
    #: reconstruct WHY a run was graded the way it was, never trusted on the grade alone.
    source: Literal["machine_sovereignty", "cloud_machine", "connection"]
    source_id: UUID


def actual_workflow_sovereignty(records) -> Sovereignty:
    """The workflow's REAL, post-run figure: the same weakest-link reduction as
    `workflow_sovereignty`, over every `NodeSovereigntyRecord` of the nodes that actually executed
    this run. Display this ALONGSIDE (never instead of) `revision_sovereignty`'s pre-run figure,
    the latter always labelled a PREDICTION — the two can legitimately disagree (a fallback
    placement, a criteria-routed agent resolving to a different provider than usual)."""
    return workflow_sovereignty(record.sovereignty for record in records)


class SovereigntyRequired(WireModel):
    """A workflow's HARD placement constraint (threat-model #6): "sovereign required" REFUSES
    dispatch to a node whose PREDICTED figure fails to meet `min_sovereignty`, and REFUSES to
    accept a run whose ACTUAL figure fails it after the fact — never a soft warning, never a
    silent autoscaler fallback to a cheaper non-sovereign region. Declared once per
    `WorkflowInterface`; `meets_requirement` is the ONE place both the pre-dispatch scheduler and
    the post-run auditor check it, so the two can never drift into different rules."""

    min_sovereignty: Sovereignty = "self_hosted"


def meets_requirement(observed: Sovereignty, requirement: SovereigntyRequired | None) -> bool:
    """Whether `observed` satisfies `requirement` — `True` with no requirement declared (today's
    unconstrained default). A LOWER rank is MORE sovereign (`_SOVEREIGNTY_RANK`), so "meets" is
    "at least as sovereign as the floor", never an exact match."""
    return requirement is None or _SOVEREIGNTY_RANK[observed] <= _SOVEREIGNTY_RANK[requirement.min_sovereignty]


class NodeSovereigntyEntry(WireModel):
    """One node's PREDICTED figure, for the workflow canvas's per-node badge — `sovereignty` is
    `None` only where genuinely undecidable ahead of a run (a criteria-routed agent); a builtin
    carries `None` too (it never participates in the workflow-wide reduction,
    `WorkflowSovereigntySummary.sovereignty` below). `tier`/`jurisdiction` are the SAME finer
    jurisdiction figures a Models-area entry carries (visual-critic 2026-09-25 round 1: the node
    badge read "Unknown" while the Models page correctly said "Non-EU (US)" for the identical
    provider, because this entry carried only the routing figure — never the tier — leaving the
    badge with nothing but the fallback branch)."""

    node_id: UUID
    sovereignty: Sovereignty | None
    tier: DataSovereigntyTier | None = None
    jurisdiction: str | None = None


class WorkflowSovereigntySummary(WireModel):
    """The workflow header's PREDICTED figures, read fresh on every open (never cached — a
    referenced agent's model connection, a machine's declared sovereignty, or a provider's sourced
    jurisdiction can all change without the workflow itself changing). `data_tier` is `None` when
    at least one data-receiving node's jurisdiction is not yet resolved (`workflow_data_tier`) —
    today's coverage is model nodes (CSV-bound per actual endpoint) and machine placement; an
    agent or connector node makes this `None` until their own tier resolution is built.
    `jurisdiction` names the WEAKEST node's own country (visual-critic 2026-09-25 round 1: the
    header's tooltip dropped the country the Models page shows for the same model) — `None` when
    `data_tier` itself is `None`, or the weakest node is a machine (no vendor jurisdiction)."""

    sovereignty: Sovereignty
    data_tier: DataSovereigntyTier | None
    jurisdiction: str | None
    require_sovereign: SovereigntyRequired | None
    nodes: tuple[NodeSovereigntyEntry, ...]


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
    connectors: tuple[ConnectorDefinition, ...] = Field(min_length=1, max_length=16)


class ConnectorCheckRequest(WireRequest):
    connector: ConnectorKind
    connection: ConnectionResourceRef | None = None


class ConnectorCheck(WireModel):
    connector: ConnectorKind
    connection: ConnectionResourceRef | None = None
    status: Literal["ready", "unconfigured", "unauthorized", "rate_limited", "unavailable"]
    auth_kind: ConnectorAuthKind
    required_access: tuple[str, ...] = Field(min_length=1, max_length=8)
    credential_expires_at: datetime | None = None


class ConnectorBrowseRequest(WireRequest):
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


#: `list_repositories` stays on the existing read-only `ConnectorAgentTool` /
#: `ConnectorBrowseActionName` path (pure read, no owner approval, fits its query/cursor/limit
#: shape); `read_file` needs a repository + path + ref together, so it lives here beside the four
#: write operations, direct like `SshAgentTool`'s own read/list operations. `repository` is
#: normalized as "owner/repo" for both providers (GitLab accepts that as its URL-encoded project
#: path), so one input shape serves both.
GitConnector = Literal["github", "gitlab"]
GitOperationName = Literal["read_file", "create_branch", "commit_file", "open_pull_request", "add_comment"]


class GitAgentTool(WireModel):
    """A git-hosting operation beyond plain listing, bound to a pinned `service_connector`
    connection. `read_file` runs directly; the other four each change something outside this app
    (a branch, a commit, a pull/merge request, a comment) and are gated by the same owner-approval
    ledger SSH writes use — the same read-direct/write-approved split `SshAgentTool` draws."""

    kind: Literal["git"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    connector: GitConnector
    operation: GitOperationName
    connection: ConnectionResourceRef
    input_schema: ToolInputSchema


MailOperationName = Literal["read_inbox"]


class MailAgentTool(WireModel):
    """One IMAP read against a pinned `mail_server` connection, run directly like Gmail's reads.
    Sending is `SendMessageTool` with `provider="mail_server"`."""

    kind: Literal["mail_server"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    operation: MailOperationName
    connection: ConnectionResourceRef
    input_schema: ToolInputSchema


class WebhookAgentTool(WireModel):
    """Posts to a pinned `webhook` connection's URL — a Discord/Teams incoming webhook or a
    Netlify/Vercel deploy hook. The connection grants exactly one verb ("write"); this tool is
    always owner-approval-gated, since a webhook post is an effect on someone else's system (a
    message sent, a deploy triggered) with no read counterpart to check first."""

    kind: Literal["webhook"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    connection: ConnectionResourceRef
    input_schema: ToolInputSchema


# ---- Sending a message: one shape for every channel ----
#
# An email and a chat post are the same act: a body to a recipient, through a provider the
# sender picks. One tool kind carries it; the provider is a field ON the node (not a block per
# vendor), and every provider reads the same envelope — the fields it cannot carry are refused
# by name at run time, never silently dropped.

#: Where a message goes out. `mail_server` is any SMTP server (its `mail_server` connection's
#: `root`); `gmail` / `microsoft` are connected Google / Microsoft 365 accounts; the rest are
#: bot-token chat APIs on a `service_connector` connection.
MessageProvider = Literal["mail_server", "gmail", "microsoft", "slack", "discord", "telegram", "whatsapp"]
#: The envelope every provider reads. `attachments` is a file port (stored artifacts); the rest
#: are text.
MessageField = Literal["to", "subject", "body", "cc", "in_reply_to", "attachments"]
MESSAGE_FIELDS: tuple[MessageField, ...] = get_args(MessageField)


class MessageProviderSpec(WireModel):
    """What one provider needs and carries: where its credential lives (`auth`: a saved
    connection of `connection_kind`, or a connected OAuth account) and which envelope fields it
    delivers. A chat provider shows `subject` as the message's first line."""

    provider: MessageProvider
    name: str = Field(min_length=1, max_length=80)
    channel: Literal["email", "chat"]
    auth: Literal["connection", "google_oauth", "microsoft_oauth"]
    connection_kind: Literal["mail_server", "service_connector"] | None = None
    fields: tuple[MessageField, ...]
    #: What `to` means for this provider, shown beside the port.
    recipient: str = Field(min_length=1, max_length=120)

    @classmethod
    def of(cls, provider: str) -> "MessageProviderSpec":
        return _MESSAGE_PROVIDER_SPECS[provider]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.auth == "connection") != (self.connection_kind is not None):
            raise ValueError("a connection-backed provider names its connection kind, an OAuth one none")
        if not {"to", "body"} <= set(self.fields):
            raise ValueError("every provider carries a recipient and a body")
        return self


_EMAIL: tuple[MessageField, ...] = MESSAGE_FIELDS
_CHAT: tuple[MessageField, ...] = ("to", "subject", "body")
MESSAGE_PROVIDERS: tuple[MessageProviderSpec, ...] = (
    MessageProviderSpec(provider="mail_server", name="SMTP server", channel="email", auth="connection", connection_kind="mail_server", fields=_EMAIL, recipient="Email address(es), comma-separated"),
    MessageProviderSpec(provider="gmail", name="Gmail", channel="email", auth="google_oauth", fields=_EMAIL, recipient="Email address(es), comma-separated"),
    MessageProviderSpec(provider="microsoft", name="Microsoft 365 (Outlook)", channel="email", auth="microsoft_oauth", fields=_EMAIL, recipient="Email address(es), comma-separated"),
    MessageProviderSpec(provider="slack", name="Slack", channel="chat", auth="connection", connection_kind="service_connector", fields=_CHAT, recipient="Channel id or name"),
    MessageProviderSpec(provider="discord", name="Discord", channel="chat", auth="connection", connection_kind="service_connector", fields=_CHAT, recipient="Channel id"),
    MessageProviderSpec(provider="telegram", name="Telegram", channel="chat", auth="connection", connection_kind="service_connector", fields=_CHAT, recipient="Chat id"),
    MessageProviderSpec(provider="whatsapp", name="WhatsApp", channel="chat", auth="connection", connection_kind="service_connector", fields=_CHAT, recipient="Phone number, E.164"),
)
_MESSAGE_PROVIDER_SPECS = {spec.provider: spec for spec in MESSAGE_PROVIDERS}
_MESSAGE_FIELD_WORDS: dict[str, str] = {"to": "Recipient", "subject": "Subject", "body": "Message text", "cc": "Copy recipients, comma-separated", "in_reply_to": "Message id this answers (threads the reply)"}


class SendMessageTool(WireModel):
    """Sends one message through the provider named on it — an email (SMTP, Gmail, Microsoft
    365) or a chat post (Slack, Discord, Telegram, WhatsApp). A saved connection
    (`connection`) or a connected OAuth account (`account`, its selector) carries the
    credential, per `MessageProviderSpec.auth`. Always an effect on someone else's inbox: the
    server gates it behind the owner-approval ledger by default (`send_mode="approval"`); a
    workflow the owner switched to `"auto"` sends without asking, within `auto_per_hour` sends and
    to `auto_recipients` only when that list is set — past either, the send waits for approval
    again. Every automatic send is recorded in the same ledger (the audit trail)."""

    kind: Literal["send_message"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1, max_length=240)
    provider: MessageProvider
    connection: ConnectionResourceRef | None = None
    account: str | None = Field(default=None, min_length=1, max_length=320)
    send_mode: Literal["approval", "auto"] = "approval"
    #: Automatic sends allowed per rolling hour from this node's principal; beyond it, approval.
    auto_per_hour: int = Field(default=10, ge=1, le=1000)
    #: When set, automatic sends go only to these addresses / chat ids, or `@domain` suffixes.
    auto_recipients: tuple[str, ...] = Field(default=(), max_length=100)
    #: Derived from the envelope (text fields only; a model passes no stored file); filled in when
    #: absent, refused when it disagrees.
    input_schema: ToolInputSchema

    @property
    def spec(self) -> MessageProviderSpec:
        return _MESSAGE_PROVIDER_SPECS[self.provider]

    @classmethod
    def schema_for(cls, provider: MessageProvider) -> ToolInputSchema:
        spec = _MESSAGE_PROVIDER_SPECS[provider]
        words = {**_MESSAGE_FIELD_WORDS, "to": spec.recipient}
        return ToolInputSchema(properties={name: ToolInputProperty(type="string", description=words[name]) for name in spec.fields if name != "attachments"}, required=("to", "body"))

    @model_validator(mode="before")
    @classmethod
    def derived_schema(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("input_schema") is None and data.get("provider") in _MESSAGE_PROVIDER_SPECS:
            return {**data, "input_schema": cls.schema_for(data["provider"])}
        return data

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.input_schema != self.schema_for(self.provider):
            raise ValueError("a send-message tool's input is its provider's envelope")
        if self.connection is not None and self.connection.capability != "write":
            raise ValueError("sending needs the connection's write grant")
        if self.spec.auth != "connection" and self.connection is not None:
            raise ValueError(f"{self.spec.name} sends from a connected account, not a saved connection")
        if self.spec.auth == "connection" and self.account is not None:
            raise ValueError(f"{self.spec.name} sends through a saved connection, not an account")
        return self

    def signature(self) -> tuple[PortSpec, ...]:
        """The node's ports: the WHOLE envelope for every provider, so switching the provider on a
        placed node never re-wires it; `to` and `body` are required (a wire or a constant)."""
        inputs = tuple(PortSpec(name=name, direction="input", value_type="artifact" if name == "attachments" else "text", required=name in {"to", "body"}, multiple=name == "attachments") for name in MESSAGE_FIELDS)
        return (*inputs, PortSpec(name="result", direction="output", value_type="text"))


class MessageAccount(WireModel):
    """One credential a provider can use in this workspace: a saved connection or a connected
    account, and what it may do. The send node's and the mail trigger's pickers list these."""

    provider: MessageProvider
    name: str = Field(min_length=1, max_length=320)
    connection: ConnectionResourceRef | None = None
    account: str | None = Field(default=None, min_length=1, max_length=320)
    can_send: bool
    can_read: bool
    #: Why it cannot do one of them, in the owner's words (a missing grant, a missing credential).
    note: str | None = Field(default=None, max_length=240)


class MessageProviderChoice(WireModel):
    spec: MessageProviderSpec
    accounts: tuple[MessageAccount, ...] = Field(default=(), max_length=256)
    #: Where to add one when `accounts` is empty (the Connections page section).
    connect_hint: str = Field(min_length=1, max_length=240)


class HarnessToolDescriptor(WireModel):
    """One tool an agent's harness can be given (`AgentRevision.harness_tools` names), with the
    category its server declares (galaius's tools publish it in MCP `_meta`), for grouping."""

    name: str = Field(min_length=1, max_length=160)
    category: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=600)


AgentCapability = Annotated[HttpAgentTool | DelegatedAgentTool | GmailAgentTool | ConnectorAgentTool | SshAgentTool | ObjectStorageAgentTool | GitAgentTool | MailAgentTool | WebhookAgentTool | SendMessageTool | WorkflowFunctionTool, Field(discriminator="kind")]


#: A server an agent's MCP binding names: the tools shipped with galaius (`galaius`), a remote
#: server this company added (its UUID), or one a PC declared (`pc:<machine UUID>:<name>`).
MCP_SERVER_ID = r"^(interact|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|pc:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:[A-Za-z0-9_.-]{1,64})$"
#: An MCP tool's name, as the protocol allows it.
MCP_TOOL_NAME = r"^[A-Za-z0-9_.-]{1,128}$"
McpToolName = Annotated[str, Field(pattern=MCP_TOOL_NAME)]


class McpBinding(WireModel):
    """One MCP server an agent is connected to: which of its tools the agent may call (`all`: every
    tool the server lists, new ones included) and, per model-using tool, the model it asks for
    (overriding the account-wide rule for that tool). A tool left out is denied to the agent's
    harness, never merely hidden."""

    server_id: str = Field(pattern=MCP_SERVER_ID)
    enabled_tools: Literal["all"] | tuple[McpToolName, ...] = "all"
    tool_models: dict[McpToolName, "ModelChoice"] = Field(default_factory=dict, max_length=64)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.enabled_tools != "all":
            if len(set(self.enabled_tools)) != len(self.enabled_tools) or len(self.enabled_tools) > 512:
                raise ValueError("an MCP binding names each tool once, at most 512")
            if set(self.tool_models) - set(self.enabled_tools):
                raise ValueError("a tool model rule names a tool this binding enables")
        return self

    def enables(self, tool: str) -> bool:
        return self.enabled_tools == "all" or tool in self.enabled_tools


class AgentRevision(WireModel):
    id: UUID
    revision: UUID
    parent_revision: UUID | None = None
    name: str = Field(min_length=1, max_length=120)
    name_fr: str = Field(default="", max_length=120, pattern=r"^[^\r\n]*$")
    role_key: str | None = Field(default=None, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=120)
    description: str = Field(default="", max_length=8192)
    summary: str = Field(default="", max_length=120, pattern=r"^[^\r\n]*$")
    summary_fr: str = Field(default="", max_length=120, pattern=r"^[^\r\n]*$")
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
    #: The MCP servers this agent is connected to, each with its tool switches (`McpBinding`).
    mcp: tuple[McpBinding, ...] = Field(default=(), max_length=32)
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
        if len({binding.server_id for binding in self.mcp}) != len(self.mcp):
            raise ValueError("an agent binds each MCP server once")
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


class AgentGraphUpdate(WireRequest):
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

    @model_serializer(mode="wrap")
    def sealed_content(self, handler: SerializerFunctionWrapHandler, info: SerializationInfo):
        """Every dump, nested in any response too, is the default-free content the cursor seals: a
        reader whose release lacks a defaulted field would keep it as an extra and hash it."""
        if info.exclude_defaults:
            return handler(self)
        return self.model_dump(mode=info.mode, include=info.include, exclude=info.exclude, context=info.context, by_alias=info.by_alias,
                               exclude_unset=info.exclude_unset, exclude_none=info.exclude_none, exclude_computed_fields=info.exclude_computed_fields,
                               round_trip=info.round_trip, serialize_as_any=info.serialize_as_any, polymorphic_serialization=info.polymorphic_serialization,
                               exclude_defaults=True)

    @classmethod
    def create(cls, agents: tuple[AgentRevision, ...], paradigms: tuple[PromptRevision, ...],
               root_agent: AgentRevisionRef | None = None,
               prompt_heads: tuple[PromptExecutionRef, ...] = ()):
        agents = tuple(sorted(agents, key=lambda item: str(item.id)))
        paradigms = tuple(sorted(paradigms, key=lambda item: (item.key.namespace, item.key.slug, item.digest)))
        prompt_heads = tuple(sorted(prompt_heads, key=lambda item: (item.key.namespace, item.key.slug)))
        content = {"agents": agents, "paradigms": paradigms, "root_agent": root_agent, "prompt_heads": prompt_heads}
        sealed = cls.model_construct(**content, cursor="").model_dump(mode="json", exclude_defaults=True)
        return cls(**content, cursor=cls.content_cursor(sealed))

    @staticmethod
    def content_cursor(content: dict[str, Any]) -> str:
        """The cursor sealing a JSON dump: canonical JSON of everything but the cursor, an absent
        root and empty prompt heads left out. What is sent is what is sealed (`canonical_content`);
        a projection for an older reader reseals what it sends (`projected`)."""
        payload = {key: value for key, value in content.items() if key != "cursor"
                   and not (key == "root_agent" and value is None) and not (key == "prompt_heads" and not value)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()

    @classmethod
    def projected(cls, data: dict[str, Any]) -> dict[str, Any]:
        return {**data, "cursor": cls.content_cursor(data)}

    @model_validator(mode="after")
    def coherent_snapshot(self) -> Self:
        if self.content_cursor(self.model_dump(mode="json", exclude_defaults=True)) != self.cursor:
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

    @classmethod
    def found_in(cls, value: object) -> tuple["ArtifactRef", ...]:
        """Every file reference anywhere in a JSON value (inside lists and objects too); raises
        ValueError on a reference-shaped object that is not a valid one."""
        if isinstance(value, list):
            return tuple(found for item in value for found in cls.found_in(item))
        if isinstance(value, dict):
            if set(value) == set(cls.model_fields):
                return (cls.model_validate(value),)
            return tuple(found for item in value.values() for found in cls.found_in(item))
        return ()


class Placement(WireModel):
    """Where a node runs: on the server (hosted APIs), or on one enrolled machine (local
    checkpoints, the owner's own GPU — data stays on it). A machine placement is either PINNED
    (`machine` set: the owner or the graph editor named an exact machine) or AUTO (`requirement`
    set, `machine` unset: the run-time placement scheduler — `MachineChannel.dispatch`, server-
    side — resolves the cheapest fit at dispatch time instead: the workspace's own online
    hardware first, then a pooled peer machine, then an on-demand cloud launch, or a named refusal
    when nothing fits). Never both, never neither, for `target == "machine"`."""

    target: Literal["server", "machine"] = "server"
    machine: MachineRef | None = None
    requirement: ResourceRequirement | None = None

    @model_validator(mode="after")
    def machine_named(self) -> Self:
        if self.target != "machine":
            if self.machine is not None or self.requirement is not None:
                raise ValueError("a server placement names no machine and no resource requirement")
            return self
        if (self.machine is not None) == (self.requirement is not None):
            raise ValueError("a machine placement names its machine (pinned) or its resource requirement (auto), and only one")
        return self


DirectTool = Annotated[HttpAgentTool | GmailAgentTool | ConnectorAgentTool | SshAgentTool | ObjectStorageAgentTool | GitAgentTool | MailAgentTool | WebhookAgentTool | SendMessageTool, Field(discriminator="kind")]


class NodeLibraryRef(WireModel):
    id: UUID


# ---- The workflow node (docs: .github/memory/generic-node-contract.md) ----
#
# Every node is ONE shape: what it runs (`impl`, data only), its `ports`, its `config` (settings
# AND the constants of unwired input ports) and its `placement`. The server and the machine runner
# dispatch on `impl.kind`; an implementation declares its effects, where it may run, the ports it
# fixes by itself and the rules binding it to its config.

#: What a node reaches outside the workflow: a model provider, a connector / API, or an enrolled
#: machine. A pure function workflow has none, at any depth.
Effect = Literal["model", "connector", "machine"]


class _Implementation(WireModel):
    #: What running it reaches; `placement` on a machine adds "machine" (WorkflowNode.effects).
    effects: ClassVar[frozenset[str]] = frozenset()
    #: Where it may run (`Placement.target`).
    placements: ClassVar[frozenset[str]] = frozenset({"server"})
    #: Where the palette files it (`WorkflowBlockAvailability.category`).
    category: ClassVar["BlockCategory"]
    #: Set when it is plumbing (`Plumbing`); every non-builtin kind is a step: it does the work itself.
    plumbing: ClassVar["Plumbing | None"] = None

    def signature(self, placement: Literal["server", "machine"] = "machine", config: dict[str, "WorkflowValue"] | None = None) -> tuple[PortSpec, ...] | None:
        """The ports this implementation fixes by itself, or None when they come from outside it
        (an input's chosen value type, a machine's advertised function, a tool's schema, a
        subgraph's interface, an agent). `placement` matters only to `ModelImplementation` (a
        hosted vs machine-local model of the same task can expose different ports), `config` only
        to a builtin whose ports follow its settings (a switch's cases)."""
        return None

    def check_ports(self, ports: tuple[PortSpec, ...]) -> None:
        """Rules binding this implementation to ports it does not fix itself; raises ValueError."""

    def described(self) -> dict[str, object]:
        """What a palette block of it says by default: its category (a builtin adds its summary,
        search words and settings schema)."""
        return {"category": self.category, "plumbing": self.plumbing}

    def required_config(self) -> tuple[str, ...]:
        """Config keys a node needs before it can run (the palette's `required_config_fields`)."""
        return ()

    def check(self, config: dict[str, WorkflowValue]) -> None:
        """Rules binding this implementation to its config; raises ValueError."""

    def check_placement(self, placement: "Placement") -> None:
        """Rules binding this implementation to where it is placed, beyond `placements`; raises ValueError."""

    #: Control ports this kind adds to `CONTROL_PORTS` (an agent's `tools`).
    control_ports: ClassVar[tuple[PortSpec, ...]] = ()


BuiltinOp = Literal[
    "input", "output", "write_artifact", "read_file", "uppercase", "lowercase", "identity", "http_get",
    "condition", "switch", "merge", "wait", "approval", "fail",
    "set_fields", "transform", "filter", "sort", "limit", "dedupe", "aggregate", "sql", "parse", "format", "calculate", "encode",
    "template", "replace_text", "extract_text", "split_text", "date_time",
]
#: Where a palette block sits and what a search for a kind of step finds (`WorkflowBlockAvailability.category`).
BlockCategory = Literal["input_output", "flow", "data", "text", "files", "time", "people", "web", "agents", "models", "apps", "machines", "code", "workflows"]
class PlumbingCondition(WireModel):
    """Plumbing only while setting `field` matches `pattern`: a text template holding nothing but
    `{{fields}}` passes them on. The editor runs it (JavaScript): keep to syntax both engines read alike."""

    field: str = Field(min_length=1, max_length=64)
    pattern: str = Field(min_length=1, max_length=200)


class Plumbing(WireModel):
    """What makes a node kind PLUMBING rather than a step: it only carries a value to the next step
    in the form that step needs — its format converted (text <-> data), fields picked, set or
    renamed, values merged or split, or passed on as is — deciding nothing and writing nothing the
    reader would call content (a step changes the value itself: its case, its words, which items
    are kept). The editor folds plumbing into the wire or step it serves, saying `badge` there; the
    workflow and its runs are unchanged by it.

    `badge` / `badge_fr` are templates over the node's settings: `{name}` is setting `name` (an
    object gives its keys, a list its items); " · " parts whose setting is empty are left out."""

    badge: str = Field(min_length=1, max_length=80)
    badge_fr: str = Field(min_length=1, max_length=80)
    only_if: PlumbingCondition | None = None

    def fields(self) -> set[str]:
        """The settings its words and condition read."""
        return set(re.findall(r"\{(\w+)\}", self.badge + self.badge_fr)) | ({self.only_if.field} if self.only_if else set())


class NoSettings(BaseModel):
    """An op with no settings of its own (its config holds only port constants, a connection, a path)."""

    model_config = {"extra": "ignore"}


class BuiltinOpSpec:
    """One builtin operation's contract, declared as data (`galaius_core.builtin_ops`): what it
    is for, its settings model and the ports it fixes. Subclassing with an `op` registers it in
    `BUILTIN_OPS`; the server keys its runners by the same name. A node's `config` holds these
    settings AND the constants of its unwired input ports (checked by the port, not here)."""

    op: ClassVar[str]
    category: ClassVar[BlockCategory]
    title: ClassVar[str]
    #: The same name in French (the editor shows the reader's language).
    title_fr: ClassVar[str] = ""
    #: What it does, in one sentence a search reads.
    summary: ClassVar[str]
    #: The same sentence in French (the editor shows the reader's language).
    summary_fr: ClassVar[str] = ""
    keywords: ClassVar[tuple[str, ...]] = ()
    #: Set when it is plumbing, not a step (`Plumbing`): what the editor says where it folds it.
    plumbing: ClassVar[Plumbing | None] = None
    Config: ClassVar[type[BaseModel]] = NoSettings
    #: Config keys a node needs before it can run.
    required: ClassVar[tuple[str, ...]] = ()
    placements: ClassVar[frozenset[str]] = frozenset({"server"})
    #: False: the editor chooses the ports (`ports` is only the palette's starting shape).
    fixed_ports: ClassVar[bool] = True

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if "op" in cls.__dict__:
            if cls.op in BUILTIN_OPS:
                raise TypeError(f"builtin op {cls.op} is declared twice")
            unknown = cls.plumbing.fields() - set(cls.Config.model_fields) if cls.plumbing else set()
            if unknown:
                raise TypeError(f"builtin op {cls.op}: its plumbing reads settings it does not have: {sorted(unknown)}")
            BUILTIN_OPS[cls.op] = cls

    @classmethod
    def ports(cls, config: dict[str, "WorkflowValue"]) -> tuple[PortSpec, ...]:
        raise NotImplementedError

    @classmethod
    def defaults(cls) -> dict[str, "WorkflowValue"]:
        """Every setting's default: the config a new node starts with."""
        return {name: field.get_default(call_default_factory=True) for name, field in cls.Config.model_fields.items() if not field.is_required()} if cls.fixed_ports else {}

    @classmethod
    def config_schema(cls) -> dict[str, object]:
        """The settings as JSON Schema — what the node editor's one generic form renders."""
        return cls.Config.model_json_schema() if cls.fixed_ports else {}

    @classmethod
    def settings(cls, config: dict[str, "WorkflowValue"]) -> BaseModel:
        """The node's settings, validated: its config without the constants of its input ports."""
        inputs = {port.name for port in cls.ports(config) if port.direction == "input"}
        try:
            return cls.Config.model_validate({name: value for name, value in config.items() if name not in inputs})
        except ValidationError as error:
            raise ValueError("; ".join(f"{'.'.join(str(part) for part in item['loc']) or cls.op}: {item['msg']}" for item in error.errors())) from None


#: Every builtin op's spec, by op (filled by `galaius_core.builtin_ops`, imported with the package).
BUILTIN_OPS: dict[str, type[BuiltinOpSpec]] = {}


class BuiltinImplementation(_Implementation):
    """Runs in the server itself (`BUILTIN_OPS[op]` says what it does and its ports) — or a FILE op
    (`FILE_OPS`), which runs where the file is: on the server against workspace storage
    (`config.connection`), or on an enrolled machine against its working directory
    (`artifact_path` relative to it, within the machine's permission ceiling)."""

    kind: Literal["builtin"]
    op: BuiltinOp
    #: Ops that read or write one file, wherever the node is placed.
    FILE_OPS: ClassVar[frozenset[str]] = frozenset({"write_artifact", "read_file"})

    @property
    def spec(self) -> type[BuiltinOpSpec]:
        return BUILTIN_OPS[self.op]

    @property
    def placements(self) -> frozenset[str]:
        return self.spec.placements

    @property
    def category(self) -> BlockCategory:
        return self.spec.category

    def described(self) -> dict[str, object]:
        spec = self.spec
        return {"category": spec.category, "plumbing": spec.plumbing, "summary": spec.summary, "summary_fr": spec.summary_fr, "name_fr": spec.title_fr, "keywords": spec.keywords, "config_schema": spec.config_schema()}

    def signature(self, placement: Literal["server", "machine"] = "machine", config: dict[str, "WorkflowValue"] | None = None) -> tuple[PortSpec, ...] | None:
        return self.spec.ports(config or {}) if self.spec.fixed_ports else None

    def required_config(self) -> tuple[str, ...]:
        return self.spec.required

    def check(self, config: dict[str, WorkflowValue]) -> None:
        if self.spec.fixed_ports:
            self.spec.settings(config)
            return
        if self.op == "input" and "value" not in config:
            raise ValueError("a workflow input holds its value")
        if config.get("connection") is not None:
            ConnectionResourceRef.model_validate(config["connection"])
        path = config.get("artifact_path")
        if path is not None and not (isinstance(path, str) and 1 <= len(path) <= 512):
            raise ValueError("artifact path must be a relative path of 1-512 characters")


class AgentImplementation(_Implementation):
    """An agent revision. Every node wired into its `tools` port is a tool it may call
    (`WorkflowNode.tool_schema`); the call runs that node like any other step."""

    kind: Literal["agent"]
    category: ClassVar[BlockCategory] = "agents"
    agent: AgentRevisionRef
    effects: ClassVar[frozenset[str]] = frozenset({"model"})
    placements: ClassVar[frozenset[str]] = frozenset({"server", "machine"})
    control_ports: ClassVar[tuple[PortSpec, ...]] = (PortSpec(name="tools", direction="input", value_type="any", required=False, multiple=True),)


class ModelChoice(WireModel):
    """A model asked for by what it must DO and how it must SCORE, instead of by id — re-resolved
    every run. Written in the ONE criteria grammar agents already use (`galaius_core.criteria`):
    `constraints` are hard clauses (a model failing one, or with no known value for it, is out),
    `rank_by` orders the survivors by weighted normalised benchmark. An empty `rank_by` means the
    task's own primary benchmark(s), so the default follows the benchmark registry, not the day
    the node was saved."""

    task: ModelTask
    rank_by: tuple[CriteriaWeight, ...] = Field(default=(), max_length=16)
    constraints: tuple[CriteriaClause, ...] = Field(default=(), max_length=32)
    #: True: the node runs the model it names, whatever today's ranking says — the benchmarks and
    #: constraints are kept for when the person chooses by them again.
    pinned: bool = False

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if len({weight.name for weight in self.rank_by}) != len(self.rank_by):
            raise ValueError("each benchmark ranks once")
        if any(clause.kind == "provider" for clause in self.constraints):
            raise ValueError("a model choice constrains properties, not a CLI provider")
        return self

    @property
    def criteria(self) -> str:
        """The constraints as an agent would store them."""
        return format_criteria(self.constraints)

    @property
    def weights(self) -> str:
        return format_criteria_weights(self.rank_by)


class ModelImplementation(_Implementation):
    """Any model typed by its task (`ModelTask`, Hugging Face pipeline tags). With an unpinned
    `choice`, `provider`/`model` are the model that choice resolved to when last edited (what the
    editor shows and prices); every run re-resolves the choice and records its answer in the run."""

    kind: Literal["model"]
    category: ClassVar[BlockCategory] = "models"
    provider: str = Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9_]*$")
    model: str = Field(min_length=1, max_length=160)
    task: ModelTask
    choice: ModelChoice | None = None
    effects: ClassVar[frozenset[str]] = frozenset({"model"})
    placements: ClassVar[frozenset[str]] = frozenset({"server", "machine"})

    @model_validator(mode="after")
    def choice_matches_task(self) -> Self:
        if self.choice is not None and self.choice.task != self.task:
            raise ValueError("a model choice is for the node's own task")
        return self

    # No return annotation: an annotated one would replace this model's own schema with a bare
    # object in every serialization-mode schema (the generated TypeScript contracts among them).
    @model_serializer(mode="wrap")
    def _no_empty_choice(self, handler: SerializerFunctionWrapHandler):
        """No choice, no key: a resolved node sent to a machine serializes exactly as a runner built
        before `choice` existed reads and signs it (its model forbids unknown keys)."""
        data = handler(self)
        if self.choice is None:
            data.pop("choice", None)
        return data

    def signature(self, placement: Literal["server", "machine"] = "machine", config: dict[str, "WorkflowValue"] | None = None) -> tuple[PortSpec, ...]:
        return model_task_ports(self.task, placement)


class FunctionImplementation(_Implementation):
    """A function an enrolled machine declared (`MachineFunctionSummary`), pinned to the exact
    signature version it was wired against: a later change on the machine refuses the call
    instead of silently running under a different shape."""

    kind: Literal["function"]
    category: ClassVar[BlockCategory] = "machines"
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")
    version: str = Field(pattern=r"^[0-9a-f]{64}$")
    effects: ClassVar[frozenset[str]] = frozenset({"machine"})
    placements: ClassVar[frozenset[str]] = frozenset({"machine"})


class ScriptFile(WireRequest):
    """A script that already lives on its machine (`ScriptImplementation.origin == "machine_file"`):
    its path inside the machine owner's script folders (`galaius machine script-roots`, never
    where workflow file steps write), the sha256 of the file's bytes when it was picked, and how it
    is started. Paths are relative to the machine's working directory; the machine re-checks the
    folders and the file's digest before every run."""

    path: str = Field(min_length=1, max_length=1024)
    file_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    args: tuple[Annotated[str, Field(max_length=4096)], ...] = Field(default=(), max_length=32)
    #: The folder it starts in; None: the script's own folder.
    cwd: str | None = Field(default=None, min_length=1, max_length=1024)
    #: The program that runs it (a command name or an absolute path); None: the language's own
    #: program on that machine (`ScriptLanguage`; a Python script declaring packages, PEP 723, runs through uv).
    interpreter: str | None = Field(default=None, min_length=1, max_length=256, pattern=r"^[^\s\x00]+$")

    @field_validator("path", "cwd")
    @classmethod
    def machine_relative(cls, value: str | None) -> str | None:
        if value is not None and (value.startswith("/") or "\\" in value or ":" in value or "\x00" in value
                                  or any(part in {"", ".", ".."} or part.startswith(".") for part in value.split("/"))):
            raise ValueError("a script path is a plain relative path inside the machine's script folders (no hidden names, no '..')")
        return value

    @field_validator("args")
    @classmethod
    def plain_args(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any("\x00" in arg for arg in value):
            raise ValueError("script arguments cannot hold NUL characters")
        return value

    #: Starts every hashed invocation: its NUL keeps file digests apart from inline source digests
    #: (inline source never holds NUL), so approving some inline text never approves a file run.
    DIGEST_TAG: ClassVar[str] = "interact.script.machine_file.v1\x00"

    def invocation_digest(self, language: str) -> str:
        """What a machine owner approves: the file's content AND how it is started (path, folder,
        arguments, interpreter), after `DIGEST_TAG`. Canonical JSON (sorted keys, no spaces, UTF-8)
        — the editor computes the same bytes (frontend graph/scriptSource.ts `scriptFileDigest`)."""
        payload = {"language": language, **self.model_dump(mode="json")}
        return hashlib.sha256((self.DIGEST_TAG + json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)).encode()).hexdigest()


#: What a Script step is written in; each fixes the program that runs it on the machine: Python (the
#: runner's own, or uv for declared packages), shell (`/bin/sh`: Linux / macOS), PowerShell (`pwsh`,
#: else Windows PowerShell), cmd (Windows only). A machine says which it runs (`script:<language>`
#: among its features); a step never lands on one that does not.
ScriptLanguage = Literal["python", "shell", "powershell", "cmd"]


class ScriptImplementation(_Implementation):
    """Python, shell, PowerShell or cmd code run on one enrolled machine — a code-execution boundary: the machine
    owner approves the step's exact `approval_digest` before the server dispatches it
    (`MachineStore.script_approved`), on top of the per-command signature. `origin` says where the
    code comes from: "inline", source written in the editor (`config["source"]`); "machine_file", a
    script already on the machine (`config` is a `ScriptFile`). The digest covers what runs AND what
    runs it: inline code with its language (the language fixes the program, `ScriptLanguage`), a file with its content, path, arguments, folder and
    program. Any edit — of the code, its language, the file, how it starts — changes the digest and
    re-arms that approval."""

    kind: Literal["script"]
    category: ClassVar[BlockCategory] = "code"
    language: ScriptLanguage
    #: Inline code: the sha256 of its source, as every released runner checks it. A machine file:
    #: its `invocation_digest`. What an owner approves is `approval_digest`, never this pin.
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: Written only when not "inline": every Script saved before this field existed, and every
    #: runner or client released before it (extra fields forbidden), reads inline code unchanged.
    origin: Literal["inline", "machine_file"] = Field(default="inline", exclude_if=lambda origin: origin == "inline")
    effects: ClassVar[frozenset[str]] = frozenset({"machine"})
    placements: ClassVar[frozenset[str]] = frozenset({"machine"})
    #: Starts every hashed inline payload: its NUL keeps inline digests apart from the bare sha256
    #: of any source (source never holds NUL) and from `ScriptFile.DIGEST_TAG` digests.
    INLINE_DIGEST_TAG: ClassVar[str] = "interact.script.inline.v1\x00"

    @classmethod
    def inline_digest(cls, language: str, source: str) -> str:
        """What a machine owner approves for inline code: its language and its exact source, after
        `INLINE_DIGEST_TAG`, as canonical JSON (sorted keys, no spaces, UTF-8) — the editor computes
        the same bytes (frontend graph/scriptSource.ts `inlineDigest`)."""
        payload = json.dumps({"language": language, "source": source}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256((cls.INLINE_DIGEST_TAG + payload).encode()).hexdigest()

    @classmethod
    def inline(cls, language: ScriptLanguage, source: str) -> "ScriptImplementation":
        """An inline Script step pinned to `source` in `language`."""
        return cls(kind="script", language=language, source_digest=hashlib.sha256(source.encode()).hexdigest())

    def required_config(self) -> tuple[str, ...]:
        return ("source",) if self.origin == "inline" else ("path", "file_digest")

    def script_file(self, config: dict[str, WorkflowValue]) -> ScriptFile:
        """The machine file a "machine_file" script runs, from its config; raises ValueError."""
        if self.origin != "machine_file":
            raise ValueError("this script is written inline, not a file on the machine")
        try:
            return ScriptFile.model_validate(config)
        except ValidationError as error:
            raise ValueError(f"script file settings are invalid: {error.errors()[0]['msg']}") from error

    def approval_digest(self, config: dict[str, WorkflowValue]) -> str:
        """The digest a machine owner must have approved for this step to run, computed from what
        runs (never trusted from the stored `source_digest`)."""
        if self.origin == "machine_file":
            return self.script_file(config).invocation_digest(self.language)
        return self.inline_digest(self.language, self._source(config))

    def check_placement(self, placement: "Placement") -> None:
        if self.origin == "machine_file" and placement.machine is None:
            raise ValueError("a script file lives on one machine: choose that machine")

    @staticmethod
    def _source(config: dict[str, WorkflowValue]) -> str:
        source = config.get("source")
        if not isinstance(source, str) or len(source) > 1 << 16 or "\x00" in source:
            raise ValueError("script node source must be text of at most 64 KiB without NUL characters")
        return source

    def check(self, config: dict[str, WorkflowValue]) -> None:
        if self.origin == "machine_file":
            if self.script_file(config).invocation_digest(self.language) != self.source_digest:
                raise ValueError("script file settings do not match their pinned digest")
            return
        source = self._source(config)
        if self.source_digest != hashlib.sha256(source.encode()).hexdigest():
            raise ValueError("script node source does not match its pinned digest")


class ConnectorImplementation(_Implementation):
    """A configured connector or API operation (`DirectTool`), executable without a model call.
    Its config holds scalar arguments only (the tool's declared input schema)."""

    kind: Literal["connector"]
    category: ClassVar[BlockCategory] = "apps"
    tool: DirectTool
    effects: ClassVar[frozenset[str]] = frozenset({"connector"})

    def signature(self, placement: Literal["server", "machine"] = "machine", config: dict[str, "WorkflowValue"] | None = None) -> tuple[PortSpec, ...] | None:
        """A send-message node fixes its ports (the envelope); any other tool's come from its
        declared input schema, chosen where the block is built."""
        return self.tool.signature() if isinstance(self.tool, SendMessageTool) else None

    def check(self, config: dict[str, WorkflowValue]) -> None:
        if any(not isinstance(value, (str, int, float, bool)) for value in config.values()):
            raise ValueError("connector arguments are scalar values")


class SubgraphImplementation(_Implementation):
    """Another graph run as one node: a saved workflow pinned to a revision, or a reusable node from
    the workspace's library (its current definition). Its effects are its contents' (resolved by
    whoever can read them: the server expands it)."""

    kind: Literal["subgraph"]
    category: ClassVar[BlockCategory] = "workflows"
    ref: WorkflowRevisionRef | NodeLibraryRef
    #: A MAP: the saved workflow runs once per item of this input — ONE list of any items (a JSON
    #: list, a text list), each item checked against the workflow's own input when it runs — in
    #: order, one at a time, every other input the same each time; each output is the list of its
    #: per-item values.
    each: str | None = Field(default=None, min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def maps_a_workflow(self) -> Self:
        if self.each is not None and isinstance(self.ref, NodeLibraryRef):
            raise ValueError("a reusable node runs once where it is placed; map a saved workflow instead")
        return self

    def check_ports(self, ports: tuple[PortSpec, ...]) -> None:
        if self.each is None:
            return
        if not any(port.direction == "input" and port.name == self.each and port.value_type == "any" and not port.multiple for port in ports):
            raise ValueError(f"a map runs once per item of its list input {self.each} (one value of type any)")
        if any(port.direction == "output" and not port.multiple for port in ports):
            raise ValueError("a map answers one list per output")


Implementation = Annotated[BuiltinImplementation | AgentImplementation | ModelImplementation | FunctionImplementation | ScriptImplementation | ConnectorImplementation | SubgraphImplementation, Field(discriminator="kind")]
_IMPLEMENTATIONS: TypeAdapter[Implementation] = TypeAdapter(Implementation)
#: What an enrolled machine runs itself (`MachineCommand.impl`).
MachineImplementation = Annotated[AgentImplementation | ModelImplementation | FunctionImplementation | ScriptImplementation | BuiltinImplementation, Field(discriminator="kind")]


#: Ports every node has without declaring them (never stored in `WorkflowNode.ports`; their names
#: are reserved). `when`: wires into it gate the node — it runs once they deliver, their values are
#: never read — so a fallback node waits on a failure. `error`: the node's failure as a
#: `NodeError`, carried only under `NodePolicy.on_error == "route"`. An agent adds `tools`.
CONTROL_PORTS: tuple[PortSpec, ...] = (
    PortSpec(name="when", direction="input", value_type="any", required=False, multiple=True),
    PortSpec(name="error", direction="output", value_type="json", required=False),
)
#: A wire into one of these delivers no value to the runner.
CONTROL_INPUTS = frozenset({"when", "tools"})

JoinRule = Literal["all", "any"]
OnError = Literal["fail_run", "route", "continue_with_default"]


class NodePolicy(WireModel):
    """How any node runs and fails, whatever it runs. A failed attempt is retried `retries` times,
    the n-th retry after `backoff_seconds * 2**(n-1)`; `timeout_seconds` bounds each attempt. Once
    every attempt failed, `on_error` decides: `fail_run` stops the run (the default); `route` sends
    a `NodeError` out of the `error` port and skips everything the node's other outputs feed;
    `continue_with_default` emits `defaults` (one per output port) as if the node had succeeded.
    `join` is the fan-in rule of each input port fed by several wires: `all` (the default) waits
    for every wire and is skipped when one of them was, `any` takes the first value to arrive (a
    list port: every value that arrived) and is skipped only when every wire was."""

    retries: int = Field(default=0, ge=0, le=10)
    backoff_seconds: float = Field(default=1.0, ge=0, le=600)
    timeout_seconds: float | None = Field(default=None, gt=0, le=86400)
    on_error: OnError = "fail_run"
    defaults: dict[str, WorkflowValue] = Field(default_factory=dict, max_length=64)
    join: dict[str, JoinRule] = Field(default_factory=dict, max_length=64)

    def delay(self, retry: int) -> float:
        """Seconds before the `retry`-th retry (1-based)."""
        return self.backoff_seconds * 2 ** (retry - 1)


class NodeError(WireModel):
    """One node's failure once its retries are spent: the `error` port's value, and what a run
    records about each error it survived (`WorkflowRun.recovered`)."""

    node: UUID
    label: str = Field(min_length=1, max_length=120)
    kind: str = Field(min_length=1, max_length=40)
    code: Literal["failed", "timeout"]
    message: str = Field(max_length=400)
    #: The attempt that failed last (1 = no retry happened).
    attempt: int = Field(ge=1)


class WorkflowNode(WireModel):
    """One node, whatever it runs. `config` holds the implementation's settings AND the constant
    value of any input port left unwired (a wire, when present, wins). `policy` says how it
    retries and where its errors go; `description` tells an agent calling it as a tool what it does."""

    id: UUID
    label: str = Field(min_length=1, max_length=120)
    x: float
    y: float
    impl: Implementation
    ports: tuple[PortSpec, ...] = Field(max_length=64)
    config: dict[str, WorkflowValue] = Field(default_factory=dict, max_length=32)
    placement: Placement = Field(default_factory=Placement)
    policy: NodePolicy = Field(default_factory=NodePolicy)
    description: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        # Settings first: ports that follow them (a switch's cases) are only meaningful once they hold.
        self.impl.check(self.config)
        signature = self.impl.signature(self.placement.target, self.config)
        if signature is not None and self.ports != signature:
            raise ValueError("node ports must be its implementation's signature")
        self.impl.check_ports(self.ports)
        if self.placement.target not in self.impl.placements:
            raise ValueError(f"a {self.impl.kind} node cannot run on the {self.placement.target}")
        self.impl.check_placement(self.placement)
        reserved = {port.name for port in self.control_ports} & {port.name for port in self.ports}
        if reserved:
            raise ValueError(f"port name {sorted(reserved)[0]} is reserved for every node")
        inputs = {port.name for port in self.all_ports if port.direction == "input"}
        if set(self.policy.join) - inputs:
            raise ValueError("a join rule names one of the node's input ports")
        outputs = {port.name: port for port in self.ports if port.direction == "output"}
        if self.policy.on_error == "continue_with_default":
            if set(self.policy.defaults) != set(outputs):
                raise ValueError("continuing with defaults needs one default per output port")
            if any(not outputs[name].accepts(value) for name, value in self.policy.defaults.items()):
                raise ValueError("a default must be a value of its output port's type (a file port has none)")
        elif self.policy.defaults:
            raise ValueError("defaults are only read when the node continues with defaults")
        return self

    @property
    def control_ports(self) -> tuple[PortSpec, ...]:
        return (*CONTROL_PORTS, *self.impl.control_ports)

    @property
    def all_ports(self) -> tuple[PortSpec, ...]:
        """Declared ports, then control ports: what a wire may attach to."""
        return (*self.ports, *self.control_ports)

    def join(self, port: str) -> JoinRule:
        return self.policy.join.get(port, "all")

    @property
    def tool_name(self) -> str:
        """The name an agent calls this node by (its label, slugged; the caller de-duplicates)."""
        return _port_slug(self.label)[:64]

    def tool_schema(self, bound: Iterable[str] = ()) -> "ToolInputSchema":
        """This node's inputs as one tool's arguments: every declared input except the `bound`
        ones (wired in the graph), required unless the port is optional or holds a constant.
        Raises ValueError naming an input an agent could never supply (a file it never saw)."""
        bound = set(bound)
        properties: dict[str, ToolInputProperty] = {}
        required: list[str] = []
        for port in self.ports:
            if port.direction != "input" or port.name in bound:
                continue
            argument = port.tool_property()
            if argument is None:
                if port.required and self.constant(port.name) is None:
                    raise ValueError(f"{self.label}: input {port.name} ({port.value_type}) cannot be passed by an agent; wire it")
                continue
            properties[port.name] = argument
            if port.required and self.constant(port.name) is None:
                required.append(port.name)
        return ToolInputSchema(properties=properties, required=tuple(required))

    @property
    def tool_description(self) -> str:
        outputs = ", ".join(f"{port.name} ({port.value_type})" for port in self.ports if port.direction == "output")
        return (self.description or f"Runs the {self.impl.kind} node '{self.label}'.") + (f" Returns {outputs}." if outputs else "")

    @property
    def effects(self) -> frozenset[str]:
        return self.impl.effects | ({"machine"} if self.placement.target == "machine" else frozenset())

    def constant(self, port: str) -> WorkflowValue | None:
        """The constant an unwired input port takes, or None."""
        value = self.config.get(port)
        return None if value in (None, "") else value


class MachineCommand(WireModel):
    """One owner-scoped workflow step requested from one enrolled machine: the node's
    implementation, its config (settings, a script's source) and its resolved input values
    (an agent's `task`, a model's `images`, a function's arguments). The runner dispatches on
    `impl.kind` and re-checks everything it can locally (signature, function version, script
    digest) on top of the server's own checks."""

    id: UUID
    nonce: UUID
    machine: MachineRef
    workspace_id: UUID
    run_id: UUID
    workflow: WorkflowRevisionRef
    node_id: UUID
    impl: MachineImplementation
    config: dict[str, WorkflowValue] = Field(default_factory=dict, max_length=32)
    inputs: dict[str, WorkflowValue] = Field(default_factory=dict, max_length=32)
    expires_at: datetime
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: "owner": the machine's own workspace runs its own workflow — today's only shape, the
    #: runner's direct-subprocess path. "pooled": `workspace_id` is a DIFFERENT (tenant) workspace
    #: borrowing this machine, signed by the OWNER's key — the runner must route this through its
    #: gVisor sandbox (`galaius.sandbox.run_pooled`), never the direct path a same-workspace
    #: command gets. A pooled `EgressPolicy` travels inside `config["_pool_egress_allow"]` as a
    #: plain list of `{host, port}` — not a typed field here, since `galaius_core.pool` (which
    #: owns `EgressPolicy`) imports FROM this module for `MachineRef`; a typed field would cycle.
    tenancy: Literal["owner", "pooled"] = "owner"
    #: The account the run acts for (`RunInitiator.account`): the runner's local audit log names it.
    initiator_account: UUID | None = None

    @model_validator(mode="after")
    def runnable(self) -> Self:
        self.impl.check(self.config)
        # The vendor catalog (`MACHINE_MODELS`, `provider == "huggingface"`) is the only model
        # source this contract can check by a fixed registry — a workspace's OWN registered model
        # (`provider == "workspace"`) is a `UserModel` UUID the SERVER already validated at
        # save/registration time; this wire
        # contract has no workspace database to check it against and must never reject it.
        if self.impl.kind == "model" and self.impl.provider != "workspace" and self.impl.model not in MACHINE_MODELS:
            raise ValueError("machine model is not in the machine model registry")
        if self.impl.kind == "agent" and not isinstance(self.inputs.get("task"), str):
            raise ValueError("agent commands carry their task")
        if self.tenancy == "pooled" and self.impl.kind == "agent":
            raise ValueError("a pooled (cross-workspace) command cannot run an agent step — script, function and model only, for now")
        if self.impl.kind == "builtin" and (self.impl.op not in BuiltinImplementation.FILE_OPS or self.tenancy == "pooled"):
            raise ValueError("a machine runs a builtin only to read or write one of its own files")
        if self.tenancy == "pooled" and self.impl.kind == "script" and self.impl.origin != "inline":
            raise ValueError("a pooled (cross-workspace) command runs inline code only, never a file of the machine")
        return self

    # A file crosses machines through the server, over HTTP authenticated by the machine's own
    # token and scoped to THIS command while it is in flight: an input file (an `ArtifactRef`
    # value) is downloaded by the machine that runs the command, a file the command produces is
    # uploaded and comes back as an `ArtifactRef`. Both sides check its sha256 and size limit.

    def input_file_path(self, port: str, index: int = 0) -> str:
        return f"/v1/machine-channel/commands/{self.id}/inputs/{port}?index={index}"

    @property
    def upload_path(self) -> str:
        return f"/v1/machine-channel/commands/{self.id}/files"


class MachineFileQuery(WireModel):
    """The server asks one connected machine what is at `path` inside its owner's script folders
    ("" = the folders themselves), for a person picking a script there. Read-only: a folder answers its
    entries, a file its size and sha256; nothing runs. Signed with the machine's key like a
    command and short-lived; the machine refuses anything outside those folders."""

    id: UUID
    machine: MachineRef
    workspace_id: UUID
    path: str = Field(default="", max_length=1024)
    expires_at: datetime
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")


class MachineFileEntry(WireModel):
    name: str = Field(min_length=1, max_length=255)
    kind: Literal["file", "folder"]
    size: int | None = Field(default=None, ge=0)


class MachineGitOrigin(WireModel):
    """The git checkout a picked file sits in, as the machine read it: where it came from, at
    which commit, and whether the file differs from that commit. Shown, never pinned (the file's
    digest is what a run is pinned to)."""

    repository: str | None = Field(default=None, max_length=500)
    commit: str = Field(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")
    path: str = Field(min_length=1, max_length=1024)
    clean: bool


class MachineFileListing(WireModel):
    path: str = Field(max_length=1024)
    kind: Literal["folder", "file"]
    entries: tuple[MachineFileEntry, ...] = Field(default=(), max_length=500)
    #: A folder holding more entries than `entries` carries.
    truncated: bool = False
    size: int | None = Field(default=None, ge=0)
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    git: MachineGitOrigin | None = None

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.kind == "file") != (self.digest is not None and self.size is not None) or (self.kind == "file" and self.entries):
            raise ValueError("a file answers its size and digest; a folder its entries")
        return self


class MachineFileQueryResult(WireModel):
    query_id: UUID
    listing: MachineFileListing | None = None
    error: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def one_answer(self) -> Self:
        if (self.listing is None) == (self.error is None):
            raise ValueError("a file query answers a listing or an error")
        return self


#: Largest slice of a file one MachineDataRequest reads (a view reads a file slice by slice).
MACHINE_DATA_CHUNK = 1024 * 1024


class MachineDataRequest(WireRequest):
    """The server asks one connected machine about its owner's FILE ROOTS for the Data screen (and
    the agents its owner allowed): list a folder, stat a file, or read one slice of it. Read-only:
    nothing is written, nothing runs. `root` is the file root the request stays beneath ("" only
    to list the roots themselves); the machine walks `path` from that root part by part and refuses
    links, hidden names and anything but a plain file. Signed with the machine's key, short-lived;
    `type` is inside the signed payload so no other signed message can be replayed as this one."""

    type: Literal["data_request"] = "data_request"
    id: UUID
    machine: MachineRef
    workspace_id: UUID
    op: Literal["list", "stat", "read"]
    root: str = Field(default="", max_length=240)
    path: str = Field(default="", max_length=1024)
    offset: int = Field(default=0, ge=0)
    length: int = Field(default=0, ge=0, le=MACHINE_DATA_CHUNK)
    expires_at: datetime
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if not self.root and (self.op != "list" or self.path):
            raise ValueError("only a listing of the roots names no root")
        if self.op == "read" and self.length == 0:
            raise ValueError("a read names how many bytes it wants")
        return self


class MachineDataAnswer(WireModel):
    """The machine's answer to one MachineDataRequest: a folder's entries, a file's facts, or one
    slice of it (base64). `identity` ("<device>:<inode>:<mtime_ns>") lets the server refuse a file
    that changed between two slices."""

    request_id: UUID
    kind: Literal["file", "folder"] | None = None
    entries: tuple[MachineFileEntry, ...] = Field(default=(), max_length=500)
    truncated: bool = False
    size: int | None = Field(default=None, ge=0)
    modified_at: datetime | None = None
    identity: str | None = Field(default=None, max_length=80)
    offset: int | None = Field(default=None, ge=0)
    data: str | None = Field(default=None, max_length=(MACHINE_DATA_CHUNK * 4) // 3 + 8)
    error: str | None = Field(default=None, max_length=400)

    @model_validator(mode="after")
    def one_answer(self) -> Self:
        if (self.error is None) == (self.kind is None):
            raise ValueError("a data request answers facts or an error")
        return self


#: Largest window of one agent run's event stream a `tail` request carries.
MACHINE_AGENT_TAIL = 256 * 1024
#: What an agent started from the web may do on a machine — the launcher's touch scopes
#: (`galaius.agents.vocabulary.TouchScope`), set on the machine by its owner.
AgentTouchScope = Literal["read_only", "workspace_write", "full_access"]
#: The touch scopes least first: a narrower one is earlier.
AGENT_TOUCH_SCOPES: tuple[AgentTouchScope, ...] = get_args(AgentTouchScope)
AgentRole = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=80)]
AgentModelId = Annotated[str, Field(pattern=r"^[A-Za-z0-9._:/@+-]+$", min_length=1, max_length=256)]
AgentProvider = Literal["claude", "codex"]
#: `agent`: a role run by the launcher (`galaius agents spawn`); `session`: an open conversation
#: like the editor's chat, which ASKS before a command or a file change (the approvals the web
#: answers); `continued`: a copy of one of the owner's own editor conversations, continued here.
AgentRunKind = Literal["agent", "session", "continued"]


class AgentStartSpec(WireRequest):
    """What the owner asks for when starting an agent on one of his machines: the folder (an agent
    root, and a path beneath it), the role, the CLI (None: the best available by the role's
    criterion) and the brief."""

    root: str = Field(min_length=1, max_length=240)
    path: str = Field(default="", max_length=1024)
    kind: Literal["agent", "session"] = "agent"
    #: The role an `agent` runs as; a `session` has none.
    role: AgentRole | None = None
    provider: AgentProvider | None = None
    #: A model this machine offers (`MachineAgentAnswer.models` / `session_models`); None: the
    #: role's criterion (agent) or the route's default (session) decides.
    model: AgentModelId | None = None
    text: str = Field(min_length=1, max_length=8000, pattern=r"\S")
    #: What this run may do, never more than the PC's own `agent_permission` (the PC clamps it).
    #: Left out of the message when None, so a runner released before the field still verifies it.
    permission: AgentTouchScope | None = Field(default=None, exclude_if=lambda value: value is None)
    #: The project this run works on.
    project: UUID | None = Field(default=None, exclude_if=lambda value: value is None)
    #: Hand the project's secrets (the server's vault) to this run, as the .env of its checkout; off
    #: unless asked for this start, so a run started without it has none to leak.
    with_secrets: bool = Field(default=False, exclude_if=lambda value: value is False)

    @model_validator(mode="after")
    def role_for_an_agent(self) -> Self:
        if (self.kind == "agent") != (self.role is not None):
            raise ValueError("an agent names its role; a session names none")
        return self


class AgentMessageSpec(WireRequest):
    text: str = Field(min_length=1, max_length=8000, pattern=r"\S")


class MachineAgentRequestBase(WireRequest):
    """The machine OWNER, signed in on the web, drives coding agents on one of his computers the
    way the editor panel does. One subclass per op, chosen by `op`. Signed with the machine's key,
    short-lived, `type` inside the signed payload; the machine refuses anything outside its agent
    roots and any run it did not start for this channel. `action`: changes something there (its
    id is accepted once, it is audited); `seconds`: how long the server waits for the answer."""

    type: Literal["agent_request"] = "agent_request"
    id: UUID
    machine: MachineRef
    workspace_id: UUID
    initiator_account: UUID
    expires_at: datetime
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")
    action: ClassVar[bool] = False
    seconds: ClassVar[float] = 15
    #: What the machine's runner must say it supports (its hello's `features`) to be asked this:
    #: an older runner is told apart from a silent one.
    feature: ClassVar[str] = "agent_control"


class AgentFoldersRequest(MachineAgentRequestBase):
    """The agent roots and the roles the machine can start ("" root), or the plain folders beneath one."""

    op: Literal["folders"] = "folders"
    root: str = Field(default="", max_length=240)
    path: str = Field(default="", max_length=1024)

    @model_validator(mode="after")
    def beneath_a_root(self) -> Self:
        if self.path and not self.root:
            raise ValueError("a path lies beneath a root")
        return self


class AgentRunsRequest(MachineAgentRequestBase):
    op: Literal["runs"] = "runs"


class AgentTailRequest(MachineAgentRequestBase):
    """One run's stream after byte `cursor` (None: its last lines)."""

    op: Literal["tail"] = "tail"
    run_id: UUID
    cursor: int | None = Field(default=None, ge=0)


#: The images a run's step may name, one row per kind: suffixes, content type, the first bytes that
#: prove it (WEBP: "RIFF" then "WEBP" at byte 8). Every image rule below is built from this table.
IMAGE_TYPES: tuple[tuple[tuple[str, ...], str, tuple[bytes, ...]], ...] = (
    ((".png",), "image/png", (b"\x89PNG\r\n\x1a\n",)),
    ((".jpg", ".jpeg"), "image/jpeg", (b"\xff\xd8\xff",)),
    ((".gif",), "image/gif", (b"GIF87a", b"GIF89a")),
    ((".webp",), "image/webp", (b"RIFF",)),
)
ImageContentType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]
_IMAGE_SUFFIX = "|".join(re.escape(suffix[1:]) for suffixes, _, _ in IMAGE_TYPES for suffix in suffixes)
#: An image path a step names, as its input wrote it: a whole quoted string, double or single (spaces
#: allowed: JSON `"file_path": "…"`, the stream's own `file_path='…'`, a `file:` URL), or an unquoted
#: absolute / `~/` path inside a command (no spaces; never the `//host/…` of a URL).
MEDIA_PATH = re.compile(
    rf'"(?:file:(?://)?)?(?P<double>(?:~/|/)(?!/)[^"\n]+?\.(?:{_IMAGE_SUFFIX}))"'
    rf"|'(?:file:(?://)?)?(?P<single>(?:~/|/)(?!/)[^'\n]+?\.(?:{_IMAGE_SUFFIX}))'"
    rf"|(?:(?<=file:)|(?<=file://)|(?<![\w/:.~'\"-]))(?P<bare>(?:~/|/)(?!/)[^\s\"'<>`|;&()]+?\.(?:{_IMAGE_SUFFIX}))(?![\w.])", re.I)
#: A run's image as the web names it: `media_key` of the path its step wrote.
MEDIA_KEY = rf"^[0-9a-f]{{20}}\.(?:{_IMAGE_SUFFIX})$"
#: Largest image one AgentMediaRequest carries back (raw bytes, before base64).
MACHINE_AGENT_MEDIA = 6 * 1024 * 1024


def image_type(data: bytes) -> ImageContentType | None:
    """The content type `data`'s first bytes prove (`IMAGE_TYPES`), or None: never the file's name."""
    for _, kind, signatures in IMAGE_TYPES:
        if any(data.startswith(signature) for signature in signatures) and (kind != "image/webp" or data[8:12] == b"WEBP"):
            return kind  # type: ignore[return-value]
    return None


def media_key(path: str) -> str:
    """The name an image path keeps on the web: sha256 of the path EXACTLY as the step wrote it
    (never resolved: the server and the PC derive it from the same stream line), plus its suffix.
    The path itself never travels in a request."""
    suffix = PurePosixPath(path).suffix.lower()
    if not re.fullmatch(rf"\.(?:{_IMAGE_SUFFIX})", suffix):
        raise ValueError("not an image path")
    return hashlib.sha256(path.encode()).hexdigest()[:20] + suffix


def media_paths(text: str) -> tuple[str, ...]:
    """The image paths one step's input names, in order, each once."""
    return tuple(dict.fromkeys(match["double"] or match["single"] or match["bare"] for match in MEDIA_PATH.finditer(text)))


class AgentMediaRequest(MachineAgentRequestBase):
    """One image a run's step names (`media_key` of its path). The machine serves it only when a
    tool step of THAT run (one it started for the web, or launched by one) names a path with this
    key, and only a plain image file no larger than MACHINE_AGENT_MEDIA."""

    op: Literal["media"] = "media"
    run_id: UUID
    name: str = Field(pattern=MEDIA_KEY)
    feature: ClassVar[str] = "agent_media"


class AgentMedia(WireModel):
    """An image's bytes (base64) and the content type its first bytes prove: an answer whose bytes are
    not base64, or not the image it claims, never validates."""

    content_type: ImageContentType
    data: str = Field(max_length=(MACHINE_AGENT_MEDIA * 4) // 3 + 8)

    @model_validator(mode="after")
    def proven(self) -> Self:
        try:
            content = base64.b64decode(self.data, validate=True)
        except ValueError:
            raise ValueError("image bytes are not base64") from None
        if image_type(content) != self.content_type:
            raise ValueError(f"these bytes are not {self.content_type}")
        return self

    @property
    def content(self) -> bytes:
        return base64.b64decode(self.data)


#: A project secret's name: an environment variable's, never one that steers the PC's own tools or
#: the agent CLI itself (its path, its shell, its credentials, galaius's own settings).
_RESERVED_SECRET = re.compile("^(" + "|".join((
    # the shell, the user, the system (POSIX and Windows)
    r"PATH", r"HOME", r"SHELL", r"USER", r"LOGNAME", r"TERM", r"TMPDIR", r"TEMP", r"TMP", r"PWD", r"IFS", r"LANG", r"LC_.*", r"DISPLAY",
    r"SYSTEMROOT", r"WINDIR", r"COMSPEC", r"PATHEXT", r"APPDATA", r"LOCALAPPDATA", r"USERPROFILE", r"XDG_.*", r"DBUS_.*",
    # shell start-up and what runs before every command
    r"BASH_ENV", r"ENV", r"ZDOTDIR", r"PROMPT_COMMAND", r"PS4", r"SHELLOPTS", r"BASHOPTS",
    # how programs and libraries load
    r"LD_.*", r"DYLD_.*", r"GLIBC_TUNABLES", r"GCONV_PATH", r"LOCPATH", r"MALLOC_.*", r"HOSTALIASES",
    r"PYTHON.*", r"NODE_OPTIONS", r"NODE_PATH", r"NODE_EXTRA_CA_CERTS", r"NODE_TLS_REJECT_UNAUTHORIZED", r"PERL5.*", r"RUBYOPT", r"RUBYLIB",
    r"JAVA_TOOL_OPTIONS", r"_JAVA_OPTIONS",
    # where traffic goes and which certificates it trusts
    r"HTTPS?_PROXY", r"https?_proxy", r"ALL_PROXY", r"all_proxy", r"NO_PROXY", r"no_proxy", r"SSL_CERT_.*", r"REQUESTS_CA_BUNDLE", r"CURL_CA_BUNDLE",
    # package tools, editors, pagers, credentials helpers, containers
    r"PIP_.*", r"UV_.*", r"NPM_CONFIG_.*", r"CARGO_.*", r"RUSTC_WRAPPER", r"GOPROXY", r"GOFLAGS", r"EDITOR", r"VISUAL", r"PAGER", r"LESSOPEN",
    r"SUDO_ASKPASS", r"GNUPGHOME", r"DOCKER_.*", r"KUBECONFIG", r"GIT_.*", r"SSH_.*", r"GPG_.*",
    # galaius itself and every agent CLI it drives
    r"INTERACT_.*", r"GALAIUS_.*", r"ANTHROPIC_.*", r"OPENAI_.*", r"CLAUDE_.*", r"CODEX_.*", r"GEMINI_.*", r"GOOGLE_GENAI_.*", r"MISTRAL_.*",
    r"OPENROUTER_.*", r"GH_.*", r"GITHUB_TOKEN", r"MCP_.*", r"BUN_.*", r"NODE_.*", r"VIRTUAL_ENV", r"CONDA_.*",
)) + ")$")


def _secret_name(value: str) -> str:
    if _RESERVED_SECRET.match(value):
        raise ValueError(f"{value} steers the PC's own tools or the agent itself; a project secret never sets it")
    return value


ProjectSecretName = Annotated[str, Field(pattern=r"^[A-Z_][A-Z0-9_]{0,127}$"), AfterValidator(_secret_name)]


def _quotable(value: str) -> str:
    """Written `NAME='value'` in the project's .env: a quote, a line break or a NUL would let a value
    run code when the file is sourced or add another name - refused (a PEM key: store it base64)."""
    if any(character in value for character in "'\r\n\0"):
        raise ValueError("a secret value is one line without a single quote (store a multi-line key base64-encoded)")
    return value


#: A project's vault holds at most this many secrets, each value at most this long, all values
#: together (as one delivery carries them) at most `SECRETS_TOTAL`.
SECRETS_PER_PROJECT = 100
SECRET_VALUE_MAX = 32 * 1024
SECRETS_TOTAL = 64 * 1024
ProjectSecretText = Annotated[str, Field(min_length=1, max_length=SECRET_VALUE_MAX), AfterValidator(_quotable)]


class ProjectSecretValue(WireRequest):
    """A project secret's value as its owner sets it from the web (write-only: never read back)."""

    value: SecretStr = Field(min_length=1, max_length=SECRET_VALUE_MAX)

    @field_validator("value")
    @classmethod
    def one_quotable_line(cls, value: SecretStr) -> SecretStr:
        _quotable(value.get_secret_value())
        return value


class ProjectSecretInfo(WireModel):
    """One secret of a project as the web shows it: its name, who set it, when - never its value."""

    name: ProjectSecretName
    updated_by: UUID
    #: Who that is, as the workspace shows people (their name, else their email).
    updated_by_name: str = Field(default="", max_length=320)
    updated_at: datetime


class SealedSecrets(WireModel):
    """A project's secrets sealed FOR ONE MACHINE (`galaius_core.sealing.SecretsSeal`): AES-256-GCM
    under a key derived from that machine's signing key, the request id, the project and its
    repository origin as associated data. Keeps the values out of any frame log or proxy capture; it
    is no boundary against the server, which can derive the same key. `origin`: the project's
    repository - the PC writes the secrets only into a checkout of it. An empty set is sealed too:
    it clears the checkout's .env."""

    project: UUID
    origin: str = Field(max_length=512)
    nonce: str = Field(pattern=r"^[0-9a-f]{24}$")
    ciphertext: str = Field(min_length=32, max_length=2 * (SECRETS_TOTAL + 64), pattern=r"^[0-9a-f]+$")
    #: How many secrets are inside (what the run's log may say; never their names).
    count: int = Field(ge=0, le=SECRETS_PER_PROJECT)


class AgentStartRequest(MachineAgentRequestBase, AgentStartSpec):
    op: Literal["start"] = "start"
    #: The named project's secrets, sealed for this machine by the server at start (None: none).
    secrets: SealedSecrets | None = Field(default=None, exclude_if=lambda value: value is None)
    action: ClassVar[bool] = True
    #: The machine gives its launcher 120 s; the server waits longer, so it never gives up on a
    #: start that then happens anyway (a retry would start a second agent).
    seconds: ClassVar[float] = 135


class AgentSendRequest(MachineAgentRequestBase, AgentMessageSpec):
    op: Literal["send"] = "send"
    run_id: UUID
    action: ClassVar[bool] = True
    seconds: ClassVar[float] = 75


class AgentStopRequest(MachineAgentRequestBase):
    op: Literal["stop"] = "stop"
    run_id: UUID
    action: ClassVar[bool] = True


class AgentOptionsRequest(MachineAgentRequestBase):
    """What can be started here: roles, models per CLI, and whether a session can open. With
    `role`: `models` is what that role's own rule picks on each CLI here (best first), so the owner
    sees before starting which model it will run on, and which CLI cannot run it at all."""

    op: Literal["options"] = "options"
    role: AgentRole | None = None
    seconds: ClassVar[float] = 45


class AgentAnswerSpec(WireRequest):
    """The owner's answer to one approval a session asked for (`AgentInteraction`)."""

    interaction_id: str = Field(min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._:@+-]+$")
    #: `AgentInteraction.digest` of the request the owner saw: the machine refuses the answer if
    #: what is pending there differs (another command, another change).
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    values: dict[Annotated[str, Field(max_length=160)], str | bool] = Field(max_length=32)


class AgentAnswerRequest(MachineAgentRequestBase, AgentAnswerSpec):
    op: Literal["answer"] = "answer"
    run_id: UUID
    action: ClassVar[bool] = True
    seconds: ClassVar[float] = 30


class AgentSessionsRequest(MachineAgentRequestBase):
    """The owner's own editor conversations whose folder lies inside an agent root."""

    op: Literal["sessions"] = "sessions"
    seconds: ClassVar[float] = 30


class AgentContinueRequest(MachineAgentRequestBase, AgentMessageSpec):
    """Continue one of the owner's editor conversations here: a copy of it resumes with `text`;
    the editor's own conversation is never written to."""

    op: Literal["continue"] = "continue"
    session_id: UUID
    action: ClassVar[bool] = True
    seconds: ClassVar[float] = 135


class AgentLogsRequest(MachineAgentRequestBase):
    """The machine connection's recent log lines, or one run's error output (`run_id`)."""

    op: Literal["logs"] = "logs"
    run_id: UUID | None = None


class AgentSettingsRequest(MachineAgentRequestBase):
    """Which CLIs may run agents here, and the model each of galaius's own tools resolves to here
    (its rule is the owner's, synced from his account; the keys that decide what clears it are this
    machine's)."""

    op: Literal["settings"] = "settings"
    seconds: ClassVar[float] = 45
    feature: ClassVar[str] = "agent_settings"


class AgentProgramsRequest(MachineAgentRequestBase):
    """Each agent program here: installed, signed in, an install or sign-in under way (`providers`)."""

    op: Literal["programs"] = "programs"
    feature: ClassVar[str] = "agent_programs"


class AgentProgramInstallRequest(MachineAgentRequestBase):
    """Install `provider`'s program for this PC's user with its vendor's own installer (fixed on the
    PC, never sent), then start its sign-in; answered at once with its state (`providers`). Asked
    again while under way or ready: the same state, nothing started twice; after a failure or an
    expired sign-in: started again."""

    op: Literal["program_install"] = "program_install"
    provider: AgentProvider
    action: ClassVar[bool] = True
    feature: ClassVar[str] = "agent_programs"


class AgentProviderSwitchRequest(MachineAgentRequestBase):
    """Let one CLI run agents here, or stop it (VS Code: "Providers That Run Agents")."""

    op: Literal["provider"] = "provider"
    provider: AgentProvider
    active: bool
    action: ClassVar[bool] = True
    feature: ClassVar[str] = "agent_settings"


#: How far one folder of a PC is open to workflows, Data and agents, least first; each level
#: implies the ones before it. `hidden`: not listed (every folder starts here); `see`: names, sizes
#: and dates; `read`: bytes; `write_on_review`: writes land in a staging copy the owner accepts on
#: the PC, by digest; `sandbox`: read + write in place, apart from every other root; `write`: read +
#: write in place. Set on the PC or from the web, applied at once either way; whatever the web asks,
#: the PC keeps its home folder itself, everything outside it, hidden names and credential stores
#: closed (runner feature `places_direct`; an older runner holds a widening as a `MachinePlaceChange`
#: until its owner runs `galaius machine approve` there).
PlaceLevel = Literal["hidden", "see", "read", "write_on_review", "sandbox", "write"]
PLACE_LEVELS: tuple[PlaceLevel, ...] = get_args(PlaceLevel)
#: A folder of the PC relative to its working directory (its home folder): "/"-separated names, none
#: hidden, empty, "." or "..".
def _place_path(value: str) -> str:
    parts = value.split("/")
    if any(not part or part in {".", ".."} or part.startswith(".") or "\\" in part or ":" in part for part in parts):
        raise ValueError("a place is a folder relative to the working directory: '/'-separated names, none hidden, empty, '.', '..', ':' or '\\'")
    return value


PlacePath = Annotated[str, Field(min_length=1, max_length=1024), AfterValidator(_place_path)]
#: A folder beneath another, by the same rules ("" the folder itself).
FolderPath = Annotated[str, Field(max_length=1024), AfterValidator(lambda value: value and _place_path(value))]


class MachinePlace(WireModel):
    """One folder the owner set a level on. `refused`: why that level is not in force now (a
    credential store, a link on the way, a sandbox overlapping another root), in plain words."""

    path: str = Field(max_length=1024)
    level: PlaceLevel
    refused: str = Field(default="", max_length=400)


class MachinePlaceChange(WireModel):
    """A widening asked from the web, waiting on the PC until its owner confirms it there (`PlaceLevel`: older runners only).
    `digest`: sha256 of the canonical change (machine, path, level, previous, id), what the PC's
    local log and the server's audit both keep."""

    id: UUID
    path: str = Field(max_length=1024)
    level: PlaceLevel
    previous: PlaceLevel
    asked_at: datetime
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class MachinePlaceEntry(WireModel):
    """One name in a whole-PC browse page, with the level in force on it (names only, never bytes)."""

    name: str = Field(min_length=1, max_length=255)
    kind: Literal["folder", "file"]
    level: PlaceLevel


class MachineReviewFile(WireModel):
    path: str = Field(max_length=1024)
    change: Literal["added", "changed", "deleted"]
    size: int | None = Field(default=None, ge=0)


class MachinePlaceReview(WireModel):
    """Writes into a `write_on_review` folder, held in a staging copy on the PC. `working`: its
    agent still runs; `ready`: the owner may accept `digest` on the PC; `blocked`: it cannot be
    applied (`reason`: e.g. it touches `.git` internals), only discarded."""

    id: UUID
    place: str = Field(max_length=1024)
    origin: Literal["agent", "workflow"]
    run_id: UUID | None = None
    created_at: datetime
    state: Literal["working", "ready", "blocked"]
    reason: str = Field(default="", max_length=400)
    files: tuple[MachineReviewFile, ...] = Field(default=(), max_length=500)
    truncated: bool = False
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class MachineFence(WireModel):
    """Whether agent CLIs on this PC run inside an OS fence built from the levels. `on`: the owner
    switched it on there (`galaius machine fence on`); `available`: this PC can build one
    (`reason` says why not); agents are fenced only when both hold."""

    platform: str = Field(max_length=40)
    available: bool
    on: bool
    reason: str = Field(default="", max_length=400)


class MachinePlacesView(WireModel):
    """The PC's levels as the PC holds them now, the widenings waiting on it (`PlaceLevel`: older
    runners only), whether browsing lists file names and folders outside the home folder too
    (`browse`, `galaius machine browse on`; folder names inside it are always listed, `places_direct`),
    its fence, and the folder it suggests as a sandbox."""

    places: tuple[MachinePlace, ...] = Field(default=(), max_length=256)
    pending: tuple[MachinePlaceChange, ...] = Field(default=(), max_length=64)
    browse: bool = False
    fence: MachineFence
    suggested_sandbox: str = Field(default="", max_length=240)


class PlacesRequest(MachineAgentRequestBase):
    """The levels, pending widenings, browse switch and fence of this PC."""

    op: Literal["places"] = "places"
    feature: ClassVar[str] = "places"


class PlaceLevelRequest(MachineAgentRequestBase):
    """Set one folder's level (`PlaceLevel`: applied at once)."""

    op: Literal["place_level"] = "place_level"
    path: PlacePath
    level: PlaceLevel
    action: ClassVar[bool] = True
    feature: ClassVar[str] = "places"


class PlaceCancelRequest(MachineAgentRequestBase):
    """Withdraw a widening still waiting on the PC."""

    op: Literal["place_cancel"] = "place_cancel"
    change_id: UUID
    action: ClassVar[bool] = True
    feature: ClassVar[str] = "places"


class PlaceBrowseRequest(MachineAgentRequestBase):
    """One page of names beneath `path` ("" the working directory): folder names inside the home
    folder always, file names and folders outside it only while `browse` is on (`MachinePlacesView`)."""

    op: Literal["place_browse"] = "place_browse"
    path: str = Field(default="", max_length=1024)
    cursor: int = Field(default=0, ge=0, le=100_000)
    feature: ClassVar[str] = "places"


class PlaceReviewsRequest(MachineAgentRequestBase):
    op: Literal["place_reviews"] = "place_reviews"
    seconds: ClassVar[float] = 30
    feature: ClassVar[str] = "places"


class PlaceReviewRequest(MachineAgentRequestBase):
    """One review and its unified diff (`lines`), for reading; accepting happens on the PC."""

    op: Literal["place_review"] = "place_review"
    review_id: UUID
    seconds: ClassVar[float] = 30
    feature: ClassVar[str] = "places"


class PlaceDiscardRequest(MachineAgentRequestBase):
    """Drop a review's staged writes (nothing reaches the folder)."""

    op: Literal["place_discard"] = "place_discard"
    review_id: UUID
    action: ClassVar[bool] = True
    feature: ClassVar[str] = "places"


#: The scope the web may give agents on a PC: `full_access` (no sandbox) is set on the PC only.
WebAgentScope = Literal["read_only", "workspace_write"]
#: A folder name a workspace is cloned into, inside an agent root: no hidden, relative, or
#: Windows-reserved name, no trailing dot.
#: The names Windows keeps for devices, lowercased: no file or folder is named so, whatever follows a
#: dot. Microsoft, "Naming Files, Paths, and Namespaces" (learn.microsoft.com/windows/win32/fileio/naming-a-file),
#: as recalled 2026-10-09, not re-fetched: COM0 / LPT0 added (later Windows versions list them; refusing more is safe).
RESERVED_DEVICE_NAMES = frozenset({"con", "prn", "aux", "nul", "conin$", "conout$", *(f"{port}{index}" for port in ("com", "lpt") for index in (*range(10), "¹", "²", "³"))})


def _workspace_name(value: str) -> str:
    if value.endswith(".") or value.split(".")[0].lower() in RESERVED_DEVICE_NAMES:
        raise ValueError("a workspace folder name never ends with '.' and is never a reserved device name")
    return value


#: The longest project folder name.
WORKSPACE_NAME_MAX = 64
WorkspaceName = Annotated[str, Field(pattern=rf"^[A-Za-z0-9][A-Za-z0-9._-]{{0,{WORKSPACE_NAME_MAX - 1}}}$"), AfterValidator(_workspace_name)]
WORKSPACE_NAME: TypeAdapter[str] = TypeAdapter(WorkspaceName)


class GitRemote(WireModel):
    """A repository a PC may clone, parsed once from its address: `https://host[:port]/path`,
    `ssh://[user@]host[:port]/path` or `user@host:path`. Never credentials in the address (they stay
    in the PC's own git setup), never another transport (file, ext, git), never a part starting
    with '-' (no option injection), never '.' or '..' parts."""

    #: A refused address is never echoed in the error: a checkout's remote may hold a token.
    model_config = ConfigDict(hide_input_in_errors=True)
    url: str = Field(min_length=1, max_length=512)
    host: str = Field(max_length=253)
    path: str = Field(max_length=512)
    _SHAPES: ClassVar[tuple[re.Pattern[str], ...]] = (
        re.compile(r"^https://(?P<host>[A-Za-z0-9][A-Za-z0-9.-]*)(?::[0-9]{1,5})?/(?P<path>[A-Za-z0-9._~/-]+)$"),
        re.compile(r"^ssh://(?:[A-Za-z0-9._-]+@)?(?P<host>[A-Za-z0-9][A-Za-z0-9.-]*)(?::[0-9]{1,5})?/(?P<path>[A-Za-z0-9._~/-]+)$"),
        re.compile(r"^[A-Za-z0-9._-]+@(?P<host>[A-Za-z0-9][A-Za-z0-9.-]*):(?P<path>[A-Za-z0-9._~/-]+)$"),
    )

    @model_validator(mode="before")
    @classmethod
    def parsed(cls, value: object) -> object:
        url = value if isinstance(value, str) else value.get("url") if isinstance(value, dict) else None
        if not isinstance(url, str):
            return value
        match = next((found for shape in cls._SHAPES if (found := shape.match(url))), None)
        if match is None:
            raise ValueError("a repository address is https://host/path, ssh://host/path or user@host:path, with no password in it")
        path = match["path"].strip("/")
        if any(part in {"", ".", ".."} or part.startswith("-") for part in path.split("/")):
            raise ValueError("a repository path never holds an empty, '.' or '..' part, or one starting with '-'")
        return {"url": url, "host": match["host"].lower(), "path": path}

    #: A URL's scheme, and everything up to the last '@' before the first '/' after it (its userinfo).
    _USERINFO: ClassVar[re.Pattern[str]] = re.compile(r"^(?P<scheme>[A-Za-z][A-Za-z0-9+.-]*://)(?:(?P<userinfo>[^/]*)@)?")

    @classmethod
    def stripped(cls, url: str) -> Self | None:
        """The repository a checkout's remote names, its credentials dropped: `https://user:token@host/…`
        -> `https://host/…`, `ssh://user:password@host/…` -> `ssh://user@host/…` (an ssh user is a login
        name, never a secret); None for a remote no PC clones (another transport, a query string...).
        What a PC reports of a folder's origin; a password never leaves it, not even in an error."""
        def kept(match: re.Match[str]) -> str:
            scheme, user = match["scheme"].lower(), (match["userinfo"] or "").split(":", 1)[0]
            return scheme + (f"{user}@" if scheme == "ssh://" and user else "")
        try:
            return cls(url=cls._USERINFO.sub(kept, url.strip(), count=1))
        except ValueError:
            return None

    @property
    def origin(self) -> str:
        """`host/path` without a trailing `.git`: what `MachineAgentSettings.clone_origins` patterns match."""
        return f"{self.host}/{self.path.removesuffix('.git')}"

    @property
    def name(self) -> str:
        """The folder a clone lands in by default: the repository's last path part, `.git` dropped."""
        return self.path.rsplit("/", 1)[-1].removesuffix(".git")

    def allowed_by(self, origins: Iterable[str]) -> bool:
        """Whether one of `origins` covers this repository: `host/owner/repo` exactly, or
        `host/owner/*` for every repository directly under it."""
        mine = self.origin.lower()
        return any(mine == origin.lower() or (origin.endswith("/*") and mine.rsplit("/", 1)[0] == origin[:-2].lower()) for origin in origins)


#: A repository address on the wire (a plain string), parsed into a `GitRemote` where it is read.
GitUrl = Annotated[GitRemote, BeforeValidator(lambda value: GitRemote(url=value) if isinstance(value, str) else value), PlainSerializer(lambda remote: remote.url, return_type=str)]


#: One allowed clone origin: `host/owner/repo` or `host/owner/*`.
CloneOrigin = Annotated[str, Field(pattern=r"^[A-Za-z0-9.-]+(?:/[A-Za-z0-9._~-]+)+(?:/\*)?$", max_length=200)]


class MachineAgentSettings(WireModel):
    """Everything about agents on one PC its owner may set from the web: whether agents run there,
    the folders they may be started in (relative to the PC's working directory, a file root's
    rules), the permission they start with, the two opt-ins, and the repositories the PC may clone
    into an agent folder. Nothing else (file / script roots, script approvals, the token, the
    server address, upgrades) is ever reachable from the web."""

    run_agents: bool = False
    agent_roots: tuple[PlacePath, ...] = Field(default=(), max_length=32)
    agent_permission: AgentTouchScope = "workspace_write"
    continue_conversations: bool = False
    answer_approvals: bool = False
    clone_origins: tuple[CloneOrigin, ...] = Field(default=(), max_length=32)

    def narrows(self, than: "MachineAgentSettings") -> bool:
        """Whether these settings grant nothing `than` does not (agents off, fewer folders or
        origins, a narrower scope, opt-ins off): such a change never needs to be built from the
        PC's latest revision - it can only take power away."""
        return ((not self.run_agents or than.run_agents) and set(self.agent_roots) <= set(than.agent_roots) and set(self.clone_origins) <= set(than.clone_origins)
                and AGENT_TOUCH_SCOPES.index(self.agent_permission) <= AGENT_TOUCH_SCOPES.index(than.agent_permission)
                and (not self.continue_conversations or than.continue_conversations) and (not self.answer_approvals or than.answer_approvals))


class WebAgentSettings(MachineAgentSettings):
    """Agent settings as the web may set them: never `full_access` (no sandbox), which the PC's
    owner sets on the PC itself."""

    agent_permission: WebAgentScope = "workspace_write"


class MachineAgentSettingsChange(WireRequest):
    """The owner's web edit of one PC's agent settings, built from the state the PC last reported:
    `based_on` is that state's `revision`, so a change made on the PC since is never undone."""

    settings: WebAgentSettings
    based_on: int = Field(ge=0)


class MachineAgentSettingsUpdate(WireModel):
    """The server's signed desired agent settings for one PC (`type` inside the signed body). The
    PC applies a `version` above the one it applied last, only when `based_on` is its current
    `revision` and its owner has not switched web control off there."""

    type: Literal["agent_settings"] = "agent_settings"
    machine: MachineRef
    workspace_id: UUID
    version: int = Field(ge=1)
    based_on: int = Field(ge=0)
    settings: WebAgentSettings
    changed_by: UUID
    changed_at: datetime
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")


#: Every machine token starts so: the server finds a machine's token in an `Authorization` header by it.
MACHINE_TOKEN_PREFIX = "iwm_"


class MachineTokenSwap(WireModel):
    """The replacement the server hands a machine for its one-time bootstrap token, sent once on the
    machine channel to a connection opened with that bootstrap token (a token that travelled where
    others can read it). Signed with the bootstrap token's key as every signed machine message is:
    HMAC-SHA256 of the canonical JSON, `signature` left out, keyed by SHA-256 of the token. The
    machine saves `token` and connects again with it; the bootstrap token is then spent."""

    type: Literal["token_swap"] = "token_swap"
    machine: MachineRef
    token: str = Field(pattern=rf"^{MACHINE_TOKEN_PREFIX}[A-Za-z0-9_-]{{32,252}}$", repr=False)
    signature: str = Field(pattern=r"^[0-9a-f]{64}$")


class MachineAgentSettingsState(WireModel):
    """What a PC holds now: `revision` counts every change of its agent settings (web or local),
    `version` the last web version it applied (0: none), `remote` whether it takes web changes
    (`galaius machine remote on|off`, the PC's own kill switch), `refused` each folder it cannot
    use and why, `detail` why the last web version was not applied (empty: it was)."""

    revision: int = Field(ge=0)
    version: int = Field(ge=0)
    remote: bool
    settings: MachineAgentSettings
    refused: tuple[str, ...] = Field(default=(), max_length=64)
    detail: str = Field(default="", max_length=400)
    #: The web version `detail` refused (0: none): the server stops offering it.
    refused_version: int = Field(default=0, ge=0)


class TranscriptMedia(WireModel):
    """An image a run's step points at, where the reader can fetch it (`src`: the app path that
    serves it; "" for the server's own mirror of the run)."""

    name: str
    label: str
    kind: Literal["image"] = "image"
    src: str = ""


class TranscriptItem(WireModel):
    """One line of an agent run as a person reads it: what it said (`said`, its last word `final`),
    its « [role] … » opening line alone (`stated`: a label, or the whole answer of a one-line run),
    a tool step, what was sent to it, a harness check, an error."""

    at: float
    kind: Literal["said", "final", "stated", "step", "sent", "check", "error"]
    text: str
    tool: str | None = None
    input: str = ""
    media: tuple[TranscriptMedia, ...] = ()


class MachineRunEvents(WireModel):
    """A window of a run on one of the owner's PCs: its lines after a byte cursor, and where the next starts."""

    run_id: UUID
    cursor: int | None = None
    truncated: bool = False
    items: tuple[TranscriptItem, ...] = ()


class MachineAgentSettingsView(WireModel):
    """One PC's agent settings as its page shows them: what the PC last reported (None: it never
    did - its galaius predates web settings, or it never connected since), the web version still
    waiting to reach it (None: none), who changed it and when, and whether its galaius takes web
    settings at all (`supported`)."""

    state: MachineAgentSettingsState | None = None
    pending: MachineAgentSettings | None = None
    pending_version: int = Field(default=0, ge=0)
    changed_by: UUID | None = None
    changed_at: datetime | None = None
    supported: bool = False
    online: bool = False


class MachineAgentHistoryEntry(WireModel):
    """One line of a PC's agent history on its page: a web settings version (`version`, and
    `changes`: field -> [before, after] against the version before it) or another agent action the
    owner took there (start, stop, a workspace prepared...), by whom and when."""

    action: str = Field(max_length=60)
    account_id: UUID | None = None
    at: datetime
    version: int | None = Field(default=None, ge=1)
    changes: dict[str, tuple[Any, Any]] = Field(default_factory=dict)


#: Where starting a project on a PC stands, in the order it is checked: the PC is not connected; its
#: galaius predates web settings (only an install on that PC fixes it); agents are off or it has no
#: agent folder; the project names no repository; the project is not on that PC yet; being cloned;
#: the last clone failed; ready to start in.
ProjectReadinessState = Literal["offline", "outdated", "agents_off", "no_repository", "no_workspace", "preparing", "failed", "ready"]


class MachineProjectReadiness(WireModel):
    """Whether a project can be started on a PC now, and if not what is missing there in plain words
    (`said`) and the one action that fixes it from here (`action`, None when none can: the PC is
    off, or only an install on it can). `install`: the installer lines to run on the PC once
    (state `outdated`). Ready: the folder to start in (`root`, `path`)."""

    state: ProjectReadinessState
    said: str = Field(max_length=600)
    action: str | None = Field(default=None, max_length=200)
    install: tuple[str, ...] = Field(default=(), max_length=4)
    root: str | None = Field(default=None, max_length=240)
    path: str = Field(default="", max_length=1024)
    repository: str | None = Field(default=None, max_length=512)


class ProjectPrepare(WireRequest):
    """What the one-click fix may need that the project does not hold yet: its repository address."""

    repository: GitUrl | None = None


#: Why a PC could not put a project in place, as a code the server words for the owner (French):
#: `git_missing` git is not installed there; `clone_auth_refused` the repository's host refused the
#: PC's git sign-in or SSH key; `host_unknown` the PC never connected to that SSH host (its key is not
#: known there); `host_unreachable` the host's name does not resolve, or it is not a public host;
#: `repository_not_found` the host says no such repository (or the PC's account cannot see it);
#: `disk_full` too little free space; `timeout` it took too long; `exists` every one of
#: `WorkspaceSpec.candidates` is taken (an existing folder is never touched);
#: `source_missing` the folder to copy is not there; `transfer_failed` the bytes did not arrive whole;
#: `unsafe_archive` the copy names a path no PC writes (absolute, `..`, a link, `.git`, a reserved
#: name); `interrupted` galaius restarted mid-way; `failed` anything else (`detail` says it).
WorkspaceFailureCode = Literal["git_missing", "clone_auth_refused", "host_unknown", "host_unreachable", "repository_not_found", "disk_full",
                               "timeout", "exists", "source_missing", "transfer_failed", "unsafe_archive", "interrupted", "failed"]
#: A clone the PC made from the web, as a launch there found it: fast-forwarded to its remote
#: (`updated`), already there (`current`), left as it was because it holds uncommitted changes or its
#: git settings changed since the clone (`dirty`) or commits its remote lacks (`diverged`), or the
#: remote could not be read (`unreachable`). Its submodules stay as they were: an `updated` clone may
#: hold submodules behind the commits it now names.
WorkspaceRefresh = Literal["updated", "current", "dirty", "diverged", "unreachable"]
#: The most a project copied from one PC to another may hold (its files' bytes, before compression), and its most files.
WORKSPACE_COPY_MAX_BYTES = 1024**3
WORKSPACE_COPY_MAX_FILES = 50_000
#: Most paths a PC looks at when measuring a folder to copy (past it, `WorkspaceSkip.more`).
WORKSPACE_COPY_EXAMINED = 4 * WORKSPACE_COPY_MAX_FILES
#: The largest archive of such a copy: its files, a tar header with its long-name record per file
#: (2 KiB at most for a 1024-character path), and gzip's own framing on data it cannot shrink.
WORKSPACE_ARCHIVE_MAX_BYTES = WORKSPACE_COPY_MAX_BYTES + 2048 * WORKSPACE_COPY_MAX_FILES + 16 * 1024**2
#: One part of a copy's archive on the wire: the source PC uploads it part by part, the target downloads it so.
WORKSPACE_TRANSFER_PART = 8 * 1024 * 1024


#: Why a copy leaves files out: `over_limit` past `WORKSPACE_COPY_MAX_BYTES` / `_FILES`; `link` a link
#: or not a plain file; `credential_store` a key or credential file; `name` a name another system cannot hold.
WorkspaceSkipReason = Literal["over_limit", "link", "credential_store", "name"]
#: How many of the files left out for one reason a copy names.
WORKSPACE_SKIP_EXAMPLES = 20


class WorkspaceSkip(WireModel):
    """Files a copy leaves out for one `reason`, how many, and the first few of them (paths in the
    folder). `more`: the PC stopped looking (a folder of more than `WORKSPACE_COPY_EXAMINED` paths),
    so `count` is a floor."""

    reason: WorkspaceSkipReason
    count: int = Field(ge=1)
    more: bool = False
    examples: tuple[Annotated[str, Field(max_length=1024)], ...] = Field(default=(), max_length=WORKSPACE_SKIP_EXAMPLES)


class WorkspacePack(WireModel):
    """What copying one folder of a PC sends, measured before anything is sent: its files and their
    bytes (within the copy limits, always), whether they are the git checkout's tracked files
    (`tracked`; else the folder walked without hidden names, dependency and build folders), the
    repository it comes from (`origin`, credentials dropped), and what it leaves out (`skipped`)."""

    files: int = Field(ge=0, le=WORKSPACE_COPY_MAX_FILES)
    size: int = Field(ge=0, le=WORKSPACE_COPY_MAX_BYTES)
    tracked: bool
    origin: GitRemote | None = None
    skipped: tuple[WorkspaceSkip, ...] = Field(default=(), max_length=len(get_args(WorkspaceSkipReason)))

    @model_validator(mode="after")
    def one_per_reason(self) -> Self:
        if len({skip.reason for skip in self.skipped}) != len(self.skipped):
            raise ValueError("a copy lists its left-out files once per reason")
        return self


class WorkspaceArchive(WireModel):
    """A folder's copy as it crossed the server: a gzip'd tar of plain files only, `size` bytes with
    sha256 `digest`, held under transfer `transfer`. The source PC uploads it part by part
    (`WORKSPACE_TRANSFER_PART` bytes each, `part_route`), then posts a `WorkspaceUpload` to `route`;
    the target PC downloads the same parts and checks `digest` over them all."""

    transfer: UUID
    size: int = Field(ge=1, le=WORKSPACE_ARCHIVE_MAX_BYTES)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: int = Field(ge=0, le=WORKSPACE_COPY_MAX_FILES)

    @property
    def parts(self) -> int:
        return -(-self.size // WORKSPACE_TRANSFER_PART)

    #: The server's routes for one transfer, under the machine's own token: `PUT` / `GET` one part,
    #: `POST` the `WorkspaceUpload` that ends it.
    ROUTE: ClassVar[str] = "/v1/machine/transfers/{transfer}"

    @classmethod
    def route(cls, transfer: UUID) -> str:
        return cls.ROUTE.format(transfer=transfer)

    @classmethod
    def part_route(cls, transfer: UUID, index: int) -> str:
        return f"{cls.route(transfer)}/{index}"


class WorkspaceFailure(WireModel):
    """Why a copy's upload stopped: `code` the server words for the owner, `detail` the PC's English for the log."""

    code: WorkspaceFailureCode
    detail: str = Field(default="", max_length=600)


class WorkspaceUpload(WireModel):
    """How a copy's upload ended, posted by the source PC to `WorkspaceArchive.route`: the archive
    (every part uploaded), or why not."""

    outcome: WorkspaceArchive | WorkspaceFailure


class MachineWorkspaceJob(WireModel):
    """One project folder a PC was asked to put in place (`source`: a `clone` of `origin`
    (`workspace_prepare`), a `copy` received from another PC or an `empty` folder (`workspace_create`)): `running` (`done` of `total` bytes received for
    a copy), `ready` (a folder agents can start in, under `root`), or `failed` with `code` (the
    server's French words) and `detail` (the PC's own, for the log)."""

    id: UUID
    root: str = Field(max_length=240)
    name: str = Field(max_length=64)
    #: The repository a clone comes from (`host/owner/repo`); "" for a copy or an empty folder.
    origin: str = Field(default="", max_length=512)
    state: Literal["running", "ready", "failed"]
    detail: str = Field(default="", max_length=600)
    started_at: datetime
    finished_at: datetime | None = None
    #: An existing checkout of the repository the PC found and registered as an agent folder (its
    #: owner's own code: its agent settings load), instead of cloning it.
    found: bool = False
    source: Literal["clone", "copy", "empty"] = "clone"
    code: WorkspaceFailureCode | None = None
    done: int | None = Field(default=None, ge=0)
    total: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.code is not None and self.state != "failed":
            raise ValueError("only a failed job carries a failure code")
        if (self.done is not None or self.total is not None) and (self.source != "copy" or self.done is None or self.total is None or self.done > self.total):
            raise ValueError("a copy's progress is `done` of `total` bytes, nothing else's")
        return self


#: An agent root as a request names it (a released runner refuses one longer than 240 characters).
AgentRootPath = Annotated[PlacePath, Field(max_length=240)]


class WorkspaceSpec(WireRequest):
    """A new project folder `name` beneath agent root `root` on one PC. An existing folder is never
    touched: the PC takes the first free one of `candidates(name)`, and its job names the folder it used."""

    root: AgentRootPath
    name: WorkspaceName | None = None
    #: How many names a taken one is tried as (`candidates`).
    TRIES: ClassVar[int] = 99

    @classmethod
    def candidates(cls, name: str) -> Iterator[str]:
        """`name`, then `name-2` ... `name-99`, each a `WorkspaceName` (the stem cut to make room for its suffix)."""
        yield WORKSPACE_NAME.validate_python(name)
        for index in range(2, cls.TRIES + 1):
            suffix = f"-{index}"
            yield WORKSPACE_NAME.validate_python(name[:WORKSPACE_NAME_MAX - len(suffix)] + suffix)


class WorkspacePrepareSpec(WorkspaceSpec):
    """Clone `url` (and its submodules) into a new folder `name` beneath agent root `root`, with
    the PC's own git credentials; `name` defaults to the repository's."""

    url: GitUrl
    submodules: bool = True


class WorkspacePrepareRequest(MachineAgentRequestBase, WorkspacePrepareSpec):
    op: Literal["workspace_prepare"] = "workspace_prepare"
    action: ClassVar[bool] = True
    feature: ClassVar[str] = "workspaces"


class WorkspaceCreateSpec(WorkspaceSpec):
    """Create the project folder `name` beneath agent root `root` from `copy_of`, a folder another
    PC uploaded (`WorkspacePackRequest`), or empty when `copy_of` is None. The PC downloads the
    archive, checks its digest, writes plain files only into a hidden partial folder, then renames it
    into place; `copy_of.size` is the COMPRESSED size, so the PC holds the files it writes to
    `copy_of.files` and their bytes to `WORKSPACE_COPY_MAX_BYTES` as it extracts them."""

    name: WorkspaceName
    copy_of: WorkspaceArchive | None = None


class WorkspaceCreateRequest(MachineAgentRequestBase, WorkspaceCreateSpec):
    op: Literal["workspace_create"] = "workspace_create"
    action: ClassVar[bool] = True
    feature: ClassVar[str] = "workspace_copy"


class WorkspacePackRequest(MachineAgentRequestBase):
    """Measure one folder beneath agent root `root` for a copy to another PC (`WorkspacePack`), and
    with `transfer` set also send it: the PC answers the measure at once and uploads the archive
    in the background (`WorkspaceArchive`), ending with a `WorkspaceUpload`. Past the copy limits,
    the files that fit are sent and the rest listed (`WorkspaceSkip` `over_limit`)."""

    op: Literal["workspace_pack"] = "workspace_pack"
    root: AgentRootPath
    path: FolderPath = ""
    transfer: UUID | None = None
    action: ClassVar[bool] = True
    seconds: ClassVar[float] = 60
    feature: ClassVar[str] = "workspace_copy"


class WorkspacesRequest(MachineAgentRequestBase):
    """The project folders this PC was asked for, newest first."""

    op: Literal["workspaces"] = "workspaces"
    feature: ClassVar[str] = "workspaces"


MachineAgentRequest = Annotated[
    AgentFoldersRequest | AgentRunsRequest | AgentTailRequest | AgentStartRequest | AgentSendRequest | AgentStopRequest
    | AgentOptionsRequest | AgentAnswerRequest | AgentSessionsRequest | AgentContinueRequest | AgentLogsRequest
    | AgentSettingsRequest | AgentProviderSwitchRequest | PlacesRequest | PlaceLevelRequest | PlaceCancelRequest | PlaceBrowseRequest
    | PlaceReviewsRequest | PlaceReviewRequest | PlaceDiscardRequest | WorkspacePrepareRequest | WorkspaceCreateRequest | WorkspacePackRequest | WorkspacesRequest | AgentMediaRequest
    | AgentProgramsRequest | AgentProgramInstallRequest,
    Field(discriminator="op"),
]
MACHINE_AGENT_REQUESTS: TypeAdapter[MachineAgentRequest] = TypeAdapter(MachineAgentRequest)


class AgentInteractionField(WireModel):
    key: str = Field(min_length=1, max_length=160)
    kind: Literal["choice", "text", "boolean"]
    label: str = Field(min_length=1, max_length=500)
    required: bool = True
    options: tuple[str, ...] = Field(default=(), max_length=32)


class AgentInteraction(WireModel):
    """An approval a session is waiting for: a command to run, a file change, a question."""

    id: str = Field(min_length=1, max_length=160)
    kind: Literal["command_approval", "file_change_approval", "user_input", "permission_approval"]
    title: str = Field(min_length=1, max_length=500)
    fields: tuple[AgentInteractionField, ...] = Field(min_length=1, max_length=32)
    disclosure: tuple[str, ...] = Field(default=(), max_length=32)
    #: sha256 of everything above as the machine holds it; an answer carries it back.
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class MachineAgentModel(WireModel):
    provider: str = Field(max_length=40)
    model: str = Field(max_length=256)


#: One catalog row's name, `<catalog provider>/<model id>` (galaius `Model.catalog_id`).
CatalogId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.:@/+-]+$", max_length=256)]


class AgentRanking(WireModel):
    """A model criterion as the server ranks it on its own benchmark board (`GET …/agent-rankings`):
    the catalog ids clearing it, best first. A PC with no board of its own launches from it (a
    board is fetched with its reader's own key); no score leaves the server."""

    criterion: str = Field(max_length=4096)
    weights: str = Field(default="", max_length=1024)
    ranked: tuple[CatalogId, ...] = Field(default=(), max_length=1000)
    ranked_at: datetime


#: Where one agent program stands on a PC, installed from the web (`AgentProgramInstallRequest`):
#: `missing` (not installed, nothing under way), `installing`, `signed_out` (installed, not signed
#: in, no sign-in under way), `signing_in` (its sign-in waiting for the person), `ready` (installed
#: and signed in), `failed` (`AgentProviderState.failure`).
AgentProgramStep = Literal["missing", "installing", "signed_out", "signing_in", "ready", "failed"]
#: Why an install or a sign-in from the web stopped, worded by the server for the owner: the sign-in
#: waited its whole time (`sign_in_expired`), or the program's sign-in ended without signing in.
AgentProgramFailure = Literal["download_failed", "install_failed", "unsupported_system", "sign_in_expired", "sign_in_failed"]
#: A one-time code a program signs in with on its vendor's page (Codex: `N5ST-Z1ZME`).
DEVICE_CODE = r"^[A-Z0-9]{4,12}(-[A-Z0-9]{4,12})?$"


class AgentSignIn(WireModel):
    """A program's sign-in waiting for the person, until `expires_at`. It happens between the person
    and the program's vendor: on the PC's own browser, or, for a program signing in by device code,
    on the vendor's page (a constant the server holds, never a link the PC names) with `code`."""

    code: str | None = Field(default=None, pattern=DEVICE_CODE)
    expires_at: datetime


class AgentProviderState(WireModel):
    """One CLI galaius can drive agents through, on this machine: switched on, installed, and where
    its install and sign-in from the web stand."""

    provider: AgentProvider
    #: The owner lets it run agents here (unmentioned: on).
    active: bool
    #: Its program is installed here.
    available: bool
    step: AgentProgramStep = "missing"
    #: Signed in (its own status command said so); None: not asked yet.
    signed_in: bool | None = None
    version: str = Field(default="", max_length=80)
    #: Who it is signed in as, as its status command says (an e-mail, an organisation), once ready.
    account: str = Field(default="", max_length=200)
    sign_in: AgentSignIn | None = None
    failure: AgentProgramFailure | None = None
    #: The program's or installer's last line, for the log (English, never a secret).
    detail: str = Field(default="", max_length=400)
    changed_at: datetime | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.step == "failed") != (self.failure is not None):
            raise ValueError("a failed program names its failure, and only a failed one does")
        if self.sign_in is not None and self.step != "signing_in":
            raise ValueError("only a sign-in under way waits for the person")
        return self


#: galaius's own tools that pick a model by rule: screenshots and images, UI element detection,
#: video, audio, and the quick (low / medium) review tier.
ToolModelRole = Literal["image", "component", "video", "audio", "sovereign"]


class ToolRoleModels(WireModel):
    """What one tool's rule resolves to on this machine, best first; `reason` says why nothing (or
    nothing stronger) clears it here."""

    role: ToolModelRole
    criterion: str = Field(max_length=4096)
    #: The owner wrote this rule; False: the built-in default is in force.
    configured: bool
    models: tuple[MachineAgentModel, ...] = Field(default=(), max_length=5)
    reason: str = Field(default="", max_length=600)


class MachineAgentSession(WireModel):
    """One of the owner's own editor conversations on this machine."""

    session_id: UUID
    provider: str = Field(max_length=40)
    title: str = Field(default="", max_length=400)
    last: str = Field(default="", max_length=400)
    root: str = Field(max_length=240)
    path: str = Field(default="", max_length=1024)
    updated_at: float = Field(ge=0)
    #: Open in the editor right now (continuing makes a copy; the editor's is never written to).
    live: bool = False


class PassedOverCandidate(WireModel):
    """A (CLI, model) the machine's launcher ranked ABOVE the one a run started on, and why it did
    not start there: `reason` is the launcher's stable code (`quota_exceeded`, `cli_missing`,
    `unauthenticated`...), `until` when a quota refusal clears (epoch seconds; None when the
    reason names no instant). What lets the web say « Claude indisponible jusqu'à 22:01, lancé
    avec Codex » instead of showing a Codex run as if Codex had been the choice."""

    provider: str = Field(max_length=40)
    model: str = Field(max_length=120)
    reason: str = Field(max_length=40)
    until: float | None = Field(default=None, ge=0)


class MachineAgentRun(WireModel):
    """One run the machine started for the web, as the machine's registry has it now."""

    run_id: UUID
    name: str = Field(max_length=120)
    role: str | None = Field(default=None, max_length=80)
    provider: str = Field(max_length=40)
    model: str | None = Field(default=None, max_length=120)
    status: str = Field(max_length=40)
    root: str = Field(max_length=240)
    path: str = Field(default="", max_length=1024)
    task: str = Field(default="", max_length=8000)
    last: str = Field(default="", max_length=400)
    started_at: float = Field(ge=0)
    finished_at: float | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    #: Set for a run a web-started run launched itself (its tree's root is `parent`).
    parent_run_id: UUID | None = None
    kind: AgentRunKind = "agent"
    #: Approvals this session waits for, oldest first.
    pending: tuple[AgentInteraction, ...] = Field(default=(), max_length=16)
    #: Started inside the PC's OS fence built from its levels (None: not recorded).
    fenced: bool | None = None
    #: The ranked candidates passed over before `provider`/`model` started, best first (a client
    #: older than this field sends none).
    passed_over: tuple[PassedOverCandidate, ...] = Field(default=(), max_length=16)


class WebLoad(WireModel):
    """A PC's conversations started from the web working now (`working`) against how many may at once (`limit`):
    `full`, it refuses one more start until one ends."""

    working: int = Field(ge=0)
    limit: int = Field(ge=1)

    @property
    def full(self) -> bool:
        return self.working >= self.limit


class MachineAgentAnswer(WireModel):
    """The machine's answer to one MachineAgentRequest: `folders` a folder listing (`roots` and the
    startable `roles` when no root was named) plus the permission agents start with; `runs` the runs; `tail` raw stream lines
    after the cursor; `start` the new run id; `send` / `stop` a one-line outcome in `detail`."""

    request_id: UUID
    error: str | None = Field(default=None, max_length=400)
    roots: tuple[str, ...] = Field(default=(), max_length=32)
    entries: tuple[MachineFileEntry, ...] = Field(default=(), max_length=500)
    truncated: bool = False
    permission: AgentTouchScope | None = None
    #: The roles this machine's launcher can start (its active catalog), with the roots listing.
    roles: tuple[str, ...] = Field(default=(), max_length=500)
    #: `options`: models an agent can be started on, per CLI, best first; a session's route.
    models: tuple[MachineAgentModel, ...] = Field(default=(), max_length=500)
    session_models: tuple[str, ...] = Field(default=(), max_length=200)
    session_reason: str = Field(default="", max_length=500)
    sessions: tuple[MachineAgentSession, ...] = Field(default=(), max_length=100)
    runs: tuple[MachineAgentRun, ...] = Field(default=(), max_length=200)
    #: With `runs`: its conversations started from the web working now and its maximum (None: a PC too old to say).
    web: WebLoad | None = None
    run_id: UUID | None = None
    lines: tuple[str, ...] = Field(default=(), max_length=4000)
    cursor: int | None = Field(default=None, ge=0)
    detail: str = Field(default="", max_length=400)
    #: `settings` / `programs` / `program_install`: the CLIs agents may run through here (and, for
    #: `settings`, each tool's model here).
    providers: tuple[AgentProviderState, ...] = Field(default=(), max_length=16)
    tool_models: tuple[ToolRoleModels, ...] = Field(default=(), max_length=16)
    #: `places` / `place_level` / `place_cancel`: the PC's levels after the request; `change`: the
    #: widening now waiting on the PC (None: applied at once).
    places: MachinePlacesView | None = None
    change: MachinePlaceChange | None = None
    #: `place_browse`: one page of names (`cursor`: where the next page starts, None: last page).
    browse: tuple[MachinePlaceEntry, ...] = Field(default=(), max_length=500)
    #: `place_reviews` / `place_review` (its diff in `lines`).
    reviews: tuple[MachinePlaceReview, ...] = Field(default=(), max_length=100)
    #: `workspaces` / `workspace_prepare`: the project folders asked of this PC, newest first.
    workspaces: tuple[MachineWorkspaceJob, ...] = Field(default=(), max_length=50)
    #: `folders` naming a folder: the repository that folder is a checkout of (credentials dropped).
    origin: GitRemote | None = None
    #: `workspace_pack`: what copying the folder sends.
    pack: WorkspacePack | None = None
    #: A refusal the server words for the owner (with `error`, the PC's English for the log).
    code: WorkspaceFailureCode | None = None
    #: `start` in a project folder the PC cloned from the web: whether it was fast-forwarded to its
    #: remote first (`WorkspaceRefresh`; None: not such a folder).
    refresh: WorkspaceRefresh | None = None
    #: `start`: the agent runs inside the PC's OS fence (False: it can read every file its user can).
    fenced: bool | None = None
    #: A place change (`place_level` / `place_cancel` / `place_discard`): sha256 of what changed, as the
    #: PC's own log keeps it; the server's audit keeps only this, never the path.
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    #: `media`: the image asked for.
    media: AgentMedia | None = None


def _port_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "node"
    return slug if slug[0].isalpha() else f"n_{slug}"


class NodeLibraryDefinition(WireModel):
    """One reusable node: a configured node, or a selection of nodes with the wires between them.
    Its BOUNDARY — inner input ports no inner edge feeds, inner output ports no inner edge reads —
    is the port list every node referencing it (`SubgraphImplementation` with a `NodeLibraryRef`)
    carries. Positions are relative to the reference node's own position."""

    id: UUID
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=400)
    nodes: tuple[WorkflowNode, ...] = Field(min_length=1, max_length=50)
    edges: tuple[WorkflowEdge, ...] = Field(default=(), max_length=150)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def wired_inside(self) -> Self:
        ids = {node.id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("library node identifiers must be unique")
        if any(node.impl.kind == "subgraph" and isinstance(node.impl.ref, NodeLibraryRef) for node in self.nodes):
            raise ValueError("a reusable node cannot hold another reusable node")
        if any(edge.source.node not in ids or edge.target.node not in ids for edge in self.edges):
            raise ValueError("library edges must stay inside the definition")
        return self

    def boundary(self) -> tuple[tuple[PortSpec, PortAddress], ...]:
        """Every boundary port, named uniquely across the definition (the inner port's name, then
        `<node slug>_<name>`, then a counter), beside the inner address it stands for."""
        fed = {edge.target for edge in self.edges}
        read = {edge.source for edge in self.edges}
        taken: set[str] = set()
        result: list[tuple[PortSpec, PortAddress]] = []
        for node in self.nodes:
            for port in node.ports:
                address = PortAddress(node=node.id, port=port.name)
                if (port.direction == "input" and address in fed) or (port.direction == "output" and address in read):
                    continue
                candidates = [port.name, f"{_port_slug(node.label)}_{port.name}"]
                name = next((candidate for candidate in candidates if candidate not in taken), None)
                if name is None:
                    counter = 2
                    while f"{candidates[1]}_{counter}" in taken:
                        counter += 1
                    name = f"{candidates[1]}_{counter}"
                taken.add(name)
                # An inner input holding a constant has its default: the boundary port is optional
                # (a wire into the reusable node, when present, still wins).
                optional = port.direction == "input" and node.constant(port.name) is not None
                result.append((port.model_copy(update={"name": name[:80], **({"required": False} if optional else {})}), address))
        return tuple(result)

    def ports(self) -> tuple[PortSpec, ...]:
        return tuple(port for port, _address in self.boundary())


class NodeLibraryUse(WireModel):
    workflow: WorkflowKey
    name: str = Field(min_length=1, max_length=120)


class NodeLibraryEntry(WireModel):
    definition: NodeLibraryDefinition
    used_in: tuple[NodeLibraryUse, ...] = ()


class WorkflowBlockAvailability(WireModel):
    """One block the editor's palette offers: the node it places (implementation, ports, default
    config, placement) and whether it runs as placed. The editor copies these four fields onto a
    new `WorkflowNode`; nothing is per kind."""

    impl: Implementation
    name: str = Field(min_length=1, max_length=120)
    ports: tuple[PortSpec, ...] = Field(max_length=64)
    config: dict[str, WorkflowValue] = Field(default_factory=dict, max_length=32)
    placement: Placement = Field(default_factory=Placement)
    readiness: Literal["executable", "config_required", "unavailable"]
    required_config_fields: tuple[str, ...] = Field(default=(), max_length=32)
    reason: str = Field(min_length=1, max_length=400)
    #: Set for every kind that can reach a named provider or run somewhere (agent, model, function,
    #: script, connector) — `None` only when it is genuinely undecidable ahead of time (a
    #: criteria-routed agent, whose route is resolved per run), never for "builtin" or "subgraph"
    #: (a pure server transform has no vendor to name; a subgraph's figure is its contents',
    #: computed by expanding it, never guessed at the collapsed node).
    sovereignty: Sovereignty | None = None
    #: What a search finds it under; a builtin's comes from its spec, every other kind's from its kind.
    category: BlockCategory | None = None
    #: What it does (a builtin's spec summary), and the words a search also matches.
    summary: str = Field(default="", max_length=400)
    summary_fr: str = Field(default="", max_length=400)
    #: A builtin's name in French ("" for kinds named by their own record).
    name_fr: str = Field(default="", max_length=120)
    #: Set when its kind is plumbing (`Plumbing`): the editor folds it into the wire or step it serves.
    plumbing: Plumbing | None = None
    keywords: tuple[str, ...] = Field(default=(), max_length=32)
    #: JSON Schema of the node's settings (`config`), rendered by the editor's one generic form;
    #: empty when the node has none or its kind edits them elsewhere.
    config_schema: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def described(cls, data: object) -> object:
        """Category, summary, search words and settings schema follow the implementation, each
        unless given (the implementation read once here, whatever form it arrived in)."""
        if not isinstance(data, dict) or "impl" not in data:
            return data
        impl = _IMPLEMENTATIONS.validate_python(data["impl"])
        return {**impl.described(), **{key: value for key, value in data.items() if value is not None}, "impl": impl}

    @model_validator(mode="after")
    def coherent_source(self) -> Self:
        signature = self.impl.signature(self.placement.target, self.config)
        if signature is not None and self.ports != signature:
            raise ValueError("catalog ports must be its implementation's signature")
        if self.sovereignty is not None and self.impl.kind in ("builtin", "subgraph"):
            raise ValueError("sovereignty does not apply to builtin or subgraph blocks")
        if len({port.name for port in self.ports}) != len(self.ports):
            raise ValueError("catalog ports must have unique names")
        return self


class NodeSignatureRequest(WireRequest):
    """The ports a node of `impl` takes under `config` — what the editor asks after a setting that
    shapes them changes (a switch's cases). Invalid settings are refused by name."""

    impl: Implementation
    config: dict[str, WorkflowValue] = Field(default_factory=dict, max_length=32)
    placement: Placement = Field(default_factory=Placement)

    @model_validator(mode="after")
    def valid_settings(self) -> Self:
        self.impl.check(self.config)
        return self

    @property
    def ports(self) -> tuple[PortSpec, ...] | None:
        """None: the editor chooses them (an input, a tool, a subgraph's interface)."""
        return self.impl.signature(self.placement.target, self.config)


class ModelProperty(WireModel):
    """One fact a model can be constrained or ranked by. `choice` properties take a word from
    `values` (a sovereignty tier, where it runs) through a membership clause."""

    name: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=1, max_length=400)
    source: str = Field(max_length=160)
    kind: Literal["flag", "number", "choice"]
    weightable: bool
    rankable: bool = False
    percentile: bool
    values: tuple[str, ...] = Field(default=(), max_length=32)
    unit: str | None = Field(default=None, max_length=40)
    higher_is_better: bool | None = None
    url: HttpUrl | None = None
    #: A short human name ("top3d.ai geometry"), when the source publishes one.
    label: str | None = Field(default=None, max_length=120)


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


class CanvasPoint(WireModel):
    x: float = Field(ge=-1e7, le=1e7)
    y: float = Field(ge=-1e7, le=1e7)


class CanvasGroupPort(WireModel):
    """A port a collapsed group shows on its edge, standing for a member's port."""

    name: str = Field(min_length=1, max_length=80)
    direction: Literal["input", "output"]
    target: PortAddress | None = None


class CanvasGroup(CanvasPoint):
    """A frame drawn around steps on the board: its members move and fold with it."""

    id: str = Field(min_length=1, max_length=120)
    label: str = Field(default="", max_length=120)
    width: float = Field(gt=0, le=1e6)
    height: float = Field(gt=0, le=1e6)
    collapsed: bool = False
    ports: tuple[CanvasGroupPort, ...] = Field(default=(), max_length=128)


class CanvasNote(CanvasPoint):
    """A note written on the board; it never runs."""

    id: str = Field(min_length=1, max_length=120)
    text: str = Field(default="", max_length=10_000)


class WorkflowCanvas(WireModel):
    """How the editor draws the board beside the graph it runs — frames around steps (`groups`,
    with `members`: node id -> group id), where the run block and each trigger sit (`placements`,
    keyed by the editor's own ids), and notes. Saved with the revision so every browser and device
    shows the same board; the engine never reads it."""

    groups: tuple[CanvasGroup, ...] = Field(default=(), max_length=500)
    members: dict[str, str] = Field(default_factory=dict, max_length=5000)
    placements: dict[str, CanvasPoint] = Field(default_factory=dict, max_length=1000)
    notes: tuple[CanvasNote, ...] = Field(default=(), max_length=500)

    @model_validator(mode="after")
    def known_groups(self) -> Self:
        ids = [group.id for group in self.groups]
        if len(set(ids)) != len(ids):
            raise ValueError("canvas group identifiers must be unique")
        if not set(self.members.values()) <= set(ids):
            raise ValueError("a canvas member names a group the canvas does not have")
        return self


class WorkflowRevision(WireModel):
    key: WorkflowKey
    revision: UUID
    parent_revision: UUID | None = None
    name: str = Field(min_length=1, max_length=120)
    nodes: tuple[WorkflowNode, ...] = Field(min_length=1, max_length=500)
    edges: tuple[WorkflowEdge, ...] = Field(max_length=1500)
    interface: WorkflowInterface
    created_at: datetime
    #: The board as drawn (groups, notes, run block / trigger places); absent on revisions saved
    #: before it existed and on those no editor drew.
    canvas: WorkflowCanvas | None = None

    @model_validator(mode="after")
    def unique_nodes(self) -> Self:
        if len({node.id for node in self.nodes}) != len(self.nodes):
            raise ValueError("workflow node identifiers must be unique")
        return self


#: Model providers reached over the OpenAI HTTP API at the connection's own base URL (`endpoint`,
#: e.g. `https://host/v1`): their model list is `GET {endpoint}/models`, their text is
#: `POST {endpoint}/chat/completions`. Hosted OpenAI itself is one; the other two are any server
#: speaking that API — the owner's own (`self_hosted`, sovereign) or anyone's (`openai_compatible`,
#: sovereignty unknown until graded).
OPENAI_WIRE_PROVIDERS: frozenset[str] = frozenset({"openai", "self_hosted", "openai_compatible"})
#: Of those, the ones whose server may need no key at all.
OPENAI_WIRE_KEYLESS: frozenset[str] = frozenset({"self_hosted", "openai_compatible"})


class ConnectionResource(WireModel):
    id: UUID
    revision: UUID
    kind: Literal["workspace_storage", "http_server", "provider_api", "service_connector", "ssh_server", "object_storage", "mail_server", "webhook"]
    name: str = Field(min_length=1, max_length=120)
    #: Reused per kind (documented at each validator branch below): workspace storage's folder,
    #: OR — for `mail_server` only — the `smtp://host:port` send endpoint, alongside `endpoint`
    #: holding the `imap://host:port` read endpoint.
    root: str | None = Field(default=None, max_length=1024)
    endpoint: str | None = Field(default=None, max_length=2048)
    credential: CredentialRef | None = None
    provider: Literal["openai", "anthropic", "gemini", "fal", "replicate", "roboflow", "mistral", "elevenlabs", "self_hosted", "openai_compatible", "google_drive", "github", "slack", "notion", "gmail", "sharepoint", "onedrive", "s3_compatible", "azure_blob", "discord", "telegram", "whatsapp", "gitlab", "discord_webhook", "teams_webhook"] | None = None
    models: tuple[str, ...] = Field(default=(), max_length=256)
    capabilities: tuple[Literal["read", "write", "list", "http", "command"], ...]
    credential_expires_at: datetime | None = None
    #: SSH login name, the object-storage access-key-id / storage-account name, a mail server's
    #: login, or a WhatsApp Business phone_number_id — the one identity string every non-OAuth
    #: remote credential needs alongside its secret.
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
        # A service connector's grantable verbs generalized beyond read-only browsing: "list"
        # covers every read/browse action (unchanged), "write" now also covers a messaging send or
        # a git write operation — the same owner-approval ledger SSH and object storage writes use.
        if self.kind == "service_connector" and (self.endpoint is not None or self.root is not None or self.provider is None or self.credential is None or not self.capabilities or set(self.capabilities) - {"list", "write"}):
            raise ValueError("service connector requires a credential, provider, and at least one of list/write")
        # A self-hosted endpoint is the owner's own machine, typically on a trusted/private
        # network with no vendor credential to hold — unlike openai/anthropic/gemini, which
        # always require one. Any other OpenAI-compatible server (`openai_compatible`: a vLLM or
        # Ollama box, OpenRouter, Groq, Together...) may or may not want a key.
        if self.kind == "provider_api" and self.provider not in OPENAI_WIRE_KEYLESS and self.credential is None:
            raise ValueError("provider connection requires a credential reference")
        if self.kind not in {"provider_api", "service_connector", "object_storage", "webhook"} and self.provider is not None:
            raise ValueError("provider kind is required only for provider, service, object-storage and webhook connections")
        if self.kind == "provider_api" and self.provider not in {"openai", "anthropic", "gemini", "fal", "replicate", "roboflow", "mistral", "elevenlabs", "self_hosted", "openai_compatible"}:
            raise ValueError("provider API kind requires a model provider")
        if self.kind not in {"service_connector", "ssh_server", "object_storage"} and self.credential_expires_at is not None:
            raise ValueError("credential expiry belongs only to connections with vendor-issued expiry")
        if self.kind == "service_connector" and self.provider not in {"google_drive", "github", "slack", "notion", "gmail", "sharepoint", "onedrive", "discord", "telegram", "whatsapp", "gitlab"}:
            raise ValueError("service connector provider is unsupported")
        if self.kind != "provider_api" and self.models:
            raise ValueError("only provider connections declare models")
        if len(set(self.models)) != len(self.models) or any(not model or len(model) > 160 for model in self.models):
            raise ValueError("provider models must be unique bounded identifiers")
        if self.kind == "ssh_server" and (self.endpoint is None or not self.endpoint.startswith("ssh://") or self.root is not None or self.credential is None or self.username is None or not self.capabilities or set(self.capabilities) - {"command", "read", "write", "list"}):
            raise ValueError("SSH connection requires an ssh:// endpoint, credential, username and at least one of command/read/write/list")
        if self.kind == "object_storage" and (self.endpoint is None or self.root is not None or self.provider not in {"s3_compatible", "azure_blob"} or self.credential is None or self.username is None or not self.capabilities or set(self.capabilities) - {"read", "write", "list"}):
            raise ValueError("object storage connection requires an endpoint, s3_compatible or azure_blob provider, credential, username and at least one of read/write/list")
        # Two host:port pairs on one connection: `endpoint` reads (IMAP), `root` sends (SMTP) — the
        # same generic-field-reuse-per-kind convention `root`/`endpoint` already carry elsewhere on
        # this model, never a config blob.
        if self.kind == "mail_server" and (self.endpoint is None or not self.endpoint.startswith("imap://") or self.root is None or not self.root.startswith("smtp://") or self.provider is not None or self.credential is None or self.username is None or not self.capabilities or set(self.capabilities) - {"read", "write"}):
            raise ValueError("mail connection requires an imap:// endpoint, an smtp:// root, credential, username and at least one of read/write")
        if self.kind == "webhook" and (self.endpoint is None or not self.endpoint.startswith("https://") or self.root is not None or self.username is not None or self.provider not in {None, "discord_webhook", "teams_webhook"} or self.capabilities != ("write",)):
            raise ValueError("webhook connection requires an https:// endpoint and write capability only")
        if self.kind != "ssh_server" and self.host_key_fingerprint is not None:
            raise ValueError("host key pinning belongs only to SSH connections")
        if self.kind != "object_storage" and self.region is not None:
            raise ValueError("region belongs only to object storage connections")
        if self.kind not in {"ssh_server", "object_storage", "mail_server"} and self.username is not None:
            raise ValueError("username belongs only to SSH, object storage and mail connections")
        return self


class RunInitiator(WireModel):
    """Who started a run. `account` is the person it acts for: the signed-in account itself, the
    account that created the workspace API key, or the account that saved the trigger - None when
    no person can be named (a key or trigger older than this record). A machine checks `account`
    against the people its owner lets run file nodes on it."""

    kind: Literal["account", "api_key", "trigger"]
    account: UUID | None = None
    api_key: UUID | None = None
    trigger: UUID | None = None


#: What a run is doing when no step result says so. The run's own stage (`WorkflowRun.phase`):
#: `queued` (its record exists, it waits for a run slot: `ahead` runs of its workspace hold them),
#: `preparing` (the server resolves its agents, tools and models before the first step),
#: `running` (steps execute). A step waiting on something outside the server (`WorkflowRun.waits`):
#: `waiting_for_machine` (sent to a machine that has not started it), `running_on_machine`,
#: `installing` / `loading_model` / `running_model` (the machine says so), `waiting_for_model` (a
#: model or agent step waits on its provider's answer), `pausing` (a Wait node, until `until`),
#: `retrying` (the step failed and its next attempt starts at `until`), `waiting_for_approval`.
RunPhaseKind = Literal["queued", "preparing", "running", "waiting_for_machine", "running_on_machine", "installing", "loading_model", "running_model", "waiting_for_model", "pausing", "retrying", "waiting_for_approval"]
#: The kinds a machine runner may report for the step it runs (`MachineEvent` progress payload
#: `{"kind": "phase", "phase": <kind>, "detail": <text>}`).
MACHINE_PHASES: tuple[RunPhaseKind, ...] = ("installing", "loading_model", "running_model")


class RunPhase(WireModel):
    """Why a run (or one of its steps) is where it is, since when. `reason` is the sentence a
    reader is shown ("Queued behind 3 runs of this workspace (8 run at once)"); the other fields
    are the same facts, typed, for a client that words or links them itself."""

    kind: RunPhaseKind
    since: datetime
    reason: str = Field(max_length=300)
    node_id: UUID | None = None
    machine_id: UUID | None = None
    machine_name: str | None = Field(default=None, max_length=200)
    #: `queued`: runs of this workspace holding or waiting for a slot when this one began waiting.
    ahead: int | None = Field(default=None, ge=0)
    #: `retrying`: when the next attempt starts; `waiting_for_machine` on a machine not connected:
    #: when the step stops waiting for it and fails.
    until: datetime | None = None


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
    #: What this run actually cost, per node and in total — set once the run leaves "running";
    #: `None` on a run still queued/running, or one this server version never metered.
    cost: RunCostActual | None = None
    #: Node failures the run survived (routed to an error path or replaced by defaults), in order.
    recovered: tuple[NodeError, ...] = Field(default=(), max_length=500)
    #: Who started it (None on a run recorded before initiators were).
    initiator: RunInitiator | None = None
    #: Each exposed output the run did not produce, with why (the path feeding it was not taken):
    #: a run that succeeded with an absent output says so, instead of a bare `None` result.
    skipped_outputs: dict[str, str] = Field(default_factory=dict, max_length=64)
    #: The run's stage now; None once it ended (or recorded before phases were).
    phase: RunPhase | None = None
    #: Each step waiting now on something outside the server, oldest first. The timeline: the
    #: run's `queued` / `started` events begin those phases; every other change to `phase` or
    #: `waits` is a `progress` event whose payload is a `PhaseEvent`.
    waits: tuple[RunPhase, ...] = Field(default=(), max_length=64)


class PhaseEvent(WireModel):
    """A run `progress` event's payload when `phase` or `waits` changed: the phase that began, or
    (`ended`) the step wait that ended, at the event's timestamp. A step's new wait supersedes its
    previous one (no `ended` for that one): a wait lasts until its node's next phase event."""

    type: Literal["phase"] = "phase"
    phase: RunPhase
    ended: bool = False


class WorkflowEvent(WireModel):
    run_id: UUID
    sequence: int = Field(ge=1)
    node_path: tuple[UUID, ...] = ()
    kind: Literal["queued", "started", "progress", "result", "error", "cancelled", "interrupted"]
    timestamp: datetime
    payload: dict[str, object] = Field(default_factory=dict)


#: Bounds a machine-built thumbnail: the runner (an enrolled machine reading or producing an
#: image/mask/video file that never leaves it) and the server (re-serving it on a step event)
#: both hold to these — the runner encodes to fit them, the server never re-encodes.
VALUE_PREVIEW_MAX_PIXELS = 256
VALUE_PREVIEW_MAX_BYTES = 64 * 1024


class ValuePreview(WireModel):
    """One step event's `preview` payload: a small, JSON-safe glimpse of a node's result or of
    one input it read — never the full value (a file's bytes stay on the machine that holds it,
    or behind the run's own activity content store). The ONE shape a machine runner's result and
    the server's step executor both hold to for an image/mask/video thumbnail, so neither drifts
    from the other's field names or size bounds."""

    kind: Literal["text", "number", "boolean", "json", "list", "artifact", "image", "empty"]
    text: str = Field(max_length=400)
    items: int | None = None
    media_type: str | None = None
    path: str | None = Field(default=None, max_length=512)
    #: Base64 image bytes, verbatim. A NEW preview keeps to VALUE_PREVIEW_MAX_PIXELS /
    #: VALUE_PREVIEW_MAX_BYTES at the producer (the runner encodes to fit, never this field —
    #: an older, differently-sized preview already reaches this same shape); unbounded here, so
    #: this type never rejects a value its own producer already promised to bound. Set only for
    #: `kind == "image"`.
    image: str | None = None


StepStatus = Literal["started", "retrying", "succeeded", "failed", "skipped"]


class StepEvent(WireModel):
    """One node step's progress, the payload of a run's step `WorkflowEvent` (`type == "step"`).
    `failed` + `handled` = the node failed and its policy kept the run going (`route`: the error
    path ran, `continue_with_default`: its defaults flowed on); `retrying` = an attempt failed and
    another starts in `retry_in_seconds`; `skipped` = the node did not run because a path it waits
    on was not taken (`skipped_because` names it). `called_by` = the agent node that ran this node
    as a tool."""

    type: Literal["step"] = "step"
    node_id: UUID
    status: StepStatus
    duration_ms: int | None = Field(default=None, ge=0)
    preview: ValuePreview | None = None
    error: str | None = Field(default=None, max_length=400)
    failure: NodeError | None = None
    handled: Literal["route", "continue_with_default"] | None = None
    attempt: int | None = Field(default=None, ge=1)
    retry_in_seconds: float | None = Field(default=None, ge=0)
    skipped_because: str | None = Field(default=None, max_length=400)
    called_by: UUID | None = None
    cost: NodeCostActual | None = None
    #: The model a model choice resolved to for this step (the server's `ModelPin`).
    model: dict[str, object] | None = None


class ProviderUsage(WireModel):
    """One provider response's tokens in ONE additive shape, whatever the vendor — normalized at
    the edge that parses the response: `input_tokens` = prompt tokens neither read from nor
    written to a cache, the two cache fields = the cached rest of the prompt, `output_tokens`
    includes reasoning. Anthropic reports this shape natively; OpenAI and Gemini count cached
    tokens INSIDE their prompt figure, so they are carved out on the way in, never counted twice."""

    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)

    @property
    def prompt_tokens(self) -> int:
        """The whole prompt the model saw: uncached + cache reads + cache writes."""
        return self.input_tokens + (self.cache_read_input_tokens or 0) + (self.cache_creation_input_tokens or 0)

    @property
    def counted_tokens(self) -> int:
        """Prompt + output from the normalized fields — what every total sums. The vendor's own
        `total_tokens` is kept verbatim but never summed: OpenAI's includes cached tokens, Gemini's
        includes thinking, Anthropic sends none."""
        return self.prompt_tokens + self.output_tokens

    def metered(self) -> NodeUsage:
        """The billable units, each cache kind at its own rate."""
        return NodeUsage(token_in=self.input_tokens, token_cache_read=self.cache_read_input_tokens, token_cache_write=self.cache_creation_input_tokens, token_out=self.output_tokens)

    @classmethod
    def from_metered(cls, usage: NodeUsage) -> Self | None:
        """The token half of a node's metered usage; None for a node not priced per token."""
        if usage.token_in is None and usage.token_out is None:
            return None
        return cls(input_tokens=usage.token_in or 0, output_tokens=usage.token_out or 0, cache_read_input_tokens=usage.token_cache_read, cache_creation_input_tokens=usage.token_cache_write)

    @classmethod
    def combined(cls, usages: Iterable[Self]) -> Self:
        """Several responses (an agent's turns) as one; a cache field stays None when no response
        reported it, never a synthetic zero."""
        usages = tuple(usages)

        def total(field: str) -> int | None:
            reported = [value for usage in usages if (value := getattr(usage, field)) is not None]
            return sum(reported) if reported else None

        return cls(input_tokens=sum(usage.input_tokens for usage in usages), output_tokens=sum(usage.output_tokens for usage in usages),
                   total_tokens=sum(usage.counted_tokens for usage in usages),
                   cache_creation_input_tokens=total("cache_creation_input_tokens"), cache_read_input_tokens=total("cache_read_input_tokens"))


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


class ConversationCreateRequest(WireRequest):
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


class ConversationAppendRequest(WireRequest):
    expected_revision: int = Field(ge=0)
    input: str = Field(min_length=1, max_length=1 << 20)
    idempotency_key: str = Field(min_length=1, max_length=160)
    max_output_tokens: int = Field(default=2048, ge=1, le=32768)


class ChatRun(WireModel):
    """A chat as the run feed lists it beside workflow runs (`GET /runs` `chats`), so every screen
    that says who is working reads the same record."""

    id: UUID
    origin: Literal["chat"] = "chat"
    title: str = Field(max_length=120)
    agent: AgentRevisionRef | None = None
    state: Literal["queued", "running", "succeeded", "failed", "cancelled", "unknown_outcome"]
    started_at: datetime
    updated_at: datetime


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
    #: The workspace's monthly cost ceiling in USD, across every metered node kind (not only
    #: LLM tokens) — `None` means no ceiling. A run whose estimate would cross it is refused
    #: until the caller confirms it explicitly (`BudgetOverrun`).
    monthly_cost_usd_limit: float | None = Field(default=None, ge=0)


class UsageSummary(WireModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    provider_calls: int = Field(ge=0)
    period_start: datetime
