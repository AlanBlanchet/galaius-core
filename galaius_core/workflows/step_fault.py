"""Why a step failed, typed: what its provider answered (a refusal, no answer, an answer nobody can read, an address the
server will not follow), what it sent back (nothing, or cut at its output limit), what the step lacks to run, why no
machine took it, or what stopped the run on the server. Its reader's words come from its `code` and figures; `message` is the same fact in English,
for a log only, said once here."""

from collections.abc import Mapping
from datetime import date
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, ValidationInfo, computed_field, field_validator, model_validator

from galaius_core.models.criteria import CriteriaName, PropertyKind
from galaius_core.wire import Cause, ProviderId
from galaius_core.models.task import ModelTask
from galaius_core.machine.features import UPDATE_COMMAND, PcFeature
from galaius_core.machine.need import InstanceTypeName, MachineName, RegionName, ResourceRequirement, ScriptLanguage


class StepFault(Cause):
    """One way a step failed: its provider, its machine, its input or the server. A kind states its English once (`SAID`,
    a template over its own fields); `detail` is the outside party's own words (a provider, a machine, a store),
    verbatim, when it wrote any, and `vendor_code` its own code for the failure (an error code, a stop or block reason),
    when it gave one: never a sentence of ours."""

    #: By default a provider failed the call and the server is its gateway; each kind declares its own.
    status: ClassVar[int] = 502
    SAID: ClassVar[str]
    detail: str | None = Field(default=None, max_length=300)
    vendor_code: str | None = Field(default=None, max_length=60)

    @field_validator("detail", "vendor_code", mode="before")
    @classmethod
    def _clipped(cls, value: object, info: ValidationInfo) -> object:
        """A provider's words run to its own length: kept to the field's ceiling, never refused for it."""
        limit = next((rule.max_length for rule in cls.model_fields[info.field_name].metadata if hasattr(rule, "max_length")), None)
        return value[:limit] if isinstance(value, str) and limit is not None else value

    @computed_field
    @property
    def message(self) -> str:
        return f"{self.said}: {self.detail}" if self.detail else self.said


class ReasonedFault(StepFault):
    """A fault whose `reason` (a closed set, declared by each kind) selects its sentence: `SAID` holds one per reason."""

    SAID: ClassVar[Mapping[str, str]]
    reason: str

    @property
    def said(self) -> str:
        return self.SAID[self.reason].format_map(vars(self))


#: What a provider's refusal asks of its reader: a key to replace or to widen, a model id to fix, an account to fund, a
#: wait, or nothing he can fix (`failed`; `unfinished` its job ended without the result).
VendorReason = Literal["key_refused", "key_forbidden", "model_unknown", "no_funds", "rate_limited", "unfinished", "failed"]


class VendorRefused(ReasonedFault):
    """The provider refused the call: `vendor_status` its HTTP answer, when it gave one (None: its job reported the
    failure itself)."""

    code: Literal["vendor_refused"] = "vendor_refused"
    reason: VendorReason
    vendor_status: int | None = None
    #: The provider whose Galaius key the call went on, not the organisation's: its reader is told whose key, never the
    #: vendor's words (`detail` stays empty; the server log keeps them).
    platform_provider: ProviderId | None = None

    #: What each HTTP status means for the reader; any other status is a failure he cannot fix (`failed`).
    BY_STATUS: ClassVar[Mapping[int, VendorReason]] = {401: "key_refused", 403: "key_forbidden", 404: "model_unknown", 402: "no_funds", 429: "rate_limited"}
    #: A vendor's own error code, read before its status (a vendor answers an empty account 401 or 429): OpenAI
    #: (platform.openai.com/docs/guides/error-codes), ElevenLabs (elevenlabs.io error docs, read 2026-10-03).
    CODES: ClassVar[Mapping[VendorReason, tuple[str, ...]]] = {
        "key_refused": ("invalid_api_key",), "key_forbidden": ("missing_permissions", "needs_authorization"),
        "no_funds": ("quota_exceeded", "insufficient_quota"), "model_unknown": ("model_not_found",)}
    SAID: ClassVar[Mapping[str, str]] = {
        "key_refused": "the API key was refused", "key_forbidden": "the API key is not allowed to use this model",
        "model_unknown": "the provider does not know this model", "no_funds": "the provider account has no funds left",
        "rate_limited": "the provider's quota or rate limit was hit", "unfinished": "the provider could not finish the job",
        "failed": "the provider answered with an error"}

    @model_validator(mode="after")
    def _platform_words_stay_in_the_log(self) -> Self:
        if self.platform_provider is not None and self.detail is not None:
            raise ValueError("a refusal on a Galaius key carries no vendor words: the server log keeps them")
        return self

    @classmethod
    def of_answer(cls, status: int, vendor_code: str | None = None, detail: str | None = None) -> Self:
        reason = next((reason for reason, codes in cls.CODES.items() if vendor_code in codes), None) or cls.BY_STATUS.get(status, "failed")
        return cls(reason=reason, vendor_status=status, vendor_code=vendor_code, detail=detail)

    def on_platform_key(self, provider: str) -> Self:
        """The same refusal told to an organisation whose call went on Galaius's `provider` key: without the vendor's
        words, which were written to Galaius's account."""
        return type(self).model_validate({**self.model_dump(), "detail": None, "platform_provider": provider})

    @property
    def said(self) -> str:
        sentence = f"the provider answered {self.vendor_status}" if self.reason == "failed" and self.vendor_status else super().said
        return f"Galaius's {self.platform_provider} key: {sentence}" if self.platform_provider else sentence


class ProviderSilent(StepFault):
    """The provider did not answer in time, or its address could not be reached."""

    status: ClassVar[int] = 504
    code: Literal["provider_silent"] = "provider_silent"
    SAID: ClassVar[str] = "the provider did not answer"


class ProviderInvalid(StepFault):
    """The provider answered something that is not the answer its API promises (`detail`: the part it got wrong)."""

    code: Literal["provider_invalid"] = "provider_invalid"
    SAID: ClassVar[str] = "provider response is invalid"


#: Why the server will not follow a provider's address or answer: it is not https, does not resolve, resolves to a
#: private address (a request forged toward this host's network), is larger than a step reads, redirects without end,
#: or points outside the provider's own host.
EndpointReason = Literal["not_https", "unresolved", "private_address", "too_large", "redirects", "foreign_host"]


class EndpointRefused(ReasonedFault):
    """The server refused a provider's address or answer, for this host's safety."""

    code: Literal["endpoint_refused"] = "endpoint_refused"
    reason: EndpointReason
    SAID: ClassVar[Mapping[str, str]] = {
        "not_https": "the provider endpoint must use https", "unresolved": "the provider endpoint does not resolve",
        "private_address": "the provider endpoint resolves to a private address", "too_large": "the provider answer is too large",
        "redirects": "the provider file redirected too many times", "foreign_host": "the provider returned an address outside its own host"}


class NoSpeech(StepFault):
    """A transcription heard no speech in its audio."""

    status: ClassVar[int] = 422
    code: Literal["no_speech"] = "no_speech"
    SAID: ClassVar[str] = "the model heard no speech in this audio"


class OutputLimit(StepFault):
    """The model stopped at its output ceiling before it finished: `ceiling` tokens (thinking included) when the step set
    it, None when the provider's own default held."""

    code: Literal["output_limit"] = "output_limit"
    ceiling: int | None = Field(default=None, ge=0)
    SAID: ClassVar[str] = "the model stopped at its output limit ({ceiling} tokens, thinking included): raise Max output tokens"

    @property
    def said(self) -> str:
        return super().said if self.ceiling is not None else "the model stopped at its output limit"


#: What a step expected back from its model.
Returned = Literal["content", "text", "image", "audio", "music", "sound", "video", "vector", "pose", "mesh", "depth_map", "predictions", "pages", "match"]


class TimedOut(StepFault):
    """Something the step waited on passed its `seconds`."""

    status: ClassVar[int] = 504
    seconds: float = Field(gt=0)


class JobTimeout(TimedOut):
    """A provider job still unfinished after `seconds`: the step stopped waiting for its `what`, the provider may finish it."""

    code: Literal["job_timeout"] = "job_timeout"
    what: Returned
    SAID: ClassVar[str] = "the {what} took longer than {seconds:g} s; the provider may still finish it"


class NothingReturned(StepFault):
    """The model answered without the `what` the step asked for; `filtered`: the provider said its safety filter held it
    back (`detail`: the provider's stated reason, verbatim)."""

    code: Literal["nothing_returned"] = "nothing_returned"
    what: Returned
    filtered: bool = False
    SAID: ClassVar[str] = "the model returned no {what}"


class ClipTooShort(StepFault):
    """The provider makes a `resolution` clip only `seconds` long."""

    status: ClassVar[int] = 422
    code: Literal["clip_too_short"] = "clip_too_short"
    resolution: str = Field(min_length=1, max_length=20)
    seconds: int = Field(gt=0)
    SAID: ClassVar[str] = "{resolution} needs a clip of {seconds} seconds"


class ModelRetired(StepFault):
    """The provider shut the step's model down on `on`: `successor` is the model it names in its place."""

    status: ClassVar[int] = 410
    code: Literal["model_retired"] = "model_retired"
    on: date
    successor: str = Field(min_length=1, max_length=120)
    SAID: ClassVar[str] = "the provider shut this model down on {on}: pick {successor} in this step"


class NoKey(StepFault):
    """The organisation holds no key for `provider`: one is added in Connections."""

    status: ClassVar[int] = 409
    code: Literal["no_key"] = "no_key"
    provider: ProviderId
    SAID: ClassVar[str] = "no {provider} key in this organisation: add one in Connections"


#: What a step lacks to run: a source image wired in, an audio file (not a text), an enrolled machine chosen, or an
#: input its model does not take (a reference image on a text-only model), file storage in its workspace, its model in
#: the step's connection, or a request small enough to send.
InputReason = Literal["image_missing", "audio_not_file", "machine_missing", "image_unused", "no_storage", "model_unconfigured", "request_too_large",
                      "model_unregistered"]


class InputFault(ReasonedFault):
    """The step's inputs or settings cannot run its model."""

    status: ClassVar[int] = 422
    code: Literal["input_fault"] = "input_fault"
    reason: InputReason
    SAID: ClassVar[Mapping[str, str]] = {
        "image_missing": "a source image is required", "audio_not_file": "the audio input is not a file",
        "machine_missing": "this model runs on an enrolled machine; choose one",
        "image_unused": "this model makes video from text only; wire nothing into its reference image port",
        "no_storage": "this workspace has no file storage for the step's files",
        "model_unconfigured": "the step's connection does not offer this model",
        "request_too_large": "the step's request is larger than the provider call accepts",
        "model_unregistered": "the step's own model is no longer registered in this workspace; pick another"}


class InputEmpty(StepFault):
    """The step's required input `port` has no value, wired or set."""

    status: ClassVar[int] = 422
    code: Literal["input_empty"] = "input_empty"
    port: str = Field(min_length=1, max_length=120)
    SAID: ClassVar[str] = 'input "{port}" is empty'


class NoRunner(StepFault):
    """No runner makes `task` with `provider`'s `model`: the step names a model its provider does not serve for it."""

    status: ClassVar[int] = 422
    code: Literal["no_runner"] = "no_runner"
    task: ModelTask
    provider: ProviderId
    model: str = Field(min_length=1, max_length=200)
    SAID: ClassVar[str] = "no runner for {task} on {provider} ({model})"


class PlatformUnpriced(StepFault):
    """`model` has no price Galaius can bill: it never runs on Galaius's `provider` key, only on the organisation's own."""

    status: ClassVar[int] = 409
    code: Literal["platform_unpriced"] = "platform_unpriced"
    provider: ProviderId
    model: str = Field(min_length=1, max_length=200)
    SAID: ClassVar[str] = "{model} has no price Galaius can bill, so it does not run on Galaius's {provider} key"


class NoModelPasses(StepFault):
    """No `task` model passes the step's choice (`criteria`, as its node states them; None: the choice names none)."""

    status: ClassVar[int] = 422
    code: Literal["no_model_passes"] = "no_model_passes"
    task: ModelTask
    criteria: str | None = None

    @property
    def said(self) -> str:
        return f"no {self.task} model passes {self.criteria or 'this choice'}"


#: What a step's model choice names that its task's catalog does not have: a property, a percentile of a property
#: publishing none, a benchmark.
CriteriaReason = Literal["not_a_property", "no_percentile", "not_a_benchmark"]


class CriteriaInvalid(ReasonedFault):
    """The step's choice names `name`, which `task` models do not offer (`reason`)."""

    status: ClassVar[int] = 422
    code: Literal["criteria_invalid"] = "criteria_invalid"
    reason: CriteriaReason
    task: ModelTask
    name: CriteriaName
    SAID: ClassVar[Mapping[str, str]] = {
        "not_a_property": "'{name}' is not a property of {task} models", "no_percentile": "'{name}' has no published field to take a percentile of",
        "not_a_benchmark": "'{name}' is not a benchmark of {task}"}


class CriteriaMiswritten(StepFault):
    """The step's choice writes `name`, a `task` property, as another kind than the one it is (`kind`)."""

    status: ClassVar[int] = 422
    code: Literal["criteria_miswritten"] = "criteria_miswritten"
    task: ModelTask
    name: CriteriaName
    kind: PropertyKind
    SAID: ClassVar[str] = "'{name}' is a {kind}, not written that way"


class OnMachine(StepFault):
    """A fault about the step's machine: `machine` its name, when one was chosen, said first."""

    machine: MachineName | None = None

    def on_machine(self, name: str | None) -> Self:
        return type(self).model_validate({**self.model_dump(exclude={"message"}), "machine": name})

    @property
    def said(self) -> str:
        said = super().said
        return f"{self.machine}: {said}" if self.machine else said


#: Why a step's machine did not run it. Reaching it: `offline` (not connected, `waited_s` for it), `disconnected`
#: mid-step, `unanswered` (connected, said nothing in time). Placing it: `unplaced` (no machine and no need set). Its
#: owner's rules: `unsaved` (a machine runs only workflows its owner saved), `not_allowed` (file steps for people its
#: owner allows), `script_unapproved` (its owner approves each script digest), `cloud_forbidden` / `pool_forbidden` (this
#: workspace may not launch cloud machines / use pooled ones). The pool: `pool_empty` (none shared, idle and fitting),
#: `pool_unscrubbed` (no GPU one scrubbed since its last tenant), `pool_claimed` (another run took it first).
MachineReason = Literal["offline", "disconnected", "unanswered", "outdated", "unplaced", "unsaved", "not_allowed", "script_unapproved", "cloud_forbidden",
                        "pool_forbidden", "pool_empty", "pool_unscrubbed", "pool_claimed", "pool_failed", "declined"]


class MachineFault(OnMachine, ReasonedFault):
    """A step's machine did not run it (`MachineReason`)."""

    status: ClassVar[int] = 409
    code: Literal["machine_fault"] = "machine_fault"
    reason: MachineReason
    waited_s: float | None = Field(default=None, ge=0)
    #: What an `outdated` machine's runner lacks, as its hello would name it.
    feature: PcFeature | None = None
    SAID: ClassVar[Mapping[str, str]] = {
        "offline": "the machine is offline", "disconnected": "the machine disconnected", "unanswered": "the machine did not answer in time",
        "outdated": f"the Galaius on this machine is too old. It updates itself; otherwise its owner runs {UPDATE_COMMAND} there",
        "unplaced": "this step is not placed on a machine",
        "unsaved": "a machine runs only workflows its owner saved: save this workflow, then run it again",
        "not_allowed": "file steps on this machine run only for people its owner allows",
        "script_unapproved": "this script is not approved on this machine: its owner approves its exact digest first",
        "cloud_forbidden": "this workspace may not launch cloud machines", "pool_forbidden": "this workspace may not use pooled machines",
        "pool_empty": "no pooled machine is shared, idle and fit for this step", "pool_unscrubbed": "no pooled GPU machine is scrubbed since its last tenant",
        "pool_claimed": "another run took the pooled machine first; run it again",
        "pool_failed": "the pooled machine failed this step: run it again, or place it on one of your own machines",
        "declined": "the run read private data and this machine step was not allowed to run"}

    @model_validator(mode="after")
    def _waited_offline(self) -> Self:
        if self.waited_s is not None and self.reason != "offline":
            raise ValueError("waited_s: only an offline machine was waited for")
        if (self.feature is None) == (self.reason == "outdated"):
            raise ValueError("feature: what an outdated machine lacks, and only then")
        return self


#: Why the cloud launched no machine either, when it was tried: no instance type `fits`, the workspace's cloud spending
#: `limit` (money), no `sovereign` instance where the need requires one (security), or the provider `failed` (its words in
#: `detail`).
CloudRefusal = Literal["fits", "limit", "sovereign", "failed"]


class MachineUnfit(StepFault):
    """No connected machine, pooled peer or cloud launch meets the step's `requirement` (and its script `language`):
    `cloud` why a cloud launch did not either, when one was tried."""

    status: ClassVar[int] = 409
    code: Literal["no_fit"] = "no_fit"
    requirement: ResourceRequirement
    language: ScriptLanguage | None = None
    cloud: CloudRefusal | None = None
    SAID: ClassVar[str] = "no connected machine, pooled peer or cloud launch meets this step's needs"
    CLOUD: ClassVar[Mapping[str, str]] = {
        "fits": "no cloud instance type fits them", "limit": "a cloud launch would pass this workspace's cloud spending limit",
        "sovereign": "no sovereign cloud instance fits them", "failed": "the cloud provider refused the launch"}

    @property
    def said(self) -> str:
        return f"{self.SAID}; {self.CLOUD[self.cloud]}" if self.cloud else self.SAID


class CloudLaunch(StepFault):
    """The step's machine is a cloud `instance` launched in `region` at `usd_per_hour`: the step ends here and a rerun
    once it connects places on it; `dry_run` when no cloud credentials are configured, so nothing was launched."""

    status: ClassVar[int] = 503
    code: Literal["cloud_launch"] = "cloud_launch"
    dry_run: bool
    instance: InstanceTypeName
    region: RegionName
    usd_per_hour: float = Field(ge=0)
    SAID: ClassVar[str] = "a {instance} machine is starting in {region}; rerun once it connects"

    @property
    def said(self) -> str:
        return "dry run only: no cloud credentials are configured, so no machine was launched" if self.dry_run else super().said


class PoolBudget(StepFault):
    """A pooled run would bring the workspace's pooled spend this period to `projected_usd`, over its `ceiling_usd`."""

    status: ClassVar[int] = 402
    code: Literal["pool_budget"] = "pool_budget"
    projected_usd: float = Field(ge=0)
    ceiling_usd: float = Field(ge=0)
    SAID: ClassVar[str] = "a pooled run would bring this period's pooled spend to ${projected_usd:.2f}, over its ${ceiling_usd:.2f} ceiling"


class StepTimedOut(OnMachine, TimedOut):
    """The step ran past its `seconds`; `waiting` what it still waited on (its `machine` to connect), if anything."""

    code: Literal["timed_out"] = "timed_out"
    waiting: Literal["machine"] | None = None
    SAID: ClassVar[str] = "timed out after {seconds:g} s"

    @property
    def said(self) -> str:
        said = super().said
        return f"{said} waiting for its machine to connect" if self.waiting == "machine" else said


#: What stopped a run on the server: it `restarted` mid-run, it `crashed` (an error of the server's own, logged), or its
#: execution settings were `config_unavailable`.
ServerReason = Literal["restarted", "crashed", "config_unavailable"]


class ServerFault(ReasonedFault):
    """The server could not finish the run (`ServerReason`)."""

    status: ClassVar[int] = 503
    code: Literal["server_fault"] = "server_fault"
    reason: ServerReason
    SAID: ClassVar[Mapping[str, str]] = {
        "restarted": "the server restarted before the run finished", "crashed": "the run stopped on a server error",
        "config_unavailable": "the workflow's execution settings are unavailable"}


#: Why a workspace's file storage did not serve a step. Its connection: `credential_missing` (no secret saved),
#: `credential_malformed` (the secret lacks a key the store needs), `provider_unsupported`. The store: `refused` (its
#: own words in `detail`), `answer_invalid` (an answer that does not parse), `too_large` (past what a step reads).
StorageReason = Literal["credential_missing", "credential_malformed", "provider_unsupported", "not_granted", "refused", "answer_invalid", "too_large"]
#: What a file storage connection may be granted (`ConnectionResource.capabilities`).
StorageGrant = Literal["list", "read", "write"]


class StorageFault(ReasonedFault):
    """A workspace's file storage did not serve the step (`StorageReason`); `not_granted` names the `grant` its
    connection lacks."""

    status: ClassVar[int] = 502
    code: Literal["storage_fault"] = "storage_fault"
    reason: StorageReason
    grant: StorageGrant | None = None
    SAID: ClassVar[Mapping[str, str]] = {
        "credential_missing": "the file storage connection has no saved secret",
        "credential_malformed": "the file storage connection's secret lacks a key the store needs",
        "provider_unsupported": "this file storage provider is not supported",
        "not_granted": "this file storage connection is not granted {grant}",
        "refused": "file storage refused the request", "answer_invalid": "file storage sent an answer that could not be read",
        "too_large": "the file is larger than a step moves"}

    @model_validator(mode="after")
    def _grant_not_granted(self) -> Self:
        if (self.grant is None) == (self.reason == "not_granted"):
            raise ValueError("grant: named exactly when the connection is not granted it")
        return self


#: Why a step failed, one member per `code`.
StepFaultCause = Annotated[VendorRefused | ProviderSilent | ProviderInvalid | EndpointRefused | NoSpeech | OutputLimit | JobTimeout
                           | NothingReturned | ClipTooShort | ModelRetired | NoKey | InputFault | InputEmpty | NoRunner | PlatformUnpriced
                           | NoModelPasses | CriteriaInvalid | CriteriaMiswritten | MachineFault | MachineUnfit | CloudLaunch | PoolBudget | StepTimedOut
                           | ServerFault | StorageFault, Field(discriminator="code")]

