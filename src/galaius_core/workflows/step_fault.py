"""Why a model step failed one hop past the server, typed: what its provider answered (a refusal, no answer, an answer
nobody can read, an address the server will not follow), what it sent back (nothing, or cut at its output limit), or
what the step lacks to run. Its reader's words come from its `code` and figures; `message` is the same fact in English,
for a log only, said once here."""

from collections.abc import Mapping
from datetime import date
from typing import Annotated, ClassVar, Literal, Self

from pydantic import Field, ValidationInfo, computed_field, field_validator, model_validator

from galaius_core.wire import Cause
from galaius_core.models.task import ModelTask


class StepFault(Cause):
    """One way a model step failed past the server. A kind states its English once (`SAID`, a template over its own
    fields); `detail` is the provider's own words, verbatim, when it wrote any, and `vendor_code` its own code for the
    failure (an error code, a stop or block reason), when it gave one: never a sentence of ours."""

    #: A provider failed the call: the server is its gateway.
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

    @property
    def said(self) -> str:
        return self.SAID.format_map(vars(self))

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
    platform_provider: str | None = Field(default=None, max_length=40)

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


class JobTimeout(StepFault):
    """A provider job still unfinished after `seconds`: the step stopped waiting for its `what`, the provider may finish it."""

    status: ClassVar[int] = 504
    code: Literal["job_timeout"] = "job_timeout"
    what: Returned
    seconds: float = Field(gt=0)
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
    provider: str = Field(min_length=1, max_length=40)
    SAID: ClassVar[str] = "no {provider} key in this organisation: add one in Connections"


#: What a step lacks to run: a source image wired in, an audio file (not a text), an enrolled machine chosen, or an
#: input its model does not take (a reference image on a text-only model), file storage in its workspace, its model in
#: the step's connection, or a request small enough to send.
InputReason = Literal["image_missing", "audio_not_file", "machine_missing", "image_unused", "no_storage", "model_unconfigured", "request_too_large"]


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
        "request_too_large": "the step's request is larger than the provider call accepts"}


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
    provider: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=200)
    SAID: ClassVar[str] = "no runner for {task} on {provider} ({model})"


#: Why a model step failed past the server, one member per `code`.
StepFaultCause = Annotated[VendorRefused | ProviderSilent | ProviderInvalid | EndpointRefused | NoSpeech | OutputLimit | JobTimeout
                           | NothingReturned | ClipTooShort | ModelRetired | NoKey | InputFault | InputEmpty | NoRunner, Field(discriminator="code")]

