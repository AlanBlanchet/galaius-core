from datetime import date
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from galaius_core import (
    Cause,
    ClipTooShort,
    Coded,
    EndpointRefused,
    FailureCause,
    InputEmpty,
    InputFault,
    JobTimeout,
    ModelRetired,
    NoKey,
    NoRunner,
    NoSpeech,
    NothingReturned,
    OutputLimit,
    ProviderInvalid,
    ProviderSilent,
    ReasonedFault,
    VendorRefused,
    cause_of,
)

CAUSE = TypeAdapter(FailureCause)


@pytest.mark.parametrize(("fault", "message"), [
    (VendorRefused.of_answer(401, detail="Incorrect API key provided"), "the API key was refused: Incorrect API key provided"),
    (VendorRefused.of_answer(429), "the provider's quota or rate limit was hit"),
    (VendorRefused.of_answer(503), "the provider answered 503"),
    (VendorRefused.of_answer(401, "missing_permissions").on_platform_key("openai"), "Galaius's openai key: the API key is not allowed to use this model"),
    (ProviderSilent(detail="timed out"), "the provider did not answer: timed out"),
    (ProviderInvalid(), "provider response is invalid"),
    (EndpointRefused(reason="private_address"), "the provider endpoint resolves to a private address"),
    (NoSpeech(), "the model heard no speech in this audio"),
    (OutputLimit(ceiling=800), "the model stopped at its output limit (800 tokens, thinking included): raise Max output tokens"),
    (OutputLimit(vendor_code="MAX_TOKENS"), "the model stopped at its output limit"),
    (JobTimeout(what="video", seconds=540), "the video took longer than 540 s; the provider may still finish it"),
    (NothingReturned(what="image", filtered=True, detail="SAFETY"), "the model returned no image: SAFETY"),
    (ClipTooShort(resolution="1080p", seconds=8), "1080p needs a clip of 8 seconds"),
    (ModelRetired(on=date(2026, 9, 1), successor="gemini-omni-flash"), "the provider shut this model down on 2026-09-01: pick gemini-omni-flash in this step"),
    (NoKey(provider="openai"), "no openai key in this organisation: add one in Connections"),
    (InputFault(reason="image_missing"), "a source image is required"),
    (InputEmpty(port="prompt"), 'input "prompt" is empty'),
    (NoRunner(task="text-to-video", provider="mistral", model="mistral-small"), "no runner for text-to-video on mistral (mistral-small)"),
    (NothingReturned(what="content", filtered=True, vendor_code="SAFETY"), "the model returned no content"),
])
def test_a_step_fault_says_its_english_once_and_reads_back_through_the_cause_union(fault, message):
    dumped = CAUSE.dump_python(fault, mode="json")
    assert dumped["message"] == fault.message == message
    assert CAUSE.validate_python(dumped) == fault
    assert {"code", "message"} <= set(CAUSE.json_schema(mode="serialization")["$defs"][type(fault).__name__]["required"])


def _concrete(kind: type) -> set[type]:
    below = {found for sub in kind.__subclasses__() for found in _concrete(sub)}
    return below | ({kind} if "code" in kind.model_fields else set())


def test_every_cause_is_a_member_of_the_union_and_answers_a_status():
    members = {member for arm in get_args(get_args(FailureCause)[0]) for member in (get_args(get_args(arm)[0]) or (arm,))}
    assert _concrete(Cause) == members
    assert all(isinstance(member.status, int) for member in members)


@pytest.mark.parametrize("kind", sorted(_concrete(ReasonedFault), key=lambda kind: kind.__name__))
def test_a_reasoned_fault_says_one_sentence_per_reason(kind):
    assert set(kind.SAID) == set(get_args(kind.model_fields["reason"].annotation))


@pytest.mark.parametrize(("status", "code", "reason"), [
    (401, None, "key_refused"), (403, None, "key_forbidden"), (404, None, "model_unknown"), (402, None, "no_funds"),
    (429, None, "rate_limited"), (500, None, "failed"), (429, "insufficient_quota", "no_funds"), (401, "quota_exceeded", "no_funds"),
])
def test_the_answer_a_vendor_gives_names_what_its_reader_can_do(status, code, reason):
    assert VendorRefused.of_answer(status, code).reason == reason


def test_a_refusal_on_a_galaius_key_never_carries_the_vendor_words():
    with pytest.raises(ValidationError):
        VendorRefused(reason="key_refused", platform_provider="openai", detail="sk-... is wrong")


class _Failed(Coded[FailureCause | None]):
    pass


def test_a_provider_writing_past_a_field_ceiling_is_clipped_never_refused():
    fault = VendorRefused.of_answer(500, "x" * 90, "y" * 900)
    assert (len(fault.vendor_code), len(fault.detail)) == (60, 300)


def test_a_step_fault_raised_under_another_error_is_still_its_cause():
    fault = NothingReturned(what="audio")
    try:
        try:
            raise _Failed(fault, fault.message)
        except _Failed as inner:
            raise RuntimeError("step failed") from inner
    except RuntimeError as outer:
        assert cause_of(outer) == fault
