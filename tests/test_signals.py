"""A stored signal never carries a credential, a URL query value or (in captured text) an email,
whatever the browser sent; ids that point at the fault survive."""

import base64

import pytest
from pydantic import ValidationError

from interact_core.signals import SignalRedaction, SignalScreenshot, SignalSubmission

RUN = "3f2b8c1e-9a4d-4c7e-8b1a-0d2e4f6a8b9c"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
CONTEXT = {
    "route": f"/#workflows/{RUN}", "screen": "workflows", "locale": "fr-FR",
    "viewport": {"width": 1280, "height": 800, "pixel_ratio": 2}, "user_agent": "Mozilla/5.0",
}


@pytest.mark.parametrize(("raw", "kept", "gone"), [
    ("GET /v1/data?cursor=abc123&q=hello failed", "cursor=[redacted]", "abc123"),
    ("Authorization: Bearer abcdefghijklmnop", "[redacted]", "abcdefghijklmnop"),
    ("key sk-proj-A1b2C3d4E5f6G7h8 refused", "[redacted]", "sk-proj-A1b2C3d4E5f6G7h8"),
    ("jwt eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl", "[redacted]", "eyJhbGciOiJIUzI1NiJ9"),
    ('{"password": "hunter22"}', '"password": "[redacted]', "hunter22"),
    ("mail bob.smith@example.com bounced", "[email]", "bob.smith@example.com"),
    ("blob " + "x9Kf" * 12, "[redacted]", "x9Kf" * 12),
    (f"run {RUN} failed", RUN, "[redacted]"),
    (f"/#workflows/{RUN}/runs/{RUN}", f"workflows/{RUN}/runs/{RUN}", "[redacted]"),
])
def test_captured_text_is_redacted_and_ids_survive(raw: str, kept: str, gone: str) -> None:
    clean = SignalRedaction.anonymous(raw)
    assert kept in clean and gone not in clean


def test_typed_description_keeps_emails_but_drops_keys() -> None:
    value = SignalSubmission.model_validate({
        "id": RUN, "description": "Mail to bob@example.com failed, key iwk_abcdef123456", "context": CONTEXT,
    })
    assert "bob@example.com" in value.description and "iwk_abcdef123456" not in value.description


def test_context_strings_are_redacted_on_validation() -> None:
    value = SignalSubmission.model_validate({"id": RUN, "description": "broken", "context": {
        **CONTEXT, "route": "/?token=s3cr3t#runs?project=workspace&state=xyz",
        "console": [{"level": "error", "message": "ann@corp.fr: 500", "at": "2026-09-26T10:00:00Z"}],
        "requests": [{"method": "GET", "path": "/v1/x?code=zzz", "status": 500, "at": "2026-09-26T10:00:00Z"}],
    }})
    assert value.context.route == "/?token=[redacted]#runs?project=workspace&state=[redacted]"
    assert value.context.console[0].message == "[email]: 500"
    assert "zzz" not in value.context.requests[0].path


@pytest.mark.parametrize("route", ["//evil.example/#x", "https://evil.example/", "/\\evil", "workflows", "/v1/operator/accounts", "javascript:alert(1)"])
def test_route_is_same_origin_only(route: str) -> None:
    with pytest.raises(ValidationError):
        SignalSubmission.model_validate({"id": RUN, "description": "x", "context": {**CONTEXT, "route": route}})


@pytest.mark.parametrize(("media_type", "data", "ok"), [
    ("image/png", PNG, True),
    ("image/jpeg", PNG, False),
    ("image/webp", b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 16, True),
    ("image/webp", b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 16, False),
    ("image/png", b"<svg onload=alert(1)>" + b"\x00" * 16, False),
])
def test_screenshot_bytes_must_be_the_declared_image(media_type: str, data: bytes, ok: bool) -> None:
    payload = {"media_type": media_type, "data_base64": base64.b64encode(data).decode()}
    if ok:
        assert SignalScreenshot.model_validate(payload).data == data
    else:
        with pytest.raises(ValidationError):
            SignalScreenshot.model_validate(payload)


def test_redaction_that_grows_a_value_is_cut_to_its_bound_and_reads_back() -> None:
    path = "/p?" + "&a=" * 130
    value = SignalSubmission.model_validate({"id": RUN, "description": "x", "context": {
        **CONTEXT, "element": {"selector": "a@b.cc " * 85},
        "requests": [{"method": "GET", "path": path, "status": 500, "at": "2026-09-26T10:00:00Z"}],
    }})
    assert len(value.context.requests[0].path) <= 400 and len(value.context.element.selector) <= 600
    assert SignalSubmission.model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize("route", ["/", "/?enter#login", "/#runs?project=x"])
def test_route_accepts_the_app_address(route: str) -> None:
    assert SignalSubmission.model_validate({"id": RUN, "description": "x", "context": {**CONTEXT, "route": route}}).context.route == route
