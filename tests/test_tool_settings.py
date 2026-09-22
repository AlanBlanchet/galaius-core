import pytest
from pydantic import ValidationError

from interact_core import PortableToolSettings, PortableToolSettingsUpdate, PortableToolSettingsValues


def test_sparse_settings_roundtrip_and_explicit_clear():
    record = PortableToolSettings(revision=7, values=PortableToolSettingsValues(image_criteria="cap.vlm and price.in < 5"))
    assert PortableToolSettings.model_validate_json(record.model_dump_json()) == record
    cleared = PortableToolSettingsUpdate(expected_revision=7, values={})
    assert cleared.model_dump(exclude_none=True) == {"expected_revision": 7, "values": {}}


@pytest.mark.parametrize("values", [
    {"OPENAI_API_KEY": "excluded"}, {"debug_dir": "excluded"}, {"media_billing": "api_allowed"},
    {"image_model": "excluded"}, {"image_fallbacks": "excluded"}, {"media_criteria": "excluded"},
    {"media_provider_order": []}, {"media_provider_order": ["same", "same"]},
    {"media_provider_order": [" padded "]}, {"vlm_min_dim": 2000, "vlm_max_dim": 1000},
    {"video_duration": float("inf")}, {"viewport_width": 0}, {"image_criteria": "x" * 2049},
    {"vlm_min_dim": 1500}, {"vlm_max_dim": 500}, {"media_provider_order": ["unsupported"]},
])
def test_settings_reject_nonportable_and_invalid_values(values):
    with pytest.raises(ValidationError):
        PortableToolSettingsValues.model_validate(values)


def test_sparse_dimensions_use_shared_defaults():
    assert PortableToolSettingsValues(vlm_max_dim=1500).vlm_min_dim is None
    assert PortableToolSettingsValues(vlm_min_dim=1500, vlm_max_dim=2000).vlm_max_dim == 2000


def test_every_media_role_takes_a_criterion_not_a_pinned_model():
    """A role names a REQUIREMENT, never a model — resolution happens where the ranker and the
    project's availability facts both are (the server), not by freezing an id here."""
    fields = PortableToolSettingsValues.model_fields
    for role in ("image", "video", "audio", "component", "claude_media", "tier_sovereign"):
        assert f"{role}_criteria" in fields
        assert f"{role}_model" not in fields
    assert "criteria_weights" in fields
    assert "media_criteria" not in fields
    assert "image_fallbacks" not in fields
