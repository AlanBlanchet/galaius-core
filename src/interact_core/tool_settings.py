"""Portable personal tool preferences; machine paths, credentials and grants excluded."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .wire import WireModel


class PortableToolSettingsValues(WireModel):
    image_model: str | None = Field(default=None, max_length=4096)
    video_model: str | None = Field(default=None, max_length=4096)
    component_model: str | None = Field(default=None, max_length=4096)
    audio_model: str | None = Field(default=None, max_length=4096)
    claude_media_model: str | None = Field(default=None, max_length=4096)
    image_fallbacks: str | None = Field(default=None, max_length=4096)
    video_fallbacks: str | None = Field(default=None, max_length=4096)
    component_fallbacks: str | None = Field(default=None, max_length=4096)
    audio_fallbacks: str | None = Field(default=None, max_length=4096)
    tier_sovereign_model: str | None = Field(default=None, max_length=4096)
    media_criteria: str | None = Field(default=None, max_length=4096)
    media_criteria_weights: str | None = Field(default=None, max_length=4096)
    media_provider_order: tuple[Annotated[str, Field(max_length=4096)], ...] | None = Field(default=None, max_length=32)
    media_timeout: int | None = Field(default=None, gt=0)
    media_max_items: int | None = Field(default=None, gt=0)
    media_max_total_bytes: int | None = Field(default=None, gt=0)
    media_max_context_chars: int | None = Field(default=None, gt=0)
    viewport_width: int | None = Field(default=None, gt=0)
    viewport_height: int | None = Field(default=None, gt=0)
    video_fps: int | None = Field(default=None, gt=0)
    video_duration: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    video_max_frames: int | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    wait_timeout: int | None = Field(default=None, gt=0)
    vlm_max_dim: int | None = Field(default=None, gt=0)
    vlm_min_dim: int | None = Field(default=None, gt=0)
    detection_max_retries: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_portable_values(self) -> Self:
        if self.vlm_min_dim is not None and self.vlm_max_dim is not None and self.vlm_min_dim > self.vlm_max_dim:
            raise ValueError("minimum image dimension cannot exceed maximum")
        order = self.media_provider_order
        if order is not None and (not order or len(set(order)) != len(order) or any(not name.strip() or name != name.strip() for name in order)):
            raise ValueError("provider order must contain distinct nonempty names")
        return self


class PortableToolSettings(WireModel):
    schema_version: Literal[1] = 1
    revision: int = Field(ge=0)
    values: PortableToolSettingsValues


class PortableToolSettingsUpdate(WireModel):
    expected_revision: int = Field(ge=0)
    values: PortableToolSettingsValues
