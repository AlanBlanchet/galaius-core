"""Portable personal tool preferences; machine paths, credentials and grants excluded.

A role field never names a model. It carries a CRITERION — a requirement string the resolver
reads against whatever the calling project actually has reachable (a saved provider key, or a
logged-in CLI session) — never a pinned id that silently stops being the right choice, or a right
choice nobody's key can reach. The resolver lives where the ranker and the availability facts
both are (`interact.criteria.Criteria`, called from the server that also knows the project's
connections); this module only carries the requirement text and its bounds.
"""

from typing import Literal, Self

from pydantic import Field, model_validator

from .wire import WireModel

VLM_MIN_DIM_DEFAULT = 768
VLM_MAX_DIM_DEFAULT = 1280
MediaSessionProviderName = Literal["claude"]
#: One requirement per role interact resolves a model for. A fallback CHAIN is no longer a
#: separate field: `Criteria.qualifying` already returns every clearing model in rank order, so
#: the second-ranked entry the resolver already computed IS the fallback — a second pinned list
#: would just be a second, driftable copy of the same ranking.
_CRITERIA_MAX_LENGTH = 2048


class PortableToolSettingsValues(WireModel):
    image_criteria: str | None = Field(default=None, max_length=_CRITERIA_MAX_LENGTH)
    video_criteria: str | None = Field(default=None, max_length=_CRITERIA_MAX_LENGTH)
    audio_criteria: str | None = Field(default=None, max_length=_CRITERIA_MAX_LENGTH)
    component_criteria: str | None = Field(default=None, max_length=_CRITERIA_MAX_LENGTH)
    claude_media_criteria: str | None = Field(default=None, max_length=_CRITERIA_MAX_LENGTH)
    tier_sovereign_criteria: str | None = Field(default=None, max_length=_CRITERIA_MAX_LENGTH)
    #: Shared across every role above: a weight names a benchmark variable (`aa.intelligence=1`),
    #: not a role, so one role-specific copy per role would only ever hold the same text.
    criteria_weights: str | None = Field(default=None, max_length=_CRITERIA_MAX_LENGTH)
    media_provider_order: tuple[MediaSessionProviderName, ...] | None = Field(default=None, max_length=32)
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
        minimum = VLM_MIN_DIM_DEFAULT if self.vlm_min_dim is None else self.vlm_min_dim
        maximum = VLM_MAX_DIM_DEFAULT if self.vlm_max_dim is None else self.vlm_max_dim
        if minimum > maximum:
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
