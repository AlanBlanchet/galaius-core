"""An image drawn by the platform for whoever asks — a person, a script, an agent on one of his PCs — so
every picture is paid with the company's provider connection and counted where its money is: the
run it makes, the person charged, the ceilings above him, the month's journal.

    POST /v1/workspaces/{workspace}/images        ImageRequest -> 201 ImageAnswer

Each image is one run of one model step: refused before it runs past a ceiling (402) or with paid use
of its provider off (403), never a 2xx with the refusal inside. `Idempotency-Key` (optional, 1-200
printable ASCII): the same key from the same person answers the first request's images again,
never paying twice."""

from typing import Literal
from uuid import UUID

from pydantic import Field, computed_field

from galaius_core.wire import WireModel, WireRequest

#: The sizes, qualities and backgrounds an image model draws at — the server's image step takes the
#: same values (its node config), so a request and a workflow step never disagree on them.
ImageSize = Literal["1024x1024", "1536x1024", "1024x1536", "auto"]
ImageQuality = Literal["low", "medium", "high", "auto"]
ImageBackground = Literal["auto", "opaque", "transparent"]
#: Images one request draws at most: each is its own run and its own charge.
IMAGES_PER_REQUEST = 4


class ImageRequest(WireRequest):
    prompt: str = Field(min_length=1)
    provider: str = Field(default="openai", min_length=1, max_length=64)
    model: str = Field(default="gpt-image-2", min_length=1, max_length=120)
    size: ImageSize = "1024x1024"
    quality: ImageQuality = "medium"
    background: ImageBackground = "auto"
    n: int = Field(default=1, ge=1, le=IMAGES_PER_REQUEST)
    #: The agent run asking (its `GALAIUS_RUN_ID` on the PC): the journal links each image's charge to
    #: that run's page. Said by the caller, never trusted for anything else.
    caller_run: UUID | None = None


class GeneratedImage(WireModel):
    #: The run that drew it: its record holds the cost and the file.
    run_id: UUID
    media_type: str
    #: The image itself, base64.
    data_base64: str
    size_bytes: int = Field(ge=0)
    #: Provider cost + platform fee; None when the provider's answer gave no usage to price it by.
    cost_usd: float | None = None


class ImageAnswer(WireModel):
    provider: str
    model: str
    images: tuple[GeneratedImage, ...]
    #: Why each image asked for and not drawn was not (a ceiling met midway, the provider's refusal):
    #: a request answers 201 when at least one image was drawn, a real error status when none was.
    failed: tuple[str, ...] = ()

    @computed_field
    @property
    def cost_usd(self) -> float | None:
        """Every image's cost together; None when one of them is unpriced."""
        costs = [image.cost_usd for image in self.images]
        return None if None in costs else round(sum(costs), 8)
