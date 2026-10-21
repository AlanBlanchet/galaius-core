"""Media the Galaius platform makes for whoever asks — an agent on one of his PCs, a script, a person — in one shape for
every capability: a typed spec per kind (`MediaSpec`), one request, one answer. The platform picks the model, so every
call is paid with the company's provider connection and counted where its money is (the run it makes, the person
charged, the ceilings above him, the month's journal).

    POST /v1/workspaces/{workspace}/media        MediaRequest -> 200 MediaAnswer (a quote) | 201 MediaAnswer (made)

A request without `max_cost_usd` is a quote: the price range of the job, nothing run, nothing charged. With it, the job
runs when its quote's high bound is within it, else 402 with the quote (never a 2xx with the refusal inside). A job
with no known price is refused (400): it would spend without a charge. Each job is one run of one model step,
refused before it runs past a ceiling (402) or with paid use of its provider off (403). `Idempotency-Key` (optional,
1-200 printable ASCII): the same key from the same person answers the first request's job again, never paying twice.
Its output comes back as files (`MediaFile`), each kept for the agent run that asked and shown in its person's chat as
a file to download (`FILE_TOOLS`)."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from .file_tools import FileTool
from .wire import WireModel, WireRequest

MediaKind = Literal["translation", "model_3d", "speech", "music", "sound"]
#: The agent tool that asks for each kind, and the files it keeps (`workflows.FILE_TOOL_REGISTRY`).
_AUDIO = ((".mp3", "audio/mpeg"), (".wav", "audio/wav"), (".ogg", "audio/ogg"))
MEDIA_TOOLS: dict[MediaKind, FileTool] = {
    "translation": FileTool(tool="translate", writes=((".txt", "text/plain"),)),
    "model_3d": FileTool(tool="generate_3d", writes=((".glb", "model/gltf-binary"),)),
    "speech": FileTool(tool="generate_speech", writes=_AUDIO),
    "music": FileTool(tool="generate_music", writes=_AUDIO),
    "sound": FileTool(tool="generate_sound", writes=_AUDIO),
}
#: The image types a job reads (`image_type` proves them by their first bytes): `ImageContentType` but GIF, which no
#: 3D model takes (an animation); restated here, since `workflows` imports this module.
MediaImageType = Literal["image/png", "image/jpeg", "image/webp"]
#: The largest input file a job takes, raw bytes (a 3D model's source image).
MEDIA_INPUT_BYTES = 20 * 1024 * 1024
#: The longest text one translation takes, in characters.
TRANSLATION_CHARS = 50_000


class MediaImage(WireModel):
    """An image a job reads (a 3D model's source): its bytes, base64, and what they are (`image_type` of them: the
    server checks the bytes, never the word)."""

    media_type: MediaImageType
    data_base64: str = Field(min_length=4, max_length=(MEDIA_INPUT_BYTES * 4) // 3 + 8)


class TranslationSpec(WireModel):
    """A text translated into `target_language` (a BCP 47 tag: `fr`, `en-GB`), its layout kept as written."""

    kind: Literal["translation"] = "translation"
    text: str = Field(min_length=1, max_length=TRANSLATION_CHARS)
    target_language: str = Field(pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
    #: The text's language when known; None: the model reads it from the text.
    source_language: str | None = Field(default=None, pattern=r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
    #: What the text is and for whom (« a reply to a client », « app labels »): the register to keep.
    context: str = Field(default="", max_length=2000)


class Model3DSpec(WireModel):
    """A 3D model (a binary glTF, `.glb`) of the object in `image`."""

    kind: Literal["model_3d"] = "model_3d"
    image: MediaImage
    #: Physically-based materials: slower, and not every viewer needs them.
    pbr: bool = False


class SpeechSpec(WireModel):
    """`text` said aloud, as an audio file."""

    kind: Literal["speech"] = "speech"
    text: str = Field(min_length=1, max_length=TRANSLATION_CHARS)
    #: The model's voice (a Gemini voice name, an ElevenLabs voice id); "" its default.
    voice: str = Field(default="", max_length=64)
    #: How to say it (« calm and warm »), for a model that takes it; "" neutral.
    style: str = Field(default="", max_length=200)


class MusicSpec(WireModel):
    """Music made from `prompt`: genre, mood, instruments, structure, lyrics, length."""

    kind: Literal["music"] = "music"
    prompt: str = Field(min_length=1, max_length=32_000)


class SoundSpec(WireModel):
    """A sound effect made from `prompt` (« rain on a tin roof »), never words to say."""

    kind: Literal["sound"] = "sound"
    prompt: str = Field(min_length=1, max_length=32_000)
    #: Its length, 0.5 to 30 s; None: the model picks it from the description.
    seconds: float | None = Field(default=None, ge=0.5, le=30)
    #: Made to loop seamlessly.
    loop: bool = False


MediaSpec = Annotated[TranslationSpec | Model3DSpec | SpeechSpec | MusicSpec | SoundSpec, Field(discriminator="kind")]


class MediaRequest(WireRequest):
    spec: MediaSpec
    #: The most it may cost, platform fee included; None: a quote, nothing run.
    max_cost_usd: float | None = Field(default=None, ge=0)
    #: The agent run asking (its `GALAIUS_RUN_ID` on the PC): the journal links the job's charge to that run's page.
    #: Said by the caller, never trusted for anything else.
    caller_run: UUID | None = None


class MediaQuote(WireModel):
    """What a job costs, platform fee included, before it runs: a range when its size is only known once made."""

    provider: str
    model: str
    low_usd: float = Field(ge=0)
    high_usd: float = Field(ge=0)


class MediaFile(WireModel):
    """One file a job made: the name it is saved under, what it is, its bytes (base64)."""

    name: str = Field(min_length=1, max_length=200)
    media_type: str
    data_base64: str
    size_bytes: int = Field(ge=0)


class MediaAnswer(WireModel):
    kind: MediaKind
    quote: MediaQuote
    #: The run that made it; None for a quote.
    run_id: UUID | None = None
    #: Provider cost + platform fee, as the journal lists it; None for a quote or a job its provider gave no usage for.
    cost_usd: float | None = None
    files: tuple[MediaFile, ...] = ()
    #: The job's text when it makes one (a translation): the same as its file's.
    text: str | None = None
