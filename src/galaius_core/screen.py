"""One computer's screen as a person watches it from the web: the rectangles the OS reports
(`ScreenRect`, `ScreenMonitor`), what a viewer asks to see (`ScreenSpec`), one encoded frame
(`ScreenFrame`), what the stream cost (`ScreenStats`), and the acts a viewer sends back
(`SCREEN_ACTS`).

Units: every coordinate a viewer sends is in the PIXELS OF THE FRAME IT READ (`ScreenPoint.frame`
names it), and the computer maps them to its own input units from that frame's own origin and
scale — a viewer never multiplies, so a resized or re-scaled stream can never land a click
somewhere else.
"""

import statistics
from datetime import UTC, datetime
from typing import Annotated, ClassVar, Literal, Self, Union

from pydantic import ConfigDict, Field, TypeAdapter

from .wire import WireModel, WireRequest

#: What a runner answers screen requests with (`MachineRunner.FEATURES`): without it, its PC is too
#: old to be viewed and the web says so instead of waiting on frames that never come.
SCREEN_FEATURE = "screen_view"

#: The widths a stream steps through as the link gets worse or better, narrowest first. A phone
#: screen is ~390 CSS px wide at 2-3x, so 1280 already over-serves it; 480 keeps a desktop readable
#: as text blocks while a tethered link recovers.
SCREEN_WIDTHS: tuple[int, ...] = (480, 720, 960, 1280, 1600, 1920)

#: JPEG qualities the same ladder walks (Pillow's scale). Below 35 text stops being readable, so a
#: failing link loses size before it loses letters.
SCREEN_QUALITIES: tuple[int, ...] = (35, 45, 55, 65, 75)

#: One frame's hard ceiling on the wire. 1920-wide JPEG at quality 75 measures ~210 KB, so this is
#: 5x the worst real frame: it bounds a malformed or hostile sender, never a true frame.
SCREEN_FRAME_MAX = 1024 * 1024


class ScreenRect(WireModel):
    """An integer rectangle in desktop pixels (the unit input is sent in), origin top-left of the
    primary monitor — so a monitor left of it has a negative ``x``."""

    x: int
    y: int
    w: int
    h: int

    @property
    def geometry(self) -> str:
        """``WxH+X+Y``, the X11 geometry spelling agents and maim already read."""
        return f"{self.w}x{self.h}{self.x:+d}{self.y:+d}"


class ScreenMonitor(ScreenRect):
    """One physical monitor: ``index`` is its position in the OS's own list (``screen:<index>``),
    ``name`` the connector (``DP-1``) where the OS has one, else ``monitor<index>``."""

    index: int
    name: str


class ScreenSpec(WireRequest):
    """What a viewer asks to see: one monitor (`monitor`, its OS index) or the whole desktop
    (None), no wider than `width`, at `quality`, no faster than `max_fps`.

    A stream never holds a viewer to what it asked: `lighter` and `sharper` walk the ladders
    (`SCREEN_WIDTHS`, `SCREEN_QUALITIES`) as the link it measures gets worse or better, so a
    tethered phone keeps a live picture instead of a frozen sharp one."""

    monitor: int | None = None
    width: Annotated[int, Field(ge=SCREEN_WIDTHS[0], le=SCREEN_WIDTHS[-1])] = 1280
    quality: Annotated[int, Field(ge=SCREEN_QUALITIES[0], le=SCREEN_QUALITIES[-1])] = 55
    max_fps: Annotated[float, Field(gt=0, le=30)] = 10.0

    def _stepped(self, by: int) -> Self:
        """The next spec along both ladders (`by` -1 = cheaper, +1 = richer), clamped at each end.
        Width moves first and quality follows it: at a given byte budget a wider blurrier picture
        reads worse than a narrower sharp one."""
        width = SCREEN_WIDTHS[min(max(SCREEN_WIDTHS.index(self.nearest_width) + by, 0), len(SCREEN_WIDTHS) - 1)]
        quality = SCREEN_QUALITIES[min(max(SCREEN_QUALITIES.index(self.nearest_quality) + by, 0), len(SCREEN_QUALITIES) - 1)]
        return self.model_copy(update={"width": width, "quality": quality})

    @property
    def nearest_width(self) -> int:
        return min(SCREEN_WIDTHS, key=lambda step: abs(step - self.width))

    @property
    def nearest_quality(self) -> int:
        return min(SCREEN_QUALITIES, key=lambda step: abs(step - self.quality))

    def lighter(self) -> Self:
        """Fewer bytes per frame: the link could not carry the last one in its frame budget."""
        return self._stepped(-1)

    def sharper(self) -> Self:
        """More bytes per frame: the link carried the last one well inside its frame budget."""
        return self._stepped(+1)

    @property
    def frame_budget_ms(self) -> float:
        """How long one frame may take end to end at `max_fps` — the number the adaptation reads."""
        return 1000.0 / self.max_fps


class ScreenFrame(WireModel):
    """One encoded picture of a screen: JPEG bytes and the rectangle they show.

    JPEG, every frame a keyframe, no inter-frame codec: Pillow encodes 1280-wide in ~3 ms where
    WebP takes ~18 ms and PNG ~32 ms (measured 2026-10-23, Pillow 12.2, 1920x1080), every browser
    and phone decodes it with no WebCodecs, and a dropped frame costs nothing because the next one
    stands alone. `scale` is image pixels per input unit, `origin_*` the captured rectangle's
    top-left in input units: a viewer reports a click in THIS frame's pixels and the computer maps
    it back with them."""

    #: JPEG bytes are base64 on the wire, both ways: the default (UTF-8) cannot encode a JPEG at
    #: all — `model_dump(mode="json")` raised `UnicodeDecodeError` on the very first frame.
    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    sequence: int
    at: datetime
    origin_x: int
    origin_y: int
    w: int
    h: int
    scale: float = 1.0
    quality: int = 0
    #: Where the computer's own pointer is, in this frame's pixels (None: the OS would not say). The
    #: capture APIs leave the cursor out of the picture, so the viewer draws it from this.
    pointer_x: int | None = None
    pointer_y: int | None = None
    jpeg: bytes = b""

    def input_point(self, x: int, y: int) -> tuple[int, int]:
        """A point in this frame's pixels, as the computer's own input units."""
        return (self.origin_x + round(x / self.scale), self.origin_y + round(y / self.scale))


class Timing(WireModel):
    """What one step of a stream cost, in milliseconds, over the samples it kept."""

    count: int = 0
    p50: float = 0.0
    p95: float = 0.0
    worst: float = 0.0

    @classmethod
    def of(cls, samples: list[float]) -> Self:
        """`samples` summarised; all zeros when there are none. p95 of fewer than 20 samples is
        their worst — a percentile needs the samples to exist, never an interpolation that reads
        like a measurement."""
        if not samples:
            return cls()
        ordered = sorted(samples)
        return cls(count=len(ordered), p50=round(statistics.median(ordered), 2),
                   p95=round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 2), worst=round(ordered[-1], 2))


class ScreenStats(WireModel):
    """What one screen session cost and delivered, as its PC measured it — the numbers a
    performance review reads: per step (grab, encode, send) and overall."""

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    frames: int = 0
    #: Frames the PC did not send because the screen had not changed: bandwidth the link never paid.
    unchanged: int = 0
    bytes_sent: int = 0
    grab: Timing = Timing()
    encode: Timing = Timing()
    #: Server-acknowledged time per frame: the only number that includes the link itself.
    send: Timing = Timing()
    width: int = 0
    quality: int = 0
    #: How many times the link made the stream step down, and back up.
    lightened: int = 0
    sharpened: int = 0

    @property
    def seconds(self) -> float:
        return (datetime.now(UTC) - self.started_at).total_seconds()

    @property
    def fps(self) -> float:
        """Frames a second over the session's whole life — 0 until there are two frames: one frame
        in a session a millisecond old is not a rate, and reporting one reads like a measurement."""
        return round((self.frames - 1) / self.seconds, 2) if self.frames > 1 and self.seconds > 0 else 0.0

    @property
    def kbps(self) -> float:
        return round(self.bytes_sent * 8 / 1000 / self.seconds, 1) if self.frames > 1 and self.seconds > 0 else 0.0

    @property
    def said(self) -> str:
        """One line a person reads: the rate, the picture and the link's own share of a frame."""
        return (f"{self.fps} fps at {self.width}px q{self.quality}, {self.kbps} kbit/s, "
                f"frame p50 {self.send.p50} ms p95 {self.send.p95} ms, {self.unchanged} unchanged")


class ScreenPoint(WireRequest):
    """Where a viewer pointed, in the pixels of the frame it read (`frame`: that frame's
    `sequence`). The computer maps it with that frame's own origin and scale."""

    x: int
    y: int
    frame: int = 0


class ScreenMove(ScreenPoint):
    type: Literal["move"] = "move"


class ScreenClick(ScreenPoint):
    type: Literal["click"] = "click"
    button: Literal["left", "right", "middle"] = "left"
    count: Annotated[int, Field(ge=1, le=3)] = 1


class ScreenScroll(ScreenPoint):
    type: Literal["scroll"] = "scroll"
    clicks: Annotated[int, Field(ge=-20, le=20)] = 0
    horizontal: bool = False


class ScreenKey(WireRequest):
    """One key or chord, spelled as every other galaius action spells it (`super`, `alt+tab`)."""

    type: Literal["key_press"] = "key_press"
    key: Annotated[str, Field(min_length=1, max_length=64)]


class ScreenText(WireRequest):
    """Text typed into whatever has focus on that computer."""

    type: Literal["type_text"] = "type_text"
    text: Annotated[str, Field(max_length=8000)]


ScreenAct = Union[ScreenMove, ScreenClick, ScreenScroll, ScreenKey, ScreenText]
#: The acts a viewer may send, by their galaius action names: one validator for the whole vocabulary.
SCREEN_ACTS: TypeAdapter[ScreenAct] = TypeAdapter(Annotated[ScreenAct, Field(discriminator="type")])

#: How many acts one request may carry: a drag is a short burst of moves, never a thousand.
SCREEN_ACTS_MAX = 64


class ScreenAllowance(WireModel):
    """Whether this computer lets its owner watch it, as the computer itself decides.

    The computer, not the server, holds this: `allowed` False refuses every frame whatever the
    server sends, and `until` expires an allowance the owner forgot to switch off."""

    allowed: bool = False
    #: When the allowance runs out (None: no limit while it is on).
    until: datetime | None = None
    #: Why it is off, in the owner's own words, when it is.
    why: str = ""

    #: The default life of an allowance the owner switches on: long enough for an evening's work,
    #: short enough that a forgotten one is not a standing invitation.
    DEFAULT_HOURS: ClassVar[int] = 8

    def live(self, now: datetime | None = None) -> bool:
        """True while this allowance still stands."""
        return self.allowed and (self.until is None or self.until > (now or datetime.now(UTC)))


class ScreenReadiness(WireModel):
    """What a computer says about being watched: whether it is allowed, what it can show, and
    whether someone is watching right now."""

    allowance: ScreenAllowance = ScreenAllowance()
    monitors: tuple[ScreenMonitor, ...] = ()
    viewing: bool = False
    #: Why it cannot be watched at all (no desktop, Wayland, a capture library missing); "" when it can.
    unsupported: str = ""
    stats: ScreenStats | None = None
