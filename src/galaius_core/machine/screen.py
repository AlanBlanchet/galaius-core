"""One computer's screen as a person watches it from the web: the rectangles the OS reports
(`ScreenRect`, `ScreenMonitor`), what a viewer asks to see (`ScreenSpec`), one encoded frame
(`ScreenFrame`), what the stream cost (`ScreenStats`), and the acts a viewer sends back
(`SCREEN_ACTS`).

Units: every coordinate a viewer sends is in the PIXELS OF THE FRAME IT READ (`ScreenPoint.frame`
names it), and the computer maps them to its own input units from that frame's own origin and
scale — a viewer never multiplies, so a resized or re-scaled stream can never land a click
somewhere else.
"""

import base64
import hashlib
import statistics
from datetime import UTC, datetime, timedelta
from typing import Annotated, ClassVar, Literal, Self, Union
from uuid import UUID

from pydantic import ConfigDict, Field, TypeAdapter, computed_field

from galaius_core.wire import WireModel, WireRequest

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

    def sharper(self, cap: "ScreenSpec | None" = None) -> Self:
        """More bytes per frame: the link carried the last ones well inside its budget — never past
        `cap` on either ladder (what the viewer asked for)."""
        richer = self._stepped(+1)
        if cap is None:
            return richer
        return richer.model_copy(update={"width": min(richer.width, cap.width), "quality": min(richer.quality, cap.quality)})

    @property
    def level(self) -> tuple[int, int]:
        """Where this spec stands on the two ladders: what tells one step from another."""
        return (self.width, self.quality)


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
    #: From a frame leaving the PC to the viewer's ack of it drawn: the link both ways plus the
    #: viewer's decode and paint — the only number that includes the network.
    send: Timing = Timing()
    #: The viewer's own decode-and-paint time per frame, as it reported it (`ScreenViewerMessage.drawn_ms`).
    drawn: Timing = Timing()
    width: int = 0
    quality: int = 0
    #: How many times the link made the stream step down, and back up.
    lightened: int = 0
    sharpened: int = 0

    @property
    def seconds(self) -> float:
        return (datetime.now(UTC) - self.started_at).total_seconds()

    @computed_field
    @property
    def fps(self) -> float:
        """Frames a second over the session's whole life — 0 until there are two frames: one frame
        in a session a millisecond old is not a rate, and reporting one reads like a measurement."""
        return round((self.frames - 1) / self.seconds, 2) if self.frames > 1 and self.seconds > 0 else 0.0

    @computed_field
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


#: Why a computer refused or ended a view, typed: the page words it in its reader's language, which
#: the computer cannot know (its own locale is not the page's), and the computer's English sentence
#: stays beside it for the logs (`MachineAgentAnswer.error`, `ScreenAllowance.why`, `ScreenEnded.why`).
#: Three situations, three types: a switch that is off, an offer refused, a view that ended.
ScreenOffReason = Literal[
    "sharing_off",      # its switch is off: never on, `galaius screen off`, or its hours ran out
    "stopped_here",     # Stop on its badge, or Ctrl+Alt+Shift+S
]
ScreenOpenRefusal = Literal[
    "sharing_off",      # its switch is off
    "passkey_new",      # a passkey it has not pinned: `galaius screen trust <id>` typed there lets it watch
    "passkey_unknown",  # a passkey it does not trust (forgotten there, or another browser's)
    "offer_expired",    # the offer ran out, or asked for more than `SCREEN_SESSION_MAX`
    "offer_used",       # this offer already opened a view: a replay
    "wrong_site",       # the offer or the passkey came from another address than the one it serves
    "not_verified",     # the passkey was used without its owner verified on the device
    "refused",          # malformed, forged, or for another computer
]
ScreenEndReason = Literal[
    ScreenOffReason,    # its switch went off during the view, or Stop on its badge
    "minutes_over",     # the minutes its offer named are over
    "no_badge",         # it could not show that it is being watched, so it is not
    "message_lost",     # a viewer message was lost on the way: the view ended rather than act out of order
    "viewer_gone",      # no frame taken for a while: a phone locked, a tab closed
    "link_lost",        # its link to the server stopped carrying frames
    "failed",           # something on it failed under the view (its display, its capture)
    "newer_view",       # a newer view of this computer was opened
    "ended",            # the viewer ended it (or the server, on the viewer's behalf)
]
#: Every word a page has to say about it, for one table of wordings.
ScreenRefusalReason = ScreenOffReason | ScreenOpenRefusal | ScreenEndReason


class ScreenAllowance(WireModel):
    """Whether this computer lets its owner watch it, as the computer itself decides.

    The computer, not the server, holds this: `allowed` False refuses every frame whatever the
    server sends, and `until` expires an allowance the owner forgot to switch off."""

    allowed: bool = False
    #: When the allowance runs out (None: no limit while it is on).
    until: datetime | None = None
    #: Why it is off, when it is: typed for the page to word (`ScreenOffReason`), and the
    #: computer's own sentence.
    reason: ScreenOffReason | None = None
    why: str = ""

    #: The default life of an allowance the owner switches on: long enough for an evening's work,
    #: short enough that a forgotten one is not a standing invitation.
    DEFAULT_HOURS: ClassVar[int] = 8

    def live(self, now: datetime | None = None) -> bool:
        """True while this allowance still stands."""
        return self.allowed and (self.until is None or self.until > (now or datetime.now(UTC)))


# --- Who may watch: the viewer's passkey, checked by the computer itself --------------------------
#
# The server carries every byte of a session and decides none of it. The computer pins its owner's
# passkey (`ScreenPasskey`; the first one while its own switch is on, any later one only when
# `galaius screen trust` is run AT the computer) and opens a session only for an offer that passkey
# signed (`ScreenOffer`, `ScreenAssertion`). The offer carries the viewer's ephemeral key, the
# computer answers with its own (`ScreenOpened`), and both derive the session's keys (`ScreenKeys`):
# every frame and every act is sealed end to end (`ScreenSealed`), so the server relays ciphertext
# it can neither read nor forge, and a replayed act is refused by its sequence number.

#: COSE algorithm numbers a pinned passkey may use (WebAuthn `getPublicKeyAlgorithm()`): ES256,
#: EdDSA, RS256 — what every platform authenticator registers.
SCREEN_PASSKEY_ALGORITHMS: tuple[int, ...] = (-7, -8, -257)
#: The longest one signed offer lives: a phone left on the page is not a standing grant, and
#: watching longer asks the passkey again.
SCREEN_SESSION_MAX = timedelta(minutes=30)
#: Clock difference between the viewer's device and the computer an offer may carry.
SCREEN_CLOCK_SKEW = timedelta(minutes=2)
#: Ceiling of the offer text the passkey signs (it is a few hundred bytes).
SCREEN_OFFER_MAX = 4096
#: Ceiling of one sealed viewer message (64 acts of 8000 characters of text, sealed).
SCREEN_INPUT_MAX = 600 * 1024
#: HKDF ``info`` per direction (HKDF-SHA256, salt = the offer's challenge, 32-byte AES-GCM keys).
SCREEN_KEY_INFO: dict[str, bytes] = {"pc": b"galaius screen v1 pc", "viewer": b"galaius screen v1 viewer"}
#: First byte of each sealed message's IV, per direction (`ScreenSealed.iv`).
SCREEN_IV_DIRECTION: dict[str, int] = {"pc": 1, "viewer": 2}
#: What the computer's identity key signs to prove an answer is its own (`ScreenOpened.proof`).
SCREEN_PROOF_LABEL = b"galaius screen v1 opened"
#: First byte of a plaintext the computer seals (`ScreenSealed`): a frame, or the stream's numbers.
SCREEN_PLAIN_FRAME, SCREEN_PLAIN_STATS = 1, 2


def b64url(raw: bytes) -> str:
    """base64url without padding: WebAuthn's own spelling of bytes, and the browser's."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def unb64url(text: str) -> bytes:
    """`b64url` read back (padding optional); raises ValueError on anything else."""
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, TypeError) as error:
        raise ValueError("not base64url") from error


B64Url = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]*$", max_length=8192)]


class ScreenPasskey(WireModel):
    """One passkey of the owner as a computer pins it: `credential_id` (WebAuthn ``rawId``),
    `public_key` (``getPublicKey()``: DER SubjectPublicKeyInfo), `algorithm` (COSE number). Sent
    with an offer only when the browser just created it (`ScreenAssertion.passkey`); the computer
    then trusts it on its own terms (`ScreenTrust`), never because the server relayed it."""

    credential_id: B64Url
    public_key: B64Url
    algorithm: int
    label: Annotated[str, Field(max_length=80)] = ""
    #: The highest authenticator counter seen (0: the authenticator keeps none, as synced passkeys do).
    counter: int = 0
    added_at: datetime | None = None


class ScreenTrust(WireModel):
    """The passkeys a computer opens sessions for (`trusted`), and those offered since that wait
    for its owner AT the computer (`pending`, `galaius screen trust`)."""

    trusted: tuple[ScreenPasskey, ...] = ()
    pending: tuple[ScreenPasskey, ...] = ()

    #: Pending passkeys kept: the server could offer junk, never fill a disk with it.
    PENDING_MAX: ClassVar[int] = 8


class ScreenOffer(WireModel):
    """What the viewer's passkey signs, as JSON TEXT: the browser sends the exact string it signed
    and the computer parses that string, so no canonical form exists to disagree on.

    `machine_id` binds it to one computer, `nonce` to one use, `expires_at` (at most
    `SCREEN_SESSION_MAX` ahead) to one sitting, `mode` to what was asked (control is its own
    offer, its own passkey press), `viewer_key` to the browser holding the matching private key,
    `origin` to the page that asked (it must be the computer's own server address)."""

    version: Literal[1] = 1
    machine_id: UUID
    nonce: Annotated[B64Url, Field(min_length=22, max_length=64)]
    mode: Literal["view", "control"] = "view"
    expires_at: datetime
    #: The viewer's ephemeral ECDH P-256 public key, raw uncompressed (65 bytes), base64url.
    viewer_key: Annotated[B64Url, Field(min_length=86, max_length=88)]
    origin: Annotated[str, Field(max_length=200)]
    spec: ScreenSpec = ScreenSpec()

    @staticmethod
    def challenge(text: str) -> bytes:
        """The WebAuthn challenge for the offer `text`: SHA-256 of its UTF-8 bytes."""
        return hashlib.sha256(text.encode("utf-8")).digest()


class ScreenAssertion(WireModel):
    """``navigator.credentials.get()``'s answer to the offer's challenge, every field base64url."""

    credential_id: B64Url
    authenticator_data: B64Url
    client_data_json: B64Url
    signature: B64Url
    #: Only when this browser created the passkey just now: its public key, for the computer to pin
    #: (first passkey while its switch is on) or hold for its owner (`ScreenTrust.pending`).
    passkey: ScreenPasskey | None = None


class ScreenOpenBody(WireRequest):
    """What the browser posts to open a session: the offer text it signed and the assertion."""

    offer: Annotated[str, Field(min_length=2, max_length=SCREEN_OFFER_MAX)]
    assertion: ScreenAssertion


class ScreenOpened(WireModel):
    """The computer's answer to a verified offer: the session, its own ephemeral key (`pc_key`,
    raw P-256, base64url), and `proof` — its identity key (`identity`, raw P-256) signing
    ``SCREEN_PROOF_LABEL || challenge || pc_key || session.bytes`` (ECDSA P-256 SHA-256, r||s, as
    WebCrypto verifies). A browser pins `identity` per computer the first time, so a server that
    swaps `pc_key` is caught: it cannot sign for the computer."""

    session: UUID
    pc_key: B64Url
    identity: B64Url
    proof: B64Url
    mode: Literal["view", "control"]
    ends_at: datetime
    monitors: tuple[ScreenMonitor, ...] = ()
    #: True when this offer's passkey was pinned just now (the computer's first).
    paired: bool = False


class ScreenSealed(WireModel):
    """One sealed message of a session, as the machine link carries it (`type` ``screen_frame``
    PC → server, ``screen_input`` server → PC): AES-256-GCM under the direction's key, IV =
    direction byte (`SCREEN_IV_DIRECTION`), 3 zero bytes, `seq` as 8 bytes big-endian; AAD = the
    session id's 16 bytes. `seq` starts at 1 and only grows: a receiver drops anything not above
    the last it opened, so a replay or a reordering by whoever relays is refused, never acted on.

    The browser's own socket carries the same thing as one binary message: `seq` (8 bytes
    big-endian) then `data`."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64")

    session: UUID
    seq: Annotated[int, Field(ge=1, lt=2**63)]
    data: bytes

    def iv(self, direction: Literal["pc", "viewer"]) -> bytes:
        return bytes([SCREEN_IV_DIRECTION[direction], 0, 0, 0]) + self.seq.to_bytes(8, "big")

    def binary(self) -> bytes:
        """The browser socket's spelling."""
        return self.seq.to_bytes(8, "big") + self.data

    @classmethod
    def from_binary(cls, session: UUID, raw: bytes) -> Self:
        if len(raw) < 8 + 16:
            raise ValueError("a sealed message is its sequence, then at least a GCM tag")
        return cls(session=session, seq=int.from_bytes(raw[:8], "big"), data=raw[8:])


class ScreenEnded(WireModel):
    """The computer ended a view (`type` ``screen_ended`` on its machine link): which, and why — typed
    for the page to word (`reason`), with its own English sentence for the logs (`why`)."""

    type: Literal["screen_ended"] = "screen_ended"
    session: UUID
    reason: ScreenEndReason
    why: str = Field(default="", max_length=400)


class ScreenViewerMessage(WireRequest):
    """What a viewer seals back (`ScreenSealed`, direction ``viewer``): the frame it drew (`ack`,
    with `drawn_ms` its own decode-and-paint time) and / or acts to take (control sessions only)."""

    ack: int | None = None
    drawn_ms: Annotated[float, Field(ge=0, le=60_000)] | None = None
    acts: Annotated[tuple[dict, ...], Field(max_length=SCREEN_ACTS_MAX)] = ()


class ScreenReadiness(WireModel):
    """What a computer says about being watched, before any passkey is asked: whether its owner
    switched it on there (`allowance`), what it can show, who it would open for, and whether
    someone is watching right now."""

    allowance: ScreenAllowance = ScreenAllowance()
    monitors: tuple[ScreenMonitor, ...] = ()
    viewing: bool = False
    #: Why it cannot be watched at all (no desktop, Wayland, a capture library missing); "" when it can.
    unsupported: str = ""
    stats: ScreenStats | None = None
    #: The passkeys it opens sessions for (none yet: the first offer while its switch is on pairs).
    passkeys: tuple[ScreenPasskey, ...] = ()
    #: Passkeys offered that wait for `galaius screen trust` at the computer.
    pending: int = 0
    #: Its identity key (raw P-256, base64url) a browser pins and checks `ScreenOpened.proof` with.
    identity: str = ""
    #: The page origin its offers must come from, and the WebAuthn relying party it checks.
    origin: str = ""
    rp_id: str = ""
