"""A coding-agent subscription (a vendor login: Claude, Codex, ...) as the web and a PC share it: its
caps, its usage windows, a run's allowance past a cap. The credential itself never crosses the wire:
it stays in the subscription's own folder on the PC that signed it in."""

from datetime import datetime
from typing import ClassVar, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .wire import Kinded, WireModel, WireRequest

#: The coding-agent CLIs a PC runs agents through, each a vendor whose login a subscription is.
AgentProvider = Literal["claude", "codex"]
#: What a PC's runner says (its hello's `features`) once it takes every request below: a start carries one
#: subscription per provider (`SubscriptionChoices`), a refusal says why typed (`SubscriptionRefusal`).
SUBSCRIPTIONS_FEATURE = "agent_subscription_choices"
#: How many subscriptions one account keeps on one PC (all of them a start may offer).
SUBSCRIPTIONS_PER_PC = 8
#: The usage periods a vendor meters a subscription over. A vendor's own token maps onto the nearest
#: (`UsageWindow.nearest`): Claude `seven_day` is `weekly`, Codex `window_minutes: 10080` is `weekly`.
UsageWindowName = Literal["five_hour", "weekly", "monthly"]
#: What a run does when its subscription reaches a cap: `finish` lets the current turn end and starts
#: no new turn or session on it until the window drops; `cut` stops the run at that reading.
AtCap = Literal["finish", "cut"]
#: What a PC's runner says once it takes a `RunMove` that does not resume (`resume` False).
RUN_MOVE_WAITS_FEATURE = "agent_subscription_moves_wait"
#: What a PC's runner says once it takes a `RunMove` from or to its own login of the run's CLI (`subscription_id` None).
RUN_MOVE_LOGIN_FEATURE = "agent_login_moves"


class UsageWindow(WireModel):
    """One usage period of a subscription as its vendor last reported it, on the PC running it."""

    window: UsageWindowName
    #: Share of the period's allowance used (vendor scale 0-100; a vendor past its pool may say more).
    used_percent: float = Field(ge=0)
    resets_at: datetime | None = None
    observed_at: datetime

    MINUTES: ClassVar[dict[UsageWindowName, int]] = {"five_hour": 5 * 60, "weekly": 7 * 24 * 60, "monthly": 30 * 24 * 60}

    @classmethod
    def nearest(cls, minutes: float) -> UsageWindowName:
        """The window whose length is closest to `minutes` (a vendor naming its period by length)."""
        return min(cls.MINUTES, key=lambda name: abs(cls.MINUTES[name] - minutes))


class SubscriptionCap(WireModel):
    """A cap a subscription reached: which window, the reading, the cap, what the run did
    (`at_cap`), and when the window reopens. The server words it (« coupé à 80 % de la semaine »)."""

    window: UsageWindowName
    used_percent: float = Field(ge=0)
    cap_percent: int = Field(ge=1, le=100)
    at_cap: AtCap
    resets_at: datetime | None = None


class SubscriptionAllowance(WireModel):
    """A run's owner letting it go past its subscription's cap on one window (the one it was held or
    cut at), up to `up_to_percent` (the vendor's 100 % stays the ceiling)."""

    window: UsageWindowName
    up_to_percent: int = Field(ge=1, le=100)


class SubscriptionLimits(WireModel):
    """The owner's caps on one subscription, in percent of each window (None: no cap, the vendor's
    own 100 % stays), and what a run does on reaching one."""

    five_hour_percent: int | None = Field(default=None, ge=1, le=100)
    weekly_percent: int | None = Field(default=None, ge=1, le=100)
    at_cap: AtCap = "finish"

    def cap(self, window: UsageWindowName, allowance: SubscriptionAllowance | None = None) -> int | None:
        """The cap in force on `window`; a run's `allowance` raises its own window's cap, never sets one."""
        own = {"five_hour": self.five_hour_percent, "weekly": self.weekly_percent}.get(window)
        return own if own is None or allowance is None or allowance.window != window else max(own, allowance.up_to_percent)

    def reached(self, usage: tuple[UsageWindow, ...], allowance: SubscriptionAllowance | None = None) -> SubscriptionCap | None:
        """The first window of `usage` at or past its cap (the highest share first), else None."""
        for reading in sorted(usage, key=lambda item: item.used_percent, reverse=True):
            if (cap := self.cap(reading.window, allowance)) is not None and reading.used_percent >= cap:
                return SubscriptionCap(window=reading.window, used_percent=reading.used_percent, cap_percent=cap,
                                       at_cap=self.at_cap, resets_at=reading.resets_at)
        return None


class Lent(WireRequest):
    """`lent`: the initiator is a member using a PC lent to him: no CLI login of that PC runs it (its owner's or a
    subscription's); only API-key and local models do. Asked of a runner saying `SUBSCRIPTIONS_FEATURE`."""

    lent: bool = Field(default=False, exclude_if=lambda value: value is False)

    @property
    def names_subscription(self) -> bool:
        """It asks for a subscription (a session or continued editor conversation, the default login's only, never does)."""
        return False


class SubscriptionUse(Lent):
    """Which subscription a turn or a continued conversation runs on, with the owner's caps on it now (the server's
    record; the PC checks them again before anything starts). Absent: the PC's own default login, as before."""

    subscription_id: UUID | None = Field(default=None, exclude_if=lambda value: value is None)
    limits: SubscriptionLimits | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def limits_of_a_subscription(self) -> Self:
        if self.limits is not None and self.subscription_id is None:
            raise ValueError("limits belong to a subscription: name its subscription_id")
        if self.lent and self.subscription_id is not None:
            raise ValueError("a subscription is never lent: a member's run on a lent PC names none")
        return self

    @property
    def names_subscription(self) -> bool:
        return self.subscription_id is not None


class SubscriptionChoice(WireModel):
    """The subscription a start runs on when its launch walk reaches a candidate of `provider`, with its caps now."""

    provider: AgentProvider
    subscription_id: UUID
    limits: SubscriptionLimits


class SubscriptionChoices(Lent):
    """The starter's own signed-in subscriptions a start may run on, in his order. The launch walk binds each candidate to
    the first of its provider's that is under its caps (read afresh on the PC), passing over a capped one saying so; a
    provider named none runs on the PC's own login; once a candidate starts, the run is pinned to the subscription it
    started on (`RunSubscription`), never two inside one run."""

    subscriptions: tuple[SubscriptionChoice, ...] = Field(default=(), max_length=SUBSCRIPTIONS_PER_PC, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def each_once(self) -> Self:
        if len({choice.subscription_id for choice in self.subscriptions}) != len(self.subscriptions):
            raise ValueError("a start names each subscription once")
        if self.lent and self.subscriptions:
            raise ValueError("a subscription is never lent: a member's run on a lent PC names none")
        return self

    def of(self, provider: str) -> tuple[SubscriptionChoice, ...]:
        """`provider`'s, in his order."""
        return tuple(choice for choice in self.subscriptions if choice.provider == provider)

    @property
    def names_subscription(self) -> bool:
        return bool(self.subscriptions)


#: Why a PC (or the server) refused something about a subscription: at a cap (`cap` says which), a member's run on a lent
#: PC, a subscription it holds no folder of, a turn or child naming another than its run's, a session or continued editor
#: conversation (the default login's only), a turn of a run whose subscription was removed (or is not the sender's).
#: `usage_limit`: the vendor's own usage limit stopped the run and its window has not reopened yet (`resets_at`); `session_missing`:
#: the session to continue elsewhere is not in its subscription's folder.
SubscriptionRefusalReason = Literal["capped", "lent", "not_signed_in", "pinned_elsewhere", "default_login_only", "subscription_removed", "usage_limit", "session_missing"]


class SubscriptionRefusal(Kinded):
    """A PC's refusal about a subscription, typed: the server words it (`reason`, `cap` when `capped`, `resets_at` when
    the vendor's `usage_limit` window reopens). One kind of `MachineRefusal`."""

    kind: Literal["subscription"] = "subscription"
    reason: SubscriptionRefusalReason
    cap: SubscriptionCap | None = None
    resets_at: datetime | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def cap_of_capped(self) -> Self:
        if (self.reason == "capped") != (self.cap is not None):
            raise ValueError("a cap refusal names its cap, and only it does")
        if self.resets_at is not None and self.reason != "usage_limit":
            raise ValueError("only a usage limit refusal names when its window reopens apart from a cap")
        return self


class RunMove(WireRequest):
    """A run of his continues its session on `subscription_id`, another of his own subscriptions of the same provider on
    the same PC, under `limits`; None: on that PC's own login of its CLI (contract A9: the PC carries the session's
    transcript over, never a credential, and pins the run there; no login is signed out). Never between accounts.

    `resume`: the PC resumes it there at once (his « Continuer sur … », the switch at a cap); False: it only moves, and
    its next turn runs there (his pick in a conversation, the switch before a turn). Asked False only of a runner saying
    `RUN_MOVE_WAITS_FEATURE`; from or to the PC's own login only of one saying `RUN_MOVE_LOGIN_FEATURE`."""

    run_id: UUID
    subscription_id: UUID | None
    limits: SubscriptionLimits | None = Field(default=None, exclude_if=lambda value: value is None)
    resume: bool = Field(default=True, exclude_if=lambda value: value)

    @model_validator(mode="after")
    def limits_of_a_subscription(self) -> Self:
        if (self.limits is None) != (self.subscription_id is None):
            raise ValueError("a move to a subscription carries its caps (limits), one to the PC's own login none")
        return self


class RunSubscription(SubscriptionUse):
    """The subscription a run is pinned to, as the PC records and reports it: every turn and child
    runs on it (`lent`: on none of the PC's CLI logins), under `limits`, raised on one window for this
    run only by `allowance`; `capped` is the cap reached while it ran (held between turns at
    `finish`, stopped at `cut`)."""

    allowance: SubscriptionAllowance | None = None
    capped: SubscriptionCap | None = None

    @model_validator(mode="after")
    def pinned_somewhere(self) -> Self:
        if self.subscription_id is None and not self.lent:
            raise ValueError("a run is pinned to a subscription, or lent: a default-login run carries none")
        if self.subscription_id is not None and self.limits is None:
            raise ValueError("a run on a subscription carries its caps (limits; no cap: SubscriptionLimits())")
        if self.lent and (self.allowance is not None or self.capped is not None):
            raise ValueError("a lent run runs on no subscription: it has no cap to reach or go past")
        return self


class RunAllowance(WireRequest):
    """The run's owner lets one run held or cut at its subscription's cap go past it, up to
    `up_to_percent` of the window it was capped on (the vendor's 100 % stays the ceiling); it dies
    with the run."""

    run_id: UUID
    up_to_percent: int = Field(ge=1, le=100)


class LoginIdentity(WireModel):
    """Which vendor login a PC's CLI is signed in to and as whom, as its own status command says: the server tells one
    login from another by `login` (the same vendor account in two folders or on two PCs is one login)."""

    #: Who it is signed in as, as its status command says (an e-mail, an organisation), once signed in.
    account: str = Field(default="", max_length=200)
    #: Which login that is: a digest of the account its status answer names, equal for the same vendor login wherever it
    #: is signed in (the server refuses one login on two accounts); "" when not signed in, or not said (an older PC).
    login: str = Field(default="", pattern=r"^([0-9a-f]{16})?$", exclude_if=lambda value: not value)


class SubscriptionUsage(LoginIdentity):
    """One subscription's last usage reading on a PC, and the login its folder holds now."""

    subscription_id: UUID
    windows: tuple[UsageWindow, ...] = Field(default=(), max_length=8)


class LoginUsage(LoginIdentity):
    """The last usage reading of one program's default login on a PC (the login its owner's own editor
    sessions use, outside any subscription), and which login that is; no window: none now (signed out, not
    installed, a login its vendor meters none for)."""

    provider: AgentProvider
    windows: tuple[UsageWindow, ...] = Field(default=(), max_length=8)


class SubscriptionUsageReport(WireModel):
    """What a PC pushes to its server when the usage of a subscription (`usage`) or of a program's default
    login (`logins`) moved (`due`)."""

    type: Literal["subscription_usage"] = "subscription_usage"
    usage: tuple[SubscriptionUsage, ...] = Field(default=(), max_length=64)
    logins: tuple[LoginUsage, ...] = Field(default=(), max_length=8)

    REPORT_POINTS: ClassVar[float] = 1.0
    REPORT_SECONDS: ClassVar[float] = 60.0

    @classmethod
    def due(cls, usage: tuple[UsageWindow, ...], reported: tuple[UsageWindow, ...], reported_at: datetime | None, now: datetime) -> bool:
        """Whether `usage` goes to the server now, `reported` at `reported_at` last: a window moved `REPORT_POINTS`
        or more, appeared or went, or anything changed and `REPORT_SECONDS` passed since that report."""
        if usage == reported:
            return False
        last = {window.window: window.used_percent for window in reported}
        moved = {window.window for window in usage} != set(last) or any(abs(window.used_percent - last[window.window]) >= cls.REPORT_POINTS for window in usage)
        return moved or reported_at is None or (now - reported_at).total_seconds() >= cls.REPORT_SECONDS
