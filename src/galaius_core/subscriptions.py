"""A coding-agent subscription (a vendor login: Claude, Codex, ...) as the web and a PC share it: its
caps, its usage windows, a run's allowance past a cap. The credential itself never crosses the wire:
it stays in the subscription's own folder on the PC that signed it in."""

from datetime import datetime
from typing import ClassVar, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .wire import WireModel, WireRequest

#: The usage periods a vendor meters a subscription over. A vendor's own token maps onto the nearest
#: (`UsageWindow.nearest`): Claude `seven_day` is `weekly`, Codex `window_minutes: 10080` is `weekly`.
UsageWindowName = Literal["five_hour", "weekly", "monthly"]
#: What a run does when its subscription reaches a cap: `finish` lets the current turn end and starts
#: no new turn or session on it until the window drops; `cut` stops the run at that reading.
AtCap = Literal["finish", "cut"]


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


class SubscriptionUse(WireRequest):
    """Which subscription a run, a turn or a continued conversation runs on, with the owner's caps
    on it now (the server's record; the PC checks them again before anything starts). Absent: the
    PC's own default login, as before subscriptions (asked of a runner saying "agent_subscriptions").
    `lent`: the initiator is a member using a PC lent to him: no CLI login of that PC runs it (its
    owner's or a subscription's); only API-key and local models do."""

    subscription_id: UUID | None = Field(default=None, exclude_if=lambda value: value is None)
    limits: SubscriptionLimits | None = Field(default=None, exclude_if=lambda value: value is None)
    lent: bool = Field(default=False, exclude_if=lambda value: value is False)

    @model_validator(mode="after")
    def limits_of_a_subscription(self) -> Self:
        if self.limits is not None and self.subscription_id is None:
            raise ValueError("limits belong to a subscription: name its subscription_id")
        if self.lent and self.subscription_id is not None:
            raise ValueError("a subscription is never lent: a member's run on a lent PC names none")
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


class SubscriptionUsage(WireModel):
    """One subscription's last usage reading on a PC."""

    subscription_id: UUID
    windows: tuple[UsageWindow, ...] = Field(default=(), max_length=8)


class SubscriptionUsageReport(WireModel):
    """What a PC pushes to its server when a subscription's usage moved (by `REPORT_POINTS` or more,
    or `REPORT_SECONDS` after the last report of a change)."""

    type: Literal["subscription_usage"] = "subscription_usage"
    usage: tuple[SubscriptionUsage, ...] = Field(max_length=64)

    REPORT_POINTS: ClassVar[float] = 1.0
    REPORT_SECONDS: ClassVar[float] = 60.0
