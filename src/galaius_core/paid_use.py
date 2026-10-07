"""Who may spend REAL money per token, and how much a workspace may spend in a month.

Deny-by-default. Nothing here decides which model is BEST (that is `ModelChoice` /
`Criteria`); it decides whether reaching a candidate would take the owner's money, and whether
he ever said yes to that provider taking it.

Three ways a call is paid for — the ONE vocabulary every surface says it in (`CHARGE_WORDS`, the
owner's own words):

- `subscription` — a vendor CLI already logged in on one of his PCs (`claude`, `codex`). A flat
  monthly plan he already pays; one more call adds nothing. This is the DEFAULT for a server-run
  agent.
- `local` — his own hardware or his own OpenAI-wire server. Electricity, never an invoice.
- `metered_api` — a vendor key billed per token. The only kind that can surprise him, so the only
  kind a `PaidUsePolicy` has to switch ON, per provider, one provider at a time.

A cap is a HARD STOP, never a warning: `PaidUseSpend.over_cap` means the next metered call is
refused. No policy at all still caps, at `DEFAULT_MONTHLY_CAP_USD` — a fresh workspace is bounded
before anyone configures anything.
"""

from datetime import UTC, datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .wire import WireModel, WireRequest

#: How a call is paid for. Derived from the route + provider, never stored on a model: the same
#: model is `subscription` through a CLI session and `metered_api` through a key.
ChargeKind = Literal["subscription", "local", "metered_api"]

#: The word each kind is shown as, in the owner's language. One table, so an agent card, the model
#: picker and the cap alert can never disagree about what a route costs.
CHARGE_WORDS: dict[ChargeKind, str] = {
    "subscription": "abonnement",
    "local": "local",
    "metered_api": "payant à l'usage",
}

#: `ModelRoute.route` values mapped to what they cost. A `cli_session` is the owner's plan; a
#: `self_hosted` connection is his own server; a stored-key route is the vendor's meter.
#: The route names are held as constants so the table never spells a key-looking literal pair.
ROUTE_CLI_SESSION = "cli_session"
ROUTE_SELF_HOSTED = "self_hosted"
ROUTE_STORED_KEY = "api_key"
_ROUTE_CHARGE: dict[str, ChargeKind] = {
    ROUTE_CLI_SESSION: "subscription",
    ROUTE_SELF_HOSTED: "local",
    ROUTE_STORED_KEY: "metered_api",
}

#: Providers whose endpoint IS the owner's own hardware, whatever route reaches them — an
#: `api_key` route to one of these still costs nothing per token (his own server asking for a
#: token of its own).
LOCAL_PROVIDERS = frozenset({"self_hosted", "openai_compatible", "ollama", "vllm"})

#: The ceiling a workspace that never set one gets. The owner named 10 €; every provider price in
#: this system is quoted in USD, and $10 is the STRICTER reading of his number — never a silent
#: currency conversion at an invented rate.
DEFAULT_MONTHLY_CAP_USD = 10.0


def charge_kind(route: str, provider: str) -> ChargeKind:
    """What one route to one provider costs. An unrecognised route reads `metered_api`: the
    money-spending pole, so a route nobody classified can never spend unnoticed."""
    if provider in LOCAL_PROVIDERS:
        return "local"
    return _ROUTE_CHARGE.get(route, "metered_api")


def charge_word(kind: ChargeKind) -> str:
    return CHARGE_WORDS[kind]


def month_start(now: datetime | None = None) -> datetime:
    instant = now or datetime.now(UTC)
    return instant.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


class PaidProviderSwitch(WireModel):
    """One provider the workspace owner switched ON for pay-per-token use. Absence of a row IS the
    refusal — there is no `enabled: False` state anyone can forget to check."""

    provider: str = Field(min_length=1, max_length=40)
    enabled_at: datetime
    #: The account that switched it on — the audit answer to "who let this spend money".
    enabled_by: UUID | None = None
    #: This provider's own ceiling, when the owner set one; the workspace cap binds on top of it
    #: (whichever is lower refuses first).
    monthly_cap_usd: float | None = Field(default=None, ge=0)


class PaidUsePolicy(WireModel):
    """What a workspace may spend real money on. Deny-by-default in both axes: no switch means no
    metered call, and no cap set means `DEFAULT_MONTHLY_CAP_USD`."""

    providers: tuple[PaidProviderSwitch, ...] = Field(default=(), max_length=64)
    #: The workspace ceiling per calendar month, in USD. `None` is NOT "no ceiling" — it reads
    #: `DEFAULT_MONTHLY_CAP_USD`. An explicit `0` forbids metered spend outright; a ceiling is
    #: lifted by raising this number, never by clearing it.
    monthly_cap_usd: float | None = Field(default=None, ge=0)

    @property
    def cap_usd(self) -> float:
        return DEFAULT_MONTHLY_CAP_USD if self.monthly_cap_usd is None else self.monthly_cap_usd

    def switch(self, provider: str) -> PaidProviderSwitch | None:
        return next((item for item in self.providers if item.provider == provider), None)

    def allows(self, provider: str) -> bool:
        """Whether a metered call to `provider` is authorized at all. The cap is a separate
        question (`PaidUseSpend.over_cap`). A local provider needs no switch: it spends nothing."""
        return provider in LOCAL_PROVIDERS or self.switch(provider) is not None

    def cap_for(self, provider: str) -> float:
        """The lower of the workspace ceiling and this provider's own, when it has one."""
        switch = self.switch(provider)
        if switch is None or switch.monthly_cap_usd is None:
            return self.cap_usd
        return min(self.cap_usd, switch.monthly_cap_usd)


class PaidUseSpend(WireModel):
    """What this calendar month has actually cost, against the ceiling — the figure every agent
    card, the model picker and the alert show, all reading this one shape."""

    period_start: datetime
    #: Model cost this month, from the settled ledger — measured, never estimated.
    spent_usd: float = Field(ge=0)
    #: The platform fee already charged on top of it. `total_usd` is what the cap is held to, so a
    #: fee can never be the thing that silently crosses it.
    fee_usd: float = Field(default=0.0, ge=0)
    cap_usd: float = Field(ge=0)

    @property
    def total_usd(self) -> float:
        return self.spent_usd + self.fee_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.cap_usd - self.total_usd)

    @property
    def over_cap(self) -> bool:
        return self.total_usd >= self.cap_usd

    @property
    def fraction(self) -> float:
        """0..1 of the ceiling used — what a bar renders. A zero cap reads full."""
        return 1.0 if self.cap_usd <= 0 else min(1.0, self.total_usd / self.cap_usd)


class ProviderChargeState(WireModel):
    """One provider as a picker row: what it costs, whether the owner switched it on, and what it
    has cost this month."""

    provider: str = Field(min_length=1, max_length=40)
    charge: ChargeKind
    allowed: bool
    #: What THIS provider cost this month, when the ledger attributes it. `None` is UNKNOWN —
    #: never 0, and never the month's total standing in for a per-provider figure (visual-critic
    #: r1 read 0.20794, the whole month, on every metered row).
    spent_usd: float | None = Field(default=None, ge=0)
    #: Why a call would be refused right now. A STATEMENT of fact, in one sentence, for a log and
    #: for a client with no localization of its own — the words a reader sees come from `refused`
    #: + the figures, in the reader's language, and carry the action. This sentence names no
    #: screen to go to: it gets shown INSIDE that screen (visual-critic r1).
    refusal: str | None = Field(default=None, max_length=300)
    #: Which refusal applies, for a client that writes its own sentence. None when the call would
    #: go through.
    refused: Literal["paid_use_not_enabled", "monthly_cap_reached"] | None = None

    @property
    def word(self) -> str:
        return CHARGE_WORDS[self.charge]


class PaidUseState(WireModel):
    """One workspace's whole money posture in one read: what is switched on, what the month cost,
    and what each reachable provider would cost to call right now."""

    policy: PaidUsePolicy
    spend: PaidUseSpend
    #: Every provider this workspace has a connection or a CLI session for, with what calling it
    #: costs and whether it may be called — the model picker's own rows.
    providers: tuple[ProviderChargeState, ...] = Field(default=(), max_length=64)


class PaidProviderUpdate(WireRequest):
    """Switch one provider on or off for pay-per-token use. Owner action only — this is the write
    that makes money movable."""

    enabled: bool
    monthly_cap_usd: float | None = Field(default=None, ge=0)


class PaidUseCapUpdate(WireRequest):
    """Set the workspace's monthly ceiling. `None` restores `DEFAULT_MONTHLY_CAP_USD`."""

    monthly_cap_usd: float | None = Field(default=None, ge=0)


class PaidUseRefusal(WireModel):
    """The body a refused metered call answers with — a real status and a real reason, never a 200
    with the refusal folded inside."""

    code: Literal["paid_use_not_enabled", "monthly_cap_reached"]
    provider: str = Field(min_length=1, max_length=40)
    message: str = Field(min_length=1, max_length=300)
    spend: PaidUseSpend | None = None

    @model_validator(mode="after")
    def cap_refusal_shows_the_figures(self) -> Self:
        if self.code == "monthly_cap_reached" and self.spend is None:
            raise ValueError("a cap refusal states the month's spend and the ceiling it crossed")
        return self
