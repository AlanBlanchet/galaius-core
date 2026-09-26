"""Money that moves: the prepaid credit wallet, auto top-up, the Stripe-driven workspace
subscription, on-demand machine reservations, and invoices/tax profile — typed contracts only,
no Stripe SDK types leak here.

Money unit: the wallet ledger is integer **USD micro-units** (`1_000_000` == $1), never `float`.
A ledger is many small additions over a long life; float drift there is a real, compounding
correctness bug, not a style question — the cost engine's own `NodeCostActual.cost_usd`/`fee_usd`
(`cost.py`) stay `float` (a per-run figure, converted to micro-units once at the wallet boundary
via `usd_to_micros`), so this module is the ONE place the conversion happens.
"""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator

from .wire import WireModel, WireRequest


def usd_to_micros(usd: float) -> int:
    return round(usd * 1_000_000)


def micros_to_usd(micros: int) -> float:
    return micros / 1_000_000


# ---------------------------------------------------------------------------------------------
# Credit wallet
# ---------------------------------------------------------------------------------------------

LedgerEntryKind = Literal[
    # Money in
    "topup", "auto_topup", "subscription_credit", "reservation_release", "refund", "pooled_payout",
    # Money out
    "run_debit", "reservation_hold", "reservation_charge", "adjustment", "pooled_run_debit",
]

#: Which kinds ever carry a positive (credit) amount vs a negative (debit) one — the ONE mapping
#: `WalletStore` and any caller validate an entry against, so a debit kind can never be posted
#: with a positive amount by a future call site that forgets to negate it.
#: `pooled_run_debit`/`pooled_payout`: a run dispatched onto ANOTHER workspace's shared machine
#: (`interact_core.pool`) — the tenant is debited cost+fee (the SAME two-line surface every other
#: metered surface shows, the "vendor" here being the owner workspace's own hardware), the owner
#: is credited the cost portion only (their price; the platform's fee is its own cut, never paid
#: out) — Alan, 2026-09-25: "machines reserved/shared cost credits."
CREDIT_KINDS: frozenset[LedgerEntryKind] = frozenset({"topup", "auto_topup", "subscription_credit", "reservation_release", "refund", "pooled_payout"})
DEBIT_KINDS: frozenset[LedgerEntryKind] = frozenset({"run_debit", "reservation_hold", "reservation_charge", "adjustment", "pooled_run_debit"})


class WalletLedgerEntry(WireModel):
    """One append-only ledger row. `idempotency_key` is UNIQUE per workspace at the store layer —
    replaying the same key (a retried webhook, a retried run-completion callback) returns the
    ORIGINAL entry, never posts twice. `amount_usd_micros` is signed: positive for a `CREDIT_KIND`,
    negative for a `DEBIT_KIND`; `balance_after_usd_micros` is the wallet balance the moment this
    entry was posted, so a reader never recomputes a running sum to audit one row."""

    id: UUID
    workspace_id: UUID
    kind: LedgerEntryKind
    amount_usd_micros: int
    balance_after_usd_micros: int
    idempotency_key: str = Field(min_length=1, max_length=200)
    #: What this entry refers to — a workflow run id, a reservation id, a Stripe payment intent
    #: or invoice id — always present, so every debit/credit traces to the thing that caused it.
    reference: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=400)
    #: The provider-cost / platform-fee split of `amount_usd_micros`, both magnitudes (never
    #: signed) — set together for a debit priced through the cost engine (`run_debit`,
    #: `reservation_hold`, `reservation_charge`; `PLATFORM_PRICING.fee_rate` on top of a vendor
    #: cost), left `None` together for anything else (a top-up, a manual adjustment, a release —
    #: nothing to split). The ledger is the SAME two-line cost/fee surface the run panel and node
    #: card already show, never a third figure a reader has to cross-reference another endpoint
    #: to explain (visual-critic finding, 2026-09-25: a ledger row showed one pre-summed amount).
    cost_usd_micros: int | None = Field(default=None, ge=0)
    fee_usd_micros: int | None = Field(default=None, ge=0)
    created_at: datetime

    @model_validator(mode="after")
    def signed_correctly(self) -> Self:
        if (self.cost_usd_micros is None) != (self.fee_usd_micros is None):
            raise ValueError("cost_usd_micros and fee_usd_micros are set or omitted together")
        if self.cost_usd_micros is not None and self.fee_usd_micros is not None and self.cost_usd_micros + self.fee_usd_micros != abs(self.amount_usd_micros):
            raise ValueError("cost_usd_micros + fee_usd_micros must equal the entry's own amount")
        if self.kind in CREDIT_KINDS and self.amount_usd_micros < 0:
            raise ValueError(f"{self.kind} is a credit kind and must not carry a negative amount")
        if self.kind in DEBIT_KINDS and self.amount_usd_micros > 0:
            raise ValueError(f"{self.kind} is a debit kind and must not carry a positive amount")
        return self


class CreditWallet(WireModel):
    workspace_id: UUID
    balance_usd_micros: int
    #: True once the balance has dropped below the workspace's own `AutoTopUpPolicy.threshold`
    #: (or, with no policy, below `$0`) — the ONE flag every surface that shows the balance reads
    #: to warn, rather than each screen re-deriving its own threshold comparison.
    low_balance: bool
    #: Set by a failed auto top-up charge (`AutoTopUpFailure`); while true, `execute_workflow`'s
    #: pre-run check refuses every run for this workspace regardless of balance, until the owner
    #: clears it (a successful manual top-up, or a successful retried auto top-up, clears it).
    runs_paused: bool
    updated_at: datetime


class WalletTopUpRequest(WireRequest):
    """A manual, owner-initiated top-up: an amount to charge the workspace's default payment
    method right now, via a Stripe Checkout session (redirect) — never a server-side off-session
    charge (that path is `AutoTopUpPolicy`'s alone, and only after the owner opts in)."""

    amount_usd: float = Field(gt=0, le=10_000)
    success_url: str = Field(min_length=1, max_length=2000)
    cancel_url: str = Field(min_length=1, max_length=2000)


class AutoTopUpPolicy(WireModel):
    workspace_id: UUID
    enabled: bool
    #: Charge when the balance drops below this many USD.
    threshold_usd: float = Field(default=10.0, ge=0)
    #: How much to add each time the threshold is crossed.
    topup_usd: float = Field(default=25.0, ge=1, le=10_000)
    updated_at: datetime

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.enabled and self.threshold_usd >= self.topup_usd:
            raise ValueError("auto top-up amount must exceed its own trigger threshold, or it re-triggers immediately")
        return self


class AutoTopUpPolicyUpdate(WireRequest):
    enabled: bool
    threshold_usd: float = Field(default=10.0, ge=0)
    topup_usd: float = Field(default=25.0, ge=1, le=10_000)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.enabled and self.threshold_usd >= self.topup_usd:
            raise ValueError("auto top-up amount must exceed its own trigger threshold, or it re-triggers immediately")
        return self


#: What KIND of failure this was, never inferred by a UI parsing `reason`'s free-text English
#: prose (visual-critic finding, 2026-09-25: one banner sentence, "the card charge did not go
#: through", was shown for every failure — false whenever Stripe actually charged the card and
#: only the wallet credit afterward failed). `card_declined`: Stripe answered and it was a no
#: (declined, needs authentication, bad payment method) — the customer must act.
#: `charge_status_unknown`: the REQUEST to Stripe itself failed (network/timeout) — whether the
#: card was charged is genuinely unknown until an operator reconciles against the Stripe
#: dashboard. `credit_failed`: Stripe confirmed the charge SUCCEEDED and only crediting the
#: wallet afterward failed — the card WAS charged, this is our own bookkeeping, never a customer
#: payment problem.
AutoTopUpFailureReason = Literal["card_declined", "charge_status_unknown", "credit_failed"]


class AutoTopUpFailure(WireModel):
    """Recorded when an off-session auto top-up charge fails (card declined, requires
    authentication, expired) — the durable form of "email + runs pause" so a support/ops read
    never depends on the transient email having been delivered."""

    workspace_id: UUID
    stripe_payment_intent_id: str | None = Field(default=None, max_length=120)
    reason_code: AutoTopUpFailureReason
    #: Raw English operator/ops detail (Stripe error text, "lock held", "manual reconciliation")
    #: — always in English regardless of the viewer's locale, same posture as a stack trace; a
    #: reader's PRIMARY, localized message is `reason_code`, never this field parsed.
    reason: str = Field(min_length=1, max_length=400)
    occurred_at: datetime
    resolved_at: datetime | None = None


# ---------------------------------------------------------------------------------------------
# Stripe customer + subscription (Checkout + Customer Portal, webhook-driven state)
# ---------------------------------------------------------------------------------------------

class StripeCustomerLink(WireModel):
    workspace_id: UUID
    stripe_customer_id: str = Field(min_length=1, max_length=120)
    default_payment_method_id: str | None = Field(default=None, max_length=120)
    updated_at: datetime


#: Stripe's own subscription status vocabulary (Billing API), reused verbatim rather than
#: re-spelled — a webhook payload's `status` field maps onto this with no translation table.
StripeSubscriptionStatus = Literal[
    "incomplete", "incomplete_expired", "trialing", "active", "past_due", "canceled", "unpaid", "paused",
]

#: Which statuses gate the app open — `entitled(status)` is the ONE function every route/UI reads,
#: so "what counts as paid" is decided once. Trialing counts: Stripe only reaches this platform's
#: webhook with a subscription object once a Checkout session completed, which already requires a
#: saved payment method.
ENTITLED_STATUSES: frozenset[StripeSubscriptionStatus] = frozenset({"trialing", "active"})


class WorkspaceSubscriptionState(WireModel):
    workspace_id: UUID
    stripe_subscription_id: str = Field(min_length=1, max_length=120)
    status: StripeSubscriptionStatus
    current_period_end: datetime
    cancel_at_period_end: bool
    updated_at: datetime

    @property
    def entitled(self) -> bool:
        return self.status in ENTITLED_STATUSES


class CheckoutSessionRequest(WireRequest):
    success_url: str = Field(min_length=1, max_length=2000)
    cancel_url: str = Field(min_length=1, max_length=2000)


class CheckoutSessionCreated(WireModel):
    url: str = Field(min_length=1, max_length=2000)


class PortalSessionCreated(WireModel):
    url: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------------------------------------
# Machine reservation
# ---------------------------------------------------------------------------------------------

MachineReservationStatus = Literal["held", "active", "completed", "cancelled"]


class MachineReservationRequest(WireRequest):
    provider: Literal["scaleway"] = "scaleway"
    region: str = Field(min_length=1, max_length=40)
    instance_type: str = Field(min_length=1, max_length=40)
    hours: float = Field(gt=0, le=720)


class MachineReservation(WireModel):
    id: UUID
    workspace_id: UUID
    provider: Literal["scaleway"]
    region: str
    instance_type: str
    usd_per_hour: float = Field(ge=0)
    #: `PLATFORM_PRICING.fee_rate` at booking time, applied on top of `usd_per_hour` — the same
    #: two-line (cost, fee) shape every other metered surface shows.
    fee_rate: float = Field(ge=0, le=1)
    hours_reserved: float = Field(gt=0)
    #: What was DEBITED from the wallet at booking (`hours_reserved * usd_per_hour * (1+fee_rate)`,
    #: in micro-units) — released back for the unused portion, or converted to `reservation_charge`
    #: entries as the machine actually runs.
    held_usd_micros: int = Field(ge=0)
    cost_usd_so_far_micros: int = Field(default=0, ge=0)
    status: MachineReservationStatus
    starts_at: datetime
    ends_at: datetime
    cloud_machine_id: UUID | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.ends_at <= self.starts_at:
            raise ValueError("a reservation must end after it starts")
        if self.status in ("completed", "cancelled") and self.cloud_machine_id is None and self.status == "completed":
            raise ValueError("a completed reservation must reference the machine it ran")
        return self


# ---------------------------------------------------------------------------------------------
# Invoices / receipts + EU VAT
# ---------------------------------------------------------------------------------------------

class TaxProfile(WireModel):
    """The workspace's own billing-tax facts — set by the owner, never inferred from an IP or a
    card's issuing country. Sourced 2026-09-24
    (`~/.github/research/pricing-model-usage-subscriptions-2026-09-24.md` §D): B2C usage-credit
    sales are taxed in the CUSTOMER's country (EU OSS one-stop-shop); B2B EU sales to a VALID VAT
    number are reverse-charged (0%, customer self-assesses). `vat_verified` is the VIES check
    RESULT, never a self-report the customer typed — a stale/unverified VAT number never grants
    reverse charge (see `invoice_vat.py`'s `rate_for`)."""

    workspace_id: UUID
    country_code: str = Field(min_length=2, max_length=2, pattern=r"^[A-Z]{2}$")
    business: bool
    vat_number: str | None = Field(default=None, max_length=32)
    vat_verified: bool = False
    vat_checked_at: datetime | None = None
    updated_at: datetime

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.vat_number is not None and not self.business:
            raise ValueError("a VAT number requires business=True")
        if self.vat_verified and self.vat_number is None:
            raise ValueError("vat_verified requires a vat_number to have been verified")
        return self


class TaxProfileUpdate(WireRequest):
    country_code: str = Field(min_length=2, max_length=2, pattern=r"^[A-Z]{2}$")
    business: bool
    vat_number: str | None = Field(default=None, max_length=32)


InvoiceKind = Literal["subscription", "topup", "reservation"]


class InvoiceLine(WireModel):
    description: str = Field(min_length=1, max_length=200)
    subtotal_usd: float = Field(ge=0)


class InvoiceRecord(WireModel):
    id: UUID
    workspace_id: UUID
    kind: InvoiceKind
    stripe_invoice_id: str | None = Field(default=None, max_length=120)
    lines: tuple[InvoiceLine, ...] = Field(min_length=1, max_length=50)
    subtotal_usd: float = Field(ge=0)
    vat_rate: float = Field(ge=0, le=1)
    vat_amount_usd: float = Field(ge=0)
    #: True when this invoice was reverse-charged (`vat_rate == 0` for that reason, not because
    #: no VAT applies at all) — shown on the invoice line the OSS/reverse-charge rule requires.
    reverse_charge: bool
    total_usd: float = Field(ge=0)
    hosted_invoice_url: str | None = Field(default=None, max_length=2000)
    created_at: datetime

    @model_validator(mode="after")
    def coherent(self) -> Self:
        computed_subtotal = round(sum(line.subtotal_usd for line in self.lines), 6)
        if abs(computed_subtotal - self.subtotal_usd) > 0.01:
            raise ValueError("invoice subtotal must equal the sum of its lines")
        if abs(self.subtotal_usd * self.vat_rate - self.vat_amount_usd) > 0.01:
            raise ValueError("VAT amount must equal subtotal times VAT rate")
        if abs(self.subtotal_usd + self.vat_amount_usd - self.total_usd) > 0.01:
            raise ValueError("invoice total must equal subtotal plus VAT")
        if self.reverse_charge and self.vat_rate != 0:
            raise ValueError("a reverse-charged invoice carries a zero VAT rate")
        return self


class OperatorGrant(WireModel):
    """An EXPLICIT, visible exemption from the credit gate for ONE workspace — never a hidden
    dry-run bypass. Lets an operator exercise cloud/pooled paths (which genuinely debit credits,
    2026-09-25 decision) before Stripe funds a real balance: the wallet still posts every debit
    for a granted workspace (its balance can go negative, visibly, in its own ledger) — this only
    lifts the pre-run REFUSAL, never the record. `note` is mandatory so a grant always carries why
    it exists, readable by anyone who finds it later."""

    workspace_id: UUID
    granted_by_account_id: UUID
    note: str = Field(min_length=1, max_length=400)
    granted_at: datetime


class OperatorGrantRequest(WireRequest):
    note: str = Field(min_length=1, max_length=400)


class InsufficientCredits(WireModel):
    """Returned instead of starting a run whose estimate would take the wallet balance below
    zero — the wallet's own `BudgetOverrun` analog, same 402 shape, so a client's existing
    budget-exceeded handling extends to this with one more `code` branch."""

    balance_usd: float
    estimate_usd: float
    runs_paused: bool
