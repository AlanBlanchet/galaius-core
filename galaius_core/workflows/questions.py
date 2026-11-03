"""A question an agent puts to the owner WITHOUT waiting on it: posted as a card in his conversation with one-tap answers,
his pick handed back later as his own message, like one he types while it works (owner 2026-11-02: « A user expects work
to be done automatically. So only ask for critical things »; 15:24: asking never blocks the asker, and only the
coordinator — his entry point — asks him).

Only a conversation's coordinator asks (a run no other run launched), only for a reason in `Criticality`; every other
fork its agent decides, says in one line, and keeps working. A question stays open on its run until a message from the
person carries its `answer_id`: his answer is sent under that id, so a retried tap is the same message and the card
closes on the one that landed."""

from typing import Annotated, Literal, Self, get_args
from secrets import token_hex
from time import time
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

#: The only reasons a coordinator may put a question to the owner: an act that cannot be taken back, touches production,
#: publishes to an audience he does not control, needs a credential he holds, opens a new access, or spends past his cap.
Criticality = Literal["destructive", "prod", "public", "credential", "new_access", "spend_past_cap"]
CRITICALITIES: tuple[Criticality, ...] = get_args(Criticality)
#: Questions one run may leave open at once: a fourth is refused until he answers one.
OPEN_MAX = 3
#: How an option is named on its button and in his answer, in order.
QUESTION_LETTERS = "ABCD"
#: Every answer id is derived in this namespace from its question's id (`OwnerQuestion.answer_id`).
_ANSWERS = UUID("6f1c2a52-3b0e-4f43-9b39-6a9f1f7d2c11")


def _line(longest: int) -> type[str]:
    """One line of at most `longest` characters, stripped, never empty."""
    return Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=longest, pattern=r"^[^\r\n]*$")]  # type: ignore[return-value]


class QuestionOption(BaseModel):
    """One answer he can tap: what it is, and what happens when he picks it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: _line(60)
    consequence: _line(160)


class QuestionAnswer(BaseModel):
    """What he tapped: an option by its index, or his own words (`OwnerQuestion.allow_custom`) — exactly one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    option: int | None = Field(default=None, ge=0, lt=len(QUESTION_LETTERS))
    text: _line(500) | None = None

    @model_validator(mode="after")
    def _one(self) -> Self:
        if (self.option is None) == (self.text is None):
            raise ValueError("an answer is one option or his own words")
        return self


class OwnerQuestion(BaseModel):
    """A critical question of one run, as its card shows it and its record keeps it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=lambda: token_hex(16), pattern=r"^[0-9a-f]{32}$")
    at: float = Field(default_factory=time)
    question: _line(200)
    options: tuple[QuestionOption, ...] = Field(min_length=2, max_length=len(QUESTION_LETTERS))
    #: The option the agent would take, by index: marked on the card.
    recommended: int = Field(ge=0)
    #: Whether he may answer in his own words (« Autre »).
    allow_custom: bool = True
    criticality: Criticality
    #: Why this one cannot be decided without him, in one line.
    reason: _line(160)

    @model_validator(mode="after")
    def _recommends_one(self) -> Self:
        if self.recommended >= len(self.options):
            raise ValueError(f"recommended names one of the {len(self.options)} options (0-based)")
        if len({option.label.casefold() for option in self.options}) != len(self.options):
            raise ValueError("each option has its own label")
        return self

    @property
    def short(self) -> str:
        return self.id[:8]

    @property
    def answer_id(self) -> str:
        """The id his answer is sent and recorded under: the same for every tap, so a retry never answers twice."""
        return str(uuid5(_ANSWERS, self.id))

    def answer_text(self, answer: QuestionAnswer) -> str:
        """His answer as the run reads it: which question, then the option he picked or his own words. Raises ValueError
        for an option the question does not have, or his own words where it allows none."""
        asked = f"[question {self.short}] « {self.question} » →"
        if answer.text is not None:
            if not self.allow_custom:
                raise ValueError("this question takes one of its options only")
            return f"{asked} « {answer.text} »"
        if answer.option is None or answer.option >= len(self.options):
            raise ValueError(f"this question has {len(self.options)} options")
        picked = self.options[answer.option]
        return f"{asked} {QUESTION_LETTERS[answer.option]} : {picked.label} ({picked.consequence})"
