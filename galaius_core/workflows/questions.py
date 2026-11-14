"""A question an agent puts to the owner WITHOUT waiting on it: posted as a card in his conversation with one-tap answers,
his pick handed back later as his own message, like one he types while it works (owner 2026-11-02: « A user expects work
to be done automatically. So only ask for critical things »; 15:24: asking never blocks the asker, and only the
coordinator — his entry point — asks him).

Only a conversation's coordinator asks (a run no other run launched), only for a reason in `Criticality`; every other
fork its agent decides, says in one line, and keeps working. A question stays open on its run until a message from the
person carries its `answer_id`: his answer is sent under that id, so a retried tap is the same message and the card
closes on the one that landed.

Every card ends with his own words (« Autre », owner 2026-11-07: « I'd like a final choice, which is what I want … I could
just manually type or speak »): no question refuses them. A question names what it is about and what each answer costs him
(owner 2026-11-12: « You must make your questions more concret if you want something from me »)."""

import re
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
#: An internal id (a run, commit or question id): it names nothing he knows, so no new question carries one. A hex word of 8+
#: mixing digits and letters (never a colour after « # », nor an amount and its unit like « 15000fcfa »), or any id-shaped
#: word after the noun naming it (« commit 9530592 »).
_INTERNAL_ID = re.compile(r"(?<![#\w])(?=[0-9a-f]*[0-9])(?=[0-9a-f]*[a-f])(?![0-9]+[a-f]+\b)[0-9a-f]{8,}\b"
                          r"|\b(?:run|commit|sha|question|card|carte)\s+[0-9a-f]{6,}\b", re.IGNORECASE)
#: How agents name each other, never words he chose: a role slug (« backend-engineer »), « community agent », « sub-agent »,
#: « agent's », « coordinator ».
_AGENT_TALK = re.compile(r"\b[a-z]+-(?:engineer|critic|tester|analy[sz]er|manager|strategist|researcher|scientist|modeler|auditor|"
                         r"optimizer|synthesizer|reviewer)\b|\bcommunity agents?\b|\bsub-?agents?\b|\bsous-agents?\b|\bagent's\b|"
                         r"\bcoordinat(?:or|eur)\b", re.IGNORECASE)
#: Every answer id is derived in this namespace from its question's id (`OwnerQuestion.answer_id`).
_ANSWERS = UUID("6f1c2a52-3b0e-4f43-9b39-6a9f1f7d2c11")


def _line(longest: int) -> type[str]:
    """One line of at most `longest` characters, stripped, never empty."""
    return Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=longest, pattern=r"^[^\r\n]*$")]  # type: ignore[return-value]


class QuestionOption(BaseModel):
    """One answer he can tap: what it is, what happens for him when he picks it, and what that costs him."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: _line(60)
    consequence: _line(160)
    #: What picking it costs him, in time or money (« 2 min de coupure », « 0,15 $ », « rien »); None only on a question an
    #: older galaius asked (`OwnerQuestion.unclear` requires it of every new one).
    cost: _line(80) | None = None


class QuestionAnswer(BaseModel):
    """What he tapped: an option by its index, or his own words (« Autre », on every card) — exactly one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    option: int | None = Field(default=None, ge=0, lt=len(QUESTION_LETTERS))
    #: His own words, as typed or spoken: line breaks kept (the card's box takes Shift+Enter).
    text: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)] | None = None

    @model_validator(mode="after")
    def _one(self) -> Self:
        if (self.option is None) == (self.text is None):
            raise ValueError("an answer is one option or his own words")
        return self


class OwnerQuestion(BaseModel):
    """A critical question of one run, as its card shows it and its record keeps it. A record an older galaius wrote reads too
    (no `subject` nor costs, an `allow_custom` key it ignores): its card still shows and takes his answer. What a NEW
    question must say is `unclear`'s."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str = Field(default_factory=lambda: token_hex(16), pattern=r"^[0-9a-f]{32}$")
    at: float = Field(default_factory=time)
    #: What it is about, named as he knows it: the file, screen, server, account or amount it acts on (None: asked by an older
    #: galaius).
    subject: _line(120) | None = None
    question: _line(200)
    options: tuple[QuestionOption, ...] = Field(min_length=2, max_length=len(QUESTION_LETTERS))
    #: The option the agent would take, by index: marked on the card.
    recommended: int = Field(ge=0)
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

    def unclear(self) -> tuple[str, ...]:
        """Why he could not answer this question cold, from his phone, without the asker's context — what a NEW question is
        refused for (galaius `ask_owner`); empty when it is clear."""
        problems = []
        if self.subject is None or len(self.subject.split()) < 2:
            problems.append("subject: name the thing it acts on as he knows it, in a few words (« base preprod du serveur galaius »)")
        if self.question.count("?") != 1 or not self.question.endswith("?"):
            problems.append("question: ask him ONE thing, ending with « ? »")
        for letter, option in zip(QUESTION_LETTERS, self.options):
            if option.cost is None or not any(char.isalnum() for char in option.cost) or "?" in option.cost:
                problems.append(f"option {letter} cost: what picking it costs him in time or money, « rien » when nothing")
        said = " ".join([self.subject or "", self.question, self.reason, *(f"{option.label} {option.consequence} {option.cost or ''}" for option in self.options)])
        if found := _INTERNAL_ID.search(said):
            problems.append(f"« {found.group()} » is an internal id he does not know: name the thing itself (its file, screen, server or account)")
        if found := _AGENT_TALK.search(said):
            problems.append(f"« {found.group()} » is how agents name each other: say what it does for him, in his words")
        return tuple(problems)

    @property
    def short(self) -> str:
        return self.id[:8]

    @property
    def answer_id(self) -> str:
        """The id his answer is sent and recorded under: the same for every tap, so a retry never answers twice."""
        return str(uuid5(_ANSWERS, self.id))

    def answer_text(self, answer: QuestionAnswer) -> str:
        """His answer as the run reads it: which question, then the option he picked or his own words. Raises ValueError
        for an option the question does not have."""
        asked = f"[question {self.short}] « {self.question} » →"
        if answer.text is not None:
            return f"{asked} « {answer.text} »"
        if answer.option is None or answer.option >= len(self.options):
            raise ValueError(f"this question has {len(self.options)} options")
        picked = self.options[answer.option]
        return f"{asked} {QUESTION_LETTERS[answer.option]} : {picked.label} ({picked.consequence})"
