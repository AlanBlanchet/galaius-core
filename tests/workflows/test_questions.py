"""`OwnerQuestion`: one line, 2-4 one-line options, a reason from the closed critical set; his answer reads back as one line."""

import pytest
from pydantic import ValidationError

from galaius_core import OwnerQuestion, QuestionAnswer

OPTIONS = ({"label": "Supprimer", "consequence": "la base repart vide"}, {"label": "Garder", "consequence": "rien ne change"})


def _question(**changes) -> OwnerQuestion:
    return OwnerQuestion(**{"question": "Supprimer la base preprod ?", "options": OPTIONS, "recommended": 1, "criticality": "destructive",
                            "reason": "irréversible", **changes})


@pytest.mark.parametrize("changes", [
    pytest.param({"criticality": "style"}, id="not-critical"),
    pytest.param({"recommended": 2}, id="recommends-no-option"),
    pytest.param({"question": "Supprimer ?\nVraiment ?"}, id="two-lines"),
    pytest.param({"options": OPTIONS[:1]}, id="one-option"),
    pytest.param({"options": (OPTIONS[0], OPTIONS[0])}, id="same-label-twice"),
])
def test_a_question_the_owner_could_not_answer_in_one_tap_is_refused(changes):
    with pytest.raises(ValidationError):
        _question(**changes)


@pytest.mark.parametrize(("answer", "reads"), [
    pytest.param(QuestionAnswer(option=1), "→ B : Garder (rien ne change)", id="option"),
    pytest.param(QuestionAnswer(text="garde-la jusqu'à lundi"), "→ « garde-la jusqu'à lundi »", id="custom"),
])
def test_his_answer_reads_as_one_line_naming_the_question(answer, reads):
    question = _question()
    assert question.answer_text(answer) == f"[question {question.short}] « Supprimer la base preprod ? » {reads}"


@pytest.mark.parametrize(("question", "answer"), [
    pytest.param({"allow_custom": False}, QuestionAnswer(text="autre chose"), id="custom-not-offered"),
    pytest.param({}, QuestionAnswer(option=3), id="option-it-lacks"),
])
def test_an_answer_the_question_did_not_offer_is_refused(question, answer):
    with pytest.raises(ValueError):
        _question(**question).answer_text(answer)


def test_its_answer_id_is_stable_and_its_own():
    first, second = _question(), _question()
    assert first.answer_id == OwnerQuestion.model_validate(first.model_dump(mode="json")).answer_id != second.answer_id
