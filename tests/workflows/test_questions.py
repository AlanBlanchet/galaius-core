"""`OwnerQuestion`: one line naming what it is about, 2-4 one-line options each with what it costs him, a reason from the closed
critical set, no internal id; his answer — an option or his own words, on every card — reads back as one line."""

import pytest
from pydantic import ValidationError

from galaius_core import OwnerQuestion, QuestionAnswer

OPTIONS = ({"label": "Supprimer", "consequence": "la base repart vide", "cost": "1 min de coupure"},
           {"label": "Garder", "consequence": "rien ne change", "cost": "rien"})


def _question(**changes) -> OwnerQuestion:
    return OwnerQuestion(**{"subject": "base de données preprod (serveur galaius)", "question": "Supprimer la base preprod ?", "options": OPTIONS, "recommended": 1, "criticality": "destructive",
                            "reason": "irréversible", **changes})


@pytest.mark.parametrize("changes", [
    pytest.param({"criticality": "style"}, id="not-critical"),
    pytest.param({"recommended": 2}, id="recommends-no-option"),
    pytest.param({"question": "Supprimer ?\nVraiment ?"}, id="two-lines"),
    pytest.param({"options": OPTIONS[:1]}, id="one-option"),
    pytest.param({"options": (OPTIONS[0], OPTIONS[0])}, id="same-label-twice"),
    # Owner 2026-11-12 « I don't understand what the question is about »: what it acts on, what each answer costs, a question.
    pytest.param({"subject": ""}, id="names-nothing"),
    pytest.param({"options": ({"label": "Supprimer", "consequence": "la base repart vide"}, OPTIONS[1])}, id="option-without-cost"),
    pytest.param({"question": "Community agent « T » asks for the browser signed in on the site lent to it"}, id="not-a-question"),
    pytest.param({"question": "Lancer le run 0c62da1e sur galaius ?"}, id="internal-id"),
    pytest.param({"reason": "source entry 2045a5d9-42ae-4d21-8c82-7b3b9c70ca0d"}, id="internal-id-in-reason"),
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


def test_an_option_the_question_did_not_offer_is_refused():
    with pytest.raises(ValueError):
        _question().answer_text(QuestionAnswer(option=3))


def test_a_question_that_once_refused_his_own_words_reads_no_more():
    """« Autre » ends every card (owner 2026-11-07): the switch that refused it (2026-11-12: « Non envoyé : réessayez ») is gone."""
    with pytest.raises(ValidationError):
        _question(allow_custom=False)


def test_its_answer_id_is_stable_and_its_own():
    first, second = _question(), _question()
    assert first.answer_id == OwnerQuestion.model_validate(first.model_dump(mode="json")).answer_id != second.answer_id
