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


def test_a_question_an_older_galaius_asked_still_reads_and_takes_his_own_words():
    """« Autre » ends every card (owner 2026-11-07); a record whose question refused it (2026-11-12: « Non envoyé : réessayez »)
    or carries no subject nor costs still shows its card and takes his words, with or without line breaks."""
    old = OwnerQuestion.model_validate({"id": "a" * 32, "question": "Supprimer ?", "recommended": 0, "allow_custom": False, "criticality": "destructive",
                                        "reason": "r", "options": [{"label": "Oui", "consequence": "x"}, {"label": "Non", "consequence": "y"}]})
    assert old.subject is None and old.answer_text(QuestionAnswer(text="ligne un\nligne deux")).endswith("→ « ligne un\nligne deux »")


# Owner 2026-11-12 « I don't understand what the question is about. You must make your questions more concret if you want
# something from me »: a NEW question names what it acts on and each answer's cost, asks one thing, carries no internal id
# nor agent talk (tester questions r1 probe rows).
@pytest.mark.parametrize("changes", [
    pytest.param({"subject": None}, id="names-nothing"),
    pytest.param({"subject": "renommage"}, id="one-word-subject"),
    pytest.param({"options": ({"label": "Supprimer", "consequence": "la base repart vide"}, OPTIONS[1])}, id="option-without-cost"),
    pytest.param({"options": ({"label": "Supprimer", "consequence": "la base repart vide", "cost": "?"}, OPTIONS[1])}, id="cost-placeholder"),
    pytest.param({"question": "Community agent « T » asks for the browser signed in on the site lent to it"}, id="not-a-question"),
    pytest.param({"question": "Supprimer la base preprod ?  Et le cache ?"}, id="two-questions"),
    pytest.param({"question": "Lancer 0c62da1e sur galaius ?"}, id="internal-id"),
    pytest.param({"question": "Relancer ABCDEF12 ?"}, id="upper-hex"),
    pytest.param({"question": "Déployer le commit 9530592 ?"}, id="named-short-sha"),
    pytest.param({"reason": "source entry 2045a5d9-42ae-4d21-8c82-7b3b9c70ca0d"}, id="uuid-in-reason"),
    pytest.param({"question": "Le community agent « Ui ux tester » peut-il avoir le navigateur ?"}, id="community-agent"),
    pytest.param({"question": "Le backend-engineer peut-il redémarrer le serveur ?"}, id="role-slug"),
])
def test_a_question_he_could_not_understand_cold_is_unclear(changes):
    assert _question(**changes).unclear()


@pytest.mark.parametrize("changes", [
    pytest.param({}, id="the-example"),
    pytest.param({"question": "Payer 15000fcfa pour le serveur ?"}, id="amount-and-unit"),
    pytest.param({"subject": "couleur #ffaa0033 du bouton"}, id="css-colour"),
    pytest.param({"subject": "facture 2026ac1 de Scaleway"}, id="invoice-ref"),
    pytest.param({"question": "Quelle clé pyannote.ai Arthur utilise-t-il sur son PC Windows ?", "subject": "clé pyannote.ai du PC d'Arthur"}, id="owner-example-3"),
])
def test_a_concrete_question_is_clear(changes):
    assert _question(**changes).unclear() == ()


def test_its_answer_id_is_stable_and_its_own():
    first, second = _question(), _question()
    assert first.answer_id == OwnerQuestion.model_validate(first.model_dump(mode="json")).answer_id != second.answer_id
