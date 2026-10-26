"""The one status table: every word a client shows has its words in both languages, and a word said about a number of
things has both its forms; a word missing its text, or text for a word outside the set, refuses to import."""

from typing import get_args

import pytest

from galaius_core import STATUS_TEXT, StatusWord


def test_every_word_has_its_text_in_both_languages() -> None:
    assert tuple(STATUS_TEXT) == get_args(StatusWord)
    blank = [word for word, text in STATUS_TEXT.items() if not all(text.label) or not all(text.meaning) or (text.next is not None and not all(text.next))]
    assert blank == []


@pytest.mark.parametrize(("word", "locale", "said"), [("stopped", "fr", "Arrêtée"), ("stopped", "en", "Stopped"), ("needs-you", "de", "Vous attend")])
def test_a_word_is_said_in_its_reader_language_french_by_default(word: StatusWord, locale: str, said: str) -> None:
    assert STATUS_TEXT[word].say(locale) == said
