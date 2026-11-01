"""`Locale` reads the reader's language from whatever a request or a stored row says, never raising on a stranger."""

from types import SimpleNamespace

import pytest

from galaius_core.locale import Locale


@pytest.mark.parametrize(("value", "spoken"), [("fr", Locale.FR), ("en", Locale.EN), ("de", Locale.EN), ("", Locale.EN), (None, Locale.EN)])
def test_a_stored_language_reads_as_ours_or_english(value: str | None, spoken: Locale) -> None:
    assert Locale.spoken(value) is spoken


@pytest.mark.parametrize(
    ("query", "headers", "read"),
    [
        ({"locale": "en"}, {"accept-language": "fr-FR"}, Locale.EN),
        ({"locale": "de"}, {"accept-language": "fr-FR,fr;q=0.9"}, Locale.FR),
        ({}, {"accept-language": "EN-gb"}, Locale.EN),
        ({}, {"accept-language": "de-DE"}, Locale.EN),
        ({}, {}, Locale.FR),
        ({}, {"accept-language": ""}, Locale.FR),
    ],
)
def test_a_request_reads_its_query_then_its_browser(query: dict[str, str], headers: dict[str, str], read: Locale) -> None:
    assert Locale.of(SimpleNamespace(query_params=query, headers=headers)) is read


def test_each_language_picks_its_own_sentence() -> None:
    assert [locale.pick("hello", "bonjour") for locale in Locale] == ["hello", "bonjour"]


def test_the_defaults_are_names_for_languages_never_languages_of_their_own() -> None:
    assert list(Locale) == [Locale.EN, Locale.FR] and (Locale.UNSAID, Locale.FALLBACK) == (Locale.FR, Locale.EN)


def test_a_phrase_missing_a_language_fails_where_it_is_read() -> None:
    with pytest.raises(ValueError, match="1 sentences for 2 languages"):
        Locale.FR.pick("hello")


@pytest.mark.parametrize(("value", "english", "french"), [
    (20_000, "20,000", "20 000"),
    (3 * 1024**3 // 2, "1.5 GB", "1,5 Go"),
    (1024**3, "1 GB", "1 Go"),
    (25 * 1024**2, "25 MB", "25 Mo"),
    (100, "1 kB", "1 ko"),
], ids=["count", "fraction-of-gb", "whole-gb", "mb", "under-a-kb"])
def test_a_number_and_a_size_read_as_each_language_writes_them(value: int, english: str, french: str) -> None:
    """A refusal names its limit as its reader writes it (« 20 000 », « 1,5 Go »), never the English digits on his French screen."""
    said = Locale.number if value == 20_000 else Locale.size
    assert (said(Locale.EN, value), said(Locale.FR, value)) == (english, french)
