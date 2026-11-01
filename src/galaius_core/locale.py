"""The languages Galaius speaks (`Locale`): the one closed set every account, page, mail and message is written in."""

import re
from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar, Protocol, TypeVar

T = TypeVar("T")


class Asking(Protocol):
    """What a request says about its reader's language: its query and its headers (a web framework's request)."""

    @property
    def query_params(self) -> Mapping[str, str]: ...

    @property
    def headers(self) -> Mapping[str, str]: ...


class Locale(StrEnum):
    """A language Galaius is written in."""

    EN = "en"
    FR = "fr"
    #: A request that names no language (no `?locale=`, no Accept-Language) is read in this one.
    UNSAID: ClassVar["Locale"]
    #: A language no member matches, or no reader at all (a stored value gone, an account deleted), is this one.
    FALLBACK: ClassVar["Locale"]

    @classmethod
    def of(cls, request: Asking) -> "Locale":
        """The reader's language for one request: `?locale=` when it names one, else the browser's first language
        (`UNSAID` when it names none, `FALLBACK` when it names one we do not write)."""
        if asked := cls.named(request.query_params.get("locale")):
            return asked
        said = (request.headers.get("accept-language") or cls.UNSAID).casefold()
        return next((locale for locale in cls if said.startswith(locale)), cls.FALLBACK)

    @classmethod
    def spoken(cls, value: str | None) -> "Locale":
        """`value` (a stored or announced language) when it is one of ours, else `FALLBACK`."""
        return cls.named(value) or cls.FALLBACK

    @classmethod
    def named(cls, value: object) -> "Locale | None":
        """The member `value` names, else None (`value in cls` raises on Python 3.11 for a non-member)."""
        return next((locale for locale in cls if locale == value), None)

    def pick(self, *sentences: T) -> T:
        """This language's own of `sentences`, one per member in member order: a new language is one more member and
        one more sentence per phrase, and a phrase still missing it fails where it is read."""
        if len(sentences) != len(_ORDER):
            raise ValueError(f"{len(sentences)} sentences for {len(_ORDER)} languages: {sentences!r}")
        return sentences[_ORDER[self]]

    def say(self, *sentences: str) -> str:
        """`pick`, set as this language writes: French puts a narrow no-break space before « : ; ? ! » and inside « »,
        where a line never breaks, and its own apostrophe (owner's pages, 2026-11-01)."""
        return self.pick(_as_written, _as_french_writes)(self.pick(*sentences))

    def number(self, value: int) -> str:
        """`value` with this language's thousands separator (« 20,000 », « 20 000 »)."""
        return self.pick(f"{value:,}", f"{value:,}".replace(",", "\u202f"))

    def size(self, size: int) -> str:
        """`size` bytes in this language's units, rounded, never broken from its unit (« 1.5 GB », « 1,5 Go », « 25 Mo », « 1 ko »)."""
        unit = self.pick(("GB", "MB", "kB"), ("Go", "Mo", "ko"))
        if size >= 1024**3:
            shown = f"{size / 1024**3:.1f}".removesuffix(".0")
            return f"{self.pick(shown, shown.replace('.', ','))}\u00a0{unit[0]}"
        return f"{max(1, round(size / 1024**2))}\u00a0{unit[1]}" if size >= 1024**2 else f"{max(1, round(size / 1024))}\u00a0{unit[2]}"


def _as_written(sentence: str) -> str:
    return sentence


#: A plain space French puts before « : ; ? ! » or after « / before », and the straight apostrophe between two letters.
_FRENCH_MARKS = re.compile(r" (?=[:;?!»])|(?<=«) |(?<=\w)'(?=\w)")


def _as_french_writes(sentence: str) -> str:
    return _FRENCH_MARKS.sub(lambda mark: "\u2019" if mark.group() == "'" else "\u202f", sentence)


# Set past the class: an alias assigned in its body would be one more value on the wire (a JSON schema `enum`).
Locale.UNSAID, Locale.FALLBACK = Locale.FR, Locale.EN

#: Each language's place in a phrase.
_ORDER = {locale: place for place, locale in enumerate(Locale)}

#: One sentence per `Locale`, in member order.
Said = tuple[str, str]
