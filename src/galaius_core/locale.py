"""The languages Galaius speaks (`Locale`): the one closed set every account, page, mail and message is written in."""

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


# Set past the class: an alias assigned in its body would be one more value on the wire (a JSON schema `enum`).
Locale.UNSAID, Locale.FALLBACK = Locale.FR, Locale.EN

#: Each language's place in a phrase.
_ORDER = {locale: place for place, locale in enumerate(Locale)}

#: One sentence per `Locale`, in member order.
Said = tuple[str, str]
