"""Token counts in ONE type (`TokenCounts`): one additive shape and one spelling whatever the vendor, its raw names mapped
once at the edge that reads its response."""

from collections.abc import Iterable
from typing import Self

from pydantic import Field

from .cost import NodeUsage, UnitPrice, priced_cost
from .wire import WireModel

#: The figures of a token count, in its one spelling.
FIGURES = ("input", "output", "cache_read", "cache_write", "reasoning")


class TokenCounts(WireModel):
    """Tokens one model call used, or a sum of calls: `input` the prompt tokens neither read from nor written to a cache
    (the full input rate), `cache_read` / `cache_write` the cached rest of the prompt (each at its own rate), `output`
    the reply, reasoning included (`reasoning` the part of it a vendor names apart). Anthropic reports this shape
    natively; OpenAI and Gemini count cached tokens inside their prompt figure: carved out at the edge reading their
    response, never counted twice. A figure nothing reported is None, never a synthetic zero."""

    input: int | None = Field(default=None, ge=0)
    output: int | None = Field(default=None, ge=0)
    cache_read: int | None = Field(default=None, ge=0)
    cache_write: int | None = Field(default=None, ge=0)
    reasoning: int | None = Field(default=None, ge=0)

    def __add__(self, other: Self) -> Self:
        """Figure by figure: None only when neither reported it."""
        def both(figure: str) -> int | None:
            mine, theirs = getattr(self, figure), getattr(other, figure)
            return None if mine is None and theirs is None else (mine or 0) + (theirs or 0)
        return type(self)(**{figure: both(figure) for figure in FIGURES})

    def __sub__(self, other: Self) -> Self:
        """What this count holds beyond `other`, figure by figure, never below zero (a count is never negative): None
        only when neither reported it."""
        def beyond(figure: str) -> int | None:
            mine, theirs = getattr(self, figure), getattr(other, figure)
            return None if mine is None and theirs is None else max((mine or 0) - (theirs or 0), 0)
        return type(self)(**{figure: beyond(figure) for figure in FIGURES})

    def covers(self, other: Self) -> bool:
        """Every figure at least `other`'s (an unreported figure counts as none): true of a cumulative counter and
        any earlier report of it."""
        return all((getattr(self, figure) or 0) >= (getattr(other, figure) or 0) for figure in FIGURES)

    def since(self, previous: Self) -> Self:
        """What a cumulative counter grew since `previous`; one that went backwards restarted from zero: all of it."""
        return self - previous if self.covers(previous) else self

    @classmethod
    def sum(cls, counts: Iterable[Self]) -> Self:
        """Several calls (an agent's turns, a period's runs) as one; none: an empty count."""
        total = cls()
        for count in counts:
            total = total + count
        return total

    @classmethod
    def of_prompt(cls, prompt: int | None, output: int | None, cache_read: int | None = None, cache_write: int | None = None) -> Self:
        """A count whose source reports the WHOLE prompt (cache reads and writes inside it: OpenAI's `prompt_tokens`):
        its uncached part carved out, never counted twice."""
        return cls(input=None if prompt is None else max(prompt - (cache_read or 0) - (cache_write or 0), 0), output=output, cache_read=cache_read, cache_write=cache_write)

    @property
    def prompt(self) -> int:
        """The whole prompt the model saw: uncached + cache reads + cache writes."""
        return (self.input or 0) + (self.cache_read or 0) + (self.cache_write or 0)

    @property
    def counted(self) -> int:
        """Prompt + output: what every total sums."""
        return self.prompt + (self.output or 0)

    @property
    def reported(self) -> bool:
        """It holds at least one figure."""
        return any(getattr(self, figure) is not None for figure in FIGURES)

    def metered(self) -> NodeUsage | None:
        """The billable units, each cache kind at its own rate; None when it reports no input or output."""
        if self.input is None and self.output is None:
            return None
        return NodeUsage(token_in=self.input, token_cache_read=self.cache_read, token_cache_write=self.cache_write, token_out=self.output)

    def priced(self, prices: tuple[UnitPrice, ...]) -> float:
        """Its USD cost at a model's unit `prices` (`priced_cost`); a figure it does not report costs nothing."""
        return 0.0 if (usage := self.metered()) is None else priced_cost(prices, usage)

    @classmethod
    def from_metered(cls, usage: NodeUsage) -> Self | None:
        """The token half of a node's metered usage; None for a node not priced per token."""
        if usage.token_in is None and usage.token_out is None:
            return None
        return cls(input=usage.token_in, output=usage.token_out, cache_read=usage.token_cache_read, cache_write=usage.token_cache_write)
