"""What a party refuses, typed once (`Refused`): the reason, for its reader to word in its own language (the refusing party
cannot know it), and the refusing party's English sentence for the logs (`str(error)`). A PC's runtime raises it, the
server raises and reads it: one shape on both sides."""

from typing import Any, ClassVar, Generic, TypeVar, get_args

from pydantic import BaseModel, TypeAdapter

from .workflows import MachineRefusal

R = TypeVar("R")
#: The kinds a machine answer carries (`MachineAgentAnswer.refusal`): an agent, a subscription or a screen refusal.
ANSWERED: tuple[type[BaseModel], ...] = get_args(get_args(MachineRefusal)[0])


def _restored(kind: type["Refused[Any]"], state: dict[str, Any], sentence: str) -> "Refused[Any]":
    """A pickled or copied refusal, rebuilt without its kind's own constructor (`Refused.__reduce__`)."""
    error = kind.__new__(kind)
    PermissionError.__init__(error, sentence)
    error.__dict__.update(state)
    return error


class Refused(PermissionError, Generic[R]):
    """A refusal: `refusal` its reason, typed `R`. A kind declares its type parameter alone (`class AgentRefused(
    Refused[AgentRefusal])`); a bare reason builds that kind's model, validated here (`AgentRefused("web_full", "4
    conversations … stop one first")`) — every refusal model names its reason `reason`.

    A `PermissionError` (an `OSError`): every path that answers a refusal already catches it; an `except OSError` around
    a call that refuses catches it too."""

    _reasons: ClassVar[TypeAdapter[Any]]
    _model: ClassVar[bool]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declared = next((get_args(base)[0] for base in cls.__dict__.get("__orig_bases__", ()) if getattr(base, "__origin__", None) is Refused), None)
        if declared is not None:
            cls._reasons = TypeAdapter(declared)
            cls._model = isinstance(declared, type) and issubclass(declared, BaseModel)

    def __init__(self, refusal: R | str | None, sentence: str) -> None:
        super().__init__(sentence)
        self.refusal: R = self._reasons.validate_python({"reason": refusal} if self._model and isinstance(refusal, str) else refusal)

    def __reduce__(self) -> tuple[Any, ...]:
        return _restored, (type(self), dict(self.__dict__), str(self))

    @staticmethod
    def refusal_of(error: BaseException) -> MachineRefusal | None:
        """The refusal a machine answer carries for `error` (`MachineAgentAnswer.refusal`): its typed reason when it is a
        `Refused` of one of the answer's kinds; None for any other error."""
        refusal = error.refusal if isinstance(error, Refused) else None
        return refusal if isinstance(refusal, ANSWERED) else None
