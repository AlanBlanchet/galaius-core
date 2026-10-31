"""The base every typed reason for a failure derives from (`FailureCause`, `failure.py`)."""

from typing import ClassVar

from .wire import Kinded


class Cause(Kinded):
    """Why something failed, typed: its page narrows it on `code` and words it from its figures; `message` is the same
    fact in English, for a log. `status` is the HTTP status a route answers with it."""

    status: ClassVar[int]
