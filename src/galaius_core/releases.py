"""Public release metadata shared by app, terminal, and editor clients."""

from datetime import date

from pydantic import Field

from .wire import WireModel


class ChangelogEntry(WireModel):
    version: str = Field(min_length=1, max_length=64)
    released_on: date
    changes: tuple[str, ...] = Field(min_length=1, max_length=24)


class ReleaseInfo(WireModel):
    version: str = Field(min_length=1, max_length=64)
    core_version: str = Field(min_length=1, max_length=64)
    build_id: str | None = Field(default=None, max_length=128)
    changelog: tuple[ChangelogEntry, ...] = Field(max_length=100)
