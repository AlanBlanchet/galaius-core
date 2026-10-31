"""The folders of a PC open to workflows, Data and agents: each folder's level, pending widenings,
reviews of staged writes, and the fence around them."""

from datetime import datetime
from typing import Annotated, Literal, get_args
from uuid import UUID

from pydantic import AfterValidator, Field

from galaius_core.wire import WireModel

#: How far one folder of a PC is open to workflows, Data and agents, least first; each level
#: implies the ones before it. `hidden`: not listed (every folder starts here); `see`: names, sizes
#: and dates; `read`: bytes; `write_on_review`: writes land in a staging copy the owner accepts on
#: the PC, by digest; `sandbox`: read + write in place, apart from every other root; `write`: read +
#: write in place. Set on the PC or from the web, applied at once either way; whatever the web asks,
#: the PC keeps its home folder itself, everything outside it, hidden names and credential stores
#: closed (runner feature `places_direct`; an older runner holds a widening as a `MachinePlaceChange`
#: until its owner runs `galaius machine approve` there).
PlaceLevel = Literal["hidden", "see", "read", "write_on_review", "sandbox", "write"]
PLACE_LEVELS: tuple[PlaceLevel, ...] = get_args(PlaceLevel)
#: A folder of the PC relative to its working directory (its home folder): "/"-separated names, none
#: hidden, empty, "." or "..".
def place_path(value: str) -> str:
    parts = value.split("/")
    if any(not part or part in {".", ".."} or part.startswith(".") or "\\" in part or ":" in part for part in parts):
        raise ValueError("a place is a folder relative to the working directory: '/'-separated names, none hidden, empty, '.', '..', ':' or '\\'")
    return value


PlacePath = Annotated[str, Field(min_length=1, max_length=1024), AfterValidator(place_path)]
#: A folder beneath another, by the same rules ("" the folder itself).
FolderPath = Annotated[str, Field(max_length=1024), AfterValidator(lambda value: value and place_path(value))]


class MachinePlace(WireModel):
    """One folder the owner set a level on. `refused`: why that level is not in force now (a
    credential store, a link on the way, a sandbox overlapping another root), in plain words."""

    path: str = Field(max_length=1024)
    level: PlaceLevel
    refused: str = Field(default="", max_length=400)


class MachinePlaceChange(WireModel):
    """A widening asked from the web, waiting on the PC until its owner confirms it there (`PlaceLevel`: older runners only).
    `digest`: sha256 of the canonical change (machine, path, level, previous, id), what the PC's
    local log and the server's audit both keep."""

    id: UUID
    path: str = Field(max_length=1024)
    level: PlaceLevel
    previous: PlaceLevel
    asked_at: datetime
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class MachinePlaceEntry(WireModel):
    """One name in a whole-PC browse page, with the level in force on it (names only, never bytes)."""

    name: str = Field(min_length=1, max_length=255)
    kind: Literal["folder", "file"]
    level: PlaceLevel


class MachineReviewFile(WireModel):
    path: str = Field(max_length=1024)
    change: Literal["added", "changed", "deleted"]
    size: int | None = Field(default=None, ge=0)


class MachinePlaceReview(WireModel):
    """Writes into a `write_on_review` folder, held in a staging copy on the PC. `working`: its
    agent still runs; `ready`: the owner may accept `digest` on the PC; `blocked`: it cannot be
    applied (`reason`: e.g. it touches `.git` internals), only discarded."""

    id: UUID
    place: str = Field(max_length=1024)
    origin: Literal["agent", "workflow"]
    run_id: UUID | None = None
    created_at: datetime
    state: Literal["working", "ready", "blocked"]
    reason: str = Field(default="", max_length=400)
    files: tuple[MachineReviewFile, ...] = Field(default=(), max_length=500)
    truncated: bool = False
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class MachineFence(WireModel):
    """Whether agent CLIs on this PC run inside an OS fence built from the levels. `on`: the owner
    switched it on there (`galaius machine fence on`); `available`: this PC can build one
    (`reason` says why not); agents are fenced only when both hold."""

    platform: str = Field(max_length=40)
    available: bool
    on: bool
    reason: str = Field(default="", max_length=400)


class MachinePlacesView(WireModel):
    """The PC's levels as the PC holds them now, the widenings waiting on it (`PlaceLevel`: older
    runners only), whether browsing lists file names and folders outside the home folder too
    (`browse`, `galaius machine browse on`; folder names inside it are always listed, `places_direct`),
    its fence, and the folder it suggests as a sandbox."""

    places: tuple[MachinePlace, ...] = Field(default=(), max_length=256)
    pending: tuple[MachinePlaceChange, ...] = Field(default=(), max_length=64)
    browse: bool = False
    fence: MachineFence
    suggested_sandbox: str = Field(default="", max_length=240)
