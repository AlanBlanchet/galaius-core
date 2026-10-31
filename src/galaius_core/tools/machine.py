"""Software a PC installs, on its owner's press, so an agent's tool runs there locally and at no cost: one row per
capability (`MACHINE_TOOLS`). Each is a pinned Python package the PC installs into a folder of its own (never the
user's site-packages), plus the programs it needs on PATH, which the PC reports missing rather than installs.

    AgentToolsRequest        -> MachineAgentAnswer.tools: every capability's state on that PC
    AgentToolInstallRequest  -> the same, its install started (a repeat while under way or ready starts nothing)

A capability not installed makes its agent tools answer one line naming where to install it: never a silent install."""

from datetime import date
from typing import Literal

from pydantic import Field

from galaius_core.tools.files import FileTool
from galaius_core.wire import WireModel

MachineToolName = Literal["video_editing"]
MachineToolPhase = Literal["absent", "installing", "ready", "failed"]
#: A runner saying it reports and installs `MACHINE_TOOLS` (`AgentToolsRequest`, `AgentToolInstallRequest`).
MACHINE_TOOLS_FEATURE = "machine_tools"


class MachineToolSpec(WireModel):
    """One capability: the package installed for it (`package`==`version`, `license`; its dependencies as released by
    `resolved_before`, so every PC installs the same set), the programs it calls from PATH, and the agent tools it backs,
    each with the files it keeps for a run (`FileTool`, read into `FILE_TOOL_REGISTRY`)."""

    name: MachineToolName
    label: str = Field(min_length=1, max_length=80)
    package: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    version: str = Field(pattern=r"^[0-9][0-9A-Za-z.+-]*$")
    license: str = Field(min_length=1, max_length=80)
    resolved_before: date
    programs: tuple[str, ...] = ()
    tools: tuple[FileTool, ...] = Field(min_length=1)


MACHINE_TOOLS: dict[MachineToolName, MachineToolSpec] = {spec.name: spec for spec in (
    # Kinocut (Apache-2.0) drives FFmpeg: cut, join, caption, overlay text, resize, convert, extract the sound.
    MachineToolSpec(name="video_editing", label="Video editing", package="kinocut", version="1.16.1", license="Apache-2.0", resolved_before=date(2026, 10, 21),
                    programs=("ffmpeg", "ffprobe"),
                    tools=(FileTool(tool="edit_video", writes=((".mp4", "video/mp4"), (".webm", "video/webm"), (".mp3", "audio/mpeg"), (".wav", "audio/wav"))),)),
)}


class MachineToolState(WireModel):
    """One capability on one PC: `phase`, the version installed (None: none), the programs PATH lacks, and why the
    last install failed (`detail`)."""

    name: MachineToolName
    phase: MachineToolPhase
    version: str | None = None
    missing: tuple[str, ...] = ()
    detail: str = Field(default="", max_length=400)

    @property
    def ready(self) -> bool:
        """Its agent tools run: installed at the pinned version, every program it calls found."""
        return self.phase == "ready" and self.version == MACHINE_TOOLS[self.name].version and not self.missing
