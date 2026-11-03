"""What a step needs of the machine it runs on: its accelerator, its resources, the language of its script. A leaf
module: `workflows` places a node by it and `step_fault` names the need no machine met."""

from typing import Annotated, Literal

from pydantic import Field

from galaius_core.wire import WireModel

AcceleratorKind = Literal["cuda", "mps", "rocm", "none"]

#: A machine's name, as its owner wrote it: every record naming one (its summary, a step it refused) holds the same ceiling.
MachineName = Annotated[str, Field(min_length=1, max_length=120)]

#: A cloud provider's names for what it sells and where: the catalog, a launched machine, a reservation and a launch
#: refusal hold the same ceilings.
InstanceTypeName = Annotated[str, Field(min_length=1, max_length=40)]
RegionName = Annotated[str, Field(min_length=1, max_length=40)]


#: What a Script step is written in; each fixes the program that runs it on the machine: Python (the
#: runner's own, or uv for declared packages), shell (`/bin/sh`: Linux / macOS), PowerShell (`pwsh`,
#: else Windows PowerShell), cmd (Windows only). A machine says which it runs (`script:<language>`
#: among its features); a step never lands on one that does not.
ScriptLanguage = Literal["python", "shell", "powershell", "cmd"]


class ResourceRequirement(WireModel):
    """What one workflow node needs to run, derived from the model registry / the user's model
    record. Every field optional: unset means "no constraint from this axis", never zero — a node
    with no declared requirement places on any connected machine, today's behaviour unchanged."""

    cpu_count: int | None = Field(default=None, ge=1, le=256)
    ram_mb: int | None = Field(default=None, ge=1, le=1 << 22)
    gpu_kind: AcceleratorKind | None = None
    vram_mb: int | None = Field(default=None, ge=1, le=1 << 20)
    disk_gb: int | None = Field(default=None, ge=1, le=1 << 16)
