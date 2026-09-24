"""Contract tests for on-demand cloud placement: a node's resource requirement fitting a
connected machine's reported resources/accelerators, the cheapest catalog instance for a
requirement, and the pure scheduler decision (`choose_placement`) that picks a connected machine
or signals a cloud launch is needed — the exact fork the scheduler and the server route on."""

from uuid import uuid4

import pytest

from interact_core import (
    MachineAccelerator,
    MachineRef,
    MachineResources,
    ResourceRequirement,
    cheapest_fit,
    choose_placement,
    resources_fit,
)


def _machine() -> MachineRef:
    return MachineRef(id=uuid4())


def test_resources_fit_passes_when_every_declared_axis_is_met() -> None:
    requirement = ResourceRequirement(cpu_count=4, ram_mb=8192, gpu_kind="cuda", vram_mb=16000)
    resources = MachineResources(cpu_count=8, ram_mb=16384, disk_free_gb=100)
    accelerators = (MachineAccelerator(kind="cuda", name="RTX 4090", memory_mb=24000),)
    assert resources_fit(requirement, resources, accelerators) is True


@pytest.mark.parametrize(
    "requirement",
    [
        ResourceRequirement(cpu_count=99),
        ResourceRequirement(ram_mb=1 << 21),
        ResourceRequirement(disk_gb=1 << 15),
        ResourceRequirement(gpu_kind="cuda", vram_mb=99000),
        ResourceRequirement(gpu_kind="rocm"),
    ],
)
def test_resources_fit_fails_on_any_unmet_axis(requirement: ResourceRequirement) -> None:
    resources = MachineResources(cpu_count=8, ram_mb=16384, disk_free_gb=100)
    accelerators = (MachineAccelerator(kind="cuda", name="RTX 4090", memory_mb=24000),)
    assert resources_fit(requirement, resources, accelerators) is False


def test_resources_fit_never_assumes_unreported_resources_are_enough() -> None:
    requirement = ResourceRequirement(cpu_count=4)
    assert resources_fit(requirement, None, ()) is False
    assert resources_fit(ResourceRequirement(), None, ()) is True


def test_cheapest_fit_picks_the_smallest_catalog_entry_meeting_the_requirement() -> None:
    cpu_only = cheapest_fit(ResourceRequirement(cpu_count=2, ram_mb=2000))
    assert cpu_only is not None and cpu_only.gpu_kind == "none"
    gpu = cheapest_fit(ResourceRequirement(gpu_kind="cuda", vram_mb=40000))
    assert gpu is not None and gpu.vram_mb >= 40000
    assert cheapest_fit(ResourceRequirement(gpu_kind="cuda", vram_mb=1 << 19)) is None


def test_choose_placement_prefers_a_fitting_connected_machine_over_cloud() -> None:
    fitting = _machine()
    candidates = (
        (fitting, MachineResources(cpu_count=8, ram_mb=16384, disk_free_gb=200), (MachineAccelerator(kind="cuda", name="A100", memory_mb=40000),)),
        (_machine(), MachineResources(cpu_count=2, ram_mb=2048, disk_free_gb=10), ()),
    )
    decision = choose_placement(ResourceRequirement(gpu_kind="cuda", vram_mb=20000), candidates)
    assert decision.needs_cloud_launch is False
    assert decision.machine == fitting


def test_choose_placement_signals_cloud_launch_when_nothing_connected_fits() -> None:
    candidates = ((_machine(), MachineResources(cpu_count=2, ram_mb=2048, disk_free_gb=10), ()),)
    decision = choose_placement(ResourceRequirement(gpu_kind="cuda", vram_mb=20000), candidates)
    assert decision.needs_cloud_launch is True
    assert decision.machine is None


def test_choose_placement_rejects_a_machine_with_no_gpu_at_all_when_one_is_required() -> None:
    no_gpu = _machine()
    candidates = ((no_gpu, MachineResources(cpu_count=8, ram_mb=16384, disk_free_gb=200), ()),)
    decision = choose_placement(ResourceRequirement(gpu_kind="cuda"), candidates)
    assert decision.needs_cloud_launch is True
