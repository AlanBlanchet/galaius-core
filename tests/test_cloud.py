"""Contract tests for on-demand cloud placement: a node's resource requirement fitting a
connected machine's reported resources/accelerators, the cheapest catalog instance for a
requirement over an EXPLICIT catalog (interact-core ships no real compute data of its own — a
server-side CSV-bound registry supplies it), the sovereignty-tier floor (threat-model #6, a hard
constraint), and the pure scheduler decision (`choose_placement`) that picks a connected machine
or signals a cloud launch is needed."""

from uuid import uuid4

import pytest

from interact_core import (
    CloudInstanceType,
    MachineAccelerator,
    MachineRef,
    MachineResources,
    ResourceRequirement,
    cheapest_fit,
    choose_placement,
    meets_tier,
    resources_fit,
)


def _machine() -> MachineRef:
    return MachineRef(id=uuid4())


def _instance(name: str, price: float, sovereignty: str = "eu_sovereign", **overrides) -> CloudInstanceType:
    defaults = dict(provider="scaleway", name=name, region="fr-par-2", cpu_count=2, ram_mb=2048, disk_gb=20, price_per_hour=price, currency="EUR", jurisdiction="FR", sovereignty=sovereignty)
    defaults.update(overrides)
    return CloudInstanceType(**defaults)


_CATALOG = (
    _instance("DEV1-S", 0.008976),
    _instance("POP2-8C-32G", 0.29, cpu_count=8, ram_mb=32 * 1024),
    _instance("L4-1-24G", 0.7875, cpu_count=8, ram_mb=48 * 1024, gpu_kind="cuda", gpu_count=1, vram_mb=24 * 1024),
    _instance("H100-1-80G", 2.8665, cpu_count=24, ram_mb=240 * 1024, gpu_kind="cuda", gpu_count=1, vram_mb=80 * 1024),
    _instance("NON-EU-CHEAP", 0.001, sovereignty="non_eu"),
)


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
    cpu_only = cheapest_fit(ResourceRequirement(cpu_count=2, ram_mb=2000), _CATALOG)
    assert cpu_only is not None and cpu_only.gpu_kind == "none" and cpu_only.name == "NON-EU-CHEAP"
    gpu = cheapest_fit(ResourceRequirement(gpu_kind="cuda", vram_mb=20000), _CATALOG)
    assert gpu is not None and gpu.name == "L4-1-24G"
    assert cheapest_fit(ResourceRequirement(gpu_kind="cuda", vram_mb=1 << 19), _CATALOG) is None


def test_meets_tier_ranks_eu_sovereign_above_foreign_law_above_non_eu() -> None:
    assert meets_tier("eu_sovereign", "eu_sovereign") is True
    assert meets_tier("eu_sovereign", "non_eu") is True
    assert meets_tier("non_eu", "eu_sovereign") is False
    assert meets_tier("non_eu", None) is True


def test_cheapest_fit_min_tier_excludes_a_cheaper_non_sovereign_row() -> None:
    """Threat-model mitigation #6, hard constraint: the cheapest UNCONSTRAINED pick would be the
    non_eu row, but a sovereignty floor must never silently fall back to it."""
    unconstrained = cheapest_fit(ResourceRequirement(cpu_count=1), _CATALOG)
    assert unconstrained is not None and unconstrained.name == "NON-EU-CHEAP"
    constrained = cheapest_fit(ResourceRequirement(cpu_count=1), _CATALOG, min_tier="eu_sovereign")
    assert constrained is not None and constrained.sovereignty == "eu_sovereign"
    assert cheapest_fit(ResourceRequirement(gpu_kind="cuda"), _CATALOG, min_tier="eu_sovereign") is not None
    assert cheapest_fit(ResourceRequirement(gpu_kind="cuda", vram_mb=1 << 19), _CATALOG, min_tier="eu_sovereign") is None


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
