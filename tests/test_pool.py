from uuid import uuid4

import pytest
from interact_core import (
    BLOCKED_EGRESS_HOSTS,
    EgressAllowEntry,
    EgressPolicy,
    MachinePoolSettingsUpdate,
    MachineRef,
    MachineResources,
    PooledMachineCandidate,
    ResourceRequirement,
    RunBudgetCheck,
    check_budget,
    choose_pooled_placement,
    pooled_run_cost,
)


def test_egress_policy_denies_by_default_and_permits_only_the_exact_allowed_pair() -> None:
    policy = EgressPolicy(allow=(EgressAllowEntry(host="huggingface.co", port=443),))

    assert policy.permits("huggingface.co", 443)
    assert not policy.permits("huggingface.co", 80)
    assert not policy.permits("evil.example", 443)


def test_egress_policy_never_permits_a_blocked_host_even_if_explicitly_allow_listed() -> None:
    for blocked in BLOCKED_EGRESS_HOSTS:
        policy = EgressPolicy(allow=(EgressAllowEntry(host=blocked, port=443),))
        assert not policy.permits(blocked, 443)


def test_check_budget_refuses_a_run_that_would_cross_the_ceiling() -> None:
    check = RunBudgetCheck(workspace_id=uuid4(), committed_usd_this_period=9.5, estimated_run_usd=1.0, ceiling_usd_this_period=10.0)

    decision = check_budget(check)

    assert decision.allowed is False
    assert "10.0000" in decision.reason


def test_check_budget_allows_a_run_that_lands_exactly_on_the_ceiling() -> None:
    check = RunBudgetCheck(workspace_id=uuid4(), committed_usd_this_period=9.0, estimated_run_usd=1.0, ceiling_usd_this_period=10.0)

    decision = check_budget(check)

    assert decision.allowed is True
    assert decision.projected_usd == 10.0


def test_sharing_a_machine_without_a_price_is_refused() -> None:
    with pytest.raises(ValueError, match="needs a price"):
        MachinePoolSettingsUpdate(shared=True, price_usd_per_hour=None)
    MachinePoolSettingsUpdate(shared=True, price_usd_per_hour=0.0)  # free sharing is a valid, explicit price
    MachinePoolSettingsUpdate(shared=False)  # turning sharing off never needs a price


def test_choose_pooled_placement_picks_the_cheapest_fitting_candidate_never_the_first() -> None:
    requirement = ResourceRequirement(cpu_count=4, ram_mb=8192)
    small_but_cheap = PooledMachineCandidate(machine=MachineRef(id=uuid4()), owner_workspace_id=uuid4(), price_usd_per_hour=0.05, resources=MachineResources(cpu_count=2, ram_mb=4096, disk_free_gb=10))
    expensive_but_fits = PooledMachineCandidate(machine=MachineRef(id=uuid4()), owner_workspace_id=uuid4(), price_usd_per_hour=0.50, resources=MachineResources(cpu_count=8, ram_mb=16384, disk_free_gb=100))
    cheap_and_fits = PooledMachineCandidate(machine=MachineRef(id=uuid4()), owner_workspace_id=uuid4(), price_usd_per_hour=0.10, resources=MachineResources(cpu_count=4, ram_mb=8192, disk_free_gb=50))

    chosen = choose_pooled_placement(requirement, (small_but_cheap, expensive_but_fits, cheap_and_fits))

    assert chosen is cheap_and_fits


def test_choose_pooled_placement_returns_none_when_nothing_fits() -> None:
    requirement = ResourceRequirement(vram_mb=80_000, gpu_kind="cuda")
    candidate = PooledMachineCandidate(machine=MachineRef(id=uuid4()), owner_workspace_id=uuid4(), price_usd_per_hour=0.05, resources=MachineResources(cpu_count=64, ram_mb=999999, disk_free_gb=999))
    assert choose_pooled_placement(requirement, (candidate,)) is None


def test_pooled_run_cost_is_price_times_elapsed_hours() -> None:
    assert pooled_run_cost(price_usd_per_hour=0.7875, elapsed_seconds=1800) == 0.393750
