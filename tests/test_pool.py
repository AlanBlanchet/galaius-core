from uuid import uuid4

from interact_core import BLOCKED_EGRESS_HOSTS, EgressAllowEntry, EgressPolicy, RunBudgetCheck, check_budget


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
