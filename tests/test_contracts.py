"""Contract behavior and package-resource acceptance tests."""

import hashlib
import json
from datetime import UTC, datetime
from importlib.resources import files
from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core import (
    Account,
    AccountUpdate,
    PromptCreateRequest,
    PromptKey,
    PromptRevision,
    WorkflowKey,
    WorkspaceSubscription,
    SubscriptionPlanRef,
    AgentRevision,
    AgentCatalogSnapshot,
    AgentGraph,
    AgentGraphUpdate,
    AgentRevisionRef,
    ConnectorLeaf,
    ConfiguredModelRef,
    ConnectionResourceRef,
    ModelEligibility,
    ModelProperty,
    PromptExecutionRef,
)


@pytest.mark.parametrize("status", ("unconfigured", "active", "suspended", "cancelled"))
@pytest.mark.parametrize("has_plan,has_time", ((False, False), (True, False), (False, True), (True, True)))
def test_subscription_assignment_is_complete_for_its_status(status, has_plan, has_time) -> None:
    now = datetime.now(UTC)
    values = dict(workspace_id=uuid4(), status=status, updated_at=now,
                  plan=SubscriptionPlanRef(id=uuid4(), revision=uuid4()) if has_plan else None,
                  assigned_at=now if has_time else None)
    valid = (not has_plan and not has_time) if status == "unconfigured" else has_plan and has_time
    if valid:
        assert WorkspaceSubscription(**values).status == status
    else:
        with pytest.raises(ValidationError, match="assignment"):
            WorkspaceSubscription(**values)


def test_account_update_distinguishes_omitted_locale_from_invalid_null() -> None:
    assert AccountUpdate().model_dump(exclude_unset=True) == {}
    assert AccountUpdate(locale="fr").model_dump(exclude_unset=True) == {"locale": "fr"}
    assert AccountUpdate(display_name=None).model_dump(exclude_unset=True) == {"display_name": None}
    with pytest.raises(ValidationError):
        AccountUpdate(locale=None)
    schema = AccountUpdate.model_json_schema()
    assert "locale" not in schema.get("required", ())
    assert "default" not in schema["properties"]["locale"]
    assert schema["properties"]["locale"]["type"] == "string"


def test_prompt_revision_rejects_content_digest_mismatch() -> None:
    with pytest.raises(ValidationError, match="digest does not match"):
        PromptRevision(
            key=PromptKey(namespace="interact", slug="system"),
            revision=uuid4(),
            digest="0" * 64,
            content="not the digest's content",
            source_commit="a" * 40,
            created_at=datetime.now(UTC),
        )


def test_agent_catalog_is_complete_verifiable_and_rejects_broken_reporting() -> None:
    content = "Verify conclusions against evidence."
    prompt = PromptRevision(key=PromptKey(namespace="paradigms", slug="evidence"), revision=uuid4(),
                            content=content, digest=hashlib.sha256(content.encode()).hexdigest(),
                            source_commit="a" * 40, created_at=datetime.now(UTC))
    reference = PromptExecutionRef(key=prompt.key, revision=prompt.revision, digest=prompt.digest, channel="draft")
    lead = AgentRevision(id=uuid4(), revision=uuid4(), name="Lead", role_key="lead", prompt=reference,
                         resources=(), created_at=datetime.now(UTC))
    child = AgentRevision(id=uuid4(), revision=uuid4(), name="Reviewer", role_key="reviewer", prompt=reference,
                          resources=(), reports_to=lead.id, created_at=datetime.now(UTC))
    snapshot = AgentCatalogSnapshot.create((child, lead), (prompt,))
    assert AgentCatalogSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
    assert AgentCatalogSnapshot.create((lead, child), (prompt,)).cursor == snapshot.cursor
    # Existing caches remain readable; new roots and authoritative heads enter the digest.
    legacy = snapshot.model_dump(mode="json", exclude={"root_agent", "prompt_heads"})
    assert AgentCatalogSnapshot.model_validate(legacy) == snapshot
    rooted = AgentCatalogSnapshot.create((lead, child), (prompt,),
        AgentRevisionRef(id=lead.id, revision=lead.revision), (reference,))
    assert rooted.cursor != snapshot.cursor
    assert AgentCatalogSnapshot.model_validate_json(rooted.model_dump_json()) == rooted
    with pytest.raises(ValidationError, match="root revision"):
        AgentCatalogSnapshot.create((lead, child), (prompt,), AgentRevisionRef(id=child.id, revision=child.revision))
    with pytest.raises(ValidationError, match="duplicate prompt heads"):
        AgentCatalogSnapshot.create((lead, child), (prompt,), prompt_heads=(reference, reference))
    with pytest.raises(ValidationError, match="prompt head is unavailable"):
        AgentCatalogSnapshot.create((lead, child), (prompt,),
            prompt_heads=(reference.model_copy(update={"revision": uuid4()}),))
    graph = AgentGraph(revision=rooted.cursor, root_agent=rooted.root_agent, agents=rooted.agents)
    assert graph.root_agent.id == lead.id
    assert "root_agent" not in AgentGraphUpdate(expected_revision=graph.revision).model_fields_set
    assert "root_agent" in AgentGraphUpdate(expected_revision=graph.revision, root_agent=None).model_fields_set
    damaged = snapshot.model_dump(mode="json")
    damaged["agents"][0]["name"] = "Tampered"
    with pytest.raises(ValidationError, match="cursor"):
        AgentCatalogSnapshot.model_validate(damaged)
    with pytest.raises(ValidationError, match="missing a pinned"):
        AgentCatalogSnapshot.create((lead, child), ())
    with pytest.raises(ValidationError, match="reporting hierarchy"):
        AgentCatalogSnapshot.create((child,), (prompt,))
    with pytest.raises(ValidationError, match="reporting hierarchy"):
        AgentCatalogSnapshot.create((lead.model_copy(update={"reports_to": child.id}), child), (prompt,))


def test_contracts_are_immutable_and_reject_unknown_fields() -> None:
    account = Account(
        account_id=uuid4(),
        email="owner@example.com",
        locale="en",
        verified=True,
    )
    with pytest.raises(ValidationError):
        Account(
            account_id=account.account_id,
            email=account.email,
            locale="en",
            verified=True,
            unexpected=True,
        )
    with pytest.raises(ValidationError):
        account.locale = "fr"


def test_standalone_package_exports_contracts_and_bundles_all_schemas() -> None:
    assert PromptCreateRequest.__module__ == "interact_core.prompts"
    assert WorkflowKey.__module__ == "interact_core.workflows"
    schema_dir = files("interact_core").joinpath("schema")
    schema_names = {path.name for path in schema_dir.iterdir() if path.name.endswith(".json")}
    assert schema_names == {
        "account-contracts.schema.json",
        "admin-contracts.schema.json",
        "prompt-contracts.schema.json",
        "workflow-contracts.schema.json",
    }
    for path in schema_dir.iterdir():
        if path.name.endswith(".json"):
            assert "$defs" in json.loads(path.read_text())


def test_agent_paradigms_are_ordered_and_unique() -> None:
    prompt = PromptExecutionRef(
        key=PromptKey(namespace="test", slug="main"),
        channel="stable", digest="1" * 64, revision=uuid4(),
    )
    paradigm = PromptExecutionRef(
        key=PromptKey(namespace="test", slug="safety"),
        channel="stable", digest="2" * 64, revision=uuid4(),
    )
    value = AgentRevision(
        id=uuid4(), revision=uuid4(), name="Agent", prompt=prompt,
        paradigms=(paradigm,), resources=(), created_at=datetime.now(UTC),
    )
    assert value.paradigms == (paradigm,)
    with pytest.raises(ValidationError, match="unique"):
        AgentRevision(
            id=uuid4(), revision=uuid4(), name="Agent", prompt=prompt,
            paradigms=(paradigm, paradigm), resources=(), created_at=datetime.now(UTC),
        )


def test_connector_leaf_keeps_value_type_explicit() -> None:
    assert ConnectorLeaf(name="count", type="number", value=3).value == 3
    with pytest.raises(ValidationError, match="does not match"):
        ConnectorLeaf(name="count", type="number", value="3")


def test_model_property_rankability_is_additive_and_defaults_false() -> None:
    value = ModelProperty(name="price.in", description="input cost", source="catalog", kind="number", weightable=False, percentile=True)
    assert value.rankable is False
    assert value.model_dump(mode="json")["rankable"] is False


def test_model_eligibility_rank_is_optional_positive_selection_order() -> None:
    evidence = ({"criterion": "cap.vlm", "kind": "capability", "outcome": "satisfied", "expected": True, "actual": True, "reason": "available"},)
    model = ConfiguredModelRef(connection=ConnectionResourceRef(id=uuid4(), revision=uuid4(), capability="http"), id="fixture/model")
    assert ModelEligibility(model=model, criteria="cap.vlm", outcome="eligible", evidence=evidence).rank is None
    assert ModelEligibility(model=model, criteria="cap.vlm", outcome="eligible", rank=1, evidence=evidence).rank == 1
    with pytest.raises(ValidationError):
        ModelEligibility(model=model, criteria="cap.vlm", outcome="eligible", rank=0, evidence=evidence)
