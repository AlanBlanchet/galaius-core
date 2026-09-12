"""Contract behavior and package-resource acceptance tests."""

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
