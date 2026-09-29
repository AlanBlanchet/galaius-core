"""The generic resource contract: the verb ladder, grants, project rules."""

from datetime import UTC, datetime
from typing import get_args
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from interact_core.resources import VERBS, GrantRequest, Project, ProjectCreate, ProjectRule, Verb


def test_the_verb_ladder_is_the_literal_in_order():
    assert VERBS == get_args(Verb) == ("see", "read", "use", "write_on_review", "write", "manage")


@pytest.mark.parametrize(("grantee_kind", "covers", "valid"), [("agent", "team", True), ("account", "team", False), ("group", "self", True)])
def test_only_an_agent_grant_covers_a_team(grantee_kind, covers, valid):
    body = {"resource_kind": "workflow", "resource_id": str(uuid4()), "grantee_kind": grantee_kind, "grantee_id": str(uuid4()), "verb": "use", "covers": covers}
    if valid:
        GrantRequest.model_validate(body)
    else:
        with pytest.raises(ValidationError):
            GrantRequest.model_validate(body)


def test_project_rules_are_one_tagged_union():
    rules = TypeAdapter(tuple[ProjectRule, ...]).validate_python(({"kind": "folder", "path": "/work/site"}, {"kind": "agent", "id": str(uuid4())}))
    assert [rule.kind for rule in rules] == ["folder", "agent"]
    with pytest.raises(ValidationError):
        ProjectCreate(name="x" * 61, colour="accent")
    with pytest.raises(ValidationError):
        ProjectCreate(name="Site", colour="Not a key")


@pytest.mark.parametrize(("root", "agent", "claimed"), [
    ("/work/site", None, True), ("/work/site/api", None, True), ("/work/site-old", None, False),   # a folder, never a name prefix
    ("", None, False), ("/elsewhere", "bound", True), ("/elsewhere", "other", False),
])
def test_a_project_claims_runs_by_folder_or_agent(root, agent, claimed):
    bound = uuid4()
    project = Project(id=uuid4(), workspace_id=uuid4(), name="Site", colour="accent", created_by=uuid4(), created_at=datetime.now(UTC),
                      rules=({"kind": "folder", "path": "/work/site/"}, {"kind": "agent", "id": bound}))
    assert project.claims_run(root, {"bound": bound, "other": uuid4(), None: None}[agent]) is claimed
