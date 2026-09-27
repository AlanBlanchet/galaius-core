"""Direction-decided strictness over every contract, and older-release projection."""

import hashlib
import importlib
import json
import pkgutil
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel, ValidationError, create_model

import interact_core
from interact_core import AgentCatalogSnapshot, AgentGraphUpdate, AgentRevision, PromptExecutionRef, PromptKey, PromptRevision, WorkflowNode, WorkflowRevision
from interact_core.builtin_ops import _Settings
from interact_core.wire import ContractDump, ContractView, WireModel, WireRequest

for module in pkgutil.iter_modules(interact_core.__path__):
    importlib.import_module(f"interact_core.{module.name}")


def contracts(root: type[BaseModel] = BaseModel) -> list[type[BaseModel]]:
    found = []
    for model in root.__subclasses__():
        if model.__module__.startswith("interact_core"):
            found.append(model)
        found.extend(contracts(model))
    return sorted(set(found), key=lambda model: model.__qualname__)


AUTHORED = (WireRequest, _Settings, ContractView)
"""What a client or user writes: an unknown field is a typo. Everything else is read from a peer."""


@pytest.mark.parametrize("model", contracts(), ids=lambda model: model.__qualname__)
def test_a_reader_keeps_unknown_fields_and_an_author_refuses_them(model: type[BaseModel]) -> None:
    try:
        parsed = model.model_validate({"field_from_a_newer_release": 1})
        errors = []
    except ValidationError as error:
        parsed, errors = None, [item["type"] for item in error.errors()]
    assert ("extra_forbidden" in errors) == issubclass(model, AUTHORED)
    if parsed is not None and issubclass(model, WireModel):
        assert parsed.model_dump()["field_from_a_newer_release"] == 1


def test_contract_count_covers_every_module() -> None:
    assert {model.__module__ for model in contracts()} >= {f"interact_core.{module.name}" for module in pkgutil.iter_modules(interact_core.__path__)} - {"interact_core.wire", "interact_core.__init__"}


def agent(**fields) -> tuple[AgentRevision, PromptRevision]:
    content = "Verify conclusions against evidence."
    prompt = PromptRevision(key=PromptKey(namespace="paradigms", slug="evidence"), revision=uuid4(), content=content,
                            digest=hashlib.sha256(content.encode()).hexdigest(), source_commit="a" * 40, created_at=datetime.now(UTC))
    reference = PromptExecutionRef(key=prompt.key, revision=prompt.revision, digest=prompt.digest, channel="draft")
    return AgentRevision(id=uuid4(), revision=uuid4(), name="Lead", role_key="lead", prompt=reference, resources=(),
                         created_at=datetime.now(UTC), **fields), prompt


def older(*hidden: str) -> ContractView:
    """The current contract minus `Class.field` names: a release before those fields existed."""
    fields = {model.__name__: frozenset(model.model_fields) for model in contracts()}
    for name in hidden:
        model, field = name.split(".")
        fields[model] -= {field}
    return ContractView(release="test", fields=fields)


def test_a_dump_under_a_view_keeps_its_content_and_only_the_response_is_projected() -> None:
    lead, prompt = agent(summary="Checks every claim.")
    view = older("AgentRevision.summary")
    with view.applied():
        dumped = lead.model_dump(mode="json")
        assert isinstance(dumped, ContractDump) and dumped == lead.model_dump(mode="json")
        assert lead.model_dump_json() == AgentRevision.model_dump_json(lead)
        body = view.render({"agents": [dumped], "total": 1})
    assert type(lead.model_dump()) is dict and lead.model_dump()["summary"] == "Checks every claim."
    assert "summary" not in body["agents"][0] and body["total"] == 1
    assert AgentRevision.model_validate(body["agents"][0]).summary == ""


def test_projection_drops_unknown_extras_on_known_classes_and_keeps_renamed_required_fields() -> None:
    lead, _ = agent()
    carried = AgentRevision.model_validate({**lead.model_dump(), "field_from_a_newer_release": 1})
    view = older("AgentRevision.name", "AgentRevision.summary_fr")
    projected = view.project(carried, carried.model_dump(mode="json"))
    assert "field_from_a_newer_release" not in projected and "summary_fr" not in projected
    assert projected["name"] == "Lead"
    unknown = ContractView(release="test", fields={})
    assert unknown.project(carried, carried.model_dump(mode="json"))["field_from_a_newer_release"] == 1


def test_a_projected_catalog_is_resealed_for_the_older_reader() -> None:
    lead, prompt = agent(summary="Checks every claim.")
    view = older("AgentRevision.summary")
    with view.applied():
        snapshot = AgentCatalogSnapshot.create((lead,), (prompt,))
        body = json.loads(json.dumps(view.render(snapshot.model_dump(mode="json"))))
    assert "summary" not in body["agents"][0] and body["cursor"] != snapshot.cursor
    assert body["cursor"] == AgentCatalogSnapshot.content_cursor(body)


def test_a_write_from_an_older_release_keeps_what_it_cannot_express_from_its_parent() -> None:
    stored, _ = agent(summary="Checks every claim.", summary_fr="Vérifie chaque affirmation.")
    view = older("AgentRevision.summary", "AgentRevision.summary_fr")
    released = view.project(stored, stored.model_dump(mode="json"))
    edit = AgentRevision.model_validate({**released, "revision": str(uuid4()), "parent_revision": str(stored.revision), "reasoning": "low"})
    fresh = AgentRevision.model_validate({**released, "id": str(uuid4())})
    cleared = edit.model_copy(update={"summary": ""})
    update = AgentGraphUpdate(expected_revision="0" * 64, agents=(edit, fresh))
    parents = {AgentRevision: lambda value: stored if value.parent_revision == stored.revision else None}
    # No stored parent to read: the write would reset both fields of the edit, so it is named.
    assert view.carry(update, {})[1] == ("AgentRevision.summary", "AgentRevision.summary_fr")
    carried, missed = view.carry(update, parents)
    assert missed == ()
    assert (carried.agents[0].summary, carried.agents[0].summary_fr, carried.agents[0].reasoning) == (stored.summary, stored.summary_fr, "low")
    assert (carried.agents[1].summary, carried.agents[1].summary_fr) == ("", "")  # a new agent holds nothing to reset
    own = view.carry(cleared, parents)[0]
    assert (own.summary, own.summary_fr) == ("", stored.summary_fr)  # a field the sender sent is its own
    assert view.carry(AgentGraphUpdate(expected_revision="0" * 64, root_agent=None), {}) == (AgentGraphUpdate(expected_revision="0" * 64, root_agent=None), ())


def test_a_catalog_from_a_writer_without_a_defaulted_field_verifies() -> None:
    lead, prompt = agent(summary="Checks every claim.")
    snapshot = AgentCatalogSnapshot.create((lead,), (prompt,))
    older_writer = json.loads(snapshot.model_dump_json(exclude_defaults=True))
    assert "summary_fr" not in older_writer["agents"][0]
    assert AgentCatalogSnapshot.model_validate(older_writer) == snapshot


def test_carry_refuses_an_edit_conflicting_with_a_preserved_policy() -> None:
    node = WorkflowNode(id=uuid4(), label="Script", x=0, y=0,
                        impl={"kind": "script", "language": "python", "source_digest": hashlib.sha256(b"print(1)").hexdigest()},
                        config={"source": "print(1)"}, placement={"target": "machine", "machine": {"id": uuid4()}},
                        ports=({"name": "value", "direction": "input", "value_type": "text"},), policy={"join": {"value": "any"}})
    stored = WorkflowRevision(key={"id": uuid4()}, revision=uuid4(), name="Work", nodes=(node,), edges=(), interface={}, created_at=datetime.now(UTC))
    view = older("WorkflowNode.policy")
    raw = view.project(stored, stored.model_dump(mode="json"))
    raw.update(revision=str(uuid4()), parent_revision=str(stored.revision))
    raw["nodes"][0]["ports"] = []
    edit = WorkflowRevision.model_validate(raw)
    carried, missed = view.carry(edit, {WorkflowRevision: lambda _: stored})
    assert (carried, missed) == (edit, ("WorkflowNode.policy",))  # named: the server answers 426


@pytest.mark.parametrize("new_field", ["", "new value"])
def test_catalog_hash_matches_wire_bytes_with_different_reader_writer_schemas(new_field) -> None:
    # A genuinely newer agent schema; the older reader retains its unknown default as an extra.
    WriterAgent = create_model("WriterAgent", __base__=AgentRevision, newer_default=(str, ""))
    WriterCatalog = create_model("WriterCatalog", __base__=AgentCatalogSnapshot, agents=(tuple[WriterAgent, ...], ...))
    lead, prompt = agent()
    newer = WriterAgent.model_validate({**lead.model_dump(), "newer_default": new_field})
    snapshot = WriterCatalog.create((newer,), (prompt,))
    for raw in (snapshot.model_dump(mode="json"), json.loads(snapshot.model_dump_json())):
        assert raw["cursor"] == AgentCatalogSnapshot.content_cursor(raw)
        assert AgentCatalogSnapshot.model_validate(raw).cursor == snapshot.cursor
        # Models added to a newer reader also cannot change what an older writer sealed.
        NewReader = create_model("NewReader", __base__=AgentCatalogSnapshot, agents=(tuple[WriterAgent, ...], ...))
        assert NewReader.model_validate(raw).cursor == snapshot.cursor
    # Sealing never takes over a caller's own dump options or the contract's schema.
    assert "cursor" not in snapshot.model_dump(exclude={"cursor"}) and "agents" in AgentCatalogSnapshot.model_json_schema(mode="serialization")["properties"]
