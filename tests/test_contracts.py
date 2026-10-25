"""Contract behavior and package-resource acceptance tests."""

import base64
import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from importlib.resources import files
from typing import get_args
from uuid import uuid4

import pytest
from typing import get_args

from galaius_core import MACHINE_AGENT_REQUESTS, PROJECTS_DIR, PROJECTS_ROOT, AgentCompactRequest, ContextGauge, TranscriptItem, MachineAgentAnswer, WorkspacePrepareRequest, WorkspaceSpec, AgentMedia, media_key, media_paths, GitRemote, MachineAgentSettings, MachineAgentSettingsChange, ModelChoice, ModelImplementation, ModelTask, format_criteria, model_task_ports, parse_criteria, AGENT_IMAGE_BYTES, AGENT_IMAGES_FEATURE, AGENT_IMAGES_MAX, AgentMessageSpec, AgentSendRequest, MEDIA_KEY, MACHINE_AGENT_MEDIA, KEPT_FILE_MAX
from pydantic import ValidationError
from galaius_core.sealing import SecretsSeal
from galaius_core.wire import fitted, internals, internals_from, readable
from galaius_core import ASKED_BY_MAX, MachineAgentRun, MachineFence
from galaius_core import SCREEN_QUALITIES, SCREEN_WIDTHS, ScreenAllowance, ScreenFrame, ScreenSpec
from cryptography.exceptions import InvalidTag

from galaius_core import (
    Account,
    AccountUpdate,
    PlatformError,
    PlatformErrorCode,
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
    ConnectorAction,
    ConnectorCatalog,
    ConnectorLeaf,
    ConfiguredModelRef,
    ConnectionResource,
    ConnectionResourceRef,
    CredentialRef,
    ModelEligibility,
    ModelProperty,
    PromptExecutionRef,
    ToolInputSchema,
    WorkflowFunctionTool,
    WorkflowRevisionRef,
    MachineAccelerator,
    MachineCommand,
    MachineCommandResult,
    MachineRef,
    MachineRuntime,
    MachineSummary,
    MachineEvent,
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
            key=PromptKey(namespace="galaius", slug="system"),
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


def test_contracts_are_immutable() -> None:
    account = Account(account_id=uuid4(), email="owner@example.com", locale="en", verified=True)
    with pytest.raises(ValidationError):
        account.locale = "fr"


def test_standalone_package_exports_contracts_and_bundles_all_schemas() -> None:
    assert PromptCreateRequest.__module__ == "galaius_core.prompts"
    assert WorkflowKey.__module__ == "galaius_core.workflows"
    schema_dir = files("galaius_core").joinpath("schema")
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


def test_machine_wire_contract_binds_commands_and_results() -> None:
    now = datetime.now(UTC)
    machine = MachineSummary(
        id=uuid4(), name="Workstation", state="online",
        runtimes=(MachineRuntime(provider="codex", version="1.2.3"),),
        last_seen_at=now,
    )
    assert machine.runtimes[0].provider == "codex"
    assert machine.accelerators == ()
    # Connected says nothing about being able to run anything: a PC list reads `runs_nothing`, and a
    # server that does not say (an older one, a payload written before the field) leaves it False
    # rather than « connected, runs nothing » (visual-critic 2026-10-15).
    assert machine.runs_nothing is False
    assert MachineSummary.model_validate(machine.model_dump(mode="json", exclude={"runs_nothing"})).runs_nothing is False
    assert machine.model_copy(update={"runs_nothing": True}).runs_nothing is True
    gpu_machine = machine.model_copy(update={"accelerators": (MachineAccelerator(kind="cuda", name="RTX 2070", memory_mb=8192),)})
    assert gpu_machine.accelerators[0].kind == "cuda"
    assert gpu_machine.accelerators[0].memory_mb == 8192
    with pytest.raises(ValidationError):
        MachineAccelerator(kind="tpu", name="x", memory_mb=1)
    assert MachineAccelerator(kind="none", name="none", memory_mb=0).memory_mb == 0
    assert MachineRef(id=machine.id).id == machine.id
    command = MachineCommand(
        id=uuid4(), nonce=uuid4(), machine=MachineRef(id=machine.id),
        workspace_id=uuid4(), run_id=uuid4(),
        workflow=WorkflowRevisionRef(key=WorkflowKey(id=uuid4()), revision=uuid4()),
        node_id=uuid4(), impl={"kind": "agent", "agent": {"id": str(uuid4()), "revision": str(uuid4())}},
        inputs={"task": "summarize"}, expires_at=now, signature="a" * 64,
    )
    assert command.signature == "a" * 64
    assert MachineCommandResult(command_id=command.id, nonce=command.nonce, status="succeeded", result="done")
    assert MachineEvent(command_id=command.id, sequence=1, kind="started", timestamp=now)
    with pytest.raises(ValidationError, match="requires an error"):
        MachineCommandResult(command_id=command.id, nonce=command.nonce, status="failed")


def test_a_dead_link_is_its_own_wire_failure_never_a_credential_failure() -> None:
    """A one-time link that is gone (expired / consumed / never issued) and a WRONG CREDENTIAL
    are different things to the person reading the screen: retyping the password can fix one
    and can never fix the other. The wire vocabulary therefore names them separately."""
    codes = set(get_args(PlatformErrorCode))
    assert {"link_expired", "authentication_failed"} <= codes
    assert PlatformError(code="link_expired").code == "link_expired"
    with pytest.raises(ValidationError):
        PlatformError(code="link_gone")
    assert set(get_args(PlatformError.model_fields["code"].annotation)) == codes


def test_google_unlink_failures_are_their_own_wire_codes() -> None:
    """Disconnecting a linked Google identity can fail two distinct ways a shared code would
    blur: the account has no other way to sign in (`last_sign_in_method`), or the named email
    is not one of this account's linked identities at all (`not_linked`) — never folded into
    the generic `invalid_request` / `not_found` pair another failure already uses for a
    different reason."""
    codes = set(get_args(PlatformErrorCode))
    assert {"last_sign_in_method", "not_linked"} <= codes
    assert PlatformError(code="last_sign_in_method").code == "last_sign_in_method"
    assert PlatformError(code="not_linked").code == "not_linked"
    assert set(get_args(PlatformError.model_fields["code"].annotation)) == codes


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


def test_connector_catalog_carries_sharepoint_and_onedrive_with_microsoft_oauth() -> None:
    """The owner named SharePoint and OneDrive; the catalog must accept them, distinct from the
    existing Google-OAuth-backed connectors, on their own auth kind."""
    search = ToolInputSchema(properties={}, required=())
    sharepoint = ConnectorCatalog(connectors=(
        {
            "connector": "sharepoint", "name": "SharePoint", "auth_kinds": ("microsoft_oauth",),
            "docs_url": "https://learn.microsoft.com/en-us/graph/api/site-search",
            "actions": ({
                "connector": "sharepoint", "name": "list_files", "method": "GET",
                "endpoint": "https://graph.microsoft.com/v1.0/sites", "auth_kind": "microsoft_oauth",
                "required_access": ("Sites.Read.All",), "input_schema": search.model_dump(mode="json"),
                "item_kind": "file", "docs_url": "https://learn.microsoft.com/en-us/graph/api/site-search",
            },),
        },
    ))
    assert sharepoint.connectors[0].connector == "sharepoint"
    onedrive = ConnectorCatalog(connectors=(
        {
            "connector": "onedrive", "name": "OneDrive", "auth_kinds": ("microsoft_oauth",),
            "docs_url": "https://learn.microsoft.com/en-us/graph/api/driveitem-list-children",
            "actions": ({
                "connector": "onedrive", "name": "list_files", "method": "GET",
                "endpoint": "https://graph.microsoft.com/v1.0/me/drive/root/children", "auth_kind": "microsoft_oauth",
                "required_access": ("Files.Read",), "input_schema": search.model_dump(mode="json"),
                "item_kind": "file", "docs_url": "https://learn.microsoft.com/en-us/graph/api/driveitem-list-children",
            },),
        },
    ))
    assert onedrive.connectors[0].connector == "onedrive"
    with pytest.raises(ValidationError, match="auth_kind"):
        ConnectorAction(
            connector="sharepoint", name="list_files", method="GET",
            endpoint="https://graph.microsoft.com/v1.0/sites", auth_kind="not_a_real_auth_kind",
            required_access=("Sites.Read.All",), input_schema=search, item_kind="file",
            docs_url="https://learn.microsoft.com/en-us/graph/api/site-search",
        )


def test_self_hosted_provider_connection_needs_no_credential() -> None:
    """Sovereignty's third route: a model the owner runs on his own machine. Same
    `provider_api` shape vendor connections already use, but a self-hosted endpoint on a
    trusted/private network carries no vendor credential — unlike openai/anthropic/gemini,
    which still require one."""
    resource = ConnectionResource(
        id=uuid4(), revision=uuid4(), kind="provider_api", provider="self_hosted",
        name="Home GPU box", endpoint="http://192.168.1.50:11434", models=("llama3.1:8b",),
        capabilities=("http",),
    )
    assert resource.credential is None
    with pytest.raises(ValidationError, match="requires a credential"):
        ConnectionResource(id=uuid4(), revision=uuid4(), kind="provider_api", provider="openai",
                            name="Vendor", endpoint="https://api.openai.com", capabilities=("http",))


def test_workflow_function_tool_is_a_named_agent_capability() -> None:
    """A function-as-tool is a pinned WORKFLOW exposed as a callable, named + described +
    schema'd exactly like every other agent capability — never bespoke code."""
    tool = WorkflowFunctionTool(
        kind="function", name="shout", description="Uppercase the given text.",
        workflow=WorkflowRevisionRef(key=WorkflowKey(id=uuid4()), revision=uuid4()),
        input_schema=ToolInputSchema(properties={"value": {"type": "string"}}, required=("value",)),
    )
    assert tool.kind == "function"
    assert AgentRevision(
        id=uuid4(), revision=uuid4(), name="Caller",
        prompt=PromptExecutionRef(key=PromptKey(namespace="test", slug="caller"), channel="stable", digest="1" * 64, revision=uuid4()),
        resources=(), capabilities=(tool,), created_at=datetime.now(UTC),
    ).capabilities == (tool,)


@pytest.mark.parametrize("text", ["cost_per_call_usd <= 1", "sovereignty in self_hosted|eu_sovereign", "ready", "top3d.texture >= 90%"])
def test_criteria_grammar_reads_back_what_it_writes(text: str) -> None:
    assert format_criteria(parse_criteria(text)) == text


def test_model_choice_is_for_its_node_task() -> None:
    choice = ModelChoice(task="image-to-3d", constraints=parse_criteria("cost_per_call_usd <= 1"))
    with pytest.raises(ValueError):
        ModelImplementation(kind="model", provider="fal", model="m", task="text-to-image", choice=choice)
    with pytest.raises(ValueError):
        ModelChoice(task="image-to-3d", constraints=parse_criteria("provider = fal"))


@pytest.mark.parametrize("placement", ["server", "machine"])
def test_every_model_task_names_each_port_once(placement: str) -> None:
    for task in get_args(ModelTask):
        names = [port.name for port in model_task_ports(task, placement)]
        assert len(names) == len(set(names)), task


@pytest.mark.parametrize(("path", "valid"), [
    ("Documents", True), ("Documents/Clients 2026", True),
    ("", False), ("/etc", False), ("a/", False), ("a//b", False), (".ssh", False), ("code/.git", False), ("a/..", False), ("C:/Users", False), ("a\\b", False),
])
def test_a_place_is_a_plain_folder_path_below_the_working_directory(path: str, valid: bool) -> None:
    fields = {"id": str(uuid4()), "machine": {"id": str(uuid4())}, "workspace_id": str(uuid4()), "initiator_account": str(uuid4()),
              "expires_at": datetime.now(UTC).isoformat(), "signature": "0" * 64, "op": "place_level", "path": path, "level": "read"}
    if valid:
        assert MACHINE_AGENT_REQUESTS.validate_python(fields).feature == "places"
    else:
        with pytest.raises(ValidationError):
            MACHINE_AGENT_REQUESTS.validate_python(fields)


@pytest.mark.parametrize(("name", "valid"), [
    ("0b0532af5ed750b2c900.png", True), ("0b0532af5ed750b2c900.jpeg", True),
    ("../0b0532af5ed750b2c90.png", False), ("0b0532af5ed750b2c900/x.png", False), ("0b0532af5ed750b2c900.svg", False), ("shot.png", False), ("", False),
])
def test_a_run_image_is_asked_by_its_key_never_by_a_path(name: str, valid: bool) -> None:
    fields = {"id": str(uuid4()), "machine": {"id": str(uuid4())}, "workspace_id": str(uuid4()), "initiator_account": str(uuid4()),
              "expires_at": datetime.now(UTC).isoformat(), "signature": "0" * 64, "op": "media", "run_id": str(uuid4()), "name": name}
    if valid:
        assert MACHINE_AGENT_REQUESTS.validate_python(fields).feature == "agent_media"
    else:
        with pytest.raises(ValidationError):
            MACHINE_AGENT_REQUESTS.validate_python(fields)


@pytest.mark.parametrize(("op", "extra", "feature"), [("send", {"run_id": str(uuid4()), "text": "and the footer"}, "agent_send_once"),
                                                     ("start", {"root": "projects", "role": "main", "text": "and the footer"}, "agent_start_once")])
def test_an_action_asked_again_is_signed_as_the_first(op, extra, feature) -> None:
    """A send or a start asked again under its id is signed as the first one: its idempotence is the runner's
    (`idempotent_feature`), never a field on the wire."""
    fields = {"id": str(uuid4()), "machine": {"id": str(uuid4())}, "workspace_id": str(uuid4()), "initiator_account": str(uuid4()),
              "expires_at": datetime.now(UTC).isoformat(), "signature": "0" * 64, "op": op, **extra}
    request = MACHINE_AGENT_REQUESTS.validate_python(fields)
    assert (request.idempotent_feature, request.idempotent_seconds) == (feature, 3600) and set(fields) <= set(request.model_dump(mode="json"))
    assert not {"idempotent_feature", "idempotent_seconds"} & set(request.model_dump(mode="json"))


@pytest.mark.parametrize("reads_at", [None, "now", "next_step", "turn_end"])
def test_a_send_is_signed_as_before_and_its_answer_says_when_it_is_read(reads_at) -> None:
    """A send asked again is signed as the first one (its idempotence is the runner's, never a field);
    an answer from an older PC (no `reads_at`, no `read_messages`) reads the same as one saying nothing."""
    fields = {"id": str(uuid4()), "machine": {"id": str(uuid4())}, "workspace_id": str(uuid4()), "initiator_account": str(uuid4()),
              "expires_at": datetime.now(UTC).isoformat(), "signature": "0" * 64, "op": "send", "run_id": str(uuid4()), "text": "and the footer"}
    send = MACHINE_AGENT_REQUESTS.validate_python(fields)
    assert set(send.model_dump(mode="json")) == {*fields, "type"}
    answer = MachineAgentAnswer(request_id=send.id, reads_at=reads_at, read_messages=(send.id,) if reads_at else ())
    dumped = answer.model_dump(mode="json")
    assert ("reads_at" in dumped) == (reads_at is not None) and MachineAgentAnswer.model_validate({**dumped, "later": 1}).reads_at == reads_at


@pytest.mark.parametrize(("text", "paths"), [
    ('{"file_path": "/tmp/run/shot one.PNG", "also": "~/caps/b.jpg"}', ("/tmp/run/shot one.PNG", "~/caps/b.jpg")),
    # The stream's own rendering of a tool input (Read, screenshot, run_actions).
    ("file_path='/tmp/s/input image.png' description='Read the image'", ("/tmp/s/input image.png",)),
    ("[{'type': 'screenshot', 'path': '/tmp/s/a2-conv.png'}, {'type': 'wait'}]", ("/tmp/s/a2-conv.png",)),
    ("{'path': 'pc-da/index-1440.png'}", ()),
    ("target='file:/tmp/s/02-models cold.png' region='617,6'", ("/tmp/s/02-models cold.png",)),
    ("open file:///tmp/s/b.png now", ("/tmp/s/b.png",)),
    ('{"command": "see (/tmp/c.webp) and /tmp/d.gif;"}', ("/tmp/c.webp", "/tmp/d.gif")),
    # A command naming two paths keeps each whole; a bare `~` or a URL is never an image path.
    ('{"command": "open /tmp/a and then x.png"}', ()),
    ('{"command": "ls ~ && cat /etc/x.png"}', ("/etc/x.png",)),
    ('{"url": "https://example.com/b.png"}', ()),
    ('{"command": "cp /tmp/a.png /tmp/b.png.bak"}', ("/tmp/a.png",)),
])
def test_both_sides_name_a_step_image_alike(text: str, paths: tuple[str, ...]) -> None:
    assert media_paths(text) == paths


def test_an_image_key_is_the_written_path_and_its_bytes_prove_its_type() -> None:
    assert media_key("/tmp/run/shot one.PNG").endswith(".png") and media_key("/a/x.png") != media_key("/a/./x.png")
    with pytest.raises(ValueError):
        media_key("/etc/passwd")
    png = b"\x89PNG\r\n\x1a\n" + b"\0" * 8
    assert AgentMedia(content_type="image/png", data=base64.b64encode(png).decode()).content == png
    for content_type, data in (("image/jpeg", base64.b64encode(png).decode()), ("image/png", "not base64!"), ("image/webp", base64.b64encode(b"RIFF0000WAVE").decode())):
        with pytest.raises(ValidationError):
            AgentMedia(content_type=content_type, data=data)


def test_a_message_says_words_or_shows_images_and_asks_a_runner_that_takes_them() -> None:
    png = AgentMedia(content_type="image/png", data=base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 8).decode())
    assert AgentMedia.of(png.content) == png
    shown = AgentMessageSpec(text="", images=(png,))
    assert re.fullmatch(MEDIA_KEY, png.name) and png.name.endswith(".png")
    assert "images" not in AgentMessageSpec(text="hi").model_dump(mode="json")
    for text, images in (("", ()), ("  ", ()), ("hi", (png,) * (AGENT_IMAGES_MAX + 1)),
                         ("hi", (AgentMedia(content_type="image/png", data=base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * AGENT_IMAGE_BYTES).decode()),))):
        with pytest.raises(ValidationError):
            AgentMessageSpec(text=text, images=images)
    fields = dict(id=uuid4(), machine={"id": uuid4()}, workspace_id=uuid4(), initiator_account=uuid4(), expires_at=datetime.now(UTC), signature="0" * 64, run_id=uuid4())
    assert AgentSendRequest(**fields, text="hi").needs == {AgentSendRequest.feature}
    assert AgentSendRequest(**fields, **shown.model_dump()).needs == {AgentSendRequest.feature, AGENT_IMAGES_FEATURE}


@pytest.mark.parametrize(("url", "origin"), [
    ("https://github.com/owner/repo.git", "github.com/owner/repo"), ("git@github.com:owner/repo.git", "github.com/owner/repo"),
    ("ssh://git@gitlab.example:2222/group/repo", "gitlab.example/group/repo"),
    ("https://user:secret@github.com/owner/repo", None), ("file:///srv/repo", None), ("ext::sh -c id", None), ("-uhttps://x/y", None),
    ("git@github.com:-oProxyCommand=id/x", None), ("git@-oProxy:x/y", None), ("https://github.com/owner/../repo", None), ("http://github.com/owner/repo", None),
])
def test_a_clone_address_is_one_plain_repository_the_origins_list_can_name(url: str, origin: str | None) -> None:
    fields = {"id": str(uuid4()), "machine": {"id": str(uuid4())}, "workspace_id": str(uuid4()), "initiator_account": str(uuid4()),
              "expires_at": datetime.now(UTC).isoformat(), "signature": "0" * 64, "op": "workspace_prepare", "root": "agents", "url": url}
    if origin is None:
        with pytest.raises(ValidationError):
            MACHINE_AGENT_REQUESTS.validate_python(fields)
        return
    request = MACHINE_AGENT_REQUESTS.validate_python(fields)
    remote = request.url
    assert remote.origin == origin and remote.allowed_by([origin.rsplit("/", 1)[0] + "/*"]) and not remote.allowed_by(["github.com/other/*"])
    assert request.model_dump(mode="json")["url"] == url  # the wire (and its signature) keeps the plain address


@pytest.mark.parametrize(("configured", "url"), [
    ("https://x-access-token:ghp_secret@github.com/Owner/Repo.git", "https://github.com/Owner/Repo.git"), ("https://user@github.com/o/r", "https://github.com/o/r"),
    ("ssh://git:secret@gitlab.example:2222/g/r", "ssh://git@gitlab.example:2222/g/r"), ("git@github.com:o/r.git", "git@github.com:o/r.git"),
    ("HTTPS://x:secret@github.com/o/r", "https://github.com/o/r"), ("https://u:p@secret@h.com/o/r", "https://h.com/o/r"),
    ("https://h.com/o/r?private_token=secret", None), ("http://oauth2:secret@gitlab.lan/o/r", None),
])
def test_a_checkout_origin_is_reported_without_its_credentials(configured: str, url: str | None) -> None:
    remote = GitRemote.stripped(configured)
    assert (remote and remote.url) == url and "secret" not in (remote.model_dump_json() if remote else "")
    if url is None:
        with pytest.raises(ValidationError) as refused:
            GitRemote(url=configured)
        assert "secret" not in str(refused.value)


@pytest.mark.parametrize(("root", "named"), [
    (PROJECTS_ROOT, True), ("galaius/projets", True), (PROJECTS_DIR, False), (".galaius/subscriptions", False),
    ("galaius:projets/../.galaius", False), ("galaius/../.galaius", False), ("other:projets", False),
])
def test_a_request_names_the_projects_root_by_its_token_never_a_hidden_path(root: str, named: bool) -> None:
    spec = {"root": root, "name": "aino"}
    if named:
        assert WorkspaceSpec.model_validate(spec).root == root
    else:
        with pytest.raises(ValidationError):
            WorkspaceSpec.model_validate(spec)
    with pytest.raises(ValidationError):   # the web never adds, moves nor removes it
        MachineAgentSettings(agent_roots=(PROJECTS_ROOT,))


def test_a_clone_request_keeps_the_fields_a_released_runner_verifies() -> None:
    assert set(WorkspacePrepareRequest.model_fields) == {"type", "id", "machine", "workspace_id", "initiator_account", "expires_at", "signature", "op", "root", "name", "url", "submodules"}



@pytest.mark.parametrize(("change", "narrows"), [
    ({"run_agents": False}, True), ({"agent_roots": ()}, True), ({"agent_permission": "read_only"}, True),
    ({"agent_roots": ("agents", "more")}, False), ({"agent_permission": "full_access"}, False), ({"answer_approvals": True}, False),
    ({"clone_origins": ("github.com/owner/*", "github.com/other/*")}, False),
])
def test_a_settings_change_narrows_only_when_it_grants_nothing_new(change: dict, narrows: bool) -> None:
    held = MachineAgentSettings(run_agents=True, agent_roots=("agents",), clone_origins=("github.com/owner/*",))
    assert held.model_copy(update=change).narrows(held) is narrows
    with pytest.raises(ValidationError):
        MachineAgentSettingsChange.model_validate({"settings": {"agent_permission": "full_access"}, "based_on": 0})



@pytest.mark.parametrize(("name", "value", "kept"), [
    ("DATABASE_URL", "postgres://u:p@h/db", True), ("AINO_TOKEN", "abc", True),
    ("PATH", "x", False), ("HTTPS_PROXY", "x", False), ("ANTHROPIC_API_KEY", "x", False), ("BASH_ENV", "x", False), ("lower", "x", False),
    ("API_KEY", "it's", False), ("API_KEY", "a\nB=1", False), ("API_KEY", "$(curl x|sh)", True),
])
def test_project_secrets_travel_sealed_and_only_as_safe_dotenv_lines(name: str, value: str, kept: bool) -> None:
    seal, request, project = SecretsSeal.for_token("t" * 40), uuid4(), uuid4()
    if not kept:
        with pytest.raises(ValidationError):
            seal.seal({name: value}, request=request, project=project, origin="github.com/o/r")
        return
    sealed = seal.seal({name: value}, request=request, project=project, origin="github.com/o/r")
    assert value not in sealed.model_dump_json() and seal.open(sealed, request=request) == {name: value}
    with pytest.raises(InvalidTag):
        seal.open(sealed, request=uuid4())  # replayed into another request: refused
    with pytest.raises(InvalidTag):
        seal.open(sealed.model_copy(update={"origin": "github.com/o/other"}), request=request)  # pointed at another repository: refused


@pytest.mark.parametrize("name", ["aino", "a" * 64, "x." + "b" * 62])
def test_every_name_a_taken_project_folder_tries_is_a_folder_name(name: str) -> None:
    tried = list(WorkspaceSpec.candidates(name))
    assert tried[0] == name and tried[-1].endswith("-99") and len(set(tried)) == WorkspaceSpec.TRIES and all(len(each) <= 64 for each in tried)


@pytest.mark.parametrize(("code", "kept"), [("ABCD-12345", True), ("WXYZ1234", True), ("abcd-1234", False), ("https://evil.example/x", False), ("AB-1", False)])
def test_a_sign_in_carries_only_a_device_code_never_a_link(code: str, kept: bool) -> None:
    """The web shows a program's sign-in code on its vendor's page, a constant of the server's: what a
    PC reports is a code of plain capitals and digits, never an address."""
    from galaius_core import AgentSignIn

    if kept:
        assert AgentSignIn(code=code, expires_at=datetime.now(UTC)).code == code
    else:
        with pytest.raises(ValidationError):
            AgentSignIn(code=code, expires_at=datetime.now(UTC))
    assert "url" not in AgentSignIn.model_fields


def test_a_compaction_request_is_its_own_op_asked_only_of_a_runner_that_supports_it() -> None:
    now = datetime.now(UTC)
    request = MACHINE_AGENT_REQUESTS.validate_python({
        "op": "compact", "id": str(uuid4()), "machine": {"id": str(uuid4())}, "workspace_id": str(uuid4()),
        "initiator_account": str(uuid4()), "expires_at": now.isoformat(), "signature": "a" * 64, "run_id": str(uuid4()),
    })
    assert isinstance(request, AgentCompactRequest) and request.action and request.feature == "agent_compact"
    item = TranscriptItem(at=1.0, kind="compacted", text="", compaction="auto", compacted_from=969_118, context_tokens=20_143)
    assert TranscriptItem.model_validate_json(item.model_dump_json()) == item
    assert "compaction" not in TranscriptItem(at=1.0, kind="said", text="hi").model_dump()
    # A provider's figure that is no count reads as not reported: never a 0 % ring, never a crash.
    assert ContextGauge.reported({"context_tokens": True, "context_window": 0, "compactions": -1}) == {}
    assert ContextGauge.reported({"context_tokens": 0, "context_window": 258_400, "other": 1}) == {"context_tokens": 0, "context_window": 258_400}
    with pytest.raises(ValidationError):
        ContextGauge(context_window=0)
    # Thinking counts travel when reported, and an event line that reports none carries no key for them.
    assert ContextGauge.reported({"thinking_tokens": 604, "thinking_live": -1}) == {"thinking_tokens": 604}
    assert "thinking_tokens" not in ContextGauge().model_dump_json() and ContextGauge(thinking_live=550).model_dump()["thinking_live"] == 550


def test_a_run_names_the_runs_it_works_for_and_a_dump_without_any_reads_as_before() -> None:
    """`asked_by` lists the runs that sent a run work, first asker first; an older server reads a run without it."""
    base = {"run_id": str(uuid4()), "name": "deployer", "provider": "claude", "status": "running", "root": "/w", "started_at": 1.0}
    asker = uuid4()
    run = MachineAgentRun.model_validate({**base, "asked_by": [str(asker)]})
    assert run.asked_by == (asker,) and MachineAgentRun.model_validate_json(run.model_dump_json()).asked_by == (asker,)
    assert "asked_by" not in MachineAgentRun.model_validate(base).model_dump_json()
    with pytest.raises(ValidationError):
        MachineAgentRun.model_validate({**base, "asked_by": [str(uuid4()) for _ in range(ASKED_BY_MAX + 1)]})


def test_a_machine_text_field_cuts_what_it_cannot_carry_keeping_what_was_written_first() -> None:
    """What a PC writes about itself (a refusal, a fault, why its fence is off) is bounded by the
    contract, never by a number copied beside the words: `fitted` reads the bound from the field.
    2026-10-15: a 464-character fence reason raised instead, and the web showed the dump."""
    room = MachineFence.model_fields["reason"].metadata[0].max_length
    long = "the fix first. " + "then what the program said. " * 40
    cut = fitted(MachineFence, "reason", long)
    assert len(cut) <= room and cut.startswith("the fix first.") and cut.endswith("…")
    assert MachineFence(platform="linux", available=False, on=False, reason=cut).reason == cut
    assert fitted(MachineFence, "reason", "short enough") == "short enough"
    assert fitted(MachineFence, "platform", "linux") == "linux"  # a field with its own, smaller room


def test_text_written_for_code_is_recognised_so_no_screen_shows_it() -> None:
    """Either side may hold a fault of its own making; neither relays nor renders its text."""
    try:
        MachineFence(platform="linux", available=False, on=False, reason="x" * 401)
    except ValidationError as error:
        dump = str(error)
    assert internals(dump) and "MachineFence" in dump
    assert internals("Traceback (most recent call last):\n  File \"/srv/app.py\", line 2")
    assert internals("<galaius_core.workflows.MachineFence object at 0x7f3a>")
    assert not internals("bubblewrap cannot build its sandbox here: once, as root, run sudo apparmor_parser -r …")
    assert not internals("the agent fence needs Linux (bubblewrap and Landlock)")
    # An exception class by its name is written as code spells it, never as a sentence spells it.
    # 2026-10-15: a PC's own words on the web read « KeyError: 'token' ».
    assert internals("KeyError: 'token'")
    assert internals("sign-in failed: KeyError: 'token'")
    assert internals("RuntimeError")
    assert not internals("error: this computer refused the run")
    assert not internals("Erreur de connexion : ce PC ne répond pas")


def test_a_line_that_turns_into_program_text_keeps_the_words_a_person_wrote_first() -> None:
    """A journal line opens with what was being done and ends in a dump: the reader is told the
    first part and never the rest, so the cut is a place in the text, not a yes or no."""
    line = "agent places refused: 1 validation error for MachineFence\nreason\n  String should have…"
    assert internals_from(line) == line.index("1 validation error")
    assert line[: internals_from(line)].strip() == "agent places refused:"
    assert internals_from("the run is over") is None
    # What a page may show of such a line: the words a person wrote, nothing of the rest.
    assert readable(line) == "agent places refused"
    assert readable("KeyError: 'token'") == ""
    assert readable("no route to the server") == "no route to the server"


@pytest.mark.parametrize("tools, dumped", [(None, None), ((), []), (("Read", "Grep"), ["Read", "Grep"])])
def test_an_agent_with_no_tool_list_has_every_tool_and_an_empty_list_has_none(tools, dumped) -> None:
    """`harness_tools` None (every tool) is left out of a dump, so a reader released before None existed reads its
    `()`, which meant every tool there; `()` (no tool at all) and a list are written and read back as they are."""
    content = "Answer only."
    reference = PromptExecutionRef(key=PromptKey(namespace="paradigms", slug="answer"), revision=uuid4(), digest=hashlib.sha256(content.encode()).hexdigest(), channel="draft")
    agent = AgentRevision(id=uuid4(), revision=uuid4(), name="Blob", role_key="blob", prompt=reference, resources=(), harness_tools=tools, created_at=datetime.now(UTC))
    wire = agent.model_dump(mode="json")
    assert wire.get("harness_tools", None) == dumped and ("harness_tools" in wire) == (tools is not None)
    assert AgentRevision.model_validate_json(agent.model_dump_json()).harness_tools == tools


def test_a_created_file_reaches_the_web_only_by_its_answers_token_keyed_on_its_bytes() -> None:
    """Threat-modeler W2 r1: a key derived from a step's INPUT made even a refused `create_file(path=".env")` name his
    real `.env`. The answer names the file kept, by the hash of its bytes; a step's input names none."""
    from galaius_core import FILE_KEY, created_files, file_key, file_token
    key = file_key(b"PK\x03\x04", ".docx")
    answer = "Wrote /w/out/summary.docx (docx, 4 bytes). " + file_token(key, "summary]\n.docx")
    assert created_files(answer) == ((key, "summary  .docx"),) and re.fullmatch(FILE_KEY, key)
    assert created_files(json.dumps({"path": ".env"})) == () and created_files("[galaius-file 0123 x.exe]") == ()
    from galaius_core import FILE_REFUSED, refused_file
    refusal = FILE_REFUSED + "'.env': a path inside your working folder"
    # As a stream records it: plain, or an MCP structured result (live run 2b600385: every answer came wrapped).
    for recorded in (refusal, json.dumps({"result": refusal})):
        assert refused_file(recorded) == "'.env': a path inside your working folder"
    assert (refused_file(answer), refused_file(json.dumps({"result": answer})), created_files(json.dumps({"result": answer}))) == (None, None, ((key, "summary  .docx"),))
    with pytest.raises(ValueError):
        file_key(b"MZ", ".exe")


def test_every_media_job_is_one_request_shape_and_its_tool_keeps_files_its_chat_shows():
    """One request for every capability, a typed spec per kind; each kind's tool is a file tool, and what a job makes that
    create_file does not write (a .glb) is a kept file all the same."""
    from pydantic import ValidationError

    from galaius_core import FILE_KEY, FILE_TOOLS, MEDIA_TOOLS, CREATE_FILE_TYPES, MediaRequest, file_key

    assert {tool.tool for tool in MEDIA_TOOLS.values()} | {"create_file"} <= FILE_TOOLS
    assert ".glb" not in dict(CREATE_FILE_TYPES) and re.fullmatch(FILE_KEY, file_key(b"glTF", ".glb"))
    quote = MediaRequest.model_validate({"spec": {"kind": "translation", "text": "Bonjour", "target_language": "en-GB"}})
    assert (quote.spec.kind, quote.max_cost_usd) == ("translation", None)
    for refused in ({"kind": "lyrics", "text": "x"}, {"kind": "translation", "text": "x", "target_language": "french"}):
        with pytest.raises(ValidationError):
            MediaRequest.model_validate({"spec": refused})


@pytest.mark.parametrize("task", ["text-to-music", "text-to-sound"])
def test_music_and_sounds_are_audio_from_words_and_their_files_show_in_chat(task):
    from galaius_core import FILE_CONTENT_TYPES, MediaRequest, file_key

    assert [(port.name, port.direction, port.value_type) for port in model_task_ports(task, "server")] == [("prompt", "input", "text"), ("audio", "output", "audio")]
    assert file_key(b"ID3", ".mp3").endswith(".mp3") and FILE_CONTENT_TYPES[".wav"] == "audio/wav"
    with pytest.raises(ValidationError):
        MediaRequest.model_validate({"spec": {"kind": "sound", "prompt": "rain", "seconds": 31}})


@pytest.mark.parametrize("size, offset, part, ok", [
    (10, 0, 10, True),                                            # a small file comes whole
    (MACHINE_AGENT_MEDIA + 5, 0, MACHINE_AGENT_MEDIA, True),      # a large one in full parts…
    (MACHINE_AGENT_MEDIA + 5, MACHINE_AGENT_MEDIA, 5, True),      # …and its shorter last
    (MACHINE_AGENT_MEDIA + 5, 0, 5, False),                       # a short part before the end
    (10, 4, 10, False),                                           # past the file's end
    (10, 10, 0, False),                                           # nothing left to send
    (KEPT_FILE_MAX + 1, 0, MACHINE_AGENT_MEDIA, False),           # larger than any kept file
])
def test_a_kept_file_travels_in_full_parts_each_naming_its_place_in_the_whole(size, offset, part, ok) -> None:
    """A video an agent cut is larger than one answer carries: its PC answers it part by part, each full but the last,
    so a reader joining them by `end` rebuilds the file and never mistakes a short read for its end."""
    from galaius_core import AgentFile
    key = "0" * 20 + ".mp4"
    if ok:
        answer = AgentFile.of(b"\0" * part, key, offset, size)
        assert (answer.content_type, answer.end) == ("video/mp4", offset + part)
    else:
        with pytest.raises(ValidationError):
            AgentFile.of(b"\0" * part, key, offset, size)


def test_a_capabilitys_tools_keep_files_of_their_own_formats_and_only_files_come_in_parts() -> None:
    """edit_video is a file tool like create_file (its answer's token makes the card); the formats it writes are kept
    files without being create_file formats; an image is never asked in parts; a file's key hashes its parts as one."""
    from galaius_core import CREATE_FILE_TYPES, FILE_KEY, FILE_TOOLS, MACHINE_TOOLS, AgentMediaRequest, MachineToolState, file_key
    assert {"create_file", "edit_video"} <= FILE_TOOLS and ".mp4" not in dict(CREATE_FILE_TYPES)
    assert re.fullmatch(FILE_KEY, file_key((b"ab", b"c"), ".mp4")) and file_key((b"ab", b"c"), ".mp4") == file_key(b"abc", ".mp4")
    signed = {"id": str(uuid4()), "machine": {"id": str(uuid4())}, "workspace_id": str(uuid4()), "initiator_account": str(uuid4()),
              "expires_at": datetime.now(UTC).isoformat(), "signature": "0" * 64, "run_id": str(uuid4())}
    assert AgentMediaRequest.model_validate({**signed, "name": "1" * 20 + ".webm", "offset": 7}).offset == 7
    assert "offset" not in AgentMediaRequest.model_validate({**signed, "name": "1" * 20 + ".webm"}).model_dump()
    with pytest.raises(ValidationError):
        AgentMediaRequest.model_validate({**signed, "name": "1" * 20 + ".png", "offset": 7})
    pinned = MACHINE_TOOLS["video_editing"].version
    assert MachineToolState(name="video_editing", phase="ready", version=pinned).ready
    assert not MachineToolState(name="video_editing", phase="ready", version=pinned, missing=("ffmpeg",)).ready
    assert not MachineToolState(name="video_editing", phase="ready", version="0.1").ready


def test_the_blob_is_a_tool_of_the_ui_not_a_member_of_his_team() -> None:
    from galaius_core import BLOB_ROLE, listed_role
    assert [listed_role(role) for role in (BLOB_ROLE, "main", None)] == [False, True, True]


def test_a_screen_frame_travels_as_base64_and_comes_back_the_same_bytes():
    """Owner 2026-10-23 asked to watch one computer from another. A frame's JPEG is bytes, and the
    default byte encoding (UTF-8) cannot carry them: `model_dump(mode="json")` raised
    `UnicodeDecodeError` on the first frame. Base64 both ways, round trip identical."""
    jpeg = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0xFF, 0xD9])
    frame = ScreenFrame(sequence=7, at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC), origin_x=-1920, origin_y=60,
                        w=960, h=540, scale=0.5, quality=55, jpeg=jpeg)

    body = json.dumps({"type": "screen_frame", "frame": frame.model_dump(mode="json")})
    read = ScreenFrame.model_validate(json.loads(body)["frame"])

    assert read.jpeg == jpeg and read.sequence == 7
    # The point a viewer read off this frame, in the computer's own input units.
    assert read.input_point(100, 50) == (-1720, 160)


def test_a_sealed_screen_message_is_the_same_bytes_on_the_machine_link_and_the_browser_socket():
    """Threat review 2026-10-23: the server relays a session's frames and acts as ciphertext. The
    machine link carries them as JSON (base64), the browser socket as binary (seq, then the sealed
    bytes): both spellings hold the same bytes, and a message too short to be sealed is refused."""
    from uuid import uuid4

    from galaius_core import ScreenSealed

    session, data = uuid4(), bytes(range(256)) * 3
    sealed = ScreenSealed(session=session, seq=2**40 + 7, data=data)
    over_link = ScreenSealed.model_validate(json.loads(json.dumps(sealed.model_dump(mode="json"))))
    assert over_link == sealed == ScreenSealed.from_binary(session, sealed.binary())
    assert sealed.iv("pc") != sealed.iv("viewer") and len(sealed.iv("pc")) == 12
    with pytest.raises(ValueError):
        ScreenSealed.from_binary(session, sealed.binary()[:20])


def test_a_screen_spec_walks_its_ladders_and_stops_at_both_ends():
    """A link that worsens narrows the picture before it blurs it, and neither step runs off its end."""
    asked = ScreenSpec()
    assert (asked.width, asked.quality) == (1280, 55)
    assert (asked.lighter().width, asked.lighter().quality) == (960, 45)
    assert ScreenSpec(width=SCREEN_WIDTHS[0], quality=SCREEN_QUALITIES[0]).lighter().width == SCREEN_WIDTHS[0]
    assert ScreenSpec(width=SCREEN_WIDTHS[-1], quality=SCREEN_QUALITIES[-1]).sharper().quality == SCREEN_QUALITIES[-1]
    assert ScreenSpec(max_fps=10).frame_budget_ms == 100


def test_an_allowance_runs_out_by_itself():
    """A switch the owner forgot is not a standing invitation."""
    assert not ScreenAllowance().live()
    assert ScreenAllowance(allowed=True).live(), "no end means on while it is on"
    now = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    assert ScreenAllowance(allowed=True, until=now + timedelta(minutes=1)).live(now)
    assert not ScreenAllowance(allowed=True, until=now - timedelta(seconds=1)).live(now)
