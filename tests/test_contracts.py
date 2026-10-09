"""Contract behavior and package-resource acceptance tests."""

import base64
import hashlib
import json
from datetime import UTC, datetime
from importlib.resources import files
from typing import get_args
from uuid import uuid4

import pytest
from typing import get_args

from galaius_core import MACHINE_AGENT_REQUESTS, WorkspacePrepareRequest, WorkspaceSpec, AgentMedia, media_key, media_paths, GitRemote, MachineAgentSettings, MachineAgentSettingsChange, ModelChoice, ModelImplementation, ModelTask, format_criteria, model_task_ports, parse_criteria
from pydantic import ValidationError
from galaius_core.sealing import SecretsSeal
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
