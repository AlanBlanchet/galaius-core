"""Contract tests for the Connections family added for messaging, git hosting, mail and
webhooks: `ConnectionResource.kind_fields` accepting the two new connection kinds and the
service-connector capability relaxation (list AND/OR write, never write-forbidden), and the four
new `AgentCapability` tool kinds round-tripping through BOTH discriminated unions that gate a
workflow node (`AgentRevision.capabilities` and `ConnectorImplementation.tool` / `DirectTool`) —
the exact place a new tool kind is silently invisible to the workflow graph if only one of the two
unions is extended."""

import base64
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from galaius_core import (
    refusal_of,
    MACHINE_AGENT_REQUESTS,
    MachineAgentAnswer,
    ProjectFileRefusal,
    ProjectFileRefused,
    ProjectTextAsk,
    PROJECT_BYTES_PART,
    ProjectFileAsk,
    ProjectFileBytes,
    ProjectFileBytesRequest,
    ProjectFileContent,
    ProjectFileRequest,
    AgentRevision,
    ConnectionResource,
    ConnectionResourceRef,
    ConnectorImplementation,
    CredentialRef,
    GitAgentTool,
    MailAgentTool,
    MailTrigger,
    SendMessageTool,
    TriggerBinding,
    WorkflowRevisionRef,
    WorkflowKey,
    PromptExecutionRef,
    PromptKey,
    ToolInputSchema,
    ValuePreview,
    WebhookAgentTool,
)


def _agent(capabilities) -> AgentRevision:
    return AgentRevision(
        id=uuid4(), revision=uuid4(), name="Caller",
        prompt=PromptExecutionRef(key=PromptKey(namespace="test", slug="caller"), channel="stable", digest="1" * 64, revision=uuid4()),
        resources=(), capabilities=capabilities, created_at=datetime.now(UTC),
    )


def test_service_connector_grants_list_and_or_write_never_neither() -> None:
    """The connector family generalized beyond read-only browsing (messaging send, git write
    operations): "write" is no longer forbidden on a service connector, but at least one of
    list/write is still required, and no other capability name is accepted."""
    discord = ConnectionResource(id=uuid4(), revision=uuid4(), kind="service_connector", name="Discord bot", provider="discord", credential=CredentialRef(id=uuid4()), capabilities=("write",))
    assert discord.capabilities == ("write",)
    github = ConnectionResource(id=uuid4(), revision=uuid4(), kind="service_connector", name="GitHub", provider="github", credential=CredentialRef(id=uuid4()), capabilities=("list", "write"))
    assert set(github.capabilities) == {"list", "write"}
    with pytest.raises(ValidationError, match="list/write"):
        ConnectionResource(id=uuid4(), revision=uuid4(), kind="service_connector", name="Empty", provider="slack", credential=CredentialRef(id=uuid4()), capabilities=())
    with pytest.raises(ValidationError, match="list/write"):
        ConnectionResource(id=uuid4(), revision=uuid4(), kind="service_connector", name="Bad", provider="slack", credential=CredentialRef(id=uuid4()), capabilities=("command",))


def test_mail_server_connection_requires_imap_endpoint_and_smtp_root() -> None:
    """`endpoint` and `root` are reused per kind, as this model already does elsewhere: for
    `mail_server` they hold the IMAP read address and the SMTP send address side by side."""
    mail = ConnectionResource(
        id=uuid4(), revision=uuid4(), kind="mail_server", name="Company mail",
        endpoint="imap://imap.example.com:993", root="smtp://smtp.example.com:587",
        credential=CredentialRef(id=uuid4()), username="bot@example.com", capabilities=("read", "write"),
    )
    assert mail.endpoint.startswith("imap://") and mail.root.startswith("smtp://")
    with pytest.raises(ValidationError, match="mail connection requires"):
        ConnectionResource(id=uuid4(), revision=uuid4(), kind="mail_server", name="Bad", endpoint="smtp://smtp.example.com:587", root="smtp://smtp.example.com:587", credential=CredentialRef(id=uuid4()), username="bot@example.com", capabilities=("read",))
    with pytest.raises(ValidationError, match="mail connection requires"):
        ConnectionResource(id=uuid4(), revision=uuid4(), kind="mail_server", name="Bad", endpoint="imap://imap.example.com:993", root="smtp://smtp.example.com:587", credential=CredentialRef(id=uuid4()), username="bot@example.com", capabilities=())


def test_webhook_connection_requires_https_endpoint_and_write_only() -> None:
    hook = ConnectionResource(id=uuid4(), revision=uuid4(), kind="webhook", name="Deploy hook", endpoint="https://api.netlify.com/hooks/abc", capabilities=("write",))
    assert hook.credential is None
    with pytest.raises(ValidationError, match="webhook connection requires"):
        ConnectionResource(id=uuid4(), revision=uuid4(), kind="webhook", name="Bad", endpoint="http://insecure.example.com/hook", capabilities=("write",))
    with pytest.raises(ValidationError, match="webhook connection requires"):
        ConnectionResource(id=uuid4(), revision=uuid4(), kind="webhook", name="Bad", endpoint="https://hooks.example.com/x", capabilities=("write", "list"))


@pytest.mark.parametrize("tool", [
    SendMessageTool(kind="send_message", name="send_discord", description="Send a Discord message.", provider="discord", connection=ConnectionResourceRef(id=uuid4(), revision=uuid4(), capability="write")),
    SendMessageTool(kind="send_message", name="send_gmail", description="Send from Gmail.", provider="gmail", account="alan"),
    GitAgentTool(kind="git", name="commit_readme", description="Commit a file on GitHub.", connector="github", operation="commit_file", connection=ConnectionResourceRef(id=uuid4(), revision=uuid4(), capability="write"), input_schema=ToolInputSchema(properties={"repository": {"type": "string"}, "branch": {"type": "string"}, "path": {"type": "string"}, "content": {"type": "string"}, "message": {"type": "string"}}, required=("repository", "branch", "path", "content", "message"))),
    MailAgentTool(kind="mail_server", name="read_mail", description="Read the inbox.", operation="read_inbox", connection=ConnectionResourceRef(id=uuid4(), revision=uuid4(), capability="read"), input_schema=ToolInputSchema(properties={"limit": {"type": "integer"}}, required=())),
    WebhookAgentTool(kind="webhook", name="post_hook", description="Post to a webhook.", connection=ConnectionResourceRef(id=uuid4(), revision=uuid4(), capability="write"), input_schema=ToolInputSchema(properties={"text": {"type": "string"}}, required=())),
])
def test_new_tool_kinds_are_valid_agent_capabilities_and_direct_tools(tool) -> None:
    """Each new tool kind must round-trip through BOTH discriminated unions a workflow node needs:
    an agent binding it as a callable capability, and a `connector`-kind graph node running it
    directly. A kind present in only one union is invisible on the other side (the exact shape of
    bug `ConnectorImplementation`'s narrower `DirectTool` union would otherwise hide)."""
    assert _agent((tool,)).capabilities == (tool,)
    assert ConnectorImplementation(kind="connector", tool=tool).tool == tool


@pytest.mark.parametrize(("fields", "error"), [
    ({"provider": "mail_server", "account": "alan"}, "saved connection, not an account"),
    ({"provider": "gmail", "connection": {"id": str(uuid4()), "revision": str(uuid4()), "capability": "write"}}, "connected account, not a saved connection"),
    ({"provider": "slack", "connection": {"id": str(uuid4()), "revision": str(uuid4()), "capability": "list"}}, "write grant"),
    ({"provider": "slack", "input_schema": {"properties": {"to": {"type": "string"}}, "required": ["to"]}}, "provider's envelope"),
])
def test_send_message_tool_refuses_a_credential_or_input_its_provider_does_not_take(fields, error) -> None:
    with pytest.raises(ValidationError, match=error):
        SendMessageTool(kind="send_message", name="send", description="Send.", **fields)


def test_send_message_node_ports_are_the_envelope_whatever_the_provider() -> None:
    """Switching the provider on a placed node must never re-wire it: every provider exposes the
    same ports, and a chat provider's input schema is the envelope subset it carries."""
    email = SendMessageTool(kind="send_message", name="send", description="Send.", provider="microsoft", account="me")
    chat = SendMessageTool(kind="send_message", name="send", description="Send.", provider="telegram")
    assert email.signature() == chat.signature()
    assert [port.name for port in email.signature() if port.required and port.direction == "input"] == ["to", "body"]
    assert set(chat.input_schema.properties) == {"to", "subject", "body"}
    assert set(email.input_schema.properties) == {"to", "subject", "body", "cc", "in_reply_to"}


def test_mail_trigger_maps_only_message_fields() -> None:
    workflow = WorkflowRevisionRef(key=WorkflowKey(id=uuid4()), revision=uuid4())
    mail = MailTrigger(kind="mail", provider="gmail", account="alan")
    assert TriggerBinding(workflow=workflow, configuration=mail, input_mapping=({"source": "body", "target_kind": "input", "target": "mail"},)).configuration == mail
    with pytest.raises(ValidationError, match="maps only"):
        TriggerBinding(workflow=workflow, configuration=mail, input_mapping=({"source": "headers", "target_kind": "input", "target": "mail"},))
    with pytest.raises(ValidationError, match="read grant"):
        MailTrigger(kind="mail", provider="mail_server", connection={"id": str(uuid4()), "revision": str(uuid4()), "capability": "write"})


def test_value_preview_image_accepts_a_larger_pre_existing_thumbnail() -> None:
    """`ValuePreview.image` carries a NEW machine-read/produced thumbnail bounded at the producer
    to VALUE_PREVIEW_MAX_PIXELS / VALUE_PREVIEW_MAX_BYTES — but the SAME shape already carries an
    older, differently-sized preview (a vision model's output overlay, historically unbounded).
    The field itself must never reject that pre-existing value: bounding is the producer's job,
    never a validation ceiling the shared wire type enforces on every past shape it also serves."""
    oversized = "a" * 200_000  # far past a 64 KB-bounded thumbnail's ~87K base64 characters
    preview = ValuePreview(kind="image", text="29 detection: book 9", media_type="image/jpeg", image=oversized)
    assert preview.image == oversized


@pytest.mark.parametrize(("path", "parsed"), [
    ("src/app.py", True), (".github/workflows/ci.yml", True), ("keys/ID_RSA", False), ("deploy/prod.pem", False),
    ("ops/service-account-prod.json", False), (".env.local", False), ("conf/.npmrc", False), ("a/./b", False),
    ("keys/signing.der", False), ("keys/app.pk8", False), ("keys/AuthKey.p8", False), ("android/store.bks", False),
    ("ops/kubeconfig.yaml", False), ("ops/prod.kubeconfig", False), ("docs/kube-config.md", True),
])
def test_a_project_path_naming_a_credential_store_or_hidden_name_is_refused_at_parse(path: str, parsed: bool) -> None:
    if parsed:
        assert ProjectFileAsk(path=path).path == path
    else:
        with pytest.raises(ValidationError):
            ProjectFileAsk(path=path)


@pytest.mark.parametrize(("fields", "parsed"), [
    ({"path": "media/clip.mp4", "offset": 0, "length": PROJECT_BYTES_PART}, True), ({"path": "a.glb", "offset": 10**12, "length": 1}, True),
    ({"path": "a.glb", "length": PROJECT_BYTES_PART + 1}, False), ({"path": "a.glb", "length": 0}, False), ({"path": "a.glb", "offset": -1}, False),
    ({"path": ".env", "length": 1}, False), ({"path": "", "length": 1}, False), ({"path": "../x.mp4", "length": 1}, False),
])
def test_a_bytes_read_asks_one_bounded_part_of_a_plain_project_path(fields: dict, parsed: bool) -> None:
    body = {"id": str(uuid4()), "machine": {"id": str(uuid4())}, "workspace_id": str(uuid4()), "initiator_account": str(uuid4()),
            "expires_at": datetime.now(UTC).isoformat(), "signature": "0" * 64, "op": "bytes", "root": "dev", "base": "app", **fields}
    if parsed:
        assert isinstance(MACHINE_AGENT_REQUESTS.validate_python(body), ProjectFileBytesRequest)
    else:
        with pytest.raises(ValidationError):
            MACHINE_AGENT_REQUESTS.validate_python(body)


@pytest.mark.parametrize(("size", "offset", "data", "parsed"), [(10, 0, b"x" * 10, True), (10, 8, b"xy", True), (10, 10, b"", True), (10, 50, b"", True),
                                                                (10, 8, b"xyz", False), (10, 10, b"x", False)])
def test_a_bytes_part_never_runs_past_its_file(size: int, offset: int, data: bytes, parsed: bool) -> None:
    fields = {"path": "a.mp4", "size": size, "offset": offset, "data": base64.b64encode(data).decode()}
    if parsed:
        assert ProjectFileBytes(**fields).content == data
    else:
        with pytest.raises(ValidationError):
            ProjectFileBytes(**fields)


@pytest.mark.parametrize(("offset", "sent"), [(0, False), (4096, True), (2**62, True), (2**62 + 1, None)])
def test_a_text_read_from_its_start_is_sent_as_a_runner_before_offsets_reads_it(offset: int, sent: bool | None) -> None:
    """Offset 0 is never sent; past 2**62 (beyond any PC's seek) none is asked."""
    fields = {"id": uuid4(), "machine": {"id": uuid4()}, "workspace_id": uuid4(), "initiator_account": uuid4(), "expires_at": datetime.now(UTC), "signature": "0" * 64,
              "root": "dev", "path": "logs/big.log", "offset": offset}
    if sent is None:
        with pytest.raises(ValidationError):
            ProjectFileRequest(**fields)
        return
    assert ("offset" in ProjectFileRequest(**fields).model_dump(mode="json")) is sent


@pytest.mark.parametrize(("offset", "end", "parsed"), [(0, 10, True), (4, 10, True), (4, 4, True), (4, 3, False), (0, 11, False)])
def test_a_text_page_lies_within_its_file(offset: int, end: int, parsed: bool) -> None:
    fields = {"path": "big.log", "size": 10, "kind": "text", "text": "x", "offset": offset, "end": end}
    if parsed:
        assert ProjectFileContent(**fields).end == end
    else:
        with pytest.raises(ValidationError):
            ProjectFileContent(**fields)


def _request(op: str, **fields) -> dict:
    return {"id": str(uuid4()), "machine": {"id": str(uuid4())}, "workspace_id": str(uuid4()), "initiator_account": str(uuid4()),
            "expires_at": datetime.now(UTC).isoformat(), "signature": "0" * 64, "op": op, "root": "dev", "base": "app", "path": "a.log", **fields}


@pytest.mark.parametrize("op", ["file", "bytes"])
def test_a_file_read_opens_unless_told_it_follows_one_and_a_released_pc_never_hears_the_default(op: str) -> None:
    first = MACHINE_AGENT_REQUESTS.validate_python(_request(op))
    later = MACHINE_AGENT_REQUESTS.validate_python(_request(op, opens=False, offset=5))
    assert first.opens and not later.opens
    assert "opens" not in first.model_dump(mode="json") and later.model_dump(mode="json")["opens"] is False


def test_a_text_read_from_a_line_start_is_a_person_s_ask_the_bytes_read_never_takes() -> None:
    assert ProjectTextAsk.model_validate({"path": "a.log", "offset": "9", "line_start": "1"}).line_start
    assert "line_start" not in MACHINE_AGENT_REQUESTS.validate_python(_request("file")).model_dump(mode="json")
    assert MACHINE_AGENT_REQUESTS.validate_python(_request("file", line_start=True)).line_start
    with pytest.raises(ValidationError):
        MACHINE_AGENT_REQUESTS.validate_python(_request("bytes", line_start=True))
    with pytest.raises(ValidationError):
        ProjectTextAsk.model_validate({"path": "a.log", "opens": "0"})


@pytest.mark.parametrize("reason", ["secret", "unchecked", "sealed", "text", "busy"])
def test_a_project_file_refusal_travels_typed_in_a_machine_answer(reason: str) -> None:
    answer = MachineAgentAnswer(request_id=uuid4(), error="a sentence", refusal=refusal_of(ProjectFileRefused(reason, "a sentence")))
    assert MachineAgentAnswer.model_validate_json(answer.model_dump_json()).refusal == ProjectFileRefusal(reason=reason)
