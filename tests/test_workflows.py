"""Contract tests for the Connections family added for messaging, git hosting, mail and
webhooks: `ConnectionResource.kind_fields` accepting the two new connection kinds and the
service-connector capability relaxation (list AND/OR write, never write-forbidden), and the four
new `AgentCapability` tool kinds round-tripping through BOTH discriminated unions that gate a
workflow node (`AgentRevision.capabilities` and `ConnectorImplementation.tool` / `DirectTool`) —
the exact place a new tool kind is silently invisible to the workflow graph if only one of the two
unions is extended."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core import (
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
