"""Phase A of the generic node contract: every stored node kind reads as ONE GenericNode shape,
without losing a field, and the value-type lattice is the one the canvas uses."""

import hashlib
from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core import (
    MACHINE_MODELS, VALUE_TYPES, VISION_MODEL_TASKS, GenericNode, MachineCommand, model_task_ports, value_type_accepts,
)
from pydantic import TypeAdapter

from interact_core.workflows import Node

NODE = TypeAdapter(Node)


def port(name, direction, value_type="text", **extra):
    return {"name": name, "direction": direction, "value_type": value_type, **extra}


def base(kind, **fields):
    return {"kind": kind, "id": str(uuid4()), "label": kind.title(), "x": 1.5, "y": -2, **fields}


CONNECTION = {"id": str(uuid4()), "revision": str(uuid4()), "capability": "write"}
MACHINE = {"id": str(uuid4())}
SOURCE = "print('hi')"
TOOL = {"kind": "http", "name": "lookup", "description": "Look up", "method": "POST", "path": "v1/items", "connection": {"id": str(uuid4()), "revision": str(uuid4()), "capability": "http"}, "json_body_from_arguments": True, "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": []}}
KINDS = {
    "input": (base("input", value="hello", ports=[port("value", "output")]), "builtin", {"value": "hello"}),
    "processing": (base("processing", operation="http_get", connection={**CONNECTION, "capability": "http"}, ports=[port("value", "input"), port("result", "output")]), "builtin", None),
    "result": (base("result", connection=CONNECTION, artifact_path="out/a.txt", ports=[port("value", "input"), port("result", "output", "artifact")]), "builtin", None),
    "composite": (base("composite", workflow={"key": {"id": str(uuid4())}, "revision": str(uuid4())}, variables={"n": 2.0}, ports=[]), "subgraph", {"n": 2.0}),
    "agent_task": (base("agent_task", agent={"id": str(uuid4()), "revision": str(uuid4())}, parameters={"tone": "short"}, machine=MACHINE, ports=[port("value", "input"), port("result", "output")]), "agent", {"tone": "short"}),
    "model": (base("model", provider="openai", model="gpt-image", task="text-to-image", config={"prompt": "a cat"}, placement={"target": "server"}, ports=[p.model_dump() for p in model_task_ports("text-to-image")]), "model", {"prompt": "a cat"}),
    "tool_task": (base("tool_task", tool=TOOL, arguments={"q": "x"}, ports=[port("q", "input"), port("result", "output", "json")]), "connector", {"q": "x"}),
    "machine_function": (base("machine_function", machine=MACHINE, function="resize", function_version="a" * 64, arguments={"width": 2.0}, ports=[port("width", "input", "number")]), "function", {"width": 2.0}),
    "script": (base("script", machine=MACHINE, language="python", source=SOURCE, source_digest=hashlib.sha256(SOURCE.encode()).hexdigest(), ports=[port("result", "output")]), "script", {"source": SOURCE}),
    "library": (base("library", library={"id": str(uuid4())}, ports=[port("value", "input")]), "subgraph", {}),
}


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_kind_reads_as_one_generic_node_keeping_ids_ports_and_values(kind: str) -> None:
    raw, impl_kind, config = KINDS[kind]
    node = NODE.validate_python(raw)
    generic = node.generic()
    assert GenericNode.model_validate_json(generic.model_dump_json()) == generic, "the generic shape round-trips through JSON"
    assert (generic.id, generic.label, generic.x, generic.y, generic.ports) == (node.id, node.label, node.x, node.y, node.ports)
    assert generic.impl.kind == impl_kind
    if config is not None:
        assert generic.config == config
    on_machine = "machine" in raw or raw.get("placement", {}).get("target") == "machine"
    assert ("machine" in generic.effects) == (on_machine or impl_kind in {"function", "script"})


def test_effects_name_what_a_pure_function_workflow_may_not_reach() -> None:
    effects = {kind: NODE.validate_python(raw).generic().effects for kind, (raw, _impl, _config) in KINDS.items()}
    assert effects["input"] == effects["processing"] == effects["result"] == effects["composite"] == effects["library"] == frozenset()
    assert effects["agent_task"] == {"model", "machine"} and effects["model"] == {"model"} and effects["tool_task"] == {"connector"}
    assert effects["machine_function"] == effects["script"] == {"machine"}


def test_generic_node_refuses_what_its_implementation_rules_out() -> None:
    model = NODE.validate_python(KINDS["model"][0]).generic()
    with pytest.raises(ValidationError, match="signature"):
        GenericNode.model_validate({**model.model_dump(), "ports": [port("prompt", "input")]})
    script = NODE.validate_python(KINDS["script"][0]).generic()
    with pytest.raises(ValidationError, match="digest"):
        GenericNode.model_validate({**script.model_dump(), "config": {"source": "print('other')"}})


@pytest.mark.parametrize(("source", "target", "accepted"), [
    ("text", "text", True), ("mask", "image", True), ("mask", "artifact", True), ("mask", "json", True),
    ("boxes", "json", True), ("text", "image", True), ("image", "artifact", True),
    ("json", "mask", False), ("artifact", "image", False), ("text", "number", False), ("json", "text", False),
])
def test_one_value_type_lattice(source: str, target: str, accepted: bool) -> None:
    assert value_type_accepts(source, target) is accepted


def test_one_machine_model_registry() -> None:
    assert {spec.name for spec in VALUE_TYPES} == set(__import__("typing").get_args(__import__("interact_core").workflows.ValueType))
    assert VISION_MODEL_TASKS == {model: spec.task for model, spec in MACHINE_MODELS.items()}
    with pytest.raises(ValidationError, match="registry"):
        MachineCommand.model_validate({"id": str(uuid4()), "nonce": str(uuid4()), "machine": MACHINE, "workspace_id": str(uuid4()), "run_id": str(uuid4()), "workflow": {"key": {"id": str(uuid4())}, "revision": str(uuid4())}, "node_id": str(uuid4()), "action": "model", "model": "someone/else", "image_paths": ["a.png"], "expires_at": "2026-09-24T00:00:00Z", "signature": "a" * 64})
