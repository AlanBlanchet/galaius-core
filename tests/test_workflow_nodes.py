"""The workflow node contract: ONE node shape for every implementation, the machine command built
from it, the value-type lattice the canvas shares and the machine model registry."""

import hashlib
from typing import get_args
from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core import (
    MACHINE_MODELS, VALUE_TYPES, VISION_MODEL_TASKS, MachineCommand, MachineFunctionSummary, NodeLibraryDefinition, WorkflowBlockAvailability,
    WorkflowNode, model_task_ports, value_type_accepts, provider_sovereignty, workflow_sovereignty,
    NodeSovereigntyRecord, SovereigntyRequired, actual_workflow_sovereignty, meets_requirement,
)
from interact_core.workflows import ValueType


def port(name, direction, value_type="text", **extra):
    return {"name": name, "direction": direction, "value_type": value_type, **extra}


MACHINE = {"id": str(uuid4())}
ON_MACHINE = {"target": "machine", "machine": MACHINE}
SOURCE = "print('hi')"
DIGEST = hashlib.sha256(SOURCE.encode()).hexdigest()
CONNECTION = {"id": str(uuid4()), "revision": str(uuid4()), "capability": "write"}
TOOL = {"kind": "http", "name": "lookup", "description": "Look up", "method": "POST", "path": "v1/items", "connection": {**CONNECTION, "capability": "http"}, "json_body_from_arguments": True, "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}, "required": []}}
#: One realistic node per implementation: (impl, ports, config, placement, effects).
NODES = {
    "input": ({"kind": "builtin", "op": "input"}, [port("value", "output")], {"value": "hello"}, {}, set()),
    "write_artifact": ({"kind": "builtin", "op": "write_artifact"}, [port("value", "input"), port("result", "output", "artifact")], {"connection": CONNECTION, "artifact_path": "out/a.txt"}, {}, set()),
    "subgraph": ({"kind": "subgraph", "ref": {"key": {"id": str(uuid4())}, "revision": str(uuid4())}}, [], {"n": 2}, {}, set()),
    "library": ({"kind": "subgraph", "ref": {"id": str(uuid4())}}, [port("value", "input")], {}, {}, set()),
    "agent": ({"kind": "agent", "agent": {"id": str(uuid4()), "revision": str(uuid4())}}, [port("value", "input"), port("result", "output")], {"tone": "short"}, ON_MACHINE, {"model", "machine"}),
    "model": ({"kind": "model", "provider": "openai", "model": "gpt-image", "task": "text-to-image"}, [p.model_dump() for p in model_task_ports("text-to-image")], {"prompt": "a cat"}, {}, {"model"}),
    "connector": ({"kind": "connector", "tool": TOOL}, [port("q", "input"), port("result", "output")], {"q": "x", "limit": 5}, {}, {"connector"}),
    "function": ({"kind": "function", "name": "double", "version": "a" * 64}, [port("value", "input", "number"), port("result", "output", "number")], {"value": 21}, ON_MACHINE, {"machine"}),
    "script": ({"kind": "script", "language": "python", "source_digest": DIGEST}, [port("result", "output")], {"source": SOURCE}, ON_MACHINE, {"machine"}),
}


def node(name: str, **override) -> dict:
    impl, ports, config, placement, _effects = NODES[name]
    return {"id": str(uuid4()), "label": name.title(), "x": 1.5, "y": -2, "impl": impl, "ports": ports, "config": config, "placement": placement or {"target": "server"}, **override}


@pytest.mark.parametrize("name", sorted(NODES))
def test_every_implementation_is_one_node_shape_that_round_trips(name: str) -> None:
    value = WorkflowNode.model_validate(node(name))
    assert WorkflowNode.model_validate_json(value.model_dump_json()) == value
    assert value.effects == NODES[name][4]


def test_integers_stay_integers_in_config() -> None:
    assert WorkflowNode.model_validate(node("connector")).config["limit"] == 5
    assert isinstance(WorkflowNode.model_validate(node("function")).config["value"], int)


@pytest.mark.parametrize(("name", "override", "message"), [
    ("model", {"ports": [port("prompt", "input")]}, "signature"),
    ("script", {"config": {"source": "print('other')"}}, "digest"),
    ("function", {"placement": {"target": "server"}}, "cannot run on the server"),
    ("connector", {"placement": ON_MACHINE}, "cannot run on the machine"),
    ("input", {"config": {}}, "holds its value"),
    ("connector", {"config": {"q": {"nested": True}}}, "scalar"),
    ("write_artifact", {"config": {"connection": {"id": "x"}, "artifact_path": "a"}}, "validation error"),
])
def test_a_node_refuses_what_its_implementation_rules_out(name: str, override: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        WorkflowNode.model_validate(node(name, **override))


def test_a_constant_is_any_config_value_but_empty() -> None:
    value = WorkflowNode.model_validate(node("function", config={"value": 21, "blank": ""}))
    assert value.constant("value") == 21 and value.constant("blank") is None and value.constant("missing") is None


def test_a_reusable_node_holds_no_other_reference_and_exposes_constants_as_optional() -> None:
    inner = node("function")
    definition = {"id": str(uuid4()), "name": "Double", "nodes": [inner], "created_at": "2026-09-24T00:00:00Z", "updated_at": "2026-09-24T00:00:00Z"}
    ports = {item.name: item for item in NodeLibraryDefinition.model_validate(definition).ports()}
    assert ports["value"].required is False and ports["result"].direction == "output"
    with pytest.raises(ValidationError, match="another reusable node"):
        NodeLibraryDefinition.model_validate({**definition, "nodes": [inner, node("library")]})


def test_a_palette_block_is_the_node_it_places() -> None:
    impl, ports, config, _placement, _effects = NODES["function"]
    block = WorkflowBlockAvailability.model_validate({"impl": impl, "name": "Double", "ports": ports, "placement": ON_MACHINE, "readiness": "executable", "reason": "Runs on this PC."})
    placed = WorkflowNode.model_validate({"id": str(uuid4()), "label": block.name, "x": 0, "y": 0, **block.model_dump(include={"impl", "ports", "config", "placement"})})
    assert placed.impl == block.impl and placed.placement == block.placement
    # A function block CAN carry a computed sovereignty (it runs somewhere, on a named machine);
    # only a pure server transform (builtin) or a collapsed subgraph cannot.
    WorkflowBlockAvailability.model_validate({**block.model_dump(), "sovereignty": "self_hosted"})
    builtin, builtin_ports, _config, _placement, _effects = NODES["input"]
    with pytest.raises(ValidationError, match="sovereignty"):
        WorkflowBlockAvailability.model_validate({"impl": builtin, "name": "Input", "ports": builtin_ports, "readiness": "executable", "reason": "Holds a constant.", "sovereignty": "vendor_api"})


def test_provider_sovereignty_reads_the_sourced_registry_and_unknown_otherwise() -> None:
    assert provider_sovereignty(None) is None  # no vendor reached (builtin, bare connector)
    assert provider_sovereignty("self_hosted") is None  # the owner's own endpoint, resolved elsewhere
    assert provider_sovereignty("gemini") == "vendor_api"  # sourced in PROVIDER_SOVEREIGNTY
    assert provider_sovereignty("not_a_real_provider") == "unknown"  # a vendor with no registry entry


def test_workflow_sovereignty_is_the_weakest_node_never_assumed_sovereign() -> None:
    assert workflow_sovereignty([]) == "self_hosted"  # nothing external at all
    assert workflow_sovereignty(["self_hosted", "self_hosted"]) == "self_hosted"
    assert workflow_sovereignty(["self_hosted", "vendor_api"]) == "vendor_api"
    assert workflow_sovereignty(["vendor_api", None]) == "unknown"  # an undecidable node outranks a known vendor
    assert workflow_sovereignty(["vendor_api", "unknown"]) == "unknown"


def test_actual_sovereignty_reduces_only_the_nodes_that_ran() -> None:
    """Threat-model #6: the ACTUAL figure comes from execution records, never the static graph —
    an untaken branch contributes nothing."""
    ran = (
        NodeSovereigntyRecord(node_id=uuid4(), sovereignty="self_hosted", source="machine_sovereignty", source_id=uuid4()),
        NodeSovereigntyRecord(node_id=uuid4(), sovereignty="vendor_api", source="connection", source_id=uuid4()),
    )
    assert actual_workflow_sovereignty(ran) == "vendor_api"
    assert actual_workflow_sovereignty(()) == "self_hosted"


def test_sovereign_required_is_a_hard_floor_not_a_preference() -> None:
    """A LOWER rank is MORE sovereign: "meets" means at least as sovereign as the declared floor."""
    requirement = SovereigntyRequired(min_sovereignty="self_hosted")
    assert meets_requirement("self_hosted", requirement) is True
    assert meets_requirement("vendor_api", requirement) is False
    assert meets_requirement("unknown", requirement) is False
    assert meets_requirement("vendor_api", None) is True  # no requirement declared: unconstrained
    looser = SovereigntyRequired(min_sovereignty="vendor_api")
    assert meets_requirement("vendor_api", looser) is True
    assert meets_requirement("unknown", looser) is False


def command(impl: dict, **fields) -> dict:
    return {"id": str(uuid4()), "nonce": str(uuid4()), "machine": MACHINE, "workspace_id": str(uuid4()), "run_id": str(uuid4()),
            "workflow": {"key": {"id": str(uuid4())}, "revision": str(uuid4())}, "node_id": str(uuid4()), "impl": impl,
            "expires_at": "2026-09-24T00:00:00Z", "signature": "a" * 64, **fields}


def test_a_machine_command_carries_the_node_impl_config_and_inputs() -> None:
    model = MachineCommand.model_validate(command({"kind": "model", "provider": "huggingface", "model": "facebook/detr-resnet-50", "task": "object-detection"}, config={"score_threshold": 0.4}, inputs={"images": ["photo.jpg"]}))
    assert model.inputs["images"] == ["photo.jpg"] and VISION_MODEL_TASKS[model.impl.model] == model.impl.task
    function = MachineCommand.model_validate(command(NODES["function"][0], inputs={"value": 21}))
    assert function.impl.version == "a" * 64 and function.inputs == {"value": 21}
    script = MachineCommand.model_validate(command(NODES["script"][0], config={"source": SOURCE}))
    assert script.config["source"] == SOURCE
    with pytest.raises(ValidationError, match="digest"):
        MachineCommand.model_validate(command(NODES["script"][0], config={"source": "rm -rf ."}))
    with pytest.raises(ValidationError, match="registry"):
        MachineCommand.model_validate(command({"kind": "model", "provider": "huggingface", "model": "someone/else", "task": "object-detection"}, inputs={"images": ["a.png"]}))
    with pytest.raises(ValidationError, match="task"):
        MachineCommand.model_validate(command(NODES["agent"][0]))
    with pytest.raises(ValidationError):
        MachineCommand.model_validate(command(NODES["connector"][0]))  # a connector never runs on a machine


def test_a_machine_function_declares_typed_ports_and_one_output() -> None:
    summary = MachineFunctionSummary(name="double", description="Doubles a number.", version="a" * 64, permission="read_only", ports=NODES["function"][1])
    assert summary.ports[1].direction == "output"
    with pytest.raises(ValidationError):
        MachineFunctionSummary(name="Bad Name", description="x", version="a" * 64, permission="read_only", ports=())


@pytest.mark.parametrize(("source", "target", "accepted"), [
    ("text", "text", True), ("mask", "image", True), ("mask", "artifact", True), ("mask", "json", True),
    ("boxes", "json", True), ("text", "image", True), ("image", "artifact", True),
    ("json", "mask", False), ("artifact", "image", False), ("text", "number", False), ("json", "text", False),
])
def test_one_value_type_lattice(source: str, target: str, accepted: bool) -> None:
    assert value_type_accepts(source, target) is accepted


def test_one_value_type_table_and_one_machine_model_registry() -> None:
    assert {spec.name for spec in VALUE_TYPES} == set(get_args(ValueType))
    assert VISION_MODEL_TASKS == {model: spec.task for model, spec in MACHINE_MODELS.items()}
