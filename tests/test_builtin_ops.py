"""The builtin op registry: every `BuiltinOp` has exactly one spec, every spec's defaults make a
valid node and a valid palette block, and config-derived ports follow their config."""

import re
import time
from typing import get_args
from uuid import uuid4

import pytest
from pydantic import ValidationError

from galaius_core import BUILTIN_OPS, BuiltinOp, Plumbing, WorkflowBlockAvailability, WorkflowNode
from galaius_core.builtin_ops import BUILTIN_OP_SPECS, TemplateOp, TemplateSettings, switch_cases

OPS = sorted(BUILTIN_OPS)


def node(op: str, config: dict | None = None, ports=None) -> WorkflowNode:
    spec = BUILTIN_OPS[op]
    config = spec.defaults() if config is None else config
    return WorkflowNode.model_validate({"id": uuid4(), "label": spec.title, "x": 0, "y": 0, "impl": {"kind": "builtin", "op": op}, "ports": [port.model_dump() for port in (spec.ports(config) if ports is None else ports)], "config": config})


def test_every_op_has_one_spec():
    assert set(BUILTIN_OPS) == set(get_args(BuiltinOp))
    assert len({spec.op for spec in BUILTIN_OP_SPECS}) == len(BUILTIN_OP_SPECS) == len(BUILTIN_OPS)


@pytest.mark.parametrize("op", OPS)
def test_defaults_make_a_node_and_a_block(op):
    spec = BUILTIN_OPS[op]
    config = spec.defaults() | ({"value": "x"} if op == "input" else {})
    placed = node(op, config)
    assert placed.impl.placements == spec.placements
    block = WorkflowBlockAvailability.model_validate({"impl": {"kind": "builtin", "op": op}, "name": spec.title, "ports": [port.model_dump() for port in spec.ports(config)], "config": config, "readiness": "executable", "reason": "Runs on the server."})
    assert block.category == spec.category and block.summary == spec.summary and block.plumbing == spec.plumbing
    assert block.config_schema == ({} if not spec.fixed_ports else spec.Config.model_json_schema())
    # A block survives its own JSON (the editor and the CLI read it back).
    assert WorkflowBlockAvailability.model_validate(block.model_dump(mode="json")) == block


@pytest.mark.parametrize("op", [op for op in OPS if BUILTIN_OPS[op].fixed_ports])
def test_fixed_ports_are_enforced(op):
    spec = BUILTIN_OPS[op]
    wrong = (*spec.ports(spec.defaults())[:-1],)
    if wrong == spec.ports(spec.defaults()):
        pytest.skip("single port")
    with pytest.raises(ValidationError, match="signature"):
        node(op, ports=wrong)


@pytest.mark.parametrize(("op", "config", "message"), [
    ("condition", {"operator": "bigger"}, "operator"),
    ("switch", {"cases": [str(n) for n in range(17)]}, "16"),
    ("switch", {"cases": ["ok", ""]}, "cases"),
    ("set_fields", {"fields": [1]}, "fields"),
    ("wait", {"seconds": 7200}, "seconds"),
    ("limit", {"count": 5, "surprise": 1}, "surprise"),
    ("replace_text", {"find": ""}, "find"),
])
def test_bad_settings_refused(op, config, message):
    with pytest.raises(ValidationError, match=message):
        node(op, config, ports=BUILTIN_OPS[op].ports(BUILTIN_OPS[op].defaults()))


@pytest.mark.parametrize("op", OPS)
def test_every_op_reads_its_settings(op):
    """What the server does first for every op, whatever its kind."""
    spec = BUILTIN_OPS[op]
    spec.settings({"value": "x", "connection": {"id": "c"}, "artifact_path": "a.txt"} if not spec.fixed_ports else spec.defaults())


def test_port_constants_are_not_settings():
    """An unwired input's constant lives in config beside the settings."""
    assert node("condition", {"operator": "equals", "compare_to": "urgent"}).constant("compare_to") == "urgent"


def test_switch_ports_follow_cases():
    config = {"cases": ["Urgent!", "urgent", "Billing"]}
    assert [name for _case, name in switch_cases(config)] == ["case_urgent", "case_urgent_2", "case_billing"]
    assert [port.name for port in BUILTIN_OPS["switch"].ports(config) if port.direction == "output"] == ["case_urgent", "case_urgent_2", "case_billing", "other"]
    with pytest.raises(ValidationError, match="signature"):
        node("switch", {"cases": ["a", "b"]}, ports=BUILTIN_OPS["switch"].ports({"cases": ["a"]}))


def test_map_over_a_workflow():
    ref = {"key": {"id": str(uuid4())}, "revision": str(uuid4())}
    ports = [{"name": "item", "direction": "input", "value_type": "any"}, {"name": "result", "direction": "output", "value_type": "text", "multiple": True}]
    mapped = WorkflowNode.model_validate({"id": uuid4(), "label": "Each", "x": 0, "y": 0, "impl": {"kind": "subgraph", "ref": ref, "each": "item"}, "ports": ports})
    assert mapped.impl.each == "item"
    with pytest.raises(ValidationError, match="list"):
        WorkflowNode.model_validate({**mapped.model_dump(), "ports": [{**ports[0], "value_type": "text"}, ports[1]]})
    with pytest.raises(ValidationError, match="list"):
        WorkflowNode.model_validate({**mapped.model_dump(), "ports": [ports[0], {**ports[1], "multiple": False}]})
    with pytest.raises(ValidationError, match="reusable"):
        WorkflowNode.model_validate({**mapped.model_dump(), "impl": {"kind": "subgraph", "ref": {"id": str(uuid4())}, "each": "item"}})


def test_every_kind_is_filed_under_its_category():
    script = WorkflowBlockAvailability(impl={"kind": "script", "language": "python", "source_digest": "0" * 64}, name="Script", ports=(), config={"source": ""}, readiness="config_required", reason="Runs on a machine.")
    assert script.category == "code" and script.summary == "" and script.config_schema == {}


@pytest.mark.parametrize("template", ["{{\n" + " " * 16000 + "x", "{{" + " " * 16000, "{{a}}" + " " * 16000 + "!"])
def test_template_patterns_never_backtrack_on_a_long_template(template):
    # The editor and the server read every saved template with them: one hostile template froze both.
    began = time.perf_counter()
    re.match(TemplateOp.plumbing.only_if.pattern, template)
    re.findall(TemplateSettings.PLACEHOLDER, template)
    assert time.perf_counter() - began < 0.05


def test_plumbing_reads_only_settings_its_kind_has():
    with pytest.raises(TypeError, match="nope"):
        type("BadOp", (BUILTIN_OPS["merge"],), {"op": "bad_plumbing", "plumbing": Plumbing(badge="{nope}", badge_fr="x")})
