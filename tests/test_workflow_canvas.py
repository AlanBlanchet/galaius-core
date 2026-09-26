"""The board as drawn travels with the workflow revision: it round-trips whole, and a group
membership naming a group the canvas lacks is refused (a save would otherwise lose the frame)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from interact_core.workflows import WorkflowRevision

NODE = str(uuid4())


def revision(canvas: object) -> dict[str, object]:
    return {
        "key": {"id": str(uuid4())}, "revision": str(uuid4()), "name": "w", "created_at": datetime.now(UTC).isoformat(),
        "nodes": [{"id": NODE, "label": "Input", "x": 0, "y": 0, "impl": {"kind": "builtin", "op": "input"}, "config": {"value": ""},
                   "ports": [{"name": "result", "direction": "output", "value_type": "text", "required": True}]}],
        "edges": [], "interface": {}, **({} if canvas is None else {"canvas": canvas}),
    }


GROUP = {"id": "g1", "label": "Intake", "x": -40, "y": -40, "width": 300, "height": 200}


@pytest.mark.parametrize("canvas", [
    None,
    {},
    {"groups": [GROUP], "members": {NODE: "g1"}},
    {"groups": [{**GROUP, "collapsed": True, "ports": [{"name": "out", "direction": "output", "target": {"node": NODE, "port": "result"}}]}], "members": {NODE: "g1"},
     "placements": {"run-block": {"x": -300, "y": 0}}, "notes": [{"id": "n1", "text": "check totals", "x": 10, "y": 400}]},
])
def test_canvas_round_trips_with_the_revision(canvas: object) -> None:
    value = WorkflowRevision.model_validate(revision(canvas))
    again = WorkflowRevision.model_validate_json(value.model_dump_json())
    assert again == value
    assert (again.canvas is None) == (canvas is None)


@pytest.mark.parametrize("canvas", [
    {"groups": [], "members": {NODE: "missing"}},
    {"groups": [GROUP, GROUP]},
    {"groups": [{**GROUP, "width": 0}]},
])
def test_an_incoherent_canvas_is_refused(canvas: object) -> None:
    with pytest.raises(ValidationError):
        WorkflowRevision.model_validate(revision(canvas))
