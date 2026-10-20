import pytest

from galaius_core import RunState, tally


@pytest.mark.parametrize("states, tree", [
    (("done", "working"), "working"),  # his coordinator's turn over, an agent still at it: the tree works
    (("done", "waiting", "failed"), "waiting"),
    (("done", "stopped", "failed", "lost"), "lost"),
    (("done", "stopped"), "stopped"),
    ((), "done"),
])
def test_a_tree_reads_as_its_strongest_run(states: tuple[RunState, ...], tree: RunState) -> None:
    assert tally(states) == tree


def test_a_forest_names_each_runs_top_and_its_trees_state_without_walking_twice() -> None:
    """A loop of launches (a record naming its own descendant) and a launcher no longer held are cut, never walked forever."""
    from galaius_core import Tree, forest
    launchers = {"main": None, "lead": "main", "maker": "lead", "critic": "lead", "orphan": "pruned", "a": "b", "b": "a"}
    states: dict[str, RunState] = {"main": "done", "lead": "done", "maker": "working", "critic": "failed", "orphan": "done", "a": "done", "b": "waiting"}
    trees = forest(launchers, states)
    assert {run: trees[run] for run in ("main", "lead", "maker", "critic", "orphan")} == {
        "main": Tree("main", "working"), "lead": Tree("main", "working"), "maker": Tree("main", "working"), "critic": Tree("main", "failed"),
        "orphan": Tree("orphan", "done")}
    assert {trees["a"].state, trees["b"].state} == {"waiting"}
