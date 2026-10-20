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


@pytest.mark.parametrize("word, state", [("running", "working"), ("cancelled", "stopped"), ("crashed", "lost"), ("paused-by-a-newer-pc", "working")])
def test_a_pcs_word_reads_as_one_state_an_unknown_one_at_work(word: str, state: RunState) -> None:
    from galaius_core import state_of
    assert state_of(word) == state


def test_a_row_built_without_its_tree_fails() -> None:
    from pydantic import ValidationError

    from galaius_core import RunTree
    with pytest.raises(ValidationError):
        RunTree(root_run_id="main", tree_state="working", tree_project="galaius")  # type: ignore[call-arg]
