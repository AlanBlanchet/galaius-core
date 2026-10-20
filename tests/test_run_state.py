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
    """A loop of launches (a record naming its own descendant) and a launcher no longer held are cut, never walked forever;
    a run whose tree was already placed keeps it, and a top run it launched joins that tree."""
    from galaius_core import RunRow, RunTree, forest
    rows = {run: RunRow(launcher, state, ()) for run, (launcher, state) in {
        "main": (None, "done"), "lead": ("main", "done"), "maker": ("lead", "working"), "critic": ("lead", "failed"), "orphan": ("pruned", "done"),
        "a": ("b", "done"), "b": ("a", "waiting"), "placed": ("coordinator", "done"), "joined": ("placed", "waiting")}.items()}
    held = {"placed": RunTree(launcher_run_id="coordinator", root_run_id="coordinator", tree_state="working", tree_project="galaius")}
    trees = {run: (tree.launcher_run_id, tree.root_run_id, tree.tree_state, tree.tree_project) for run, tree in forest(rows, held).items()}
    assert {run: trees[run] for run in ("main", "lead", "maker", "critic", "orphan", "placed", "joined")} == {
        "main": (None, "main", "working", ""), "lead": ("main", "main", "working", ""), "maker": ("lead", "main", "working", ""),
        "critic": ("lead", "main", "failed", ""), "orphan": ("pruned", "orphan", "done", ""),
        "placed": ("coordinator", "coordinator", "working", "galaius"), "joined": ("placed", "coordinator", "waiting", "galaius")}
    assert {trees["a"][2], trees["b"][2]} == {"waiting"}


def test_a_run_a_tree_asked_counts_in_its_state_and_stays_in_its_own_tree() -> None:
    """Code-reviewer + generalizer W1 r3: a deployer the coordinator asked, working, listed its conversation as working
    while the coordinator's tree read done. Its state counts in its asker's tree (and up it); it stays the top of its own."""
    from galaius_core import RunRow, forest
    trees = forest({"coordinator": RunRow(None, "done", ()), "maker": RunRow("coordinator", "done", ()), "deployer": RunRow(None, "working", ("maker",))})
    assert {run: (tree.root_run_id, tree.tree_state) for run, tree in trees.items()} == {
        "coordinator": ("coordinator", "working"), "maker": ("coordinator", "working"), "deployer": ("deployer", "working")}


@pytest.mark.parametrize("word, state", [("running", "working"), ("cancelled", "stopped"), ("crashed", "lost"), ("paused-by-a-newer-pc", "working")])
def test_a_pcs_word_reads_as_one_state_an_unknown_one_at_work(word: str, state: RunState) -> None:
    from galaius_core import state_of
    assert state_of(word) == state


def test_a_row_built_without_its_tree_fails() -> None:
    from pydantic import ValidationError

    from galaius_core import RunTree
    with pytest.raises(ValidationError):
        RunTree(root_run_id="main", tree_state="working", tree_project="galaius")  # type: ignore[call-arg]
