"""A run's state, as every reader of it says it: one vocabulary for an agent run, a conversation, a tree of runs.

A run is at work (`working`), waiting for its person (`waiting`), or ended: by itself (`done`), on an error (`failed`), by
someone's stop (`stopped`), or with its process gone and no word of its own (`lost`). A tree of runs (a conversation and
every agent it launched, at any depth) is one state too, by one rule (`tally`)."""

from collections.abc import Iterable, Mapping
from typing import Literal, NamedTuple, get_args

from pydantic import BaseModel

RunState = Literal["working", "waiting", "done", "failed", "stopped", "lost"]
RUN_STATES: tuple[RunState, ...] = get_args(RunState)
#: The states a run never leaves by itself.
ENDED: frozenset[RunState] = frozenset({"done", "failed", "stopped", "lost"})
#: A PC's own word for its run (`MachineAgentRun.status`, its launcher's record) as every reader says it.
STATE_OF_STATUS: dict[str, RunState] = {"running": "working", "starting": "working", "waiting": "waiting", "done": "done", "failed": "failed",
                                        "cancelled": "stopped", "stopped": "stopped", "interrupted": "lost", "crashed": "lost"}


def state_of(word: str) -> RunState:
    """A PC's word for its run (`MachineAgentRun.status`, its record's `status`) as every reader says it. A word this reader
    does not know (a newer PC's) reads `working`: a run is never shown ended on a guess, and the host's process check
    ends it once its process is gone."""
    return STATE_OF_STATUS.get(word, "working")
#: How several runs read as one, strongest first: any at work, else any waiting for its person, else the worst ending.
_RANK: tuple[RunState, ...] = ("working", "waiting", "lost", "failed", "stopped", "done")


def tally(states: Iterable[RunState]) -> RunState:
    """Several runs' states as one (a tree: the run and every run below it): the first of `_RANK` any of them is in;
    `done` for none."""
    held = set(states)
    return next((state for state in _RANK if state in held), "done")


class Tree(NamedTuple):
    """Where one run sits (`forest`): the run at the top of its tree and the state of the tree below it."""

    root: str
    state: RunState


class RunTree(BaseModel):
    """Where a run sits, decided by the server for every row it serves (`forest`), never walked by a reader: the run that
    launched it (`parent_run_id`, else the run whose session launched it; None for a top run), the run at the top of its
    tree (its own id for a top run), the state of the tree below it (`tally` of it and every run it launched, any depth),
    and the project of its conversation — its top run's, whatever folder it works in ("" when not known). Mixed into each
    row model carrying it, its fields at the top level, none defaulted: a row built without its tree fails, never reads
    « done »."""

    launcher_run_id: str | None
    root_run_id: str
    tree_state: RunState
    tree_project: str

    def tree_fields(self) -> dict[str, object]:
        """These fields alone, to build or update another row that carries them."""
        return {name: getattr(self, name) for name in RunTree.model_fields}

    @staticmethod
    def alone(run_id: str, state: RunState, launcher_run_id: str | None = None, project: str = "") -> dict[str, object]:
        """The fields of a run heading its own tree: before `forest` placed it, or with no launcher known."""
        return {"launcher_run_id": launcher_run_id, "root_run_id": run_id, "tree_state": state, "tree_project": project}


def forest(launchers: Mapping[str, str | None], states: Mapping[str, RunState]) -> dict[str, Tree]:
    """Every run's tree, one walk for all: `launchers` maps each run to the run that launched it (None for a top run),
    `states` its own state. A launcher outside the runs given leaves its run the top of its own tree; a launch loop is
    cut where it closes. A tree's state is `tally` of the run and every run below it, any depth."""
    up = {run: launcher for run, launcher in launchers.items() if launcher in states and launcher != run}
    below: dict[str, list[str]] = {}
    for run, launcher in up.items():
        below.setdefault(launcher, []).append(run)
    roots: dict[str, str] = {}
    for run in states:
        path, current = [], run
        while current not in roots and current in up and current not in path:
            path.append(current)
            current = up[current]
        top = roots.get(current, current)
        roots.update({step: top for step in (*path, current)})
    rolled: dict[str, RunState] = {}
    for run in states:  # below before above, without recursion: a tree is as deep as its agents went
        stack, pushed = [run], {run}
        while stack:
            current = stack[-1]
            pending = [child for child in below.get(current, ()) if child not in rolled and child not in pushed]
            if pending:
                pushed.update(pending)
                stack.extend(pending)
                continue
            stack.pop()
            if current not in rolled:
                rolled[current] = tally((states[current], *(rolled[child] for child in below.get(current, ()) if child in rolled)))
    return {run: Tree(roots[run], rolled[run]) for run in states}


__all__ = ["ENDED", "RUN_STATES", "STATE_OF_STATUS", "RunState", "RunTree", "Tree", "forest", "state_of", "tally"]
