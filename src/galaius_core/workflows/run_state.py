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
#: Every word a run's own record says of it as every reader says it, ONE table: a PC's agent run (`MachineAgentRun.status`, its
#: launcher's record) and a workflow or chat run (`WorkflowRun.status`, `ChatRun.status`). Queued or being cancelled, it still
#: goes; a cancel is a stop; an outcome never known is lost.
STATE_OF_STATUS: dict[str, RunState] = {"queued": "working", "starting": "working", "running": "working", "cancelling": "working", "waiting": "waiting",
                                        "done": "done", "succeeded": "done", "failed": "failed", "cancelled": "stopped", "stopped": "stopped",
                                        "interrupted": "lost", "crashed": "lost", "unknown_outcome": "lost"}


def state_of(word: str) -> RunState:
    """A run's own word (a PC's agent run, a workflow or chat run: `STATE_OF_STATUS`) as every reader says it. A word this
    reader does not know (a newer PC's or server's) reads `working`: a run is never shown ended on a guess, and the host's process check
    ends it once its process is gone."""
    return STATE_OF_STATUS.get(word, "working")


#: Every word a run's own record says of it (`state_of` reads them): a line that is only one of them says nothing of its own.
RUN_WORDS: frozenset[str] = frozenset(STATE_OF_STATUS)
#: How several runs read as one, strongest first: any at work, else any waiting for its person, else the worst ending.
_RANK: tuple[RunState, ...] = ("working", "waiting", "lost", "failed", "stopped", "done")


def tally(states: Iterable[RunState]) -> RunState:
    """Several runs' states as one (a tree: the run and every run below it): the first of `_RANK` any of them is in;
    `done` for none."""
    held = set(states)
    return next((state for state in _RANK if state in held), "done")


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


class RunRow(NamedTuple):
    """One run as `forest` places it: the run that launched it (None for a top run), its own state, and the runs it
    works for beside its launcher (`asked_by`: the runs that sent it work)."""

    launcher: str | None
    state: RunState
    asked_by: tuple[str, ...]


def forest(rows: Mapping[str, RunRow], held: Mapping[str, RunTree] | None = None) -> dict[str, RunTree]:
    """Every row's tree, one walk for all. A run in `held` keeps the tree already placed for it (a reader's own forest, e.g. a server's mirror); the
    rest are placed over themselves, and a top one whose launcher is held joins that launcher's tree (its top and
    project). A launcher outside both leaves its run the top of its own tree; a launch loop is cut where it closes. A
    tree's state is `tally` of the run, every run below it (any depth) and every run one of them asked (`asked_by`, its own
    state alone: a deployer working makes its asker's conversation work; the asked run stays in its own tree, whatever
    asked it). `tree_project` is the held tree's, else "":
    the caller, who knows each top run's project, names it."""
    held = held or {}
    loose = {run: row for run, row in rows.items() if run not in held}
    up = {run: row.launcher for run, row in loose.items() if row.launcher in loose and row.launcher != run}
    # The asked runs' own states, by the run that asked them (any row's: a held run's state counts in a loose asker's tree).
    asked: dict[str, list[RunState]] = {}
    for run, row in rows.items():
        for asker in row.asked_by:
            if asker in loose and asker not in (run, row.launcher):
                asked.setdefault(asker, []).append(row.state)
    below: dict[str, list[str]] = {}
    for run, launcher in up.items():
        below.setdefault(launcher, []).append(run)
    roots: dict[str, str] = {}
    for run in loose:
        path, current = [], run
        while current not in roots and current in up and current not in path:
            path.append(current)
            current = up[current]
        top = roots.get(current, current)
        roots.update({step: top for step in (*path, current)})
    rolled: dict[str, RunState] = {}
    for run in loose:  # below before above, without recursion: a tree is as deep as its agents went
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
                rolled[current] = tally((loose[current].state, *asked.get(current, ()), *(rolled[child] for child in below.get(current, ()) if child in rolled)))
    placed = {run: RunTree(**held[run].tree_fields()) for run in rows if run in held}
    for run, row in loose.items():
        joined = held.get(loose[roots[run]].launcher or "")
        placed[run] = RunTree(launcher_run_id=row.launcher, root_run_id=roots[run] if joined is None else joined.root_run_id, tree_state=rolled[run],
                              tree_project="" if joined is None else joined.tree_project)
    return placed


__all__ = ["ENDED", "RUN_STATES", "RUN_WORDS", "STATE_OF_STATUS", "RunRow", "RunState", "RunTree", "forest", "state_of", "tally"]
