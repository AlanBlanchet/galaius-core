//! A run's state, as every reader of it says it: one vocabulary for an agent run, a conversation, a tree of runs.
//!
//! A run is at work (`working`), waiting for its person (`waiting`), or ended: by itself (`done`), on an error
//! (`failed`), by someone's stop (`stopped`), or with its process gone and no word of its own (`lost`).

use std::collections::{HashMap, HashSet};

macro_rules! states {
    ($($name:ident = $word:literal),*) => {
        /// In the order a tree reads as one, strongest first (`tally`).
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        pub enum RunState { $($name),* }

        impl RunState {
            pub const ALL: [Self; [$($word),*].len()] = [$(Self::$name),*];

            pub fn word(self) -> &'static str {
                match self { $(Self::$name => $word),* }
            }
        }
    };
}

states!(Working = "working", Waiting = "waiting", Lost = "lost", Failed = "failed", Stopped = "stopped", Done = "done");

impl RunState {
    pub fn parse(word: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|state| state.word() == word)
    }
}

/// A run's own word (an agent run, a workflow or chat run) as every reader says it.  A word this reader does not know
/// (a newer writer's) reads `working`: a run is never shown ended on a guess.
pub fn state_of(word: &str) -> RunState {
    use RunState::*;
    match word {
        "waiting" => Waiting,
        "done" | "succeeded" => Done,
        "failed" => Failed,
        "cancelled" | "stopped" => Stopped,
        "interrupted" | "crashed" | "unknown_outcome" => Lost,
        _ => Working,
    }
}

/// Several runs' states as one (a tree: the run and every run below it): any at work, else any waiting for its person,
/// else the worst ending; `done` for none.
pub fn tally(states: impl IntoIterator<Item = RunState>) -> RunState {
    states.into_iter().min().unwrap_or(RunState::Done)
}

/// One run as `forest` places it: the run that launched it (None for a top run), its own state, and the runs it works
/// for beside its launcher.
#[derive(Clone, Debug)]
pub struct RunRow {
    pub launcher: Option<String>,
    pub state: RunState,
    pub asked_by: Vec<String>,
}

/// Where a run sits: its launcher, the run at the top of its tree (its own id for a top run), the state of the tree
/// below it, and its conversation's project ("" when not known).
#[derive(Clone, Debug, PartialEq)]
pub struct RunTree {
    pub launcher_run_id: Option<String>,
    pub root_run_id: String,
    pub tree_state: RunState,
    pub tree_project: String,
}

/// Every row's tree, one walk for all, in the rows' order (the held ones first).  A run in `held` keeps the tree already
/// placed for it; the rest are placed over themselves, and a top one whose launcher is held joins that launcher's tree.
/// A launcher outside both leaves its run the top of its own tree; a launch loop is cut where it closes.  A tree's state
/// is `tally` of the run, every run below it and every run one of them asked (its own state alone).  `rows` names each
/// run once.
pub fn forest<'a>(rows: &'a [(String, RunRow)], held: &HashMap<String, RunTree>) -> Vec<(&'a str, RunTree)> {
    let loose: HashMap<&str, &RunRow> = rows.iter().filter(|(run, _)| !held.contains_key(run)).map(|(run, row)| (run.as_str(), row)).collect();
    let order: Vec<&str> = rows.iter().map(|(run, _)| run.as_str()).filter(|run| loose.contains_key(run)).collect();
    let up: HashMap<&str, &str> =
        order.iter().filter_map(|&run| loose[run].launcher.as_deref().filter(|launcher| loose.contains_key(launcher) && *launcher != run).map(|launcher| (run, launcher))).collect();
    // The asked runs' own states, by the run that asked them (any row's: a held run's state counts in a loose asker's tree).
    let mut asked: HashMap<&str, Vec<RunState>> = HashMap::new();
    for (run, row) in rows {
        for asker in &row.asked_by {
            if loose.contains_key(asker.as_str()) && asker != run && Some(asker) != row.launcher.as_ref() {
                asked.entry(asker.as_str()).or_default().push(row.state);
            }
        }
    }
    let mut below: HashMap<&str, Vec<&str>> = HashMap::new();
    for &run in &order {
        if let Some(&launcher) = up.get(run) {
            below.entry(launcher).or_default().push(run);
        }
    }
    let mut roots: HashMap<&str, &str> = HashMap::new();
    for &run in &order {
        let (mut path, mut walked, mut current) = (Vec::new(), HashSet::new(), run);
        while !roots.contains_key(current) && !walked.contains(current) {
            let Some(&above) = up.get(current) else { break };
            path.push(current);
            walked.insert(current);
            current = above;
        }
        let top = roots.get(current).copied().unwrap_or(current);
        roots.extend(path.into_iter().chain([current]).map(|step| (step, top)));
    }
    // Below before above, without recursion: a tree is as deep as its agents went.
    let mut rolled: HashMap<&str, RunState> = HashMap::new();
    for &run in &order {
        let (mut stack, mut pushed) = (vec![run], HashSet::from([run]));
        while let Some(&current) = stack.last() {
            let children = below.get(current).map(Vec::as_slice).unwrap_or_default();
            let pending: Vec<&str> = children.iter().copied().filter(|child| !rolled.contains_key(child) && !pushed.contains(child)).collect();
            if !pending.is_empty() {
                pushed.extend(pending.iter().copied());
                stack.extend(pending);
                continue;
            }
            stack.pop();
            if !rolled.contains_key(current) {
                let held_states = children.iter().filter_map(|child| rolled.get(child).copied());
                let state = tally([loose[current].state].into_iter().chain(asked.get(current).into_iter().flatten().copied()).chain(held_states));
                rolled.insert(current, state);
            }
        }
    }
    let kept = rows.iter().filter_map(|(run, _)| held.get(run).map(|tree| (run.as_str(), tree.clone())));
    let placed = order.iter().map(|&run| {
        let top = roots[run];
        let joined = held.get(loose[top].launcher.as_deref().unwrap_or(""));
        let tree = RunTree {
            launcher_run_id: loose[run].launcher.clone(),
            root_run_id: joined.map_or_else(|| top.to_owned(), |tree| tree.root_run_id.clone()),
            tree_state: rolled[run],
            tree_project: joined.map(|tree| tree.tree_project.clone()).unwrap_or_default(),
        };
        (run, tree)
    });
    kept.chain(placed).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(launcher: Option<&str>, state: RunState, asked_by: &[&str]) -> RunRow {
        RunRow { launcher: launcher.map(str::to_owned), state, asked_by: asked_by.iter().map(|&run| run.to_owned()).collect() }
    }

    #[test]
    fn a_tree_reads_as_its_strongest_state_and_a_loop_is_cut() {
        let rows = [
            ("top".to_owned(), row(None, RunState::Done, &[])),
            ("child".to_owned(), row(Some("top"), RunState::Failed, &[])),
            ("asked".to_owned(), row(None, RunState::Working, &["child"])),
            ("a".to_owned(), row(Some("b"), RunState::Done, &[])),
            ("b".to_owned(), row(Some("a"), RunState::Lost, &[])),
        ];
        let placed: HashMap<_, _> = forest(&rows, &HashMap::new()).into_iter().collect();
        assert_eq!((placed["top"].tree_state, placed["child"].root_run_id.as_str()), (RunState::Working, "top"));
        assert_eq!(placed["asked"].root_run_id, "asked");
        assert_eq!(placed["a"].root_run_id, placed["b"].root_run_id);
        assert_eq!(tally([]), RunState::Done);
        assert_eq!(state_of("never heard of"), RunState::Working);
    }
}
