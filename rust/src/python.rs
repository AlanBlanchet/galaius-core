//! The Python module: each rule over plain values (words for states, tuples for rows).

use std::collections::HashMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt};

use crate::run_state::{RunRow, RunState, RunTree};
use crate::{files, roles, run_state, tokens};

/// `(launcher, state, asked_by)`.
type Row = (Option<String>, String, Vec<String>);
/// `(launcher_run_id, root_run_id, tree_state, tree_project)`.
type Tree = (Option<String>, String, String, String);
/// `(maker, at, replaces, size, sources)`.
type Told = (String, f64, Option<String>, Option<u64>, Vec<String>);

fn state(word: &str) -> PyResult<RunState> {
    RunState::parse(word).ok_or_else(|| PyValueError::new_err(format!("no run state {word:?}")))
}

#[pyfunction]
fn state_of(word: &str) -> &'static str {
    run_state::state_of(word).word()
}

#[pyfunction]
fn tally(states: Vec<String>) -> PyResult<&'static str> {
    Ok(run_state::tally(states.iter().map(|word| state(word)).collect::<PyResult<Vec<_>>>()?).word())
}

#[pyfunction]
fn forest(rows: Vec<(String, Row)>, held: Vec<(String, Tree)>) -> PyResult<Vec<(String, Tree)>> {
    let rows = rows.into_iter().map(|(run, (launcher, word, asked_by))| Ok((run, RunRow { launcher, state: state(&word)?, asked_by }))).collect::<PyResult<Vec<_>>>()?;
    let held = held
        .into_iter()
        .map(|(run, (launcher_run_id, root_run_id, word, tree_project))| Ok((run, RunTree { launcher_run_id, root_run_id, tree_state: state(&word)?, tree_project })))
        .collect::<PyResult<HashMap<_, _>>>()?;
    let placed = run_state::forest(&rows, &held);
    Ok(placed.into_iter().map(|(run, tree)| (run.to_owned(), (tree.launcher_run_id, tree.root_run_id, tree.tree_state.word().to_owned(), tree.tree_project))).collect())
}

#[pyfunction]
#[pyo3(signature = (role))]
fn listed_role(role: Option<&str>) -> bool {
    roles::listed_role(role)
}

/// The gauge fields `values` holds: whole numbers only (no bool, no float) each field takes.
#[pyfunction]
fn reported<'py>(py: Python<'py>, values: &Bound<'py, PyDict>) -> PyResult<Bound<'py, PyDict>> {
    let whole = |field: &str| {
        let found = values.get_item(field).ok().flatten()?;
        (found.is_instance_of::<PyInt>() && !found.is_instance_of::<PyBool>()).then(|| found.extract().ok()).flatten()
    };
    let held = PyDict::new(py);
    for (field, figure) in roles::reported(whole) {
        held.set_item(field, figure)?;
    }
    Ok(held)
}

/// `(input, output, cache_read, cache_write)` of a whole-prompt figure.
#[pyfunction]
#[pyo3(signature = (prompt, output, cache_read = None, cache_write = None))]
fn of_prompt(prompt: Option<i64>, output: Option<i64>, cache_read: Option<i64>, cache_write: Option<i64>) -> (Option<i64>, Option<i64>, Option<i64>, Option<i64>) {
    let counts = tokens::TokenCounts::of_prompt(prompt, output, cache_read, cache_write);
    (counts.input, counts.output, counts.cache_read, counts.cache_write)
}

#[pyfunction]
fn created_files(text: &str) -> Vec<(String, String)> {
    files::created_files(text).into_iter().map(|(key, name)| (key.to_owned(), name.to_owned())).collect()
}

#[pyfunction]
fn file_stories(text: &str) -> Vec<(String, Told)> {
    files::file_stories(text).into_iter().map(|(key, story)| (key.to_owned(), (story.maker, story.at, story.replaces, story.size, story.sources))).collect()
}

#[pyfunction]
fn tool_answer(text: &str) -> String {
    files::tool_answer(text)
}

#[pyfunction]
fn refused_file(text: &str) -> Option<String> {
    files::refused_file(text)
}

#[pymodule]
fn galaius_rules(module: &Bound<'_, PyModule>) -> PyResult<()> {
    macro_rules! functions {
        ($($name:ident),*) => { $(module.add_function(wrap_pyfunction!($name, module)?)?;)* };
    }
    functions!(state_of, tally, forest, listed_role, reported, of_prompt, created_files, file_stories, tool_answer, refused_file);
    Ok(())
}
