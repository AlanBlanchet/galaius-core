# galaius-rules

The rules every reader of galaius data applies, in Rust (`src/`), with a Python module (`galaius_rules`, built by
maturin: `uv build --wheel`; `cargo test` runs the Rust checks).

| Rule | Rust | Same rule in `galaius_core` |
|---|---|---|
| A run's state, a tree's state, where each run sits | `run_state` | `workflows.run_state` (`state_of`, `tally`, `forest`) |
| Roles listed with a person's runs; a conversation gauge as reported | `roles` | `listed_role`, `ContextGauge.reported` |
| Token counts from a whole-prompt figure | `tokens` | `TokenCounts.of_prompt` |
| The files a file tool's answer names, their stories, a refusal | `files` | `created_files`, `file_stories`, `tool_answer`, `refused_file` |

Installed clients still run the Python definitions: a client release carries no native wheel for them yet. The Python
definition of each rule is deleted in the release that ships these wheels; until then a change to a rule is made in both.
