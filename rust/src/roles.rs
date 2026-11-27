//! What a run's role and its gauge say.

/// The role of the web's own assistant: its runs are in no list of runs, trees, conversations or counts.
const UI_ROLES: [&str; 1] = ["blob"];

/// Whether a run of `role` is listed and counted with the team's runs.
pub fn listed_role(role: Option<&str>) -> bool {
    role.is_none_or(|role| !UI_ROLES.contains(&role))
}

/// The fields of a conversation gauge, each with the least value it takes.
pub const GAUGE: [(&str, i64); 5] = [("context_tokens", 0), ("context_window", 1), ("compactions", 0), ("thinking_tokens", 0), ("thinking_live", 0)];

/// The gauge fields `value` holds, each kept only when it is a whole number its field takes: anything else reads as not
/// reported, never as zero.
pub fn reported<'a>(value: impl Fn(&str) -> Option<i64> + 'a) -> impl Iterator<Item = (&'static str, i64)> + 'a {
    GAUGE.into_iter().filter_map(move |(field, least)| value(field).filter(|&figure| figure >= least).map(|figure| (field, figure)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_assistant_is_unlisted_and_an_empty_window_unreported() {
        assert!(listed_role(None) && listed_role(Some("tester")) && !listed_role(Some("blob")));
        let held = reported(|field| match field {
            "context_tokens" => Some(0),
            "context_window" => Some(0),
            "compactions" => Some(-1),
            _ => None,
        });
        assert_eq!(held.collect::<Vec<_>>(), [("context_tokens", 0)]);
    }
}
