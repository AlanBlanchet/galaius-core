//! Token counts in one additive shape: `input` the prompt tokens neither read from nor written to a cache,
//! `cache_read` / `cache_write` the cached rest of the prompt, `output` the reply, reasoning included (`reasoning` the
//! part of it a vendor names apart).  A figure nothing reported is None, never zero.

use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct TokenCounts {
    pub input: Option<i64>,
    pub output: Option<i64>,
    pub cache_read: Option<i64>,
    pub cache_write: Option<i64>,
    pub reasoning: Option<i64>,
}

impl TokenCounts {
    /// A count whose source reports the WHOLE prompt (cache reads and writes inside it): its uncached part carved out,
    /// never counted twice.
    pub fn of_prompt(prompt: Option<i64>, output: Option<i64>, cache_read: Option<i64>, cache_write: Option<i64>) -> Self {
        let cached = cache_read.unwrap_or(0).saturating_add(cache_write.unwrap_or(0));
        Self { input: prompt.map(|prompt| prompt.saturating_sub(cached).max(0)), output, cache_read, cache_write, reasoning: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_whole_prompt_is_never_counted_twice() {
        assert_eq!(TokenCounts::of_prompt(Some(10), Some(2), Some(7), Some(5)).input, Some(0));
        assert_eq!(TokenCounts::of_prompt(Some(10), None, Some(4), None).input, Some(6));
        assert_eq!(TokenCounts::of_prompt(None, Some(2), Some(4), None).input, None);
    }
}
