//! The files a file tool's answer names.  A kept file reaches a reader only through its tool's own answer, never a
//! step's input: a refused call names nothing.

use std::collections::HashSet;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

/// How a file tool's answer that wrote nothing opens, its reason after it.
const REFUSED: &str = "No file written: ";
/// Characters of a refusal's reason kept.
const REASON: usize = 400;
/// The largest file kept, and the most source files one file's story names.
const SIZE_MAX: u64 = 2 * 1024 * 1024 * 1024;
const SOURCES_MAX: usize = 20;
/// A kept file's key: the start of its bytes' digest and its plain suffix.
macro_rules! key {
    () => {
        r"[0-9a-f]{20}(?:\.[a-z0-9]{1,16})?"
    };
}

macro_rules! patterns {
    ($($name:ident = $pattern:expr;)*) => {
        $(static $name: LazyLock<Regex> = LazyLock::new(|| Regex::new($pattern).expect(stringify!($name)));)*
    };
}

patterns! {
    KEY = concat!("^", key!(), "$");
    TOKEN = concat!(r"\[galaius-file (?P<key>", key!(), r") (?P<name>[^\]\n\r]{1,200})\]");
    STORY = concat!(r"\[galaius-story (?P<key>", key!(), r") maker=(?P<maker>[0-9A-Za-z-]{1,64}) at=(?P<at>\d+(?:\.\d+)?)(?: size=(?P<size>\d+))?",
        r"(?: from=(?P<sources>[0-9a-z.,]+))?(?: replaces=(?P<replaces>", key!(), r"))?\]");
}

/// What a kept file's key carries: who made it, from which kept files and in place of which, when, and its size.
#[derive(Clone, Debug, PartialEq)]
pub struct Story {
    pub maker: String,
    pub at: f64,
    pub replaces: Option<String>,
    pub size: Option<u64>,
    pub sources: Vec<String>,
}

/// The (key, name) of each file the answer names, in order, each once.
pub fn created_files(text: &str) -> Vec<(&str, &str)> {
    let mut seen = HashSet::new();
    TOKEN.captures_iter(text).map(|found| found.extract::<2>().1).map(|[key, name]| (key, name)).filter(|named| seen.insert(*named)).collect()
}

/// The story each file token of the answer carries, by key, the first that holds up; a file with none has no entry.
pub fn file_stories(text: &str) -> Vec<(&str, Story)> {
    let mut seen = HashSet::new();
    STORY
        .captures_iter(text)
        .filter_map(|found| {
            let size = match found.name("size") {
                Some(size) => Some(size.as_str().parse::<u64>().ok().filter(|&size| size <= SIZE_MAX)?),
                None => None,
            };
            let sources: Vec<String> = found.name("sources").map_or_else(Vec::new, |sources| sources.as_str().split(',').map(str::to_owned).collect());
            if sources.len() > SOURCES_MAX || !sources.iter().all(|source| KEY.is_match(source)) {
                return None;
            }
            let at = found["at"].parse().ok()?;
            let story = Story { maker: found["maker"].to_owned(), at, replaces: found.name("replaces").map(|key| key.as_str().to_owned()), size, sources };
            let key = found.name("key")?.as_str();
            seen.insert(key).then_some((key, story))
        })
        .collect()
}

/// A tool result's words as its stream records them: a structured result (`{"result": "…"}`) unwrapped.
pub fn tool_answer(text: &str) -> String {
    match serde_json::from_str::<Value>(text) {
        Ok(Value::Object(mut fields)) => match fields.remove("result") {
            Some(Value::String(result)) => result,
            _ => text.to_owned(),
        },
        _ => text.to_owned(),
    }
}

/// Why a file tool's answer wrote nothing, None for an answer that wrote a file.
pub fn refused_file(text: &str) -> Option<String> {
    let answer = tool_answer(text);
    let space = |character: char| character.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&character);
    answer.strip_prefix(REFUSED).map(|reason| reason.trim_matches(space).chars().take(REASON).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_named_file_is_listed_once_and_a_refusal_says_why() {
        let key = "0123456789abcdef0123.docx";
        let said = format!("[galaius-file {key} a.docx] [galaius-file {key} a.docx] [galaius-story {key} maker=run-1 at=2.5 size=9 from={key}]");
        assert_eq!(created_files(&said), [(key, "a.docx")]);
        assert_eq!(file_stories(&said)[0].1.size, Some(9));
        assert!(file_stories(&format!("[galaius-story {key} maker=r at=1 from=bad]")).is_empty());
        assert_eq!(refused_file(r#"{"result":"No file written:  too large "}"#).as_deref(), Some("too large"));
        assert_eq!(refused_file("written"), None);
    }
}
