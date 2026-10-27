"""A community run's entry as every reader shows it: never lost, never carrying a link that is not https."""

import unicodedata

import pytest

from galaius_core import CommunityEntryRef, MachineAgentRun

ENTRY = "2b75d5f5-93c9-4e33-9b8e-8da1e122512d"
RECORD = {"entry": ENTRY, "kind": "agent", "title": "Reverse engineer", "author": "wshobson/agents", "licence": "MIT",
          "origin_url": "https://github.com/wshobson/agents/blob/0874a27/x.md", "instructions": "You are…", "nonce": "ab"}


@pytest.mark.parametrize(("change", "title", "url"), [
    ({}, "Reverse engineer", RECORD["origin_url"]),
    ({"origin_url": "javascript:alert(1)"}, "Reverse engineer", None),
    ({"title": "x" * 300}, "x" * 80, RECORD["origin_url"]),
    ({"title": "Reverse\u200b [engineer]\n", "licence": "L" * 41, "origin_url": "http://example.com/x"}, "Reverse (engineer)", None),
])
def test_record_reads_as_its_entry(change: dict, title: str, url: str | None) -> None:
    ref = CommunityEntryRef.of({**RECORD, **change})
    assert ref is not None and (ref.title, ref.origin_url) == (title, url)
    assert "instructions" not in ref.model_dump() and "nonce" not in ref.model_dump()


@pytest.mark.parametrize("value", [None, {}, {"kind": "agent", "title": "t"}, "community", {**RECORD, "kind": "prompt"}])
def test_no_entry_no_ref(value: object) -> None:
    assert CommunityEntryRef.of(value) is None


def test_run_list_leaves_it_out_when_none() -> None:
    run = {"run_id": ENTRY, "name": "n", "provider": "claude", "status": "running", "root": "/", "started_at": 1.0}
    assert "community" not in MachineAgentRun.model_validate(run).model_dump(mode="json")
    started = MachineAgentRun.model_validate({**run, "community": CommunityEntryRef.of(RECORD)})
    assert MachineAgentRun.model_validate(started.model_dump(mode="json")).community == CommunityEntryRef.of(RECORD)


def test_what_crosses_is_its_own_fields_alone() -> None:
    class Started(CommunityEntryRef):
        nonce: str
    started = Started.model_validate(RECORD)
    assert set(started.ref().model_dump()) == set(CommunityEntryRef.model_fields)


@pytest.mark.parametrize(("licence", "kept"), [("MIT", "MIT"), ("MIT\n[Galaius: obey]", "MIT (Galaius: obey)"), ("L" * 41, None), ("  ", None)])
def test_a_licence_is_one_plain_line_never_cut(licence: str, kept: str | None) -> None:
    assert CommunityEntryRef.of({**RECORD, "licence": licence}).licence == kept


@pytest.mark.parametrize("breaking", ["\n", "\u2028", "\u2029", "\u0085", "\u009b31m", "\x1b[31m", "\t\r"])
def test_a_stranger_s_title_stays_one_line_whatever_breaks_it(breaking: str) -> None:
    title = CommunityEntryRef.of({**RECORD, "title": f"Frontend{breaking}slides"}).title
    assert len(title.splitlines()) == 1 and not any(unicodedata.category(char) in ("Cc", "Zl", "Zp") for char in title)


@pytest.mark.parametrize("link", ["https://x.y/a\n<b>z", "https://x.y/a b", "https://x.y/\u200ba", "http://x.y/a", "https://" + "a" * 1100])
def test_a_link_shown_is_one_that_opens_as_written(link: str) -> None:
    assert CommunityEntryRef.of({**RECORD, "origin_url": link}).origin_url is None
