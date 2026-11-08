# Changelog

## Unreleased

- `AgentRevision.relations` (`AgentRelation`: caller role or `*`, callers one / many, callees one / per_project / many,
  `max_live`, derived `kind` 1-1 / 1-N / N-1 / N-N) replaces `AgentRevision.instances`: who may call a role and how many
  of its runs answer, every relation of a role keeping the same `callees`; `AgentRelation.answering` picks the relation a
  call runs under (a run of no role reaches `*` only, the person is never refused). `AgentInstances` is `AgentCallees`.
- The package sits at the repository root (`galaius_core/`), no `src/` folder.
- The built-in MCP server's id is `galaius`; script approval digests and the sealed project-secrets key
  derive from `galaius.*` tags (saved script approvals are asked again once).

## 0.45.0 — 2026-10-07

- Renamed: the package is `galaius-core`, imported as `galaius_core`.

## 0.42.0 — 2026-09-13

- Shared trigger configuration snapshots and compare-before-save updates for schedule and webhook block editors.

## 0.41.0 — 2026-09-13

- Editable agent graphs with an explicit Assistant root and revision conflict checks.
- Authoritative prompt heads in verifiable agent catalogs; existing 0.40 catalogs remain readable.
- Resumable conversation summaries and revision-checked message requests.
- Direct HTTP, email and connector workflow tasks, including tool activity without a parent agent.
- Company profiles, establishment lookup, logo uploads and shared release information.

## 0.40.0 — 2026-09-13

- Standalone shared contracts and generated schemas for Galaius clients and services.
