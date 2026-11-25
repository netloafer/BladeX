# BladeX 0.4.0

0.4.0 delivers everything the 0.3.0 README listed under *Coming next*. It also announces a breaking change
that arrives in 0.5.0.

## Delivered in 0.4.0

- **Compaction keeps the session.** When your agent compacts a long conversation on its own, BladeX keeps
  it as the same session: the ledger stays bound and the turn count carries on.
- **Handoff and import.** `bladex ledger handoff` produces a one-page portable handoff of a ledger, and you
  can create or edit a ledger yourself with `bladex ledger create | edit`. The dashboard has a handoff page too.
- **Contradiction flagging (experimental, off by default).** Contradicting facts can be flagged for you to
  settle; BladeX never deletes on its own judgment. Turn it on with `BLADEX_CONTRADICTION_ENABLED=1`.
- **Codex over `/v1/responses`.** Reasoning parameters pass through unchanged, and no thinking text leaks
  into Codex output.
- **Interruptible rebuilds.** Full rebuilds can be interrupted and resumed, survive upstream outages, and
  never leave a half-written index.

## Heads-up: 0.5.0 is a breaking change

0.5.0 is a breaking change for Matters, fact retrieval, and the existing Obsidian export. Memory will be
rebuilt around the ledger, and `bladex_memory_search` will search ledgers and task summaries instead of
fragment-level facts. Do not build workflows on top of Matter pages or on fragment-level fact search.

## Known limitations that matter for 0.4.0

- **Contradiction flagging is experimental.** In our tests only about one flag in four was a real
  contradiction (a small sample), so expect mostly false alarms. It only flags; nothing is deleted or changed
  for you. It makes extra model calls while on, capped by default at 400,000 input tokens a day
  (`BLADEX_CONTRADICTION_DAILY_INPUT_TOKENS`). Only new memories are checked. What it operates on will change
  in 0.5.0.
- **Codex `apply_patch` needs a catalog declaration.** Codex's model catalog must declare
  `apply_patch_tool_type` for Codex to call `apply_patch` through BladeX; without it Codex edits files
  through the shell instead.
- **Handoff packs are not filtered for personal information**, and an imported Goal is text the receiving
  model reads as an instruction. Read a pack before importing it, and do not import one from a source you do
  not trust.
- **No evidence yet that a handoff saves turns.** BladeX makes no claim about how many turns a handoff saves.
- **Sub-task and sub-agent traffic has no live samples.** Ledger behavior for sub-tasks and post-compaction
  long tasks of some agents, Claude Code sub-agents, and Codex collaboration sub-tasks has not been observed
  on real traffic.
- **Edits to a ledger's `.md` file take effect on the next agent request.**

See *Known Limitations* in the README for the full list.
