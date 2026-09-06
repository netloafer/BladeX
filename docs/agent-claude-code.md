# Connecting Claude Code to BladeX

Claude Code speaks the Anthropic Messages API. BladeX serves it natively at `/v1/messages`,
so connecting is two environment variables — no plugin, no config file, no code change.

> Verification status: the endpoint and its tool-call conversion are covered by automated
> tests; the full real-traffic acceptance run is part of the beta release checklist.
> Report anything that looks off.

## Setup

```bash
export ANTHROPIC_BASE_URL=http://localhost:38080
export ANTHROPIC_API_KEY=<your BladeX client key>   # printed by `bladex init`
```

Note the base URL has **no `/v1` suffix** here — Claude Code appends the path itself.
(The OpenAI-format agents in the other guides do include `/v1`.)

Start a session and ask it something. Then:

```bash
bladex status --traffic
```

You should see a row with `agent=claude-code`. If you see nothing, the traffic never
reached BladeX — check the base URL and that `bladex status` reports the proxy running.

## What BladeX does with Claude Code traffic

| Behaviour | How BladeX handles it |
|---|---|
| No `X-Agent-ID` header | Passive fingerprinting from the system prompt and tool signatures identifies it as `claude-code`. You are never asked to add headers. |
| `tool_result` blocks inside user messages | Converted to standalone `role:tool` messages before forwarding — OpenAI-compatible upstreams reject them otherwise (this was a real 502 source). |
| Very long user messages (transcripts, file dumps) | Agent-internal envelopes are stripped before distillation only. **The forwarded request body is never modified.** |
| Sub-agent calls with no system prompt | Session stickiness ties them to the parent task so they do not fragment your memory. |
| `count_tokens` requests | Served; without it Claude Code sees a 404 every turn. |

## Verifying memory actually works

```bash
bladex doctor --e2e     # forward -> store -> distil -> recall, end to end
```

Cross-agent check (the point of the product): mention a durable fact in another agent
(Hermes, Codex), wait for the consolidator cycle (60s by default), then ask Claude Code
about it. The recalled memory shows up in `bladex status --traffic` as a non-zero
injected item count.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Claude Code errors immediately | Wrong base URL shape — do **not** append `/v1`. |
| 401 from BladeX | `ANTHROPIC_API_KEY` must be your BladeX **client** key, not the upstream provider key. |
| Traffic shows up but nothing is ever recalled | Consolidator not running (`bladex consolidator status`) or upstream distillation key missing (`bladex config check`). |
| Turns show `stored=skipped` | Redis is down; `bladex status` prints the degradation banner. Turns spill to disk and replay when Redis returns. |
| Responses are slower than direct | Check the banner for `inject timeout`; raise `BLADEX_HOTPATH_BUDGET_MS`. |

To fall back at any moment, unset `ANTHROPIC_BASE_URL`. Claude Code goes straight to
Anthropic again and nothing in `data/` is lost.
