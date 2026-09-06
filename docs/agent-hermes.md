# Connecting Hermes to BladeX

Hermes speaks the OpenAI Chat Completions API, which is BladeX's primary inbound format.
This is the agent BladeX has the most real traffic hours with.

## Setup

In your Hermes provider configuration, set:

```
base_url = http://localhost:38080/v1
api_key  = <your BladeX client key>      # printed by `bladex init`
```

Then send a message and check:

```bash
bladex status --traffic     # expect rows with agent=hermes:default (or hermes:<profile>)
```

## What BladeX does with Hermes traffic

| Behaviour | How BladeX handles it |
|---|---|
| No `X-Agent-ID` header | Passive fingerprinting (system prompt + tool signatures) identifies Hermes, and the profile suffix (`hermes:default`, `hermes:accept`) is read from the agent's own declaration. |
| Internal maintenance calls (checkpoints, compaction prompts, memory-save sub-tasks) | Detected as auxiliary: skipped for distillation so they do not pollute memory. Cheap-tier routing applies only to the calls where it is actually correct. |
| Mixture-of-agents reference calls | Recognised so the same turn is not stored twice. |
| Hermes' own context compression | BladeX extracts facts incrementally on every turn, so nothing depends on surviving compression. |

Hermes keeps its built-in memory features; BladeX adds a layer in front and does not
modify the agent. If both are active you will see some duplication in the model's context —
that is expected in beta.

## Verifying memory actually works

```bash
bladex doctor --e2e         # full loop probe
bladex status --traffic     # per-turn: injected hard rules + items, route source, stored
```

A healthy row shows a non-zero injected item count once the consolidator has processed a
few turns. Right after a fresh install the store is empty, so early turns legitimately
inject hard rules only.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Everything works but no memory is ever recalled | Consolidator not running — `bladex consolidator status`. It is a separate process by design. |
| `agent=unknown` rows | An internal sub-request without a system prompt; session stickiness normally attaches it to the parent. Persistent cases are a BladeX-side fix. |
| Injection is empty on every turn | Check the degradation banner in `bladex status` for injection timeouts or an unavailable P2 layer. |
| Distillation produces almost nothing | Check `[distill].model` in `routing.toml` and that its `api_key_env` resolves (`bladex config check`). |

To fall back, point Hermes' `base_url` back at your provider.
