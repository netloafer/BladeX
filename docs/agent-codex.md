# Connecting Codex CLI to BladeX

Codex CLI speaks the OpenAI **Responses** API. BladeX serves it natively at
`/v1/responses`, alongside Chat Completions and Anthropic Messages.

> Verification status: the endpoint, streaming event sequence and Codex fingerprints are
> covered by automated tests; the full real-traffic acceptance run is part of the beta
> release checklist.

## Setup

Point the OpenAI base URL at BladeX and use your BladeX client key:

```bash
export OPENAI_BASE_URL=http://localhost:38080/v1
export OPENAI_API_KEY=<your BladeX client key>    # printed by `bladex init`
```

Then confirm:

```bash
bladex status --traffic     # expect a row with agent=codex
```

## What BladeX does with Codex traffic

| Behaviour | How BladeX handles it |
|---|---|
| Stateless per turn (full `input` each time) | Matches BladeX's model exactly — nothing is kept server-side between turns. |
| `previous_response_id` | Rejected with a clear 400. BladeX is stateless by design; silently ignoring it would produce wrong context. |
| `instructions` field | Mapped to the leading system message. |
| Flat tool definitions, `function_call` / `function_call_output` | Normalised to OpenAI chat shape so storage and rebuild stay format-independent. |
| `reasoning.effort` | Passed through; reasoning output is accumulated and stored. |
| Tool loops | Evidence from closed task units is degraded to placeholders before forwarding, keeping the prefix byte-stable so upstream prompt caching still hits. |

Requests are always stored in OpenAI chat format regardless of the inbound protocol, so
memory rebuilt from the ledger is identical across agents.

## Verifying memory actually works

```bash
bladex doctor --e2e
```

Cross-agent check: state a decision in Codex, wait one consolidator cycle, then ask
Claude Code or Hermes about it. Shared memory across agents is the product claim — this
is how you confirm it on your own machine.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| 400 mentioning `previous_response_id` | Disable stateful mode in your client; BladeX needs the full input each turn. |
| 404 on `/v1/responses` | Older BladeX build, or the base URL is missing `/v1`. |
| Turns appear as `agent=unknown` | The fingerprint did not match. Report it — the fix belongs in BladeX, not in your Codex config. |
| Route always picks the cheap model | An auxiliary-detection rule matched. Static routing in `routing.toml` (`[strategies.agent]`) takes precedence over auxiliary downgrading; see `route_static_over_aux` in the log. |

To fall back, restore `OPENAI_BASE_URL` to the provider's URL.
