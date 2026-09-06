# BladeX - AI Collaboration Guide

This is the authoritative reference for any AI coding tool (Claude Code, Cursor, etc.) working in this repo. Read it before writing code.

## What BladeX is

BladeX is a **memory-first LLM proxy**. An agent points its `base_url` at BladeX; BladeX injects the user's accumulated memory, routes to an upstream model, captures the full reply, and distills the conversation into durable facts - all without the agent changing a line of code.

Three roles, in priority order:

1. **Memory = the only moat** (and the basis for routing quality).
2. **Proxy = the main integration path** (OpenAI `/v1/chat/completions` + Anthropic `/v1/messages`).
3. **Routing = an entrance, not a product** (no moat; borrow mature tooling, don't sell it).

## Architecture (essentials)

- **Proxy** (FastAPI): identity → memory injection → routing → forward (via the Router gateway) → capture full reply → async store.
- **Three storage tiers, one direction (`P1 → P3 → P2`):**
  - `P1` Redis Streams (AOF, ack-on-store) - intake buffer.
  - `P3` RocksDB - the single source of truth, append-only, every turn.
  - `P2` LanceDB + facts + MatterGraph - derived; rebuilt from P3 by a background consolidator.
- **Hot path budget ≤ 50 ms p99** (identity + retrieval + injection); degrade to hard-rules-only on timeout.
- **Memory is extracted incrementally**, before the agent's context compression can lose it. Hard rules (`MUST`/`NEVER`) inject every turn at full trust.

## Borrow vs. build (discipline)

1. **Borrow mature components, don't fork them:** multi-model access via the **LiteLLM SDK** (don't vendor its adapters); storage via **Redis / RocksDB / LanceDB** (don't build storage engines).
2. **Build the moat yourself:** memory orchestration - prefetch (relevant injection), consolidation (incremental distillation), consistency (contradiction detection), recall (cross-session). This is the only code worth polishing.
3. **Adapt to agents, never the reverse.** BladeX is a service agents consume. If an agent behaves oddly (no `X-Agent-ID`, sub-tasks without a system prompt, unusual request shapes), fix it on the BladeX side (passive fingerprinting, sticky association, identity inference). **Do not require the agent to add headers, change its system prompt, or alter its request format.** This is a rigid principle.

## Code organization

```
packages/
  bladex-core/    memory orchestration (the moat) - agent-neutral
  bladex-proxy/   the proxy: FastAPI + LiteLLM, tiered storage, routing
  bladex-mcp/     (soon) MCP server for active model recall
config/           .env / routing.toml templates + run scripts
deploy/           Dockerfile + compose (single-machine self-host)
docs/             deployment guide, contributor conventions
```

## Python conventions

- **Python 3.11+.** Type annotations on all signatures (including returns).
- **Pydantic v2** for domain objects. No bare dicts carrying business data.
- **Call LLMs through the Router gateway** (`bladex_proxy/router_sdk.py`: `acompletion` / `completion` /
  `embedding`) - never hand-build provider HTTP, and never `import litellm` anywhere else. The gateway
  wraps the LiteLLM SDK and is the single place that owns global switches, log capture and error wording;
  user-facing output says **Router**, not the vendor name.
- **Logging by process:** BladeX's own processes (proxy server, background workers) use `structlog`; library code loaded into another process uses stdlib `logging`. **No `print()` for debugging.**
- **Exceptions:** specific types + structured log + safe fallback. Never bare `except: pass`.

```python
# Right
try:
    facts = prefetch.retrieve(query)
except RetrievalError as e:
    logger.warning("prefetch_fallback", error=str(e), query_len=len(query))
    facts = prefetch.hard_rules_only()
```

## Naming & domain vocabulary

| Type | Rule | Example |
|------|------|---------|
| Class | PascalCase | `Prefetcher`, `ProxyCore` |
| Function / var | snake_case | `retrieve_facts`, `consolidate` |
| Constant | UPPER_SNAKE | `MAX_PREFETCH_K` |
| Package | kebab-case | `bladex-core`, `bladex-proxy` |
| Module | snake_case | `consolidation.py` |

| Term | Meaning |
|------|---------|
| `Fact` | a distilled structured fact (stored in P2 / derived) |
| `Prefetch` | per-turn recall + injection (the only injection source) |
| `Consolidation` | incremental distillation of facts from conversations |
| `HardRule` | `MUST`/`NEVER` rule (trust 1.0, injected every turn) |
| `P1 / P3 / P2` | intake buffer (Redis) / complete ledger (RocksDB) / derived (LanceDB + facts + MatterGraph) |
| `Matter` | a "thing" - a cross-session logical unit of work |
| `Tombstone` | append-only deletion marker; P2 rebuild must not resurrect tombstoned data |

## Testing

- **Unit** (`tests/unit/`, `packages/*/tests/`): mock external deps (Redis / RocksDB / Router gateway).
  Patch `bladex_proxy.route.router_sdk`, not the vendor SDK.
- **Integration** (`tests/integration/`): real proxy + Redis + RocksDB, real multi-agent flow.
- New behavior ships with a test. New storage paths ship with a rebuild-equivalence check (P2 must rebuild from P3).
- Run: `uv run pytest -q`. Integration tests need Redis on `127.0.0.1:6379`.

## Before committing

- `uv run ruff check .` clean (`ruff check --fix .` for safe fixes).
- `uv run pytest -q` passes.
- Every change should be explainable to a non-technical user - say what changed and why.

## What's out of scope (beta)

Managed SaaS / multi-user hosting; HA/clustering; a separate admin key (admin endpoints share the client-key system); at-rest encryption of the data directory (operator's responsibility).

For deployment, security, and the data-flow statement, see `docs/deployment-selfhost.md` and `SECURITY.md`.
