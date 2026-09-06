# BladeX Engineering Conventions (contributor baseline)

> Purpose: keep contributors (human and AI) on one style so the codebase doesn't drift. **Architecture decisions live in ADRs; engineering details live here.** When they conflict, ADRs win on architecture, this doc wins on coding detail.

## 0. Iron rules (skim before coding)

1. Full type annotations; Pydantic v2 for domain objects; no bare dicts crossing modules with business data.
2. No `print()` debugging - `structlog` for BladeX processes, stdlib `logging` for library code.
3. Don't fork or vendor upstream adapters (LiteLLM via SDK; storage via embedded components).
   The model SDK is imported in exactly one file -- `bladex_proxy/router_sdk.py`, surfaced as **Router**.
4. Respect module boundaries and dependency direction (`bladex-core` is agent-neutral; no reverse deps).
5. Use the domain vocabulary - no synonyms.
6. New/changed code ships with a test; `pytest` green before handing off.
7. Commits explain "what changed" in plain language.

## 1. File & directory naming

| Object | Rule | Example |
|---|---|---|
| Package | kebab-case, `bladex-` prefix | `bladex-proxy`, `bladex-core` |
| Package python dir | snake_case (matches package) | `bladex_proxy/`, `bladex_core/` |
| Module (.py) | snake_case, noun | `identity.py`, `p1_redis.py` |
| Test file | `test_<module>.py` | `test_inject.py` |
| Config | `config/.env` (**not in git**) + `config/.env.example` (template) | - |

Package layout is consistent:

```
packages/<pkg>/
├── pyproject.toml
├── <pkg_snake>/            # source
│   ├── __init__.py         # explicit public exports
│   └── <module>.py
└── tests/                  # unit tests for this package
```

## 2. Naming

| Type | Rule | Example |
|---|---|---|
| Class | PascalCase | `Prefetcher`, `ProxyConfig`, `P1Redis` |
| Function / var | snake_case | `retrieve_facts`, `injected_text` |
| Constant / env | UPPER_SNAKE; env uses `BLADEX_` prefix | `MAX_PREFETCH_K`, `BLADEX_REDIS_URL` |
| Private | `_` prefix | `_dedup_facts` |
| Boolean | `is_/has_/enabled` | `auth_enabled`, `is_stream` |
| Enum | PascalCase class; OpenAI-compatible values lowercased | `class TurnStatus(str, Enum): OK="ok"` |
| structlog event | snake_case, past-tense/noun | `"turn_enqueued"`, `"auth_failed"` |

**Domain vocabulary (mandatory, no synonyms):**

| Term | Meaning |
|---|---|
| `Fact` | a distilled structured fact (P2 / derived) |
| `Prefetch` | per-turn recall + injection (the only injection source) |
| `Consolidation` | incremental distillation of facts from conversations |
| `HardRule` | `MUST`/`NEVER` rule (trust 1.0, injected every turn) |
| `P1 / P3 / P2` | intake buffer (Redis) / complete ledger (RocksDB) / derived (LanceDB + facts + MatterGraph) |
| `Matter` | a "thing" - a cross-session logical unit of work |
| `Tombstone` | append-only deletion marker; rebuild must not resurrect it |

## 3. Interfaces

- **Full type annotations** on all signatures including returns. Avoid `Any` unless genuinely dynamic (and comment why).
- **Pydantic v2** for domain objects (`Identity`, `Turn`, `ToolEvent`); `dataclass` OK for config. No bare dicts carrying business data across modules.
- **Nullable:** `X | None`, not `Optional`.
- **Stubs are marked** `# stub: will become <target>` with the signature frozen.
- **Dependency direction:** `bladex-core` (agent-neutral) ← `bladex-proxy` (depends on core). `bladex-core` must not import `bladex_proxy` or any agent code. Storage flows only `P1 -> P3 -> P2`.

### HTTP API (OpenAI-compatible)

- Endpoints: `POST /v1/chat/completions`, `POST /v1/messages` (Anthropic), `GET /health`, `GET /metrics`, `GET /v1/models`.
- Headers: `Authorization: Bearer <key>`, `X-Agent-ID`, `X-Session-ID` (identity hints; not required - BladeX fingerprints).
- Error body: `{"error": {"message": <str>, "type": <str>}}`. 401 auth failure; 4xx client; 5xx upstream (502, `type: "upstream_error"`).
- **Passthrough:** `tools` / `tool_choice` / `response_format` / `seed` etc. pass through the Router gateway untouched.
- **Streaming:** SSE, `data: {json}\n\n`, ends with `data: [DONE]\n\n`; chunks passed through verbatim (including `tool_calls`).

### Storage contracts

- **P3 key:** `user/agent/session/seq` (via `Identity.storage_key()`).
- **Serialization:** `msgpack` (`use_bin_type=True`).
- **P1 (Redis Streams):** XADD -> XREADGROUP -> **XACK only after store succeeds** (no loss, no dup); never `XADD MAXLEN` to trim unprocessed data.
- **P3 (RocksDB):** single source of truth, append-only; no hidden deletes (use tombstones).

## 4. Errors & logging

- Specific exception + structured log + safe fallback. No bare `except` / swallowing.
- Custom exceptions extend a `BladeXError` base (`RetrievalError`, `StorageError`, …) for layered handling.
- `structlog`, snake_case events, structured fields (`key=...`, `error=str(e)`). No `print()`.

## 5. Testing

- **Unit** (`tests/unit/`, `packages/*/tests/`): mock external deps (Redis / RocksDB / Router gateway).
- **Integration** (`tests/integration/`): real local deps, end-to-end.
- **Naming:** `test_<behavior>`, one assertion theme per test.
- **Discipline:** new features need tests; **fix a bug by first adding a test that reproduces it**. `pytest` green before handoff.
- New storage paths need a rebuild-equivalence check (P2 rebuilds from P3 identically).

## 6. Config & dependencies

- Config via env vars (`BLADEX_` prefix) + `config/.env`. Secrets never in git - only placeholders in `.env.example`.
- `uv` manages deps; `requires-python = ">=3.11"`. `uv sync --all-extras` installs workspace packages editable + dev tools.
- Don't fork upstream or vendor adapters.

## 7. Commits & collaboration

- Format: `<type>(<scope>): <subject>` where type ∈ `feat/fix/docs/test/refactor/chore`, scope ∈ `proxy/core/storage/config/…`. Subject imperative; body in plain language explains *why*.
- One concern per commit. Work on a branch or worktree.
- **Broadcast interface changes:** if you change a cross-module signature, HTTP contract, or storage key/serialization, call it out prominently in the PR.
- When implementation conflicts with an ADR or `CLAUDE.md`, stop and align - don't silently diverge.

## 8. Pre-commit checklist (Definition of Done)

- [ ] Full type annotations; no bare dicts for business data.
- [ ] No `print()`; structlog/stdlib logging with snake_case events.
- [ ] Tests added; `uv run pytest -q` green.
- [ ] Module boundaries and dependency direction respected.
- [ ] Naming follows §2; domain vocabulary used.
- [ ] Storage changes honor contracts (key format, msgpack, ack-after-store, no MAXLEN trim).
- [ ] `uv run ruff check .` clean.
- [ ] Commit explains the change in plain language; interface changes broadcast.

---

> This is the engineering baseline; it grows with the project. Changes to this doc follow the commit convention (`docs(convention): …`).
