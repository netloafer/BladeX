<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/bladex-lockup-horizontal-dark.svg">
    <img src="assets/brand/bladex-lockup-horizontal.svg" alt="BladeX" width="340">
  </picture>
</p>

<p align="center">
  <a href="https://github.com/netloafer/BladeX/actions/workflows/ci.yml"><img src="https://github.com/netloafer/BladeX/actions/workflows/ci.yml/badge.svg" alt="CI status"></a>
</p>

> **Build your private data assets.**
> **Connect all your agents and LLMs like a blade.**

BladeX is a **self-hosted, memory-first proxy** that sits between your AI agents and the model
providers. Point any agent's API base URL at BladeX and it gains cross-session, **cross-agent**
memory plus a per-task ledger — with **no code changes to the agent**. Everything stays on your
machine: the memory is a private data asset you own, not a feature you rent.

中文版：[README.zh.md](README.zh.md)

## The 4S

| | What it means |
|---|---|
| **Save all Memory** | Every turn lands in the Memory Hub **losslessly** — full text and metadata, the single source of truth. Nothing is silently truncated, and nothing is deleted to make room: when capacity is tight, trust decays and facts are archived. |
| **Saving your Token** | The injected surface is no longer a memory dump. Each turn carries hard rules, a stable self-description, and the task ledger; everything else the model **pulls on demand** through `bladex_memory_search`. A ledger that survives compaction also cuts the re-explaining rounds, which is where the real spend is. |
| **Smart LLM Router** | One endpoint, many models. Configuration-driven deterministic routing (explicit request model → team → principal → agent), optional LLM judging, session stickiness, health-based failover. **Static configuration always beats runtime inference.** |
| **Sift out experience** | Raw turns are not memory. A background distiller turns them into structured facts, groups them into Matters (a "thing" you work on across sessions and agents), and the model records what it has actually established in the ledger's `Verified` section. |

## What problem does it solve?

Agents forget across sessions, and their built-in memory has four recurring failures:

1. **Token bloat** — dump all memory every turn, pay more and get slower as context grows.
2. **Memory erosion** — context compaction drops older facts: *"why don't you remember last week?"*
3. **Contradictions** — conflicting information from different sessions, with no single source of truth.
4. **Forced forgetting** — when context fills up, old facts are deleted outright.

And one more that only shows up once you use several agents: **nothing carries across them.** What
you established in one coding agent is invisible to the next. That handoff is the thing BladeX is
built for, and the reason the memory lives in a proxy rather than inside any one agent.

## How it works: ledger + memory + agency

### 1. Ledger — keeping the model's attention on the task

Long tasks drift. Context gets compacted, the session restarts, you move from one agent to another —
and every time, the model loses the thread and you re-explain it. Run several agents against several
models and that tax compounds: each one needs the goal, the constraints, and what is already settled,
and each one wants it in its own configuration file, its own format, its own conventions. So the same
task ends up described four times, in four standards, drifting apart from the moment you write them.

BladeX keeps **one task state, in one format, in the proxy**. Every agent and every model behind it
reads the same ledger — no per-agent setup, no per-model convention, nothing to keep in sync:

| Section | Holds |
|---|---|
| **Goal** | What this task is for. **Only the user can change it** — see the red lines below. |
| **Core** | Constraints and givens that hold for the whole task. |
| **Verified** | What has actually been established, ideally backed by a tool or test result. |
| **Open** | Known unknowns. |
| **Next** | The agreed next step. |

What that buys you:

- **Attention lands on the open part.** The five sections are short by construction and the block
  rides at the very end of the message array — where a model in a long tool loop actually reads. It
  arrives already knowing what this task is and what is settled, so its attention goes to what isn't.
- **Focus without loss.** The ledger is not a summary of the transcript — summaries are lossy and
  drift. It is a structured record the model writes *as it establishes things*, so compaction can
  drop the conversation without dropping the state. Concentration and fidelity are not traded here.
- **Correctness, then cost.** `Verified` is for what was actually established, ideally backed by a
  tool or test result — which is what stops a long task from quietly building on a guess. And the
  savings come from the rounds you never spend re-establishing context, which are worth far more
  than any bytes trimmed off an injection.
- **Handoff is free.** One ledger per task, one active ledger bound per session, both independent of
  which agent is driving. Finish a stretch in one agent, pick it up in another, and the state is
  already there.
- **Resuming is a real option.** When the model considers switching tasks, BladeX shows it both
  *what you worked on recently* and *ledgers that look like the same task* — so returning to a task
  from three weeks ago is something the model can choose, instead of silently starting a fifth
  ledger for a thing you already have four of.

### 2. Memory — four layers, one direction

Writes flow **Pipeline → Hub → Index → Flash**, always in that direction; each layer downstream of
the Hub can be rebuilt from it.

| Layer | Backend | Role |
|---|---|---|
| **Pipeline** | Redis (AOF) | Incoming buffer. Acked only after the turn is stored; disk spill if Redis is down, so turns are never lost. |
| **Memory Hub** | RocksDB | Full journal, **single source of truth**, long-term. Everything else can be rebuilt from it. |
| **Memory Index** | LanceDB + FTS | Derived: distilled facts, embeddings, the Matter graph. Rebuildable from the Hub. |
| **Memory Flash** | Files | Your ledgers as editable Markdown (edit one in your own editor and BladeX adopts the change), plus a behavior profile (`behavior.json`) you can view on the dashboard. Projections of the Hub, never a second source of truth. |

### 3. Agency — BladeX is visible to the model

Purely passive vector recall has a ceiling: the party making the decisions doesn't know there is
anything to look up. So BladeX introduces itself (`config/system/AGENT.md`, `TOOLS.md`, `SKILLS.md`)
and offers a small tool face the model can call:

- `bladex_memory_search` — pull past facts on demand (the only path to memory; nothing is force-fed)
- `bladex_ledger_read` / `bladex_ledger_update` / `bladex_ledger_switch` — read, record, and move
  between task ledgers

Calls to these tools are intercepted by BladeX and never reach your upstream bill as agent work: a
turn containing only `bladex_*` calls is served in an internal loop; a mixed turn is split, handled,
and spliced back. You can see every one of them in `bladex status --traffic`.

### The three red lines

1. **BladeX tools only read and write memory and ledger objects.** They cannot run code, read your
   files, or reach the network. The tool face is stationery, not a secretary.
2. **Only the user can change the Goal.** A model that can rewrite its own objective undermines
   the entire point of having one.
3. **If the model never calls the tools, behaviour is identical.** Remove the tool face and the
   system falls back to the passive form. This is also the measurement channel: the value of the
   tool face can be A/B-measured directly.

## How is it different from other options?

| | BladeX | Built-in agent memory | mem0 |
|---|---|---|---|
| **Zero agent rewrites** | ✅ change one base URL | — | requires agent-side integration |
| **Works across different agents** | ✅ one memory behind all of them | ❌ per-agent silo | depends on integration |
| **Task state, not just facts** | ✅ five-section ledger | ❌ | ❌ |
| **You own all data** | ✅ 100% on your infrastructure | depends on the agent | cloud by default |

## Quick Start (v0.3.0)

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Redis server (auto-started by `bladex start` if installed but not running)

No system RocksDB is needed — storage uses the prebuilt `rocksdict` wheel.

### Install & Run

```bash
git clone https://github.com/YOUR-ORG/bladex.git
cd bladex
uv sync

# Put the `bladex` command on your PATH. `uv sync` installs it into .venv but does
# not activate anything, so a fresh shell will say "command not found" without this.
source .venv/bin/activate          # Windows: .venv\Scripts\activate
# Prefer not to activate? Prefix every command with `uv run`, e.g. `uv run bladex init`.

# Interactive setup: writes config/.env (chmod 600) + config/routing.toml,
# generates a client key and an admin key, and enables auth by default.
bladex init

# Fill in your upstream API key when prompted (or edit config/.env afterwards).

bladex doctor          # environment health check
bladex start           # Redis + consolidator + proxy
bladex doctor --e2e    # end-to-end: is memory actually being produced and recalled?
```

`bladex start` prints the dashboard URL and, if anything in the memory pipeline is missing, a
`degraded` line naming the consequence (for example *"Redis unreachable -> turns are NOT stored, no
memory will be produced"*). A green `ready` line means the loop is live; `bladex doctor --e2e`
proves it by sending a probe turn and recalling it. For scripts: `bladex start` exits `0` when ready,
`1` when it failed, and `3` when it started degraded.

> **First start downloads the embedding model** (~70 MB, from Hugging Face) before the proxy answers
> `/ready`. `bladex start` recognises the download and waits for it, printing
> `downloading embedding model … first start only` plus progress, instead of giving up at the 30 s
> readiness window; the window only restarts once the download finishes. If the download fails, or
> makes no progress for 2 minutes, `start` reports `failed -- the embedding model download failed`
> with the reason (and stops the proxy it just started). The second start is instant.

### Configure your agent

In your agent's settings, change the **API base URL** from the default (e.g.
`https://api.openai.com/v1`) to:

```
http://localhost:38080/v1
```

Use your **BladeX client key** as the "API key" (printed by `bladex init`, stored in `config/.env`).

**Supported agent formats natively**:

- OpenAI `/v1/chat/completions` — all OpenAI-compatible clients
- Anthropic `/v1/messages` — native Claude Code compatibility
- OpenAI Responses `/v1/responses` — native Codex CLI compatibility

Per-agent three-step setup:

| Agent | What to change | Verify |
|---|---|---|
| Hermes | `base_url` → `http://localhost:38080/v1`, API key → your BladeX client key | `bladex status --traffic` shows the turn with `agent=hermes:*` |
| Claude Code | `export ANTHROPIC_BASE_URL=http://localhost:38080` and `ANTHROPIC_API_KEY=<bladex client key>` | `bladex status --traffic` shows `agent=claude-code` |
| Codex CLI | point the OpenAI base URL at `http://localhost:38080/v1` (Responses API) | `bladex status --traffic` shows `agent=codex` |

Full per-agent guides, including known behaviours and troubleshooting:
[Hermes](docs/agent-hermes.md) · [Claude Code](docs/agent-claude-code.md) · [Codex CLI](docs/agent-codex.md)

Nothing needs to change on the agent side beyond that URL. BladeX adapts to how each agent
behaves — no custom headers, no system-prompt edits, no request-format changes.

### Is it working?

```bash
bladex status --traffic     # one line per recent turn: agent / injected / route / stored
bladex doctor --e2e         # full loop: forward -> store -> distil -> recall
bladex ledger list          # the task ledgers, newest first
bladex ledger show <id>     # one ledger's five sections
bladex ledger close <id>    # mark a finished ledger closed (undo: bladex ledger reopen <id>)
bladex memory search "..."  # what BladeX has actually learned
```

`bladex status` also shows a warning banner when memory is silently degrading — injection timeouts,
turns that were not stored, or a consolidator that has fallen behind.

### If BladeX breaks

BladeX sits in front of all your agent traffic, so it is worth knowing the escape hatch before you
need it. **Point the agent back at its original base URL.** That is the whole recovery procedure —
your agents keep working immediately, and nothing in `data/` is lost; BladeX will pick up again when
you restart it.

```bash
bladex stop      # stop the proxy and consolidator (Redis is left running)
bladex start     # bring it back
bladex consolidator restart   # restart only the distiller, without interrupting live sessions
```

### Extra LLM calls and cost

BladeX adds calls to your upstream account beyond what your agent sends. Budget for them:

| Call | When | How to control |
|---|---|---|
| Fact distillation | Background, per stored turn (in the consolidator, not the hot path) | Point `[distill].model` in `routing.toml` at a cheap model |
| Conclusion / progress distillation | Background, on turns that produce a result | Same as above |
| Tool-face inner loop | When the model calls a `bladex_*` tool and BladeX has to continue the turn | Turn the tool face off (`BLADEX_MODULE_TOOLFACE=0`) |
| LLM routing judge | Only when `[strategies.filter]` routing is enabled | Leave routing off, or restrict the candidate pool |

Distillation runs off the request path, so it does not slow your agent down — but it does show up on
your bill. Internal agent calls detected as auxiliary are routed to the cheap tier automatically.
Injection and retrieval themselves make **no** LLM calls.

**On the "Saving your Token" claim**: we publish mechanisms, not a percentage. The measured figure so
far (~87% less) is **memory-injection accounting only** — end-to-end request tokens are still being
baselined, and BladeX will not quote an end-to-end saving until that number exists.

### Small context windows

If you run a local model with a small context window (32K and below), tell BladeX the window:
add `context_window = <tokens>` (the `num_ctx` from `ollama show <model>`) to that model's
`[[models]]` entry in `config/routing.toml`. With it set, dynamic routing skips a model that cannot
hold the request plus an output reserve (`BLADEX_CONTEXT_OUTPUT_FLOOR_TOKENS`, default 2048), and
BladeX trims **its own** injected surface to the remaining budget — switch candidates first, then
the ledger block, then the ledger tool family, then the system notes; hard rules are never trimmed,
and your agent's messages, tools and `max_tokens` are never touched. A model pinned by a static
strategy is never swapped: you get a `route_static_context_window_exceeded` warning pointing at
`routing.toml` instead. Without `context_window` nothing changes. `bladex doctor` lists local
models that have no window configured; `bladex status --traffic` and the `upstream_output_truncated`
warning still tell you when output is being cut.

### Non-English users

The default embedding model is `BAAI/bge-small-en-v1.5` (English-first, 0.07 GB). If your
conversations are mostly in another language, pin a multilingual model **before** you build up
memory — switching later requires a full re-embed:

```toml
# config/routing.toml
[embedding]
model = "intfloat/multilingual-e5-large"   # 2.24 GB, the model BladeX's thresholds are calibrated on
```

### No local embedding model? Use an embedding API

The embedding model runs locally by default. If your machine cannot host one (small VPS, no
spare RAM), point BladeX at a hosted embedding API instead — the setting already exists:

```toml
# config/routing.toml
[embedding]
backend = "api"
model = "openai/text-embedding-3-small"   # Router format: provider/model
api_base = ""                              # optional; provider default when empty
api_key_env = "OPENAI_API_KEY"             # name of the env var holding the key; the key itself never goes in this file
```

Two things to know before you switch:

- **Privacy**: in `api` mode *every* distilled fact and *every* query is sent to the embedding
  provider. BladeX logs `EMBED_API_PRIVACY` at startup to make this visible. Keep `local` if your
  memory must stay on-device.
- **Changing backend or model changes the vector space.** Do it before you build up memory, or
  stop both processes and run `python scripts/reembed_index.py` first — otherwise the Memory Index
  refuses to start on a model mismatch.

The `api` backend does not use the optional embedding module (`BLADEX_MODULE_EMBED`); that module
only exists to share a *local* model between processes.

## Where is my data stored?

All data is stored locally on your machine, under `data/`:

- **Pipeline**: Redis AOF persistence
- **Memory Hub**: RocksDB — full conversation journal, single source of truth
- **Memory Index**: LanceDB — embeddings, distilled facts, Matter graph (rebuildable from the Hub)
- **Memory Flash**: plain files in `data/bladex_flash` — one Markdown file per task ledger
  (`<principal>/personal/ledgers/…/<id>.md`; edit it and BladeX adopts your version) and
  `behavior.json`, a per-agent / per-project activity profile. Point `BLADEX_FLASH_PATH` somewhere
  handier if you want to edit ledgers alongside your own files. (Before 0.3.0 Flash also wrote a
  `User → Agent → Project → Session` directory tree; that tree is no longer written — see the
  release notes.)

BladeX **never sends your memory to third-party servers**. The only outgoing requests are to the LLM
API providers you configured, for model inference. Encryption at rest is delegated to your disk
(FileVault / LUKS); the application layer does not add its own.

## Features

### Core (v0.3.0)

- 🧾 **Five-section task ledger** — Goal / Core / Verified / Open / Next; survives compaction,
  restarts, and cross-agent handoffs. Goal is user-owned. Finished ledgers can be **closed or
  superseded** — by the agent (`bladex_ledger_close`) or by you (`bladex ledger close | reopen |
  supersede --by <id>`); a closed ledger drops out of the "recently worked on" list and stops
  absorbing new turns. When what you say matches its content (within a day of closing, and not superseded), it shows
  up again in the matching-ledgers list marked as closed — from any session or agent — and
  switching back to it reopens it. `bladex ledger show` follows the supersede chain.
- 🧠 **Memory quality as the product** — relevance over volume, incremental extraction before any
  compaction, trust decay and archiving instead of deletion.
- 🛠️ **Tool face** — `bladex_memory_search` plus four ledger tools, with a strict boundary
  (memory and ledger objects only) and zero behavioural difference when unused.
- 🔌 **Zero integration** — any agent speaking OpenAI Chat Completions, Anthropic Messages, or
  OpenAI Responses.
- 🚦 **Router** — configuration-driven deterministic routing, optional LLM judge, session
  stickiness, health-based failover. Static config outranks every runtime inference.
- 🔒 **Privacy-first** — fully self-hosted; you control everything.
- 👥 **Multi-agent, multi-user** — multiple API keys, optional team/identity segregation
  (enterprise self-hosted).
- 🖥️ **Web dashboard** at `http://localhost:38080/dashboard` — status, fact search, Matter graph,
  ledgers, session replay, hard rules.
- 🔌 **MCP server** (`bladex-mcp`) — a second read/write surface for clients that speak MCP.
- 📤 **Export/import** — versioned JSONL snapshots, plus continuous **two-way** sync with Obsidian
  and PostgreSQL. Edits made there flow back into the Memory Hub as journaled admin events, so they
  survive a full rebuild and later automatic distillation never reverts them:
  - **Obsidian** — edit a fact line (correction), move it to another Matter page (re-assignment;
    into `Inbox.md` = unassign), or delete it (forget). On by default; `backflow = false` turns it
    off. Edits to `Hard Rules.md` are journaled too and take effect within seconds (hard rules are
    hot-reloaded, see below).
  - **PostgreSQL** — the exported tables stay a read-only mirror; to change something, `INSERT` a
    request into the append-only `bladex_edit_inbox` table and BladeX writes back `accepted` /
    `rejected` with a reason. **Off by default** (`inbox = true` in `config/export.toml`), and it
    **never accepts hard-rule changes** — whoever can write that database would otherwise be able
    to inject instructions into every agent.
  - Mass deletions are refused, not applied: a whole page disappearing, or more fact lines vanishing
    in one scan than `BLADEX_BACKFLOW_MAX_DELETES` (default 5), only logs a warning.

### Delivered in 0.3.0

Everything the 0.2.0 README listed under *Coming next*:

- Context-window-aware routing and budget-aware trimming of the injected surface (see *Small
  context windows* above; `[[models]].context_window`, `BLADEX_CHARS_PER_TOKEN`,
  `BLADEX_CONTEXT_OUTPUT_FLOOR_TOKENS`). The stable part of the injected surface (system notes and
  tool schemas) is now frozen at the first turn of a session and re-anchored by content afterwards,
  so its position no longer drifts when the agent appends system reminders.
- `close` / `supersede` primitives for finished ledgers: the model gets a `bladex_ledger_close`
  tool (optionally naming the ledger that supersedes it), and you get `bladex ledger close | reopen |
  supersede`. A closed ledger leaves the "recently worked on" list; when what you say matches
  its content, it reappears among the matching ledgers marked as closed, and switching back to it
  reopens it.
- Bidirectional sync with external tools: edits you make in the Obsidian vault (a fact's line,
  moving it to another Matter page, deleting it, `Hard Rules.md`) flow back into memory as
  management events (`backflow = true` by default in `export.toml`; set `false` to keep the old
  one-way export). PostgreSQL gets an opt-in append-only inbox table (`inbox = true`).
- Hot-reloadable hard rules: change `BLADEX_HARD_RULES` in `config/.env` (or `Hard Rules.md`) and
  every process picks it up within `BLADEX_HARD_RULES_RELOAD_S` (default 5 s), no restart.
- `bladex start` waits for the first-start embedding model download instead of reporting a false
  `/ready` timeout.

Also in 0.3.0:

- `bladex start` returns exit code 3 when it starts degraded (proxy up, part of the memory
  pipeline missing) instead of 0; 1 still means it failed.
- `bladex status` shows a queue trend line, and `/metrics` exposes `bladex_backlog_*` series; a
  `backlog_rising` warning is logged when the backlog keeps climbing.
- BladeX refuses to start a second Redis on the same `data/` directory, and `bladex repair-aof`
  repairs a Redis append-only manifest that points at missing files (dry run by default).
- Long admin commands such as `bladex memory tombstone-project` run in the background on the
  proxy and no longer stall it.
- The idle consolidator round is about 5× cheaper.

### Coming next (0.4.0)

- When your agent compacts a long conversation on its own, BladeX keeps it as the same session:
  the ledger stays bound and the turn count carries on.
- Hand a task to another agent with `bladex ledger handoff`, a one-page portable handoff of the
  ledger; you can also create or edit a ledger yourself with `bladex ledger create | edit`.
- Contradicting facts are flagged for you to settle; BladeX never deletes on its own judgment.
- Reasoning parameters pass through `/v1/responses` unchanged; no thinking text leaks into Codex
  output.
- Full rebuilds can be interrupted and resumed, survive upstream outages, and never leave a
  half-written index.

## Known Limitations (v0.3.0)

- Tuned for **individual self-hosting** — enterprise multi-tenant features are still in development.
- The ledger tool face is enabled by default only for `medium` and `strong` model tiers
  (`BLADEX_LEDGER_TIERS`), or when the session already owns an active ledger. Weak-tier models still
  see the ledger; they just are not asked to maintain it.
- On small-context local models the injected surface competes with the output budget unless you
  declare the model's `context_window` (see *Small context windows* above); BladeX does not
  auto-detect windows, and it does not change the agent's `max_tokens`.
- LanceDB ANN indexing is built once a table passes 256 rows; below that, search is brute-force
  (fast at that size, but tiny stores do skip the index).
- Only OpenAI Chat Completions, Anthropic Messages, and OpenAI Responses endpoints are supported
  natively — more formats will be added based on demand.
- **Exported memory can be stale in the target**: continuous sync to Obsidian/PostgreSQL pushes new
  and updated facts, but a Matter card's `summary`/`open_issues` are not filtered by sensitivity
  level the way individual facts are. Review before pointing sync at a shared location.
- The dashboard shows hard rules read-only. Edit them in `config/.env` (`BLADEX_HARD_RULES`,
  `||`-separated) or in Obsidian's `Hard Rules.md`; **no restart needed** — the proxy, consolidator
  and Flash daemon each pick the change up within `BLADEX_HARD_RULES_RELOAD_S` (default 5 s) and log
  `hard_rules_reloaded`. If both places were edited, the later edit wins. Clearing the line does not
  clear your rules (the previous set is kept and a warning is logged).
- The CLI reads `config/`, `data/` and `logs/` relative to the current directory (or `BLADEX_HOME`).
  Running `bladex status` from an unrelated directory shows an empty store — your data is not gone,
  you are just pointed somewhere else.
- **No contradiction detection yet.** Two conflicting facts can coexist in memory without a flag.
- **Model routing does not use your memory.** Routing is configuration-driven (plus an optional
  LLM judge); memory-informed routing is not implemented.
- **When an agent compacts its own history** (e.g. Hermes or DeepSeek Harness on long sessions),
  BladeX currently treats the next request as a new session: the turn count restarts and the
  ledger is no longer bound automatically. The ledger itself is kept — the model is offered it as
  a switch candidate.
- **Codex with a reasoning model upstream**: reasoning parameters on `/v1/responses` are not
  forwarded yet, so thinking text can leak into the output and `apply_patch` may fail.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Memory quality design informed by analysis of [Hermes](https://hermes-agent.nousresearch.com/)
  memory architecture
- Multi-model routing uses [LiteLLM](https://www.litellm.ai/) SDK for provider compatibility
- Built with FastAPI, Redis, RocksDB, and LanceDB
