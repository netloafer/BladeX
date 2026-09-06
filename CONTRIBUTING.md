# Contributing to BladeX

Thanks for considering a contribution. BladeX is a memory-first LLM proxy; the codebase is small and the boundaries are deliberate. This guide gets you productive fast.

## Setup

```bash
git clone … && cd bladex
uv sync --all-extras          # installs workspace packages (editable) + dev tools
uv run ruff check .           # lint
uv run pytest -q              # tests (integration tests need Redis on 127.0.0.1:6379)
```

If Redis isn't running, the Redis-dependent integration tests skip automatically; the rest still run. To run them: `brew install redis && redis-server` (or `apt install redis-server`).

The e5 embedding model (~2 GB) downloads on the first test that exercises real embeddings; subsequent runs use the cache at `data/fastembed_cache/`.

## What to work on

- **The moat lives in `bladex-core`** (prefetch, consolidation, consistency) and the proxy's storage tiering in `bladex-proxy`. These are the places where careful work pays off.
- **Borrow, don't fork.** Multi-model access goes through the LiteLLM SDK; storage uses Redis / RocksDB / LanceDB off the shelf. Don't vendor adapters or build storage engines - that's not where the value is.
- **Adapt to agents, never the reverse.** BladeX is a service agents consume. If an agent behaves oddly (no `X-Agent-ID`, sub-tasks without a system prompt, unusual request shapes), fix it on the BladeX side (fingerprinting, sticky association, inference). Do not require the agent to change.

## Before you open a PR

- `uv run ruff check .` is clean (auto-fix what you can: `ruff check --fix .`).
- `uv run pytest -q` passes.
- New behavior comes with a test. New storage paths come with a rebuild-equivalence check (P2 must rebuild from P3).
- Don't introduce `print()` for debugging - use `structlog` (proxy/worker processes) or stdlib `logging` (library code loaded into other processes).
- Pydantic v2 for domain objects; no bare dicts carrying business data.

## Architecture decisions (ADRs)

From v0.1.0, new architecture decisions are recorded as ADRs (Architecture Decision Records), numbered from 0001. If your change is architecturally significant, write one: context, decision, consequences. Small changes don't need one.

## Commit & PR style

- Commits: imperative subject, body explains *why*. Keep history readable.
- PRs: describe what changed and why; link any related issue; call out anything that touches the storage pipeline or injection logic (these are load-bearing).

## Releasing

Releases are tagged `v*` and published via the `release.yml` workflow (PyPI trusted publishing + a GitHub Release draft). The release manager curates the draft notes (bilingual, with the data-flow statement) before publishing.

## Questions

Open a discussion or issue. Be kind.
