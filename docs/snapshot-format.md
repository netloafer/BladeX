# BladeX Snapshot Format (schema 1)

`bladex export` / `bladex import` use a line-delimited JSON (JSONL) snapshot.
Each line is one JSON object with a `type` and `schema` field. The first line
is always `meta`.

## Record types

| type | contents | notes |
|---|---|---|
| `meta` | `schema`, `exported_at`, `embed_model_id`, `include_turns` | first line; importers must reject `schema` greater than what they support |
| `hard_rule` | `content` | configuration-plane; **not** written to storage on import — merge manually into `BLADEX_HARD_RULES` |
| `fact` | full Fact fields (id, content, kind, item_kind, subject, attribute, audience, matter_id, temporal fields, importance, scope, exposure_ceiling, …) | embeddings are **not** exported; the importer re-embeds with its local model |
| `matter` | full Matter fields (matter_id, title, summary, status, origin, participants, lifecycle, open_issues, …) | centroid/embedding excluded (recomputed locally) |
| `edge` | full MatterEdge fields (edge_id, matter_id, target_type, target_key, relation, provenance, …) | |
| `admin_event` | AdminEvent fields (event_type, matter_id, target_key, ts, payload) | replayed into the local P3 journal on import (fingerprint-deduplicated) |
| `turn` | P3 Turn records with `key` | only with `bladex export --full`; **not** imported (offline archive only) |

## Semantics

- **Tombstoned data never appears** in a snapshot: P2 cascade deletion has
  already removed it, and P3 scans skip tombstone-covered entries
  (ADR-0012 §3.6).
- **Import is idempotent by id**: an existing fact/matter/edge with the same id
  is skipped. Re-importing the same snapshot is a no-op.
- **Manual beats auto**: a local manual matter/edge is never overwritten by a
  snapshot auto entry; a snapshot manual entry may replace a local auto one.
- **Conflicts are non-blocking**: a fact with the same id but different content
  keeps the local version; the conflict is counted and logged
  (`snapshot_import_fact_conflict`).
- **Rebuild equivalence**: every imported entity is also journaled into P3 as a
  `fact_import` / `matter_import` / `edge_import` admin event. A later
  `bladex storage rebuild` (full P2 rebuild) replays the journal, so imported
  memories survive rebuilds even though they have no local conversation
  history behind them.
- Embeddings are always recomputed by the importing instance — snapshots are
  portable across embedding models (the P2 vector-space invariant is never
  violated by an import).

## Operational notes

- `bladex export` is read-only (P3 secondary + P2 read-only handles); the proxy
  and consolidator can keep running.
- `bladex import` requires the proxy and consolidator to be stopped (it takes
  the P2/P3 write locks). Run `bladex stop` first.
