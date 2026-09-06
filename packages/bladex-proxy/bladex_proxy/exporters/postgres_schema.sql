-- BladeX PostgreSQL connector schema v1（Beta T15/B3.2；随包发布）
-- 幂等：全部 IF NOT EXISTS；版本记录在 bladex_schema_version。

CREATE TABLE IF NOT EXISTS bladex_schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS facts (
    id                TEXT PRIMARY KEY,
    content           TEXT NOT NULL,
    kind              TEXT NOT NULL DEFAULT 'general',
    item_kind         TEXT NOT NULL DEFAULT 'assertion',
    subject           TEXT NOT NULL DEFAULT '',
    attribute         TEXT NOT NULL DEFAULT '',
    source_user_id    TEXT NOT NULL DEFAULT '',
    scope             TEXT NOT NULL DEFAULT '',
    exposure_ceiling  TEXT NOT NULL DEFAULT 'public',
    importance        DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    created_at        TIMESTAMPTZ,
    t_invalid         TIMESTAMPTZ,
    payload           JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS matters (
    matter_id   TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    summary     TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'active',
    origin      TEXT NOT NULL DEFAULT 'auto',
    scope       TEXT NOT NULL DEFAULT '',
    updated_at  TIMESTAMPTZ,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS matter_edges (
    edge_id     TEXT PRIMARY KEY,
    matter_id   TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_key  TEXT NOT NULL,
    relation    TEXT NOT NULL DEFAULT 'belongs',
    provenance  TEXT NOT NULL DEFAULT 'auto',
    weight      DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS hard_rules (
    content TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS sync_state (
    exporter_name TEXT PRIMARY KEY,
    cursor_value  TEXT NOT NULL DEFAULT '',
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_facts_user   ON facts (source_user_id);
CREATE INDEX IF NOT EXISTS idx_edges_matter ON matter_edges (matter_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON matter_edges (target_key);

INSERT INTO bladex_schema_version (version) VALUES (1)
ON CONFLICT (version) DO NOTHING;
