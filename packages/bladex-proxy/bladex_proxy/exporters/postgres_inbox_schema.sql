-- BladeX PostgreSQL 收件表（Z4 / MQ-P33；随包发布）
-- 🔴 只在 export.toml 里该 postgres exporter 的 options.inbox = true 时应用——
--    默认关的功能不在用户库里留表。幂等：全部 IF NOT EXISTS。
--
-- 用法：外部程序只 INSERT；BladeX 轮询 status='pending' 的行，经 proxy 落成
-- Memory Hub 管理事件，然后**原地回写** status / result_reason / event_ref / processed_at。
-- BladeX 永不 DELETE 本表的行（审计留档）。
--
--   target_type  'fact'（'hard_rule' 等其它值一律 rejected）
--   op           'update'  new_value = 新文字
--                'assign'  new_value = 目标 matter_id（'' = 摘除归属）
--                'delete'  new_value 忽略
--   status       'pending' → 'accepted' | 'rejected'

CREATE TABLE IF NOT EXISTS bladex_edit_inbox (
    id             BIGSERIAL PRIMARY KEY,
    target_type    TEXT NOT NULL,
    target_id      TEXT NOT NULL DEFAULT '',
    op             TEXT NOT NULL,
    new_value      TEXT NOT NULL DEFAULT '',
    submitter      TEXT NOT NULL DEFAULT '',
    submitted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    status         TEXT NOT NULL DEFAULT 'pending',
    result_reason  TEXT NOT NULL DEFAULT '',
    event_ref      TEXT NOT NULL DEFAULT '',
    processed_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_edit_inbox_pending
    ON bladex_edit_inbox (id) WHERE status = 'pending';
