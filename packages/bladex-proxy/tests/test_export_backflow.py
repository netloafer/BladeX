"""Z4（MQ-P33）外部工具回流验收：Obsidian vault 直编 + Postgres 收件表 ⇒ Hub 管理事件。

判据（prompt-batch-Z4 §5）在本文件的落点：
  ① Obsidian 改一行 ⇒ Hub 多一条 FACT_CORRECT ⇒ 重建后搜得到新文字、搜不到旧文字
     —— `test_e2e_obsidian_correct_survives_full_rebuild`；阴性：回流关 ⇒ 零事件
  ② Postgres 收件更正 ⇒ 同①，收件行回写 accepted；硬规则 ⇒ rejected、Hub 零事件
     —— `test_e2e_postgres_inbox_*`（fake 连接；真 PG = NO-DATA，见结算稿）
  ③ full rebuild 后更正仍在 —— 同①（重放分支拆掉即红：结算稿反向变异自证）
  ④ manual 压过 auto：full rebuild 把来源轮次再蒸一遍，内容不回退 —— 同①
  ⑤ 单写者静态守卫 —— `test_single_writer_static_guard`
  ⑥ 自写回声 0 事件；大面积消失 0 墓碑 —— `test_obsidian_*`
  ⑦ 换归属只动点名的那一条 —— `test_e2e_obsidian_move_only_touches_named_fact`
"""

from __future__ import annotations

import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.distillation import DistillFact, DistillOutput, MatterProposal
from bladex_core.exporters import ExportBatch
from bladex_core.fact import Fact
from bladex_core.matter import Matter, MatterOrigin, MatterStatus
from bladex_proxy.export_backflow import (
    AdminBackflowClient,
    BackflowEdit,
    BackflowResult,
)
from bladex_proxy.export_sync import ExporterConfig, ExportSyncWorker
from bladex_proxy.exporters.obsidian import ObsidianExporter, parse_hard_rules_page
from bladex_proxy.exporters.postgres import PostgresExporter, _load_schema_sql
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    SessionIdSource,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

_PKG = Path(__file__).resolve().parents[1] / "bladex_proxy"


def _fact(fid: str, content: str, matter_id: str = "") -> Fact:
    return Fact(id=fid, content=content, source_user_id="u1",
                matter_id=matter_id, created_at=datetime.now(UTC))


def _matter(mid: str, title: str) -> Matter:
    return Matter(matter_id=mid, title=title, status=MatterStatus.ACTIVE,
                  origin=MatterOrigin.AUTO)


def _vault(tmp_path: Path, **opts) -> tuple[ObsidianExporter, Path]:
    root = tmp_path / "vault" / "BladeX"
    return ObsidianExporter("v", "private", {"vault_dir": str(root), **opts}), root


def _seed_vault(exp: ObsidianExporter) -> None:
    exp.sync_incremental(ExportBatch(
        facts=[_fact("f1", "用户偏好中文", "m1"), _fact("f2", "部署在 arm64", "m1"),
               _fact("f3", "无归属事实")],
        matters=[_matter("m1", "发布"), _matter("m2", "迁移")],
        hard_rules=["MUST be polite", "NEVER use emojis"]), "")


def _drop_line(path: Path, needle: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    (hit,) = [ln for ln in lines if needle in ln]
    path.write_text("\n".join(ln for ln in lines if ln is not hit), encoding="utf-8")
    return hit


def _edit_page(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new), encoding="utf-8")


# ═══ Obsidian：快照对账（纯文件系统）════════════════════════════════


def test_obsidian_self_echo_zero_edits(tmp_path):
    """⑥ 阴性对照：连续两次正常导出、用户什么都没改 ⇒ 0 编辑。"""
    exp, _ = _vault(tmp_path)
    _seed_vault(exp)
    assert exp.collect_edits(5) == []
    exp.sync_incremental(ExportBatch(facts=[_fact("f1", "用户偏好中文（新版）", "m1")]), "c")
    assert exp.collect_edits(5) == []
    # 新实例（进程重启）读回同一份快照，照样 0
    exp2 = ObsidianExporter("v", "private", {"vault_dir": str(exp._root)})
    assert exp2.collect_edits(5) == []


def test_obsidian_line_edit_becomes_correct_and_ack_stops_echo(tmp_path):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    _edit_page(root / "Matters/m1.md", "- 用户偏好中文 [[", "- 用户偏好简体中文 [[")
    edits = exp.collect_edits(5)
    assert [(e.kind, e.fact_id, e.content, e.origin) for e in edits] == [
        ("correct", "f1", "用户偏好简体中文", "obsidian")]
    exp.ack(edits[0], BackflowResult("accepted"))
    assert exp.collect_edits(5) == []
    # 导出器随后把 Index 里的更正版写回（行被规范化）——仍是回声
    exp.sync_incremental(ExportBatch(facts=[_fact("f1", "用户偏好简体中文", "m1")]), "c")
    assert exp.collect_edits(5) == []


def test_obsidian_whitespace_only_change_is_not_an_edit(tmp_path):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    _edit_page(root / "Matters/m1.md", "- 用户偏好中文 [[", "-   用户偏好中文   [[")
    assert exp.collect_edits(5) == []


def test_obsidian_move_line_between_pages(tmp_path):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    line = _drop_line(root / "Matters/m1.md", "bladex-fact:f2")
    exp._append_fact_line(line, "m2")  # 等同用户把行粘进 m2 页
    edits = exp.collect_edits(5)
    assert [(e.kind, e.fact_id, e.matter_id, e.from_matter_id) for e in edits] == [
        ("assign", "f2", "m2", "m1")]
    exp.ack(edits[0], BackflowResult("accepted"))
    assert exp.collect_edits(5) == []


def test_obsidian_move_to_inbox_is_detach(tmp_path):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    line = _drop_line(root / "Matters/m1.md", "bladex-fact:f1")
    inbox = root / "Inbox.md"
    inbox.write_text(inbox.read_text().rstrip("\n") + "\n" + line + "\n")
    edits = exp.collect_edits(5)
    assert [(e.kind, e.fact_id, e.from_matter_id) for e in edits] == [("detach", "f1", "m1")]


def test_obsidian_single_line_delete_is_tombstone_edit(tmp_path):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    m1 = root / "Matters/m1.md"
    m1.write_text("\n".join(ln for ln in m1.read_text().splitlines()
                            if "bladex-fact:f2" not in ln))
    edits = exp.collect_edits(5)
    assert [(e.kind, e.fact_id) for e in edits] == [("delete", "f2")]


def test_obsidian_moving_matters_dir_away_deletes_nothing(tmp_path, capsys):
    """§3 阴性对照：把 Matters/ 整个挪走 ⇒ 0 删除 + 一行告警点名数量。"""
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    shutil.move(str(root / "Matters"), str(tmp_path / "elsewhere"))
    edits = exp.collect_edits(5)
    assert [e for e in edits if e.kind == "delete"] == []
    out = capsys.readouterr().out
    assert "obsidian_backflow_pages_missing_no_delete" in out and "facts=2" in out


def test_obsidian_mass_vanish_over_threshold_deletes_nothing(tmp_path, capsys):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    for p in (root / "Matters/m1.md", root / "Inbox.md"):
        p.write_text("\n".join(ln for ln in p.read_text().splitlines()
                               if "bladex-fact:" not in ln))
    assert [e for e in exp.collect_edits(2) if e.kind == "delete"] == []
    out = capsys.readouterr().out
    assert "obsidian_backflow_mass_delete_refused" in out
    assert "vanished=3" in out and "threshold=2" in out
    # 同一现场、阈值放宽 ⇒ 三条都删（证明上面的 0 是闸拦的，不是没识别出来）
    assert sorted(e.fact_id for e in exp.collect_edits(3) if e.kind == "delete") == [
        "f1", "f2", "f3"]


def test_obsidian_broken_marker_or_frontmatter_is_ignored(tmp_path, capsys):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    m1 = root / "Matters/m1.md"
    # 删 f1 的定位标记（wikilink 还在）+ 改 f2 文字
    _edit_page(m1, " <!--bladex-fact:f1-->", "")
    assert [e.fact_id for e in exp.collect_edits(5)] == []  # f1 认不出 ⇒ 不删不改
    assert "obsidian_backflow_marker_missing" in capsys.readouterr().out
    # 页 frontmatter id 被改 ⇒ 整页不回流（连 f2 的改动也不认）
    _edit_page(m1, "- 部署在 arm64 [[", "- 部署在 x86 [[")
    _edit_page(m1, "bladex_id: m1", "bladex_id: m9")
    assert exp.collect_edits(5) == []
    assert "obsidian_backflow_page_id_broken" in capsys.readouterr().out


def test_obsidian_hard_rules_page_edit(tmp_path):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    hr = root / "Hard Rules.md"
    _edit_page(hr, "- NEVER use emojis", "- NEVER use emojis\n- MUST cite sources")
    edits = exp.collect_edits(5)
    assert [(e.kind, e.rules, e.previous) for e in edits] == [(
        "hard_rules", ["MUST be polite", "NEVER use emojis", "MUST cite sources"],
        ["MUST be polite", "NEVER use emojis"])]
    exp.ack(edits[0], BackflowResult("accepted"))
    assert exp.collect_edits(5) == []
    # 清空 ⇒ 不回流（清成零条是最坏的失败方式）
    hr.write_text("---\nbladex_id: hard-rules\n---\n\n# Hard Rules\n\n")
    assert exp.collect_edits(5) == []


def test_parse_hard_rules_page():
    assert parse_hard_rules_page(
        "---\nbladex_id: hard-rules\n---\n\n# Hard Rules\n\n- A\n* B\n  - C\n-\ntext\n"
    ) == ["A", "B", "C"]


def test_obsidian_old_vault_without_snapshot_is_baselined(tmp_path):
    """升级前导出的 vault 没有快照 ⇒ 现状收作基线、零事件（认不出哪些是用户改的）。"""
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    (root / ".bladex/sync_snapshot.json").unlink()
    _edit_page(root / "Matters/m1.md", "- 用户偏好中文 [[", "- 升级前就改过 [[")
    exp2 = ObsidianExporter("v", "private", {"vault_dir": str(root)})
    assert exp2.collect_edits(5) == []
    assert exp2.collect_edits(5) == []
    _edit_page(root / "Matters/m1.md", "- 升级前就改过 [[", "- 基线之后再改 [[")
    assert [e.content for e in exp2.collect_edits(5)] == ["基线之后再改"]


def test_obsidian_backflow_option_off(tmp_path):
    exp, _ = _vault(tmp_path, backflow=False)
    assert exp.backflow_enabled is False


# ═══ Postgres 收件表（fake 连接）════════════════════════════════════


class _PgState:
    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.sql: list[str] = []

    def insert(self, **kw) -> int:
        rid = len(self.rows) + 1
        self.rows.append({"id": rid, "status": "pending", "result_reason": "",
                          "event_ref": "", "submitter": "", "new_value": "", **kw})
        return rid


class _PgCursor:
    def __init__(self, st: _PgState) -> None:
        self.st = st
        self._res: list = []

    def execute(self, sql: str, params: tuple = ()) -> None:
        self.st.sql.append(sql)
        if sql.startswith("SELECT COALESCE(MAX(id), 0)"):
            self._one = (max((r["id"] for r in self.st.rows), default=0),)
        elif sql.startswith("SELECT id, target_type"):
            self._res = [(r["id"], r["target_type"], r["target_id"], r["op"],
                          r["new_value"], r["submitter"])
                         for r in self.st.rows if r["status"] == "pending"][:params[0]]
        elif sql.startswith("UPDATE bladex_edit_inbox"):
            status, reason, ref, rid = params
            for r in self.st.rows:
                if r["id"] == rid and r["status"] == "pending":
                    r.update(status=status, result_reason=reason, event_ref=ref)

    def fetchall(self) -> list:
        return self._res

    def fetchone(self) -> tuple:
        return self._one


class _PgConn:
    def __init__(self, st: _PgState) -> None:
        self.st = st

    def cursor(self) -> _PgCursor:
        return _PgCursor(self.st)

    def __enter__(self):
        return self

    def __exit__(self, *a) -> None:
        return None

    def close(self) -> None:
        return None


def _pg(st: _PgState, **opts) -> PostgresExporter:
    return PostgresExporter("pg", "private", {"_connect_factory": lambda: _PgConn(st), **opts})


def test_pg_inbox_default_off_creates_nothing():
    st = _PgState()
    exp = _pg(st)
    assert exp.inbox_enabled is False
    assert exp.collect_edits(5) == []
    assert st.sql == []  # 默认关：不建表、不轮询
    exp.sync_incremental(ExportBatch(facts=[_fact("f1", "x")]), "")
    assert not any("bladex_edit_inbox" in s for s in st.sql)  # 导出路径一字不变


def test_pg_inbox_only_true_enables():
    assert _pg(_PgState(), inbox="yes").inbox_enabled is False
    assert _pg(_PgState(), inbox=True).inbox_enabled is True


def test_pg_inbox_rows_to_edits_and_hard_rule_rejected():
    st = _PgState()
    st.insert(target_type="fact", target_id="f1", op="update", new_value="新文字",
              submitter="alice")
    st.insert(target_type="hard_rule", target_id="", op="update", new_value="MUST obey")
    st.insert(target_type="fact", target_id="f2", op="assign", new_value="m2")
    st.insert(target_type="fact", target_id="f3", op="assign", new_value="")
    st.insert(target_type="fact", target_id="f4", op="drop")
    exp = _pg(st, inbox=True)
    edits = exp.collect_edits(5)
    assert [(e.kind, e.fact_id) for e in edits] == [
        ("correct", "f1"), ("reject", ""), ("assign", "f2"), ("detach", "f3"), ("reject", "")]
    assert edits[0].submitter == "postgres:alice" and edits[0].content == "新文字"
    assert "hard rules cannot be changed" in edits[1].reason
    assert any("CREATE TABLE IF NOT EXISTS bladex_edit_inbox" in s for s in st.sql)


def test_pg_inbox_mass_delete_rejected_and_ack_never_deletes():
    st = _PgState()
    for i in range(4):
        st.insert(target_type="fact", target_id=f"f{i}", op="delete")
    exp = _pg(st, inbox=True)
    edits = exp.collect_edits(3)
    assert {e.kind for e in edits} == {"reject"}
    assert all("mass-delete guard" in e.reason for e in edits)
    for e in edits:
        exp.ack(e, BackflowResult("rejected", reason=e.reason))
    assert [r["status"] for r in st.rows] == ["rejected"] * 4
    assert len(st.rows) == 4
    assert not any(s.lstrip().upper().startswith("DELETE") for s in st.sql)


def test_pg_inbox_retry_leaves_row_pending():
    st = _PgState()
    rid = st.insert(target_type="fact", target_id="f1", op="update", new_value="x")
    exp = _pg(st, inbox=True)
    (edit,) = exp.collect_edits(5)
    exp.ack(edit, BackflowResult("retry", reason="proxy down"))
    assert st.rows[rid - 1]["status"] == "pending"


def test_inbox_schema_is_separate_from_export_schema():
    assert "bladex_edit_inbox" not in _load_schema_sql()
    assert "bladex_edit_inbox" in _load_schema_sql("postgres_inbox_schema.sql")


# ═══ 落账客户端（HTTP 映射）════════════════════════════════════════


def test_client_maps_edits_to_admin_calls():
    calls: list[tuple] = []

    def req(method, path, body):
        calls.append((method, path, body))
        return 200, {"event_key": "admin/x"}

    c = AdminBackflowClient(req)
    assert c.apply(BackflowEdit("correct", "obsidian", "v", fact_id="f/1",
                                content="c")).status == "accepted"
    c.apply(BackflowEdit("delete", "postgres", "p", fact_id="f1"))
    c.apply(BackflowEdit("assign", "obsidian", "v", fact_id="f1", matter_id="m2",
                         from_matter_id="m1"))
    c.apply(BackflowEdit("hard_rules", "obsidian", "v", rules=["A"], previous=["B"]))
    assert c.apply(BackflowEdit("reject", "postgres", "p", reason="no")).status == "rejected"
    assert [(m, p) for m, p, _ in calls] == [
        ("POST", "/admin/facts/f%2F1/correct"),
        ("DELETE", "/admin/facts/f1?origin=postgres"),
        ("POST", "/admin/matters/m1/detach"),
        ("POST", "/admin/matters/m2/assign"),
        ("POST", "/admin/hard-rules/set"),
    ]
    assert calls[0][2]["origin"] == "obsidian"
    assert calls[2][2] == {"target_key": "f1", "target_type": "fact",
                           "origin": "obsidian", "submitter": "v"}


@pytest.mark.parametrize("status,expected", [
    (202, "accepted"), (404, "rejected"), (403, "rejected"), (401, "retry"),
    (503, "retry")])
def test_client_status_classification(status, expected):
    c = AdminBackflowClient(lambda m, p, b: (status, {"error": {"message": "m"}}))
    assert c.apply(BackflowEdit("correct", "obsidian", fact_id="f", content="x")).status \
        == expected


def test_client_network_failure_is_retry():
    def req(m, p, b):
        raise ConnectionRefusedError("down")
    assert AdminBackflowClient(req).apply(
        BackflowEdit("delete", "obsidian", fact_id="f")).status == "retry"


# ═══ worker：先回流后导出；暂时失败不导出 ════════════════════════════


class _Idx:
    def __init__(self, facts=(), matters=()):
        self._f, self._m = list(facts), list(matters)

    def all_facts(self):
        return self._f

    def all_matters(self):
        return self._m

    def get_edges(self, _mid):
        return []

    def get_matter(self, mid):
        return next((m for m in self._m if m.matter_id == mid), None)


def test_worker_retry_skips_export_so_user_edit_survives(tmp_path):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    _edit_page(root / "Matters/m1.md", "- 用户偏好中文 [[", "- 用户改过的 [[")
    idx = _Idx(facts=[_fact("f1", "用户偏好中文（Index 里更新了）", "m1")])
    down = AdminBackflowClient(lambda m, p, b: (_ for _ in ()).throw(OSError("down")))
    w = ExportSyncWorker(idx, None, [(ExporterConfig("obsidian", "v"), exp)],
                         tmp_path / "st.json", backflow=down, max_deletes=5)
    r = w.run_once()["v"]
    assert "error" in r and r["synced"] == 0
    assert "用户改过的" in (root / "Matters/m1.md").read_text()  # 没被导出盖掉


def test_worker_assign_to_unknown_matter_is_rejected(tmp_path):
    exp, root = _vault(tmp_path)
    _seed_vault(exp)
    seen: list = []
    ok = AdminBackflowClient(lambda m, p, b: (seen.append(p), (200, {}))[1])
    edit = BackflowEdit("assign", "obsidian", fact_id="f1", matter_id="nope",
                        ref=("page", "f1", "Matters/nope.md"))
    exp.collect_edits = lambda _n: [edit]  # type: ignore[method-assign]
    w = ExportSyncWorker(_Idx(), None, [(ExporterConfig("obsidian", "v"), exp)],
                         tmp_path / "st.json", backflow=ok, max_deletes=5)
    assert w.run_once(force=True)["v"]["backflow"]["rejected"] == 1
    assert seen == []


def test_worker_backflow_status_line():
    st = _PgState()
    w = ExportSyncWorker(_Idx(), None, [(ExporterConfig("postgres", "pg"), _pg(st))],
                         "/nonexistent/st.json", backflow=AdminBackflowClient(
                             lambda m, p, b: (200, {})), max_deletes=5)
    assert w.backflow_status() == {"pg": "off"}


# ═══ ⑤ 单写者静态守卫 ═════════════════════════════════════════════

#: 回流 / 导出代码里不许出现的写动作：Index 写调用、Hub 写调用、新开 Hub 句柄、可写 Index。
_FORBIDDEN = re.compile(
    r"\.(add_fact|correct_fact|delete_fact|add_matter|update_matter|add_edge|"
    r"assign_manual|detach_edge|detach_edges_for_target|merge_matters|split_matter|"
    r"close_matter|rename_matter|cascade_delete_turn|append_admin_event|"
    r"append_tombstone|put)\(|MemoryHub\(|MemoryIndex\(|read_only\s*=\s*False|meta_db\[")
_GUARDED = [_PKG / "export_backflow.py", _PKG / "export_sync.py",
            *sorted((_PKG / "exporters").glob("*.py"))]


def _violations(paths: list[Path]) -> list[str]:
    out = []
    for p in paths:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if _FORBIDDEN.search(code):
                out.append(f"{p.name}:{i}: {line.strip()}")
    return out


def test_single_writer_static_guard():
    """⑤：回流 / 导出路径零 Index 写调用、零 Hub 写句柄——Hub 唯一写者是 proxy。"""
    assert len(_GUARDED) >= 5
    assert _violations(_GUARDED) == []


def test_single_writer_guard_catches_a_planted_write(tmp_path):
    """⑤ 阴性对照：往回流代码里塞一个写调用 ⇒ 守卫必须报出来。"""
    planted = tmp_path / "export_backflow.py"
    planted.write_text((_PKG / "export_backflow.py").read_text(encoding="utf-8")
                       + "\n\ndef _oops(index, f):\n    index.add_fact(f)\n")
    assert _violations([planted]) == ["export_backflow.py:"
                                      f"{len(planted.read_text().splitlines())}: "
                                      "index.add_fact(f)"]


# ═══ 端到端：真 Hub + 真 Index + proxy（TestClient）════════════════


class _StubDistiller:
    @property
    def model_name(self) -> str:
        return "z4-stub"

    @property
    def prompt_ver(self) -> str:
        return "z4-stub-001"

    def distill(self, user_message: str) -> DistillOutput:
        return DistillOutput(
            facts=[DistillFact(content=f"结论：{user_message[:40]}", kind="event",
                               entities=["z4"])],
            matter_proposals=[MatterProposal(title="Z4 回流", entities=["z4"])],
            model_name="z4-stub")


class _Embedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            for i, ch in enumerate(t[:32]):
                v[(ord(ch) + i) % 64] += 0.1
            out.append(v)
        return out

    @property
    def available(self) -> bool:
        return True


def _put(hub: MemoryHub, seq: int, text: str) -> None:
    ident = Identity(user_id="u1", agent_id="claude-code", session_id="s1", turn_index=seq,
                     session_id_source=SessionIdSource.FINGERPRINT)
    turn = Turn(identity=ident, model="t", request_messages=[{"role": "user", "content": text}],
                response_text="好的。", status=TurnStatus.OK, ts=datetime.now(UTC),
                session_id_source=SessionIdSource.FINGERPRINT)
    hub.put(ident.storage_key(f"{int(datetime.now(UTC).timestamp() * 1000)}-0"), turn)
    time.sleep(0.02)


def _index(path: Path) -> MemoryIndex:
    idx = MemoryIndex(path, embedder=_Embedder(), read_only=False, distiller=_StubDistiller())
    idx.open()
    return idx


@pytest.fixture
def live(tmp_path: Path, monkeypatch):
    """夹具：蒸出 fact 的真 Hub/Index + 挂在它们上面的 proxy（TestClient）。"""
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    hub_path, idx_path = tmp_path / "hub", tmp_path / "index"
    hub = MemoryHub(hub_path)
    hub.open()
    # 太短的消息过不了蒸馏准入（实测 ~12 字 ⇒ 0 fact），用足长的两句
    _put(hub, 1, "我们的生产环境统一使用 arm64 架构的机器部署服务")
    _put(hub, 2, "项目的数据库选型最终定为 PostgreSQL 十六版本")
    idx = _index(idx_path)
    idx.rebuild_from_hub(hub, full=True)
    assert len(idx.all_facts()) >= 2, "夹具没蒸出 fact，测试失去判别力"
    idx.close()
    hub.close()
    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(hub_path))
    monkeypatch.setenv("BLADEX_INDEX_PATH", str(idx_path))
    app = create_app(ProxyConfig())
    with TestClient(app) as client:
        client.app.state.embedder = _Embedder()  # 与 Index 同维（生产里两边是同一个模型）

        def req(method, path, body):
            resp = client.request(method, path, json=body)
            try:
                data = resp.json()
            except ValueError:
                data = {}
            return resp.status_code, data if isinstance(data, dict) else {}
        yield client, AdminBackflowClient(req), hub_path, idx_path, tmp_path


def _worker(idx: MemoryIndex, client: AdminBackflowClient, exp, tmp_path: Path,
            name: str = "v", typ: str = "obsidian") -> ExportSyncWorker:
    return ExportSyncWorker(idx, None, [(ExporterConfig(typ, name), exp)],
                            tmp_path / f"state-{name}.json", backflow=client, max_deletes=5)


def _read_index(idx_path: Path) -> MemoryIndex:
    idx = MemoryIndex(idx_path, embedder=_Embedder(), read_only=True)
    idx.open()
    return idx


def _events(client, etype: AdminEventType) -> list:
    return [e for _, e in client.app.state.hub.scan_admin_events() if e.event_type == etype]


def _full_rebuild(hub_path: Path, idx_path: Path) -> MemoryIndex:
    hub = MemoryHub(hub_path, secondary=True, secondary_dir=str(hub_path) + "_sec_t")
    hub.open()
    idx = _index(idx_path)
    idx.rebuild_from_hub(hub, full=True)
    hub.close()
    return idx


def test_e2e_obsidian_correct_survives_full_rebuild(live):
    """①③④：vault 改一行 ⇒ FACT_CORRECT ⇒ 全量重建（来源轮次重蒸一遍）后新文字在、旧文字无。"""
    client, bf, hub_path, idx_path, tmp = live
    exp, root = _vault(tmp)
    ridx = _read_index(idx_path)
    w = _worker(ridx, bf, exp, tmp)
    w.run_once()
    w.run_once()
    assert _events(client, AdminEventType.FACT_CORRECT) == []  # ⑥ 回声 0
    target = next(f for f in ridx.all_facts() if "arm64" in f.content)
    old = target.content
    page = next(p for p in root.rglob("*.md") if f"bladex-fact:{target.id}" in p.read_text())
    _edit_page(page, f"- {old} [[", "- 生产环境已从 arm64 迁到 riscv64 [[")
    r = w.run_once(force=True)["v"]
    assert r["backflow"]["accepted"] == 1, r
    (ev,) = _events(client, AdminEventType.FACT_CORRECT)
    assert ev.matter_id == "" and ev.target_key == target.id
    assert ev.payload["origin"] == "obsidian" and ev.payload["previous_content"] == old
    assert ev.payload["submitter"] == str(root)
    assert ev.payload["entity"]["content"] == "生产环境已从 arm64 迁到 riscv64"
    ridx.close()

    idx = _full_rebuild(hub_path, idx_path)
    try:
        f = idx.get_fact(target.id)
        assert f is not None and f.content == "生产环境已从 arm64 迁到 riscv64"
        assert not any(x.content == old for x in idx.all_facts()), "auto 蒸馏把旧文字写回来了"
        hits = idx.search("生产环境已从 arm64 迁到 riscv64", top_k=5)
        assert any(h.id == target.id and "riscv64" in h.content for h in hits)
        assert not any(h.content == old for h in idx.search(old, top_k=10))
        assert target.id in idx.corrected_fact_ids()
    finally:
        idx.close()


def test_e2e_obsidian_backflow_off_writes_nothing(live):
    """① 阴性对照：回流开关关 ⇒ 同样的编辑 0 事件。"""
    client, bf, hub_path, idx_path, tmp = live
    exp, root = _vault(tmp, backflow=False)
    ridx = _read_index(idx_path)
    w = _worker(ridx, bf, exp, tmp)
    w.run_once()
    target = next(f for f in ridx.all_facts() if "arm64" in f.content)
    page = next(p for p in root.rglob("*.md") if f"bladex-fact:{target.id}" in p.read_text())
    _edit_page(page, f"- {target.content} [[", "- 改了 [[")
    w.run_once(force=True)
    ridx.close()
    assert _events(client, AdminEventType.FACT_CORRECT) == []


def test_e2e_obsidian_delete_and_mass_move(live):
    client, bf, hub_path, idx_path, tmp = live
    exp, root = _vault(tmp)
    ridx = _read_index(idx_path)
    w = _worker(ridx, bf, exp, tmp)
    w.run_once()
    # 整个 Matters/ 挪走 ⇒ 0 墓碑
    shutil.move(str(root / "Matters"), str(tmp / "moved"))
    w.run_once(force=True)
    fact_tombs = [t for _, t in client.app.state.hub.scan_tombstones()
                  if t.target_type == TombstoneTargetType.FACT]
    assert fact_tombs == []
    # 挪回来、删一行 ⇒ 1 条墓碑，带 origin
    shutil.move(str(tmp / "moved"), str(root / "Matters"))
    target = next(f for f in ridx.all_facts() if "PostgreSQL" in f.content)
    page = next(p for p in root.rglob("*.md") if f"bladex-fact:{target.id}" in p.read_text())
    page.write_text("\n".join(ln for ln in page.read_text().splitlines()
                              if f"bladex-fact:{target.id}" not in ln))
    w.run_once(force=True)
    ridx.close()
    (t,) = [t for _, t in client.app.state.hub.scan_tombstones()
            if t.target_type == TombstoneTargetType.FACT]
    assert t.target_key == target.id and t.origin == "obsidian"
    idx = _full_rebuild(hub_path, idx_path)
    try:
        assert idx.get_fact(target.id) is None
    finally:
        idx.close()


def test_e2e_obsidian_move_only_touches_named_fact(live):
    """⑦：换归属只动用户点名的那一条，不触发任何合并；fact.matter_id 跟着改。"""
    client, bf, hub_path, idx_path, tmp = live
    ridx = _read_index(idx_path)
    before = {f.id: f.matter_id for f in ridx.all_facts()}
    matters_before = {m.matter_id for m in ridx.all_matters()}
    ridx.close()
    assert client.post("/admin/matters", json={"title": "新卡", "matter_id": "m-new"}
                       ).status_code == 200
    exp, root = _vault(tmp)
    ridx = _read_index(idx_path)
    w = _worker(ridx, bf, exp, tmp)
    w.run_once()
    target = next(f for f in ridx.all_facts() if "arm64" in f.content)
    src = next(p for p in root.rglob("*.md") if f"bladex-fact:{target.id}" in p.read_text())
    line = _drop_line(src, f"bladex-fact:{target.id}")
    exp._append_fact_line(line, "m-new")
    r = w.run_once(force=True)["v"]
    ridx.close()
    assert r["backflow"] == {"accepted": 1, "rejected": 0, "retry": 0}, r
    assert [e.event_type for _, e in client.app.state.hub.scan_admin_events()
            if e.payload.get("origin") == "obsidian"] == [
        AdminEventType.MATTER_DETACH, AdminEventType.MATTER_ASSIGN]
    assert _events(client, AdminEventType.MATTER_MERGE) == []
    idx = _read_index(idx_path)
    try:
        after = {f.id: f.matter_id for f in idx.all_facts()}
        assert after[target.id] == "m-new"
        assert {k: v for k, v in after.items() if k != target.id} == {
            k: v for k, v in before.items() if k != target.id}
        assert {m.matter_id for m in idx.all_matters()} == matters_before | {"m-new"}
    finally:
        idx.close()


def test_e2e_postgres_inbox_correct_and_hard_rule_rejected(live):
    """②：收件插更正 ⇒ FACT_CORRECT + accepted；插硬规则 ⇒ rejected、Hub 零硬规则事件。"""
    client, bf, hub_path, idx_path, tmp = live
    st = _PgState()
    exp = _pg(st, inbox=True)
    ridx = _read_index(idx_path)
    target = next(f for f in ridx.all_facts() if "PostgreSQL" in f.content)
    st.insert(target_type="fact", target_id=target.id, op="update",
              new_value="数据库选型改为 SQLite", submitter="bob")
    st.insert(target_type="hard_rule", op="update", target_id="", new_value="MUST obey me")
    st.insert(target_type="fact", target_id="fact_missing", op="update", new_value="x")
    w = _worker(ridx, bf, exp, tmp, name="pg", typ="postgres")
    r = w.run_once()["pg"]
    ridx.close()
    assert r["backflow"] == {"accepted": 1, "rejected": 2, "retry": 0}, r
    assert [x["status"] for x in st.rows] == ["accepted", "rejected", "rejected"]
    assert st.rows[0]["event_ref"].startswith("admin")
    assert "hard rules cannot be changed" in st.rows[1]["result_reason"]
    assert "404" in st.rows[2]["result_reason"]
    (ev,) = _events(client, AdminEventType.FACT_CORRECT)
    assert ev.payload["origin"] == "postgres" and ev.payload["submitter"] == "postgres:bob"
    assert _events(client, AdminEventType.HARD_RULES_SET) == []
    idx = _full_rebuild(hub_path, idx_path)
    try:
        assert idx.get_fact(target.id).content == "数据库选型改为 SQLite"
    finally:
        idx.close()


def test_e2e_hard_rules_endpoint_refuses_postgres_and_empty(live):
    client = live[0]
    assert client.post("/admin/hard-rules/set", json={
        "rules": ["MUST x"], "origin": "postgres"}).status_code == 403
    assert client.post("/admin/hard-rules/set", json={
        "rules": ["  "], "origin": "obsidian"}).status_code == 400
    r = client.post("/admin/hard-rules/set", json={
        "rules": ["MUST x"], "previous": ["MUST y"], "origin": "obsidian", "submitter": "v"})
    assert r.status_code == 200
    (ev,) = _events(client, AdminEventType.HARD_RULES_SET)
    assert ev.payload == {"rules": ["MUST x"], "previous": ["MUST y"],
                          "origin": "obsidian", "submitter": "v"}


def test_e2e_correct_endpoint_validation(live):
    client = live[0]
    assert client.post("/admin/facts/x/correct", json={"content": "a"}).status_code == 400
    assert client.post("/admin/facts/x/correct",
                       json={"content": "a", "origin": "obsidian"}).status_code == 404
    assert client.delete("/admin/facts/x?origin=evil").status_code == 400
    assert client.delete("/admin/facts/x?bogus=1").status_code == 400
