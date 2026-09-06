"""V-F2 验收剧本：物化幂等 / 删树重渲染逐字一致 / 用户直编捕获且不被覆盖 / 收敛。"""

from __future__ import annotations

from bladex_core.ledger import (
    ACTOR_MODEL,
    LedgerEntry,
    add_entry,
    ledger_md_path,
    new_ledger,
    render_ledger_md,
)
from bladex_proxy.flash_daemon import FlashDaemon


def _pool_v1():
    led = new_ledger(ledger_id="ldg-a1", title="任务甲",
                     goal="把批二做完", goal_source="user",
                     created_at="2026-08-25T10:00:00Z")
    led = add_entry(led, "next", LedgerEntry(text="写测试", source="model"),
                    actor=ACTOR_MODEL)
    return {"ldg-a1": led}


def _mk(tmp_path, pool_holder, events):
    return FlashDaemon(
        root=str(tmp_path), principal="u1",
        ledger_source=lambda: (pool_holder["pool"], {}),
        emit_admin_event=lambda t, p: events.append((t, p)))


class TestMaterialize:
    def test_writes_pool_then_idempotent(self, tmp_path):
        holder = {"pool": _pool_v1()}
        events: list = []
        d = _mk(tmp_path, holder, events)
        assert d.run_once()["written"] == 1
        # 无新事：不脏不写
        assert d.run_once() == {"written": 0, "user_edits": 0}
        # push 后内容没变也不真写（write_if_changed）
        d.push()
        assert d.run_once()["written"] == 0

    def test_delete_tree_rerender_identical(self, tmp_path):
        holder = {"pool": _pool_v1()}
        d = _mk(tmp_path, holder, [])
        d.run_once()
        path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
        first = open(path, encoding="utf-8").read()
        import os
        os.unlink(path)
        d.push()
        d.run_once()
        assert open(path, encoding="utf-8").read() == first  # 投影：重渲染逐字一致

    def test_pool_update_rewrites(self, tmp_path):
        holder = {"pool": _pool_v1()}
        d = _mk(tmp_path, holder, [])
        d.run_once()
        led2 = add_entry(holder["pool"]["ldg-a1"], "verified",
                         LedgerEntry(text="测试全绿", source="tool", ref="gate#2"),
                         actor=ACTOR_MODEL)
        holder["pool"] = {"ldg-a1": led2}
        d.push()
        assert d.run_once()["written"] == 1
        path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
        assert "测试全绿" in open(path, encoding="utf-8").read()


class TestUserEdit:
    def test_edit_captured_as_event_and_not_clobbered(self, tmp_path):
        holder = {"pool": _pool_v1()}
        events: list = []
        d = _mk(tmp_path, holder, events)
        d.run_once()
        path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
        # 用户在 Open 段手加一行
        content = open(path, encoding="utf-8").read()
        edited = content.replace("## Open\n\n_(empty)_",
                                 "## Open\n\n- 用户手记的一个疑点")
        open(path, "w", encoding="utf-8").write(edited)

        out = d.run_once()
        assert out["user_edits"] == 1
        assert events and events[0][0] == "ledger_user_edit"
        parsed = events[0][1]["ledger"]
        assert any(e["text"] == "用户手记的一个疑点"
                   for e in parsed["sections"]["open"])
        # 🔴 池数据还没带回用户版 ⇒ 文件不被机器版覆盖
        assert "用户手记的一个疑点" in open(path, encoding="utf-8").read()

    def test_converges_after_event_flows_back(self, tmp_path):
        holder = {"pool": _pool_v1()}
        events: list = []
        d = _mk(tmp_path, holder, events)
        d.run_once()
        path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
        content = open(path, encoding="utf-8").read()
        edited = content.replace("## Open\n\n_(empty)_",
                                 "## Open\n\n- 用户手记的一个疑点")
        open(path, "w", encoding="utf-8").write(edited)
        d.run_once()                       # 捕获事件
        # 模拟事件流回 Hub → source 带回用户版
        from bladex_core.ledger import Ledger
        holder["pool"] = {"ldg-a1": Ledger.model_validate(events[0][1]["ledger"])}
        d.push()
        d.run_once()
        final = open(path, encoding="utf-8").read()
        assert "用户手记的一个疑点" in final
        # 收敛后不再重复上报
        assert d.run_once()["user_edits"] == 0
        assert len(events) == 1

    def test_garbage_edit_id_mismatch_no_event_no_clobber(self, tmp_path):
        """`- ledger:` 元数据被写坏 ⇒ 解析出随机新 id ⇒ 不发事件、不覆盖文件、
        只告警（给不存在的账本造事件比丢一次编辑更糟）。"""
        holder = {"pool": _pool_v1()}
        events: list = []
        d = _mk(tmp_path, holder, events)
        d.run_once()
        path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
        open(path, "w", encoding="utf-8").write("完全不是账本格式的东西")
        out = d.run_once()          # 不炸
        assert out["user_edits"] == 0
        assert events == []
        assert open(path, encoding="utf-8").read() == "完全不是账本格式的东西"


class TestRenderStability:
    def test_snapshot_matches_render(self, tmp_path):
        holder = {"pool": _pool_v1()}
        d = _mk(tmp_path, holder, [])
        d.run_once()
        path = ledger_md_path(str(tmp_path), "u1", "ldg-a1")
        assert open(path, encoding="utf-8").read() == \
            render_ledger_md(holder["pool"]["ldg-a1"])
