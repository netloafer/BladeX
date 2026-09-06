"""MQ-A10：agent 注册表跨重启持久化。

🔴 病例（2026-08-26 live）：`AgentRegistry` 是纯内存单例、零恢复。那次重启后
**第一个**请求恰好是 Hermes 的内部子调用（无 system prompt、单条 user），
同源继承（G11.5）没有记录可继承 ⇒ 落 `unknown-0eaa8faf`，
而它的 `origin_key=0eaa8faf` 与 `hermes:default` **完全相同**——机制本该接住它。
更糟：那一刻 `note_unrecognized` 把这个桶**永久**登记进待认领列表，每重启一次多一条。

模块原 docstring 写着"不持久化（进程重启重建）——单用户本地场景，**首请求识别后即恢复**"，
而这条前提恰恰不成立：**首请求可能就是那个识别不出来的子调用**。
（刚性原则 12：把缺陷写进文档 ≠ 处理了缺陷。）
"""

from __future__ import annotations

import json
import time

from bladex_proxy import agent_registry as ar
from bladex_proxy.agent_registry import AgentRegistry, load_cache, restore, save_cache, snapshot


def _reg_with_binding() -> AgentRegistry:
    reg = AgentRegistry()
    reg.register("local", "hermes:default", trigger="system_prompt:hermes agent",
                 origin_key="0eaa8faf")
    return reg


class TestRoundTrip:
    def test_binding_survives_restart(self, tmp_path):
        """🔴 本卡的判据：重启后**第一个**请求就是子调用时，同源继承仍能命中。"""
        reg = _reg_with_binding()
        assert save_cache(reg, root=str(tmp_path))

        fresh = AgentRegistry()                       # 模拟进程重启
        assert fresh.lookup_sticky("local", "0eaa8faf") is None   # 恢复前：接不住
        load_cache(fresh, root=str(tmp_path))
        assert fresh.lookup_sticky("local", "0eaa8faf") == "hermes:default"

    def test_pending_claims_survive_restart(self, tmp_path):
        """待认领列表也是重启即空——dashboard 只列得出"本次启动以来见过的"。"""
        reg = AgentRegistry()
        reg.note_unrecognized("local", "unknown-9eb9e3a9", basis="ua:asyncopenai",
                              headers={"user-agent": "AsyncOpenAI/Python"})
        save_cache(reg, root=str(tmp_path))

        fresh = AgentRegistry()
        load_cache(fresh, root=str(tmp_path))
        items = fresh.pending_claims(None)
        assert len(items) == 1
        assert items[0].bucket_id == "unknown-9eb9e3a9"
        assert items[0].basis == "ua:asyncopenai"

    def test_cross_origin_still_refuses_to_inherit(self, tmp_path):
        """🔴 阳性对照：恢复缓存**不得**放宽同源判据。
        2026-08-18 dsh 被静默署名成 hermes:default 就是"不同客户端也继承"造成的。"""
        reg = _reg_with_binding()
        save_cache(reg, root=str(tmp_path))
        fresh = AgentRegistry()
        load_cache(fresh, root=str(tmp_path))
        assert fresh.lookup_sticky("local", "9eb9e3a9") is None, \
            "跨 origin 继承了 ⇒ 回归 dsh 误署名事故"


class TestTTL:
    def test_stale_binding_is_dropped(self):
        """过期绑定直接丢——origin_key 粒度粗（UA+vendor），放久了可能与现实脱节。
        宁可退回 unknown，不可误署名。"""
        reg = AgentRegistry()
        data = {"version": 1, "bindings": [
            {"user_id": "local", "origin_key": "old", "agent_id": "hermes",
             "identified_at": time.time() - 30 * 86400},
            {"user_id": "local", "origin_key": "new", "agent_id": "hermes",
             "identified_at": time.time()},
        ], "pending": []}
        n_bind, _ = restore(reg, data)
        assert n_bind == 1
        assert reg.lookup_sticky("local", "old") is None
        assert reg.lookup_sticky("local", "new") == "hermes"

    def test_stale_pending_is_kept(self):
        """🔴 过期只丢绑定、不丢待认领：前者影响**归属判定**（陈旧=可能误署名），
        后者只是一张给用户看的列表（陈旧=界面多一行）。代价不对等，处置就不该一样。"""
        reg = AgentRegistry()
        _, n_pend = restore(reg, {"version": 1, "bindings": [], "pending": [
            {"user_id": "local", "bucket_id": "unknown-x", "basis": "ua:z",
             "first_seen": time.time() - 90 * 86400, "count": 3}]})
        assert n_pend == 1


class TestRobustness:
    def test_missing_file_is_not_an_error(self, tmp_path):
        assert load_cache(AgentRegistry(), root=str(tmp_path)) == (0, 0)

    def test_corrupt_file_degrades_quietly(self, tmp_path):
        p = ar.cache_path(str(tmp_path))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not json", encoding="utf-8")
        assert load_cache(AgentRegistry(), root=str(tmp_path)) == (0, 0)

    def test_version_mismatch_discards_whole_cache(self):
        """缓存没有兼容义务：字段变了整份作废重建，不做半吊子迁移。"""
        reg = AgentRegistry()
        assert restore(reg, {"version": 999, "bindings": [
            {"user_id": "u", "origin_key": "o", "agent_id": "a",
             "identified_at": time.time()}]}) == (0, 0)

    def test_one_bad_record_does_not_void_the_rest(self):
        reg = AgentRegistry()
        n_bind, _ = restore(reg, {"version": 1, "bindings": [
            {"origin_key": "o1"},                                   # 缺 agent_id
            {"user_id": "u", "origin_key": "o2", "agent_id": "a",
             "identified_at": time.time()},
        ], "pending": []})
        assert n_bind == 1

    def test_snapshot_keeps_one_row_per_origin(self):
        """快照按 (user, origin_key) 去重——否则文件随流量线性增长，
        缓存不该有这种性质。"""
        reg = AgentRegistry()
        for _ in range(5):
            reg.register("local", "hermes:default", origin_key="0eaa8faf")
        assert len(snapshot(reg)["bindings"]) == 1

    def test_records_without_origin_key_are_not_persisted(self):
        """没有 origin_key 的记录对同源继承无用，存了只是噪声。"""
        reg = AgentRegistry()
        reg.register("local", "hermes:default", origin_key="")
        assert snapshot(reg)["bindings"] == []


class TestAutosaveArming:
    def test_no_write_before_load_cache(self, tmp_path, monkeypatch):
        """🔴 自动落盘由 `load_cache()` 武装 ⇒ **单测与脚本永远不写用户的 data/**。
        沙盒残渣污染过 gate 一次，这条是那类事故的结构性防线。"""
        monkeypatch.setattr(ar, "_autosave_path", None)
        reg = AgentRegistry()
        reg.register("local", "hermes:default", origin_key="ok1")
        reg.note_unrecognized("local", "unknown-y", basis="ua:z")
        assert not ar.cache_path(str(tmp_path)).exists()

    def test_new_binding_writes_after_arming(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ar, "_autosave_path", ar.cache_path(str(tmp_path)))
        reg = AgentRegistry()
        reg.register("local", "hermes:default", origin_key="ok1")
        p = ar.cache_path(str(tmp_path))
        assert p.exists()
        assert json.loads(p.read_text(encoding="utf-8"))["bindings"][0]["origin_key"] == "ok1"

    def test_repeat_binding_does_not_rewrite(self, tmp_path, monkeypatch):
        """register 是热路径（每轮都调）——只有**新**绑定才写盘。"""
        monkeypatch.setattr(ar, "_autosave_path", ar.cache_path(str(tmp_path)))
        reg = AgentRegistry()
        reg.register("local", "hermes:default", origin_key="ok1")
        p = ar.cache_path(str(tmp_path))
        mtime = p.stat().st_mtime_ns
        for _ in range(20):
            reg.register("local", "hermes:default", origin_key="ok1")
        assert p.stat().st_mtime_ns == mtime, "重复绑定也在写盘 ⇒ 热路径每轮一次 IO"


class TestIsolationDiscipline:
    """🔴 2026-08-26 当场踩到、且是**第二次**踩：凡"按部署根发现"的新文件源，
    必须同时接进 conftest 隔离面。

    症状：server 启动会 `load_cache()`，不隔离就读写**真实部署**的
    `data/agent_registry.json` —— 一个全新用例启动时打出 `pending=4`，
    那 4 条是上一个用例留下的；跨用例污染，且按执行顺序抽风。
    `BLADEX_AGENT_RULES_PATH` 的注释里写着同一句教训（"结果随用户当天在 dashboard
    存了什么规则而变，2026-08-19 实测炸过三条"），我没照做。
    """

    def test_env_override_exists_and_is_honoured(self, monkeypatch, tmp_path):
        target = tmp_path / "elsewhere.json"
        monkeypatch.setenv("BLADEX_AGENT_REGISTRY_CACHE", str(target))
        assert ar.cache_path() == target

    def test_explicit_root_beats_env(self, monkeypatch, tmp_path):
        """显式参数比 env 更具体——反过来会让"指定了目录却写到别处"。"""
        monkeypatch.setenv("BLADEX_AGENT_REGISTRY_CACHE", "/nope/x.json")
        assert str(tmp_path) in str(ar.cache_path(str(tmp_path)))

    def test_conftest_isolates_it(self):
        """守卫本身：这个 env 必须在 conftest 的隔离表里，否则下次又会漏。"""
        import os
        val = os.environ.get("BLADEX_AGENT_REGISTRY_CACHE", "")
        assert val, "conftest 没有隔离注册表缓存 ⇒ 测试会写用户的 data/"
        assert "data/agent_registry.json" not in val or val.startswith("/tmp"), \
            f"隔离值指向了疑似真实部署路径：{val}"
