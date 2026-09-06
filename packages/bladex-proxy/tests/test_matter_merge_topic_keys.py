"""合并提案与合并完备性（matter-dedup-keywords-20260814 卡拍板 C）。

此前 `merge_matters` 只搬边 + 平均质心——fact.matter_id 不改写（U6/U7/轴B 全部
读它，合并名存实亡）、source 措辞不进 target 原生键（LLM 再用旧措辞出 proposal
照样开新的）、source 不关闭（继续参与钉卡与 L2-L4 候选）。本文件钉死完备性
四件套 + 候选检测只读语义 + 重放幂等。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from bladex_core.fact import Fact
from bladex_core.matter import Matter, MatterEdge, MatterStatus
from bladex_core.matter import EdgeTargetType
from bladex_proxy.storage.memory_index import MemoryIndex


class MockEmbedder:
    def embed(self, texts):
        import hashlib

        out = []
        for t in texts:
            v = [0.0] * 64
            h = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big") % 64
            v[h] = 0.9
            out.append(v)
        return out

    @property
    def available(self):
        return True


def _fact(fid: str, content: str, matter_id: str = "") -> Fact:
    f = Fact(id=fid, content=content, source_user_id="u1",
             source_session="s1", matter_id=matter_id)
    f.embedding = [0.1] * 64
    return f


def _matter(mid: str, title: str, keys: dict[str, int],
            status: MatterStatus = MatterStatus.ACTIVE) -> Matter:
    return Matter(matter_id=mid, title=title, status=status,
                  aliases=[title], topic_keys=keys, centroid=[0.5] * 64,
                  centroid_weight=1.0)


_KEYS = {"泰安仁信": 3, "泰山啤酒": 5, "增资扩股": 2}


def _setup(tmpdir: str) -> MemoryIndex:
    index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(),
                        read_only=False)
    index.open()
    index.add_matter(_matter("m-src", "查询泰安仁信入股泰山啤酒的方式", dict(_KEYS)))
    index.add_matter(_matter("m-dst", "核实泰安仁信入股泰山啤酒的时间与方式",
                             {**_KEYS, "股权报告": 1}))
    index.add_fact(_fact("f1", "src 成员一", matter_id="m-src"))
    index.add_fact(_fact("f2", "src 成员二", matter_id="m-src"))
    for fid in ("f1", "f2"):
        index.add_edge(MatterEdge(matter_id="m-src",
                                  target_type=EdgeTargetType.FACT,
                                  target_key=fid))
    return index


def test_merge_completeness_four_pieces():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _setup(tmpdir)
        moved = index.merge_matters("m-src", "m-dst")
        assert moved == 2

        # ① fact.matter_id 改写（U6 有限跳/U7 ②平面/轴B 边界的消费面）
        assert index.get_fact("f1").matter_id == "m-dst"
        assert index.get_fact("f2").matter_id == "m-dst"
        # 边全部指向 target
        assert {e.target_key for e in index.get_edges("m-dst")} >= {"f1", "f2"}
        assert index.get_edges("m-src") == []

        dst = index.get_matter("m-dst")
        # ② source 措辞进 target 原生键（L3 全串下次能命中）
        assert "查询泰安仁信入股泰山啤酒的方式" in dst.aliases
        # ③ topic_keys 计票合并
        assert dst.topic_keys["泰山啤酒"] == 10
        assert dst.topic_keys["股权报告"] == 1
        # ④ source 关闭 + 双方 lifecycle 有记录
        src = index.get_matter("m-src")
        assert src.status == MatterStatus.CLOSED
        assert any(ev.event == "merged_into" for ev in src.lifecycle)
        assert any(ev.event == "merged" for ev in dst.lifecycle)
        index.close()


def test_merge_replay_is_idempotent_noop_when_source_missing():
    """重建时 L3 子层拦截成功 → source 根本没被开出来 → 重放 merge 事件 = no-op。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(),
                            read_only=False)
        index.open()
        index.add_matter(_matter("m-dst", "目标", dict(_KEYS)))
        assert index.merge_matters("m-ghost", "m-dst") == 0
        assert index.merge_matters("m-dst", "m-dst") == 0     # self-merge 拒绝
        assert index.merge_matters("__unassigned__", "m-dst") == 0
        index.close()


def test_merge_candidates_detects_and_orients_but_does_not_merge():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _setup(tmpdir)
        index.add_matter(_matter("m-other", "阿根廷友谊赛赔率",
                                 {"阿根廷": 2, "佛得角": 1, "赔率": 1}))
        pairs = index.matter_merge_candidates()
        assert len(pairs) == 1, f"应只检出泰安对，实得 {pairs}"
        p = pairs[0]
        # 方向：成员少的（dst 无边=0）并入成员多的（src 2 条边）
        assert p["target_matter_id"] == "m-src" and p["source_matter_id"] == "m-dst"
        assert p["overlap"] >= 3 and "泰山啤酒" in p["shared_keys"]
        # 只检测不合并
        assert index.get_matter("m-src").status == MatterStatus.ACTIVE
        assert index.get_matter("m-dst").status == MatterStatus.ACTIVE
        index.close()


def test_merge_candidates_skips_closed_and_empty_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(),
                            read_only=False)
        index.open()
        index.add_matter(_matter("m-a", "关闭的", dict(_KEYS), status=MatterStatus.CLOSED))
        index.add_matter(_matter("m-b", "在开的", dict(_KEYS)))
        index.add_matter(Matter(matter_id="m-c", title="无键", topic_keys={}))
        assert index.matter_merge_candidates() == []
        index.close()


def test_merge_candidates_no_false_positives_on_noisy_legacy_matters():
    """🔴 存量精度红线（2026-08-14 当天实测定的边界）：存量 Matter 的派生键
    是路径/Redis key/命令名等噪声——曾用"长键包含匹配"想把存量泰安对救出来，
    live 一跑 300 对假阳性（'bladex' 藏在 N 个长键里 = N 票，MS-16 吸尘器
    借尸还魂），当天回退为**纯相等**。本测试用 live 假阳性对的真实键形态钉死：
    宁可存量检不出（手动 merge 兜底），不许把无关 dev-matter 配成对。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = MemoryIndex(Path(tmpdir) / "index", embedder=MockEmbedder(),
                            read_only=False)
        index.open()
        # live 假阳性对的真实键形态（更新路径 vs 审批权限，本不相干）
        index.add_matter(Matter(
            matter_id="m-noise-a", title="更新持久化记忆中的 BladeX 项目路径为新路径",
            status=MatterStatus.ACTIVE, topic_keys={},
            aliases=["更新持久化记忆中的 BladeX 项目路径为新路径"],
            entities=["/users/jasonye/dev/bladex/logs/consolidator-1.log",
                      "bladex", "bladex:sync:cmd", "pgrep", ".env"]))
        index.add_matter(Matter(
            matter_id="m-noise-b", title="审批 Codex 代理的权限提升命令请求",
            status=MatterStatus.ACTIVE, topic_keys={},
            aliases=["审批 Codex 代理的权限提升命令请求"],
            entities=["/proc/51591/environ", "bladex", "bladex:*", "codex",
                      "lsof", "redis-cli"]))
        assert index.matter_merge_candidates() == [], (
            "噪声 dev-matter 被配成合并提案——包含匹配回潮或毒词过滤失效")
        index.close()


def test_deferred_merge_applies_on_incremental_pass(tmp_path):
    """🔴 deferred 兑现（2026-08-14 用户实测抓的缺口）：管理事件此前只在 full
    rebuild 重放——proxy 拿不到写锁回 'deferred: 下次 rebuild 生效'，而正常
    运行永远没有 full rebuild，三条 MATTER_MERGE 落账后 Matter 纹丝不动。
    修法 = 增量 pass 水位重放。本测试走**空 Memory Hub（无新 turn）**路径
    ——B17 教训：提前 return 的分支也必须够得到重放。"""
    from bladex_proxy.models import AdminEventType
    from bladex_proxy.storage.memory_hub import MemoryHub

    ledger = MemoryHub(tmp_path / "ledger")
    ledger.open()
    index = MemoryIndex(tmp_path / "index", embedder=MockEmbedder(), read_only=False)
    index.open()
    index.add_matter(_matter("m-src", "查询泰安仁信入股泰山啤酒的方式", dict(_KEYS)))
    index.add_matter(_matter("m-dst", "核实泰安仁信入股泰山啤酒的时间与方式", dict(_KEYS)))
    index.add_fact(_fact("f1", "src 成员", matter_id="m-src"))
    index.add_edge(MatterEdge(matter_id="m-src", target_type=EdgeTargetType.FACT,
                              target_key="f1"))

    # proxy 端 deferred 的形态：事件在账上，index 未动
    ledger.append_admin_event(AdminEventType.MATTER_MERGE, "m-dst", "m-src",
                              source_matter_id="m-src")

    # 增量 pass（无新 turn）→ merge 兑现
    index.rebuild_from_hub(ledger, full=False)
    assert index.get_matter("m-src").status == MatterStatus.CLOSED
    assert index.get_fact("f1").matter_id == "m-dst"
    dst = index.get_matter("m-dst")
    merged_events = [ev for ev in dst.lifecycle if ev.event == "merged"]
    assert len(merged_events) == 1

    # 第二个增量 pass：水位挡住，不重复应用（计票字段不许重复累加）
    tk_before = dict(dst.topic_keys)
    index.rebuild_from_hub(ledger, full=False)
    dst2 = index.get_matter("m-dst")
    assert len([ev for ev in dst2.lifecycle if ev.event == "merged"]) == 1
    assert dst2.topic_keys == tk_before
    index.close()
    ledger.close()


def test_merge_closed_source_is_noop_guard():
    """幂等护栏：source 已 closed（已合并过）再应用 = no-op。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _setup(tmpdir)
        assert index.merge_matters("m-src", "m-dst") == 2
        tk = dict(index.get_matter("m-dst").topic_keys)
        assert index.merge_matters("m-src", "m-dst") == 0     # 重复应用
        assert index.get_matter("m-dst").topic_keys == tk     # 计票没脏
        index.close()
