"""MQ-L27：人设事实跨 agent 传送的三段修法（2026-08-29 Jason 拍板）。

live 病例：codex 被注入 hermes 的「用户称呼助手为"小黑"」后自称小黑。
判据经全库 8038 条判别力实测（命中 24 → 收窄后 22 真 0 误）；
阴性对照直接取实测中的两条真实误报。
"""

from __future__ import annotations

from bladex_core.fact import (
    Fact,
    effective_audience,
    is_persona_fact,
    source_agent,
)


class TestPersonaJudge:
    """判据矩阵——样本取 live Index 原文（判据一致性：A 侧原文做样本）。"""

    def test_true_positives_from_live_index(self):
        for sub, attr in [
            ("用户", "对助手的称呼"),          # 案发 fact_302b28a0cf9f 同款
            ("AI 助手", "昵称"),
            ("助手", "自称"),
            ("hermes 配置", "人设与表情设置"),
            ("hermes kawaii 人设", "表情行为"),
            ("用户与助手的称呼", "称呼习惯"),
        ]:
            assert is_persona_fact(sub, attr), (sub, attr)

    def test_measured_false_positives_stay_out(self):
        """全库实测抓到的两条误报——收窄后必须为阴性。"""
        assert not is_persona_fact("婉拒律师团队的微信文案草稿", "占位称呼"), \
            "「占位」负向词：文案占位称呼不是助手人设"
        assert not is_persona_fact("Pi agent", "npm 包名与安装方式"), \
            "收窄掉 agent/assistant 半边：包名不是人设"

    def test_work_records_are_not_persona(self):
        """行事记录类（小黑当主语）有意不拦——工作事实跨 agent 共享是价值。"""
        assert not is_persona_fact("Pi Agent 配置", "重建结果")
        assert not is_persona_fact("ornith_compare.py", "疑似用途")


class TestEffectiveAudience:
    def _fact(self, **kw):
        return Fact(id="fact_x", content="c",
                    source_ledger_key="local/hermes:default/tw:x/123-0", **kw)

    def test_unlabeled_persona_fact_locks_to_source_full_id(self):
        """2026-08-29 拍板 v2：profile=独立 agent，人设锁到完整 id
        （kawaii 是 hermes:accept 的，不该漏给 hermes:default）。"""
        f = self._fact(subject="用户", attribute="对助手的称呼")
        assert effective_audience(f) == "agent:hermes:default"

    def test_explicit_label_wins(self):
        f = self._fact(subject="用户", attribute="对助手的称呼",
                       audience="agent:codex")
        assert effective_audience(f) == "agent:codex", "显式标注永远优先于派生"

    def test_non_persona_unlabeled_stays_all(self):
        f = self._fact(subject="qwen3.8:27b", attribute="下载完成日期")
        assert effective_audience(f) == "all", "普通事实零回归"

    def test_unresolvable_source_stays_all(self):
        f = Fact(id="f", content="c", subject="助手", attribute="自称",
                 source_ledger_key="")
        assert effective_audience(f) == "all", "解不出来源就不猜（宁漏勿锁错）"
        assert source_agent(f) == ""


class TestLayeredAudienceMatch:
    """prefetch 放行集合 {all, agent:<base>, agent:<完整 id>} 的语义矩阵。
    镜像 prefetch 过滤点的判据原文（§3.2b：行为断言取 A 侧判据）。"""

    @staticmethod
    def _allowed(aud: str, agent_full: str) -> bool:
        base = agent_full.split(":")[0]
        return aud in ("all", f"agent:{base}", f"agent:{agent_full}")

    def test_profile_label_hits_only_that_profile(self):
        assert self._allowed("agent:hermes:accept", "hermes:accept")
        assert not self._allowed("agent:hermes:accept", "hermes:default"), \
            "accept 的 kawaii 人设不得漏给 default"
        assert not self._allowed("agent:hermes:accept", "codex")

    def test_base_label_covers_all_profiles(self):
        assert self._allowed("agent:hermes", "hermes:default")
        assert self._allowed("agent:hermes", "hermes:accept")
        assert not self._allowed("agent:hermes", "codex")

    def test_all_reaches_everyone(self):
        assert self._allowed("all", "codex")


class TestAdminEventApply:
    def test_audience_set_event_applies_to_fact(self, tmp_path):
        import pytest
        pytest.importorskip("lancedb")   # 沙盒无 native 依赖；用户机 gate 必跑
        from bladex_proxy.models import AdminEvent, AdminEventType
        from bladex_proxy.storage.memory_index import MemoryIndex
        idx = MemoryIndex(str(tmp_path / "idx"), embedder=None)
        idx.open()
        f = Fact(id="fact_p1", content="用户称呼助手为小黑",
                 subject="用户", attribute="对助手的称呼",
                 source_ledger_key="local/hermes:default/tw:x/1-0")
        idx.add_fact(f)
        ev = AdminEvent(event_type=AdminEventType.AUDIENCE_SET, matter_id="",
                        payload={"fact_id": "fact_p1",
                                 "audience": "agent:hermes:default"})
        assert idx._apply_admin_event(ev) is True
        assert idx.get_fact("fact_p1").audience == "agent:hermes:default"
        # 幂等：重复应用同值无害（增量重放可能重复投递）
        assert idx._apply_admin_event(ev) is True
        # 缺目标/缺值 → False 不炸
        assert idx._apply_admin_event(AdminEvent(
            event_type=AdminEventType.AUDIENCE_SET, matter_id="",
            payload={"fact_id": "nope", "audience": "agent:x"})) is False
        assert idx._apply_admin_event(AdminEvent(
            event_type=AdminEventType.AUDIENCE_SET, matter_id="",
            payload={"fact_id": "fact_p1", "audience": ""})) is False
        idx.close()

    def test_matter_id_stays_empty_per_mq_a7(self):
        """MQ-A7 护栏：新事件类型 target 走 payload 不占 matter_id
        （MQ-V9：源码断言按路径读文本，不用 inspect.getsource）。"""
        import os
        import bladex_proxy as _pkg
        from bladex_proxy.models import AdminEventType
        assert AdminEventType.AUDIENCE_SET.value == "audience_set"
        from _source_probe import package_source
        src = package_source("server")   # F0.1 拆包：server.py → server/ 包，按包拼接读源码
        assert 'append_admin_event(AdminEventType.AUDIENCE_SET, ""' in src, \
            "端点必须以空 matter_id 追加（fact_id 在 payload）"
