"""③④ 漂移检测与认领建议（2026-08-26，接入域）。

🔴 本卡的核心不是"能不能猜出是 codex"，是 **能不能把"升级"与"子代理"分开**——
两者的判据正好相反，搞反了两边都错：合并子代理是**误合并**，
给升级后的自己起新名字是**身份分裂**。

| | 版本升级失联 | 子代理 |
|---|---|---|
| 现象 | 老 agent_id **停**，新桶**起** | 老 agent_id 与新桶**并存** |
| 判据 | 接替 | 并存 |
"""

from __future__ import annotations

from bladex_proxy.agent_drift import detect, suggest_claim, transport_overlap

CODEX_BASIS = "ua:codex|vendor:codex-parent"     # 真实 basis（取自 live 日志）
KNOWN = ["codex", "hermes:default", "claude-code"]


class TestTransportOverlap:
    def test_vendor_segment_prefix_matches_base_name(self):
        """真实形态：`x-codex-parent-thread-id` 被贪婪正则拆出 vendor=`codex-parent`，
        与 base 名 `codex` 不相等但显然同源。"""
        assert transport_overlap(CODEX_BASIS, KNOWN) == "codex"

    def test_no_match_is_empty(self):
        assert transport_overlap("ua:asyncopenai", KNOWN) == ""

    def test_tools_hash_is_not_an_identity_clue(self):
        """🔴 `tools:` 是工具集哈希——**正是升级会变的那一维**。
        拿它找线索等于用最脆的信号做最重的判断。"""
        assert transport_overlap("tools:deadbeef|none", KNOWN) == ""

    def test_unknown_buckets_are_not_candidates(self):
        assert transport_overlap("ua:codex", ["unknown-9d9bdd3e"]) == ""


class TestSuccessionIsTheDiscriminator:
    def test_upgrade_shape_succession_true(self):
        """老 agent 最后一次出现**早于**新桶 ⇒ 它之后就没再说话 ⇒ 接替 ⇒ 升级。"""
        ev = detect(basis=CODEX_BASIS, known_agent_ids=KNOWN,
                    last_seen={"codex": 100.0}, bucket_first_seen=200.0)
        assert ev.succession is True

    def test_subagent_shape_succession_false(self):
        """🔴 老 agent 在新桶出现**之后**仍在跑 ⇒ 并存 ⇒ 多半是子代理，不是升级。
        guardian 就是这个形态：主 codex 一直在跑，子代理并行出现。"""
        ev = detect(basis=CODEX_BASIS, known_agent_ids=KNOWN,
                    last_seen={"codex": 300.0}, bucket_first_seen=200.0)
        assert ev.succession is False

    def test_no_history_no_claim(self):
        ev = detect(basis=CODEX_BASIS, known_agent_ids=KNOWN,
                    last_seen={}, bucket_first_seen=200.0)
        assert ev.succession is False


class TestSessionContinuation:
    def test_content_level_evidence(self):
        """唯一不依赖 header 的证据：同一个对话，前半截 codex、后半截 unknown。"""
        ev = detect(basis=CODEX_BASIS, known_agent_ids=KNOWN,
                    last_seen={"codex": 1.0}, bucket_first_seen=2.0,
                    session_match="codex")
        assert ev.session_continuation is True and ev.score == 3

    def test_session_match_alone_can_name_the_candidate(self):
        """传输维完全对不上（agent 换了 SDK），但会话接得上 —— 仍是线索。"""
        ev = detect(basis="ua:asyncopenai", known_agent_ids=KNOWN,
                    last_seen={"codex": 1.0}, bucket_first_seen=2.0,
                    session_match="codex")
        assert ev.looks_like == "codex" and ev.session_continuation is True


class TestSuggestionThreshold:
    def test_two_pieces_of_evidence_required(self):
        """🔴 单条太弱：只有传输重叠时，"另一个用同款 SDK 的新客户端"与
        "老朋友升级"长得一模一样，而误合并=0 是红线。"""
        weak = detect(basis=CODEX_BASIS, known_agent_ids=KNOWN,
                      last_seen={"codex": 300.0}, bucket_first_seen=200.0)
        assert weak.score == 1
        assert suggest_claim("unknown-x", weak) is None

    def test_strong_evidence_yields_a_prefilled_claim(self):
        ev = detect(basis=CODEX_BASIS, known_agent_ids=KNOWN,
                    last_seen={"codex": 100.0}, bucket_first_seen=200.0,
                    session_match="codex")
        sug = suggest_claim("unknown-9d9bdd3e", ev,
                            {"originator": "Codex Desktop",
                             "user-agent": "Codex Desktop/0.149.0-alpha.4.3"})
        assert sug is not None
        assert sug.to_agent_id == "codex"
        assert sug.merge is True, "目标已存在 ⇒ 认领即合并，必须走二次确认"
        assert sug.rule.get("header_patterns"), "规则没预填，用户还得自己写"

    def test_no_candidate_no_suggestion(self):
        ev = detect(basis="ua:something-new", known_agent_ids=KNOWN,
                    last_seen={}, bucket_first_seen=1.0)
        assert ev.looks_like == "" and suggest_claim("b", ev) is None


class TestWiring:
    def test_registry_exposes_the_two_readouts_drift_needs(self):
        from bladex_proxy.agent_registry import AgentRegistry
        reg = AgentRegistry()
        reg.register("local", "codex", origin_key="ok1")
        reg.register("local", "unknown-zzz", origin_key="ok2")
        assert reg.known_agent_ids() == ["codex"], "unknown 桶不该进比对面"
        assert "codex" in reg.last_seen_by_agent()

    def test_sighting_carries_drift_and_survives_restart(self, tmp_path):
        from bladex_proxy.agent_registry import AgentRegistry, load_cache, save_cache
        reg = AgentRegistry()
        reg.note_unrecognized("local", "unknown-9d9bdd3e", basis=CODEX_BASIS,
                              drift={"looks_like": "codex", "score": 3})
        save_cache(reg, root=str(tmp_path))
        fresh = AgentRegistry()
        load_cache(fresh, root=str(tmp_path))
        assert fresh.pending_claims(None)[0].drift["looks_like"] == "codex"

    def test_evidence_can_be_filled_in_later(self):
        """"接替"要等老 agent 停下来才成立 —— 首次登记时可能还没有证据。"""
        from bladex_proxy.agent_registry import AgentRegistry
        reg = AgentRegistry()
        reg.note_unrecognized("local", "b1", basis="ua:x")
        reg.note_unrecognized("local", "b1", basis="ua:x", drift={"looks_like": "codex"})
        assert reg.pending_claims(None)[0].drift["looks_like"] == "codex"

    def test_identity_emits_the_renamed_warning(self):
        import inspect

        from bladex_proxy import identity
        src = inspect.getsource(identity.resolve_identity)
        assert "agent_signal_drift" in src
        assert "agent_unrecognized" in src, "无线索时仍要走原告警，不能一刀切改口"
