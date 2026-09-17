"""MQ-L38 · 账本块的相关性候选段 + 建本告警 + 旧账本 matter 回填端点（2026-09-04）。

钉四件事：
① 有命中 ⇒ 块里多一段 `Ledgers that look like the SAME task…`，命中项只在这段出现一次，
   recency top-5 段**原样**还在（两段并列，谁也不替谁）；
② 无命中 ⇒ 块**逐字**等于修前形态（零差异：把相关性函数打成空即旧渲染）；
③ 候选里有 ≥0.5 的同题本却仍 `ledger_id=''` 建新本 ⇒ `ledger_created_despite_match` 告警；
④ `/admin/ledgers/bind-legacy-matters`：dry-run 不写；`--yes` 只对 `matter_id` 空的发事件；
   已绑本一个不动；CLI 子选项走到端点。
"""
from __future__ import annotations

import asyncio
import json

import structlog.testing

from bladex_core.ledger import new_ledger
from bladex_proxy.agency import _MATCHING_LEDGERS_HEADER, AgencyRuntime

_SYS = {"role": "system", "content": "You are a test agent."}


def _on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")


def _seed(ag: AgencyRuntime) -> None:
    """池：一本 08-25 的泰山旧本 + 6 本更新的无关本（把旧本挤出 recency top-5）。"""
    ag.pool["ldg-taishan"] = new_ledger(
        title="泰山啤酒破产重整最新进展梳理", goal="小黑，帮我再梳理一下泰山啤酒破产重整的最新进展",
        created_at="2026-08-25T08:30:00+00:00", ledger_id="ldg-taishan")
    for i, (t, g) in enumerate([
        ("设置默认报告输出目录", "把报告默认输出目录设置为 ~/Reports"),
        ("三星PIM存内计算深度分析", "深度分析三星 PIM 存内计算"),
        ("微信团队开源 embedding 模型调研", "调研微信团队开源的 embedding 模型"),
        ("评估 v5 架构开发进展", "评估 BladeX v5 架构的开发进展"),
        ("北京今日天气查询", "查询北京今日天气"),
        ("GitHub freetoken 项目调研", "调研 GitHub 上的 freetoken 项目"),
    ]):
        ag.pool[f"ldg-new{i}"] = new_ledger(
            title=t, goal=g, created_at=f"2026-09-0{i + 1}T00:00:00+00:00",
            ledger_id=f"ldg-new{i}")


def _switch(ag, agent, **args):
    return asyncio.run(ag.toolface.dispatch(
        "bladex_ledger_switch", json.dumps({"ledger_id": "", "title": "t", **args}),
        allowed_exposure="public",
        context={"agent_id": agent, "user_query": "x", "turn_index": 0}))


# ── ① 有命中：相关段出现，top-5 原样，命中项不重复 ─────────────────────────

def test_matching_section_is_added_and_recent_list_stays(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    msgs = [_SYS, {"role": "user", "content": "小黑，梳理泰山啤酒破产重整最新进展"}]
    with structlog.testing.capture_logs() as cap:
        out = ag.insert_ledger_block(msgs, "hermes:default", tier="medium",
                                     with_instruction=True)
    block = out[-1]["content"]
    assert _MATCHING_LEDGERS_HEADER in block
    assert "Other recent ledgers" in block, "recency 段必须原样还在——两段并列"
    assert block.count("ldg-taishan") == 1, "命中项只在相关段出现一次"
    i_match, i_recent = block.index(_MATCHING_LEDGERS_HEADER), block.index("Other recent ledgers")
    assert i_match < i_recent, "相关段在 recency 段之前"
    recent_part = block[i_recent:]
    assert "ldg-taishan" not in recent_part
    assert recent_part.count("- ldg-new") == 5, "top-5 一个不少"
    row = next(e for e in cap if e.get("event") == "agency_ledger_candidates")
    assert row["relevant"] == ["ldg-taishan"] and row["top_score"] >= 0.9 and row["recent"] == 5
    assert "Prefer switching to a listed matching ledger" in block


# ── ② 无命中：逐字等于修前 ─────────────────────────────────────────────────

def test_no_match_renders_exactly_the_old_block(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    msgs = [_SYS, {"role": "user", "content": "继续"}]
    new = ag.insert_ledger_block(msgs, "hermes:default", tier="medium",
                                 with_instruction=True)[-1]["content"]
    # 修前形态 = 相关性函数恒空时的渲染
    import bladex_proxy.agency.runtime as agency_mod   # F0.1 拆包：消费方 ledger_injection_message 在 runtime.py
    monkeypatch.setattr(agency_mod, "relevant_ledgers", lambda *a, **k: [])
    old = ag.insert_ledger_block(msgs, "hermes:default", tier="medium",
                                 with_instruction=True)[-1]["content"]
    assert new == old
    assert _MATCHING_LEDGERS_HEADER not in new


def test_no_instruction_means_no_candidates_at_all(monkeypatch):
    """无 switch 工具时整段是噪声——与 top-5 同门控（MQ-L7 有令无器）。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    msg = ag.ledger_injection_message("hermes:default", with_instruction=False,
                                      user_text="小黑，梳理泰山啤酒破产重整最新进展")
    assert msg is None or _MATCHING_LEDGERS_HEADER not in msg["content"]


# ── ③ 建本告警 ──────────────────────────────────────────────────────────

def test_create_despite_match_is_warned(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    msgs = [_SYS, {"role": "user", "content": "小黑，梳理泰山啤酒破产重整最新进展"}]
    ag.insert_ledger_block(msgs, "hermes:default", tier="medium", with_instruction=True)
    with structlog.testing.capture_logs() as cap:
        _switch(ag, "hermes:default")
    rows = [e for e in cap if e.get("event") == "ledger_created_despite_match"]
    assert len(rows) == 1 and rows[0]["candidate"] == "ldg-taishan" and rows[0]["score"] >= 0.5


def test_create_without_match_is_silent(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    msgs = [_SYS, {"role": "user", "content": "写一个 python 脚本统计文件数"}]
    ag.insert_ledger_block(msgs, "hermes:default", tier="medium", with_instruction=True)
    with structlog.testing.capture_logs() as cap:
        _switch(ag, "hermes:default")
    assert not [e for e in cap if e.get("event") == "ledger_created_despite_match"]


# ── ③b 🔴 MQ-L69：告警必须说清"这一轮到底给没给模型看候选" ──────────────


def test_warning_says_candidates_were_shown(monkeypatch):
    """用户面轮渲染了候选 ⇒ `candidates_shown=True`，这才是"模型无视了提示"。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    msgs = [_SYS, {"role": "user", "content": "小黑，梳理泰山啤酒破产重整最新进展"}]
    ag.insert_ledger_block(msgs, "hermes:default", tier="medium", with_instruction=True)
    with structlog.testing.capture_logs() as cap:
        _switch(ag, "hermes:default")
    rows = [e for e in cap if e.get("event") == "ledger_created_despite_match"]
    assert rows and rows[0]["candidates_shown"] is True


def test_stale_candidates_do_not_masquerade_as_shown(monkeypatch):
    """🔴 本条钉的就是 L69 的缺陷本体：`_last_candidates` **跨轮残留**。

    序列照抄 live：① 用户面轮算出候选并渲染 → ② 工具面轮（候选段只进用户面，
    `with_instruction=False` 同理）→ ③ 模型在第 ② 轮里新建账本。

    修前：告警照打，且与"看着候选仍新建"**在日志上一模一样** ——
    09-11 CC 那次告警前 60 秒零条 `agency_ledger_candidates`，属于哪一种无法判定。
    修后：`candidates_shown=False` 把它分出来。

    ⚠️ 告警**本身不取消**（只记不拦，三红线），变的只是它可被复算。
    """
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    msgs = [_SYS, {"role": "user", "content": "小黑，梳理泰山啤酒破产重整最新进展"}]
    ag.insert_ledger_block(msgs, "hermes:default", tier="medium", with_instruction=True)
    assert ag._last_candidates, "前置：第一轮必须真算出候选，否则本用例测了个空"

    ag.insert_ledger_block([_SYS, {"role": "user", "content": "继续"}],
                           "hermes:default", tier="medium", with_instruction=False)
    assert ag._last_candidates, "候选确实残留下来了——这正是缺陷的成因，不是修它"

    with structlog.testing.capture_logs() as cap:
        _switch(ag, "hermes:default")
    rows = [e for e in cap if e.get("event") == "ledger_created_despite_match"]
    assert rows, "告警不取消：只记不拦"
    assert rows[0]["candidates_shown"] is False, \
        "这一轮没给模型看候选，不能算在模型头上"


# ── ④ 旧账本 matter 回填 ────────────────────────────────────────────────────

def _app(monkeypatch):
    import tempfile
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    _on(monkeypatch)
    cfg = ProxyConfig(hard_rules=[], upstream_model="openai/test", upstream_api_key="sk-fake",
                      rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb")
    app = create_app(cfg)
    return app, TestClient, patch


def test_bind_legacy_matters_dry_run_then_write(monkeypatch):
    from bladex_core.ledger_runtime import ledger_anchor_matter_id
    app, TestClient, _ = _app(monkeypatch)
    with TestClient(app) as client:
        ag = app.state.agency
        ag.pool["ldg-old"] = new_ledger(title="旧", goal="", ledger_id="ldg-old",
                                        created_at="2026-08-25T00:00:00+00:00")
        ag.pool["ldg-bound"] = new_ledger(title="已绑", goal="", ledger_id="ldg-bound",
                                          created_at="2026-09-02T00:00:00+00:00",
                                          ).model_copy(update={"matter_id": "m-keep"})
        emitted: list = []
        monkeypatch.setattr(ag, "_emit", lambda et, payload: emitted.append((str(et), payload)))
        hdr = {"Authorization": "Bearer sk"}
        r = client.post("/admin/ledgers/bind-legacy-matters", json={}, headers=hdr)
        assert r.status_code == 200 and r.json()["status"] == "dry_run"
        assert [x["ledger_id"] for x in r.json()["ledgers"]] == ["ldg-old"]
        assert emitted == [] and ag.pool["ldg-old"].matter_id == ""
        r = client.post("/admin/ledgers/bind-legacy-matters", json={"dry_run": False}, headers=hdr)
        assert r.json()["status"] == "bound" and r.json()["unbound"] == 1
        assert ag.pool["ldg-old"].matter_id == ledger_anchor_matter_id("ldg-old")
        assert ag.pool["ldg-bound"].matter_id == "m-keep", "已绑本一个不动"
        assert len(emitted) == 1 and "LEDGER_UPDATE" in emitted[0][0].upper()
        assert emitted[0][1]["ledger"]["matter_id"] == ledger_anchor_matter_id("ldg-old")
        # 幂等：再跑一次没有可回填的
        r = client.post("/admin/ledgers/bind-legacy-matters", json={"dry_run": False}, headers=hdr)
        assert r.json()["unbound"] == 0


def test_cli_doctor_bind_flag_calls_endpoint(monkeypatch, capsys, tmp_path, isolated_home):
    from bladex_proxy import cli
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    calls: list = []

    def _rec(method, path, body=None, params=None, timeout=15.0):
        calls.append((method, path, body))
        return 200, {"status": "dry_run" if (body or {}).get("dry_run", True) else "bound",
                     "checked": 3, "unbound": 1,
                     "ledgers": [{"ledger_id": "ldg-old", "matter_id": "m-abc", "title": "旧"}]}
    from _source_probe import consumer_module
    _ldg = consumer_module("bladex_proxy.cli.ledger_cmds", "bladex_proxy.cli")   # F0.1 拆包：消费方在 cli/ledger_cmds.py
    monkeypatch.setattr(_ldg, "_admin_call", _rec)
    assert cli.main(["ledger", "doctor", "--bind-legacy-matters"]) == 0
    assert calls[-1][:2] == ("POST", "/admin/ledgers/bind-legacy-matters") and calls[-1][2]["dry_run"] is True
    assert "dry run" in capsys.readouterr().out
    assert cli.main(["ledger", "doctor", "--bind-legacy-matters", "--yes"]) == 0
    assert calls[-1][2]["dry_run"] is False
    assert "Bound 1" in capsys.readouterr().out


# ── ⑤ C6c：候选命中却写进别的本 ⇒ ledger_update_off_candidate ────────────────

def _update(ag, agent, **args):
    body = {"section": "verified", "op": "add", "text": "x", **args}
    return asyncio.run(ag.toolface.dispatch(
        "bladex_ledger_update", json.dumps(body), allowed_exposure="public",
        context={"agent_id": agent, "user_query": "x", "turn_index": 0}))


def test_update_off_candidate_is_warned(monkeypatch):
    """09-04 09:10 病例：泰山句（候选 c9ab 1.0）→ 模型 update 落进激活本 v5。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    ag.activation.restore({ag.scope_of("hermes:default"): "ldg-new3"})   # 激活本 = 评估 v5
    msgs = [_SYS, {"role": "user", "content": "小黑，梳理泰山啤酒破产重整最新进展"}]
    ag.insert_ledger_block(msgs, "hermes:default", tier="medium", with_instruction=True)
    with structlog.testing.capture_logs() as cap:
        out = _update(ag, "hermes:default")
    assert not str(out).startswith("Error"), out
    rows = [e for e in cap if e.get("event") == "ledger_update_off_candidate"]
    assert len(rows) == 1
    assert rows[0]["active"] == "ldg-new3" and rows[0]["candidate"] == "ldg-taishan"
    assert rows[0]["score"] >= 0.5 and rows[0]["section"] == "verified"


def test_update_into_candidate_or_without_match_is_silent(monkeypatch):
    _on(monkeypatch)
    ag = AgencyRuntime()
    _seed(ag)
    scope = ag.scope_of("hermes:default")
    # a) 候选命中且 update 落进候选本本身（激活本就是它）⇒ 不响（候选段排除激活本，故无候选）
    ag.activation.restore({scope: "ldg-taishan"})
    ag.insert_ledger_block([_SYS, {"role": "user", "content": "小黑，梳理泰山啤酒破产重整最新进展"}],
                           "hermes:default", tier="medium", with_instruction=True)
    with structlog.testing.capture_logs() as cap:
        _update(ag, "hermes:default")
    assert not [e for e in cap if e.get("event") == "ledger_update_off_candidate"]
    # b) 无候选命中 ⇒ 不响
    ag.activation.restore({scope: "ldg-new3"})
    ag.insert_ledger_block([_SYS, {"role": "user", "content": "写一个 python 脚本统计文件数"}],
                           "hermes:default", tier="medium", with_instruction=True)
    with structlog.testing.capture_logs() as cap:
        _update(ag, "hermes:default")
    assert not [e for e in cap if e.get("event") == "ledger_update_off_candidate"]


# ── ⑦ MQ-L46：池里混 `+00:00` 与 `+08:00` 两本，recency 顺序按真实时刻 ────────

def test_recent_list_orders_by_instant_not_string(monkeypatch):
    """`2026-09-05T01:00:00+08:00`（= 04 日 17:00Z）比 `2026-09-05T00:30:00+00:00` 早；
    字符串序会反过来（修前形态）。"""
    _on(monkeypatch)
    ag = AgencyRuntime()
    ag.pool["ldg-utc"] = new_ledger(title="utc 本", goal="x", ledger_id="ldg-utc",
                                    created_at="2026-09-05T00:30:00+00:00")
    ag.pool["ldg-cst"] = new_ledger(title="cst 本", goal="y", ledger_id="ldg-cst",
                                    created_at="2026-09-05T01:00:00+08:00")
    msgs = [_SYS, {"role": "user", "content": "完全无关的话题 zzz"}]
    out = ag.insert_ledger_block(msgs, "hermes:default", tier="medium", with_instruction=True)
    block = out[-1]["content"]
    recent = block[block.index("Other recent ledgers"):]
    assert recent.index("ldg-utc") < recent.index("ldg-cst"), recent
    # 判别力对照：改回比字符串必须变红
    assert sorted(["2026-09-05T00:30:00+00:00", "2026-09-05T01:00:00+08:00"], reverse=True)[0] \
        == "2026-09-05T01:00:00+08:00"
