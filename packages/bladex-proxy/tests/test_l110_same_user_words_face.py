"""批 L110 · MQ-L110：同一句用户话只判一次用户面，会话 id 变了也一样。

病例（11-03 hermes:accept，`report-task-l109-20261105.md` §2）：第一个请求只有一条非 system 消息，
会话落 `tw:*`；工具往返时前缀够两条，会话变 `fp:*`。判定的键含 `session_id`，`_prev` 为空 ⇒
第二次用户面 ⇒ "FIRST STEP … If it is a DIFFERENT task" 重发，模型推翻第一次「留下」的判断。

夹具 `fixtures/l110_face_shapes.json` 是 Hub 真实轮次的形状：角色、`tool_calls` 有无、用户话的内容哈希，
不含正文（生成脚本 `docs/benchmarks/l110/probe_l110_face_shapes.py`）。
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
import structlog.testing
from bladex_core.ledger import new_ledger
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.agency.runtime import _USER_WORDS_SEEN_MAX

_FIXTURE = Path(__file__).parent / "fixtures" / "l110_face_shapes.json.gz"
_SWITCH = "is it a DIFFERENT task"   # 批 SW · 步骤 3：S1 比对半句改写后仍含的片段（仍只在用户面出现）
_AGENT = "hermes:accept"

_SYS = {"role": "system", "content": "s"}
_WORDS = {"role": "user", "content": "分析一下这个市场"}
_CALL = {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}
_TOOL = {"role": "tool", "tool_call_id": "c1", "content": "ok"}
_TW = "tw:local:hermes:accept:5978865.798736a0"
_FP = "fp:681dcd4c9cbc"


@pytest.fixture(autouse=True)
def _modules(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")


def _inject(ag: AgencyRuntime, msgs: list[dict], session: str, agent: str = _AGENT,
            project: str = "") -> tuple[str, dict]:
    with structlog.testing.capture_logs() as cap:
        out = ag.insert_ledger_block(msgs, agent, tier="medium", with_instruction=True,
                                     session_id=session, project_id=project)
    row = next(e for e in cap if e.get("event") == "agency_ledger_block_injected")
    return out[-1]["content"], row


def _seeded(agent: str = _AGENT, project: str = "") -> AgencyRuntime:
    ag = AgencyRuntime()
    led = new_ledger(title="t", goal="g", ledger_id="ldg-l110000001",
                     created_at="2026-11-03T06:00:00+00:00")
    ag.pool[led.ledger_id] = led
    ag.activation.restore({ag.scope_of(agent, project): led.ledger_id})
    return ag


def _shapes() -> dict:
    with gzip.open(_FIXTURE, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _rebuild(shape: list[list]) -> list[dict]:
    """夹具的一轮 → 消息数组。用户话正文用内容哈希占位：同哈希 = 同一句话。"""
    out: list[dict] = []
    for role, payload in shape:
        if role == "assistant":
            out.append({"role": "assistant", "content": "",
                        **({"tool_calls": [{"id": "c"}]} if payload else {})})
        elif role == "user":
            out.append({"role": "user", "content": f"words:{payload}"})
        else:
            out.append({"role": role, "content": ""})
    return out


# ── 病例回放 ──────────────────────────────────────────────────────────────────

def test_case_replay_second_request_is_tool_face() -> None:
    case = _shapes()["case_20261103"]
    assert [t["session"].split(":")[0] for t in case] == ["tw", "fp"], "夹具应是 tw → fp"
    ag = _seeded()
    first, r1 = _inject(ag, _rebuild(case[0]["messages"]), case[0]["session"])
    second, r2 = _inject(ag, _rebuild(case[1]["messages"]), case[1]["session"])
    assert (r1["face"], r2["face"]) == ("user", "tool")
    assert r2["face_last_role"] == "tool"
    assert r2["instruction_kind"] == "record"
    assert _SWITCH in first and _SWITCH not in second, "切换指令不重发"


def test_case_shape_used_the_same_user_words() -> None:
    """夹具自检：两个请求里开头那句话确实同哈希，否则上面那条测的不是这个病例。"""
    case = _shapes()["case_20261103"]
    first_user = [p for r, p in case[0]["messages"] if r == "user"]
    second_user = [p for r, p in case[1]["messages"] if r == "user"]
    assert first_user == second_user and len(first_user) == 1


# ── 范围回放：L109 列出的每一次 ────────────────────────────────────────────────

def _replay_rows(*, fixed: bool) -> list[dict]:
    """L109 列的 hermes 系 / Pi 是本批要修的；codex / dsh 是另一种成因，单独断言不受影响。"""
    return [r for r in _shapes()["range"]
            if r["agent"].startswith(("hermes", "Pi")) == fixed]


def _faces(row: dict) -> list[dict]:
    ag = AgencyRuntime()
    return [_inject(ag, _rebuild(t["messages"]), t["session"], agent=row["agent"],
                    project=t["project"])[1] for t in row["turns"]]


def test_range_fixture_covers_what_l109_listed() -> None:
    """夹具自检：hermes:accept 3 + hermes:default 6 + Pi 3，codex 2，缺了就不算范围回放。"""
    rows = _shapes()["range"]
    count = {a: sum(r["agent"] == a for r in rows)
             for a in ("hermes:accept", "hermes:default", "Pi", "codex")}
    assert count == {"hermes:accept": 3, "hermes:default": 6, "Pi": 3, "codex": 2}


@pytest.mark.parametrize("row", _replay_rows(fixed=True), ids=lambda r: f"{r['agent']}-{r['log_ts']}")
def test_range_replay_no_user_face_on_tool_tail(row: dict) -> None:
    """同段内逐请求回放，修后不再出现「末条是 tool 却判 user」。"""
    faces = _faces(row)
    assert len(faces) >= 2, "至少要有开局请求和被误判的那个请求"
    assert all(not (r["face"] == "user" and r["face_last_role"] == "tool") for r in faces)
    assert faces[0]["face"] == "user", "这句话的第一个请求仍是用户面"


@pytest.mark.parametrize("row", _replay_rows(fixed=False), ids=lambda r: f"{r['agent']}-{r['log_ts']}")
def test_other_causes_are_not_changed_by_this_fix(row: dict) -> None:
    """codex 09-29、dsh 10-16 不是 `tw→fp`，本批不修：全量回放的修前修后判定相同。"""
    assert (row["old"], row["new"]) == ("user", "user")


# ── 真正的新用户轮 ────────────────────────────────────────────────────────────

def test_new_words_in_the_same_session_are_user_face() -> None:
    ag = _seeded()
    _inject(ag, [_SYS, _WORDS], _TW)
    msgs = [_SYS, _WORDS, _CALL, _TOOL]
    assert _inject(ag, msgs, _FP)[1]["face"] == "tool"
    msgs = [*msgs, {"role": "assistant", "content": "好了"},
            {"role": "user", "content": "再看另一件事"}]
    block, row = _inject(ag, msgs, _FP)
    assert row["face"] == "user" and _SWITCH in block, "指令照发"


def test_same_words_again_as_a_new_turn_are_user_face() -> None:
    """用户把同一句话再说一遍是新的一轮，哈希相同也不算。"""
    ag = _seeded()
    msgs = [_SYS, _WORDS]
    _inject(ag, msgs, _TW)
    msgs = [*msgs, {"role": "assistant", "content": "好了"}, _WORDS]
    assert _inject(ag, msgs, _FP)[1]["face"] == "user"


def test_same_words_in_a_new_conversation_are_user_face() -> None:
    """🔴 另一个会话说一模一样的话，第一个请求里没有回答 ⇒ 仍是用户面。"""
    ag = _seeded()
    _inject(ag, [_SYS, _WORDS], _TW)
    _inject(ag, [_SYS, _WORDS, _CALL, _TOOL], _FP)
    block, row = _inject(ag, [_SYS, _WORDS], "tw:local:hermes:accept:5979255.95785fd4")
    assert row["face"] == "user" and _SWITCH in block


def test_other_scope_does_not_share_user_words() -> None:
    """另一个作用域（别的 agent）见过同一句话，不影响这里。"""
    ag = _seeded()
    other = "hermes:default"
    ag.activation.restore({ag.scope_of(_AGENT): "ldg-l110000001",
                           ag.scope_of(other): "ldg-l110000001"})
    _inject(ag, [_SYS, _WORDS], _TW)
    row = _inject(ag, [_SYS, _WORDS, _CALL, _TOOL], _FP, agent=other)[1]
    assert row["face"] == "user"


def test_answered_words_never_seen_stay_user_face() -> None:
    """进程重启后第一个请求：状态清零、这句话没见过 ⇒ 修前行为不变（只会多注一次，不漏注）。"""
    ag = _seeded()
    assert _inject(ag, [_SYS, _WORDS, _CALL, _TOOL], _FP)[1]["face"] == "user"


def test_gated_first_request_does_not_consume_the_user_face() -> None:
    """第一个请求被档位门挡下（模型没看到块），第二个请求才是这句话第一次判面。"""
    ag = _seeded()
    with structlog.testing.capture_logs() as cap:
        out = ag.insert_ledger_block([_SYS, _WORDS], _AGENT, tier="weak", with_instruction=True,
                                     session_id=_TW)
    assert any(e.get("event") == "agency_ledger_block_gated" for e in cap)
    assert out == [_SYS, _WORDS]
    assert _inject(ag, [_SYS, _WORDS, _CALL, _TOOL], _FP)[1]["face"] == "user"


def test_retry_after_the_session_changed_keeps_tool_face() -> None:
    ag = _seeded()
    msgs = [_SYS, _WORDS, _CALL, _TOOL]
    _inject(ag, [_SYS, _WORDS], _TW)
    assert _inject(ag, msgs, _FP)[1]["face"] == "tool"
    assert _inject(ag, msgs, _FP)[1]["face"] == "tool", "逐字重试沿用"


def test_activation_reading_is_logged() -> None:
    """上线后的激活读数（铁律 9）：只有真正跨会话判 tool 时才打。"""
    ag = _seeded()
    with structlog.testing.capture_logs() as cap:
        ag.insert_ledger_block([_SYS, _WORDS], _AGENT, tier="medium", session_id=_TW)
        ag.insert_ledger_block([_SYS, _WORDS, _CALL, _TOOL], _AGENT, tier="medium", session_id=_FP)
        ag.insert_ledger_block([_SYS, _WORDS, _CALL, _TOOL, _CALL, _TOOL], _AGENT, tier="medium",
                               session_id=_FP)
    rows = [e for e in cap if e.get("event") == "agency_face_same_user_words"]
    assert len(rows) == 1 and rows[0]["session"] == _FP


def test_seen_table_is_bounded() -> None:
    ag = _seeded()
    for i in range(_USER_WORDS_SEEN_MAX + 5):
        _inject(ag, [_SYS, {"role": "user", "content": f"第 {i} 句"}], f"s-{i}")
    assert len(ag._user_words_seen) == _USER_WORDS_SEEN_MAX


# ── 无激活本的开局（L109 §4 A 行、MQ-L79）────────────────────────────────────

def test_no_active_ledger_tool_face_keeps_first_step_and_candidates() -> None:
    """无激活本时，修后的工具面仍有首步指令与相关性段，只少 recency 段。"""
    ag = AgencyRuntime()
    words = {"role": "user", "content": "之前 Mac mini 和 Mac Studio 的性价比分析怎么样了？"}
    other = new_ledger(title="苹果新 Mac mini / Mac Studio 亮点与性价比分析",
                       goal="分析苹果最新发布的 Mac mini 与 Mac Studio：亮点与性价比。",
                       ledger_id="ldg-l110other01", goal_source="user",
                       created_at="2026-11-02T06:00:00+00:00")
    ag.pool[other.ledger_id] = other
    # 相关性分按池内 idf 算，池里只有一本时恒为 0
    for lid, title in (("ldg-l110other02", "status 命令按成因报错"),
                       ("ldg-l110other03", "4 independent bug-fix tasks")):
        ag.pool[lid] = new_ledger(title=title, goal=title, ledger_id=lid, goal_source="user",
                                  created_at="2026-11-01T06:00:00+00:00")
    u_block, u = _inject(ag, [_SYS, words], _TW)
    t_block, t = _inject(ag, [_SYS, words, _CALL, _TOOL], _FP)
    assert "Other recent ledgers" in u_block, "用户面才有 recency 段"
    assert (u["face"], t["face"]) == ("user", "tool")
    assert "No active task ledger" in t_block and "FIRST STEP" in t_block
    assert other.ledger_id in t_block, "相关性段仍在"
    assert "Other recent ledgers" not in t_block


# ── 零行为差异 ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["claude-code", "codex", "dsh"])
def test_real_shape_blocks_are_identical_to_before(name: str, monkeypatch) -> None:
    """真实形状逐请求回放，修后的块与修前判定渲染出的块逐字相同。"""
    seg = _shapes()["zero_diff"][name]
    assert len(seg) >= 3, "夹具至少要有工具往返"

    def _old_judge(self: AgencyRuntime, scope: str, session_id: str, messages: list[dict]) -> bool:
        from bladex_core.ledger_runtime import user_turn_index
        uti = user_turn_index(messages)
        key = (scope, session_id or "")
        shape = (uti, len(messages or ()))
        prev = self._last_seen_face_turn.get(key)
        if prev is not None and prev[0] == shape:
            new = prev[1]
        else:
            new = (prev is None) or (prev[0][0] != uti)
        self._last_seen_face_turn[key] = (shape, new)
        return new

    new_ag = _seeded(name)
    new_blocks = [_inject(new_ag, _rebuild(t["messages"]), t["session"], agent=name)[0] for t in seg]
    assert any("FIRST STEP" in b for b in new_blocks) or "EVERY TURN" in new_blocks[-1]
    monkeypatch.setattr(AgencyRuntime, "_judge_new_user_turn", _old_judge)
    old_ag = _seeded(name)
    old_blocks = [_inject(old_ag, _rebuild(t["messages"]), t["session"], agent=name)[0] for t in seg]
    assert new_blocks == old_blocks


# ── 单一实现点 ────────────────────────────────────────────────────────────────

def test_face_state_has_a_single_implementation_point() -> None:
    from tests._source_probe import package_source

    src = package_source("agency")
    assert src.count("def _judge_new_user_turn(") == 1
    assert src.count("self._judge_new_user_turn(") == 1, "只有 insert_ledger_block 一个调用点"
    # 初始化 + 判定方法里一读一写，共三处
    assert src.count("self._last_seen_face_turn") == 3
    # 初始化 + 判定方法里 查 / 写 / 移到末尾 / 取长度 / 淘汰
    assert src.count("self._user_words_seen") == 6
