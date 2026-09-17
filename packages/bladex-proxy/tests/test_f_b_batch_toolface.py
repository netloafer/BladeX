"""F-B1 的 proxy 半边：schema ⇄ 运行时对账 + handler 侧"整批一次 rev / 一条事件"。

运行时语义（原子性 / match / 老回文）在 `packages/bladex-core/tests/test_f_b_batch_update.py`。
本文件只管两件 proxy 才有的事：

1. **schema 与实现是同一个闭集**（op 枚举从 `UPDATE_OPS` 读，不在测试里默写）
   与 **老参数集合 ⊆ 新参数集合**（红线 4：批量是新增，不是换调用面）；
2. **一次调用 = 一次 `rev+1` + 一条 `LEDGER_UPDATE` 事件**。逐条发事件会让
   rev 与事件数对不上，而重放是状态式（payload = 整本 dump）——多发的那几条
   是纯噪声，且会让 `probe_ledger_trajectory` 的"更新事件数"读数虚高。
"""

from __future__ import annotations

import ast
import asyncio
import pathlib

import pytest

from bladex_core.ledger import ACTOR_MODEL, Ledger, LedgerEntry, add_entry
from bladex_core.ledger_runtime import UPDATE_OPS, activation_scope
from bladex_proxy.agency import AgencyRuntime
from bladex_proxy.toolface import TOOL_SCHEMAS

_TOOLFACE_SRC = (pathlib.Path(__file__).resolve().parents[1]
                 / "bladex_proxy" / "toolface.py")


def _update_schema() -> dict:
    return next(t["function"] for t in TOOL_SCHEMAS
                if t["function"]["name"] == "bladex_ledger_update")


# ── schema ⇄ 运行时 ─────────────────────────────────────────────────────────

class TestSchemaRuntimeParity:
    def test_op_enums_come_from_the_single_definition(self):
        """`toolface` 里每一处 op 枚举都必须逐字等于 `UPDATE_OPS`。

        分叉的两个方向都坏：schema 多一个 ⇒ 模型拿到运行时不认的 op；
        运行时多一个 ⇒ 有能力却没写进 schema，永远不会被调用。
        """
        tree = ast.parse(_TOOLFACE_SRC.read_text(encoding="utf-8"))
        found: list[list[str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            if "enum" not in keys:
                continue
            enum = node.values[keys.index("enum")]
            if isinstance(enum, ast.List):
                vals = [e.value for e in enum.elts if isinstance(e, ast.Constant)]
                if set(vals) & set(UPDATE_OPS):
                    found.append(vals)
        assert len(found) >= 2, \
            "顶层与 entries 里各应有一处 op 枚举——找不到就是本守卫失去了参照系"
        for vals in found:
            assert vals == list(UPDATE_OPS), \
                f"schema op 枚举 {vals} 与 UPDATE_OPS {list(UPDATE_OPS)} 分叉"

    def test_batch_is_additive_old_params_all_survive(self):
        props = _update_schema()["parameters"]["properties"]
        legacy = {"section", "op", "text", "index", "ref", "rev",
                  "goal", "goal_change_quote"}
        assert legacy <= set(props), f"老参数被删了：{sorted(legacy - set(props))}"
        assert {"entries", "match"} <= set(props), "新参数没接上"

    def test_entries_items_carry_the_five_fields(self):
        items = _update_schema()["parameters"]["properties"]["entries"]["items"]
        # 🔴 F1.4 / MQ-L53：`section` 出 required（`op=remove` + `match` 时它是冗余；
        # 两个互不相干的 agent 各自漏掉它 ⇒ 接口的问题）。二选一的校验在
        # `_validate_spec`，对照见 `test_f1_remove_without_section.py`。
        assert set(items["required"]) == {"op"}
        assert {"section", "op", "text", "ref", "match", "index"} \
            <= set(items["properties"])

    def test_sections_are_the_same_four_in_both_places(self):
        props = _update_schema()["parameters"]["properties"]
        items = props["entries"]["items"]["properties"]
        assert items["section"]["enum"] == props["section"]["enum"], \
            "顶层与 entries 的段枚举分叉 ⇒ 模型按批量写会被运行时拒"

    def test_top_level_description_tells_it_to_batch(self):
        """判据取**性质**不取整句（措辞随 prompt 调优可变）：说清"合成一次调用"
        与"优先 match"两件事。写在顶层 description 是因为遗忘发生在**决定调不调
        这个工具**的那一刻（MQ-L29 同款理由）。"""
        low = _update_schema()["description"].lower()
        assert "entries" in low and ("batch" in low or "one call" in low), \
            "顶层没说「把相关改动合成一次调用」—— MQ-L49 的成本曲线不会自己弯"
        assert "match" in low and "index" in low, "没说优先 match 而不是 index"

    def test_tool_face_stays_within_attention_budget(self):
        """加参数不是免费的（ADR-0029）。与 `test_toolface` 同一把尺、同一上限。"""
        import json
        total = sum(len(json.dumps(t, ensure_ascii=False)) for t in TOOL_SCHEMAS)
        assert total <= 8000, f"工具面 schema 总长 {total} 超预算 8000"


# ── handler：一次 rev / 一条事件 ────────────────────────────────────────────

class _Rt(AgencyRuntime):
    """记下 `_emit` 的调用（AgencyRuntime 无 hub 时 `_emit` 是空操作）。"""

    def __init__(self):
        super().__init__(index=None, hub=None)
        self.emitted: list[tuple] = []

    def _emit(self, etype, payload):   # noqa: D401
        self.emitted.append((etype, payload))


@pytest.fixture(autouse=True)
def _modules_on(monkeypatch):
    monkeypatch.setenv("BLADEX_MODULE_LEDGER", "1")
    monkeypatch.setenv("BLADEX_MODULE_TOOLFACE", "1")


def _rt_with_ledger() -> tuple[_Rt, Ledger]:
    led = Ledger(ledger_id="ldg-fb1proxy", title="批量", rev=4)
    led = add_entry(led, "next", LedgerEntry(text="删掉我", source="model"),
                    actor=ACTOR_MODEL)
    rt = _Rt()
    rt.pool[led.ledger_id] = led
    rt.activation.restore({activation_scope("codex", ""): led.ledger_id})
    return rt, led


def _update(rt, args):
    return asyncio.run(rt._h_ledger_update(
        args, allowed_exposure="local",
        context={"agent_id": "codex", "project_id": "", "session_id": "s1"}))


class TestBatchLandsOnce:
    def test_three_edits_bump_rev_once_and_emit_one_event(self):
        rt, led = _rt_with_ledger()
        out = _update(rt, {"rev": 4, "entries": [
            {"section": "verified", "op": "add", "text": "A"},
            {"section": "core", "op": "add", "text": "B"},
            {"section": "next", "op": "remove", "match": "删掉我"},
        ]})
        assert out.count("\n") == 2, out
        new = rt.pool[led.ledger_id]
        assert new.rev == 5, "整批一次 +1（逐条 +1 会让 rev 与模型手上的视图对不上）"
        assert len(rt.emitted) == 1, "整批一条 LEDGER_UPDATE（重放是状态式）"
        assert rt.emitted[0][1]["ledger"]["rev"] == 5
        assert [e["text"] for e in rt.emitted[0][1]["ledger"]["sections"]["verified"]] \
            == ["A"]

    def test_failed_batch_leaves_no_event_and_no_rev_bump(self):
        rt, led = _rt_with_ledger()
        out = _update(rt, {"rev": 4, "entries": [
            {"section": "verified", "op": "add", "text": "A"},
            {"section": "next", "op": "remove", "index": 99},
        ]})
        assert out.startswith("Error:"), out
        assert rt.pool[led.ledger_id].rev == 4 and rt.emitted == [], \
            "整批不落 = 池不动 + 事件不发（半本账本比不落更糟）"

    def test_rejection_is_logged(self, monkeypatch):
        """🔴 09-08 事故（MQ-L52）：失败路径此前**零埋点**。hermes 连撞 6 次
        `entries[5] needs both section and op`、`rounds=6 degraded=True`、
        62 万 token，而 proxy 日志一片空白——全靠 Jason 在 agent 终端肉眼看见。

        成功有 `agency_ledger_update_applied`、失败什么都没有 ⇒ "模型反复撞同一个
        错"在读数上**不存在**（MQ-L17：没人看得见的机制等于没有）。
        """
        seen: dict = {}

        import bladex_proxy.agency.handlers as H

        class _Log:
            def warning(self, event, **kw):
                if event == "agency_ledger_update_rejected":
                    seen.update(kw)

            def info(self, event, **kw):
                pass

        monkeypatch.setattr(H, "logger", _Log())   # 打消费方子模块
        rt, led = _rt_with_ledger()
        _update(rt, {"rev": 4, "entries": [
            {"section": "verified", "op": "add", "text": "A"},
            {"match": "漏了 section 和 op"},
        ]})
        assert seen, "整批被拒却零日志 ⇒ 这类事故只能靠人肉眼发现"
        assert seen["ledger"] == led.ledger_id and seen["agent"] == "codex"
        assert seen["n_edits"] == 2, "要能数出模型一次想改几条"
        assert "entries[1]" in seen["error"], seen["error"]

    def test_long_error_is_truncated_with_an_explicit_mark(self, monkeypatch):
        """🔴 刚性原则 13：截断必须**带显式标记**。

        首版写的是 `str(e)[:200]` —— 静默截断，与 MQ-A22 那次事故（header 快照
        200 字符静默切断 JSON）**是同一个数字上的同一个错**。live 09-08 01:38:53
        那条 `error=` 恰好断在 "or drop it and r"，读日志的人无从判断后面还有没有内容。
        """
        from bladex_proxy.agency.handlers import TRUNCATED_MARK, _clip

        short = "x" * 200
        assert _clip(short) == short and TRUNCATED_MARK not in _clip(short), \
            "没超限不该加标记"
        long = "y" * 201
        out = _clip(long)
        assert out.endswith(TRUNCATED_MARK), "超限必须带标记，静默丢数据比丢数据更糟"
        assert out[:200] == long[:200]

    def test_a_real_long_rejection_carries_the_mark(self, monkeypatch):
        """接线半边：真跑一次长错误，日志字段上要看得见标记。"""
        seen: dict = {}

        import bladex_proxy.agency.handlers as H

        class _Log:
            def warning(self, event, **kw):
                if event == "agency_ledger_update_rejected":
                    seen.update(kw)

            def info(self, event, **kw):
                pass

        monkeypatch.setattr(H, "logger", _Log())
        rt, _ = _rt_with_ledger()
        # 段名很长 ⇒ 报错文本必然超过 200 字符（错误里会列出本账本所有段名）
        _update(rt, {"rev": 4, "entries": [
            {"section": "verified", "op": "add", "text": "A"},
            {"section": "x" * 300, "op": "add", "text": "B"},
        ]})
        assert seen, "长错误也要有埋点"
        assert seen["error"].endswith(H.TRUNCATED_MARK), seen["error"][-60:]

    def test_successful_batch_is_not_counted_as_rejected(self, monkeypatch):
        """阴性对照：成功不许进拒绝桶（否则分子被灌水、比率读反）。"""
        seen: list = []

        import bladex_proxy.agency.handlers as H

        class _Log:
            def warning(self, event, **kw):
                if event == "agency_ledger_update_rejected":
                    seen.append(kw)

            def info(self, event, **kw):
                pass

        monkeypatch.setattr(H, "logger", _Log())
        rt, _ = _rt_with_ledger()
        _update(rt, {"rev": 4, "entries": [
            {"section": "verified", "op": "add", "text": "A"}]})
        assert seen == []

    def test_single_form_still_works_end_to_end(self):
        rt, led = _rt_with_ledger()
        out = _update(rt, {"section": "verified", "op": "add", "text": "单条",
                           "rev": 4})
        assert out == "added to verified: 单条"
        assert rt.pool[led.ledger_id].rev == 5 and len(rt.emitted) == 1
