"""批 U · U1b（MQ-I29）—— Memory Index 的 unit_key 按现行轮次规则从原文重算，不读存量 task_units。

钉四件事：
① `task_unit.index_unit_key`：原文完整 ⇒ 按 `build_task_units` 重算（source=recomputed），与存下的旧口径
   key 不同时以重算为准；原文缺失 / `content_ref` 未还原 ⇒ 沿用存下的 key 加 `legacy:` 前缀
   （source=legacy_stored）；两样都没有 ⇒ ""。
② 单实现点守卫：`memory_index.py` 的代码里不许出现任何 `task_units` 读法，unit_key 只经 `_index_unit_key(turn)` 取。
③ 真重建路径：`rebuild_from_hub(full=True)` 扫描出的每一轮 `ConversationTurn.unit_key` 都等于按原文重算的值；
   蒸出的 fact 继承同一个 key；来源计数进 `_last_unit_key_sources`。
   阴性对照：把 memory_index 改回读存量 task_units ⇒ 本条红（见结算稿 U1b 节的反向变异）。
④ 算不出来的轮次在重建里有标记（`legacy:` 前缀）、有计数，不静默。
"""

from __future__ import annotations

import ast
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from bladex_core.task_unit import (
    LEGACY_UNIT_KEY_PREFIX,
    TaskUnit,
    build_task_units,
    index_unit_key,
    make_unit_key,
)
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage import memory_index as mi
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex

# 工具循环里插了一句话：旧口径（每条 user 一单元）的 open 单元是 "q2"，§2.2 口径也是 "q2"；
# 但若末条 user 是插话，旧口径 open 单元 = 插话，新口径 = 本轮首条。
_MSGS = [{"role": "user", "content": "做一件事：整理 README"},
         {"role": "assistant", "content": "", "tool_calls": [
             {"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
         {"role": "tool", "tool_call_id": "c1", "content": "r"},
         {"role": "user", "content": "顺便把拼写错误也改了"}]
_NEW_KEY = make_unit_key("做一件事：整理 README")
_OLD_KEY = make_unit_key("顺便把拼写错误也改了")      # 旧口径：插话自成 open 单元


def _turn(msgs, stored_key: str = "") -> Turn:
    units = [TaskUnit(unit_key=stored_key)] if stored_key else []
    return Turn(identity=Identity(user_id="u1", agent_id="claude-code", session_id="s1"),
                model="m", request_messages=msgs, response_text="ok", status=TurnStatus.OK,
                task_units=units)


# ── ① 取法 ────────────────────────────────────────────────────────────────


def test_recomputes_from_messages_not_stored_units():
    assert build_task_units(_MSGS)[-1].unit_key == _NEW_KEY
    assert index_unit_key(_turn(_MSGS, stored_key=_OLD_KEY)) == (_NEW_KEY, "recomputed")


def test_missing_messages_fall_back_to_marked_stored_key():
    assert index_unit_key(_turn([], stored_key=_OLD_KEY)) == (
        LEGACY_UNIT_KEY_PREFIX + _OLD_KEY, "legacy_stored")


def test_unresolved_content_ref_falls_back_to_marked_stored_key():
    msgs = [{"role": "user", "content_ref": "deadbeef"}]
    key, src = index_unit_key(_turn(msgs, stored_key=_OLD_KEY))
    assert src == "legacy_stored" and key.startswith(LEGACY_UNIT_KEY_PREFIX)
    assert key != _OLD_KEY, "旧口径 key 不许以新口径的样子混进库"


def test_nothing_to_go_on_is_empty():
    assert index_unit_key(_turn([])) == ("", "none")


# ── ② 单实现点守卫 ────────────────────────────────────────────────────────


def _code_without_comments_and_docstrings(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            body[0] = ast.Pass()
    return ast.unparse(tree)


def test_memory_index_has_no_task_units_reader():
    code = _code_without_comments_and_docstrings(Path(mi.__file__))
    assert "task_units" not in code, \
        "memory_index.py 又出现了 task_units 读法——unit_key 只许经 task_unit.index_unit_key 取（MQ-I29）"
    assert len(re.findall(r"_index_unit_key\(turn\)", code)) == 1


# ── ③ ④ 真重建路径 ────────────────────────────────────────────────────────


class _Embedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 64
            v[hash(t) % 64] = 1.0
            out.append(v)
        return out

    @property
    def available(self):
        return True


def test_full_rebuild_assigns_recomputed_unit_key_to_every_turn(monkeypatch):
    seen: list[tuple[str, str]] = []
    real = mi.ConversationTurn

    def _spy(*a, **kw):
        ct = real(*a, **kw)
        seen.append((ct.ledger_key, ct.unit_key))
        return ct

    monkeypatch.setattr(mi, "ConversationTurn", _spy)
    with tempfile.TemporaryDirectory() as tmp:
        hub = MemoryHub(Path(tmp) / "hub")
        hub.open()
        ident = Identity(user_id="u1", agent_id="claude-code", session_id="s1")
        cases = {
            # 可重算：存的是旧口径 key，重建必须换成新口径
            ident.storage_key("1719900001-0"): (_MSGS, _OLD_KEY, _NEW_KEY),
            # 算不出来：原文缺失 ⇒ legacy 前缀
            ident.storage_key("1719900002-0"): ([], _OLD_KEY, LEGACY_UNIT_KEY_PREFIX + _OLD_KEY),
        }
        for i, (key, (msgs, stored, _want)) in enumerate(cases.items()):
            t = _turn(msgs, stored_key=stored)
            t.ts = datetime(2026, 9, 20, 12, 0, i, tzinfo=UTC)
            hub.put(key, t)
        index = MemoryIndex(Path(tmp) / "idx", embedder=_Embedder(), read_only=False)
        index.open()
        index.rebuild_from_hub(hub, full=True)
        got = dict(seen)
        for key, (_m, _s, want) in cases.items():
            assert got.get(key) == want, (key, got.get(key), want)
        assert index._last_unit_key_sources == {"recomputed": 1, "legacy_stored": 1}
        for f in index.all_facts():
            src = getattr(f, "source_ledger_key", "") or ""
            if src in cases and f.unit_key:
                assert f.unit_key == cases[src][2], "fact 继承的 unit_key 必须是重算值"
        index.close()
        hub.close()
