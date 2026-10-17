"""MQ-CA14 守卫（批 R · R0，2026-09-18，拍板 (11)）：L2 摘要块**不许悄悄复活**。

## 这张表钉什么

CA9 清账删了 L2 摘要的**生产者**，漏了**消费侧**：`inject.py` 仍在装配结果里找
`SUMMARY_OPEN` 反推 `summary_hash` / `kept_recent_n`。全仓无人再产生该块 ⇒ 那段反推
结构性死亡，两个键恒默认值——而静态守卫（代码还在、语法正确）与
`probe_turn_field_coverage.py`（不进 `Turn.cap` 内嵌 dict）都看不见它（刚性原则 13 层 3）。

处置是**退役**（删反推段 + `cap_info` 不再声明两键），不是"保留恒默认值"。四条判据：

① `cap_info` 键集不含 `summary_hash` / `kept_recent_n`（两条路径：assembler=None 与 agent 未启用）；
② 🔴 **源码守卫：`SUMMARY_OPEN` 的生产者数 = 0** —— 生产代码里该常量只允许三种形态：
   `assembly.py` 的定义、`inject.py` 的 import / 剥离正则 / 读侧"入参里有没有残留块"判断
   （剥的是**入参**，不是 `assembled`）。任何第四处（含把反推段加回去）即红。
   这条比①值钱：①只能看见键，②钉的是"没有人能再生产这个块"。
③ 存量不受影响：`DecisionMeta.cap` 是 dict，带旧键的历史 Turn 反序列化后旧值原样读回；
④ 阴性对照：`_negative_control_lines` 里那两行（就是被删的反推段原文）喂给判别函数必须报。

## 为什么按"行的精确形态"白名单，而不是数出现次数

被删的那段写的是 `SUMMARY_OPEN in c`，与 `inject.py:305` 合法的 `SUMMARY_OPEN in content`
形态几乎相同——只数次数抓得到"多了一处"，抓不到"换了一个变量名再加回来"是不是同一件事。
白名单精确到整行（strip 后），注释行不计（注释不能生产任何东西）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_PROXY_PKG = _REPO / "packages" / "bladex-proxy" / "bladex_proxy"
_CORE_PKG = _REPO / "packages" / "bladex-core" / "bladex_core"
_SCRIPTS = _REPO / "scripts"
_TOKEN = "SUMMARY_" + "OPEN"  # 拆开构造：守卫不该抓到自己的 docstring

#: 生产代码里允许出现该常量的**整行**（strip 后精确匹配）。第四种形态 = 红。
_ALLOWED_LINES: dict[str, frozenset[str]] = {
    "assembly.py": frozenset({'SUMMARY_OPEN = "<bladex-context-summary>"'}),
    "inject.py": frozenset({
        "SUMMARY_OPEN,",                                       # from assembly import (...)
        r'rf"\s*{re.escape(SUMMARY_OPEN)}.*?{re.escape(SUMMARY_CLOSE)}\s*",',  # 剥离正则
        "if isinstance(content, str) and (MEMORY_OPEN in content or SUMMARY_OPEN in content):",  # 读侧：入参残留
    }),
}

#: 被删的反推段原文（阴性对照④的输入）——它们必须被判为"生产者/消费者复活"。
_negative_control_lines = (
    "if isinstance(c, str) and SUMMARY_OPEN in c:",
    'cap_info["summary_hash"] = hashlib.sha256(c.encode()).hexdigest()[:16]',
)


def _code_lines_with_token(src: str) -> list[str]:
    """源码里含该常量的**代码行**（strip 后），跳过注释行与 docstring 行。"""
    out: list[str] = []
    doc_ranges: list[tuple[int, int]] = []
    try:
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ds = ast.get_docstring(node, clean=False)
                if ds and node.body and isinstance(node.body[0], ast.Expr):
                    e = node.body[0]
                    doc_ranges.append((e.lineno, e.end_lineno or e.lineno))
    except SyntaxError:
        pass
    for i, line in enumerate(src.splitlines(), 1):
        if _TOKEN not in line:
            continue
        st = line.strip()
        if st.startswith("#"):
            continue
        if any(a <= i <= b for a, b in doc_ranges):
            continue
        out.append(st)
    return out


#: 退役键的字面量：生产代码里**任何**代码行出现即红（写它 = 消费侧复活）。
_RETIRED_KEY_LITERALS = ('"summary_' + 'hash"', '"kept_' + 'recent_n"')


def _code_lines_with_retired_keys(src: str) -> list[str]:
    out: list[str] = []
    for line in src.splitlines():
        st = line.strip()
        if st.startswith("#") or not any(k in st for k in _RETIRED_KEY_LITERALS):
            continue
        out.append(st)
    return out


def offenses(files: dict[str, str]) -> list[tuple[str, str]]:
    """{文件名: 源码} → [(文件名, 违规行)]。纯函数，④直接喂它。"""
    bad: list[tuple[str, str]] = []
    for name, src in files.items():
        allowed = _ALLOWED_LINES.get(name, frozenset())
        for st in _code_lines_with_token(src):
            if st not in allowed:
                bad.append((name, st))
        for st in _code_lines_with_retired_keys(src):
            bad.append((name, st))
    return bad


def _production_sources() -> dict[str, str]:
    files: dict[str, str] = {}
    for root in (_PROXY_PKG, _CORE_PKG, _SCRIPTS):
        for p in root.rglob("*.py"):
            files[str(p.relative_to(_REPO))] = p.read_text(encoding="utf-8")
    return files


def _by_basename(files: dict[str, str]) -> dict[str, str]:
    """守卫白名单按 basename 键（`assembly.py` / `inject.py`）；其余文件按相对路径原样。"""
    out: dict[str, str] = {}
    for rel, src in files.items():
        base = rel.rsplit("/", 1)[-1]
        key = base if (base in _ALLOWED_LINES and rel.startswith("packages/bladex-proxy/bladex_proxy/")
                       and rel.count("/") == 3) else rel
        out[key] = src
    return out


# ── ② 源码守卫 ──

def test_summary_open_has_no_producer_in_production_code() -> None:
    files = _by_basename(_production_sources())
    assert "assembly.py" in files and "inject.py" in files  # 扫描面本身要成立
    bad = offenses(files)
    assert bad == [], f"SUMMARY_OPEN 出现了白名单之外的形态（L2 复活？）：{bad}"


def test_guard_whitelist_is_fully_used() -> None:
    """白名单里每一行都真的存在——否则白名单是死的（守卫盯着一个不存在的形态）。"""
    files = _by_basename(_production_sources())
    for name, allowed in _ALLOWED_LINES.items():
        present = set(_code_lines_with_token(files[name]))
        assert allowed <= present, f"{name} 白名单里有已不存在的行：{allowed - present}"


# ── ④ 阴性对照 ──

@pytest.mark.parametrize("line", _negative_control_lines)
def test_negative_control_reinserted_reverse_derivation_is_caught(line: str) -> None:
    files = _by_basename(_production_sources())
    tampered = dict(files)
    tampered["inject.py"] = files["inject.py"] + "\n" + line + "\n"
    bad = offenses(tampered)
    assert ("inject.py", line) in bad, f"把反推段加回去守卫没红：{bad}"


def test_negative_control_comment_lines_do_not_count() -> None:
    """注释里提到常量不算生产者——否则本文件顶部的退役注释就把守卫弄红了。"""
    files = _by_basename(_production_sources())
    tampered = dict(files)
    tampered["inject.py"] = files["inject.py"] + "\n# SUMMARY_OPEN in c  (comment only)\n"
    assert offenses(tampered) == []


# ── ① cap_info 键集 ──

def test_cap_info_does_not_declare_retired_keys() -> None:
    from bladex_proxy.inject import _apply_cap
    msgs = [{"role": "user", "content": "hi"}]
    _, cap = _apply_cap(msgs, None, [], False, "s", agent_id="claude-code")
    assert "summary_hash" not in cap and "kept_recent_n" not in cap
    assert cap["triggered"] is False


def test_decision_meta_cap_only_carries_triggered() -> None:
    import inspect

    from bladex_proxy.server.orchestration import _build_decision_meta
    sig = inspect.signature(_build_decision_meta)
    assert "cap_summary_hash" not in sig.parameters and "cap_kept_recent_n" not in sig.parameters


# ── ③ 存量读回 ──

def test_historical_cap_dict_with_retired_keys_round_trips() -> None:
    from bladex_proxy.models import DecisionMeta
    old = {"route": {}, "cap": {"triggered": True, "summary_hash": "abcd1234abcd1234", "kept_recent_n": 7}}
    dm = DecisionMeta.model_validate(old)
    assert dm.cap["summary_hash"] == "abcd1234abcd1234" and dm.cap["kept_recent_n"] == 7
    assert DecisionMeta.model_validate(dm.model_dump()).cap == dm.cap
