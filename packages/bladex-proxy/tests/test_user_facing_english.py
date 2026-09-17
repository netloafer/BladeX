"""ADR-0027 §5.1 / T40：用户可见字符串统一英文。

策略（2026-08-04 拍板）：**用户可见字符串英文，代码注释与内部文档保持中文。**
理由：README / 治理文件 / 默认 embedding 都已按"面向全球开发者、英语优先"定调，
而 CLI 全部输出与 dashboard 整个 UI 是中文——海外用户装完包第一条报错就是读不懂的。

这组测试守住"可见面"，不碰注释：
- `cli.py`：`print(...)`、`typer.Option/Argument(help=...)`、`typer.prompt/confirm`、
  以及 typer 命令函数 docstring 的**首行**（typer 拿它当命令 help）。
- `dashboard.html`：除注释行外的一切。

之所以要机器守：英文化是一次性机械改动，但新写的一行 print 很容易又是中文，
而这类回归没人会在 review 里逐行盯。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_CJK = re.compile(r"[一-鿿]")
_CLI_FILES = sorted((_REPO / "packages" / "bladex-proxy" / "bladex_proxy" / "cli").glob("*.py"))   # F0.1 拆包


def _cli_tree() -> ast.Module:
    """cli/ 包全部文件的顶层节点拼成一棵树（断言对象仍是"整个 CLI"）。"""
    assert _CLI_FILES, "cli/ 包下没有 .py —— 结构又变了？"
    body = [n for p in _CLI_FILES for n in ast.parse(p.read_text(encoding="utf-8")).body]
    return ast.Module(body=body, type_ignores=[])
_DASHBOARD = _REPO / "packages" / "bladex-proxy" / "bladex_proxy" / "dashboard.html"


def _iter_str_constants(node: ast.AST):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub


def test_cli_print_output_is_english():
    """`print(...)` 是用户唯一看得见的东西。"""
    tree = _cli_tree()
    bad: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            continue
        for const in _iter_str_constants(node):
            if _CJK.search(const.value):
                bad.append(f"cli.py:{const.lineno}: {const.value.strip()[:70]}")
    assert not bad, "这些 print 仍是中文：\n" + "\n".join(bad)


def test_cli_option_help_is_english():
    """`--help` 是新用户的第一张地图。"""
    tree = _cli_tree()
    bad: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in ("Option", "Argument", "Typer"):
            continue
        for kw in node.keywords:
            if kw.arg == "help" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str) and _CJK.search(kw.value.value):
                bad.append(f"cli.py:{kw.value.lineno}: {kw.value.value[:70]}")
    assert not bad, "这些 help 文案仍是中文：\n" + "\n".join(bad)


def test_cli_command_docstring_summary_is_english():
    """typer 用命令函数 docstring 的**首行**当 help —— 它是用户可见面的一部分，
    不适用"docstring 保持中文"这条（其余行仍可中文，本测试只看首行）。"""
    tree = _cli_tree()
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any(".command(" in ast.unparse(d) for d in node.decorator_list):
            continue
        doc = ast.get_docstring(node)
        if not doc:
            continue
        first = doc.splitlines()[0]
        if _CJK.search(first):
            bad.append(f"cli.py:{node.lineno} {node.name}: {first[:70]}")
    assert not bad, "这些命令的 help 首行仍是中文：\n" + "\n".join(bad)


def test_cli_prompts_and_confirms_are_english():
    tree = _cli_tree()
    bad: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("prompt", "confirm")):
            continue
        for const in _iter_str_constants(node):
            if _CJK.search(const.value):
                bad.append(f"cli.py:{const.lineno}: {const.value[:70]}")
    assert not bad, "这些交互提示仍是中文：\n" + "\n".join(bad)


def test_generated_env_template_is_english():
    """`bladex init` 生成的 .env 注释是用户会打开来读的文件。"""
    from bladex_proxy.cli import _ENV_MINIMAL_TEMPLATE, _ROUTING_MINIMAL_TEMPLATE

    for name, text in (("env", _ENV_MINIMAL_TEMPLATE), ("routing", _ROUTING_MINIMAL_TEMPLATE)):
        assert not _CJK.search(text), f"init 生成的 {name} 模板含中文"


def test_dashboard_ui_is_english():
    """dashboard 五页的 UI 文案；注释行（/* … */、//）保持中文不算。"""
    lines = _DASHBOARD.read_text(encoding="utf-8").splitlines()
    bad: list[str] = []
    in_block = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if in_block:
            if "*/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/*"):
            if "*/" not in stripped:
                in_block = True
            continue
        if stripped.startswith("//") or stripped.startswith("<!--"):
            continue
        if _CJK.search(line):
            bad.append(f"dashboard.html:{i}: {stripped[:80]}")
    assert not bad, "dashboard 这些 UI 文案仍是中文：\n" + "\n".join(bad)


def test_dashboard_declares_english_lang():
    assert '<html lang="en">' in _DASHBOARD.read_text(encoding="utf-8")


def test_mcp_tool_descriptions_are_english():
    """MCP 工具描述直接进模型上下文，也是对外面。"""
    src = (_REPO / "packages" / "bladex-mcp" / "bladex_mcp" / "server.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not any("tool()" in ast.unparse(d) for d in node.decorator_list):
            continue
        doc = ast.get_docstring(node) or ""
        if _CJK.search(doc):
            bad.append(f"{node.name}: {doc.splitlines()[0][:70]}")
    assert not bad, "这些 MCP 工具描述仍是中文：\n" + "\n".join(bad)
