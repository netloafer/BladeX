"""V-A4 守卫：Memory Hub 更名后的命名卫生（ADR-0032 §2.2）。

批一（2026-08-25）做了呈现层：源码禁旧名短语（"Memory"+"Ledger"）。
符号层收口（2026-08-28，Jason 拍板推翻批一「符号不强改」）加三条：
② Hub 域旧符号（类/模块/函数名）全仓零出现——旧名与账本（LEDGER.md）撞名；
③ 新模块 `bladex_proxy.storage.memory_hub` 可导入且类名就是 `MemoryHub`；
④ 旧模块 shim 在被 git rm 前必须 fail-loud（防静默双份实现）。
目录层收口（2026-09-05，D3）加两条：
⑤ 旧目录路径字面量（`data/` + `bladex_ledger`）全仓零出现（物理目录已 mv 为 `data/bladex_hub`，不留软链）；
   🔴 但 `bladex_ledger_{read,switch,update}` 是**模型可见的工具名**，前缀恰是 `bladex_ledger`，
   守卫必须放行——判别逻辑独立成函数 `_path_literal_offenses`，⑥ 用对照用例证明它分得开
   （喂工具名必须不报、喂路径字面量必须报），否则守卫要么误报工具面、要么什么都抓不到。

历史文档（docs/）不在守卫范围——改历史 = 伪造当时的事实。
禁用符号用拼接构造，否则守卫抓到自己。
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
# tests 也在扫描根内：符号层收口时根 tests/ 差点漏网（08-28 实测盲区，
# grep 范围与守卫范围必须一致，否则残留藏在没扫的目录里）。
_SCAN_ROOTS = ("packages", "scripts", "config", "tests", "deploy")
_EXTS = {".py", ".html", ".toml", ".example", ".sh", ".yml", ".yaml"}
# 根 conftest.py 的 PRODUCTION_DEFAULTS 是默认目录的第二份（test_release_readiness 对账），
# 不在任何扫描根下，单列进来（只进 ⑤ 目录层守卫；其顶部 docstring 是事故叙述，不进短语层）。
_EXTRA_FILES = ("conftest.py",)
_BANNED_PHRASE = "Memory " + "Ledger"  # 拆开构造：守卫不该抓到自己

# Hub 域旧符号（符号层收口后的禁用清单；同样拆开构造）。
# 账本域（Ledger/LedgerEntry/ledger_id/bladex_ledger_* …）与台账域（*Journal）不在此列。
_BANNED_SYMBOLS = (
    "Ledger" + "RocksDB",          # → MemoryHub
    "rebuild_from_" + "ledger",    # → rebuild_from_hub
    "ledger_" + "rocksdb",         # 模块名 → memory_hub
    "Ledger" + "DistillJournal",   # → HubDistillJournal
    "Ledger" + "JudgmentJournal",  # → HubJudgmentJournal
    "resolve_ledger_" + "path",    # → resolve_hub_path
    "DEFAULT_LEDGER_" + "DIR",     # → DEFAULT_HUB_DIR
    "inspect_" + "ledger",         # CLI/脚本 → inspect_hub
    "state." + "ledger",           # app.state 属性 → state.hub（getattr 变体见下）
    '"ledger", None',              # getattr(<state>, "ledger", None) 形态
)

# 旧模块 shim（等用户机 git rm）；shim 自身允许出现旧名。
_SHIM_FILES = {
    "packages/bladex-proxy/bladex_proxy/storage/ledger_rocksdb.py",
    "scripts/inspect_ledger.py",
}


def _iter_source_files():
    for root in _SCAN_ROOTS:
        base = _REPO / root
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file() or p.suffix not in _EXTS:
                continue
            if "__pycache__" in p.parts or "archive" in p.parts:
                continue
            if p.name.endswith(".bak"):
                continue
            yield p


def _iter_extra_files():
    for name in _EXTRA_FILES:
        p = _REPO / name
        if p.is_file():
            yield p


# ── ⑤ 目录层：路径字面量 vs 工具名 ──
_OLD_DIR_LITERAL = "data/" + "bladex_ledger"          # 拆开构造：守卫不该抓到自己
_OLD_DIR_BASENAME = "bladex_" + "ledger"
_TOOL_NAME_WHITELIST = tuple(f"{_OLD_DIR_BASENAME}_{s}" for s in ("read", "switch", "update"))
# 路径字面量的三种写法：`data/<旧名>…`、`"data", "<旧名>"`、`"data" / "<旧名>"`（旧名 = bladex_ledger）。
# 以 `data` 为锚——工具名从不跟在 `data/` 后面，这就是两类能分开的依据。
_OLD_DIR_PATTERNS = (
    _OLD_DIR_LITERAL,
    f'"data", "{_OLD_DIR_BASENAME}"',
    f'"data" / "{_OLD_DIR_BASENAME}"',
)


def _path_literal_offenses(text: str) -> list[str]:
    """返回 `text` 中命中旧目录路径字面量的行；工具名 `bladex_ledger_*` 不算。

    独立成函数是为了让 ⑥ 的对照用例能直接喂字符串——守卫的判别力必须可单独验证。
    """
    hits: list[str] = []
    for line in text.splitlines():
        if not any(pat in line for pat in _OLD_DIR_PATTERNS):
            continue
        # 把工具名抠掉后再判：一行里既有工具名又有路径字面量时仍要报。
        stripped = line
        for tool in _TOOL_NAME_WHITELIST:
            stripped = stripped.replace(tool, "")
        if any(pat in stripped for pat in _OLD_DIR_PATTERNS):
            hits.append(line.strip())
    return hits


def test_old_hub_dir_literal_absent():
    """D3（2026-09-05）：旧目录 `bladex_ledger` 已 mv 为 `data/bladex_hub`，不留软链，
    代码 / 配置 / 脚本默认值 / 测试里不得再引用旧路径。"""
    offenders: list[str] = []
    for p in (*_iter_source_files(), *_iter_extra_files()):
        rel = str(p.relative_to(_REPO))
        if rel == "packages/bladex-proxy/tests/test_hub_naming_guard.py":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for hit in _path_literal_offenses(text):
            offenders.append(f"{rel}: {hit}")
    assert not offenders, (
        f"旧目录路径 `{_OLD_DIR_LITERAL}` 已于 2026-09-05 改为 `data/bladex_hub`（D3，ADR-0032 §2.2），"
        f"不留软链；发现: {offenders[:10]}")


def test_old_hub_dir_guard_ignores_tool_names():
    """⑥ 判别力对照·阴性：三个工具名单独出现、成对出现、带引号/括号出现，都不得报。"""
    for tool in _TOOL_NAME_WHITELIST:
        assert _path_literal_offenses(tool) == []
        assert _path_literal_offenses(f'    name="{tool}",') == []
        assert _path_literal_offenses(f"tools = [{tool}, {_TOOL_NAME_WHITELIST[0]}]") == []
    # 工具名前缀本身（无 `data/` 锚）也不报——那是账本域，不归本守卫管。
    assert _path_literal_offenses(f"{_OLD_DIR_BASENAME}_id = 3") == []


def test_old_hub_dir_guard_catches_path_literals():
    """⑥ 判别力对照·阳性：三种路径写法都必须报；与工具名同行时也必须报。"""
    assert _path_literal_offenses(f'DEFAULT_HUB_DIR = "{_OLD_DIR_LITERAL}"')
    assert _path_literal_offenses(f'os.path.join(_ROOT, "data", "{_OLD_DIR_BASENAME}")')
    assert _path_literal_offenses(f'_REPO / "data" / "{_OLD_DIR_BASENAME}"')
    assert _path_literal_offenses(f'ROCKS="{_OLD_DIR_LITERAL}_secondary"')
    mixed = f'# {_TOOL_NAME_WHITELIST[1]} 写 {_OLD_DIR_LITERAL}'
    assert len(_path_literal_offenses(mixed)) == 1


def test_banned_phrase_absent_from_source():
    offenders: list[str] = []
    for p in _iter_source_files():
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if _BANNED_PHRASE in text:
            offenders.append(str(p.relative_to(_REPO)))
    assert not offenders, (
        f"'{_BANNED_PHRASE}' is a banned term (renamed to 'Memory Hub', ADR-0032 §2.2); "
        f"found in: {offenders[:10]}")


def test_hub_domain_old_symbols_absent():
    offenders: list[str] = []
    for p in _iter_source_files():
        rel = str(p.relative_to(_REPO))
        if rel in _SHIM_FILES or rel == "packages/bladex-proxy/tests/test_hub_naming_guard.py":
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for sym in _BANNED_SYMBOLS:
            if sym in text:
                offenders.append(f"{rel}: {sym}")
    assert not offenders, (
        "Hub 域旧符号已于 V-A4 符号层收口改名（2026-08-28），新代码不得再引用；"
        f"发现: {offenders[:10]}")


def test_memory_hub_class_importable():
    pytest.importorskip("rocksdict")
    from bladex_proxy.storage.memory_hub import MemoryHub
    assert MemoryHub.__name__ == "MemoryHub"


def test_old_module_shim_fails_loud():
    """旧模块在 git rm 前必须 fail-loud——静默可导入 = 双份实现风险。"""
    shim = _REPO / "packages/bladex-proxy/bladex_proxy/storage/ledger_rocksdb.py"
    if not shim.exists():
        return  # 已被 git rm，守卫使命完成
    with pytest.raises(ImportError, match="memory_hub"):
        import importlib

        importlib.import_module("bladex_proxy.storage.ledger_rocksdb")
