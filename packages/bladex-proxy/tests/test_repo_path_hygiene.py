"""仓库路径卫生守卫（CLAUDE.md「给 AI 工具的特别说明」第 11 条）。

背景：仓库从 `~/Documents/Claude/Projects/BladeX` 迁到 `~/dev/BladeX`。旧路径**只作为
软链保留**，供 Claude Desktop app 使用（它的项目目录不可改）。软链不是第二份代码，
但**代码里一旦写死旧路径，就会在没有那个软链的机器上（CI、别人的部署、容器里）直接断**——
而本机因为软链在，永远测不出来。这正是"只在我机器上能跑"的经典形态。

守卫范围只覆盖**会被执行的东西**：`packages/` / `scripts/` / `config/`。
不覆盖 `docs/` —— 老 ADR 与历史实验记录里的旧路径是**当时的事实**，
改它们等于伪造历史（CLAUDE.md 第 11 条第二款）。
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_LEGACY = "Documents/Claude/Projects/BladeX"

#: 会被执行的目录 —— 只有这些需要守。docs/ 是历史，data/ 是数据。
_GUARDED = ("packages", "scripts", "config")

#: 受检后缀。`.json` 只在 config/ 下检查（`.claude/settings.local.json` 是本机
#: 工具配置、不随仓库执行，且不在受检目录内）。
_SUFFIXES = {".py", ".sh", ".toml", ".yml", ".yaml", ".json", ".env", ".example"}


#: 🔴 守卫必须排除自己：本文件正文里就写着 `_LEGACY` 那个字符串（常量 + 阳性对照），
#: 不排除就会稳定地抓到自己、恒红。自指是这类"全仓扫某字符串"守卫的通病。
_SELF = Path(__file__).resolve()

#: 阳性对照种的探针文件名。真扫描对它**免疫**（2026-08-25 事故：沙盒里 unlink 被
#: EPERM 拦下，finally 删不掉 → 残渣留在仓库 → 用户机 gate 恒红。守卫自己的
#: 工件不该有能力把守卫打红——把工件名纳入排除，阳性对照走 include_probe=True
#: 的显式通道，两个测试的判别力都不受损）。
_PROBE_NAME = "_tmp_path_hygiene_probe.py"


def _iter_guarded_files(*, include_probe: bool = False):
    for top in _GUARDED:
        base = _ROOT / top
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if p.resolve() == _SELF:
                continue
            if p.name == _PROBE_NAME and not include_probe:
                continue
            if any(part in {".venv", "__pycache__", "node_modules"} for part in p.parts):
                continue
            if p.suffix in _SUFFIXES or p.name.startswith(".env"):
                yield p


def test_no_executable_code_references_the_legacy_repo_path() -> None:
    """🔴 生产代码不得写死旧仓库路径。

    本机有软链所以不会报错 —— 这恰恰是危险之处：断只会断在别人的机器上。
    """
    offenders: list[str] = []
    for p in _iter_guarded_files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _LEGACY in text:
            for i, line in enumerate(text.splitlines(), 1):
                if _LEGACY in line:
                    offenders.append(f"{p.relative_to(_ROOT)}:{i}: {line.strip()[:100]}")

    assert not offenders, (
        "以下受检文件写死了旧仓库路径（权威路径是 ~/dev/BladeX，旧路径只是给 "
        "Claude Desktop 的软链）：\n  " + "\n  ".join(offenders))


def test_guard_actually_scans_something() -> None:
    """🔴 阴性对照：确认扫描范围非空。

    一个扫不到任何文件的守卫恒绿 —— 比没有守卫更糟，因为它给人已经守住的错觉。
    本轮已两次撞见同型问题（-1 哨兵伪装成读数、校准默认通过）。
    """
    files = list(_iter_guarded_files())
    assert len(files) > 50, f"受检文件仅 {len(files)} 个，扫描范围可疑（目录改名了？）"
    assert any(p.suffix == ".py" for p in files)
    assert any(p.suffix == ".sh" for p in files)


def test_guard_would_catch_a_violation(tmp_path: Path) -> None:
    """🔴 阳性对照：真的种一个违规文件进受检目录，守卫必须抓到它。

    只断言"现在是绿的"不够 —— 恒绿的守卫和没有守卫等价。这里用真文件走完整扫描路径，
    连"排除自身"的逻辑有没有把别人也一起排掉都能验出来。
    """
    planted = _ROOT / "scripts" / _PROBE_NAME
    planted.write_text(f"BAD = '/Users/alice/{_LEGACY}/data'\n", encoding="utf-8")
    try:
        hits = [p for p in _iter_guarded_files(include_probe=True)
                if _LEGACY in p.read_text(encoding="utf-8", errors="ignore")]
        assert planted in hits, "守卫扫不到种进去的违规文件 —— 它是恒绿的"
        # 真扫描（默认通道）对探针工件免疫——残渣即使删不掉也打不红 gate。
        assert planted not in set(_iter_guarded_files())
    finally:
        try:
            planted.unlink(missing_ok=True)
        except PermissionError:
            # 沙盒挂载卷禁 unlink（EPERM）。残渣已被默认扫描通道免疫（_PROBE_NAME），
            # 留下也打不红 gate——吞掉这个错优于让"删不动"伪装成守卫失败。
            pass


def test_guard_excludes_only_itself(tmp_path: Path) -> None:
    """排除自身不许扩大化：同目录的其它测试文件仍要在扫描范围内。"""
    files = set(_iter_guarded_files())
    assert _SELF not in files
    siblings = [p for p in files if p.parent == _SELF.parent]
    assert len(siblings) > 10, "同目录兄弟文件被误排除了"
