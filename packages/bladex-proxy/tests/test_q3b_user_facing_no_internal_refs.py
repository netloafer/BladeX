"""批 Q3B · MQ-L63：用户看得到的文案里不许有中文，也不许引用内部条款编号。

Jason 2026-09-08 的判据：「日志里的信息不应该是中文，应该是规范的英文报错，**不能引用 CLAUDE.md 规则**。」
`MQ-…` / `ADR-…` / `CLAUDE.md` 对仓库外的人是死链——内部立论写在 docstring 与注释里。

`test_user_facing_english.py`（ADR-0027 §5.1）只守 CLI 的 print / help / docstring **首行**里的中文。
本组补它看不见的三块：
  ① 整个 `bladex_proxy/` 的 `raise …(…)` 消息（启动报错、配置报错会原样到终端）；
  ② typer 命令 docstring 里 `--help` **实际显示**的部分（click 以 `\\f` 截断——`\\f` 之后是给读代码的人看的）；
  ③ 以上各处再加上 print / echo / prompt / help 里的 `MQ-` / `ADR-` / `CLAUDE.md`。

确需保留的（例如只在测试夹具里抛的断言），进 `_ALLOW`，并写一句为什么。
不管：代码注释、日志事件名与字段、docs/、模型面资产（`assets/`）。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "bladex_proxy"
_BAD = re.compile(r"[一-鿿]|\bM[QS]-[A-Z]*\d+|\bADR-\d{4}|CLAUDE\.md")

_SINK_CALLS = {"print", "echo", "secho", "prompt", "confirm", "BadParameter", "Exit"}
_HELP_CALLS = {"Option", "Argument", "Typer", "command"}

#: "相对路径:片段" -> 为什么可以留。片段取命中字符串的一小段，改了原文就会失配变红。
_ALLOW: dict[str, str] = {
    # 只在测试进程里抛（PYTEST_CURRENT_TEST 为真时才检查），读者是写测试的人；
    # test_z1b_live_namespace_guard 有意断言这个指针。
    "storage/pipeline_redis.py:_isolate_env). MQ-A77.": "test-process-only guard",
}


def _consts(node: ast.AST):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            yield sub


def _is_command(fn: ast.FunctionDef) -> bool:
    return any(".command(" in (u := ast.unparse(d)) or u.endswith(".command")
               for d in fn.decorator_list)


def scan(pkg: Path = _PKG) -> list[str]:
    hits: list[str] = []
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Call):
                f = node.func
                name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
                if name in _SINK_CALLS or (name == "line" and rel.startswith("cli/")):
                    targets = list(node.args) + [k.value for k in node.keywords
                                                 if k.arg not in ("file", "end", "flush", "sep")]
                elif name in _HELP_CALLS:
                    targets = [k.value for k in node.keywords if k.arg == "help"]
            elif isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                targets = list(node.exc.args)
            elif isinstance(node, ast.FunctionDef) and _is_command(node):
                shown = (ast.get_docstring(node, clean=False) or "").split("\f")[0]
                m = _BAD.search(shown)
                if m:
                    hits.append(f"{rel}:{node.lineno} {node.name} --help: …{shown[max(0, m.start()-20):m.end()+30]!r}")
                continue
            for t in targets:
                for c in _consts(t):
                    if not _BAD.search(c.value):
                        continue
                    if any(key.startswith(f"{rel}:") and key.split(":", 1)[1] in c.value
                           for key in _ALLOW):
                        continue
                    hits.append(f"{rel}:{c.lineno}: {c.value.strip()[:80]!r}")
    return hits


def test_user_facing_strings_have_no_cjk_or_internal_refs() -> None:
    hits = scan()
    assert not hits, ("用户面字符串含中文或内部编号（改成英文、去掉 MQ-/ADR-/CLAUDE.md；"
                      "命令 docstring 里给读代码的人看的部分放到 \\f 之后）：\n" + "\n".join(hits))


def test_allowlist_entries_still_match() -> None:
    """白名单不许挂着过期条目：原文改掉了，对应条目就该删。"""
    stale = []
    for key in _ALLOW:
        rel, frag = key.split(":", 1)
        src = (_PKG / rel).read_text(encoding="utf-8")
        if frag not in src:
            stale.append(key)
    assert not stale, f"过期白名单条目：{stale}"


def test_guard_catches_a_planted_violation(tmp_path: Path) -> None:
    """判别力：往一个假包里放一处违规，三类 sink 各一，必须都抓到。"""
    pkg = tmp_path / "bladex_proxy"
    (pkg / "cli").mkdir(parents=True)
    (pkg / "cli" / "x.py").write_text(
        "import typer\napp = typer.Typer()\n"
        "@app.command()\ndef go() -> None:\n"
        "    '''Do it.\n\n    Internal note (see MQ-A1).\n    \\f\n    这里之后不显示。\n    '''\n"
        "    print('done, see ADR-0027')\n"
        "    raise ValueError('配置错误')\n",
        encoding="utf-8")
    hits = scan(pkg)
    assert len(hits) == 3, hits
    assert any("--help" in h and "MQ-A1" in h for h in hits)
