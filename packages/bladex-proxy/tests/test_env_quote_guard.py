"""`.env` 引号错配检出（`task-anchor-materialize-20260901.md` T5）。

live 病例：`config/.env:105` 写成 `BLADEX_MODULE_FLASH="1'`（开 `"` 收 `'`）。
Python 侧 `parse_env_file` 走 `.strip('"').strip("'")` 读出 `1`，**运行时毫无异常**；
zsh `source config/.env` 却从该行一路吞到 EOF，丢掉三个 `BLADEX_ASSEMBLY_*` 键，
只报一句 `unmatched "`。⇒ **同一个文件，两个加载器读出两个不同的环境**
（刚性原则 12 的形状）。

检出只报不改：修值是用户的事，程序的责任是别让它静默（原则 13 同族）。
"""

from __future__ import annotations

from pathlib import Path

from bladex_proxy.deployment import parse_env_file, unbalanced_quote_lines


def _write(tmp_path: Path, body: str) -> str:
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return str(p)


# ── 该报的 ────────────────────────────────────────────────────────────────


def test_double_open_single_close_is_caught(tmp_path: Path) -> None:
    """live 病例原样。"""
    p = _write(tmp_path, 'export BLADEX_MODULE_FLASH="1\'\n')
    assert unbalanced_quote_lines(p) == [(1, "BLADEX_MODULE_FLASH")]


def test_open_without_close_is_caught(tmp_path: Path) -> None:
    p = _write(tmp_path, 'export A="unterminated\n')
    assert unbalanced_quote_lines(p) == [(1, "A")]


def test_close_without_open_is_caught(tmp_path: Path) -> None:
    p = _write(tmp_path, 'export A=trailing"\n')
    assert unbalanced_quote_lines(p) == [(1, "A")]


def test_line_number_is_the_real_one(tmp_path: Path) -> None:
    """行号要能直接定位——报错说不清在哪一行，等于没报。"""
    p = _write(tmp_path, "# c\nA=1\n\nB='oops\n")
    assert unbalanced_quote_lines(p) == [(4, "B")]


def test_lone_quote_value_is_caught(tmp_path: Path) -> None:
    """值只有一个引号字符：`len(val) < 2` 那条分支。"""
    p = _write(tmp_path, 'A="\n')
    assert unbalanced_quote_lines(p) == [(1, "A")]


# ── 不该报的（判别力：误报会让人学会忽略这条警告）─────────────────────────


def test_balanced_forms_are_clean(tmp_path: Path) -> None:
    p = _write(tmp_path, 'A="1"\nB=\'2\'\nC=3\nexport D="a b c"\n')
    assert unbalanced_quote_lines(p) == []


def test_quote_inside_a_quoted_value_is_clean(tmp_path: Path) -> None:
    """`BLADEX_HARD_RULES="MUST ... don't ..."` —— 引号内的撇号是正常英文。

    这是 live `.env` 第 50 行的真实形状，误报它 = 每次启动都刷一条假警告。
    """
    p = _write(tmp_path, 'export BLADEX_HARD_RULES="MUST answer; don\'t guess"\n')
    assert unbalanced_quote_lines(p) == []


def test_comments_are_ignored(tmp_path: Path) -> None:
    """注释里的引号在 shell 里也无害（`#` 之后整行忽略），不该报。"""
    p = _write(tmp_path, '# 这里有个"中文引号说明\nA=1\n')
    assert unbalanced_quote_lines(p) == []


def test_missing_file_is_empty_not_an_error(tmp_path: Path) -> None:
    assert unbalanced_quote_lines(str(tmp_path / "nope.env")) == []


# ── 解析行为不许被这条检查改变 ────────────────────────────────────────────


def test_parse_is_unchanged_by_the_check(tmp_path: Path) -> None:
    """🔴 只报不改：错配的行照旧读出本意。

    改行为会让一个本来能跑的部署在升级后突然起不来，而这个缺陷的危害是
    "两个加载器不一致"，告警足以让它现形。
    """
    p = _write(tmp_path, 'export BLADEX_MODULE_FLASH="1\'\nexport A="2"\n')
    vals = parse_env_file(p)
    assert vals["BLADEX_MODULE_FLASH"] == "1"
    assert vals["A"] == "2"


def test_parse_env_file_stays_pure(tmp_path: Path) -> None:
    """`parse_env_file` 不得自己报告——它在只读比对场景也被调用。

    判别力：把"解析"和"报告"混在一起，下一个人就不敢改任一边；
    报告接在 `cli._load_env_file()`，那条路本来就负责把配置问题打到 stderr。

    🔴 判据是**有没有调用**，不是"有没有提到"：首版对整段源码做字符串匹配，
    被生产函数自己 docstring 里的那句提及打成 FAIL。与同日 admin key 那次同型
    ——解释一个模式时写出了那个模式的字面量。改成走 AST 看真实调用，
    注释与文档字符串怎么写都不影响。
    """
    import ast
    import inspect
    import textwrap

    from bladex_proxy import deployment
    src = textwrap.dedent(inspect.getsource(deployment.parse_env_file))
    tree = ast.parse(src)
    called = {
        node.func.id if isinstance(node.func, ast.Name) else
        getattr(node.func, "attr", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "unbalanced_quote_lines" not in called, (
        f"parse_env_file 里调了检出函数 —— 只读比对场景会被它刷屏。调用到的：{sorted(called)}")
