"""批 F3 守卫 · `behavior.json` 的**读取点只有一处**（判据 F3.2）。

读侧（运维面 `GET /admin/behavior`；F5 前还有模型面 `bladex_flash_read`，10-17 退出注册）
必须经同一个函数；两个面各读一次文件，就会各自长出一套"文件不在怎么办 / 过期算不算
数 / 时间怎么折"的处置，而这两套只有在同一天写的时候才一致（MQ-A18 / MQ-P7 形态：
同一件事只在一个端点做对了）。

🔴 **这把尺子量的是接线不是函数**（MQ-W13：函数有测试 ≠ 接线有守卫）：
它按源码找"谁打开了这个文件"，短路掉 `read_behavior_card` 的调用、
或在某个面里另起一个 `open(behavior_path(...))`，都会红。
"""

from __future__ import annotations

import io
import os
import re
import tokenize

from _source_probe import package_source

_PKG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "bladex_proxy")

#: 允许自己拼 `behavior.json` 路径的生产文件（写侧 + 唯一读点）。
_PATH_OWNERS = {"behavior_read.py", "flash_daemon.py"}


def _code_only(src: str) -> str:
    """去掉注释与字符串字面量 —— 只留**代码**。

    🔴 不这么做这把尺子就量错了东西：本批的 docstring 里写着
    "经 `behavior_read.read_behavior_card()`"，按原文 grep 会把一句说明
    数成第二个读取点（尺子钉住了它没打算钉的那一层，MQ-CA15 同族）。
    """
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        out.append("" if tok.type in (tokenize.COMMENT, tokenize.STRING)
                   else tok.string)
    # 空白一并抹掉：断言的是**调用形态**（`read_behavior_card()`），
    # 而 tokenize 会把 `a.b()` 拆成带空格的 token 串。
    return re.sub(r"\s+", "", "".join(out))


def _py_sources() -> dict[str, str]:
    out = {}
    for dirpath, _dirs, names in os.walk(_PKG_DIR):
        if "__pycache__" in dirpath:
            continue
        for n in names:
            if n.endswith(".py"):
                p = os.path.join(dirpath, n)
                with open(p, encoding="utf-8") as f:
                    out[os.path.relpath(p, _PKG_DIR)] = _code_only(f.read())
    return out


def test_only_behavior_read_and_the_daemon_resolve_the_path():
    """`behavior_path(` / `BEHAVIOR_FILE` 只许出现在写侧与唯一读点。"""
    hits = {rel for rel, src in _py_sources().items()
            if re.search(r"behavior_path\(|BEHAVIOR_FILE", src)}
    assert {os.path.basename(h) for h in hits} <= _PATH_OWNERS, (
        f"behavior.json 的路径在这些文件里被另行拼出: {sorted(hits)}；"
        "读侧一律经 behavior_read.read_behavior_card()")


def test_the_admin_face_goes_through_read_behavior_card():
    """读者面恰好一个调用点，且不自己开文件。

    10-17 F5：模型面 `bladex_flash_read` 退出注册，调用方从两个（handlers / admin_read）
    收成一个。再长出读者时把它加进这里——**加之前先问它是不是经 `read_behavior_card()`**。
    """
    srcs = _py_sources()
    callers = {rel for rel, src in srcs.items() if "read_behavior_card(" in src
               and os.path.basename(rel) != "behavior_read.py"}
    assert {os.path.basename(c) for c in callers} == {"admin_read.py"}, (
        f"read_behavior_card 的调用方不是预期的读者面: {sorted(callers)}")
    for rel in callers:
        # 调用形态（`read_behavior_card()`）恰一处；import 行不带括号，不计入。
        assert srcs[rel].count("read_behavior_card()") == 1, f"{rel}: 不止一个读取点"


def test_the_admin_face_uses_dashboard_view_and_the_agency_reads_nothing():
    """运维面取 `dashboard_view()`，且**不许**自己再算一遍；工具面（agency 包）
    F5 后不再读画像——读回来就是把退出注册的入口从后门接回去。"""
    agency = _code_only(package_source("agency"))
    admin = _py_sources()["admin_read.py"]
    assert "read_behavior_card" not in agency and "model_view" not in agency
    assert ".dashboard_view()" in admin
    # 运维面不得重算聚合（那是 daemon 的事，dashboard 另算一遍 = 第二个口径）
    assert "aggregate(" not in admin
