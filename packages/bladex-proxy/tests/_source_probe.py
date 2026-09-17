"""按 AST 取函数源码 —— 替代 `inspect.getsource` 做源码断言（2026-08-28）。

## 为什么需要它

全仓库有 ~29 处用 `inspect.getsource(f)` 做源码断言（"这个函数里必须/不许出现
某段代码"）。`getsource` 的定位方式是：读 `f.__code__.co_firstlineno`，再从
**磁盘上的源文件**那一行开始找块边界。

⇒ **字节码的行号与磁盘文件不同步时，它会安静地返回另一个函数的源码。**

2026-08-28 live 撞到：在 `ledger_injection_message` 之前插入了 50 行的新方法，
用户机 gate 报 `assert 'run_inner_loop(' in src` 失败，而 `src` 的内容是
`ledger_injection_message` 的全文 —— `process_message` 的旧行号落进了新文件里
另一个方法的范围。（`packages/bladex-proxy/bladex_proxy/__pycache__/` 下确有
三天前的 `agency.cpython-310.pyc`。）

🔴 **红了还算走运。真正危险的是取到相邻函数、而断言恰好通过 —— 静默假绿。**
那 29 处每一处都有这个风险。

## 修法

从磁盘文件解析 AST 找定义，完全不碰 `co_firstlineno`。找不到就**报错**，
不返回空串——`"x" not in ""` 恒真，那是把假绿写进工具里。
"""
from __future__ import annotations

import ast
import inspect
import os
import textwrap
from typing import Any

_PROXY_PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bladex_proxy")


def consumer_module(preferred: str, fallback: str):
    """monkeypatch 目标：拆包后的消费方子模块；子模块尚不存在时退回门面模块。

    只为 F0.1 三个 commit（agency → server → cli）逐个落地时**中间态仍 gate 绿**而存在：
    同一个测试文件既被 agency 拆包改了 patch 目标、又要在 cli 拆包时改第二处时，
    早一个 commit 里 `bladex_proxy.cli.ledger_cmds` 还不存在。三个 commit 全落地后
    fallback 永不触发，调用点可随时改回直接 import。
    """
    import importlib
    try:
        return importlib.import_module(preferred)
    except ModuleNotFoundError:
        return importlib.import_module(fallback)


def package_source(name: str) -> str:
    """把 `bladex_proxy/<name>/` 下全部 `*.py` 按文件名排序后**拼成一段文本**（含 `__init__.py`）。

    2026-09-06 F0.1：`agency.py` / `server.py` / `cli.py` 三个巨石零行为拆成包。此前 ~30 处
    "按路径读源码做计数/包含断言"的守卫读的是单文件；拆包后**同一条断言的对象是整个包**
    （例：`_strip_execute_splice(` 期望 5 处 = def 一处 + 四个调用点，现在分布在
    `runtime.py` 与 `streams.py`）。故按包拼接，断言原样不动。
    找不到目录 **报错**而不是返回空串——`"x" not in ""` 恒真是静默假绿。
    """
    d = os.path.join(_PROXY_PKG, name)
    single = os.path.join(_PROXY_PKG, name + ".py")
    if not os.path.isdir(d):
        if os.path.isfile(single):
            # 拆包前的单文件形态（F0.1 三个 commit 逐个落地时，中间态仍要 gate 绿）
            with open(single, encoding="utf-8") as f:
                return f.read()
        raise AssertionError(f"{d} 不是包目录、{single} 也不存在 —— 结构又变了？先修这里，别让守卫假绿")
    parts = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".py"):
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                parts.append(f"# ==== {name}/{fn} ====\n" + f.read())
    assert parts, f"{d} 下没有 .py"
    return "\n".join(parts)


def source_of(target: Any, name: str = "") -> str:
    """取源码文本。`target` 可以是类/函数/模块；`name` 指定类内方法名。

        source_of(AgencyRuntime, "process_message")   # 类方法
        source_of(some_function)                      # 顶层函数
        source_of(AgencyRuntime)                      # 整个类

    与 `inspect.getsource` 的差别：**按名字在 AST 里找**，不按字节码行号，
    因而对 stale `.pyc`、行号漂移、装饰器包装都免疫。
    """
    mod = inspect.getmodule(target)
    if mod is None:
        raise AssertionError(f"取不到 {target!r} 所属模块 —— 无法做源码断言")
    path = inspect.getsourcefile(mod)
    if not path:
        raise AssertionError(f"{mod.__name__} 没有源文件 —— 无法做源码断言")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    tree = ast.parse(text)

    if inspect.ismodule(target) and not name:
        return text

    outer = target.__name__ if not inspect.ismodule(target) else ""
    node = _find(tree, outer, name)
    if node is None:
        raise AssertionError(
            f"在 {path} 里找不到 "
            f"{outer + '.' if outer and name else ''}{name or outer} 的定义。\n"
            "改名了？还是断言写错了名字？—— 这里**故意报错而不是返回空串**，"
            "因为 `\"x\" not in \"\"` 恒真，静默假绿比红更糟。"
        )
    return textwrap.dedent(ast.get_source_segment(text, node) or "")


def _find(tree: ast.AST, outer: str, name: str) -> ast.AST | None:
    """先找 `outer`（类/函数名），再在其直接子节点里找 `name`。"""
    def defs(scope):
        return [n for n in ast.iter_child_nodes(scope)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]

    if not name:
        return next((n for n in defs(tree) if n.name == outer), None)
    if not outer:
        return next((n for n in defs(tree) if n.name == name), None)
    holder = next((n for n in defs(tree) if n.name == outer), None)
    if holder is None:
        return None
    return next((n for n in defs(holder) if n.name == name), None)
