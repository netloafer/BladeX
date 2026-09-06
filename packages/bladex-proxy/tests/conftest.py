"""让本目录下的测试辅助模块可被**绝对导入**（2026-08-28，gate 事故驱动）。

## 为什么不能用相对导入

仓库里有**两个 `tests` 目录**：

    tests/                       ← 无 __init__.py（命名空间包）
    packages/bladex-proxy/tests/ ← 有 __init__.py（常规包）

`tests` 这个顶层名字归谁，取决于**导入顺序**。gate 的第一阶段把两边的文件
混在一条 pytest 命令里，根 `tests/` 先被处理时 `tests` 绑到了仓库根，
于是本目录里的 `from ._source_probe import …` 解析成 `tests._source_probe`
而找不到 —— 报错信息（"No module named 'tests._source_probe'"）完全指不到真因。

🔴 `test_stream_keepalive.py` 用同样的相对导入却一直是绿的，**只是因为它不在
NEW_TESTS 里、只在全量阶段跑，避开了那个顺序**。那不是"它没问题"，是"它没被
测到"。同一天里第二次撞见「绿只是因为没跑到那条路径」。

## 做法

把本目录加进 `sys.path`，测试改用绝对导入 `from _source_probe import source_of`。
conftest 由 pytest 特殊加载，不参与上面那场包名争夺。
"""
from __future__ import annotations

import pathlib
import sys

_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
