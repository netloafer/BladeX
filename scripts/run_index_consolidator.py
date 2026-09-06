#!/usr/bin/env python3
"""薄壳：转调包内 consolidator（ADR-0027 §4.2）。

真正的实现在 `bladex_proxy/consolidator.py`——它随 wheel 发布，所以
`pip install` / `uv tool install` 的用户也能启动 Memory Index 提炼（此前只有仓库脚本，
装包用户拿到的是一个永远不产生记忆的记忆产品）。

本文件只做两件事：把 workspace 源码目录加进 sys.path（仓库内直跑、未装包的场景），
然后转调 `main()`。仓库内的老命令行与 `pgrep -f run_index_consolidator.py` 全部照旧可用。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("packages/bladex-core", "packages/bladex-proxy"):
    _full = os.path.join(_ROOT, _p)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)

from bladex_proxy.consolidator import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
