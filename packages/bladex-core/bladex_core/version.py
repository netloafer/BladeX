"""包版本的唯一读取点（批 O，2026-09-18，O1 事故驱动）。

事故：0.2.0 打 tag 时四个 `pyproject.toml` 的 `version` 与两个 `__version__`
全停在 0.1.0——六处字面量互相抄、没有一处是"真相"。刚性原则 12 同款处置：
**单一真相源 = 各包 `pyproject.toml` 的 `[project].version`**（uv build 读的就是它），
`__version__` 不再是第二份字面量，而是从它派生：

1. 已安装（wheel / editable）：`importlib.metadata.version(dist)`；
2. 源码树直跑（PYTHONPATH 兜底、无 dist-info）：读包目录旁的 `pyproject.toml`；
3. 两条都取不到：返回 ``"0+unknown"``——**显式标记**，不伪装成某个版本
   （刚性原则 13：静默丢数据比丢数据更糟）。

四个 pyproject 之间的一致性由 `test_release_readiness.py` 对账；
tag 与包版本的一致性由 `scripts/check_release_version.py` 在 release 工作流里挡。
"""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path

UNKNOWN_VERSION = "0+unknown"


def read_pyproject_version(pyproject: Path) -> str | None:
    """读 `pyproject.toml` 的 `[project].version`；文件缺失或无该键返回 None。"""
    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project")
    if not isinstance(project, dict):
        return None
    v = project.get("version")
    return v if isinstance(v, str) and v else None


def package_version(dist: str, package_file: str) -> str:
    """给 `__init__.py` 用：`__version__ = package_version("bladex-proxy", __file__)`。

    `package_file` 是包目录里任一文件（通常 `__file__`），源码树形态下
    `pyproject.toml` 在包目录的上一级。
    """
    try:
        return _dist_version(dist)
    except PackageNotFoundError:
        pass
    pyproject = Path(package_file).resolve().parent.parent / "pyproject.toml"
    return read_pyproject_version(pyproject) or UNKNOWN_VERSION
