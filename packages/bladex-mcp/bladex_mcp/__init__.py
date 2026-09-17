"""BladeX MCP server（Beta T16/T17，B4）。"""

import tomllib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
from pathlib import Path


def _package_version() -> str:
    """单一真相源 = `pyproject.toml`（批 O，O1）；这里只派生，不写第二份字面量。

    与 `bladex_core.version.package_version` 同一套取法（已安装读 dist 元数据，
    源码树直跑读旁边的 pyproject，都取不到给显式的 ``0+unknown``）。
    不 import bladex_core：本包刻意零依赖（只有 `mcp`），不为一个版本号加一条依赖。
    """
    try:
        return _dist_version("bladex-mcp")
    except PackageNotFoundError:
        pass
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        v = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {}).get("version")
        return v if isinstance(v, str) and v else "0+unknown"
    except (OSError, tomllib.TOMLDecodeError):
        return "0+unknown"


__version__ = _package_version()
