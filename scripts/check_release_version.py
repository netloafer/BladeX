#!/usr/bin/env python3
"""发版门：git tag 必须等于包版本（批 O，2026-09-18，O1 事故驱动）。

事故：`v0.2.0` 打在一份 `pyproject.toml` 全写着 0.1.0 的代码上，`release.yml`
构建出 `bladex_core-0.1.0-py3-none-any.whl` 并无条件 `uv publish`。当时是
CI 红 + trusted publisher 未配**偶然**挡住的——PyPI 版本号不可覆盖不可重用，
真发出去就是不可逆。这个脚本是那道本来就该有的门。

用法（本地与 CI 同一条命令）::

    python scripts/check_release_version.py --tag v0.2.0     # 与仓库版本比对
    python scripts/check_release_version.py                  # 只做四处一致性对账

判据（任一不满足即非零退出，且逐条打印）：

1. 根 `pyproject.toml` 与 `packages/*/pyproject.toml` 的 `[project].version` **互相一致**；
2. 给了 `--tag`：`tag` 去掉前导 ``v`` 后**逐字等于**该版本（`v0.2.0` ↔ `0.2.0`；
   `0.2.0` 裸 tag 也接受，`v0.2.0-rc1` 与 `0.2.0` 不等）。

零依赖（stdlib only），公开树随发（`release.yml` 在公开仓里跑）。
单元测试 `tests/unit/test_check_release_version.py` 含阴性对照。
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 参与对账的 pyproject（相对仓库根）。新增 workspace 包时这里要跟着加——
#: `test_check_release_version.py` 与 `[tool.uv.workspace].members` 对账，漏加会红。
PYPROJECTS = (
    "pyproject.toml",
    "packages/bladex-core/pyproject.toml",
    "packages/bladex-proxy/pyproject.toml",
    "packages/bladex-mcp/pyproject.toml",
)


def read_version(pyproject: Path) -> str:
    """读 `[project].version`；缺文件 / 缺键直接抛——发版门不吞错。"""
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    v = data["project"]["version"]
    if not isinstance(v, str) or not v:
        raise ValueError(f"{pyproject}: [project].version 为空")
    return v


def collect_versions(repo_root: Path = REPO_ROOT) -> dict[str, str]:
    return {rel: read_version(repo_root / rel) for rel in PYPROJECTS}


def normalize_tag(tag: str) -> str:
    """`v0.2.0` → `0.2.0`；只剥一个前导 v/V，其余逐字保留。"""
    tag = tag.strip()
    if tag[:1] in ("v", "V"):
        return tag[1:]
    return tag


def check(tag: str | None, repo_root: Path = REPO_ROOT) -> list[str]:
    """返回问题清单；空列表 = 通过。"""
    problems: list[str] = []
    versions = collect_versions(repo_root)
    distinct = sorted(set(versions.values()))
    if len(distinct) != 1:
        problems.append(
            "包版本不一致："
            + "; ".join(f"{rel}={v}" for rel, v in versions.items())
        )
    if tag is not None:
        want = normalize_tag(tag)
        for rel, v in versions.items():
            if v != want:
                problems.append(f"tag {tag!r}（→ {want!r}）≠ {rel} 的 version {v!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default=None, help="git tag（形如 v0.2.0）；省略则只做一致性对账")
    ap.add_argument("--repo-root", default=str(REPO_ROOT), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = Path(args.repo_root)
    try:
        problems = check(args.tag, root)
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"release-version-check: 读不到版本：{exc}", file=sys.stderr)
        return 2
    versions = collect_versions(root)
    if problems:
        for p in problems:
            print(f"release-version-check: ✗ {p}", file=sys.stderr)
        print("release-version-check: FAIL —— 先升 pyproject 版本号再打 tag "
              "（PyPI 版本号不可覆盖，发错了不可逆）", file=sys.stderr)
        return 1
    shown = next(iter(versions.values()))
    suffix = f"，tag {args.tag} 匹配" if args.tag else ""
    print(f"release-version-check: OK version={shown}（{len(versions)} 处一致{suffix}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
