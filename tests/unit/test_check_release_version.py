"""发版门 `scripts/check_release_version.py` 的自测（批 O，O1）。

测的是门本身会不会红：四个 pyproject 集体停在旧版本时，**单元守卫无感**
（互相一致），只有 tag 比对能抓到——所以阴性对照的主角是「tag ≠ 版本 ⇒ 非零退出」。
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / "scripts" / "check_release_version.py"

if str(_SCRIPT.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT.parent))
crv = importlib.import_module("check_release_version")


def _fake_repo(tmp_path: Path, versions: dict[str, str]) -> Path:
    for rel in crv.PYPROJECTS:
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f'[project]\nname = "{p.parent.name}"\nversion = "{versions[rel]}"\n',
                     encoding="utf-8")
    return tmp_path


def _all(v: str) -> dict[str, str]:
    return dict.fromkeys(crv.PYPROJECTS, v)


# ── 阳性：一致 + tag 匹配 ─────────────────────────────────────────────────


def test_consistent_and_tag_matches(tmp_path):
    root = _fake_repo(tmp_path, _all("0.2.0"))
    assert crv.check("v0.2.0", root) == []
    assert crv.check("0.2.0", root) == []          # 裸 tag 也接受
    assert crv.check(None, root) == []             # 只对账


# ── 阴性①（这次事故的形状）：四处一致但集体停旧，tag 更新 ⇒ 必红 ───────────


def test_tag_newer_than_package_fails(tmp_path):
    root = _fake_repo(tmp_path, _all("0.1.0"))
    assert crv.check(None, root) == [], "一致性对账对这个形状无感——这正是为什么需要 tag 门"
    problems = crv.check("v0.2.0", root)
    assert len(problems) == len(crv.PYPROJECTS)
    assert all("v0.2.0" in p and "0.1.0" in p for p in problems)


# ── 阴性②：四处不一致 ⇒ 无 tag 也红 ─────────────────────────────────────────


def test_inconsistent_versions_fail_without_tag(tmp_path):
    vs = _all("0.2.0")
    vs["packages/bladex-mcp/pyproject.toml"] = "0.1.0"
    root = _fake_repo(tmp_path, vs)
    problems = crv.check(None, root)
    assert len(problems) == 1 and "不一致" in problems[0]
    assert "bladex-mcp" in problems[0]


# ── 阴性③：预发布后缀 / 多余前缀不算匹配（逐字比对，不做"差不多"）──────────


@pytest.mark.parametrize("tag", ["v0.2.0-rc1", "v0.2.0.post1", "vv0.2.0", "release-0.2.0", "v0.2"])
def test_near_miss_tags_fail(tmp_path, tag):
    root = _fake_repo(tmp_path, _all("0.2.0"))
    assert crv.check(tag, root), f"{tag!r} 不该被当成 0.2.0"


# ── 真实仓库：脚本以子进程跑，退出码就是 release.yml 看到的那个 ───────────────


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(_SCRIPT), *args],
                          capture_output=True, text=True, check=False)


def test_real_repo_is_consistent_and_script_exit_codes():
    root_v = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    ok = _run("--tag", f"v{root_v}")
    assert ok.returncode == 0, ok.stderr
    assert "OK" in ok.stdout
    bad = _run("--tag", "v99.99.99")
    assert bad.returncode == 1, bad.stdout + bad.stderr
    assert "FAIL" in bad.stderr


def test_pyprojects_list_matches_workspace_members():
    """新增 workspace 包却没进对账名单 ⇒ 这里红（名单是手写的，交付物是长出来的）。"""
    root = tomllib.loads((_REPO / "pyproject.toml").read_text(encoding="utf-8"))
    members = root["tool"]["uv"]["workspace"]["members"]
    assert members == ["packages/*"], f"workspace members 形态变了：{members}"
    on_disk = sorted(p.relative_to(_REPO).as_posix()
                     for p in (_REPO / "packages").glob("*/pyproject.toml"))
    listed = sorted(rel for rel in crv.PYPROJECTS if rel.startswith("packages/"))
    assert on_disk == listed, f"packages/*/pyproject.toml 与 PYPROJECTS 不一致：{on_disk} vs {listed}"


def test_release_workflow_calls_this_script_with_the_tag():
    """门用的是这个脚本（不是 yml 里另写一套），且喂的是 tag 名；步骤顺序见下面按 YAML 解析的那条。"""
    wf = (_REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert 'scripts/check_release_version.py --tag "${{ github.ref_name }}"' in wf, (
        "release.yml 没有用 tag 名调用 check_release_version.py")


# ── 批 O · O3：publish 不可逆 ⇒ 不能是默认行为（阴性对照：去掉 if ⇒ 红）───────


def _release_steps() -> list[dict]:
    import yaml

    wf = yaml.safe_load((_REPO / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))
    return wf["jobs"]["release"]["steps"]


def test_every_uv_publish_step_is_gated_by_explicit_opt_in():
    steps = [s for s in _release_steps() if "uv publish" in str(s.get("run", ""))]
    assert steps, "release.yml 里没有 uv publish 步骤了？"
    for s in steps:
        cond = str(s.get("if", ""))
        assert cond, f"publish 步骤无条件执行：{s.get('name')}"
        assert "BLADEX_PYPI_PUBLISH == 'true'" in cond or "inputs.target" in cond, (
            f"publish 步骤的条件不是显式开关：{s.get('name')}: {cond}")


def test_tag_push_publish_requires_repo_variable_true():
    """tag push 这条默认路径：变量不是 'true' 就不发（变量缺省 = 不发）。"""
    pypi = [s for s in _release_steps() if s.get("name") == "Publish to PyPI"]
    assert len(pypi) == 1
    cond = str(pypi[0]["if"])
    assert "github.event_name == 'push'" in cond
    assert "vars.BLADEX_PYPI_PUBLISH == 'true'" in cond


def test_version_gate_precedes_build_in_step_order():
    names = [s.get("name") or s.get("uses") for s in _release_steps()]
    gate = next(i for i, n in enumerate(names) if n == "Check tag == package version")
    build = next(i for i, n in enumerate(names) if n == "Build distributions")
    assert gate < build
