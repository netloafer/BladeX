"""MQ-A41 · 模型面运行时资产必须随 wheel 走，并被 `bladex init` 铺到部署根（2026-09-05）。

## 病例（G7 冷安装实跑，`/tmp/bladex-g4/logs/proxy-*.log`）

    ledger_template_unreadable  '/tmp/bladex-g4/config/ledger-template.md'
    system_notes_missing  dir=…/config/system files=['AGENT.md','TOOLS.md','SKILLS.md'] loaded=0
    system_notes_empty

`pyproject.toml` 没有任何 package-data ⇒ wheel 里只有 `.py`；`init` 只写 `.env` + `routing.toml`。
⇒ **pip 装出来的部署，`load_system_notes()` 返回 ""，自我介绍整块不注，
模型不知道 BladeX 存在、不知道有工具面**——而"对后端 LLM 可见"是 v5 的核心定位（ADR-0032 §1）。

## 为什么判据没抓到

G7 的判据是「`init/start/doctor --e2e` 全过」，而 e2e 测的是**记忆闭环**（存 → 蒸 → 召），
**测不到自我介绍有没有注入**。G7 过了，冷装出来却是半个 BladeX——**判据比承诺窄**。
所以本组除了钉资产，还钉 `doctor` 必须把它当一条检查项（否则同型缺陷下次还是靠人读启动日志）。

## 单一真相源与对账

真相源在**包内** `bladex_proxy/assets/`（随 wheel 走，任何安装形态都在）；
仓库 `config/` 下那份是部署副本。两份不可避免（wheel 需要源、部署根需要用户可编辑的那份），
故按本仓既有做法**让两个副本互相对账**（同 `test_flash_scope_parity.py` 的思路）。
"""
from __future__ import annotations

import pathlib

import pytest
from bladex_proxy.cli import _ASSET_TARGETS, _install_model_facing_assets, assets_dir

_REPO = pathlib.Path(__file__).resolve().parents[3]


# ── ① 资产真的在包里（= 真的进 wheel）──────────────────────────────────────

@pytest.mark.parametrize("rel_src,_rel_dst", _ASSET_TARGETS)
def test_asset_ships_inside_the_package(rel_src: str, _rel_dst: str):
    """🔴 必须在 `bladex_proxy/` 包目录内——`uv build` 是 sdist → wheel，
    放在包外（比如仓库根的 `config/`）用 force-include 拉进来，sdist 里就没有它，
    wheel 会静默少文件。"""
    src = assets_dir().joinpath(*rel_src.split("/")[1:])
    assert src.is_file(), f"{rel_src} 不在包里 —— wheel 会漏发它"
    assert src.stat().st_size > 0, f"{rel_src} 是空文件"
    assert "bladex_proxy" in src.parts, "资产必须在包目录内，否则 sdist→wheel 会丢"


# ── ② 包内副本与仓库部署副本逐字一致（两份不可避免 ⇒ 对账）──────────────────

@pytest.mark.parametrize("rel_src,rel_dst", _ASSET_TARGETS)
def test_package_copy_matches_the_repo_deployment_copy(rel_src: str, rel_dst: str):
    """两份内容必须逐字相同。改了一份忘另一份 = 用户装到的和我们测的不是同一段话。"""
    src = assets_dir().joinpath(*rel_src.split("/")[1:])
    dst = _REPO / rel_dst
    if not dst.is_file():
        pytest.skip(f"{rel_dst} 不在本部署根（公开树/精简安装形态）")
    assert src.read_text(encoding="utf-8") == dst.read_text(encoding="utf-8"), (
        f"{rel_src} 与 {rel_dst} 内容不一致 —— 单一真相源是包内那份，"
        f"改完要同步部署副本")


# ── ③ init 铺得下去，且不覆盖用户改过的那份 ────────────────────────────────

def test_install_writes_all_assets_into_a_fresh_root(tmp_path: pathlib.Path):
    lines = _install_model_facing_assets(tmp_path)
    for _src, rel_dst in _ASSET_TARGETS:
        assert (tmp_path / rel_dst).is_file(), f"{rel_dst} 没铺下去"
    assert len(lines) == len(_ASSET_TARGETS)


def test_install_never_clobbers_a_user_edited_file(tmp_path: pathlib.Path):
    """🔴 判别力对照：AGENT.md 是**给模型看的承诺**，用户改过就不能被 init 悄悄盖回去。
    （`--force` 是另一条路，语义与 `.env` / `routing.toml` 一致。）"""
    p = tmp_path / "config" / "system" / "AGENT.md"
    p.parent.mkdir(parents=True)
    p.write_text("MY OWN NOTES", encoding="utf-8")
    _install_model_facing_assets(tmp_path)
    assert p.read_text(encoding="utf-8") == "MY OWN NOTES"
    # 其余三份仍要铺下去——"有一份存在"不等于"整组都在"
    assert (tmp_path / "config" / "system" / "TOOLS.md").is_file()


def test_force_overwrites(tmp_path: pathlib.Path):
    p = tmp_path / "config" / "system" / "AGENT.md"
    p.parent.mkdir(parents=True)
    p.write_text("OLD", encoding="utf-8")
    _install_model_facing_assets(tmp_path, force=True)
    assert p.read_text(encoding="utf-8") != "OLD"


def test_missing_source_is_reported_not_silently_skipped(tmp_path, monkeypatch):
    """源不在 = wheel 打包漏了。**必须响亮报出**——静默跳过正是本条缺陷的成因
    （misconfiguration fails loud）。"""
    monkeypatch.setattr("bladex_proxy.cli.lifecycle.assets_dir", lambda: tmp_path / "nope")   # F0.1 拆包：消费方在 cli/lifecycle.py
    lines = _install_model_facing_assets(tmp_path)
    assert lines and all("missing from the package" in ln for ln in lines), lines


# ── ④ doctor 必须把它当一条检查项（判据比承诺窄的那一课）────────────────────

def test_doctor_checks_model_facing_assets():
    """没有这条，同型缺陷下次还是只能靠人读启动日志的 warning 发现。"""
    src = (_REPO / "packages" / "bladex-proxy" / "bladex_proxy" / "cli" / "lifecycle.py"
           ).read_text(encoding="utf-8")   # F0.1 拆包：消费方在 cli/lifecycle.py（doctor + _ASSET_TARGETS）
    assert '_check("Model-facing assets"' in src, "doctor 没有把模型面资产列为检查项"
    assert "_ASSET_TARGETS" in src.split('_check("Model-facing assets"')[0], \
        "检查项要复用 _ASSET_TARGETS，别再写第二份清单"
