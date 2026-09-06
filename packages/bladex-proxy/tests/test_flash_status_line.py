"""`bladex status` 的 Memory Flash 那一行（从 test_flash_docs.py 拆出，2026-09-03 S5）。

老渲染器（ADR-0031 `memory_index.refresh_flash_docs`）与 `BLADEX_FLASH_RENDER` 随 S5
删除；这一行改按 ADR-0032 布局的账本池计数，写者是 bladex-flash 维护进程。
从**本机文件系统**读，不经 /admin——proxy 起没起与它无关，经 admin 中转会让
"proxy 挂了"伪装成"Flash 没渲染"。
"""

from __future__ import annotations

import pytest
from bladex_core import flash


def test_status_line_reports_ledgers_and_user_files(monkeypatch: pytest.MonkeyPatch, tmp_path):
    from bladex_proxy.cli import _flash_status_line

    monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path))
    base = flash.scope_dir(str(tmp_path), "local")
    flash.write_if_changed(f"{base}/ledgers/0f/ldg-1.md", "a")
    flash.write_if_changed(f"{base}/ledgers/0f/ldg-1.local.md", "mine")
    line = _flash_status_line()
    assert "1 ledgers" in line
    assert "1 yours" in line
    assert "newest change" in line


def test_status_line_points_at_the_writer_when_empty(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """空树要说明**谁**该写它——少了这半句用户只知道"没有"，不知道去看哪个进程。"""
    from bladex_proxy.cli import _flash_status_line

    monkeypatch.setenv("BLADEX_FLASH_PATH", str(tmp_path / "empty"))
    line = _flash_status_line()
    assert "nothing materialized yet" in line and "bladex-flash" in line
