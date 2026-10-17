"""批 Z1 · Z1.1 —— MQ-A42：`bladex start` 的 /ready 等待窗口感知首次下载模型。

判据（提示词 `prompt-batch-Z1-20260923.md` §1 Z1.1）：
  ① 空缓存冷启动不打 `failed`，打下载进度文案（含模型名），下完照常判 ready；
  ② 🔴 阴性对照：没在下载的真失败仍在**原窗口**内报 failed（不许整体放大 30 秒）；
  ③ 下载失败 ⇒ failed 且点名是下载失败；
  ④ 信号来源可指认：唯一写入点 = `embedding.ensure_local_model`，读方只经 `read_download_state`。
"""

from __future__ import annotations

import json
import os
import re
import types
from pathlib import Path

import pytest
from bladex_proxy import embedding as em
from bladex_proxy.cli import lifecycle as lc

_PKG = Path(__file__).resolve().parents[1] / "bladex_proxy"


class _Clock:
    """假时钟：`sleep` 推进 `monotonic`，测试不真等。"""

    def __init__(self) -> None:
        self.t = 1000.0
        self.hooks: list = []

    def monotonic(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.t += s
        for h in self.hooks:
            h(self.t)


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    import time as _real
    monkeypatch.setattr(lc, "time", types.SimpleNamespace(
        monotonic=c.monotonic, sleep=c.sleep, strftime=_real.strftime))
    return c


def _write_marker(cache: Path, pid: int, **fields) -> Path:
    d = cache / em.DOWNLOAD_STATE_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{pid}.json"
    p.write_text(json.dumps({"pid": pid, **fields}), encoding="utf-8")
    return p


def test_cold_start_download_is_not_reported_as_failed(clock, tmp_path, monkeypatch, capsys):
    """判据①：下载持续 90 秒（远超 30 秒窗口），期间 /ready 不答 ⇒ 不报 failed，下完判 ready。"""
    pid = os.getpid()
    marker = _write_marker(tmp_path, pid, state="downloading",
                           model="intfloat/multilingual-e5-large", size_gb=2.24,
                           source="https://huggingface.co")
    blob = tmp_path / "models--x" / "blob.incomplete"
    blob.parent.mkdir()
    start = clock.t

    def _progress(t: float) -> None:
        if t - start < 90:
            blob.write_bytes(b"x" * int((t - start) * 1000))  # 每秒都在长
        elif marker.exists():
            marker.unlink()  # 下完：ensure_local_model 删标记

    clock.hooks.append(_progress)
    ready_after = start + 95
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: (
        {"ready": True, "checks": {"redis": True, "hub": True, "index": True}}
        if clock.t >= ready_after else None))
    rc = lc._report_startup_state("127.0.0.1", 18989, True, True, timeout_s=30,
                                  proxy_pid=pid, cache_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "failed" not in out
    assert "downloading embedding model intfloat/multilingual-e5-large" in out
    assert "first start only" in out
    assert "download finished" in out
    assert "state: ready" in out


def test_negative_control_real_failure_keeps_original_window(clock, tmp_path, monkeypatch, capsys):
    """判据② 🔴：没有下载标记、进程活着、/ready 不答 ⇒ 仍在 30 秒窗口内报 failed。"""
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: None)
    start = clock.t
    rc = lc._report_startup_state("127.0.0.1", 18989, True, True, timeout_s=30,
                                  proxy_pid=os.getpid(), cache_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert rc == 1
    assert "did not answer within 30s" in out
    assert clock.t - start <= 31, f"窗口被放大了：{clock.t - start:.0f}s"
    assert "downloading" not in out


def test_negative_control_stale_marker_of_dead_process_is_ignored(clock, tmp_path, monkeypatch, capsys):
    """被 kill 的上一次首启留下 downloading 标记：pid 已死 ⇒ 不算在下，窗口照旧。"""
    dead = 2 ** 22 + 12345  # 远超 macOS/Linux 常见 pid 上限的值，必然不存在
    _write_marker(tmp_path, dead, state="downloading", model="m")
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: None)
    start = clock.t
    rc = lc._report_startup_state("127.0.0.1", 18989, True, True, timeout_s=30,
                                  proxy_pid=os.getpid(), cache_dir=str(tmp_path))
    assert rc == 1 and clock.t - start <= 31
    assert em.read_download_state(str(tmp_path), dead) is None


def test_download_failure_is_named_as_download_failure(clock, tmp_path, monkeypatch, capsys):
    """判据③：下载失败 ⇒ failed + 点名下载失败 + 错误原文（proxy 降级起来、/ready 照答也一样）。"""
    pid = os.getpid()
    _write_marker(tmp_path, pid, state="failed", model="intfloat/multilingual-e5-large",
                  error="embedding model download failed for 'x' from https://huggingface.co: 403")
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: {
        "ready": True, "checks": {"redis": True, "hub": True, "index": False}})
    stopped: list = []
    monkeypatch.setattr(lc, "_read_pidfile", lambda p: pid)
    monkeypatch.setattr(lc, "_stop_by_pidfile", lambda p, n: stopped.append(n))
    rc = lc._report_startup_state("127.0.0.1", 18989, True, True, timeout_s=30,
                                  proxy_pid=pid, cache_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert rc == 1
    assert "the embedding model download failed" in out
    assert "403" in out
    assert "did not answer" not in out
    assert stopped == ["Proxy"], "记忆已死的 proxy 不留着"


def test_download_stall_is_reported_as_failure(clock, tmp_path, monkeypatch, capsys):
    """判据③ 的另一半：标记说在下、字节数 120 秒不动 ⇒ 按下载失败报，不无限等。"""
    pid = os.getpid()
    _write_marker(tmp_path, pid, state="downloading", model="m", size_gb=1.0)
    monkeypatch.setattr(lc, "_probe_ready", lambda *a, **k: None)
    monkeypatch.setattr(lc, "_read_pidfile", lambda p: None)
    start = clock.t
    rc = lc._report_startup_state("127.0.0.1", 18989, True, True, timeout_s=30,
                                  proxy_pid=pid, cache_dir=str(tmp_path))
    out = capsys.readouterr().out
    assert rc == 1
    assert "no download progress for 120s" in out
    assert 120 <= clock.t - start <= 130


# ── ensure_local_model 写标记（信号的生产端）────────────────────────────────


def test_ensure_local_model_writes_marker_during_download_and_clears_after(monkeypatch, tmp_path):
    seen: dict = {}

    class _FakeTE:
        def __init__(self, model_name=None, cache_dir=None):
            seen["during"] = em.read_download_state(str(tmp_path), os.getpid())

        @staticmethod
        def list_supported_models():
            return [{"model": "BAAI/bge-small-en-v1.5"}]

    import fastembed
    monkeypatch.setattr(fastembed, "TextEmbedding", _FakeTE)
    monkeypatch.setenv("HF_ENDPOINT", "https://my-mirror.internal")
    assert em.ensure_local_model("BAAI/bge-small-en-v1.5", str(tmp_path)) is True
    assert seen["during"]["state"] == "downloading"
    assert seen["during"]["model"] == "BAAI/bge-small-en-v1.5"
    assert em.read_download_state(str(tmp_path), os.getpid()) is None, "下完必须删标记"


def test_ensure_local_model_marks_failure(monkeypatch, tmp_path):
    class _BoomTE:
        def __init__(self, model_name=None, cache_dir=None):
            raise OSError("connection reset")

        @staticmethod
        def list_supported_models():
            return [{"model": "BAAI/bge-small-en-v1.5"}]

    import fastembed
    monkeypatch.setattr(fastembed, "TextEmbedding", _BoomTE)
    monkeypatch.setenv("HF_ENDPOINT", "https://my-mirror.internal")
    with pytest.raises(RuntimeError, match="download failed"):
        em.ensure_local_model("BAAI/bge-small-en-v1.5", str(tmp_path))
    st = em.read_download_state(str(tmp_path), os.getpid())
    assert st["state"] == "failed" and "connection reset" in st["error"]


def test_cached_model_writes_no_marker(tmp_path):
    # 目录名按当前 fastembed 给的真实仓库名拼（0.9 起大小写变了，见 test_cache_detect_uses_real_hf_repo）
    d = tmp_path / ("models--" + em._hf_repos_for("BAAI/bge-small-en-v1.5")[0].replace("/", "--"))
    d.mkdir(parents=True)
    (d / "model.onnx").write_text("x")
    assert em.ensure_local_model("BAAI/bge-small-en-v1.5", str(tmp_path)) is True
    assert not (tmp_path / em.DOWNLOAD_STATE_DIRNAME).exists()


# ── 判据④：信号来源单一 ─────────────────────────────────────────────────────


def test_download_marker_has_one_writer_and_one_reader():
    """写入只在 embedding.py（ensure_local_model 一族）；CLI 只经 read_download_state 读。"""
    writers, dirname_users = [], []
    for f in _PKG.rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        rel = f.relative_to(_PKG).as_posix()
        if re.search(r"_write_download_state\(", src):
            writers.append(rel)
        if ".bladex-download" in src:
            dirname_users.append(rel)
    assert writers == ["embedding.py"], writers
    assert dirname_users == ["embedding.py"], dirname_users
    emb = (_PKG / "embedding.py").read_text(encoding="utf-8")
    # 写调用只出现在 ensure_local_model / _download_local_model 两个函数体里
    body = emb.split("def ensure_local_model(", 1)[1].split("\nclass _QueryLRU", 1)[0]
    total = len(re.findall(r"(?<!def )_write_download_state\(cache_dir,", emb))
    assert total == len(re.findall(r"(?<!def )_write_download_state\(cache_dir,", body)) == 3
    cli_src = (_PKG / "cli" / "lifecycle.py").read_text(encoding="utf-8")
    assert "read_download_state" in cli_src
    assert "DOWNLOAD_STATE_DIRNAME" not in cli_src, "CLI 不许自己拼标记路径"
