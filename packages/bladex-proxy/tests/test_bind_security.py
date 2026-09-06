"""T8 安全基线：非回环 + 无 auth 启动警告测试（beta T8）。

warn_if_insecure_bind: 绑非回环且未开 auth -> 横幅 + structlog warning。
回环地址或 auth 开启 -> 不警告。
"""
from __future__ import annotations

import io

import structlog.testing
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import warn_if_insecure_bind


def test_non_loopback_no_auth_warns():
    """非回环 + auth 关 -> 横幅 + structlog warning。"""
    cfg = ProxyConfig(host="0.0.0.0", auth_enabled=False)
    buf = io.StringIO()
    with structlog.testing.capture_logs() as cap:
        warned = warn_if_insecure_bind(cfg, stream=buf)
    assert warned is True
    banner = buf.getvalue()
    assert "SECURITY WARNING" in banner
    assert "0.0.0.0" in banner
    assert any(e["event"] == "bind_non_loopback_no_auth" for e in cap), cap


def test_loopback_no_warn():
    """回环地址（默认 127.0.0.1）-> 不警告。"""
    buf = io.StringIO()
    with structlog.testing.capture_logs() as cap:
        warned = warn_if_insecure_bind(ProxyConfig(host="127.0.0.1"), stream=buf)
    assert warned is False
    assert buf.getvalue() == ""
    assert cap == []


def test_loopback_variants_no_warn():
    """localhost / ::1 同属回环 -> 不警告。"""
    for h in ("localhost", "::1"):
        buf = io.StringIO()
        with structlog.testing.capture_logs() as cap:
            warned = warn_if_insecure_bind(ProxyConfig(host=h), stream=buf)
        assert warned is False, f"{h} 应视为回环"
        assert cap == []


def test_non_loopback_with_auth_no_warn():
    """非回环但 auth 开（有 client key 鉴权）-> 不警告。"""
    cfg = ProxyConfig(
        host="0.0.0.0", auth_enabled=True, client_keys_raw="bladex-test||test",
    )
    buf = io.StringIO()
    with structlog.testing.capture_logs() as cap:
        warned = warn_if_insecure_bind(cfg, stream=buf)
    assert warned is False
    assert buf.getvalue() == ""
    assert cap == []
