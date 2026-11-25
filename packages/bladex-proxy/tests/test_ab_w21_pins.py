"""批 AB · MQ-W21：放开 fastembed 上限的前提是钉住 onnxruntime。

调度 10-22 Linux 实测：ort 同版本时 fastembed 0.8 / 0.9 向量逐位一致；不钉 ort 时 0.9 带进 1.31.0，
最大绝对差 8.6e-4。⇒ pyproject 与 lock 里 ort 版本必须是同一个确定值。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _locked(name: str) -> str:
    m = re.search(rf'^name = "{name}"\nversion = "([^"]+)"', (ROOT / "uv.lock").read_text("utf-8"),
                  re.MULTILINE)
    assert m, name
    return m.group(1)


def test_onnxruntime_pinned_to_locked_version() -> None:
    ort = _locked("onnxruntime")
    for pkg in ("bladex-proxy", "bladex-core"):
        text = (ROOT / "packages" / pkg / "pyproject.toml").read_text("utf-8")
        assert f'"onnxruntime=={ort}"' in text, f"{pkg} 必须钉 onnxruntime=={ort}"
        assert '"fastembed>=0.3,<0.10"' in text


def test_locked_fastembed_within_the_verified_range() -> None:
    major, minor = _locked("fastembed").split(".")[:2]
    assert (int(major), int(minor)) in {(0, 8), (0, 9)}, "升级前先做向量逐位一致对照"
