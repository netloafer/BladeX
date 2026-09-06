"""G9.4 H3：`bladex config show --effective` —— 生效配置一键可见。

## 它要终结的排查成本

130 个 flag 的生效值散在 env / routing.toml / flags.py 三处。已经吃过两次亏：

- ADR-0027 §5.4：五个机制开关默认关、且不在 `.env.example` 里。那次"验收通过"
  是在临时 shell 里跑的，**仓库默认形态从未变过**——发出去的是未修版本。
- ADR-0028：配置写 200、进程跑 150。

两次的形状一样：**没人能一眼看到"现在到底按什么在跑"**。

## 这组测试钉四件

1. **对账**：`flags.py` 两张默认值表里的每一项都必须出现在输出里（漏一项，
   下一次"改了没生效"的排查就还是要 grep）；且这些名字在 `.env.example` 里也在。
2. **来源三态**：同一个 flag，设 env 与不设 env 必须给出不同的 source 标注。
   只显示值不显示来源等于没解决问题——"值是 150" 和 "值是 150 因为 env 压过了模板"
   是两件事。
3. **确定性**：两次渲染逐字一致、无绝对路径、无时间戳。它要能 diff、能贴进报告。
4. **不泄密**：密钥只出指纹。这条输出天然会被贴给别人看。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bladex_core.flags import MEMORY_FLAG_DEFAULTS, MEMORY_NUMERIC_DEFAULTS

from bladex_proxy.cli import render_effective_config
from bladex_proxy.config import ProxyConfig

_ENV_EXAMPLE = Path(__file__).resolve().parents[3] / "config" / ".env.example"
_CJK = re.compile(r"[一-鿿]")


@pytest.fixture()
def text() -> str:
    return render_effective_config(ProxyConfig())


# ── 1. 与 flags.py / .env.example 对账 ─────────────────────────────────────


@pytest.mark.parametrize("name", sorted(MEMORY_FLAG_DEFAULTS) + sorted(MEMORY_NUMERIC_DEFAULTS))
def test_every_flag_is_shown(text: str, name: str) -> None:
    assert name in text, f"{name} 在 flags.py 里有默认值，却不出现在生效配置里"


@pytest.mark.parametrize("name", sorted(MEMORY_FLAG_DEFAULTS))
def test_shown_flags_are_also_in_the_template(name: str) -> None:
    """输出里的名字必须同时在模板里 —— 三方（代码/模板/视图）不许有第三种口径。

    `test_flag_defaults.py` 已经钉了"代码 ↔ 模板"，这条把视图接进同一张网。
    """
    assert name in _ENV_EXAMPLE.read_text(encoding="utf-8")


# ── 2. 来源三态 ────────────────────────────────────────────────────────────


def test_source_column_distinguishes_env_from_default(monkeypatch) -> None:
    """同一个开关：不设 env → default；设了 env → env，且值跟着翻。"""
    monkeypatch.delenv("BLADEX_HOP_EXPAND", raising=False)
    before = _line(render_effective_config(ProxyConfig()), "BLADEX_HOP_EXPAND")
    assert before.endswith("default") and " on " in before, before

    monkeypatch.setenv("BLADEX_HOP_EXPAND", "0")
    after = _line(render_effective_config(ProxyConfig()), "BLADEX_HOP_EXPAND")
    assert after.endswith("env") and " off " in after, after


def test_numeric_flag_source_and_value(monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_ADJUDICATE_TOPK", "42")
    line = _line(render_effective_config(ProxyConfig()), "BLADEX_ADJUDICATE_TOPK")
    assert "42" in line and line.endswith("env")


def test_embedding_section_reports_its_own_source(monkeypatch) -> None:
    """embedding 的来源是**合并时记下来的**，不是展示层另猜一遍。

    另猜一遍就会有第二份优先级，然后两份在某次改动后开始各说各话——
    而这正是本命令要终结的那类排查。
    """
    monkeypatch.setenv("BLADEX_EMBED_BACKEND", "local")
    line = _line(render_effective_config(ProxyConfig()), "backend")
    assert line.endswith("env"), line


# ── 3. 确定性（可 diff、可贴报告）─────────────────────────────────────────


def test_render_is_deterministic() -> None:
    assert render_effective_config(ProxyConfig()) == render_effective_config(ProxyConfig())


def test_sections_present_and_ordered(text: str) -> None:
    idx = [text.index(h) for h in ("[1] Memory flags", "[2] Embedding",
                                   "[3] Routing strategy layers", "[4] Injection & assembly")]
    assert idx == sorted(idx)


def test_no_timestamp_in_output(text: str) -> None:
    assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", text)


def test_user_facing_output_is_english(text: str) -> None:
    """ADR-0027 §5.1：用户可见字符串统一英文（`test_user_facing_english` 的同款约束）。"""
    assert not _CJK.search(text), "生效配置输出里有中文"


# ── 4. 不泄密 ──────────────────────────────────────────────────────────────


def test_secrets_are_fingerprinted_not_printed(monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_EMBED_API_KEY", "sk-supersecretvalue-0123456789")
    out = render_effective_config(ProxyConfig())
    assert "supersecret" not in out
    assert "sk-***(30 chars)" in out


def test_unset_secret_says_unset(monkeypatch) -> None:
    monkeypatch.delenv("BLADEX_EMBED_API_KEY", raising=False)
    assert "(unset)" in render_effective_config(ProxyConfig())


# ── 辅助 ───────────────────────────────────────────────────────────────────


def _line(text: str, needle: str) -> str:
    for raw in text.splitlines():
        if needle in raw:
            return raw.rstrip()
    raise AssertionError(f"输出里找不到 {needle}")
