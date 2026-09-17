"""Router 网关（2026-08-05）：改名规则 + 日志接管 + "SDK 只在一个文件里出现" 守卫。

这组测试防的是什么：

上游模型 SDK 是借来的通用品，但它的**日志面与异常措辞**会直接漏到用户眼前。实测
consolidator 一次运行的日志里有 767 行是供应商自己的输出——含 100 行重复（它给自己
挂了 handler、又 propagate 到 root，同一条打两遍）、23 条指向别人 issue 页的求助链接、
一条让用户敲 `litellm._turn_on_debug()`（在 BladeX 里敲不出来）。

根因不是"没关"，是**开关挂错了地方**：全局开关写在 `route.py` 模块级，而 consolidator
根本不 import `route.py` → 那两个开关在该进程里从未生效。所以真正要钉死的不是某条
文案，而是"**SDK 只允许在网关这一个文件里出现**"这条结构约束——它一旦破，同类问题
必然以新的形式重来（这与「修法要放对层」是同一条教训）。
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest
from bladex_proxy import router_sdk

_REPO = Path(__file__).resolve().parents[3]
_GATEWAY = "router_sdk.py"
# 网关自己 + 网关的测试（要断言开关确实生效，必须看得见 SDK）。
# 名单只有这两项，加第三项前先想清楚为什么它不能走网关。
_ALLOWED_DIRECT_IMPORT = frozenset({_GATEWAY, "test_router_sdk.py"})


# ── 1. 结构约束：SDK 只在网关里出现 ──────────────────────────────────────────


def _python_sources() -> list[Path]:
    roots = [_REPO / "packages", _REPO / "scripts", _REPO / "tests"]
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        out.extend(
            p for p in root.rglob("*.py")
            if ".venv" not in p.parts and "archive" not in p.parts
        )
    return out


def test_no_direct_vendor_import():
    """除网关外，任何文件都不许 `import litellm`（含函数内的惰性 import）。

    这条是本组的核心。用 AST 而不是 grep：注释与文档字符串里提到它是允许的
    （归因、踩坑记录），真正要拦的是 import 语句。
    """
    offenders: list[str] = []
    for path in _python_sources():
        if path.name in _ALLOWED_DIRECT_IMPORT:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # 不是本测试该管的问题
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "litellm" or n.startswith("litellm.") for n in names):
                offenders.append(f"{path.relative_to(_REPO)}:{node.lineno}")
    assert not offenders, (
        "这些文件直接 import 了上游模型 SDK，绕过 Router 网关：\n  "
        + "\n  ".join(offenders)
        + f"\n改成 `from bladex_proxy import router_sdk`。全局开关/日志接管/异常翻译"
          f"都挂在 {_GATEWAY}，分散 import 会让开关'谁先 import 谁说了算'。"
    )


def test_gateway_exposes_the_three_call_shapes():
    """调用面就这三种；少一个就会有人绕过网关自己 import。"""
    for name in ("acompletion", "completion", "embedding"):
        assert callable(getattr(router_sdk, name)), f"网关缺 {name}"


# ── 2. 改名：用户可见措辞里不出现供应商名 ────────────────────────────────────


@pytest.mark.parametrize(
    "raw",
    [
        "\nLiteLLM completion() model= deepseek-v4-flash; provider = openai",
        "litellm.InternalServerError: InternalServerError - Connection error.",
        "LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'.",
        "Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new",
        "LiteLLM: Failed to fetch remote model cost map from "
        "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices.json",
        "litellm.BadRequestError: LLM Provider NOT provided",
    ],
)
def test_rebrand_removes_vendor_name(raw: str):
    out = router_sdk.rebrand(raw)
    assert "litellm" not in out.lower(), out
    assert "berriai" not in out.lower(), out


def test_rebrand_keeps_the_diagnosable_parts():
    """只换措辞，不换语义——模型名、provider、原始错误描述必须留着。

    被我们改写过头的错误，用户拿去搜什么也搜不到，那比署别人的名更糟。
    """
    out = router_sdk.rebrand(
        "litellm.InternalServerError: OpenAIException - Connection error. "
        "model=deepseek-v4-flash provider=openai status=500"
    )
    for keep in ("Connection error", "deepseek-v4-flash", "openai", "500"):
        assert keep in out, f"{keep!r} 被改没了：{out}"
    assert out.startswith("Router.InternalServerError")


def test_debug_hint_points_at_a_real_knob():
    """改写后的提示让用户设 `BLADEX_ROUTER_LOG_LEVEL`——它必须真的生效。

    指一条敲了没反应的路，比不指还糟（ADR-0027「绿灯不许骗人」同样适用于报错）。
    """
    out = router_sdk.rebrand(
        "LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'."
    )
    assert router_sdk.LOG_LEVEL_ENV in out


def test_log_level_env_actually_resolves(monkeypatch):
    monkeypatch.setenv(router_sdk.LOG_LEVEL_ENV, "DEBUG")
    assert router_sdk.resolve_log_level() == logging.DEBUG
    monkeypatch.setenv(router_sdk.LOG_LEVEL_ENV, "not-a-level")
    assert router_sdk.resolve_log_level() == logging.INFO  # 认不出当没配，不拒启动
    monkeypatch.delenv(router_sdk.LOG_LEVEL_ENV)
    assert router_sdk.resolve_log_level() == logging.INFO


def test_error_text_is_the_single_exit():
    exc = RuntimeError("litellm.APIConnectionError: OpenAIException - Connection error.")
    out = router_sdk.error_text(exc)
    assert "litellm" not in out.lower()
    assert out.startswith("Router.APIConnectionError")


# ── 3. 日志接管 ─────────────────────────────────────────────────────────────


def test_log_bridge_strips_vendor_handlers_and_renames(caplog):
    """摘掉 SDK 自带的 handler（重复行的来源）+ 记录名改成 Router。"""
    vendor = logging.getLogger("LiteLLM")
    vendor.addHandler(logging.StreamHandler())  # 模拟 SDK import 时挂的那个
    router_sdk.install_log_bridge()

    assert vendor.handlers == [], "SDK 自带 handler 没摘干净 -> 同一条日志会打两遍"
    assert vendor.propagate is True, "要 propagate 到宿主进程才能统一格式"

    with caplog.at_level(logging.INFO):
        vendor.info("\nLiteLLM completion() model= %s; provider = %s", "m1", "openai")

    rec = caplog.records[-1]
    assert rec.name == "Router"
    assert rec.getMessage() == "Router completion() model= m1; provider = openai"
    assert not rec.getMessage().startswith("\n"), "开头的换行是 124 条空行的来源"


def test_log_bridge_is_idempotent():
    """入口点可能调多次（proxy lifespan + 测试），不该叠加 filter。"""
    router_sdk.install_log_bridge()
    router_sdk.install_log_bridge()
    vendor = logging.getLogger("LiteLLM")
    assert sum(1 for f in vendor.filters if f.__class__.__name__ == "_RebrandFilter") == 1
    assert router_sdk.log_bridge_installed()


def test_global_switches_applied_on_import():
    """开关挂在 import 上——不依赖任何调用方"记得设"（这正是 2026-08-05 的根因）。"""
    import litellm  # noqa: PLC0415 —— 本文件是网关的测试，是唯一允许直接看 SDK 的地方

    assert litellm.drop_params is True
    assert litellm.suppress_debug_info is True


# ══════════════════════════════════════════════════════════════════════════
# 🔴 D0j · 上游错误往外写必须过 error_text —— 12 个出口一个都没接（2026-09-05）
#
# 病例：Jason 用 curl 打一张图片，拿回的 502 正文是
#   {"error":{"message":"Upstream error: litellm.BadRequestError: OpenAIException - …"}}
# 而 `router_sdk.error_text` 的 docstring 写着它是「**所有**上游错误的唯一出口」，
# CLAUDE.md 编码规范也写着「对外一律叫 Router……上游错误往外写之前过一道
# router_sdk.error_text(exc)」。规范写了、**server.py 的 12 个出口（6 对 × 三协议
# × 流式/非流式）一个都没接**——与同日 MQ-L45（innerloop 说"剥离归调用方"、
# 三个调用方都没剥）完全同型：**契约写了，没有接收方**。
#
# 这一条比 L45 更贴脸：它是**用户可见面**，用户 curl 就能看见我们的依赖名。
# ══════════════════════════════════════════════════════════════════════════

def _server_src() -> str:
    from _source_probe import package_source
    return package_source("server")   # F0.1 拆包：server.py → server/ 包，按包拼接读源码


def test_upstream_errors_never_reach_the_client_raw():
    """三协议 × 流式/非流式 = 6 个出口，日志与响应各一处，共 12 处必须过 error_text。"""
    src = _server_src()
    assert 'f"Upstream error: {e}"' not in src, (
        "上游异常被直接 str() 进 HTTP 响应 —— 用户会看到 litellm.* 的类名")
    assert src.count('f"Upstream error: {router_sdk.error_text(e)}"') == 6, \
        "六个响应出口（chat/anthropic/responses × 流式/非流式）都要过 error_text"
    for ev in ("upstream_call_failed", "anthropic_upstream_call_failed",
               "responses_upstream_call_failed"):
        assert f'logger.error("{ev}", error=str(e))' not in src, \
            f"{ev} 仍把上游异常原样写进日志"
    assert src.count("error=router_sdk.error_text(e))") == 6, \
        "六个日志出口都要过 error_text"


def test_internal_errors_are_not_routed_through_error_text():
    """🔴 判别力对照：`error_text` 是**上游错误**的出口，不是万能包装。

    BladeX 自己的异常（admin 扫描失败、解析失败……）继续用 `str(e)`——
    全量替换虽然行为无害（`rebrand` 对不含供应商名的串是恒等），但会把
    "这条是我们自己的错"这个信息抹掉。首版我就手滑全替了 31 处，这条钉住边界。
    """
    src = _server_src()
    assert src.count("error=str(e))") >= 20, \
        "内部异常被一起包进 error_text 了 —— 上游出口与内部错误必须分得开"


def test_error_text_rebrands_the_real_case():
    """行为面：拿 09-05 那条真实报文验一遍，依赖名必须消失、原始描述必须留下。"""
    real = ("litellm.BadRequestError: OpenAIException - Image 0 failed: "
            "Image dimensions are too small. Minimum allowed dimension: 14 pixels.")
    out = router_sdk.rebrand(real)
    assert "litellm" not in out.lower(), out
    assert out.startswith("Router.BadRequestError"), out
    # 原始错误描述必须逐字保留——改写过的错误让用户搜不到任何东西（rebrand 的立论）
    assert "Minimum allowed dimension: 14 pixels." in out
