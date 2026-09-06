"""Router —— BladeX 与上游模型 SDK 之间的唯一网关（全仓库唯一 `import litellm` 的地方）。

## 为什么要有这一层（2026-08-05，真实日志驱动）

借来的多模型适配 SDK 是**通用品**（CLAUDE.md「通用品借用、护城河自建」），但借来的
**日志面与异常措辞不是**——它们会直接漏到用户眼前。实测 consolidator 一次运行的日志里：

    544  LiteLLM completion() model= deepseek-v4-flash; provider = openai
    124  [INFO    ] LiteLLM                          <- 上一条开头那个 \\n 造成的空行
    100  [INFO    ] LiteLLM   Wrapper: Completed Call, calling success_handler
     23  Give Feedback / Get Help: https://github.com/BerriAI/litellm/issues/new
     23  LiteLLM.Info: If you need to debug this error, use `litellm._turn_on_debug()'.
      1  LiteLLM: Failed to fetch remote model cost map from raw.githubusercontent.com/...
     10  distill_failed err=litellm.InternalServerError: OpenAIException - Connection error.

对用户来说这些是**别人家产品的自言自语**：让人以为 BladeX 出了问题、给的调试指令
（`litellm._turn_on_debug()`）在 BladeX 里根本敲不出来、报 issue 的链接指向别人的仓库。

本模块把这件事收成一处：

1. **进程级开关只设一次**。此前 `drop_params` / `suppress_debug_info` 写在 `route.py`
   模块级——consolidator 根本不 import `route.py`，于是那两个开关在 consolidator 进程里
   **从未生效**，红色 banner 照刷。全局开关必须挂在"谁都会经过"的地方。
2. **日志接管**。SDK 在 import 时给自己的三个 logger 各挂了一个带颜色的 StreamHandler，
   而那三个 logger 又 propagate 到 root——同一条消息被打**两遍**（上面 124/100 的差额
   就是重复行）。这里摘掉它自带的 handler，只留 propagate，交给宿主进程统一格式化。
3. **改名**。logger 名与消息正文里的供应商措辞统一换成 `Router`（用户拍板 2026-08-05）。
   INFO 级别保留——它确实有"这轮发给了哪个模型"的运维价值，只是不该署别人的名。
4. **异常翻译**。`error_text(exc)` 是所有往日志/HTTP 响应里写上游错误的唯一出口。

## 边界

- 这里**不做**路由决策。选哪个模型是 `route.py` + `bladex_core.routing` 的事，
  本模块只负责"把选定的调用发出去"以及"把外来的噪音挡在门口"。
  （模块名叫 `router_sdk` 而不是 `router`，就是为了不和 `route.py` 混。）
- 也**不包装**返回值：调用方拿到的仍是 SDK 原生的 response / stream 对象，
  `capture.py` 那套自迭代流式捕获逐字不变。包一层 DTO 只会多一个失真面。
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

# 🔴 必须在 import litellm **之前**：SDK 在 import 时会去
# raw.githubusercontent.com 拉模型价格表，离线/网络受限时甩一条署名 LiteLLM 的
# WARNING（实测每次启动一条）。本地表足够——BladeX 不用它的计费功能。
# setdefault：用户显式设了就尊重（想要线上价格表的人不该被我们锁死）。
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm  # noqa: E402  （上面那行 env 是它的前置条件，顺序不能调）

# ── 进程级开关（import 本模块即生效，不依赖调用方记得设）─────────────────────

# drop 上游模型不支持的参数（如 reasoning_effort——Hermes 发送，ARK 系模型不支持
# → UnsupportedParamsError 502）。
litellm.drop_params = True

# 关掉 SDK 直接 print() 到 stderr 的红色 banner（"Provider List: ..." /
# "Give Feedback / Get Help: <别人的 issue 页>"）。
# 2026-07-29 实测触发路径：`anthropic/doubao-seed-2.0-lite` 每次调用都刷一次——SDK
# 剥掉 provider 前缀后，内部旁路（cost/token 统计）拿裸名再做一次 provider 推断，
# 推断不出来就 print banner 并抛 BadRequestError，异常被它自己吞掉，主调用照常成功。
# 46 次调用 = 46 次 banner，46 次 turn_enqueued 全部正常——纯噪音。
# 它走的是 print() 不是 logging，只能用这个全局开关关掉。
litellm.suppress_debug_info = True


# ── 改名（用户可见措辞里不出现供应商名）──────────────────────────────────────

BRAND = "Router"
"""用户可见的名字。上游 SDK 是实现细节，不是产品面。"""

LOG_LEVEL_ENV = "BLADEX_ROUTER_LOG_LEVEL"
"""调 Router 这一层日志详略的旋钮（DEBUG/INFO/WARNING/ERROR，默认 INFO）。

它必须**真的生效**：改写后的错误提示里会让用户去设这个变量，而 ADR-0027 那条
"绿灯不许骗人"同样适用于报错——指一条敲了没反应的路，比不指还糟。
生效点在 `install_log_bridge()`。
"""

_VENDOR_LOGGERS = ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy")

# 有序替换：长串在前，否则短串先命中会把长串切碎。
_REBRAND_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    # 指向别人仓库的求助链接 —— 用户照着点也解决不了 BladeX 的问题。
    (re.compile(r"\s*Give Feedback / Get Help:\s*https?://\S*litellm\S*"), ""),
    (re.compile(r"https?://raw\.githubusercontent\.com/BerriAI/litellm/\S+"),
     "the upstream cost-map registry"),
    (re.compile(r"https?://\S*BerriAI/litellm\S*"), f"the {BRAND} backend"),
    # 敲不出来的调试指令 —— 换成 BladeX 里真能敲的那条（`LOG_LEVEL_ENV` 真实生效，见下）。
    (re.compile(r"If you need to debug this error, use `litellm\._turn_on_debug\(\)'?\.?"),
     f"Set {LOG_LEVEL_ENV}=DEBUG for the full upstream trace."),
    (re.compile(r"litellm\._turn_on_debug\(\)"), f"{LOG_LEVEL_ENV}=DEBUG"),
    # 异常类名：`litellm.InternalServerError:` → `Router.InternalServerError:`
    (re.compile(r"\blitellm\.(?=[A-Z])"), f"{BRAND}."),
    (re.compile(r"\bLiteLLM\.Info\b"), f"{BRAND}.Info"),
    (re.compile(r"\bLiteLLM\b"), BRAND),
    (re.compile(r"\blitellm\b"), BRAND.lower()),
)


def rebrand(text: str) -> str:
    """把一段文字里的供应商措辞换成 BladeX 的说法。

    只动措辞，不动语义——状态码、模型名、原始错误描述全部保留，
    否则用户拿着一条被我们改写过的错误去搜，什么也搜不到。
    """
    if not text:
        return text
    for pattern, repl in _REBRAND_RULES:
        text = pattern.sub(repl, text)
    return text


def error_text(exc: BaseException) -> str:
    """上游异常 → 一句可写进日志/响应的话（**所有**上游错误的唯一出口）。

    用法：`logger.warning("distill_failed", err=router_sdk.error_text(e))`。
    直接 `str(e)` 会把 `litellm.InternalServerError: ...` 原样漏出去。
    """
    return rebrand(str(exc)).strip()


class _RebrandFilter(logging.Filter):
    """改写供应商 logger 的记录名与正文。

    为什么用 filter 而不是自建 handler：filter 挂在**产生记录的 logger** 上，
    在 `Logger.handle()` 里、propagate 到 root 之前执行一次——所以宿主进程
    （proxy 的 structlog / consolidator 的 basicConfig）不用改任何配置，
    拿到的记录已经是改过名的。挂 handler 则要求每个宿主自己配一遍。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.name = rebrand(record.name)
        # 先 getMessage() 展开 %s 参数再改写：SDK 有把 provider 名塞进 args 的写法，
        # 只改 msg 模板会漏。展开后清空 args，避免下游二次格式化。
        try:
            message = record.getMessage()
        except Exception:  # 参数不匹配是 SDK 的 bug，不该让日志链路跟着炸
            message = str(record.msg)
        # SDK 有几条消息以 "\n" 开头（见文件头 124 条空行的来源），顺手削掉。
        record.msg = rebrand(message).strip()
        record.args = ()
        return True


_filter = _RebrandFilter()
_bridge_installed = False


def resolve_log_level() -> int:
    """Router 层日志级别：env `BLADEX_ROUTER_LOG_LEVEL` > INFO。

    认不出的值当没配（不是拒启动）——日志旋钮不该成为起不来的理由。
    """
    raw = (os.environ.get(LOG_LEVEL_ENV) or "").strip().upper()
    return getattr(logging, raw, logging.INFO) if raw else logging.INFO


def install_log_bridge(level: int | None = None) -> None:
    """接管上游 SDK 的日志（进程入口调一次；重复调用无副作用）。

    做三件事：
    1. 摘掉 SDK 自带的彩色 StreamHandler —— 它的 formatter 把 "LiteLLM" 硬编码在
       格式串里（`%(asctime)s - %(name)s:%(levelname)s: %(filename)s:%(lineno)s`），
       改不动；而且它与 propagate 到 root 的那份**重复**，同一条打两遍。
    2. 挂改名 filter。
    3. 保持 propagate=True —— 让宿主进程的格式统一生效。
    """
    global _bridge_installed
    if level is None:
        level = resolve_log_level()
    for name in _VENDOR_LOGGERS:
        lg = logging.getLogger(name)
        for handler in list(lg.handlers):
            lg.removeHandler(handler)
        if not any(isinstance(f, _RebrandFilter) for f in lg.filters):
            lg.addFilter(_filter)
        lg.setLevel(level)
        lg.propagate = True
    _bridge_installed = True


def log_bridge_installed() -> bool:
    """给测试与 `bladex doctor` 用：日志接管是否已生效。"""
    return _bridge_installed


# ── 调用面（薄封装，签名与上游一致）──────────────────────────────────────────


async def acompletion(**kwargs: Any) -> Any:
    """异步 completion（流式/非流式都走这里）。返回上游原生对象。"""
    return await litellm.acompletion(**kwargs)


def completion(**kwargs: Any) -> Any:
    """同步 completion（consolidator 侧的蒸馏 / L4 裁决用；非热路径）。"""
    return litellm.completion(**kwargs)


def embedding(**kwargs: Any) -> Any:
    """线上 embedding API（`BLADEX_EMBED_BACKEND=api` 档）。"""
    return litellm.embedding(**kwargs)


def retryable_error_types() -> tuple[type[BaseException], ...]:
    """值得 failover 的上游异常类型（超时 / 连接错）。

    用 getattr 取而不是直接 import：SDK 版本间这些名字动过，取不到就返回空元组，
    调用方退回按 status_code 判定——宁可少切一次 failover，不能因为 SDK 改名而崩。
    """
    named = tuple(
        t for t in (getattr(litellm, "Timeout", None), getattr(litellm, "APIConnectionError", None))
        if isinstance(t, type)
    )
    return named
