"""AgencyRuntime —— 工具面/拦截协议/账本的运行时装配与 server 接线面（批二启用卡 V-P5a）。

server.py 只拿到三个薄调用点（inbound 准备 / tools 增补 / 响应处置），全部先查
`modules.module_enabled`——**四个模块默认关，本文件在默认形态下零行为差异**
（三红线之 3；关着时每个入口第一行短路返回）。

# 组成

- ToolFace 真实 handler：memory_search 接 MemoryIndex（结果过 exposure 门，
  ADR-0032 §3.2-7）；ledger_* 接账本池 + ActivationTable，写路径全部落 Hub
  管理事件（池 = 事件重放的投影，重启 `_load_pool` 重建）。
- SpliceLedger + strip/splice：入站先剥回流（D6/D7）再拼接（V-P4）。
- 内循环：`run_inner_loop`，call_llm 由 server 侧以当轮路由偏应用注入。

# 🔴 接线纪律

- Hub 是唯一真相：账本事件经 `append_admin_event`（`AdminEvent.matter_id` 留空，
  MQ-A7 护栏）；拼接台账经 `Turn.splice_records` 入 Hub（V-P4 已落地）。
  内循环轮次经 `loop_ledger`（`bladex_proxy.loopledger.InnerLoopLedger`）缓冲、
  `server._enqueue_turn` 随主轮 drain 后以 aux 轮入 Hub（MQ-P9，2026-09-02 落地；
  `SpliceRecord` 仍只在**混合路径**产生，两者分工：拼接台账 vs 支出凭证）。
  数"模型调了多少次 bladex 工具"读 `Turn.tool_events`（拦截前抽取，两条路都在；
  MQ-N8）——aux 轮自身的 `tool_events` 是第 2 轮起的调用，尺子按 `auxiliary` 分档。
- 拦截决不制造空响应/悬空调用（hermes 重试指纹 / D2 报错形态）。
- 每请求 `RecentRequests` 去重：codex 超时整请求重发 ≥6 次实测——重发不得
  重触发内循环（返回上次结果的合成文本，宁可重复内容不重复花钱）。


# 09-06 F0.1：本文件原为 2,365 行的 `agency.py`，零行为拆成包——
#   runtime.py  `AgencyRuntime` 本体 + 注入块常量 + `tool_context`
#   handlers.py ToolFace handlers（mixin）+ 账本写路径辅助
#   streams.py  三条流式拦截 + `_Narrator` 族 + 合成事件
#   notes.py    模板 / 系统自我介绍 / AGENTS 名册 / `_last_user_text`
# 本模块只做门面：re-export 全部原公开名与测试引用的私有名（守卫 `test_facade_exports.py`）。
# 🔴 monkeypatch 目标要打在**消费方模块**上（如 `bladex_proxy.agency.runtime.tool_context`），
#    打在门面上只换门面的引用、消费方看不到。
"""

from __future__ import annotations

from typing import Any

import structlog

from bladex_proxy.agency.notes import (  # noqa: F401
    _MACHINE_TEXT_MARKS, _machine_text_mark, load_ledger_template, _SYSTEM_NOTE_FILES,
    load_system_notes, load_agents_roster, render_system_notes,
    LEDGER_SECTION_OPEN, LEDGER_SECTION_CLOSE, _GOAL_MAX_CHARS, _WRAPPER_TAG_RE, _BARE_PREFIX_RES,
    _last_user_text
)
from bladex_proxy.agency.handlers import (  # noqa: F401
    SUBTASK_TAG, _DESPITE_MATCH_SCORE, USER_NAMED_TITLE_MIN_CHARS, _user_named_ledger,
    ToolFaceHandlersMixin
)
from bladex_proxy.agency.runtime import (  # noqa: F401
    LEDGER_BLOCK_OPEN, LEDGER_BLOCK_CLOSE, ABOUT_BLOCK_OPEN, ABOUT_BLOCK_CLOSE,
    INJECTION_MARKERS, _FIRST_STEP_WITH_LEDGER, _FIRST_STEP_NO_LEDGER, _MATCHING_LEDGERS_HEADER,
    _call_names, _loop_cost, tool_context, AgencyRuntime,
    LEDGER_BLOCK_BUCKETS, LedgerBlockBreakdown, _split_ledger_md,
    rare_pending_units, _pending_text
)
from bladex_proxy.agency.streams import (  # noqa: F401
    intercept_chat_stream, _SSE_KEEPALIVE, _Narrator, _ev, _ResponsesNarrator,
    _AnthropicNarrator, _ChatNarrator, _idle_too_long, _narrate_on, _keepalive_interval_s,
    _narrate_until_done, _intercept_protocol_stream, _rewrite_stop_anthropic,
    _rewrite_stop_responses, _synth_responses_tool_call_events,
    _synth_anthropic_tool_call_events, _synth_anthropic_text_events, _SYNTH_ITEM_ID,
    _synth_responses_text_events, intercept_anthropic_stream, intercept_responses_stream
)

logger = structlog.get_logger()


def build_agency(app_state: Any) -> AgencyRuntime:
    """lifespan 装配：吃 app.state 上已有的 index/ledger（Hub），失败不阻断启动。"""
    return AgencyRuntime(index=getattr(app_state, "index", None),
                         hub=getattr(app_state, "hub", None))
