"""拦截协议的分流与剥离原语（ADR-0032 §3.2；批二卡 V-P2）。

纯函数 + 无 IO 状态机，零 litellm 依赖——沙盒可测。消费方：server 各端点
（接线随启用卡，挂 `BLADEX_MODULE_INTERCEPTION`，默认关=零行为差异）。

三分流（§3.2 拍板 #2）：

    none         无 bladex_* 调用 → 原样透传
    pure_bladex  只有 bladex_* 调用（且剥离后无实质内容）→ 内循环
                 ——不可剥空转发：hermes 对空 assistant 有重试指纹（D2 依据）
    mixed        两者都有 → 剥离-拼接（执行己方、删调用、agent 返回后拼回）

流式剥离（D 实测约束）：工具名出现在该调用的**首个** delta（chat 的 index 首片 /
anthropic 的 content_block_start / responses 的 output_item.added）⇒ 可即时决定
拦或放，只需按 index 重编号，缓冲开销 O(1)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bladex_proxy.toolface import is_bladex_tool

MODE_NONE = "none"
MODE_PURE = "pure_bladex"
MODE_MIXED = "mixed"


@dataclass
class Disposition:
    """一次响应的分流结果。"""

    mode: str
    bladex_calls: list[dict] = field(default_factory=list)   # OpenAI tool_call dicts
    agent_calls: list[dict] = field(default_factory=list)


def mint_call_id() -> str:
    """铸一个非空 tool_call id（MQ-P8）。

    上游（实测 deepseek-v4-flash）流式 delta 可以**不带 id**：我们把空 id 发给
    agent，agent 自己铸一个回传 ⇒ 拼接锚"记录侧无 id 退化哈希 / 回传侧真 id"
    结构性永不相等（哈希级实证见台账 MQ-P8）。规则：**永不外发空 id**——
    发射权威（各剥离器）在首个 fragment/block/item 上铸，wire 与锚记录同源。
    """
    import uuid
    return f"call_{uuid.uuid4().hex[:24]}"


def ensure_call_ids(tool_calls: list[dict]) -> None:
    """就地补齐空 id（非流式 mixed 路径用；有 id 的一律不动）。"""
    for tc in tool_calls or []:
        if not tc.get("id"):
            tc["id"] = mint_call_id()


def _call_name(tc: dict) -> str:
    fn = tc.get("function") if isinstance(tc, dict) else None
    if isinstance(fn, dict):
        return fn.get("name", "") or ""
    return tc.get("name", "") or ""     # anthropic tool_use block 形态


def classify_message(message: dict) -> Disposition:
    """OpenAI assistant message → 分流。判据（§3.2）：**剥离后还剩不剩 agent 调用**。

    🔴 2026-08-26 修正（live 病例：Hermes"看起来卡死"）。首版判据是"还剩 agent 调用
    **或非空正文**"，把两件不同的事当成了一件：

    - "还剩能转发的东西"（有正文 ⇒ 有）
    - "**这一轮该结束了**"（有正文 ⇒ **完全不能推出**）

    实测那一轮模型返回的是正文 `"…先把证据归档到任务账本，再给结论。"` +
    两个 `bladex_ledger_update`，零 agent 调用。按旧判据判 MIXED ⇒ 剥掉我方调用后
    转发出去的只剩这段正文，且因为转发调用数为 0 而把 `finish_reason` 改写成
    `stop` ⇒ **Hermes 认为这一轮结束了**，用户看到的就是"说要给结论然后没了"。

    **前言不是回答。** 模型明说"再给结论"，它是要拿着工具结果继续的——
    这恰恰是我们希望的良好行为，却被协议截断。所以：**只要 agent 自己没有调用，
    这一轮就不可能完整**，必须走内循环（执行我方工具 → 把结果喂回去 → 让模型
    自己决定还有没有话说）。让模型决定也符合三红线（作者是模型，不是我们）。

    代价：模型只是"记一笔然后收尾"时会多一次上游往返。这个代价必须付——
    我们没有可靠判据区分"前言"与"收尾语"，而**内循环本身就是在问模型这个问题**。
    """
    calls = message.get("tool_calls") or []
    ours = [tc for tc in calls if is_bladex_tool(_call_name(tc))]
    theirs = [tc for tc in calls if not is_bladex_tool(_call_name(tc))]
    if not ours:
        return Disposition(MODE_NONE, agent_calls=theirs)
    return Disposition(MODE_MIXED if theirs else MODE_PURE,
                       bladex_calls=ours, agent_calls=theirs)


def strip_bladex_calls(message: dict) -> tuple[dict, list[dict]]:
    """非流式剥离：删 bladex_* 调用，agent 调用**原顺序原 id** 保留。
    返回 (剥离后的消息, 被剥的调用)。纯函数，入参不改。"""
    calls = message.get("tool_calls") or []
    ours = [tc for tc in calls if is_bladex_tool(_call_name(tc))]
    theirs = [tc for tc in calls if not is_bladex_tool(_call_name(tc))]
    out = dict(message)
    if theirs:
        out["tool_calls"] = theirs
    else:
        out.pop("tool_calls", None)
    return out, ours


# ── 流式剥离：chat 协议（tool_calls delta 带 index）──────────────────────────


class ChatStreamStripper:
    """OpenAI chat 流式 chunk 过滤器。

    用法：`out_chunks = stripper.feed(chunk)` 逐 chunk 喂；结束后
    `stripper.removed_calls` = 被拦截调用的聚合形态（name/arguments/id）。

    机制：每个 tool_call index 的**首个 delta 必带 name**（协议保证，D 实测确认）
    ⇒ 见到即决定拦/放；放行的 index 重编号成连续序（agent 侧解析器按 index 聚合，
    跳号会让它建出空洞）。`finish_reason=tool_calls` 在全拦截且无其它内容时改写为
    `stop` 由调用方决定——本类不改 finish_reason（分流判定在 capture 聚合层做，
    这里只管剥）。
    """

    def __init__(self) -> None:
        self._decision: dict[int, bool] = {}      # src index -> intercept?
        self._renumber: dict[int, int] = {}       # src index -> dst index
        self._next_dst = 0
        self._removed: dict[int, dict] = {}       # src index -> aggregated call
        self._forwarded: dict[int, dict] = {}     # src index -> 已发射形态聚合（MQ-P8）

    @property
    def removed_calls(self) -> list[dict]:
        out = []
        for idx in sorted(self._removed):
            acc = self._removed[idx]
            out.append({"id": acc.get("id", ""), "type": "function",
                        "function": {"name": acc.get("name", ""),
                                     "arguments": acc.get("arguments", "")}})
        return out

    @property
    def forwarded_calls(self) -> list[dict]:
        """放行给 agent 的调用（**已发射形态**，含铸的 id）——拼接锚的唯一合法来源
        （MQ-P8：capture 看到的是上游原始形态，id 可能为空；agent 回传的是这里的形态）。"""
        out = []
        for idx in sorted(self._forwarded, key=lambda i: self._renumber.get(i, i)):
            acc = self._forwarded[idx]
            out.append({"id": acc.get("id", ""), "type": "function",
                        "function": {"name": acc.get("name", ""),
                                     "arguments": acc.get("arguments", "")}})
        return out

    def feed(self, chunk: dict) -> list[dict]:
        choices = chunk.get("choices") or []
        if not choices:
            return [chunk]
        choice = choices[0]
        delta = choice.get("delta") or {}
        tcs = delta.get("tool_calls")
        if not tcs:
            return [chunk]
        kept: list[dict] = []
        for tc in tcs:
            idx = tc.get("index", 0)
            if idx not in self._decision:
                name = _call_name(tc)
                intercept = is_bladex_tool(name)
                self._decision[idx] = intercept
                if intercept:
                    self._removed[idx] = {"id": tc.get("id", ""), "name": name,
                                          "arguments": ""}
                else:
                    self._renumber[idx] = self._next_dst
                    self._next_dst += 1
                    # MQ-P8：放行调用首个 fragment 无 id ⇒ 铸——永不外发空 id。
                    # `minted` 标记留给下面的发射逻辑：上游本来带 id 的流必须
                    # **逐字节透传**（none 零差异红线，test_none_passthrough_matrix
                    # 抓过一次"顺手把后续 fragment 的 id:null 改写掉"的越权）。
                    self._forwarded[idx] = {
                        "id": tc.get("id") or mint_call_id(),
                        "minted": not tc.get("id"),
                        "name": name, "arguments": ""}
            if self._decision[idx]:
                acc = self._removed[idx]
                if tc.get("id"):
                    acc["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    acc["name"] = fn["name"]
                if fn.get("arguments"):
                    acc["arguments"] += fn["arguments"]
                continue
            facc = self._forwarded[idx]
            fn = tc.get("function") or {}
            if fn.get("name"):
                facc["name"] = fn["name"]
            if fn.get("arguments"):
                facc["arguments"] += fn["arguments"]
            tc2 = dict(tc)
            tc2["index"] = self._renumber[idx]
            # 发射形态与记录同源，但**最小干预**（none 零差异红线：上游带 id 的
            # 流必须逐字节透传——test_none_passthrough_matrix 抓过一次"顺手把
            # 后续 fragment 的 id:null 改写掉"的越权）：
            # ① 铸过的调用：首个 kept fragment 盖铸 id；此后只有上游迟到 id
            #    （会与已发射的分叉）才覆写成权威值，其余 fragment 原样；
            # ② 上游带 id 的调用：除"中途换 id"这种分叉外一概不碰。
            if facc.get("minted"):
                if not facc.get("emitted"):
                    tc2["id"] = facc["id"]
                    facc["emitted"] = True
                elif tc.get("id"):
                    tc2["id"] = facc["id"]
            elif tc.get("id") and tc.get("id") != facc["id"]:
                tc2["id"] = facc["id"]
            kept.append(tc2)
        if not kept:
            return []          # 整个 chunk 只剩被拦调用 → 吞掉
        out_delta = dict(delta)
        out_delta["tool_calls"] = kept
        out_choice = dict(choice)
        out_choice["delta"] = out_delta
        out = dict(chunk)
        out["choices"] = [out_choice]
        return [out]


# ── 流式剥离：Anthropic messages（content_block 事件按 index）────────────────


class AnthropicStreamStripper:
    """Anthropic SSE 事件过滤器（事件已解析为 dict）。

    `content_block_start` 的 tool_use 块自带 name ⇒ 即时决定；被拦截块的
    start/delta/stop 全吞，其余块 index 重编号。收集 input_json_delta 聚合参数。
    """

    def __init__(self) -> None:
        self._decision: dict[int, bool] = {}
        self._renumber: dict[int, int] = {}
        self._next_dst = 0
        self._removed: dict[int, dict] = {}
        self._forwarded: dict[int, dict] = {}     # MQ-P8：已发射形态（含铸 id）

    @property
    def removed_calls(self) -> list[dict]:
        out = []
        for idx in sorted(self._removed):
            acc = self._removed[idx]
            out.append({"id": acc.get("id", ""), "type": "function",
                        "function": {"name": acc.get("name", ""),
                                     "arguments": acc.get("arguments", "") or "{}"}})
        return out

    @property
    def forwarded_calls(self) -> list[dict]:
        """放行 tool_use 的已发射形态（MQ-P8：拼接锚唯一合法来源）。"""
        out = []
        for idx in sorted(self._forwarded, key=lambda i: self._renumber.get(i, i)):
            acc = self._forwarded[idx]
            out.append({"id": acc.get("id", ""), "type": "function",
                        "function": {"name": acc.get("name", ""),
                                     "arguments": acc.get("arguments", "")}})
        return out

    def feed(self, event: dict) -> list[dict]:
        etype = event.get("type", "")
        if etype == "content_block_start":
            idx = event.get("index", 0)
            block = event.get("content_block") or {}
            intercept = block.get("type") == "tool_use" and is_bladex_tool(block.get("name", ""))
            self._decision[idx] = intercept
            if intercept:
                self._removed[idx] = {"id": block.get("id", ""),
                                      "name": block.get("name", ""), "arguments": ""}
                return []
            self._renumber[idx] = self._next_dst
            self._next_dst += 1
            out_ev = self._reindexed(event, idx)
            if block.get("type") == "tool_use":
                # MQ-P8：放行 tool_use 无 id ⇒ 铸，并盖进发射事件（agent 回传的
                # 就是这个 id）。深拷贝 content_block 防止改写调用方的 dict。
                minted = block.get("id") or mint_call_id()
                self._forwarded[idx] = {"id": minted,
                                        "name": block.get("name", ""),
                                        "arguments": ""}
                if not block.get("id"):
                    out_ev = dict(out_ev)
                    out_ev["content_block"] = {**block, "id": minted}
            return [out_ev]
        if etype in ("content_block_delta", "content_block_stop"):
            idx = event.get("index", 0)
            if self._decision.get(idx):
                if etype == "content_block_delta":
                    delta = event.get("delta") or {}
                    if delta.get("type") == "input_json_delta":
                        self._removed[idx]["arguments"] += delta.get("partial_json", "")
                return []
            if etype == "content_block_delta" and idx in self._forwarded:
                delta = event.get("delta") or {}
                if delta.get("type") == "input_json_delta":
                    self._forwarded[idx]["arguments"] += delta.get("partial_json", "")
            return [self._reindexed(event, idx)]
        return [event]

    def _reindexed(self, event: dict, idx: int) -> dict:
        dst = self._renumber.get(idx, idx)
        if dst == idx:
            return event
        out = dict(event)
        out["index"] = dst
        return out


# ── 流式剥离：OpenAI Responses（output_item 事件按 output_index）─────────────


class ResponsesStreamStripper:
    """Responses SSE 事件过滤器。`response.output_item.added` 的 function_call
    item 自带 name ⇒ 即时决定；该 item 的 arguments delta/done 与 item.done 全吞；
    其余 item 的 output_index 重编号。"""

    _ITEM_EVENTS = ("response.output_item.added", "response.output_item.done")
    _ARG_EVENTS = ("response.function_call_arguments.delta",
                   "response.function_call_arguments.done")
    #: 🔴 其余**同样携带 `output_index`** 的事件（2026-08-25 Codex live 事故）：
    #: 首版只对上面两族重编号，`output_text.delta` / `content_part.*` 被原样放行
    #: ⇒ 剥掉前面的 bladex item 后，message item 的 index 被改成 0、而它的文本
    #: 增量还在说 index=1 ⇒ Codex 对不上就断连（`client disconnected`，
    #: 用户侧表现为"只输出了一段话就中断"）。
    #: **判据不写死事件名**：凡带 `output_index` 的都要跟随同一张重编号表——
    #: 协议将来加新事件（如 refusal/annotation 族）不必改这里。
    _PASSTHROUGH_TYPES = ("response.created", "response.in_progress",
                          "response.completed", "response.incomplete",
                          "response.failed")

    def __init__(self) -> None:
        self._decision: dict[int, bool] = {}
        self._renumber: dict[int, int] = {}
        self._next_dst = 0
        self._removed: dict[int, dict] = {}
        self._forwarded: dict[int, dict] = {}     # MQ-P8：已发射形态（含铸 id）
        self._minted_items: dict[str, str] = {}   # item.id -> 铸的 call_id（快照盖章用）

    @property
    def forwarded_calls(self) -> list[dict]:
        """放行 function_call 的已发射形态（MQ-P8：拼接锚唯一合法来源）。"""
        out = []
        for idx in sorted(self._forwarded, key=lambda i: self._renumber.get(i, i)):
            acc = self._forwarded[idx]
            out.append({"id": acc.get("call_id", ""), "type": "function",
                        "function": {"name": acc.get("name", ""),
                                     "arguments": acc.get("arguments", "")}})
        return out

    def stamp_snapshot_items(self, items: list[dict]) -> int:
        """把铸的 call_id 盖进终止快照的 function_call item（按 item.id 对应）。

        流内 item 已带铸 id、快照还是上游的空 id ⇒ Codex 对账分叉（08-25 断连
        事故同族）。返回盖章数。"""
        stamped = 0
        for it in items or []:
            if (it or {}).get("type") == "function_call" and not it.get("call_id"):
                minted = self._minted_items.get(str(it.get("id", "")))
                if minted:
                    it["call_id"] = minted
                    stamped += 1
        return stamped

    @property
    def removed_calls(self) -> list[dict]:
        out = []
        for idx in sorted(self._removed):
            acc = self._removed[idx]
            out.append({"id": acc.get("call_id", ""), "type": "function",
                        "function": {"name": acc.get("name", ""),
                                     "arguments": acc.get("arguments", "") or "{}"}})
        return out

    def feed(self, event: dict) -> list[dict]:
        etype = event.get("type", "")
        if etype in self._ITEM_EVENTS:
            idx = event.get("output_index", 0)
            item = event.get("item") or {}
            if etype == "response.output_item.added":
                intercept = (item.get("type") == "function_call"
                             and is_bladex_tool(item.get("name", "")))
                self._decision[idx] = intercept
                if intercept:
                    self._removed[idx] = {"call_id": item.get("call_id", ""),
                                          "name": item.get("name", ""),
                                          "arguments": item.get("arguments", "")}
                    return []
                self._renumber[idx] = self._next_dst
                self._next_dst += 1
                if item.get("type") == "function_call":
                    minted = item.get("call_id") or mint_call_id()
                    self._forwarded[idx] = {"call_id": minted,
                                            "name": item.get("name", ""),
                                            "arguments": item.get("arguments", "")}
                    if not item.get("call_id"):
                        self._minted_items[str(item.get("id", ""))] = minted
                        ev2 = dict(self._reindexed(event, idx))
                        ev2["item"] = {**item, "call_id": minted}
                        return [ev2]
            elif self._decision.get(idx):
                if item.get("arguments"):
                    self._removed[idx]["arguments"] = item["arguments"]
                return []
            else:
                # item.done：放行 item 的快照带上权威 call_id（与 added 同源）
                if (item.get("type") == "function_call" and idx in self._forwarded
                        and item.get("call_id") != self._forwarded[idx]["call_id"]):
                    if item.get("arguments"):
                        self._forwarded[idx]["arguments"] = item["arguments"]
                    ev2 = dict(self._reindexed(event, idx))
                    ev2["item"] = {**item, "call_id": self._forwarded[idx]["call_id"]}
                    return [ev2]
                if item.get("type") == "function_call" and idx in self._forwarded                         and item.get("arguments"):
                    self._forwarded[idx]["arguments"] = item["arguments"]
            return [self._reindexed(event, idx)]
        if etype in self._ARG_EVENTS:
            idx = event.get("output_index", 0)
            if self._decision.get(idx):
                if etype.endswith(".delta"):
                    self._removed[idx]["arguments"] = (
                        self._removed[idx].get("arguments", "") + event.get("delta", ""))
                elif event.get("arguments"):
                    self._removed[idx]["arguments"] = event["arguments"]
                return []
            if idx in self._forwarded:
                if etype.endswith(".delta"):
                    self._forwarded[idx]["arguments"] = (
                        self._forwarded[idx].get("arguments", "") + event.get("delta", ""))
                elif event.get("arguments"):
                    self._forwarded[idx]["arguments"] = event["arguments"]
            return [self._reindexed(event, idx)]
        # 其余带 output_index 的事件（output_text.delta / content_part.* / …）：
        # 被拦 item 的全吞，其余跟随同一张重编号表（见 _PASSTHROUGH_TYPES 注释）。
        if etype not in self._PASSTHROUGH_TYPES and "output_index" in event:
            idx = event.get("output_index", 0)
            if self._decision.get(idx):
                return []
            return [self._reindexed(event, idx)]
        return [event]

    def _reindexed(self, event: dict, idx: int) -> dict:
        dst = self._renumber.get(idx, idx)
        if dst == idx:
            return event
        out = dict(event)
        out["output_index"] = dst
        return out


def all_kept_empty(strippers_removed: list[dict], kept_agent_calls: list[Any],
                   content_text: str) -> bool:
    """流式收尾后的分流判定帮手：剥完既无 agent 调用也无正文 = pure 路径
    （本应走内循环——调用方据此决定是否补发合成收尾/重放，语义同 classify_message）。"""
    return bool(strippers_removed) and not kept_agent_calls and not content_text.strip()
