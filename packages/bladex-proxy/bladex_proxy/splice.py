"""剥离-拼接的台账与入站处理（ADR-0032 §3.2；批二卡 V-P4）。

# 语义（Jason 2026-08-25 拍板的混合调用方案）

出站（响应侧，V-P2 已剥）：BladeX 执行己方工具、持有结果；下发 agent 的消息**不含**
bladex 调用 ⇒ agent 侧历史自洽（无悬空调用）。

入站（下一请求）：agent 带自己的工具结果回来 ⇒ 在锚定的 assistant 消息里**恢复**
bladex 调用并插入其结果，一并转发上游 ⇒ LLM 看到完整的工具交互史。

# 🔴 协议硬点（§3.2-1/2/4/5）

- **台账 = Hub 投影**：`SpliceRecord` 随 turn 入 Hub，进程内表可由 Hub 重建；
  丢失 = 优雅降级（agent 侧历史本就自洽，LLM 只是失忆那次查询）。
- **锚点消失即弃**：agent 压缩后找不到锚 ⇒ 静默丢弃该条，不报错不补救。
- **老化**：条目 `max_age_turns` 后退出（默认不定值，flags 可配；退出接受一次性
  cache 破裂）。
- **幂等/去重**（D4 实测：codex 超时后整请求重发 ≥6 次）：`request_fingerprint`
  给调用方做"同一请求重发"识别，内循环与工具执行不得重触发。

# 入站剥离（V-P4 T3，D6/D7 双证据，最高优先）

- 注入回流：CC 把含 `<bladex-memory>` 的历史当 user 回传（119/120 轮实测）；
- 播报回带：CC/Pi/dsh 把 thinking/reasoning 播报存档回传（D7 实测）。
两者都**只剥我们自己的块**（标记识别），agent 与用户的正文一个字不碰。
**播报与注入只存在于 BladeX↔agent 之间，LLM 与真上游永远看不到。**
（注意与 `inject.strip_previous_injection` 的分工：那边管我们上一轮注进去的块、
限 system 角色且"agent 正文含标记不剥"；这边管 agent **回流**进 user/assistant
角色的块——正是那边刻意不碰、由此累积的那部分。）
"""

from __future__ import annotations

import hashlib
import json
import re
import time

from pydantic import BaseModel, Field

#: 播报块标记（V-P3 narrate 用它开头；入站按它剥）。生产标记与探针
#: （BLX-NARRATE-PROBE）分开——探针文本不该被生产剥离规则静默吃掉。
NARRATE_MARK = "[bladex-live]"

_MEMORY_BLOCK_RE = re.compile(
    r"<bladex-(?:memory|ledger)>.*?</bladex-(?:memory|ledger)>\s*", re.DOTALL)


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def anchor_key(stripped_message: dict) -> str:
    """拼接锚：**剥离后**的 assistant 消息指纹（agent 存档并回传的正是这个形态）。

    取 content + agent 调用的 id 序列——两者是 agent 侧逐字保留的部分；
    不取 bladex 调用（agent 看不到它们）。

    🔴 content 先 `.strip()` 再哈希（2026-08-25 首轮真实流量实测）：流式捕获的
    全文尾带 `"\\n\\n"`，hermes 存档回传时把结尾空白 rstrip 掉 → 哈希不等 →
    拼接被"锚失即弃"误伤（`anchors_dropped=1 spliced=0`，bladex 工具结果
    没送到下一轮 LLM）。空白归一是安全的：两条不同消息不会因 strip 撞锚。
    """
    content = str(stripped_message.get("content") or "").strip()
    ids = ",".join((tc.get("id") or _call_name_args_hash(tc))
                   for tc in stripped_message.get("tool_calls") or [])
    return _hash(f"{content}\n#{ids}")


def _call_name_args_hash(tc: dict) -> str:
    fn = tc.get("function") or {}
    return _hash(f"{fn.get('name','')}|{fn.get('arguments','')}")


class SpliceRecord(BaseModel):
    """一次剥离-拼接的记录（随 turn 入 Hub；进程内表 = 它的投影）。"""

    session_prefix: str
    anchor: str                       # anchor_key(剥离后消息)
    calls: list[dict] = Field(default_factory=list)     # 被剥的 bladex tool_calls
    results: list[dict] = Field(default_factory=list)   # 对应 role=tool 消息
    created_ms: int = 0
    age_turns: int = 0


class SpliceLedger:
    """进程内拼接表（Hub 投影）。`restore()` 由 Hub 记录重建——重启不致失忆；
    重建失败 = 优雅降级（见模块头注）。"""

    def __init__(self, *, max_age_turns: int = 0) -> None:
        self._by_session: dict[str, list[SpliceRecord]] = {}
        self._max_age = max_age_turns          # 0 = 不老化（默认；标定后配）

    def add(self, rec: SpliceRecord) -> None:
        self._by_session.setdefault(rec.session_prefix, []).append(rec)

    def restore(self, records: list[SpliceRecord]) -> None:
        for r in records:
            self.add(r)

    def records(self, session_prefix: str) -> list[SpliceRecord]:
        return list(self._by_session.get(session_prefix, []))

    def new_in_turn(self, session_prefix: str) -> list[SpliceRecord]:
        """本轮**新增**的条目（`age_turns == 0`），供落 Hub。

        🔴 只存新增、不存全量快照：`tick_and_prune` 每轮 +1，所以 0 就是"本轮加的"。
        存全量 = 每轮把整个会话的拼接表再写一遍 = O(n²)——ADR-0008 §2.2
        「P3 每轮存全量历史」正是这个坑，本仓已经踩过一次，不再踩第二次。
        """
        return [r for r in self._by_session.get(session_prefix, [])
                if r.age_turns == 0]

    def tick_and_prune(self, session_prefix: str) -> int:
        """每轮调用一次：age+1，超龄退出。返回退出数。"""
        recs = self._by_session.get(session_prefix, [])
        if not recs:
            return 0
        for r in recs:
            r.age_turns += 1
        if self._max_age <= 0:
            return 0
        kept = [r for r in recs if r.age_turns <= self._max_age]
        dropped = len(recs) - len(kept)
        self._by_session[session_prefix] = kept
        return dropped


def splice_into_messages(
    messages: list[dict], records: list[SpliceRecord],
) -> tuple[list[dict], int, int]:
    """入站拼接：按锚找到 assistant 消息，恢复 bladex 调用并在其工具结果段插入
    我们的结果。返回 (新消息列表, 应用数, 弃置数)。

    - 锚匹配不到 ⇒ 弃（压缩/改写；§3.2-2 静默）。
    - 恢复的调用**追加在 agent 调用之后**（顺序对上游语义无关，追加最不扰动）；
      结果消息插在该 assistant 消息现有 tool 结果之后、下一条非 tool 消息之前
      ——OpenAI 语义要求每个 tool_call 有对应 role=tool 消息紧随其后成段。
    - 纯函数：不改入参。幂等：已含我们调用 id 的锚跳过（重发请求二次拼接防护）。
    """
    if not records:
        return messages, 0, 0
    by_anchor = {r.anchor: r for r in records}
    out: list[dict] = []
    applied = 0
    matched: set[str] = set()
    i = 0
    while i < len(messages):
        m = messages[i]
        out.append(m)
        if m.get("role") == "assistant":
            key = anchor_key(m)
            rec = by_anchor.get(key)
            if rec is not None and key not in matched:
                matched.add(key)
                existing_ids = {tc.get("id") for tc in m.get("tool_calls") or []}
                new_calls = [tc for tc in rec.calls if tc.get("id") not in existing_ids]
                if new_calls:
                    m2 = dict(m)
                    m2["tool_calls"] = list(m.get("tool_calls") or []) + new_calls
                    out[-1] = m2
                    # 跳过 agent 自己的 tool 结果段，再插我们的结果
                    j = i + 1
                    while j < len(messages) and messages[j].get("role") == "tool":
                        out.append(messages[j])
                        j += 1
                    out.extend(rec.results)
                    i = j
                    applied += 1
                    continue
        i += 1
    dropped = len(records) - applied
    return out, applied, dropped


# ── 入站剥离（T3）────────────────────────────────────────────────────────────


def strip_inbound_echoes(messages: list[dict]) -> tuple[list[dict], int]:
    """剥掉 agent 回流的我们自己的块。返回 (新消息列表, 剥除块数)。

    覆盖两类（只认标记，agent/用户正文零接触）：
      ① user 角色 content（str 或 list[text block]）里的 `<bladex-memory>…</>` 与
         `<bladex-ledger>…</>` 块（账本注入同样会被 CC 回流，D6 同理）；
      ② assistant 角色里回带的播报——chat 形态的 `reasoning_content` 含
         `NARRATE_MARK` 整字段删；anthropic 形态的 content list 里
         `type=thinking` 且 thinking 含标记的整块删。
    """
    out: list[dict] = []
    stripped = 0
    for m in messages:
        role = m.get("role", "")
        m2 = m
        if role in ("user", "tool"):
            # 🔴 tool 角色也要剥（2026-08-25 live 实测：hermes 把含 <bladex-memory>
            # 的历史塞进 tool 结果回传，`marker_in_non_system_message role=tool ×4`
            # ——旧 strip 按设计不碰 agent 正文，于是这部分逐轮累积）。
            m2, n = _strip_memory_blocks(m)
            stripped += n
        elif role == "assistant":
            m2, n = _strip_narrate(m)
            stripped += n
            m2, n2 = _strip_memory_blocks(m2)
            stripped += n2
        out.append(m2)
    return out, stripped


def _has_block_marker(s: str) -> bool:
    # 快速路径：两种块标记都要认（首版只查 memory，ledger 块整块漏剥——测试当场抓到）。
    return "<bladex-memory>" in s or "<bladex-ledger>" in s


def _strip_memory_blocks(m: dict) -> tuple[dict, int]:
    c = m.get("content")
    if isinstance(c, str):
        if not _has_block_marker(c):
            return m, 0
        new, n = _MEMORY_BLOCK_RE.subn("", c)
        m2 = dict(m)
        m2["content"] = new
        return m2, n
    if isinstance(c, list):
        n = 0
        new_list = []
        for blk in c:
            if (isinstance(blk, dict) and blk.get("type") == "text"
                    and _has_block_marker(str(blk.get("text", "")))):
                txt, k = _MEMORY_BLOCK_RE.subn("", blk["text"])
                n += k
                if txt.strip():
                    new_list.append({**blk, "text": txt})
                continue
            new_list.append(blk)
        if not n:
            return m, 0
        m2 = dict(m)
        m2["content"] = new_list
        return m2, n
    return m, 0


def _strip_narrate(m: dict) -> tuple[dict, int]:
    n = 0
    m2 = m
    rc = m.get("reasoning_content")
    if isinstance(rc, str) and NARRATE_MARK in rc:
        m2 = dict(m2)
        m2.pop("reasoning_content", None)
        n += 1
    c = m2.get("content")
    if isinstance(c, list):
        kept = []
        for blk in c:
            if (isinstance(blk, dict) and blk.get("type") == "thinking"
                    and NARRATE_MARK in str(blk.get("thinking", ""))):
                n += 1
                continue
            kept.append(blk)
        if len(kept) != len(c):
            m2 = dict(m2)
            m2["content"] = kept
    return m2, n


# ── 重发去重（T4）────────────────────────────────────────────────────────────


def request_fingerprint(messages: list[dict]) -> str:
    """同一入站请求的指纹（codex 超时重发识别）。取 messages 的结构化哈希——
    重发是逐字节同一份 body，锚在内容上即可；不含时间戳。"""
    return _hash(json.dumps(messages, ensure_ascii=False, sort_keys=True))


class RecentRequests:
    """(fingerprint → 首见时间) 的小型环形表：`seen()` 判重发。容量硬顶防泄漏。"""

    def __init__(self, *, capacity: int = 256, window_s: float = 600.0) -> None:
        self._cap = capacity
        self._window = window_s
        self._seen: dict[str, float] = {}

    def seen(self, fingerprint: str, *, now: float | None = None) -> bool:
        t = time.monotonic() if now is None else now
        # 清窗口外与超容量（按最旧驱逐）
        expired = [k for k, ts in self._seen.items() if t - ts > self._window]
        for k in expired:
            del self._seen[k]
        hit = fingerprint in self._seen
        if not hit:
            if len(self._seen) >= self._cap:
                oldest = min(self._seen, key=self._seen.__getitem__)
                del self._seen[oldest]
            self._seen[fingerprint] = t
        return hit
