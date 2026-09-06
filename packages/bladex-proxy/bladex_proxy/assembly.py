"""上下文装配（Context Assembly）—— ADR-0019 P1。

以"任务单元 + 证据生命周期"取代 CAP（ADR-0016）的"对话条数老化"模型：

  TaskUnit := user 消息 + 其后的 [assistant(tool_calls)/tool]* （直到下一条 user）
  状态     := closed（其后还有 user 消息 = 任务已闭合）| open（最后一个单元）

两级降解（全部确定性、零 LLM、内容纯函数 → 会话内前缀稳定，保上游 prompt cache）：

  L1 证据降解（**常开**，不看总量——阈值门控会在越线时改写历史前缀，破坏确定性）：
     closed 单元里的巨型 tool 结果 → 占位摘录（保消息骨架与 tool_call_id 配对）。
     user 消息与 final assistant（任务结论）原样保留——**结论即蒸馏**：
     主模型读完全部证据给出的回答，就是这些证据的最优压缩，且已免费产出。
     open 单元全量保留（模型答当前问题要用证据）。

  L2 单元摘要（触发式：消息数 > msg_threshold 或 装配后体积 > budget_chars）：
     从最老的 closed 单元起整体替换为一条 Memory Index 摘要 system 消息
     （承接 CAP 段B逻辑；prefix_changed 冲突检测保留——Hermes 刚压缩则不二次压缩）。

信息守恒：Memory Hub 总账永存 agent 原文（入库走 original_messages，与装配无关）；
占位符自述"全文已存 BladeX 记忆"；P2 结论蒸馏与主动召回 = ADR-0019 P2/P3 阶段。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import structlog

from bladex_proxy.modules import module_enabled
from bladex_core.fact import Fact
from bladex_core.task_unit import build_task_units, unit_indices

logger = structlog.get_logger()

# 摘要块标记 + 体积估算：唯一定义处（2026-09-03 S3：`context_manager.py`（CAP，ADR-0016）
# 删除，三符号迁入；`inject.py` 原有的那份重复定义改为从这里 import——刚性原则 12
# "同一个值两个家 = 缺陷"）。
SUMMARY_OPEN = "<bladex-context-summary>"
SUMMARY_CLOSE = "</bladex-context-summary>"


def _msg_chars(msg: dict) -> int:
    """单条消息的字符数估算（str 全长；list 只计 text part）。"""
    content = msg.get("content")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(p.get("text", ""))
            for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return 0


def estimate_context_chars(messages: list[dict]) -> int:
    """消息列表总字符数估算（L2 体积触发 + assembly_done 的 chars_* 读数）。"""
    return sum(_msg_chars(m) for m in messages)


_EVIDENCE_MARK = "[bladex-archived-evidence"


def assembly_enabled_for(agent_id: str) -> bool:
    """装配是否对该 agent 开（2026-09-03 拍板 e，MQ-CA6；默认值真身 `flags.py`）。

    `BLADEX_ASSEMBLY_AGENTS` 逗号清单：`*` 全开；命中完整 agent_id **或其 base**
    （`hermes:default` 命中 `hermes`）即开；空清单 = 全关。每轮读 env（与 `_env`
    同款、不冻结）——它是止血开关，改了要立刻生效。
    `agent_id` 为空 = 调用方没带身份（直调 `do_inject` 的测试/脚本），无对象可 gate
    ⇒ 开（生产端点恒带 identity.agent_id，unknown-* 桶按 base 不命中即关）。
    """
    from bladex_core.flags import flag_csv
    aid = (agent_id or "").strip()
    if not aid:
        return True
    allowed = flag_csv("BLADEX_ASSEMBLY_AGENTS")
    if "*" in allowed:
        return True
    return aid in allowed or aid.partition(":")[0] in allowed


def _env(key: str, default: str, legacy_key: str | None = None) -> str:
    """读 BLADEX_ASSEMBLY_* env；未设时兼容读旧 BLADEX_CAP_* 名。"""
    val = os.environ.get(key)
    if val is not None:
        return val
    if legacy_key is not None:
        legacy = os.environ.get(legacy_key)
        if legacy is not None:
            return legacy
    return default


@dataclass
class AssemblyConfig:
    """装配配置（ADR-0019 §3；BLADEX_ASSEMBLY_* 驱动，兼容旧 BLADEX_CAP_*）。"""

    enabled: bool = True
    # L1 证据降解
    evidence_min_chars: int = 500        # tool 结果超过此长度才降解
    evidence_excerpt_chars: int = 200    # 占位摘录保留的首段长度
    keep_recent_closed_units: int = 1    # 保守带：最近 K 个 closed 单元证据保全文
    # ── U9（ADR-0025 B 层）：关联降解——按与当前任务的相关性分三档 ──
    # 默认关（G10）：关闭时 closed 单元一律走 L1 现状（= B 层的下界）。
    # BLADEX_RECONSTRUCT_B 取值：0/off=关（默认）| 1/on=全开 |
    #   ab=**会话级 A/B**（ADR-0025 T5 路线②：按 session_id hash 确定性分组，
    #   50% 开 B 层；会话内一致不破 cache（Q4 拍板粒度=会话级）。分组可 grep
    #   `assembly_b_ab_assign`，因果对照数据由此产生——R1b「无对照不得宣称正收益」）。
    reconstruct_b: bool = False
    reconstruct_b_ab: bool = False       # A/B 模式（reconstruct_b=False 时才有意义）
    b_strong_min_hits: int = 3           # 指纹命中数 ≥ 此值 → 强相关（保全文）
    b_full_cap_chars: int = 6000         # 强相关单元保全文的体积上限（超限退回摘录）
    # L2 单元摘要（承接 CAP）
    msg_threshold: int = 30
    budget_chars: int = 60000
    preserve_units: int = 3              # L2 永不动最近 N 个单元（含 open）；ADR-0020 T2: 6->3 让大上下文 drop 更多
    summary_max_chars: int = 1000
    keep_recent_tool_results: int = 4    # L1 保留最近 K 条 tool 结果全文（保守带，ADR-0020 T2 扩展到 open 单元）

    @classmethod
    def from_env(cls) -> AssemblyConfig:
        return cls(
            # V-A3.2：模块开关与既有细粒度开关是**与**关系（都默认开 = 零回归）。
            # BLADEX_MODULE_ASSEMBLY 是模块粗粒度门（注册表单一真相源）；
            # BLADEX_ASSEMBLY_ENABLED 保留为细粒度兼容通道，不再是唯一门。
            enabled=(_env("BLADEX_ASSEMBLY_ENABLED", "true", "BLADEX_CAP_ENABLED").lower() in ("true", "1", "yes")
                     and module_enabled("assembly")),
            evidence_min_chars=int(_env("BLADEX_ASSEMBLY_EVIDENCE_MIN_CHARS", "500")),
            evidence_excerpt_chars=int(_env("BLADEX_ASSEMBLY_EVIDENCE_EXCERPT_CHARS", "200")),
            keep_recent_closed_units=int(_env("BLADEX_ASSEMBLY_KEEP_RECENT_CLOSED", "1")),
            msg_threshold=int(_env("BLADEX_ASSEMBLY_MSG_THRESHOLD", "30", "BLADEX_CAP_MSG_THRESHOLD")),
            budget_chars=int(_env("BLADEX_ASSEMBLY_BUDGET_CHARS", "60000")),
            preserve_units=int(_env("BLADEX_ASSEMBLY_PRESERVE_UNITS", "3", "BLADEX_CAP_PRESERVE_TURNS")),
            summary_max_chars=int(_env("BLADEX_ASSEMBLY_SUMMARY_MAX_CHARS", "1000", "BLADEX_CAP_SUMMARY_MAX_CHARS")),
            keep_recent_tool_results=int(_env("BLADEX_ASSEMBLY_KEEP_RECENT_TOOL_RESULTS", "4")),
            reconstruct_b=_env("BLADEX_RECONSTRUCT_B", "0").lower() in ("1", "true", "yes", "on"),
            reconstruct_b_ab=_env("BLADEX_RECONSTRUCT_B", "0").lower() == "ab",
            b_strong_min_hits=int(_env("BLADEX_RECONSTRUCT_B_STRONG_MIN", "3")),
            b_full_cap_chars=int(_env("BLADEX_RECONSTRUCT_B_FULL_CAP", "6000")),
        )


@dataclass
class _Unit:
    """一个任务单元：user 起始，含其后所有非 user 消息（索引区间）。"""

    indices: list[int] = field(default_factory=list)
    closed: bool = False
    key: str = ""  # U9: TaskUnit.unit_key（B 层判定冻结的锚点，I4）


class ContextAssembler:
    """每轮上下文装配器（ADR-0019 P1 + ADR-0025 B 层关联降解，U9 gated）。"""

    # B 层判定冻结缓存的会话上限（防泄漏）
    _MAX_B_SESSIONS = 500

    def __init__(self, config: AssemblyConfig | None = None) -> None:
        self._config = config or AssemblyConfig.from_env()
        # U9/I4：B 层判定冻结——session → (open_unit_key, {unit_key: tier}, fingerprint)
        # 判定只在单元边界（open unit 变化）时重算；工具循环内逐字稳定（R2）。
        self._b_freeze: dict[str, tuple[str, dict[str, str], list[str]]] = {}

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    # ── 入口 ──

    def assemble(
        self,
        messages: list[dict],
        facts: list[Fact],
        prefix_changed: bool = False,
        session_id: str = "",
        allow_l2: bool = True,
        agent_id: str = "",
    ) -> tuple[list[dict], dict]:
        """装配：L1 证据降解（常开，B 层开启时按相关性分三档）→ L2 单元摘要（触发式）。

        返回 (assembled_messages, info)。
        info: changed / evidence_degraded / units_dropped / chars_before / chars_after
              / degrade_plan（msg_index → full|excerpt|minimal，U2 落 reconstruction）
              / b_fingerprint（B 层指纹快照，可解释性）/ b_layer（B 层是否生效）
              / l2_summary_msg + l2_dropped_unit_keys（ADR-0028 E4 冻结复用用）。

        allow_l2=False（ADR-0028 E4 温热期）：只跑 L1（内容纯函数、前缀稳定），
        **不新增** L2 摘要——上一轮已应用的 L2 由调用方按冻结计划原样复用。
        """
        info = {
            "changed": False, "evidence_degraded": 0, "units_dropped": 0,
            "chars_before": 0, "chars_after": 0,
            "degrade_plan": {}, "b_fingerprint": [], "b_layer": False,
        }
        if not self._config.enabled or not messages:
            return messages, info

        chars_before = estimate_context_chars(messages)
        info["chars_before"] = chars_before

        units = self._split_units(messages)

        # ── U9（ADR-0025 B 层，默认关）：单元相关性三档判定（冻结至单元边界）──
        idx_tier: dict[int, str] = {}
        if self._b_enabled_for(session_id) and units:
            tiers, fp_terms = self._b_decide(messages, units, facts, session_id)
            info["b_fingerprint"] = fp_terms
            info["b_layer"] = True
            for u in units:
                t = tiers.get(u.key)
                if t:
                    for i in u.indices:
                        idx_tier[i] = t

        # ── L1：closed 单元证据降解（常开，内容纯函数）──
        out, degraded = self._degrade_evidence(
            messages, units, idx_tier=idx_tier, degrade_plan=info["degrade_plan"],
        )

        # ── L2：单元摘要（触发式，承接 CAP）──
        dropped = 0
        msg_count = len(out)
        chars_now = estimate_context_chars(out)
        l2_trigger = allow_l2 and (
            msg_count > self._config.msg_threshold
            or (self._config.budget_chars > 0 and chars_now > self._config.budget_chars)
        )
        # §3.5 冲突检测（承接 ADR-0016）：Hermes 刚压缩过且规模不大 → 不二次压缩
        if l2_trigger and prefix_changed and msg_count < self._config.msg_threshold * 2:
            logger.info("assembly_skip_agent_compressed", msg_count=msg_count)
            l2_trigger = False
        if l2_trigger:
            out, dropped, _l2_plan = self._summarize_old_units(out, facts)
            info["l2_summary_msg"] = _l2_plan.get("summary_msg")
            info["l2_dropped_unit_keys"] = _l2_plan.get("dropped_unit_keys", [])

        info["evidence_degraded"] = degraded
        info["units_dropped"] = dropped
        info["chars_after"] = estimate_context_chars(out)
        info["changed"] = degraded > 0 or dropped > 0
        if info["changed"]:
            # MQ-CA3 第三处：`session_id` / `agent_id` 是**归属维度**，不是装饰。
            # 没有它们，攒下来的 chars_before 无法按会话切分——而同一会话内
            # chars_before 单调递增（08-28 那跑 441768→473940 是一条曲线上的
            # 19 个点，不是 19 个独立样本）。混池取分位数 = 把"单会话的成长轨迹"
            # 与"会话之间的差异"当成同一个量，而预算门（V-C2b）要问的正是后者。
            # 分母必须挂在被统计的对象上（08-17「参照系脱钩」同型）。
            logger.info(
                "assembly_done",
                evidence_degraded=degraded, units_dropped=dropped,
                chars_before=chars_before, chars_after=info["chars_after"],
                msgs_before=len(messages), msgs_after=len(out),
                session_id=session_id, agent_id=agent_id,
            )
        return (out if info["changed"] else messages), info

    # ── 单元化 ──

    def _split_units(self, messages: list[dict]) -> list[_Unit]:
        """按 user 消息切任务单元 —— ADR-0024 T1 后委托给 `bladex_core.task_unit`。

        本方法不再自带切分算法：`build_task_units` 是全仓库唯一的切分实现
        （消灭 ADR-0024 D1「任务结构算三遍」的第一遍与第二遍分歧）。
        这里只把 `TaskUnit` 折算成装配内部用的索引视图 `_Unit`。

        索引口径：相对**传入的 messages 列表**。装配内部对 `cleaned`（剥离注入块后）
        调用，与 `do_inject` 落 Memory Hub 的那份（agent 原始 messages）索引基不同——
        两者是同一个纯函数作用在两个列表上，不是两套算法。
        """
        return [
            _Unit(indices=unit_indices(u, messages), closed=u.closed,
                  key=getattr(u, "unit_key", ""))
            for u in build_task_units(messages)
        ]

    # ── U9（ADR-0025 B 层）：关联指纹 + 单元三档判定 + 冻结 ──

    def _b_enabled_for(self, session_id: str) -> bool:
        """B 层对本会话是否生效：on=全开；ab=会话级确定性分组（T5 路线②）。

        分组 = sha256(session_id) 首字节奇偶（确定性：同会话恒同组，会话内
        一致不破 cache——Q4 拍板粒度）。ab 模式下无 session_id → 关（保守）。
        分组结果 `assembly_b_ab_assign` 可 grep（离线评估按组对比 refetch 等指标）。
        """
        if self._config.reconstruct_b:
            return True
        if not self._config.reconstruct_b_ab or not session_id:
            return False
        import hashlib
        on = hashlib.sha256(session_id.encode()).digest()[0] % 2 == 0
        if session_id not in getattr(self, "_b_ab_logged", set()):
            if not hasattr(self, "_b_ab_logged"):
                self._b_ab_logged: set[str] = set()
            if len(self._b_ab_logged) < 2000:
                self._b_ab_logged.add(session_id)
            logger.info("assembly_b_ab_assign", session=session_id[:16],
                        group="B_on" if on else "B_off")
        return on

    @staticmethod
    def _b_fingerprint(facts: list[Fact]) -> list[str]:
        """从 prefetch 已召回的 facts 组装关联指纹（R3：不新增 embedding，零 LLM）。

        词项 = fact.subject + fact.entities（判别性锚点），去重保序，封顶 40。
        确定性：同一 facts 输入 → 同一指纹。
        """
        terms: list[str] = []
        seen: set[str] = set()
        for f in facts:
            srcs = [getattr(f, "subject", "") or ""]
            srcs.extend(str(e) for e in (getattr(f, "entities", None) or []))
            for s in srcs:
                t = s.strip().lower()
                if len(t) >= 2 and t not in seen:
                    seen.add(t)
                    terms.append(t)
            if len(terms) >= 40:
                break
        return terms[:40]

    def _b_decide(
        self,
        messages: list[dict],
        units: list[_Unit],
        facts: list[Fact],
        session_id: str,
    ) -> tuple[dict[str, str], list[str]]:
        """B 层判定：closed 单元 → full（强相关）| excerpt（弱相关）| minimal（无关）。

        判定冻结至下一单元边界（I4）：同 session 的 open unit_key 未变时，
        直接复用上次判定（工具循环内逐字稳定，R2）；新单元边界才重算指纹与档位。
        本地字符串匹配（子串包含），不新增 embedding（R3/G9）。
        """
        import time as _time
        _t0 = _time.perf_counter()
        open_key = units[-1].key if units else ""
        if session_id:
            frozen = self._b_freeze.get(session_id)
            if frozen is not None and frozen[0] == open_key:
                return frozen[1], frozen[2]

        fp = self._b_fingerprint(facts)
        tiers: dict[str, str] = {}
        for u in units:
            if not u.closed or not u.key:
                continue
            text = " ".join(
                _text_of(messages[i])[:2000].lower() for i in u.indices
            )
            hits = sum(1 for t in fp if t in text) if fp else 0
            if hits >= self._config.b_strong_min_hits:
                tiers[u.key] = "full"
            elif hits >= 1:
                tiers[u.key] = "excerpt"
            else:
                tiers[u.key] = "minimal"

        if session_id:
            if len(self._b_freeze) >= self._MAX_B_SESSIONS:
                self._b_freeze.pop(next(iter(self._b_freeze)), None)
            self._b_freeze[session_id] = (open_key, tiers, fp)
        # R3 口径（ADR-0025）：本地匹配耗时实测入日志（不新增 embedding 的另一半证据）
        logger.info("assembly_b_decided", session=session_id[:16],
                    units=len(tiers), fp_terms=len(fp),
                    full=sum(1 for v in tiers.values() if v == "full"),
                    minimal=sum(1 for v in tiers.values() if v == "minimal"),
                    elapsed_ms=round((_time.perf_counter() - _t0) * 1000, 3))
        return tiers, fp

    # ── L1：证据降解 ──

    def _degrade_evidence(
        self,
        messages: list[dict],
        units: list[_Unit],
        idx_tier: dict[int, str] | None = None,
        degrade_plan: dict[int, str] | None = None,
    ) -> tuple[list[dict], int]:
        """closed 单元（保守带之外）的巨型 tool 结果 → 占位摘录。

        只改 content，消息骨架/顺序/tool_call_id 原样 → API 配对合法。
        final assistant（单元内最后一条有正文的 assistant）与 user 原样保留。
        幂等：已降解的占位符不再二次处理。

        U9（B 层，idx_tier 非空时）：按相关性分档——
          full（强相关）：单元体积 ≤ b_full_cap_chars 时**跳过降解**（保全文）；
          excerpt（弱相关 / B 层关闭）：占位摘录（= 现状，B 层的下界）；
          minimal（无关）：只留一行元信息占位（G4：占位置换，永不删消息）。
        degrade_plan（U2）：记录每条被处置消息的档位（msg_index → 档位），
        供 reconstruction 落 Memory Hub——"模型实际看到什么"从此可复原。
        """
        # ADR-0020 T2: closed 单元 tool 全降解（证据塌缩，§3.2）；
        # open 单元 tool 保最近 K（当前任务证据），更老降解。
        idx_tier = idx_tier or {}
        open_unit = units[-1] if units else None
        open_indices = set(open_unit.indices) if open_unit else set()
        open_tools = [i for i in open_indices if messages[i].get("role") == "tool"]
        keep = self._config.keep_recent_tool_results
        open_keep = set(open_tools[-keep:]) if keep > 0 else set()
        # B 层 full 档：单元体积超上限则退回 excerpt（防强相关巨型单元撑爆上下文）
        full_cap_exceeded: set[int] = set()
        if idx_tier:
            for u in units:
                if not u.closed:
                    continue
                if idx_tier.get(u.indices[0] if u.indices else -1) != "full":
                    continue
                total = sum(len(_text_of(messages[i])) for i in u.indices)
                if total > self._config.b_full_cap_chars:
                    full_cap_exceeded.update(u.indices)
        degrade_idx: set[int] = set()
        for i, m in enumerate(messages):
            if m.get("role") != "tool":
                continue
            if i in open_keep:
                continue  # open 最近 K tool 保留
            if idx_tier.get(i) == "full" and i not in full_cap_exceeded:
                if degrade_plan is not None:
                    degrade_plan[i] = "full"
                continue  # B 层强相关：保全文
            degrade_idx.add(i)  # closed tool 全降解 + open 更老 tool 降解

        if not degrade_idx:
            return messages, 0

        out: list[dict] = []
        degraded = 0
        for i, msg in enumerate(messages):
            if i not in degrade_idx:
                out.append(msg)
                continue
            tier = idx_tier.get(i, "excerpt")
            if tier == "full":
                tier = "excerpt"  # full 但超体积上限 → 摘录
            # MQ-CA2：占位符尾句按单元归属选。`open_indices` 上面已算好
            # （open 单元 = units[-1]），这里只是把它带到文案层——
            # 降解与否的判定一个字都没改，closed 路径逐字不变。
            new_msg, did = self._degrade_tool_msg(
                msg, minimal=(tier == "minimal"), open_unit=(i in open_indices),
            )
            out.append(new_msg)
            degraded += did
            if did and degrade_plan is not None:
                degrade_plan[i] = tier
        return out, degraded

    def _degrade_tool_msg(
        self, msg: dict, minimal: bool = False, open_unit: bool = False,
    ) -> tuple[dict, int]:
        """单条 tool 消息降解（content 纯函数；短结果/已降解的原样返回）。

        minimal=True（U9 无关档）：只留一行元信息（不带摘录），仍是占位置换（G4）。
        open_unit=True（MQ-CA2）：占位符尾句改说"任务进行中"——见 `_placeholder`。
        """
        content = msg.get("content")
        min_chars = self._config.evidence_min_chars
        excerpt_n = 0 if minimal else self._config.evidence_excerpt_chars
        name = msg.get("name", "") or ""

        if isinstance(content, str):
            if len(content) <= min_chars or content.startswith(_EVIDENCE_MARK):
                return msg, 0
            return {**msg, "content": _placeholder(content, name, excerpt_n, open_unit)}, 1

        if isinstance(content, list):
            total = sum(
                len(p.get("text", "")) for p in content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            )
            if total <= min_chars:
                return msg, 0
            texts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and isinstance(p.get("text"), str)
            ]
            joined = "\n".join(t for t in texts if t)
            if joined.startswith(_EVIDENCE_MARK):
                return msg, 0
            new_parts: list = [
                p for p in content
                if not (isinstance(p, dict) and isinstance(p.get("text"), str))
            ]
            new_parts.insert(
                0,
                {"type": "text", "text": _placeholder(joined, name, excerpt_n, open_unit)},
            )
            return {**msg, "content": new_parts}, 1

        return msg, 0

    # ── L2：单元摘要 ──

    def _summarize_old_units(
        self,
        messages: list[dict],
        facts: list[Fact],
    ) -> tuple[list[dict], int, dict]:
        """从最老 closed 单元起整体替换为一条摘要 system 消息（承接 CAP 段B）。

        永不动最近 preserve_units 个单元（含 open）；逐个 drop 直到
        消息数 ≤ msg_threshold 且体积 ≤ budget_chars，或无可 drop。

        ADR-0028 E4：第三个返回值是**计划**（summary_msg + 被摘要单元的 unit_key），
        供温热期原样复用——摘要正文含 Memory Index 事实，每轮重算会变，重算即打碎前缀。
        """
        units = self._split_units(messages)
        _empty_plan: dict = {"summary_msg": None, "dropped_unit_keys": []}
        max_droppable = len(units) - max(1, self._config.preserve_units)
        if max_droppable <= 0:
            return messages, 0, _empty_plan

        droppable = [u for u in units[:max_droppable] if u.closed]
        if not droppable:
            return messages, 0, _empty_plan

        drop_indices: set[int] = set()
        dropped_units = 0
        dropped_user_texts: list[str] = []
        dropped_keys: list[str] = []

        def _within_limits() -> bool:
            remaining = [m for i, m in enumerate(messages) if i not in drop_indices]
            n = len(remaining) + 1  # +1 摘要消息
            c = estimate_context_chars(remaining)
            return n <= self._config.msg_threshold and (
                self._config.budget_chars <= 0 or c <= self._config.budget_chars
            )

        for u in droppable:
            if _within_limits():
                break
            for i in u.indices:
                msg = messages[i]
                if msg.get("role") == "user":
                    dropped_user_texts.append(_text_of(msg))
                drop_indices.add(i)
            dropped_units += 1
            if u.key:
                dropped_keys.append(u.key)

        if not dropped_units:
            return messages, 0, _empty_plan

        summary_msg = self._build_summary(facts, dropped_user_texts)

        # 摘要插在最后一条"前导 system"之后（保 system 前缀稳定）
        out: list[dict] = []
        inserted = False
        for i, msg in enumerate(messages):
            if i in drop_indices:
                if not inserted:
                    out.append(summary_msg)
                    inserted = True
                continue
            out.append(msg)
        if not inserted:
            out.insert(0, summary_msg)
        return out, dropped_units, {
            "summary_msg": summary_msg, "dropped_unit_keys": dropped_keys,
        }

    def _build_summary(self, facts: list[Fact], dropped_user_texts: list[str]) -> dict:
        """摘要消息：Memory Index 蒸馏事实 + 被 drop 单元的 user 首 50 字符兜底（去重）。

        承接 ADR-0016 §3.3 语义（含 fact/snippet 互含去重）。
        """
        lines: list[str] = ["以下是之前对话的关键信息摘要："]
        fact_contents: list[str] = []
        for fact in facts:
            if fact.category == "hard_rule":
                continue
            lines.append(f"- {fact.content}")
            if fact.content:
                fact_contents.append(fact.content.strip().lower())

        seen: set[str] = set()
        for text in dropped_user_texts:
            snippet = text.strip()[:50]
            if not snippet:
                continue
            low = snippet.lower()
            if low in seen:
                continue
            if any(fc in low or low in fc for fc in fact_contents):
                continue
            seen.add(low)
            lines.append(f"- {snippet}")

        body = "\n".join(lines)
        if len(body) > self._config.summary_max_chars:
            body = body[: self._config.summary_max_chars].rsplit("\n", 1)[0]
        return {"role": "system", "content": f"{SUMMARY_OPEN}\n{body}\n{SUMMARY_CLOSE}"}


def _text_of(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


def _placeholder(
    content: str,
    tool_name: str,
    excerpt_chars: int,
    open_unit: bool = False,
) -> str:
    """证据占位摘录（内容纯函数 → 每轮逐字一致，保 prompt cache）。

    excerpt_chars=0（U9 minimal 档）：只留一行元信息，不带摘录。

    open_unit（MQ-CA2，2026-08-28）：**尾句必须说真话**。
    closed 单元「该任务已完成、结论见其后的 assistant 回复」是 ADR-0019
    「结论即蒸馏」的前提，成立；ADR-0020 T2 把降解外推到 open 单元后，
    这句对 open 单元的两个断言**都假**——任务没完成，其后也没有 assistant 回复。
    模型据此以为那一步已有结论，于是既不重取证据也不重做。
    降解本身是策略选择，说假话不是。

    🔴 closed 路径（open_unit=False）逐字不变 —— 那是本次改动的阴性对照，
    也是上游 prompt cache 的稳定面：closed 单元的占位符一旦变字节，
    历史前缀整体失效。
    """
    name = tool_name or "tool"
    if excerpt_chars <= 0:
        if open_unit:
            return (
                f"{_EVIDENCE_MARK} tool={name} chars={len(content)} unrelated]\n"
                f"…(与当前任务无关而折叠；需要时可重新调用工具获取)"
            )
        return (
            f"{_EVIDENCE_MARK} tool={name} chars={len(content)} unrelated]\n"
            f"…(与当前任务无关；证据全文已存 BladeX 记忆)"
        )
    excerpt = content[:excerpt_chars]
    if open_unit:
        return (
            f"{_EVIDENCE_MARK} tool={name} chars={len(content)} open]\n"
            f"{excerpt}\n"
            f"…(当前任务较早的工具输出，已折叠为摘录；**结论尚未产生**，"
            f"需要完整内容请重新调用工具)"
        )
    return (
        f"{_EVIDENCE_MARK} tool={name} chars={len(content)}]\n"
        f"{excerpt}\n"
        f"…(该任务已完成，结论见其后的 assistant 回复；证据全文已存 BladeX 记忆)"
    )
