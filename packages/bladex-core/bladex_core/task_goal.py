"""G12.3 `task_goal` 捕获（ADR-0031 §4.1「取」不是「炼」）。

权威：`docs/planning/g12-3-goal-criteria-20260822.md`（判据 A1–A3 / B / C / D / E
+ 阴性 N1–N10 + 阳性 P1/P2，Jason 2026-08-22 过目）——改这里先改那份判据。

# 🔴 红线（ADR-0031 §4.1，不许"好心修正"）
#
# 1. **取，不是炼。** 本模块只做"哪一段是目标陈述"的**定位**，绝不概括、改写、
#    截断存储。goal 的全部价值是**用户一眼认出是自己说的**——改一个字就没了。
#    例：用户说"帮我搞定泰山啤酒的破产重整尽调，重点看股权变动，不用管财务报表"，
#    炼成「泰山啤酒破产重整尽调」丢掉两个限定，而用户看不出哪被改了。
# 2. **只有用户能改 goal。** 模型能改目标 ⇒ 防漂移**自我拆台**：漂移的模型会把
#    目标改成它正在做的事，然后声称没漂。故本模块只在**新开一件事**那一刻产出
#    一次，之后任何路径不得改写（唯一写入点在 memory_index._apply_decision；
#    用户覆盖发生在**渲染层**，不回写真相层——判据 D1–D4）。

## 为什么准入判据是"来源"而不是"长度"

设计文档 §9.9 把问题写成「原样引用需要长度上限」。2026-08-22 对 live Memory Hub
5103 轮取读数（生产 `classify_auxiliary` 重判）**推翻了这个定义**：

    会话首轮代理集 n=200：真实用户目标 96% ≤600 字符；
    而 >600 的长尾几乎全是**机器文本**——最长 8 条里 6 条是 Codex 内部注入
    （19713 字符）/ 裁判脚手架 / 截图包壳 / runtime 快照，真实用户目标只有 1 段。

⇒ 按"截断到 N"处理，等于把一条 19713 字符的 Codex 内部提示截成 600 字符、
再当作"你的目标"写进用户能读的 Flash 文件——**把「不该取」变成「取了一段」，
比不取更糟**。与 ADR-0024 T3「不是长消息问题，是信封问题」同型第二次。

**存不截断**（可核对性）；注入体积由渲染层按 §6.7 的"goal 一行"预算处理
（600 字符，Jason 2026-08-22 拍板，落点 G12.5——本模块不声明那个常量，
免得又造一个无消费者的字段）。

## goal 是准入缺陷的显影剂

`SCAFFOLD` 轮（`DISTILL_ONLY_AUX_RULES` 那批）**可以开卡**——rebuild 只对
Hermes 原生 aux 整轮丢，信封类轮次 assistant 侧照收，因而能产 fact、能走 L5 新开。
首轮代理集里它占 13.0%。再加上指纹表的缺口（MQ-S44 后台进程通知 / MQ-S45 Codex
history 近似未命中），会有机器文本轮成为一件事的"首轮"。

**本模块不在 goal 层修准入**（那是 D1 / 指纹表的事，且在 0.1.0 范围冻结之外），
而是把它**变得可见**：拒绝取值时留分档 `reason`，调用方按档计数。
这把尺子量的就是"有多少卡是被机器文本开出来的"。

## 重建等价性

v1 的 goal 是首轮 user 文本的**纯函数**（`strip_envelopes` 确定性、零 LLM），
重放同一条 Memory Hub → 同一个 goal。**不需要进 judgment 台账**；
G12.7 的 LLM 定位（"哪一段是目标陈述"）才需要，届时 goal 读台账不重跑。
"""

from __future__ import annotations

from dataclasses import dataclass

from .envelope import strip_envelopes

# ── 轮次处置三分（生产判据的唯一定义处）────────────────────────────────
# 与 memory_index.rebuild_from_hub 的分支逐字对应，**仪器与测试从这里读、
# 不另抄字面量**（硬约束 8；MQ-S4「筛子和被测系统各算各的」正是抄字面量的病）。
# 映射 (auxiliary, aux_rule) → 本三分：`identity.classify_turn_disposition`。
TURN_REAL = "real"          # 非 aux：正常用户轮
TURN_SCAFFOLD = "scaffold"  # aux ∧ 规则 ∈ DISTILL_ONLY_AUX_RULES：user 侧是脚手架，
                            # assistant 侧照收 ⇒ **能开卡**，但 goal 必须拒
TURN_DROPPED = "dropped"    # aux ∧ 规则 ∉ 该集合：rebuild 整轮丢，永远当不了首轮
TURN_CLASSES: tuple[str, ...] = (TURN_REAL, TURN_SCAFFOLD, TURN_DROPPED)

# ── 来源标记（ADR-0031 §4.5：agent 说的 ≠ 我们蒸出来的 ≠ 我们从历史补的）──
GOAL_SOURCE_USER_FILE = "user_file"            # 第 1 级：用户手写文件（G12.5 落点，v1 不产）
GOAL_SOURCE_AGENT_DECLARED = "agent_declared"  # 第 2 级：agent 自产目标声明（v1 不产，见下）
GOAL_SOURCE_FIRST_TURN_USER = "first_turn_user"  # 第 3 级：首轮用户消息（v1 唯一实现）
GOAL_SOURCES: tuple[str, ...] = (
    GOAL_SOURCE_USER_FILE, GOAL_SOURCE_AGENT_DECLARED, GOAL_SOURCE_FIRST_TURN_USER,
)
# 🔴 第 1/2 级 v1 **有意不产**：文件在 G12.4/G12.5 才存在；agent 自产声明没有跨 agent
# 的确定性生产者，硬做就是又一个"机制存在生产不走"（`open_issues` 前车之鉴）。
# H1 纪律下这条注释就是署名——两个取值的生产者见上述任务号。

# ── 未取到的分档原因（封闭集；新加一档先让 test_task_goal 的守卫红一次）──
GOAL_ABSENT_SCAFFOLD = "scaffold_turn"        # A1：脚手架轮开的卡
GOAL_ABSENT_DROPPED = "dropped_turn"          # A1：整轮丢的轮次（正常到不了这里）
GOAL_ABSENT_EMPTY = "empty_after_strip"       # A2：剥完信封没剩下东西
GOAL_ABSENT_NO_KEY = "no_ledger_key"          # A3：fact 没有 source_ledger_key
GOAL_ABSENT_NO_CONTEXT = "no_turn_context"    # 调用路径拿不到那一轮（如 rejudge_pending）
GOAL_ABSENT_REASONS: tuple[str, ...] = (
    GOAL_ABSENT_SCAFFOLD, GOAL_ABSENT_DROPPED, GOAL_ABSENT_EMPTY,
    GOAL_ABSENT_NO_KEY, GOAL_ABSENT_NO_CONTEXT,
)


@dataclass(frozen=True)
class GoalCapture:
    """一次捕获的结果。`text` 为空 ⇔ `reason` 非空（空着比编造好）。"""

    text: str = ""
    source: str = ""
    reason: str = ""
    ledger_key: str = ""

    @property
    def captured(self) -> bool:
        return bool(self.text)


def capture_first_turn_goal(
    *,
    ledger_key: str,
    user_text: str,
    turn_class: str,
) -> GoalCapture:
    """从一件事的首轮取 goal（判据 A1→A2→A3 顺序短路，纯函数）。

    `turn_class` 取 `TURN_CLASSES` 之一，由调用方用**生产**判据算好
    （`identity.classify_turn_disposition`）——🔴 不许读冻结的
    `turn.identity.auxiliary`：2026-08-22 取读数时第一跑就栽在这里，
    指纹明明命中的轮被算成真实用户轮，整跑作废。

    `user_text` 传**末条 user 消息原文**（`query_understanding.last_user_text`），
    本函数负责剥信封——即 ADR-0031 §14.4 要求的"goal 捕获在 envelope 剥离与
    aux 判定**之后**"。理由不是流程洁癖：模型产的文本以 user role 回流
    （transcript-in-user、`<bladex-memory>` 回声、子 agent 转录），
    **那是"只有用户能改 goal"这条防线的现成绕过路径**。
    """
    # fail-closed：类别不在闭集里 = 我们不知道这轮是什么，**不猜**。
    # （不新造一档 reason：语义上就是"拿不到可信的轮次上下文"。）
    if turn_class not in TURN_CLASSES:
        return GoalCapture(reason=GOAL_ABSENT_NO_CONTEXT, ledger_key=ledger_key)
    if turn_class == TURN_SCAFFOLD:
        return GoalCapture(reason=GOAL_ABSENT_SCAFFOLD, ledger_key=ledger_key)
    if turn_class == TURN_DROPPED:
        return GoalCapture(reason=GOAL_ABSENT_DROPPED, ledger_key=ledger_key)

    clean, _kinds = strip_envelopes(user_text or "")
    if not clean:
        return GoalCapture(reason=GOAL_ABSENT_EMPTY, ledger_key=ledger_key)

    if not ledger_key:
        return GoalCapture(reason=GOAL_ABSENT_NO_KEY)

    # 🔴 定位 = 剥离后全文，**不挑段、不截断**。任何"挑哪一段"的启发式都是炼的
    # 变种；按 ADR-0031 §7.2，定位本就是首轮 LLM 的第 ② 件事，归 G12.7。
    # 本函数因此正是 §7.5 明写的那条确定性地板：「LLM 失败 → 用首轮原文当 Goal」。
    return GoalCapture(
        text=clean, source=GOAL_SOURCE_FIRST_TURN_USER, ledger_key=ledger_key,
    )
