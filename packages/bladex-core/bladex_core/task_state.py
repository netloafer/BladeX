"""D1 延续/新开判定引擎（G12.2 本体·核心；ADR-0031 §3，算法 = 决策表 v2）。

权威：`docs/planning/g12-2-decision-table-20260820.md`（v2，Jason 2026-08-20 定稿）
——**那张表是本模块的全部算法内容**，本文件只是它的代码形态；改这里先改表。

# 红线 7（ADR-0031 §3.2）：延续是时序局部的、合并是全局的，两者风险不同、
# 不该用同一个保守度。本模块的"默认延续"不违反误合并=0 红线——
# 误合并红线管的是全局合并（merge），本模块管的是相邻轮次的归属（延续）。
# 不要把这里的默认方向改回"默认新开/默认无归属"，那是 80% 碎片率的直接来源。

# 红线 6（ADR-0031 §2）：本模块只判"这一轮属于哪份档案"，不判"这件事该怎么做"。
# 任何往判定里加"下一步"语义的改动都是越界（BladeX 不做作者）。

结构（与表逐节对应）：

    AnchorTracker   延续锚，per-(agent, session)——"默认延续"延续的是锚，
                    不是窗口里随便哪张卡（表 §1）；aux 轮不动锚（表 §2）
    ActiveWindow    活跃档案窗口，per-principal（跨 agent 免费的前提），
                    最近活跃 K 份，只供候选不供默认方向
    decide_turn     R1–R5 五行决策表本体（纯函数；R5 只判"需要裁决"，
                    裁决与降级由调用方做——LLM 属 proxy 侧，本模块零 IO）

零 LLM 直通只有两条路（表 v2 修订：S3 键/S5 显式信号只提名候选、不做判据——
08-18 一天撞过五次的纪律）：S2 unit_key 唯一命中，或 manual（manual 在调用方
更上游拦截，压过本引擎一切分支）。

设计约束：叶子模块，只依赖同层叶子 `continuation`。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .continuation import DEFAULT_NEW_SUBJECT_MAX, is_task_continuation

#: 活跃窗口大小（ADR-0031 §3.3：3–5 份）。窗口大小属"新开判定旋钮"的组成部分：
#: 改值 = 改 R3/R4 的候选域 = 可能改首轮集合 → 与 margin/门槛一起进 knob_ver。
DEFAULT_WINDOW_K = 5

#: 决策表分支名（可 grep；edge.decision["branch"] / taskstate 台账 signals 用）。
BRANCH_R1_UNIT_OVERRIDE = "R1"   # 锚在，unit_key 命中窗口内另一份 → 延续到它
BRANCH_R2_ANCHOR = "R2"          # 锚在，S1 成立 → 延续到锚（默认路径）
BRANCH_R4_UNIT = "R4"            # 锚缺席，窗口内 unit_key 唯一命中
BRANCH_R5_JUDGE = "R5"           # 疑似新事/不确定 → 需要裁决（新开判定）
BRANCH_AUX = "AUX"               # aux 轮：四不（不新开/不当首轮/不动锚/不改窗口）
# 注：表 v2 的 R3① （S1 不成立后窗口内 unit_key 唯一命中）在实现里**折叠进 R1**
# ——unit_key 检查放在 S1 之前，两行动作相同（延续到命中卡），只差审计粒度；
# 不为审计粒度留一个永不产出的分支常量（H1 枚举取值对账刚抓过第七例）。

#: D1 产出的归属边统一层名——误延续仪器按 `decision["layer"]` 分层报数，
#: CONT 层落地后自动出现在它的读数里（probe_member_coherence 设计如此）。
CONT_LAYER = "CONT"

#: 🔴 新开判定旋钮的版本串（TaskStateJudgment.knob_ver）。组成旋钮：
#: DEFAULT_WINDOW_K、new_subject_max（BLADEX_ATTRIB_CONT_NEW_SUBJECT_MAX）、
#: 分支结构本身。**标定任何一个 = bump 本串**（改产物要可对账——与蒸馏
#: prompt_ver 同一条纪律；旋钮值变更 = 重建产物变更，按执行卡 §4 约束 B 攒窗口）。
D1_KNOB_VER = "d1-knob-001"


@dataclass
class TurnDecision:
    """一轮的判定结果。verdict 语义与 TaskStateJudgment 对齐（台账只收 R5 的）。"""

    branch: str                       # BRANCH_* 之一
    verdict: str                      # continue_to | needs_judgment | aux_skip
    matter_id: str = ""               # continue_to 的目标；needs_judgment 为空
    anchor_matter_id: str = ""
    anchor_alive: bool = False
    window_snapshot: list[str] = field(default_factory=list)
    signals: dict = field(default_factory=dict)


class AnchorTracker:
    """延续锚：per-(agent_id, session_id) 记最近一条**非 aux**轮的归属卡。

    🔴 作用域是 (agent, session) 不是 principal——并发双流（Claude Code 与 Codex
    同 principal 同时做两件事）下 per-principal 的"上一轮"会让 B 默认延续进 A
    刚激活的档案（表 §1 反例检验）。**这两个作用域不许合并成一个。**
    """

    def __init__(self) -> None:
        # (agent_id, session_id) -> (matter_id, 该轮参照文本 user+assistant)
        self._state: dict[tuple[str, str], tuple[str, str]] = {}

    def get(self, agent_id: str, session_id: str) -> tuple[str, str]:
        """返回 (锚 matter_id, 锚轮参照文本)；无锚 = ("", "")。"""
        return self._state.get((agent_id, session_id), ("", ""))

    def update(self, agent_id: str, session_id: str, matter_id: str,
               reference_text: str, *, auxiliary: bool = False) -> None:
        """轮次落定后更新锚。aux 轮**不动锚**（表 §2 四不之一）——
        claude-code 38% 的轮是内部调用，aux 断锚 = 大量真实轮掉进无锚分支，
        碎片病换个入口再生产。"""
        if auxiliary or not matter_id:
            return
        self._state[(agent_id, session_id)] = (matter_id, reference_text)


class ActiveWindow:
    """活跃档案窗口：per-principal 最近活跃 K 份（只供候选，不供默认方向）。

    activity 时钟 = 成员轮的 ledger stream ms（调用方传入；不要用
    Matter.created_at——全量重建会把它压成重建那二十分钟，probe_matter_split 教训）。
    """

    def __init__(self, k: int = DEFAULT_WINDOW_K) -> None:
        self._k = k
        self._active: dict[str, int] = {}   # matter_id -> last stream ms

    def touch(self, matter_id: str, stream_ms: int, *, auxiliary: bool = False) -> None:
        """aux 轮不改窗口状态（表 §2）。"""
        if auxiliary or not matter_id:
            return
        prev = self._active.get(matter_id, -1)
        if stream_ms >= prev:
            self._active[matter_id] = stream_ms

    def remove(self, matter_id: str) -> None:
        """事closed/被 merge 掉时移出（闭合档案退出活跃窗口，ADR-0031 §3.5）。"""
        self._active.pop(matter_id, None)

    def snapshot(self) -> list[str]:
        """最近活跃的 ≤K 份，按活跃时间降序。这份列表**必须**原样进台账
        window_snapshot——没有它，事后无法审计"当时窗口里有没有正确答案"。"""
        return [m for m, _ in sorted(self._active.items(),
                                     key=lambda kv: -kv[1])[:self._k]]

    def __contains__(self, matter_id: str) -> bool:
        return matter_id in self.snapshot()


def decide_turn(
    *,
    anchor_matter_id: str,
    anchor_alive: bool,
    anchor_reference_text: str,
    window: list[str],
    session_id_source: str,
    intent_text: str,
    unit_hit_matter: str = "",
    auxiliary: bool = False,
    new_subject_max: int = DEFAULT_NEW_SUBJECT_MAX,
) -> TurnDecision:
    """决策表 v2 §4 的纯函数形态。从上往下短路，每轮走且只走一行。

    `unit_hit_matter`：窗口内 **S2 unit_key 唯一命中**的卡（调用方算好传入；
    多卡命中或零命中都传 ""——唯一性是零 LLM 直通的前提，模糊即送裁决）。
    R5 的裁决与降级不在本函数（LLM 属 proxy 侧）：verdict=needs_judgment 时
    调用方先查 taskstate 台账（重放台账优先），无记录才现场判；
    裁决 uncertain / 超时 / 失败 ⇒ **降级 = 新开**（R7 保守侧：不确定时选
    可逆的错误方向——碎片可事后折叠，误挂没有对称的回退机制），且判决
    无论结果一律入台账（否定判决也记，否则旋钮一标定重放就漂首轮集合）。
    """
    snap = list(window)
    base = dict(anchor_matter_id=anchor_matter_id, anchor_alive=anchor_alive,
                window_snapshot=snap)

    # 表 §2：aux 四不。不判、不动状态；是否蒸馏是另一层的事（两层别混）。
    if auxiliary:
        return TurnDecision(branch=BRANCH_AUX, verdict="aux_skip", **base)

    anchor_ok = bool(anchor_matter_id) and anchor_alive

    # S2 unit_key 唯一命中：内容派生的同一意图，唯一许零 LLM 压过默认的证据。
    if unit_hit_matter and unit_hit_matter in snap:
        if anchor_ok and unit_hit_matter != anchor_matter_id:
            branch = BRANCH_R1_UNIT_OVERRIDE
        elif anchor_ok:
            branch = BRANCH_R2_ANCHOR      # 命中的就是锚 = 普通默认延续
        else:
            branch = BRANCH_R4_UNIT        # 跨 session 同意图回切
        return TurnDecision(branch=branch, verdict="continue_to",
                            matter_id=unit_hit_matter,
                            signals={"unit_hit": unit_hit_matter}, **base)

    if anchor_ok:
        cont, n_new = is_task_continuation(
            session_id_source=session_id_source,
            current_text=intent_text,
            reference_text=anchor_reference_text,
            new_subject_max=new_subject_max,
        )
        if cont:
            # R2 默认路径：默认延续，要"证明"是新的事才切换（举证责任反转）。
            return TurnDecision(branch=BRANCH_R2_ANCHOR, verdict="continue_to",
                                matter_id=anchor_matter_id,
                                signals={"new_subjects": n_new}, **base)
        # R3：疑似换事。unit_key 没命中（上面已查），S3/S5 只提名不判 → R5。
        return TurnDecision(branch=BRANCH_R5_JUDGE, verdict="needs_judgment",
                            signals={"new_subjects": n_new, "from": "R3"}, **base)

    # 锚缺席（session 首轮 / 时间窗切断），unit_key 也没命中 → R5。
    return TurnDecision(branch=BRANCH_R5_JUDGE, verdict="needs_judgment",
                        signals={"from": "R4_miss"}, **base)


def unique_unit_hit(unit_key: str, window_unit_index: dict[str, set[str]]) -> str:
    """窗口内 unit_key → 卡 的唯一命中判定（调用方建 `unit_key -> {matter_id}` 倒排）。

    唯一命中返回 matter_id；零命中或多卡命中返回 ""（模糊即送裁决，不猜）。
    """
    if not unit_key:
        return ""
    hits = window_unit_index.get(unit_key) or set()
    return next(iter(hits)) if len(hits) == 1 else ""
