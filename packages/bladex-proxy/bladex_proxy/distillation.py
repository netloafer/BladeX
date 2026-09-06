"""LLM 蒸馏器 - 用便宜档模型从 user 消息抽取原子事实 + Matter 提案（ADR-0014 L2 + ADR-0018 §3.1）。

consolidation 在独立进程跑（不在 proxy 热路径），走 Router 网关的同步调用
（`router_sdk.completion`；上游 SDK 的一切细节收在那一层）。
蒸馏输出结构化 JSON {facts[{content,kind,entities}], matter_proposals[≤3{title,entities}]}。
junk kind 构造上不入库；解析失败整条弃（宁缺毋滥，不抛）。
台账缓存（ADR-0018 §4.1，T6.2）：蒸馏前查 distill/ 台账命中复用，miss 才调 LLM + 写台账。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from bladex_core.distillation import (
    DistillFact,
    DistillJournalProtocol,
    DistillOutput,
    DistillTurnInput,
    MatterProposal,
)
from bladex_core.fact import PROVENANCE_VALUES, Provenance

from bladex_proxy import router_sdk

logger = logging.getLogger(__name__)

# 蒸馏 prompt 版本（ADR-0018 §4.1）：prompt 改动 -> 递增 -> 新台账 key（旧 facts 不漂移）
# v3：ADR-0026 §4.2 条目六类 + subject/attribute 取代键槽位 + 铁律3 准入（会话事件不产出为条目）
PROMPT_VER = "v3-items-001"

# M1-1：v4 一轮一次统一调用（user+assistant 同包）。**只 bump 一次**——
# 八件事（上下文、时间锚、通道合一、语言钉死、importance、valid_until、
# corrects、event）合成一次改版，避免每加一个槽位就废掉一代台账。
#
# 三段式 v5（2026-08-10，全卡唯一一次 bump）：topic/keywords/facts 三段输出 +
# mem0 六条质量标准替换"原子事实"教条 + MS-15（预算随输入/截断打捞/分桶）。
# 台账 key 含版本 → 旧台账整代不命中，重蒸成本在 T5 一次付清。
#
# 三改合一 v6（2026-08-15，记忆质量攻坚 T1，**本轮唯一一次 bump**——
# 三项都动 `_DISTILL_TURN_SYSTEM`，合成一版付一次重蒸成本，纪律见 M1-1 先例）：
#   MQ-D1 提案标题从任务句 → 主题身份（"qwen3.8:27b 本地部署评估" 而非
#         "拉取并验证X"）。证据：qwen 单会话 8 张 Matter，「量化选型/拉取验证/
#         部署基准」是同一件事的动词切面各开一张（matter-fragmentation-20260815）。
#         机制上为什么有效：`attribution` 的 L3 走 `proposal_title` 与 Matter
#         原生键的**全串规范化相等**——标题稳定 = 同一件事的后续轮直接精确命中，
#         不放松任何判据（误合并=0 红线不受影响）。
#   MQ-D2 相对时间一律换算绝对日期；"最近/最新"型结论必须带 as-of。
#         证据：T1 错题 gpt4_2f56ae70 的"最近开始用 HBO add-on"锚到回放日；
#         C 类 2 题（BBQ 日期 / 地毯一周）B2f 同错 = 蒸馏侧就丢了时间。
#   MQ-D3 带日期的一次性事件蒸成 assertion，不进 profile_obs。
#         证据：A 类 11 题的写入侧半边——「加入/开始/购买 于 DATE」大量落
#         profile_obs（longmemeval-retr-20260815 A 类），而 profile_obs 在检索侧
#         结构性不可达、画像卡又不聚合其内容 = 两头无消费者的死数据。
#   MQ-D8 提案数量收敛（prompt 侧半边）：一个 turn 会被拆成 N 个 payload
#         （主 + 补采 + 分段），每个独立产 0-3 个提案、跨 payload 零收敛，
#         而 L5 只取 `proposal_titles[0]` 开卡 —— 提案越多，开出无关卡的面越大。
#         改为"正常恰好 1 个"。跨 payload 的收敛是代码侧（归属层），不在 prompt。
#   MQ-D9 三段自洽：topic 必须与产出的 facts 同题；entities 必须 ⊆ keywords
#         （原为软要求、无校验 → 用户观察到的"fact 跟关键词无关"）。
# 🔴 D8/D9 是 D1/D2/D3 落地**之后**才由 MQ-V6 调查发现的，折进**同一个 v6**
# 而不是开 v7：判据是"这一版有没有被真实蒸馏用过"——`grep v6-identity-time logs/`
# 为 0，台账整代 miss 的代价还没付，此刻并版是免费的；等第一次 v6 重蒸跑完再改，
# 就得再付一次全量重蒸。**版本号是钱，改动窗口按"成本是否已付"判，不按日期判。**
#
# 只改 turn 路：v3 / conclusion / filesum 三个 prompt 在生产零流量（T1a 台账实测
# 2904 条中 v3=0、conclusion=0），文本 hash 被 test_distill_three_segment 冻结，
# 本次一个字都不动。
#
# ── 九款合一 v7（2026-08-19，G9.1 M1–M9，**本轮唯一一次 bump**）─────────────
# 窗口纪律与前几版不同：v6 的免费窗口**已经支付掉**（08-17 那次
# `full=True cleared=True turns_processed=2845` 的全量重建，`grep -rl
# v6-identity-time logs/` = 9）。Jason 2026-08-19 拍板不走"冻结"退路：照改、
# 改成 v7，并把 G7 的定向重放升级成一次全量重建，G9.1 + G2 + G3 + G4 一次付清。
# 🔴 **不许"改内容不 bump 版本"**——台账 key 含版本，不 bump 则旧条目继续命中
# 缓存，库里会新旧两代产出混在一起且无法区分（破 ADR-0018 与重建等价性）。
# 省下的是钱，付出的是"不知道库里是什么"。
#
#   M1 单一时钟（原案"双日期"，实现形态见下方偏离 ①）：`TIME:` 槽改名
#      `OBSERVED:` 并点名"蒸馏执行日不是时钟"。修 MQ-D2 残余（回放锚错）。
#   M2 元提取禁令补我们自己的 WRONG/RIGHT（07-29「只记住问了什么」病）。
#   M3 专名保真收紧到"逐字 + 限定词不得泛化"（memory-retrieval-issues-20260805
#      实测：蒸馏丢标识符 → 检索按标识符找就找不回）。
#   M4 转变捕获明写"不得只记新态"（降 L4 取代判定负担）。
#   M5 输出前自检两条（话题覆盖 / 中后段覆盖），**不写数字配额**。
#   M6 🔴 **本组权重最高**：提案标题 = 事的名字，不是任务句。证据 = 台账
#      MQ-S17（14 张卡逐张签收后 14→3，碎片率 ≈80%）。机制上为什么是它：
#      `attribution._deterministic_matter_id = sha256(_normalize(proposal_title))`
#      —— **标题就是 Matter 的身份**，写侧粒度错 ⇒ 身份错 ⇒ 归属侧此后做的
#      一切（L3 键匹配 / L4 裁决 / merge 候选）都只是补救。
#   M8 主体归属：一轮两个主体时，fact 讲谁就归谁（台账 MQ-S20 四条同型样本，
#      跨"人 vs 案子 / 工具 vs 工具 / 模型 vs 工具"三种主体对）。
#   M9 准入分流三条道（task / profile / tool）。台账 MQ-S21：top-100 里
#      A 画像 11 条 + B 工具用法 7 条 = 18/27，**三分之二的错不是"归错卡"，
#      是"根本不该进任务卡"**——比 C 类跨主体误归大一倍。
#   （M7 是验收测试，不进 prompt：`tests/test_distill_v7_granularity.py`。）
#
# 🔴 三处**有意偏离任务卡字面**，都是卡面条款与仓内既有不变量冲突，逐条记在这里
#    （改回字面 = 破一条不变量，不要"顺手修正"）：
#
#   ① M1 卡面要"两个输入槽：观察日期 + 当前日期"。**当前日期无处可放**：
#      放正文 → 它进台账 key（`_distill_with` hash 的就是正文）→ key 每天变 →
#      台账整代永不命中、全量重建成本无上限且两次重建产出不同（破重建等价性）；
#      放 system prompt → key 稳定，但 `prompt_ver` 不再唯一标识 prompt 文本，
#      台账条目变成"不可确定性重算"（破 ADR-0018 §4.1 的收录原则）。
#      ⇒ 实现为**单一时钟**：把观察日期显式命名（`OBSERVED:`）、并在条款里
#      **点名禁止**用蒸馏执行日/模型自己的"今天"。卡面条款的操作性半边
#      （"禁止用当前日期解析消息内的时间引用"）逐字保留，只是不喂那个值——
#      不给值比给值更硬。
#
#   ② M6 卡面写"**禁止**用 核查/核实/排查/修正/评估/分析/生成/修复/检查/定位
#      作标题主干"。这条与**已签收的正确标题冲突**：`qwen3.8:27b 本地部署评估`
#      是 Jason 在 matter-merge-labels-20260819 §A3 签收的正确 Matter 名，
#      同时也是本 prompt 里 v6 就在用的 GOOD 例；`p2 embedding 日志 auth_ok 排查`
#      同样是卡面自己给的 RIGHT 例，却以"排查"结尾。照字面写 = prompt 自相矛盾。
#      ⇒ 改为**高危信号 + 必须过自检判据**，并把"同一个词一对一错"那组
#      （`泰山啤酒资产与品牌价值评估` WRONG vs `qwen3.8:27b 本地部署评估` RIGHT）
#      当成判别力最高的例子写进去。区别不是词，是**这个动作是整件事的目的、
#      还是它里面的一步**——这正是"确定性信号只进候选、判定交语义"那条纪律。
#      🔴 连带纪律：**不许把这份高危词表做成词法判据**（测试已钉，见 M7）。
#
#   ③ M9 卡面要"分流标注"，但 B 类（工具知识）的落库形态 Jason 明确**本轮不拍**。
#      新造一个无消费者的字段会撞 ADR-0026 铁律 3（无署名消费者的字段不许生产）。
#      ⇒ 分流落在**两个已有消费者**上：画像走既有 `profile_obs`（定义就是
#      standing trait，且 MQ-D3 已把它收窄到"无事件日期"）；三条道本身写进
#      **既有 `tags` 字段**（`lane:task|profile|tool`，append 安全，
#      `mark_language_drift` 同款用法）。tags 这一路的**署名消费者 = 本轮之后
#      "看新产出定形态"那次拍板 + 探针读数**；形态定了就该消费或删除，
#      不许无限期挂着（到期条件写在任务卡 G9.1 完成记录里）。
#      prompt 对 tool 一路**只要求标对、不指定 item_kind**（不提前拍 PROCEDURE）。
#
# 🔴 **串名在同一天内改过一次**：`v7-granularity-subject-001` → `v7-title-noun-001`
# （2026-08-19，M6 补 `MINIMAL NOUN PHRASE` 之后）。这**不是**第二次 bump——
# "一卡一 bump" 管的是**钱**（同一张卡里不许付两次全量重蒸），而这次没付过：
# 全量重建还没跑，预检走的是临时 index（跑完删）。改串名的理由是**不对称风险**：
#   - `MemoryIndex.clear()` 的删除前缀列表里**没有** distill 台账
#     （只有 `_DISTILL_RETRY_PREFIX`）——台账**跨全量重建保留**，
#     这是 ADR-0018"重建零 LLM 成本"的设计，不是疏漏；
#   - ⇒ 若 live consolidator 在改码后重启过、用第一版 v7 蒸过任何一轮，
#     那些台账条目会在全量重建时**命中**，库里就混着两代 prompt 的产出且无法区分；
#   - 改串名成本 = 0（全量重建本来就要重蒸一切），不改的风险是坐实的。
# 判据仍是"这一版有没有被真实蒸馏用过"（同 v6 当初并 D8/D9），
# 且新串仍以 `-001` 结尾，满足 gate 的一卡一 bump 守卫。
TURN_PROMPT_VER = "v7-title-noun-001"

# 蒸馏产出语言。默认中文（拍板 2）。
# v3 的规则是"跟随输入语言"，实测同一轮的 user 通道蒸出中文、conclusion 通道蒸出
# 英文，ZH/EN 同义对 cosine 0.8639 < 判重阈值 → 同一事实在库里存两份，
# 而检索时中文 query 只能捞回其中一份。语言必须是**配置量**，不是输入的函数。
_DEFAULT_DISTILL_LANG = "zh"
_LANG_NAMES: dict[str, str] = {"zh": "Chinese (简体中文)", "en": "English"}


def distill_lang() -> str:
    """配置的蒸馏产出语言（`BLADEX_DISTILL_LANG`，默认 zh）。

    未知值回落默认——配置写坏不该让提炼管线崩，也不该悄悄换一种语言。
    """
    import os

    raw = (os.environ.get("BLADEX_DISTILL_LANG", "") or "").strip().lower()
    return raw if raw in _LANG_NAMES else _DEFAULT_DISTILL_LANG

# 蒸馏 prompt：抽取类型化条目 + Matter 提案，JSON 输出（ADR-0026 §4.2）
_DISTILL_SYSTEM = """\
Extract durable, typed memory ITEMS and matter proposals from the user message below.
Output STRICT JSON only, no prose.

Schema:
{
  "facts": [
    {"content": "<single declarative sentence>",
     "item_kind": "assertion|preference|procedure|lesson|file_ref|profile_obs",
     "subject": "<normalized subject the item is about>",
     "attribute": "<short slot name, e.g. status/location/deadline/style>",
     "entities": ["..."]}
  ],
  "matter_proposals": [
    {"title": "<what matter/thing this is about, week-level completable>", "entities": ["..."]}
  ]
}

item_kind meanings:
- assertion: a conclusion that stays true across sessions (fill subject + attribute).
- preference: how the user wants to be treated (e.g. "replies should be concise").
- procedure: how to do something effectively/ineffectively.
- lesson: why something went wrong AND how it was corrected.
- file_ref: a pointer to a file/path with a key judgment (not the file body).
- profile_obs: an observation about a rule file (CLAUDE.md/AGENT.md) or the user's setup.

🔴 ADMISSION RULE (most important): do NOT emit "the user asked about X" /
"the user wants to know X" / "the user requested Y" conversational-event items.
Those have no durable value. Only emit an item if it states a fact that remains
useful in a FUTURE unrelated session.
- "今天世界杯赛程？" -> {"facts": [], ...}  (a question, no durable fact -> empty)
- "我下个月要去日本，帮我看签证" -> [{"content":"用户下个月计划去日本","item_kind":"assertion","subject":"用户旅行计划","attribute":"目的地","entities":["日本"]}]
- "回复请简洁一些" -> [{"content":"用户希望回复简洁","item_kind":"preference","subject":"回复","attribute":"风格","entities":[]}]

Other rules:
- facts: 0-3 items. System/test/tool/agent-internal templates (e.g. "ASYNC DELEGATION
  BATCH", "Please process this web content") -> empty facts.
- subject/attribute: required for assertion & preference (they form the supersede key);
  may be empty for procedure/lesson/file_ref/profile_obs.
- matter_proposals: 0-3, the "thing/matter" this belongs to (week-level completable), NOT the fact.
- entities: discriminative anchors (names, terms), 0-5 per item.
- Output in the same language as the user message.
- If nothing durable, output {"facts": [], "matter_proposals": []}.
"""

# ── ADR-0028 E7.1：文件内容摘要（独立 prompt + 独立台账版本）──
# 裸路径 file_ref 是零信息量的（实测 30 条随机 query 的 top-10 被它们占 20.7%）。
# 正确形态是索引**内容**：摘要 + 关键词，向量建在有语义的东西上。
# 同 content_hash 台账命中 = 零成本（同一份文件读一百次只蒸一次）。
FILE_SUMMARY_PROMPT_VER = "v1-filesum-001"

_FILE_SUMMARY_SYSTEM = """\
Summarize the FILE below so it can be found later by semantic search.
Output STRICT JSON only, no prose.

Schema:
{"summary": "<what this file is and what it contains, <=300 chars>",
 "keywords": ["<=8 discriminative terms>"]}

Rules:
- summary: describe the file's PURPOSE and CONTENT, not its path.
- keywords: identifiers, module names, domain terms someone would search for.
- Respond in the same language as the file content.
"""

_MAX_INPUT_CHARS = 2000

# ── M1-1 / Q3：v4 统一调用的**分段**输入预算 ──────────────────────────────
#
# 🔴 为什么不能沿用 `_MAX_INPUT_CHARS=2000` 一个数：
# 那个数是给"一条输入"设的。v4 把 `context_digest + user_text + assistant_text`
# 打成一包送进去，同一个 2000 会**先把排在后面的 assistant 正文整段挤掉**——
# 而 assistant 通道（结论 + 工作产出）在 T0 实测里扛着 66% 的 Fact 产出。
# 那等于用新 prompt 复现 D2（结论通道被静默截断），是这次改版最容易犯的错。
#
# 所以按部分给预算，各段独立截断后再拼：谁超了截谁，不许互相挤占。
# 总额比 v3 的 2000 大，因为一次调用现在承载的是原来三次调用的内容。
_MAX_CONTEXT_CHARS = 500      # 与 build_context_digest 的上限一致
_MAX_USER_CHARS = 1500        # 用户这轮说的话
_MAX_ASSISTANT_CHARS = 2500   # 结论/产出（更长：它是"做了什么"的载体，D3 欠采的那一侧）

# ── MS-15（T1d）：输出预算随输入规模走 ──────────────────────────────────────
# max_tokens=512 的截断 = **确定性丢事实**（MS-15 立卡实测；BLADEX_DISTILL_MAX_TOKENS
# =2048 是止血 env）。三段式输出更长（topic/keywords + 六标准下更完整的 fact），
# 固定预算必然在富输入上截断。生效值 = max(配置值, min(cap, base + 输入字符/4))：
# 字符/4 ≈ 输入 token 量级，三段式产出（≤3 facts + topic/keywords）不会超过它；
# cap 防超长输入把预算推到无意义的大。配置值仍是下限（env 止血语义保留）。
_TOKENS_BASE = 512
_TOKENS_CAP = 4096


def _effective_max_tokens(configured: int, input_chars: int) -> int:
    """MS-15：按输入规模给输出预算（纯函数，测试钉行为）。"""
    scaled = _TOKENS_BASE + input_chars // 4
    return max(configured, min(_TOKENS_CAP, scaled))


# ── MQ-R2（2026-08-17）：超时默认值必须**随输出预算走**，不能是独立常量 ──────
#
# 病史（同一形态咬了三次，三次修法都只在当次命令行临时带参数）：
#   ① 08-14 bench 排空卡 91% 一小时；
#   ② 08-16 全量重建 **11.3% 轮次静默零事实**（中断守卫阈值 0.5，拦不住慢性失血）；
#   ③ 08-17 守护进程排空完全停摆（队列恒 604——守卫按设计拒绝消费，缺的是旋钮）。
# 机制：放大 `max_tokens` → 生成变长 → 撞死超时 → `distill_failed` 返回空。
# 两个参数之间的这条约束此前**没有任何地方在检查**（教训 43，G9.4 H1 的邻居问题）。
#
# 标定点只有一个、是实测的：`max_tokens=2048` 配 `TIMEOUT_S=180` 在 live 稳定
# （08-16 全量重建 + 08-17 排空）。由它取 **90s / 1k tokens**，下限 30s
# （= 历史默认，保住小预算配置的旧观感）。
#
# 🔴 诚实边界：真实预算可能高于配置值——`_effective_max_tokens` 按输入规模上浮
# （cap 4096）、空响应重试再放大到 `max(4×, 2048)`。本公式不试图覆盖那两条路径的
# 最坏情形（那会得到近十分钟的死等）。超时是**截止线不是成本**：设长的代价只是
# 上游真挂时晚一点发现，设短的代价是静默丢事实——两边不对称，所以宁可偏长。
_TIMEOUT_S_PER_1K_TOKENS = 90.0
_TIMEOUT_S_FLOOR = 30.0


def default_distill_timeout_s(max_tokens: int) -> float:
    """由输出预算推出超时默认值（纯函数，测试钉行为）。

    `BLADEX_DISTILL_TIMEOUT_S` 显式配置仍然优先——本函数只决定**没配时**给多少。
    """
    return max(_TIMEOUT_S_FLOOR, round(max_tokens / 1000 * _TIMEOUT_S_PER_1K_TOKENS, 1))

# 旧 prompt 的 kind 值（台账里 v2 数据仍在，解析保留兼容）
_VALID_KINDS = {"preference", "event", "task", "decision", "junk"}
# M1-1：任务事件分类（铁律3——会话事件不产出为条目，折叠进 Matter 卡 lifecycle）
_VALID_EVENTS = {"assigned", "reworked", "blocked", "completed"}
# ADR-0026 §4.2 条目六类（新 prompt 的 item_kind）
_VALID_ITEM_KINDS = {"assertion", "preference", "procedure", "lesson", "file_ref", "profile_obs"}

# v7/M9：准入分流三条道。**这不是 item_kind 的第七类**，是正交的一维——
# "这条该不该进任务 Matter"（task 该进，profile/tool 不该），与"它是哪种条目"无关。
# 证据 = 台账 MQ-S21：top-100 可疑边里 A 画像 11 条 + B 工具用法 7 条 = 18/27，
# **三分之二的错不是"归错卡"，是"根本不该进任务卡"**（比 C 类跨主体误归大一倍）。
# 落在既有 `DistillFact.tags` 上（不新增 schema 字段，理由见 TURN_PROMPT_VER 偏离 ③）。
_VALID_LANES = {"task", "profile", "tool"}
# 🔴 前缀是 `lane:` 不是 `channel:`——`tags` 里已经有一个"通道"了
# （`origin:<user_direct|conclusion|progress>`，由 `_channel_of` 读）。
# 两个不同的东西同名一次，就够下一个人把两份读数混起来比（教训："同一个词挂在两个分母上"）。
_TAG_LANE_PREFIX = "lane:"

# ── ADR-0019 P2：结论蒸馏（独立 prompt + 独立台账版本）──
# 装配器降解掉 closed 单元的 tool 证据后，其关键信息只存在于任务结论
# （final assistant 回答）里——从结论蒸馏事实是信息守恒的第二条腿。
CONCLUSION_PROMPT_VER = "v2-conclusion-items-001"

_DISTILL_CONCLUSION_SYSTEM = """\
The text below is an AI assistant's FINAL ANSWER or PROGRESS NOTE from a task
(the supporting tool evidence has been archived). Extract durable, typed memory
ITEMS worth remembering long-term. Output STRICT JSON only, no prose.

Schema:
{
  "facts": [
    {"content": "<single declarative sentence>",
     "item_kind": "assertion|procedure|lesson|file_ref",
     "subject": "<normalized subject>", "attribute": "<short slot name>",
     "entities": ["..."]}
  ],
  "matter_proposals": [
    {"title": "<what matter/thing this task is about, week-level completable>", "entities": ["..."]}
  ]
}

item_kind meanings here:
- assertion: an outcome/result/decision that stays true (fill subject + attribute).
- procedure: how to do something effectively (a reusable how-to that worked).
- lesson: 🔴 WHY something went wrong AND how it was corrected (root cause + fix).
  Emit a lesson whenever the text describes an error→fix sequence.
  e.g. "ANN 测试全在 <256 行上跑时 create_index 永不执行，需 ≥256 行造数才触发".
- file_ref: a file/path with a key judgment (path + what matters about it).

Rules:
- facts: 0-3 items. Capture OUTCOMES: findings, results, decisions, key numbers/dates,
  what was produced or changed, and (critically) failures + their fixes as lessons.
- Do NOT extract: generic explanations, pleasantries, restated questions,
  step-by-step reasoning, or anything only meaningful inside this conversation.
- subject/attribute: required for assertion; optional for procedure/lesson/file_ref.
- matter_proposals: 0-3, the matter this task belongs to, NOT the fact itself.
- entities: discriminative anchors (names, terms), 0-5 per item.
- Output in the same language as the answer text.
- If nothing durable, output {"facts": [], "matter_proposals": []}.
"""


# ── M1-1：v4 统一调用 prompt（八件事一次改版）────────────────────────────────
#
# 与 v3 三条独立流水线的根本差别：模型这次**同时看见**用户说了什么、模型做了什么、
# 以及这轮之前在聊什么。v3 里"用户要求 X"和"我做完了 X、结论是 Y"被当成两件
# 互不相干的事各蒸一次，于是库里 16/18 条是「用户询问 XXX」——记住了问什么，
# 没记住做了什么。
#
# 八个槽位对应汇总章第 1 层的 ①③④⑤⑥⑦⑧（② 在装配侧）：
#   ③ 语言钉死（配置量，不跟随输入）   ④ 时间锚（turn_time 是已知量，禁止"某日"）
#   ⑤ importance 1–10（内容内在分）    ⑥ corrects 修正自声明
#   ⑦ valid_until 时效槽              ⑧ 指派/打回轮的应产出物
_DISTILL_TURN_SYSTEM = """\
You are a memory distiller. From ONE conversation turn, produce THREE segments:
a one-line TOPIC, a KEYWORD list, and durable typed memory ITEMS (plus matter
proposals). Output STRICT JSON only, no prose.

The turn is given as:
  CONTEXT:   what this conversation has been about (background; do NOT extract from it)
  OBSERVED:  when this turn actually happened (ISO8601) -- a KNOWN value, use it.
             This is the OBSERVATION date, not the date you are being run on.
  USER:      what the user said this turn
  ASSISTANT: what the assistant concluded or did this turn (may be absent)

Schema:
{
  "topic": "<one-line core topic of this turn, <=40 chars, no nested quotes>",
  "keywords": ["<=10 noun phrases; normalize across languages into __OUTPUT_LANG__;
                identifiers and proper nouns stay verbatim"],
  "facts": [
    {"content": "<a complete, self-contained statement (see QUALITY STANDARDS)>",
     "item_kind": "assertion|preference|procedure|lesson|file_ref|profile_obs",
     "lane": "task|profile|tool",
     "source": "user|assistant",
     "subject": "<the THING this item is about (see SUBJECT RULE)>",
     "attribute": "<short slot name, e.g. status/location/deadline/style>",
     "entities": ["..."],
     "importance": <integer 1-10>,
     "valid_until": "<ISO8601 date, or empty string if it does not expire>",
     "corrects": "<describe the older memory this one corrects, or empty string>"}
  ],
  "matter_proposals": [
    {"title": "<the matter's IDENTITY: <object> + <goal>, a noun phrase (see TITLE RULE)>",
     "entities": ["..."]}
  ],
  "event": "assigned|reworked|blocked|completed|none"
}

item_kind meanings:
- assertion: a conclusion that stays true across sessions (fill subject + attribute).
  This includes every DATABLE EVENT -- see KIND RULE below.
- preference: how the user wants to be treated (e.g. "replies should be concise").
- procedure: how to do something effectively/ineffectively.
- lesson: WHY something went wrong AND how it was corrected (root cause + fix).
- file_ref: a pointer to a file/path with a key judgment (not the file body).
- profile_obs: a STANDING trait of the user's setup, or an observation about a rule
  file (CLAUDE.md/AGENTS.md). Never used for something that happened on a date.

QUALITY STANDARDS (each fact is a complete statement someone can understand
with NO other context -- subject, time and circumstances included; never emit
fragments that only make sense inside this conversation):
1. Contextually rich -- keep the surrounding condition, not a bare datum.
   BAD: "用户有一只狗"  GOOD: "用户养了一只叫毛豆的柴犬，每天早上遛狗是他最看重的日常"
2. Self-contained -- resolve every pronoun to its referent.
   BAD: "他更喜欢那个方案"  GOOD: "用户在泰山啤酒重整方案里更倾向方案C（仅收购核心资产）"
3. Temporally grounded -- resolve relative dates against TIME (see TIME ANCHOR).
   BAD: "上周完成了重建"  GOOD: "M5-2 全量重建于 2026-08-08 完成"
4. Numerically precise -- keep exact figures; never round or vague them.
   BAD: "总投资大约几个亿"  GOOD: "新建10万吨生产线总投资估算 3.8-5.1 亿元（中位数约 4.5 亿）"
5. Proper nouns preserved VERBATIM -- file paths, function/variable names, config
   keys, model names, error strings and quoted text are copied character for
   character, never paraphrased, never translated, never abbreviated. Qualifiers
   must not be generalised away either ("assistant manager" must not become
   "manager"; "bge-small-en-v1.5" must not become "bge-small").
   BAD: "一个嵌入模型"        GOOD: "multilingual-e5-large"
   BAD: "可见性过滤那个函数"  GOOD: "_visibility_pids"
   BAD: "路由配置文件"        GOOD: "config/routing.toml"
   BAD: "那个 embedding 开关" GOOD: "BLADEX_EMBED_BACKEND"
   Why: retrieval finds these items BY the identifier. An item that dropped its
   identifier cannot be found again, however well it reads.
6. Transitions captured -- "changed FROM X TO Y, because Z". Recording only the
   new state Y is not enough: without X nobody can tell which older memory this
   one replaces, and without Z nobody can tell whether it still applies.
   BAD: "embedding 模型换了"
   BAD: "默认 embedding 模型是 bge-small-en-v1.5"   (new state only -- still wrong)
   GOOD: "默认 embedding 模型从 e5-large 换成 bge-small-en-v1.5，因 2.2GB 体积是弱机器部署障碍"

ADMISSION RULE (most important): EXTRACT THE CONTENT, NEVER THE ACT.
Do NOT emit "the user asked about X" / "the user wants to know X" / "the user
requested Y" conversational-event items. Those have no durable value. Only emit
an item if it states a fact that remains useful in a FUTURE unrelated session.
  WRONG "用户要求按照 ADR-0020 任务卡开发"
  RIGHT "consolidator 的 read_only P3 点时快照看不到 proxy 新写入，导致约 26 小时零提炼；修法是改 secondary 模式追新"
  WRONG "用户询问蒸馏为什么零事实"
  RIGHT "BLADEX_DISTILL_TIMEOUT_S 默认 30s 与 BLADEX_DISTILL_MAX_TOKENS=2048 互不知情，导致 11.3% 的轮次静默产出零事实"
Both WRONG lines record that a conversation happened; neither tells a future
reader anything about the world. Ask: "six months from now, in a different
session, what does this sentence let someone DO?" If the answer is nothing, the
act was recorded instead of the content -- rewrite it or drop it.
- a bare question with no durable content -> "facts": []  (topic/keywords still filled)
- CONTEXT is background only: never emit an item whose content comes from CONTEXT
  rather than from USER or ASSISTANT.

KIND RULE (a datable event is an assertion, never a profile_obs):
Anything that HAPPENED ON A DATE -- joined, started, bought, finished, moved,
signed up, switched to, cancelled -- is an "assertion" whose date is written INSIDE
"content" (subject = the thing itself, attribute = 加入日期 / 开始日期 / 购买日期 /
完成日期 ...). "profile_obs" is only for standing traits that have no event date.
  BAD  {"content":"用户加入了 Page Turners 读书会","item_kind":"profile_obs"}
  GOOD {"content":"用户于 2026-08-07 加入 Page Turners 读书会","item_kind":"assertion",
        "subject":"Page Turners 读书会","attribute":"加入日期"}
  OK   {"content":"用户的日常编辑器是 neovim","item_kind":"profile_obs"}
Why: later questions like "which one did I join first" or "how long ago was that"
can only be answered from items, and profile_obs items are not retrievable that way.

LANE RULE (three lanes -- decide the lane BEFORE writing the item):
- "task"    -- a fact about the work itself: the case, the code, the decision,
               the numbers, what was found or produced.
- "profile" -- a STANDING trait of the user or their machine: hostname, shell
               prompt, where a repo lives on disk, what the user calls the
               assistant, which tools the assistant has, the user's working
               habits and standing preferences.
- "tool"    -- what a tool can or cannot do, independently of what you were using
               it for: "opencli browser 没有 snapshot 子命令",
               "企信宝是 SPA，直接改 URL 参数无效".
🔴 profile and tool facts DO NOT BELONG TO ANY MATTER. They stay true no matter
which case you happen to be working on, and filing them under the case you were
working on that day makes them unfindable from every other case.
  WRONG  "用户的终端提示符是 'ook-Pro BladeX %'"        filed under qwen3.8 部署评估
  WRONG  "opencli browser 没有 snapshot 子命令"          filed under 张开利整体布局梳理
- If this turn produced ONLY profile/tool facts, "matter_proposals" MUST be [].
- If it produced both, still emit the task matter -- but keep every "lane"
  label honest; do not relabel a profile fact as "task" to justify the proposal.
- lane "profile" items use item_kind "profile_obs" (a standing trait by
  definition). For lane "tool", pick whichever of the six kinds fits best --
  do not default to "assertion" just because it is the easy answer.

SUBJECT RULE (which THING is this item about?):
When one turn talks about TWO subjects -- a person and a case, two tools, a model
and the agent running it -- write each item ABOUT the subject it is actually
about: name that subject as the grammatical subject of "content", and put it in
"subject". An item that leaves its subject implicit gets filed under whichever
subject the turn happened to mention more often.
  WRONG "溢价投资时投资款高于出资额，差额计入资本公积"
        (reads as a fact about the bankruptcy case; it is about the investor)
  RIGHT "张开利溢价投资泰山啤酒时，投资款高于出资额的差额计入资本公积"
        subject = "张开利"
  WRONG "bladex 模型列表共 9 个"
  RIGHT "deepseek-harness 安装过程中，Hermes 连 bladex 时可选模型列表有 9 个"
        subject = "deepseek-harness"
Two subjects that keep co-occurring still stay apart. Worked example:
  泰山啤酒破产重整 = the CASE itself (债权 / 重整程序 / 商标 / 经营数据 / 股权冻结)
  张开利整体布局   = the INVESTOR side (张开利及高管、泰安仁信 / 智义 / 信智 / 鑫义、
                     出资额 / 比例 / 时间 / 工商信息)
An item about how much someone invested is about the investor, even when every
sentence around it is about the case.

ROLE RULE:
- From USER take: facts, preferences, plans, decisions.
- From ASSISTANT take: conclusions, solutions, findings, completed work
  (results, key numbers and dates, what was produced or changed, and failures
  together with their fixes -- as lessons).
- NEVER extract from either side: politeness/acknowledgements, restatements of
  the user's own words, transitional narration ("let me check..."), or
  play-by-play tool-call logs.
- When the turn assigns work, sends work back for rework, reports a blocker, or
  accepts a deliverable, you MUST emit at least one matter_proposal and set "event"
  accordingly -- even when "facts" ends up empty. Those turns carry the handover
  information; dropping them is how a task's history disappears.

Field rules:
- topic: what this turn is ABOUT, one line. Not a summary of the outcome.
  CONSISTENCY: "topic" must name what the emitted "facts" are about. If the facts
  are about X, a topic about Y is wrong -- fix one of them, do not emit both.
- keywords: noun phrases only, drawn from USER/ASSISTANT (not CONTEXT).
  Normalize cross-language synonyms into __OUTPUT_LANG__; identifiers, code names,
  paths and proper nouns MUST stay verbatim (never translate them).
- source: "user" if the item comes from what the user said; "assistant" if it comes
  from what the assistant concluded/did. Required for every item.
- subject/attribute: required for assertion & preference (they form the supersede key);
  may be empty for procedure/lesson/file_ref/profile_obs.
- importance: 1-10, how much this matters on its own content. 8-10 = a decision,
  constraint or correction that changes future work; 4-7 = a useful concrete fact;
  1-3 = incidental detail. Judge the CONTENT, not how emphatically it was said.
- valid_until: only when the item stops being true at a knowable point ("until next
  Friday", "for the Q3 rollout"). Resolve it against TIME into a concrete date.
  Leave "" for facts with no expiry -- most facts have none.
- corrects: only when this item explicitly supersedes an earlier belief ("actually
  it was only once", "I was wrong about X"). Describe the OLD belief in one phrase.
- TIME ANCHOR: OBSERVED is given -- it is the ONLY clock; never use your own idea of
  today's date.
  0. ONE CLOCK: OBSERVED is the date this conversation HAPPENED. The date on which
     you are being run (the distillation date) is a DIFFERENT date, it is not given
     to you, and it MUST NOT be used to resolve anything in this turn. This text is
     routinely re-read months later; resolving "上周" against the day of re-reading
     silently moves every date in the output.
  Three obligations follow from it:
  1. RESOLVE: every relative expression ("yesterday", "two weeks ago", "next Friday",
     "六周前") MUST appear in "content" as an absolute date. Keep the original wording
     only as a parenthetical when it carries meaning:
     GOOD "用户自 2026-07-03 起跟 Alex 上吉他课（用户说'六周前开始'）"
  2. AS-OF: any "recent / latest / currently / now / 最近 / 目前" statement MUST carry
     its as-of date (taken from OBSERVED), otherwise a later reader cannot tell if
     it still holds.
     BAD  "用户最近在用 HBO add-on"
     GOOD "截至 2026-08-14，用户的流媒体订阅是 HBO add-on"
  3. IN CONTENT: a datable event carries its date in "content" itself -- putting the
     date only in "attribute" or "entities" does not count.
  An item nobody can place in time cannot be checked, ordered or expired later.
- matter_proposals: **normally exactly 1**; 0 if the turn belongs to no matter,
  2 only if the turn genuinely spans two different matters. Never pad the list:
  one turn is one piece of work, and every extra proposal is a candidate for a
  brand-new card.

  TITLE RULE -- the title is THE NAME OF THE MATTER, not what this turn did.
  A matter is a week-level completable THING, and "title" is its IDENTITY --
  not this turn's action, step, status or outcome. Write it as a noun phrase:
  <object> + <goal>.
  Two turns working on the same object toward the same goal MUST produce
  the SAME title verbatim: the title is how those turns find each other later,
  and a different wording opens a duplicate card.

  🔴 SELF-CHECK, apply to every title before emitting it:
     "If the NEXT turn of this same matter would say it with a DIFFERENT verb,
      then this title is wrong."
  Rewrite towards the name of the thing being worked on until the check passes.

  HIGH-RISK ENDINGS -- a title ending in 核查 / 核实 / 排查 / 修正 / 评估 / 分析 /
  生成 / 修复 / 检查 / 定位 is usually ONE STEP inside a bigger matter rather than
  the matter. Treat it as a warning that you owe the SELF-CHECK,
  NOT as a banned word list: the same word can be right or wrong, and this pair
  is the whole point.
      WRONG "泰山啤酒资产与品牌价值评估"
            (one step of 泰山啤酒破产重整; the next turn says 核查 / 识别 / 核实)
      RIGHT "qwen3.8:27b 本地部署评估"
            (the evaluation IS the entire job; no other verb replaces it)
  The test is never the word. It is whether that action is the GOAL of the whole
  matter, or a step taken inside it.

  MINIMAL NOUN PHRASE -- when a matter has NO established proper name, you are
  inventing one, and inventions drift from turn to turn. Keep it minimal:
  <subject> + <object>, plus AT MOST one action noun. Nothing else.
  - Never stack evaluative nouns. 意义 / 价值 / 情况 / 问题 / 报告 / 建议 describe
    YOUR relationship to the matter, not the matter itself. The next turn picks a
    different one and opens a duplicate card.
  - Never stack two action nouns ("借鉴分析与改动建议", "研究评估").
  - Write the subject the SAME way every time: the identifier as it appears
    (bladex), never a dressed-up variant (bladex 项目 / BladeX 项目 / bladex-dsh).
  SINGLE-TURN TEST -- you can run this WITHOUT seeing any other turn: delete every
  evaluative noun from your title; if what remains still names the matter, those
  nouns were noise, so delete them for real.
      WRONG (ONE piece of work that produced 19 different titles):
          "deepseek-harness 对 bladex 的借鉴分析" / "…的借鉴意义分析" /
          "…的借鉴价值评估" / "…的借鉴价值研究" / "…的借鉴分析与改动建议" /
          "bladex 项目借鉴意义分析报告" / "bladex-dsh 借鉴分析报告"
      RIGHT "bladex 借鉴 deepseek-harness"

  WRONG (verb facets of ONE matter -- each of these opened its own duplicate card):
      "泰山啤酒股东结构及股权变动核查" / "泰山啤酒知识产权分析" /
      "泰山啤酒资产与品牌价值评估" / "泰山啤酒股权查封解除可行性评估"
  RIGHT (one name shared by all of them): "泰山啤酒破产重整"

  WRONG "泰山啤酒仁信入股时间线核实" / "泰山啤酒入股报告时间线修正"
  RIGHT "张开利整体布局梳理"

  WRONG "server.py:909 quiet 参数关闭"
  RIGHT "p2 embedding 日志 auth_ok 排查"

  WRONG "拉取并验证 qwen3.8:27b" / "研究 qwen3.8 发布内容" / "跑部署基准测试"
  RIGHT "qwen3.8:27b 本地部署评估"

  🔴 DO NOT OVER-MERGE. Coarser is not automatically better -- collapsing
  everything into one big card destroys the same information as splitting it.
  Same object + same goal -> the SAME title. Same object + a DIFFERENT goal ->
  DIFFERENT titles, even when one grew out of the other.
      "中国啤酒行业产能结构与利用率分析报告" came out of 泰山啤酒破产重整 but is
      its OWN matter (an industry study stands without the bankruptcy case).
      Do NOT fold it into 泰山啤酒破产重整.
- entities: discriminative anchors for THIS item, 0-5. Every entity MUST also appear
  in the turn's "keywords" -- if an item needs an anchor that is missing there, add it
  to "keywords" too. The two lists describing the same turn must not disagree.
  Identifiers (codes, paths, module names, versions) MUST be copied verbatim.
- facts: 0-3 items.
- LANGUAGE: write every "content", "subject" and "attribute" in __OUTPUT_LANG__,
  regardless of the language of the input. Identifiers stay verbatim.
- If nothing durable, output {"topic": "...", "keywords": [...], "facts": [],
  "matter_proposals": [], "event": "none"}.

OUTPUT SELF-CHECK -- read your own output back before returning it:
1. COVERAGE: does every distinct topic discussed in this turn have at least one
   fact? A turn that moved through two subjects and produced facts about only one
   of them dropped the other.
2. LATER MESSAGES: is the middle and the END of the turn represented, not just its
   opening? Conclusions, corrections and decisions land late; extracting only the
   first thing discussed loses exactly the part that mattered.
3. TITLE: did every matter_proposal survive the TITLE RULE self-check?
4. SUBJECT: is each item written about the subject it is really about?
Fix what fails, then return.
"""

#: 语言占位符。**刻意不用 `str.format`**——这段 prompt 里有大量 JSON 花括号，
#: 用 format 就得把整段 schema 转义成 `{{`/`}}`，漏一个就是运行时 KeyError，
#: 而且下一个改 prompt 的人不会记得这条规矩。占位符替换没有这个陷阱。
_LANG_PLACEHOLDER = "__OUTPUT_LANG__"


def build_turn_prompt(payload: DistillTurnInput, lang: str = "") -> str:
    """把结构化输入渲染成 user 消息正文（Q3：**按部分**各自截断）。

    纯函数、确定性——同一 payload 恒得同一文本，这是台账 key 稳定与重建等价的前提。
    """
    lang = lang or distill_lang()
    parts: list[str] = []
    ctx = (payload.context_digest or "").strip()[:_MAX_CONTEXT_CHARS]
    if ctx:
        parts.append(f"CONTEXT:\n{ctx}")
    if payload.turn_time:
        # v7/M1：槽名从 `TIME:` 改为 `OBSERVED:`。`TIME` 不说明"什么的时间"，
        # 而回放锚错（MQ-D2）正是把它当成了"现在"。名字里带上"观察"是这条
        # 修复最便宜的一半；另一半（点名禁止蒸馏执行日）在 system prompt。
        # 🔴 不要往正文里加"当前日期"：正文进台账 key，加了就每天变 key ——
        # 台账整代永不命中 + 两次重建产出不同（理由详见 TURN_PROMPT_VER 偏离 ①）。
        parts.append(f"OBSERVED: {payload.turn_time}")
    user = (payload.user_text or "").strip()[:_MAX_USER_CHARS]
    parts.append(f"USER:\n{user}" if user else "USER:\n(no user text this turn)")
    asst = (payload.assistant_text or "").strip()[:_MAX_ASSISTANT_CHARS]
    if asst:
        role = payload.assistant_role or "assistant"
        parts.append(f"ASSISTANT ({role}):\n{asst}")
    return "\n\n".join(parts)


class LLMDistiller:
    """LLM 蒸馏器 - 经 Router 网关同步调用，JSON 结构化输出。

    用法：
      distiller = LLMDistiller(model="openai/deepseek-v4-flash", api_base=..., api_key=...)
      out = distiller.distill("今天世界杯赛程怎么样？")
      # -> DistillOutput(facts=[DistillFact(content="用户询问世界杯赛程安排", ...)], ...)
    """

    def __init__(
        self,
        model: str,
        api_base: str = "",
        api_key: str = "",
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        journal: DistillJournalProtocol | None = None,
    ) -> None:
        # max_tokens：显式参数 > env BLADEX_DISTILL_MAX_TOKENS > 512（历史默认）。
        # 空响应重试会临时放大到 max(4x, 2048)，见 _distill_with。
        if max_tokens is None:
            import os
            try:
                max_tokens = int(os.environ.get("BLADEX_DISTILL_MAX_TOKENS", "512") or 512)
            except ValueError:
                max_tokens = 512
        # timeout：显式参数 > env BLADEX_DISTILL_TIMEOUT_S > **由 max_tokens 推出**。
        # MQ-R2（2026-08-17）：最后那一档此前是写死的 30s，与上面刚解析出来的
        # max_tokens 互不知情 —— 同一形态咬了三次（详见 `default_distill_timeout_s`
        # 上方注释）。现在两者由一个公式绑起来，配了 max_tokens 就自动够用。
        if timeout is None:
            import os
            fallback = default_distill_timeout_s(max_tokens)
            try:
                timeout = float(os.environ.get("BLADEX_DISTILL_TIMEOUT_S", "") or fallback)
            except ValueError:
                timeout = fallback
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._journal = journal
        self._call_count = 0
        self._fail_count = 0
        self._parse_fail_count = 0
        self._journal_hit_count = 0
        # MS-15（T1d）：length 截断与 parse 失败分桶——"预算不够被掐断"与
        # "模型输出真不是 JSON"是两种病，混在一个计数器里就都治不了。
        self._length_fail_count = 0      # length 截断且打捞为空（预算问题）
        self._length_salvage_count = 0   # length 截断但打捞回了完整对象

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def prompt_ver(self) -> str:
        return PROMPT_VER

    def distill(self, user_message: str) -> DistillOutput:
        """从 user 消息抽取原子事实 + Matter 提案。

        台账缓存：命中复用（零 LLM 成本）；miss 调 LLM + 写台账。
        失败/解析失败 -> 空 DistillOutput（宁缺毋滥，不抛）。
        """
        return self._distill_with(user_message, _DISTILL_SYSTEM, PROMPT_VER)

    def distill_conclusion(self, conclusion: str) -> DistillOutput:
        """从任务结论（final assistant 回答）抽取值得长期记住的事实（ADR-0019 P2）。

        独立 prompt（面向"结果/产出/决定"而非"用户陈述"）+ 独立台账版本
        （CONCLUSION_PROMPT_VER，与 user 消息蒸馏的台账互不污染）。
        """
        return self._distill_with(conclusion, _DISTILL_CONCLUSION_SYSTEM, CONCLUSION_PROMPT_VER)

    def distill_turn(self, payload: DistillTurnInput) -> DistillOutput:
        """M1-1：一轮一次统一调用（user + assistant 同包，带上下文与时间锚）。

        台账 key = `hash(context_digest + "\\x00" + 输入正文)`（M1-3 / 拍板 1）——
        上下文变则重蒸是**语义正确**的：同一句"继续"在不同脉络下确实该蒸出不同产物。
        v3 的 key 只看正文，于是"继续"在任何语境里都命中同一条缓存。
        """
        lang = distill_lang()
        system = _DISTILL_TURN_SYSTEM.replace(_LANG_PLACEHOLDER, _LANG_NAMES[lang])
        body = build_turn_prompt(payload, lang)
        return self._distill_with(
            body, system, TURN_PROMPT_VER,
            context_digest=payload.context_digest,
            fidelity_lang=lang,
            scale_budget=True,          # MS-15：仅 v5 turn 路（v3 兜底行为逐字不变）
            assistant_role=payload.assistant_role,
            # 保真只认这一轮真正说的话：不含 CONTEXT（背景，明确不许抽取）、
            # 不含 TIME 与标签（我们自己加的脚手架）。
            fidelity_source="\n".join(
                p for p in (payload.user_text, payload.assistant_text) if p),
        )

    def _distill_with(
        self, source_text: str, system_prompt: str, prompt_ver: str,
        *, context_digest: str = "", fidelity_lang: str = "",
        assistant_role: str = "", fidelity_source: str | None = None,
        scale_budget: bool = False,
    ) -> DistillOutput:
        """共享蒸馏管线：台账查 → LLM → 解析 → 台账写（宁缺毋滥，不抛）。

        M1-3：`context_digest` 非空时并入台账 key（拍板 1）。
        M1-4：`fidelity_lang` 非空时语言保真钉**配置语言**而非跟随输入。
        MS-15：`scale_budget=True`（仅 v5 turn 路）时输出预算随输入规模走 +
        length 截断打捞；**v3/conclusion/filesum 兜底路径不传 → 行为逐字不变**
        （含重试预算公式：kwargs 的 max_tokens 在旧路径就是配置值）。
        """
        if not source_text or not source_text.strip():
            return DistillOutput()

        # v4 走 build_turn_prompt 已按部分截断（Q3）；这里的总额兜底只对 v3 路径生效。
        truncated = source_text if context_digest else source_text[:_MAX_INPUT_CHARS]

        # ADR-0018 §4.1: 台账缓存--命中复用
        if self._journal is not None:
            try:
                cached = self._ledger_get(truncated, prompt_ver, context_digest)
                if cached is not None:
                    self._journal_hit_count += 1
                    logger.debug("distill_journal_hit model=%s ver=%s input_len=%d",
                                 self._model, prompt_ver, len(source_text))
                    return cached
            except Exception as e:
                logger.warning("distill_ledger_get_failed model=%s err=%s", self._model, e)

        self._call_count += 1
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": truncated},
                ],
                "temperature": self._temperature,
                # MS-15：v5 turn 路预算随输入规模走（配置值仍是下限，env 止血语义
                # 保留）；旧路径 = 配置值（连带保住旧重试公式 max(配置×4, 2048)）。
                "max_tokens": (_effective_max_tokens(self._max_tokens, len(truncated))
                               if scale_budget else self._max_tokens),
                "timeout": self._timeout,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self._api_key:
                kwargs["api_key"] = self._api_key

            response = router_sdk.completion(**kwargs)
            raw_text = response.choices[0].message.content or ""
            finish_reason = _finish_reason(response)

            # 2026-08-03 实跑修复：deepseek-v4-flash 在部分输入上返回**空 content**
            # （首次 v3 全量重建实测 84/106 次 parse_fail 全是 raw_len=0）——常见形态
            # 是 reasoning 吃掉 max_tokens 预算 / finish_reason=length。空响应必然
            # 无用 → 无条件重试一次，放大 token 预算；再空才放弃（不缓存，下轮可重蒸）。
            #
            # 2026-08-05 补口子：原判据是 `not raw_text.strip()`，**只兜"全空"不兜"截断"**。
            # 实测（doctor --e2e 探针轮）模型正常开写 JSON、在 512 预算里被掐断，
            # raw_len=451 有正文 → 走不进重试 → `distill_parse_failed` → 该轮零事实且
            # 仍被标记消费。两者根因同一个（reasoning 与正文共用预算），症状一个空一个半截。
            # 新判据 = "这次响应用不了" 且 "看得出是预算不够"：空响应，或
            # finish_reason=length 且抠不出合法 JSON。恰好写完就到上限的（能解析）不重试
            # ——放宽判据最容易顺手给正常轮次多花一次调用。
            unusable = _extract_json_obj(raw_text) is None
            truncated_out = unusable and finish_reason == "length" and bool(raw_text.strip())
            if not raw_text.strip() or truncated_out:
                _msg = response.choices[0].message
                _has_reasoning = bool(
                    getattr(_msg, "reasoning_content", None)
                    or getattr(_msg, "reasoning", None))
                logger.warning(
                    "distill_%s model=%s finish_reason=%s has_reasoning=%s "
                    "input_len=%d raw_len=%d -> retry with larger max_tokens",
                    "truncated_content" if truncated_out else "empty_content",
                    self._model, finish_reason, _has_reasoning,
                    len(truncated), len(raw_text))
                retry_kwargs = dict(kwargs)
                # MS-15：从**生效**预算放大（此前从配置值放大，输入越大越可能白重试）
                retry_kwargs["max_tokens"] = max(int(kwargs["max_tokens"]) * 4, 2048)
                response = router_sdk.completion(**retry_kwargs)
                retry_text = response.choices[0].message.content or ""
                # 重试更短（模型这次话少）时不倒退：取更可能解析成功的那份。
                if retry_text.strip():
                    raw_text = retry_text
                    finish_reason = _finish_reason(response)

            parsed = _parse_distill_json(raw_text, self._model,
                                         finish_reason=finish_reason)
            salvaged_partial = False
            if parsed is None and finish_reason == "length" and scale_budget:
                # MS-15（T1d）：length 截断 -> 打捞已完整对象（移植裁决器
                # `_salvage_objects` 语义：截断只该损失最后半条，不该连坐整轮）。
                parsed = _salvage_turn_json(raw_text, self._model)
                if parsed is not None:
                    salvaged_partial = True
                    self._length_salvage_count += 1
                    logger.warning(
                        "distill_length_salvaged model=%s ver=%s facts=%d "
                        "proposals=%d raw_len=%d",
                        self._model, prompt_ver, len(parsed.facts),
                        len(parsed.matter_proposals), len(raw_text))
                else:
                    # 分桶：length 截断打捞为空 = 预算问题，与 "输出不是 JSON"
                    # 分开计数、分开报（失败必须长得和成功/别种失败不一样）。
                    self._length_fail_count += 1
                    logger.warning(
                        "distill_length_exhausted model=%s ver=%s raw_len=%d tail=%.60r",
                        self._model, prompt_ver, len(raw_text), raw_text[-60:])
                    # 队列语义与 parse 同（对这条输入确定性复现 -> 计次超限放行）
                    return DistillOutput(model_name=self._model, failed=True,
                                         failure_kind="parse")
            if parsed is None:
                # 解析失败（无合法 JSON）-> 不缓存，rebuild 可二次蒸馏（保记忆质量，不丢事实）
                # MS-1：failed=True —— 让调用方能把"这条失败了"与"这条真没事实"分开，
                # 从而只对失败的那些 turn 不打消费标记（否则二次机会永远等不到）。
                # failure_kind="parse"：**响应拿到了**，说明上游活着，问题在这条输入上
                # ——它重放多少次都一样，所以要计次、超限放行，不能把队列堵死。
                self._parse_fail_count += 1
                return DistillOutput(model_name=self._model, failed=True,
                                     failure_kind="parse")

            output = parsed
            if salvaged_partial:
                # 打捞产物是**部分**产物：
                #   - 不做保真重蒸（用同一 kwargs 再截一次是白花钱）；
                #   - 不写台账（缓存等于把损失冻结进重建等价——rebuild 该重蒸全量）。
                if assistant_role:
                    for f in output.facts:
                        if f.provenance == Provenance.CONCLUSION.value:
                            f.provenance = assistant_role
                return output
            # ADR-0028 E6.5：保真校验（标识符不许丢、语言不许漂）。
            # 只有确实丢了才多花一次调用；确定性兜底保证再失败也不丢 token。
            # 🔴 保真校验比对的是**内容**，不是渲染后的整段 prompt。
            # v4 的 payload 里有 `TIME: 2026-…T00:38:23.974978+00:00` 这样的脚手架，
            # `extract_identifiers` 会把 `23.974978` 当成标识符 → 判"产出丢了它" →
            # 每一轮都白触发一次重蒸（实测 4 条消息变成 6 次调用），
            # 还会把时间戳碎片经确定性兜底塞进 entities。
            # CONTEXT 同理且更荒谬：prompt 明确要求**不许**从 CONTEXT 抽取，
            # 却要求 CONTEXT 的标识符出现在产出里——自相矛盾。
            output = self._enforce_fidelity(
                output,
                fidelity_source if fidelity_source is not None else truncated,
                kwargs, prompt_ver, want_lang=fidelity_lang)
            # M1-1：assistant 半边的条目落到具体通道（conclusion / progress）。
            # 模型只回答 "user|assistant"——它不知道这一轮的 assistant 正文是
            # 最终结论还是工具循环里的过程说明，那是**调用方**才知道的事实。
            # 让模型猜等于把一个确定量交给概率（Q4 的 provenance 双写靠它）。
            if assistant_role:
                for f in output.facts:
                    if f.provenance == Provenance.CONCLUSION.value:
                        f.provenance = assistant_role
            if not output.facts and not output.matter_proposals:
                logger.debug("distill_empty model=%s ver=%s input_len=%d",
                             self._model, prompt_ver, len(source_text))
            else:
                logger.debug("distill_ok model=%s ver=%s facts=%d proposals=%d input_len=%d",
                             self._model, prompt_ver, len(output.facts),
                             len(output.matter_proposals), len(source_text))

            # 写台账（write-if-absent）：成功解析的结果（含真空）都缓存，
            # rebuild 命中复用 = 零 LLM 成本（ADR-0018 §4.1）。解析/调用失败不缓存。
            if self._journal is not None:
                try:
                    self._journal_put(truncated, prompt_ver, output, context_digest)
                except Exception as e:
                    logger.warning("distill_journal_put_failed model=%s err=%s", self._model, e)

            return output

        except Exception as e:
            self._fail_count += 1
            logger.warning("distill_failed model=%s err=%s - return empty",
                           self._model, router_sdk.error_text(e))
            # MS-1：同上——降级不抛，但要在返回值上留下"这是失败"的痕迹。
            # failure_kind="call"：调用本身没成（连不上/超时/认证/配额），
            # 是上游的问题不是这条 turn 的问题 → **不计重试次数**，
            # 否则断线期间每条轮次都在攒次数、攒满被丢弃 = 把积压烧光。
            return DistillOutput(model_name=self._model, failed=True,
                                 failure_kind="call")

    # ── M1-3（拍板 1）：台账 key = hash(上下文摘要 + 消息) ──────────────────
    #
    # 为什么上下文要进 key：带上下文蒸馏之后，**同一句话在不同脉络下确实该蒸出
    # 不同产物**（"继续" / "还是不行" / "按上面那个改"）。v3 的 key 只 hash 正文，
    # 于是这类消息在任何语境里都命中同一条缓存，蒸出来的东西必然是错的。
    # 重建等价性由 context_digest 的**确定性装配**保证（build_context_digest 纯函数，
    # 且只取该轮自带的历史，不取批次邻居——见其 docstring）。
    #
    # 两个 helper 用 TypeError 探测旧签名：台账实现（IndexDistillJournal / 测试替身）
    # 不一定支持 context_digest 参数，缺失就退回旧调用（v3 行为逐字不变）。

    def _ledger_get(self, text: str, prompt_ver: str, context_digest: str) -> Any:
        if not context_digest:
            return self._journal.get_distill(text, self._model, prompt_ver)
        try:
            return self._journal.get_distill(
                text, self._model, prompt_ver, context_digest=context_digest)
        except TypeError:
            return self._journal.get_distill(text, self._model, prompt_ver)

    def _journal_put(self, text: str, prompt_ver: str, output: DistillOutput,
                    context_digest: str) -> None:
        if not context_digest:
            self._journal.put_distill(text, self._model, prompt_ver, output)
            return
        try:
            self._journal.put_distill(
                text, self._model, prompt_ver, output, context_digest=context_digest)
        except TypeError:
            self._journal.put_distill(text, self._model, prompt_ver, output)

    # ── 已蒸话语 memo（2026-08-08 欠采修复）──
    #
    # core 的 `_collect_candidates_v4` 用 `getattr` 探测这两个方法，缺失即降级回
    # 「只蒸 discourse[-1]」的旧行为。签名对 core 是**单参**（只问内容），
    # `prompt_ver` 在这里补 —— 换 prompt = 该重蒸一遍，与 distill/ 台账同款分代。

    def discourse_seen(self, text: str) -> bool:
        fn = getattr(self._journal, "discourse_seen", None)
        if not callable(fn):
            return True   # 台账实现不支持 -> 视作"都蒸过" = 旧行为逐字不变
        return bool(fn(text, self.prompt_ver))

    def mark_discourse(self, text: str, ledger_key: str = "") -> None:
        fn = getattr(self._journal, "mark_discourse", None)
        if callable(fn):
            fn(text, self.prompt_ver, ledger_key)

    def summarize_file(self, path: str, content: str) -> tuple[str, list[str]]:
        """ADR-0028 E7.1：文件内容摘要 + 关键词（独立台账版本，同 hash 零成本）。

        返回 (summary, keywords)；失败返回 ("", []) —— 调用方降级为只录元数据。
        """
        import json as _json

        text = f"# {path}\n\n{content}"[:_MAX_INPUT_CHARS]
        if self._journal is not None:
            try:
                cached = self._journal.get_distill(
                    text, self._model, FILE_SUMMARY_PROMPT_VER)
                if cached is not None and cached.facts:
                    self._journal_hit_count += 1
                    f0 = cached.facts[0]
                    return f0.content, list(f0.entities)
            except Exception as e:  # noqa: BLE001
                logger.warning("filesum_ledger_get_failed err=%s", e)

        self._call_count += 1
        try:
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _FILE_SUMMARY_SYSTEM},
                    {"role": "user", "content": text},
                ],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "timeout": self._timeout,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self._api_key:
                kwargs["api_key"] = self._api_key
            resp = router_sdk.completion(**kwargs)
            obj = _extract_json_obj(resp.choices[0].message.content or "")
            if not obj:
                return "", []
            summary = str(obj.get("summary", "") or "").strip()[:300]
            keywords = [str(k).strip() for k in (obj.get("keywords") or []) if str(k).strip()][:8]
            if self._journal is not None and summary:
                try:
                    self._journal.put_distill(
                        text, self._model, FILE_SUMMARY_PROMPT_VER,
                        DistillOutput(
                            facts=[DistillFact(content=summary, kind="general",
                                               entities=keywords)],
                            matter_proposals=[], model_name=self._model),
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning("filesum_journal_put_failed err=%s", e)
            _ = _json  # 保持 import 明确（解析走 _extract_json_obj）
            return summary, keywords
        except Exception as e:  # noqa: BLE001 —— 文件摘要失败只降级，不影响主管线
            self._fail_count += 1
            logger.warning("filesum_failed model=%s err=%s", self._model, router_sdk.error_text(e))
            return "", []

    def _enforce_fidelity(
        self, output: DistillOutput, source_text: str,
        kwargs: dict[str, Any], prompt_ver: str, *, want_lang: str = "",
    ) -> DistillOutput:
        """E6.5 蒸馏保真：标识符不丢 + 语言钉死（最多多花一次调用）。

        顺序有讲究：先补标识符再钉语言——语言重蒸会整批换掉 facts，
        反过来做等于把刚补好的标识符又扔了。

        M1-4：`want_lang` 非空时钉**配置语言**，而不是"跟随输入语言"。
        跟随输入是 X2 那条一处病灶三层发病的病灶本身——同一轮 user 通道蒸出中文、
        conclusion 通道蒸出英文，ZH/EN 同义对 cosine 0.8639 < 判重阈值 →
        同一事实存两份，而中文 query 只能捞回其中一份。语言必须是配置量。
        """
        if not output.facts:
            return output

        from bladex_core.distill_fidelity import (
            RETRY_HINT_IDENTIFIERS,
            RETRY_HINT_LANGUAGE,
            attach_missing_identifiers,
            check_identifiers,
            check_language,
            mark_language_drift,
        )

        # ① 标识符保真
        missing = check_identifiers(source_text, output.facts)
        if missing:
            retried = self._retry_with_hint(
                kwargs, RETRY_HINT_IDENTIFIERS.format(missing=", ".join(missing)))
            if retried is not None and retried.facts:
                still = check_identifiers(source_text, retried.facts)
                if len(still) < len(missing):
                    output, missing = retried, still
            if missing:
                attach_missing_identifiers(output.facts, missing)

        # ② 语言钉死（M1-4：配置语言优先于"跟随输入"）
        want, got = check_language(source_text, output.facts, want_lang=want_lang)
        if want != got:
            lang_name = "Chinese" if want == "zh" else "English"
            retried = self._retry_with_hint(
                kwargs, RETRY_HINT_LANGUAGE.format(lang=lang_name))
            if retried is not None and retried.facts:
                _w, g2 = check_language(source_text, retried.facts, want_lang=want_lang)
                if g2 == want:
                    # 重蒸换了整批 facts，标识符要重新兜一次
                    still = check_identifiers(source_text, retried.facts)
                    if still:
                        attach_missing_identifiers(retried.facts, still)
                    return retried
            logger.warning("distill_language_drift want=%s got=%s model=%s",
                           want, got, self._model)
            mark_language_drift(output.facts)
        return output

    def _retry_with_hint(
        self, kwargs: dict[str, Any], hint: str,
    ) -> DistillOutput | None:
        """带追加约束重蒸一次。失败/不可解析返回 None（调用方保留原产出）。"""

        retry_kwargs = dict(kwargs)
        msgs = list(retry_kwargs["messages"])
        msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n" + hint}
        retry_kwargs["messages"] = msgs
        try:
            self._call_count += 1
            resp = router_sdk.completion(**retry_kwargs)
            text = resp.choices[0].message.content or ""
            return _parse_distill_json(text, self._model,
                                       finish_reason=_finish_reason(resp))
        except Exception as e:  # noqa: BLE001 —— 重蒸失败保留原产出，绝不抛
            logger.warning("distill_fidelity_retry_failed model=%s err=%s",
                           self._model, router_sdk.error_text(e))
            return None

    def stats(self) -> dict[str, int]:
        return {
            "calls": self._call_count,
            "fails": self._fail_count,
            "parse_fails": self._parse_fail_count,
            # MS-15：length 截断分桶（与 parse_fails 互斥计数——预算问题和
            # "输出不是 JSON"是两种病，混桶就都治不了）。
            "length_fails": self._length_fail_count,
            "length_salvaged": self._length_salvage_count,
            "ledger_hits": self._journal_hit_count,
            "model": self._model,
            # 两条路各有版本号，**分开报**。此前这里硬写 PROMPT_VER，
            # 于是不管实际走的是 v4 的 distill_turn 还是旧的 distill，
            # 收尾一律打 `prompt_ver=v3-items-001` —— 2026-08-08 复盘时
            # 据此一度判定"v4 prompt 根本没生效"，查了一圈才发现是仪器在骗人。
            "prompt_ver": PROMPT_VER,
            "turn_prompt_ver": TURN_PROMPT_VER,
        }


def _finish_reason(response: Any) -> str:
    """取 finish_reason，取不到返回空串（各家 SDK 字段偶有缺席，不因此炸）。"""
    try:
        return str(getattr(response.choices[0], "finish_reason", "") or "")
    except Exception:  # noqa: BLE001 —— 诊断字段，拿不到不影响主体
        return ""


def _extract_json_obj(raw_text: str) -> dict | None:
    """从 LLM 输出里抠出 JSON dict；抠不出返回 None。

    单独抽出来是为了**给重试判据复用**：判断"这次响应到底能不能用"必须不打日志、
    不计失败，否则一次响应会留下两条 parse_failed，反倒把日志读花。
    """
    if not raw_text or not raw_text.strip():
        return None
    text = raw_text.strip()
    # 去 markdown ```json ... ``` 包裹
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1]) if lines[-1].startswith("```") else "\n".join(lines[1:])
    first = text.find("{")
    last = text.rfind("}")
    if first < 0 or last < 0 or last <= first:
        return None
    try:
        data = json.loads(text[first:last + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _salvage_strings(text: str) -> list[str]:
    """从（可能被截断的）JSON 字符串数组正文里逐个抠出完整字符串。

    `text` 从 `[` 之后开始。截断处停下——与 `_salvage_objects` 同语义。
    """
    dec = json.JSONDecoder()
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        if text[i] != '"':
            break
        try:
            s, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            break
        if isinstance(s, str) and s.strip():
            out.append(s)
        i = end
    return out


def _salvage_turn_json(raw_text: str, model_name: str) -> DistillOutput | None:
    """MS-15（T1d）：从 length 截断的 v5 输出里打捞已完整的段。

    移植裁决器 `_salvage_objects` 的语义（截断只损失最后半条，不连坐整轮）
    到蒸馏的**顶层 dict** 形态：topic（截断前通常已完整）、keywords（逐串打捞）、
    facts / matter_proposals（逐对象打捞，复用裁决器同款实现）。

    打捞结果经 `_parse_distill_json` 走一遍正常字段校验（null 安全 / kind
    白名单 / topic 广播），保证与正常路径同一套出口。全部段皆空 -> None。
    """
    import re as _re

    from bladex_proxy.adjudicator import _salvage_objects

    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = ("\n".join(lines[1:-1]) if lines[-1].startswith("```")
                else "\n".join(lines[1:]))
    first = text.find("{")
    if first < 0:
        return None
    body = text[first:]

    data: dict[str, Any] = {}
    m = _re.search(r'"topic"\s*:\s*"((?:[^"\\]|\\.)*)"', body)
    if m:
        try:
            data["topic"] = json.loads(f'"{m.group(1)}"')
        except json.JSONDecodeError:
            pass
    m = _re.search(r'"keywords"\s*:\s*\[', body)
    if m:
        data["keywords"] = _salvage_strings(body[m.end():])
    m = _re.search(r'"facts"\s*:\s*\[', body)
    if m:
        data["facts"] = _salvage_objects(body[m.end():])
    m = _re.search(r'"matter_proposals"\s*:\s*\[', body)
    if m:
        data["matter_proposals"] = _salvage_objects(body[m.end():])

    if not any(data.get(k) for k in ("topic", "keywords", "facts", "matter_proposals")):
        return None
    return _parse_distill_json(json.dumps(data, ensure_ascii=False), model_name)


def _parse_distill_json(
    raw_text: str, model_name: str, *, finish_reason: str = "",
) -> DistillOutput | None:
    """解析 LLM JSON 输出 -> DistillOutput | None。

    - 容错提取 JSON（LLM 可能裹 markdown ```json）
    - junk kind 不入 facts（构造上拦截，ADR-0018 §3.1 修 R4）
    - 解析失败（无合法 JSON dict，含空响应）-> None（调用方不缓存，rebuild 可二次蒸馏）
    - 合法 dict（含空 facts，即"真无事实"）-> DistillOutput（真空也缓存，rebuild 零 LLM 成本）
    """
    data = _extract_json_obj(raw_text)
    if data is None:
        # 2026-08-05：加 finish_reason + 尾部片段。此前只有 raw_len + 前 80 字符，
        # 于是"开头明明是合法 JSON 却解析失败"完全看不出是被谁掐断的——排查那次
        # 只能靠推断。截断的证据在**尾巴**上，不在头上。
        logger.warning(
            "distill_parse_failed raw_len=%d finish_reason=%s head=%.80r tail=%.60r",
            len(raw_text), finish_reason or "?", raw_text, raw_text[-60:])
        return None

    # facts（junk 拦截 + ADR-0026 §4.2 六类 item_kind/subject/attribute）
    facts: list[DistillFact] = []
    for item in data.get("facts", []) or []:
        if not isinstance(item, dict):
            continue
        # 🔴 M0-2（复核 E4）：JSON 里的 `null` 不是"缺失"——`.get(k, "")` 只在 **键不存在**
        # 时给默认值，键存在但值为 null 时返回 None，`str(None)` 就是字面量 `"None"`。
        # 模型返回 `{"subject": null}` 是常态（prompt 说"无则留空"，它理解成 null），
        # 于是库里出现了一批 subject/attribute == 'None' 的条目——它们会在取代键上
        # **互相撞成同一个键**（scope, 'none', 'none'），把毫不相干的事实串成取代链。
        # 统一走 `or ""`：None / 缺失 / 空串 三种"没有"折叠成同一种。
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        # 旧 kind（v2 台账兼容）；junk 构造上拦截
        kind = str(item.get("kind") or "general").strip()
        if kind not in _VALID_KINDS:
            kind = "general"
        if kind == "junk":
            continue
        # 新 item_kind（v3）：合法六类才取，否则留空由 consolidation 侧 _resolve_item_kind 兜底
        item_kind = str(item.get("item_kind") or "").strip().lower()
        if item_kind not in _VALID_ITEM_KINDS:
            item_kind = ""
        subject = str(item.get("subject") or "").strip()
        attribute = str(item.get("attribute") or "").strip()
        entities = [str(e) for e in (item.get("entities", []) or []) if isinstance(e, (str, int, float))]
        # v7/M9：三条道（task|profile|tool）落**既有** `tags` 字段，不新增 schema 字段。
        # 为什么不新增：ADR-0026 铁律 3「无署名消费者的字段不许生产」——B 类（tool）的
        # 落库形态 Jason 明确本轮不拍，此刻造一个专用字段就是造死数据。
        # tags 是自由标记位，`mark_language_drift` 已是同款用法（append，不覆盖）。
        # 未知/缺失值一律不写标记（宁缺毋滥），下游 `getattr(f, "tags", "")` 读法不变。
        lane = str(item.get("lane") or "").strip().lower()
        tags = f"{_TAG_LANE_PREFIX}{lane}" if lane in _VALID_LANES else ""
        facts.append(DistillFact(
            content=content, kind=kind, entities=entities, tags=tags,
            item_kind=item_kind, subject=subject, attribute=attribute,
            # ── M1-1 v4 新槽位（v3 输出里没有这些键 → 全部落默认值，解析零回归）──
            importance=_parse_rating(item.get("importance")),
            valid_until=str(item.get("valid_until") or "").strip(),
            corrects=str(item.get("corrects") or "").strip(),
            provenance=_parse_source(item.get("source")),
        ))

    # matter_proposals（≤3）
    proposals: list[MatterProposal] = []
    for item in (data.get("matter_proposals", []) or [])[:3]:
        if not isinstance(item, dict):
            continue
        # M0-2 同款 null 安全：`{"title": null}` 曾产出标题字面量 "None" 的 Matter 提案。
        title = str(item.get("title") or "").strip()
        if not title:
            continue
        entities = [str(e) for e in (item.get("entities", []) or []) if isinstance(e, (str, int, float))]
        proposals.append(MatterProposal(title=title, entities=entities))

    # M1-1：轮级 event（铁律3——会话事件不产出为条目，折叠进 Matter 卡 lifecycle）。
    # 它是**一轮一个**，不是每条 fact 一个：一轮里"被打回"这件事只发生一次。
    # 挂在每条 fact 上会让同一个事件按 fact 数重复折叠（H2 那个 version 虚高的形状）。
    event = str(data.get("event") or "").strip().lower()
    if event not in _VALID_EVENTS:
        event = ""
    # 🔴 event 必须落在 **DistillFact** 上，不能挂在 DistillOutput 上。
    # 台账记录 `DistillRecord` 只持久化 `facts` + `matter_proposals`（models.py:473-474），
    # 挂在 output 上的字段在**台账命中那条路径上会丢** → 首次蒸馏与缓存复用产出不一致
    # → 重建等价性（G6）当场破掉，而且只在命中时发作，最难查的那种。
    #
    # 无 facts 的轮次不带 event 也没关系：MS-5 定的折叠目标是"该轮 facts 经归属落定的
    # Matter"，没有 fact 就没有折叠目标，本来就不折叠。
    for f in facts:
        f.event = event

    # 三段式 v5（T1b）：轮级 topic / keywords。
    # 缺键 → 空串/空列表，**不算 parse 失败**（模型偶发漏键不该整轮丢；
    # v3/v4 台账反序列化同样落默认值，解析零回归）。
    # topic 与 event 同款广播到每条 fact（台账穿越，见 DistillFact.topic 注释）。
    topic = str(data.get("topic") or "").strip()
    keywords = [str(k).strip() for k in (data.get("keywords") or [])
                if isinstance(k, (str, int, float)) and str(k).strip()][:10]
    for f in facts:
        f.topic = topic
        # 2026-08-14 拍板 A：keywords 从瞬态转持久，与 topic 同款广播
        # （台账穿越理由同上；历史台账反序列化 = [] 由派生兜底）。
        f.keywords = list(keywords)

    return DistillOutput(
        facts=facts, matter_proposals=proposals, model_name=model_name,
        topic=topic, keywords=keywords,
    )


def _parse_rating(raw: Any) -> int:
    """importance 1–10。解析不出 / 越界 → 0（= 未评分，退回 kind 基线，M1-5）。

    宽松而不是报错：模型偶尔会给 "8" 或 8.0 或 "high"。前两种收下，第三种归 0。
    """
    try:
        val = int(float(raw))
    except (TypeError, ValueError):
        return 0
    return val if 1 <= val <= 10 else 0


def _parse_source(raw: Any) -> str:
    """`"source": "user"|"assistant"` → provenance 值。

    v4 一轮一次统一调用（user+assistant 同包），来源信息必须由产出侧标注，
    否则通道在合流那一刻就丢了。assistant 侧先落 `conclusion`，
    再由 `_distill_with(assistant_role=...)` 改写成真实通道
    （conclusion / progress 是**调用方**才知道的事实，不该让模型猜）。
    未标注 → 空串（未知），消费者必须能处理未知。
    """
    val = str(raw or "").strip().lower()
    if val == "user":
        return Provenance.USER_DIRECT.value
    if val == "assistant":
        return Provenance.CONCLUSION.value
    if val in PROVENANCE_VALUES:
        return val
    return ""
