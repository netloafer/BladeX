"""记忆机制的开关解析（ADR-0027 §5.4）。

背景（为什么单独一个模块）：ADR-0024/0025/0026 统一改造的五个机制各自在三个包里
用 `os.environ.get(X, "") == "1"` 就地读环境变量，默认全关。2026-08-04 发布就绪审查
发现：机制已经完成并经真实流量验收（重复注入 0、阴性误注 0），但**默认值与配置模板
从未同步**——五个开关只存在于一份 planning 文档的 export 段里，仓库默认形态仍是
07-29 那个带 66 次误注 / 171 次重复的旧注入。发出去的会是未修版本。

结论落两条纪律：
1. 默认值集中在这里，不再散落在三个包的 `os.environ.get` 字面量里；
2. `config/.env.example` 与本表由测试对账（见 `test_flag_defaults.py`），
   任何一侧改了另一侧不改 = 测试红。

关闭回滚通道保留：`BLADEX_<FLAG>=0`（或 false/no/off）退回旧行为，
既有"关闭逐字一致"测试仍然是有效的回归网。
"""

from __future__ import annotations

import os

# 开关名 -> 默认值（True = 默认开启）。改这里必须同步 config/.env.example。
MEMORY_FLAG_DEFAULTS: dict[str, bool] = {
    # （BLADEX_INJECT_INDEX_RECALL / BLADEX_INJECT_PLANES 已于 2026-09-03 S1 删除：
    #  08-30 拍板 Memory Index 不再主动注入后，121 轮 `inject_plane_counters=0`、注入路径
    #  `index_search_done=0`，主动注入检索路径（prefetch.py + 三平面）整体退役，记忆改由
    #  模型经 `bladex_memory_search` 工具面按需取。不留 A/B 开关，回滚 = git 历史。）
    # 🔴 下面三个仍保留——它们的消费点**不在**注入路径：SOFT_SCORING / HOP_EXPAND 门
    # `MemoryIndex.search`（工具面 `bladex_memory_search` 走它），PROFILE_CARDS 门写侧
    # 画像捕获（USER.md/AGENT.md 投影的上游）。
    # （BLADEX_ITEM_BOUNDARY 已于 2026-09-03 S4 删除：U6 轴B 条目边界，2026-08-15
    #  拍板默认关（−22 分事故，docs/benchmarks/longmemeval-t3ad-20260815.md）、从未回开；
    #  其唯一消费点在 prefetch.py ③平面，随 S1 一起退役。）
    # U6 软打分融合（`MemoryIndex.search`：=0 退回 T8a 实体重排）。
    "BLADEX_SOFT_SCORING": True,
    # U6 有限跳（`MemoryIndex.search`：命中已被取代的旧条时换成取代者）。
    "BLADEX_HOP_EXPAND": True,
    # U10 画像被动捕获（consolidation 写侧；USER.md/AGENT.md 投影的上游）。
    "BLADEX_PROFILE_CARDS": True,
    # （BLADEX_ATTRIB_L0 已于 2026-09-03 S4 连函数删除：ADR-0024 §4.5 L0「同单元同属」
    #  默认关、从未翻默认、零 live 读数；MQ-V6 实测碎片化瓶颈与任务单元无关。）

    # V-I2（ADR-0032 批一，2026-08-25）：实体键规范化（分隔符变体折叠，
    # `qwen3.8:27b` ≡ `qwen3.8-27B`；点号保留）。作用面 = fusion entity 通道
    # 匹配 + 同义折叠 Jaccard。=0 退回旧口径（strip+lower），单变量 A/B 与止血用。
    "BLADEX_ENTITY_CANON": True,

    # ── V-L5 账本锚层（ADR-0032 §4.5）──
    # 账本锚定 Matter 归属（主路径；D1/五层降兜底）。查找键 =
    # **`activation_scope(agent, project)` + 点时**（"这一轮发生时激活的是哪本"），
    # 🔴 **不是 session、不是终态绑定**——两条都试过、都是错的：
    #   MQ-L14 用 session 键查 scope 表 ⇒ 5390 轮命中 0，死机制却报告自己就绪；
    #   MQ-L15 用终态绑定 ⇒ hermes 全部 3622 轮history 拖进一张 CSS 评审卡（该 20 轮）。
    # ✅ **默认开（2026-09-02 翻默认，V-L5 T6；Jason 按 4a/4b 口径签收）**：
    # 4a 每张锚卡错边数对 pre（`member-coherence-pre-20260901.md`）不上升、
    # 4b 真实流量新引入 L2/L3 错边 = 0（读数表见调度台账 A1 行 / HANDOFF-20260902e）。
    # 翻之前 live `.env` 已 =1 跑了 8 天，flags 关 / live 开的分叉期间 gate 测的
    # 不是生产形态——翻默认改的是"默认形态 = 生产形态"。
    # 绝对误合并=0 不由本开关承担：ANCHOR 层的错边成因是账本切换失守
    # （`task-ledger-switch-diagnosis-20260901.md`，0.3.0）。
    # 开关保留作回滚通道（=0 ⇒ 零 ANCHOR 边，`test_flag_off_is_regression_channel`）。
    "BLADEX_LEDGER_ANCHOR": True,

    # ── 账本准入闸（ADR-0032 §4.5，2026-09-02）──
    # DPL 边落**账本锚卡**前过两道因果闸：① 轮次不得早于账本 created_at；
    # ② 轮次明确属于别的账本 ⇒ 拒。判定在 `ledger_runtime.ledger_gate`。
    #
    # 🔴 **默认开**，理由是它修的是正确性不是加特性：关掉 = 回到 09-02 之前
    # "内容相似压过账本边界"的行为（实测把 ADR-0024 的开发过程吸进「评估 v5
    # 架构」卡）。默认关会让测试与 live 分叉——`BLADEX_LEDGER_ANCHOR` 本身
    # 就是这个形状（这里 False、live 开着），别再加一个。
    #
    # 保留开关是**为了当仪器**：关/开各跑一次全量重建，锚卡成员差即本闸的
    # 拦截量（预估闸① 281 条，闸②增量未测）。锚层关着时本闸自动 no-op
    # （没有锚卡 ⇒ `target_ledger_id` 恒空 ⇒ 恒放行），不需要额外条件。
    "BLADEX_LEDGER_GATE": True,

    # ── V-P3 硬点 8：拦截播报（ADR-0032 §3.2 #8）──
    # 内循环期间以 reasoning/thinking 流播报 BladeX 在做什么，让用户感知工作进行中。
    # 🔴 **默认关**（2026-08-27），但**理由与初稿写的不同，已更正**：
    #
    # 初稿写"播报机制未经真实客户端验证"——**错的**。机制在 2026-08-25 的
    # V-R1 D7 就对五个 agent 实测过（survey §5c）：codex 过程实时可见 + 收尾折叠
    # （最理想形态、不回带）、Pi/dsh 全量渲染、CC 折叠成计时条、hermes 不渲染。
    #
    # 真实理由是：**我的实现与那次验证过的形态不一致**。参考实现就在仓库里
    # （`scripts/agent_probe_upstream.py`，D7 的 mock 上游），我却按协议名从头编，
    # 结果发出**不属于任何 item 的裸 delta**（缺 output_item.added 声明、
    # 缺 item_id/output_index/summary_index、缺收尾）。
    # 08-27 已照参考实现改成有状态发射器（`_ResponsesNarrator` 等），
    # 并加了与参考实现的对账守卫（`TestMatchesTheProbeReference`）。
    #
    # ⚠️ 同日还证伪了"播报导致 Codex 断开"这个归因：16:47 与 19:57 两次
    # **播报关闭**的内循环之后同样没有后续请求 —— 断开另有原因，仍未定位。
    #
    # 🔴 仍默认关：形态虽已对齐参考，但**这一版代码本身没在真机上跑过**。
    # 回开条件 = 真实客户端实测三协议渲染正常且不中断。
    #
    # ⚠️ 归因更正（2026-08-27 晚，MQ-L22）：此前两版注释把 Codex
    # 「内循环之后零后续请求」归给播报形态，**两次都错**。真根因在
    # `_synth_responses_text_events` 缺 `response.content_part.added`
    # ⇒ 官方解析器 IndexError ⇒ 断连，与本开关无关（关着也断）。
    # 播报确实另有一个编号撞车缺陷（同批已修），但它只在本开关开启时才发生。
    # 记下来是因为：**开关关着仍复现，本该立刻排除它**——我却又猜了一轮。
    "BLADEX_NARRATE_INTERCEPT": False,

    # ── MQ-L21：agent 自带零工具 ⇒ 不给账本面（2026-08-27 Codex 病例）──
    # 一个不带任何工具的请求要的是**一段文本**，不是"完成一件任务"。
    # Codex 的标题生成/建议生成原本零工具，我们塞给它 4 个 bladex_* 工具，
    # 它才有了建账本的能力 —— 建出来的账本占住激活位，用户真实任务只能新开。
    # 🔴 默认**开**，方向论证同 LANE_HARD_SPLIT：本项是**排除**不是合并，
    # 最坏情况是某个极简客户端漏拿账本面（可置 0 回滚），零误合并风险；
    # 而"默认关且不进模板"的反向教训在案（ADR-0027 §5.4）。
    # 代价实测见 identity.brings_no_tools 的 docstring。
    "BLADEX_LEDGER_REQUIRE_TOOLS": True,

    # ── G12.2 D1 范式（ADR-0031 §3；算法 = 决策表 v2）──
    # 默认延续 + 活跃档案窗口 + 首轮/后续二分：R1/R2/R4 直挂零 LLM，
    # R5 落回五层管线兜底（新开判决入 taskstate 台账钉首轮集合）。
    # 依赖 BLADEX_ATTRIB_CONTINUATION=1（S1 信号与 _prev_of 会话序都建在
    # GM-1 的扫描段上）。
    #
    # ✅ **2026-08-21 翻默认 False → True**，举证责任已履行（红线 2 要求的是
    # "未验证机制默认关"，不是"永远关"）。三样读数齐：
    #
    #   1. **误延续尺前后对照**（`docs/benchmarks/member-coherence-pre-g122-v11-20260820.txt`
    #      vs `member-coherence-post-g122-20260821.txt`，同参数同索引）：
    #      新出现的 **CONT 层判 116 条、BOTH = 0 (0.0%)** —— 全表最干净的一层
    #      （次好的 L4 是 0.5%），A实体零重叠只 3、B低包含只 2、交集 0。
    #      **R2 长链污染段仍为「无」**，即 58 条 R2 判决没有制造任何
    #      ≥3 连的错挂链。校准段 5/5 过，阴性对照 53.5% → 52.1%（判别力 ~50pp 稳）。
    #   2. **零回归**：L2 6→6 / L3 26→26 / C 画像挂卡 296→296 **BOTH 计数逐字不变**，
    #      合计 BOTH 绝对数 40→40（L4 −1 / L5 +1 在 642 / 78 的分母上互相抵消）。
    #      🔴 **口径更正（2026-08-21，下一棒读码抓出）**：已判 +150 里 **CONT 只占 116**，
    #      另外 34 条散在 L2 +4 / L3 +16 / L4 +11 / L5 +3——初稿写"+150 全部由 CONT
    #      贡献"是错的。结论方向不变（新增边净增 0 条 BOTH），但别按字面引用。
    #      同理 L2/L3 那两个"1.0% / 1.8% 不变"是**打印精度下不变**
    #      （6/578=1.038% → 6/582=1.031%），**计数**才是真的逐字不变。
    #      合计率 1.5%→1.4% **全部来自分母稀释（2710→2860），分子一条没减**——
    #      D1 不碰存量边，也没打算碰。
    #   3. **硬 gate**：live 21 小时 79 条层判决（R2 58 / R5 12 / AUX 5 / R4 3 /
    #      R1 1），`d1_target_missing` 与 `d1_ledger_write_failed` 在**全部 20 个
    #      consolidator 日志**里均为 0。延续:新开 = 4.8:1，正是 D1 举证责任反转
    #      要的形状（对照立卡时"四轮对话开 4 张卡"）。
    #
    # 🔴 **这份读数证明的边界，写下来免得后人读宽**：它证明的是**延续没有制造
    # 错误的成员归属**（误延续 = 0）。它**不**证明"该延续的都延续了"——
    # 误延续尺量的是"成员与卡搭不搭"，一条本该新开却被判延续的轮次，只要它的
    # fact 与卡内实体重叠，这把尺子看不见。那一面归剧本验收（ADR-0031
    # 「验收：剧本先于指标」），不归本读数。
    "BLADEX_TASK_STATE_D1": True,

    # ── MQ-S27 硬分流（2026-08-20 Jason 拍板；G12.2 决策表 v2 §7 前置边界）──
    # lane:profile / lane:tool 的 fact（含 item_kind=PROFILE_OBS）不进 Matter 归属、
    # 不建边、不进延续链：profile 走画像原料通道（profile_obs 既有消费路径），
    # tool 类留 KB 检索但不挂卡。写侧 M9 已 100% 产出 lane 标（324 profile /
    # 362 tool），读侧零消费 = 296 条画像/规则 fact 散挂任务卡（四条「小黑称呼」
    # 挂四张卡，member-coherence v1.1 实测）。本开关就是 lane: 标记的署名消费者
    # ——到期条款就此销账。
    # 🔴 默认开的方向论证（与 L0 默认关不矛盾）：本项是**排除**不是合并——最坏
    # 情况是被误标 lane 的任务 fact 漏挂（留在 KB 可检索、可 merge 补挂），
    # 不产生任何误合并风险；而"默认关且不进模板"的反向教训在案（ADR-0027 §5.4）。
    # 置 0 = 回滚通道（归属循环逐字回到改动前）。
    "BLADEX_LANE_HARD_SPLIT": True,

    # ── GM-1（MQ-S13）：对话延续信号进候选池 ──
    # 「延续(TAIL_CONTINUATION) ∧ 无新主体」→ 把上一轮 fact 所属的 Matter 加进本轮的
    # 归属候选池。🔴 **只加候选，判定仍由 L4 做**——它不改变任何判定层逻辑，只让
    # L4 看得见正确答案（四开事故的归因是"候选里没有姐妹卡 4 次 / 眼前有姐妹卡仍判
    # none 0 次"，账在候选生成不在判定层）。
    #
    # **默认开**，与 L0 的处置不同，理由是举证责任的方向不一样：L0 是**确定性合并**
    # （代表怎么判，成员就怎么走，跳过 LLM），未验证就开等于把误合并风险直接落库；
    # 本项最坏情况只是"多送一次裁决"。而 ADR-0027 §5.4 记着相反方向的教训——
    # 机制默认关且不进模板 = 那次"通过"是在临时 shell 里跑的，仓库默认形态从未变过。
    # 置 0 = 回滚通道（候选池逐字回到改动前）。
    "BLADEX_ATTRIB_CONTINUATION": True,

    # ── GM-2（MQ-S11）：按主题键反查候选 Matter ──
    # 归属候选此前只有 `search_matters(fact.embedding)` 一条纯向量通路。它在话题稳定、
    # 语料充足时是够用的（实测同一张卡跨 6 session / 3 agent 持续吸附成员，跨会话
    # 归并 43–45%）；本项补的是**向量散开的那一段**——同一对象不同动作
    # （进度评估／进展评估／问题报告／风险复核）内容差异大时向量捡不回来。
    # 判别力门与"单键永不放行"在 `key_based_matter_candidates` 里，是这条通路
    # 不退化成 MS-16 吸尘器的全部依仗。同样只加候选、判定仍由 L4 做。
    "BLADEX_ATTRIB_KEY_RECALL": True,

    # ── 第二批（ADR-0028 §4 / E2.1）：B 类"存在但不工作"全量通电 ──
    # 机制早已完整（supersede.py 精确取代键、lifecycle 事件折叠、file_ref 条目），
    # 差的只是"从未有人把开关打开"——与第一批同一类腐化，同一套对账纪律收编。
    #
    # 去重合并 + 强度评分 + 冲突覆盖（同键同义 strength+1 / 同键异值 t_invalid 取代）。
    "BLADEX_SUPERSEDE_ENABLED": True,
    # Matter 卡 lifecycle 事件（会话事件折叠进卡，铁律3）。
    "BLADEX_MATTER_LIFECYCLE": True,
    # （BLADEX_FILE_REF_ENABLED 已于 2026-09-03 S4 删除：M1-2/拍板 3 退役后它只剩
    #  "为对账不缺项"这一个存在理由，本身零消费点——与 KEYWORD_CHANNEL 同款僵尸。）
    # BLADEX_KEYWORD_CHANNEL 已删除（三段式卡 T3c，2026-08-10）：旧 substring
    # 实现随 E6.2 删除后 flag 留成僵尸——全仓库零消费方，`bladex status` 却把它
    # 打成 off 误导观感。词法路（FTS5）无条件参与融合，不由开关控制。

    # ── ADR-0028 §2/§3（E4）：Prompt Cache TTL 感知 + 冷启相关性裁剪 ──
    # 温热期：全量转发、冻结计划复用、禁止新增降解（保上游 prompt cache）；
    # 冷启期：装配正常跑 + 相关性裁剪（cache 反正要重建，裁剪只赚不赔）。
    # `=0` 回退到"每轮都按冷处理"，即本卡之前的行为。
    "BLADEX_CACHE_TTL_AWARE": True,

    # ── ADR-0028 §5（E7）：C 类新能力 ──
    # 文件内容索引：读写类工具的 result 正文 → 摘要 + 关键词 → 独立检索通道。
    # 取代裸路径 file_ref（实测 30 条随机 query 的 top-10 被裸路径占 20.7%）。
    "BLADEX_FILE_INDEX": True,

    # ── M1-2（记忆核心链改造）：蒸馏 v4 两级路由 ──
    # 先分流（话语 / 规则文件 / 粘贴物）→ 话语路才走**一轮一次统一调用**
    # （user + assistant 同包 + 上下文摘要 + 时间锚）。
    # 关掉 = 回落 v3 的三条独立流水线（user / conclusion / progress 各蒸各的），
    # 行为逐字不变——这是**回滚通道**，不是"待启用"（ADR-0027 §5.4 的教训：
    # 机制默认关且不在模板里，等于发出去的是未修版本）。
    "BLADEX_DISTILL_V4": True,

    # ── M2（记忆核心链改造）：写入时裁决 ──
    # novelty 降为"只拦重发"（0.98）→ 0.90+ 邻居打包 → 便宜档判 ADD/UPDATE/NOOP
    # → 写 judgment 台账（重放零 LLM）。它同时取代了两个失效的机制：
    # supersede 精确键（降为快路径）与冲突检测器（输出本就无人消费）。
    # 关掉 = 回到"novelty 阈值独自决定入不入库"的旧形态。
    "BLADEX_ADJUDICATE_ENABLED": True,
    # 全量重建时按 judgment 台账重放取代关系（MQ-S32 病 B 第五环，2026-08-31）。
    #   为什么默认开：它**恢复的是 ADR-0018 §4.1 明写声称过的行为**
    #   （"台账让 rebuild 重放零 LLM"），不是新功能——而该行为自建成起零消费者，
    #   八个 rebuild_id 每次都把已经付过 LLM 费用的取代关系推倒重来
    #   （实测 237 条 / 21.1% 的同键并存出自这里）。四条安全阀见
    #   `MemoryIndex._replay_supersede_judgments`。
    #   开关兼作 A/B 仪器：置 0 重跑 `why_no_supersede --batch` 应回到旧读数。
    "BLADEX_SUPERSEDE_REPLAY_ENABLED": True,

    # （BLADEX_FLASH_RENDER / BLADEX_FLASH_MATTERS 已于 2026-09-03 S5 删除：ADR-0031
    #  时代 consolidator 侧老渲染器的两个旋钮，默认关、live 未设、渲染器代码已删；
    #  Flash 树的唯一写者是 flash_daemon，模块门 = BLADEX_MODULE_FLASH。）
}

# 数值型参数的默认值（同样与 .env.example 对账）。
# 布尔开关走 MEMORY_FLAG_DEFAULTS，这里放"默认值本身有含义"的阈值。
MEMORY_NUMERIC_DEFAULTS: dict[str, float] = {
    # ── Flash 保鲜（2026-08-29 Jason 拍板：Flash 贵在精，目录数量也要维护；
    #    默认值暂定、按真实流量再调，全部用户可配）──
    # 每 project 保留最近 N 个 session 目录（清单同窗，滑出整删——文件是投影，
    # 历史在 Hub/Index 永远可查）。
    # 账本并发标注窗口（秒）：另一写者在窗口内动过激活账本 ⇒ 注入头加一行
    # 提示（2026-08-29 拍板；0=关）。写冲突的硬保护是 rev 乐观锁，这行只管可见。
    "BLADEX_LEDGER_COAUTHOR_WINDOW_S": 1800,
    "BLADEX_FLASH_SESSIONS_PER_PROJECT": 10,
    # 每 agent 保留的 project 目录上限（按最近活跃排序）。
    "BLADEX_FLASH_PROJECTS_PER_AGENT": 10,
    # project 活跃窗口（天）：窗口外的项目目录回收（PROJECTS.md 清单同窗）。
    "BLADEX_FLASH_PROJECT_ACTIVE_DAYS": 30,
    # agent 目录回收（天）：超期无流量删目录；AGENTS.md 名册行保留。
    "BLADEX_FLASH_AGENT_RETIRE_DAYS": 90,

    # 重要性遗忘门槛：importance 低于此值的条目**退出召回**（数据不删、as-of 可查，
    # 保 G6 重建等价性）。0 = 不过滤。
    #
    # 🔴 M0-8（复核 G3）+ B5 口径订正（2026-08-06 用户拍板）：0.2 → 0.0，**并且永久退役**。
    #
    # 起因：importance 公式的三个动态输入全部断粮——strength 断在合并从未运转（模块2）、
    # ref_count 断在注入长期"只注硬规则"（G1，信号恒空）、半衰减是唯一在动的项，
    # 于是 importance 事实上只剩「kind 基线 × 时间衰减」。在这种状态下开门槛
    # = **按年龄无差别遗忘**（G3 实测：均匀衰减 66 天驱逐用户偏好）。
    #
    # 🔴 不要在三时钟（M3）落地后"按新分布重开这个门槛"——复核汇总章二表第 1 行定的是
    # **一个遗忘机制，不是两个**：遗忘统一走三时钟（更新钟 × 相关钟 + kind 豁免表），
    # importance 只保留为**排序弱先验**与 dashboard 信号。两个遗忘机制并存就要额外定义
    # 它们的优先级与冲突消解，而那正是 G3 这类"没人说得清谁把数据吃了"的温床。
    "BLADEX_MIN_IMPORTANCE": 0.0,
    # 生命周期重算 job 的最小间隔（秒，6h）。consolidator 空闲轮触发，见 E2.4。
    "BLADEX_LIFECYCLE_INTERVAL_S": 21600.0,
    # 词法通道 bigram 影子表（M0-5）单次返回上限。**默认 0 = 关闭这条路。**
    #
    # 🔴 这是一条被评估集**否掉**的机制，默认值本身就是结论，不要随手改开。
    # 卡的前提是"二字中文词（美伊/伊朗/油价）在 trigram 上恒零命中，需要补一条路"。
    # 前半句为真（FTS 层面 0 → 11/16/4 命中），后半句**没有被数据支持**：
    # 三种接法在真实评估集上逐一实测，端到端全是净负（recall@10）：
    #
    #     ① 不接（基线）                 0.430   us_iran 0.217
    #     ② 展开 + 按名次对等合并        0.413   us_iran 0.133
    #     ③ 收窄到 len<3（≈等于不接）    0.430   us_iran 0.217
    #     ④ 展开 + 只追加在 trigram 之后 0.396   us_iran 0.050   ← 四跑最差
    #
    # ③→④ 的差异只有"bigram 尾部追加"这一项，净 −0.034、us_iran −0.167。
    # 原因是 RRF 只看是否在通道里出现：一条 2-gram OR 匹配来的低精度候选，
    # 哪怕排在词法通道末尾，也会拿到不可忽略的融合分，把 dense 的好结果挤出 top-10。
    # 中文 dense 通道对**话题类**查询本来就工作良好，词法路的边际价值是负的。
    #
    # 机制与影子表**保留**（写侧照常维护、`BLADEX_FTS_BIGRAM_LIMIT>0` 即可打开），
    # 留待 MS-9 融合权重标定后重新评估——那时词法通道有独立权重，
    # 不必再与 dense 抢同一个 RRF 位次。**在有新分数之前，它是关的。**
    "BLADEX_FTS_BIGRAM_LIMIT": 0.0,
    # ── M3-2：三时钟阈值（复核 5.9 拍板 6：**先立观测、真实数据标定后调**）──
    #
    # 🔴 这三个数是**起步值不是结论**。复核明确"不预先拍精度"，所以别照着它们
    # 调参——先让 M3-1 的相关钟跑出真实分布，再回来标定。
    #
    # Matter 的 2×2 出口矩阵（H1 的解）：
    #                  近期有命中          久无命中
    #   近期有更新      active（正常）      在写从不被用 → 过度采集审计信号
    #   久无更新        dormant（泰山类）   auto-close 候选（世界杯类，双钟停摆）
    "BLADEX_CLOCK_STALE_UPDATE_D": 14.0,   # 更新钟：多久没更新算"不活跃"
    "BLADEX_CLOCK_STALE_HIT_D": 30.0,      # 相关钟：多久没命中算"不相关"
    # provisional 滞留出口（H3：60% 的 Matter 困在"不注入"的中间态且无出口）
    "BLADEX_CLOCK_PROVISIONAL_MAX_D": 30.0,
    # 🔴 观测期护栏（5.9 修正 3）：`last_hit_at` 全库覆盖率低于此值时**只 log 不动状态**。
    # 理由是实测过的事故形态：07-28~08-05 注入中断期间全库零命中，
    # 若当时已有"30 天无命中即退场"，会发生**系统故障触发的大规模误杀**。
    # 钟只有在喂它的事件干净时才可信 —— 覆盖率就是"这个钟有没有在被喂"的度量。
    # 注意：M3-1 刚上线时覆盖率必然是 0，M5 全量重建又会清零，
    # 所以"上线后一段时间内只 log"是**设计使然**，不是没生效。
    "BLADEX_CLOCK_MIN_COVERAGE": 0.5,
    # （MS-6 钉卡自愈 BLADEX_PIN_COOLOFF_D / PIN_MISS_MAX 已随 S1 删除：消费点只在 prefetch。）

    # ── V-P3（ADR-0032 §3.2；agent-compat-survey D4 实测标定）──
    # 内循环墙钟预算（秒）。120 = 五 agent 全体无感区间（CC≥240 / hermes 150 起弹
    # 横幅 / codex 180 超时且会整请求重发）。per-agent 上调经 env 配，不写死。
    "BLADEX_INNER_LOOP_BUDGET_S": 120.0,
    # 内循环最大轮数（LLM 往返次数硬顶，与墙钟双保险——预算烧不完也不许无限转）。
    "BLADEX_INNER_LOOP_MAX_ROUNDS": 6.0,
    # ── V-L2/L4（ADR-0032 §4.3/§4.4）──
    # 切换防抖：同一 session 内距上次落定切换不足 N **用户轮**（role=user 消息数，
    # E0.2/MQ-L48：工具往返不计）的再切换拒绝（账本抖动 = 注入面抖动）。默认值待 E2 读数标定。
    "BLADEX_LEDGER_SWITCH_DEBOUNCE": 3.0,
    # 陈旧兜底：激活账本落后 ≥N **用户轮**（同口径）即注入陈旧标记（0=关）。默认值待 E2 读数标定。
    "BLADEX_LEDGER_STALE_TURNS": 20.0,

    # ── V-P3 内循环 keepalive（ADR-0032 §3.2）──
    # 纯 bladex 调用走内循环，那期间**两段都静默**：上游流的 bladex_* 事件被剥掉
    # 不转发，内循环本身又挂在 await 上。2026-08-27 live：Codex 23 秒零字节后
    # 判超时断开（内循环其实跑成功了，人没等到）。
    # 心跳形态 = SSE 注释行（三协议通吃、语义惰性，见 agency.py 注释）。
    # 10 秒 = chat 端点原来硬编码的 `_KEEPALIVE_INTERVAL_S`，已跑过真实流量。
    # 🔴 收进这里是因为**同一参数两个默认值 = 缺陷**（刚性原则 12）：
    # 新加协议路径时差点又拍一个 5.0，两条路径各一个值、与哪个更合理无关。
    # 0 = 关（回归通道）。
    "BLADEX_STREAM_KEEPALIVE_S": 10.0,

    # ── V-P4 硬点 1：拼接条目老化轮数（ADR-0032 §10-2）────────────────────
    # 0 = 不老化（当前默认）。ADR §10-2 明确拍板"**不预定默认值，真实流量标定**"
    # ——所以这里给的不是一个"我觉得合理"的数，是一个**旋钮**。
    #
    # 🔴 2026-08-27 补：此前它硬编码在 `SpliceLedger.__init__` 的默认参数里，
    # 既不在 flags 也不在 `.env.example`——**想标定都没有可转的旋钮**，
    # 而"标定后配"正是拍板要求的下一步。刚性原则 12 的同款形状。
    #
    # 老化的代价（ADR §3.2 硬点 5）：条目退出会让转发前缀一次性变化 ⇒
    # 上游 prompt cache 破裂一次。所以调大调小都不是白拿，要看读数。
    "BLADEX_SPLICE_MAX_AGE_TURNS": 0,
    #: agent 注册表缓存（origin_key→agent 绑定 + 待认领桶）的保留天数。
    #: 2026-08-26（MQ-A10）：注册表原本纯内存、重启即空 ⇒ 重启后的**第一个**
    #: 子调用请求必然落 `unknown-<hash8>`（同源继承没有可继承的记录），
    #: 且那一刻就被永久登记进待认领列表。TTL 存在的理由：origin_key 粒度较粗
    #: （UA+vendor），绑定放久了可能与现实脱节——过期即忘，退回 unknown 而非误署名。
    "BLADEX_AGENT_REGISTRY_TTL_DAYS": 7.0,

    # T3b（三段式卡，2026-08-10）：融合 entity 通道的热门实体衰减系数。
    # decay = 1/(1+coef·(N-1)²)，N = 实体关联 fact 数。取 mem0 的 0.001 起步
    # （N=50 时 ≈0.29、N=200 时 ≈0.025——'bladex' 这类全项目实体加权趋零）。
    "BLADEX_ENTITY_DECAY_COEF": 0.001,

    # ── M2-2：写入时裁决落地后的两个阈值（复核第 2 层）──
    #
    # novelty：0.95 → **0.98**。语义从"判重"降为"**只拦完全重发**"。
    # 0.95 那个值是让 novelty 独自承担"这条该不该入库"的判断，而 e5 在本库的
    # 噪声底就是 0.816–0.93 —— 它注定要么漏网（带日期的近重复变体互相看不见，E2）
    # 要么误杀。现在这个判断交给裁决器（看文本，不看阈值），
    # novelty 只负责把"一模一样的重发/重试"挡在门外，那件事 0.98 做得很好。
    "BLADEX_NOVELTY_THRESHOLD": 0.98,
    # 进裁决包的邻居相似度下限。3.9 用 live 库 370 条向量实测：
    #   ≥0.85 → 92% 的写入触发裁决、平均 35.8 条邻居（太宽，被噪声底决定）
    #   ≥0.90 → 62% 的写入触发、平均 4.5 条邻居  ← 正好一个裁决包
    # 按本库节奏 ≈ 每天 9 次便宜档调用，成本不构成反对理由。
    "BLADEX_ADJUDICATE_MIN_SIM": 0.90,
    # 裁决包的邻居数上限。**不是 5**：3.9 实测跨语言冲突对（EN 新事实 vs ZH 旧事实）
    # 排在 top-8，k=5 会把它整个漏掉。语言钉死（M1-4）后这类对会回到 0.95+ 区间，
    # 但存量数据上 k 必须够宽。
    "BLADEX_ADJUDICATE_TOPK": 10.0,
    # 一次裁决调用最多打包多少条候选（2026-08-08 立）。
    #   此前**不分块**：805 条塞进一次调用，返回 JSON 被输出上限截断、解析失败，
    #   805 条全部 fallback ADD —— 单次解析失败的爆炸半径 = 整个库。
    #   分块后单块失败只损失该块。25 是保守起点，未经真实流量标定。
    "BLADEX_ADJUDICATE_CHUNK": 25.0,
    # 一次嵌入调用最多送多少条文本（2026-08-31 事故驱动，MQ-A28/A29）。
    #   此前候选嵌入是**一次性整批**：全量重建把 6 万段话语塞进一个调用 ⇒
    #   ① IPC 后端 120s 超时结构上必超；② 期间零进度，"在跑"与"卡死"不可区分；
    #   ③ 打不断（`^C` 无反应，原因未定位）。
    #   512 是保守起点，**未经标定**——真实速率由 `consolidate_embed_progress`
    #   的 `rate=` 给出，标定后再调。
    "BLADEX_EMBED_BATCH": 512.0,
    # 一轮提炼里**降级成 ADD 的候选比例**达到该值 → 判为裁决侧上游中断：
    # 本轮不写事实、不标记消费，退避重试（MQ-S32 病 B，2026-08-31 立）。
    #   为什么要有它：裁决失败是**降级不抛**的，`_all_add` 把整块判成 ADD，
    #   而 ADD 是合法判决 —— "上游断了"与"这批确实都是新事实"在下游长得一样，
    #   消费标记照打，当场制造同键并存（实测 46 条）。与蒸馏中断守卫同型同默认值。
    #   分母是候选条数不是调用次数（分块后一次调用判 25 条，按调用数会被稀释）。
    #   0 = 关闭守卫，回到旧行为。
    "BLADEX_ADJUDICATE_OUTAGE_ABORT_RATIO": 0.5,
    # GM-1「无新主体」的门槛：本轮相对上一轮新增的主体信号数 ≥ 该值 → 判为引入新任务、
    # 不进候选。🔴 **这个值不承重**（候选只送裁决），探针扫过 1→12 收益列全程高于
    # 风险列、两列不交叉（14.9%→18.0% vs 6.5%→2.3%）——结论不是阈值的产物。
    # 3 = 探针呈现所用的中档值；调大只是多送几次 L4。
    "BLADEX_ATTRIB_CONT_NEW_SUBJECT_MAX": 3.0,
    # GM-2 判别力门：主题键出现在超过 `本值 × 库内可链接 Matter 数` 张卡上 → 判定为
    # 泛词，不参与按键反查（下限 2 张，见 `key_based_matter_candidates`）。
    # 🔴 **分母随库走，不是绝对数**：同一个"出现在 5 张卡上"，在 20 张卡的库里是泛词、
    # 在 5000 张卡的库里是强标识符——绝对阈值当门槛正是 MQ-S1 清查的那个设计模式。
    # 0.05 [标定]：当前库 103 张卡 → 门槛 5 张，挡掉 bladex(42)/skill.md(19)/ollama(14)，
    # 而 MQ-D11 的分布说明代价很小（903 个键里 732 个只出现在 1 张卡上）。
    # 待真实读数复标：`index_attrib_key_recall` 的 key_candidates/key_adopted 比值。
    "BLADEX_ATTRIB_KEY_DF_RATIO": 0.05,
    # T3-A（检索侧 16 点收复卡，2026-08-15 拍板「参与召回+排序折价」）：
    # profile_obs 从整类禁注（QUOTA_MAX_PROFILE_OBS=0）改为**限额参与**。
    # 根因：T1 归因 11/18 错题的时间锚（"加入/开始/购买 于 DATE"）蒸成
    # profile_obs，在类型配额层被整类丢弃 → 检索结构性不可达；而①平面画像卡
    # 实测不聚合其内容——两头无消费者=死数据。0 = 回退旧行为（回归通道）。
    # （BLADEX_ATTRIB_L0_SPAN_H 已随 ATTRIB_L0 于 2026-09-03 S4 删除。）
    "BLADEX_QUOTA_PROFILE_OBS": 2.0,
    # 2026-08-20 读数驱动（`docs/benchmarks/channel-votes-20260820.md` §11）：
    # 同义折叠的 cosine 阈值。**这是注入面丢记忆的头号原因**——仪器把「e5 排进
    # 前 5 却没进注入」按出局阶段拆开：fold 80.7% / quota 9.6% / mmr 9.6% /
    # **rank 0.0%**（通道投票根本不决定去留，dense 保底与融合权重两条修法据此销账）。
    #
    # 根因是 fusion.py 内部自相矛盾：模块开头写着「库内 cosine 噪声底 0.88–0.93，
    # 绝对阈值形同虚设」，RRF 整套就是为弃用绝对阈值而设计，而 `FOLD_COSINE = 0.92`
    # 正埋在那个噪声底里。两条独立证据都指向 0.97：
    #   ① 阈值扫描（150 条真实 query）流失率 0.92→29.1% / 0.95→24.0% /
    #      **0.97→18.1%** / 0.98→15.1%，而注入冗余在 0.97→0.98 那一步翻倍；
    #   ② 人工签收 10 组：真重复 cos 0.972/0.979，误折 cos 0.926–0.964。
    #
    # 🔴 **2026-08-21 终验后默认值回退 0.97 → 0.92**（`docs/benchmarks/
    # longmemeval-fold-ab-20260821-064603.md`）。共享库 A/B（代码指纹全程一致、
    # 两档看逐字相同的 Memory Index）：**0.92 → 72.0% ／ 0.97 → 68.0%**。
    #
    # 🔴 但 −4pp **落在噪声内，无法判定**（不是"更差"）：
    #   · McNemar 精确检验 p=0.727（discordant 仅 8 题，翻案 3 / 退步 5；
    #     8 个 discordant 上要 p<0.05 必须 8:0）；
    #   · **同阈值重跑对照**（08-20 `bench_batched.sh` 0.92 档 38/50=76.0%
    #     vs 08-21 共享库档A 0.92 36/50=72.0%，同数据同切分）批级差
    #     `[0,−1,+3,−4,0]` —— **单批就抖 ±3~4 题，而 A/B 整体只差 2 题。噪声底 ≥ 信号。**
    # ⇒ 回退的理由是**"无证据支持改动"**，不是"0.97 更差"。
    # 机制、flag、接线、测试全部保留：设 `BLADEX_FOLD_COSINE=0.97` 即复现实验档。
    #
    # 备选解释待验：`top_k` 恒定 15，少折 ⇒ 同一件事占更多名额 ⇒ 挤掉多样性。
    # 扫描用的"冗余对/轮"代理（0.92→0.97 只涨 0.12）自指且可能低估代价。
    # 🔴 更要紧的连带结论见台账 MQ-S43：**LongMemEval-50 分辨不了 4pp 级差异**，
    # 一批以"涨了 1–2 点"为验收判据的历史卡随之失效——别再用它做细粒度判据。
    "BLADEX_FOLD_COSINE": 0.92,
    # T3-A 排序折价：profile_obs 的融合总分乘数（<1 = 降权，防止画像原料
    # 挤掉同等相关的 assertion/lesson）。0.85 是保守起步值 [标定]，
    # 由内部尺子（retrieval_eval）复测守回归；1.0 = 不折价。
    "BLADEX_PROFILE_OBS_DISCOUNT": 0.85,
    # （Matter 钉卡门 BLADEX_MATTER_PIN_MIN / PIN_MARGIN / PIN_SOLO_MIN 已随 S1 删除：
    #  唯一消费点是 prefetch 三平面的②平面钉卡，标定素材在 git 历史与
    #  data/bench_batched/t3d_*_calibration.txt。）
    # MQ-R1（2026-08-17）：facts 向量表的碎片上限，超过即在 consolidation pass
    # 收尾压缩。0 = 关闭这条自维护（回归通道）。
    # 口径 = **当前版本的 fragment 数**（`fact_vector_fragments`），不是 `data/`
    # 目录的文件数——压缩有意保留旧版本文件，按文件数判会永远触发不停。
    #
    # 这个数**是从热路径预算推出来的，不是拍的**（08-17 两点线性 + 外推验证）：
    #     search ≈ 410ms + 3.63ms × 碎片数
    #     800ms 预算 − 410ms 固定底 − 22ms（embed+matter）= 368ms 余量 → ≈100 碎片
    # 所以 **`BLADEX_HOTPATH_BUDGET_MS` 改了，这个数要跟着重算**
    # （MQ-R6 把固定底降下来之后，预算回调 250ms，余量只剩几十毫秒）。
    #
    # 🔴 它防的是一个**已经兑现过两次**的事故：MS-14（08-09）手工压一次、
    # 自维护没做 → 08-16 一次全量重建把 1619 条 fact 写成 1619 个碎片 →
    # 检索 6.4s → 每轮撞 deadline 降级只注硬规则 → 注入面空转一个月。
    # 定时任务防不住这个形态（碎片是被"写"堆出来的，不是被"时间"堆出来的），
    # 所以触发点挂在写侧的 pass 收尾。
    "BLADEX_INDEX_MAX_FRAGMENTS": 100.0,
    # MQ-S9 / G12.2-pre（2026-08-20，Jason 拍板）：fp: 指纹会话桶的时间窗切分——
    # 同 (user, agent, fp) 桶相邻请求 gap 超过此秒数 → 新纪元后缀（identity.py
    # `_FpEpochWindow`）。取值依据 = probe_session_gaps 读数：gap ≤5m 占 96.5%、
    # 尾部摊平无锐利谷 → 按代价不对称取保守侧 4h（切早断锚的代价 > 切晚多留候选；
    # 4h 切 41 个明确跨会话点，激进备选 1h 切 69 个）。0 = 关闭（回旧撞桶行为）。
    # 只作用于 fp: 桶——显式 session 对照段零长尾（同一份读数），切它是替 agent 做主。
    "BLADEX_SESSION_FP_WINDOW_S": 14400.0,
    # 写入侧蒸馏并发路数（1 = 串行）。**唯一真相源**——三条消费路径都读这里：
    # `bladex sync run`、CLI 拉起的 consolidator 守护进程、`MemoryIndex.rebuild_from_hub`。
    # 收口前这三处各带一个默认值（4 / 1 / 1），没人说得出生产在跑哪个。
    #
    # 🔴 这是一个**兑现过两次**的缺陷的机制解，两次都是"默认值散落在调用方"：
    #   ① 2026-08-13：`consolidator.py --concurrency` 的 help 写"默认取 env"、
    #      实现是 `default=0 → 1`，bench 全程串行蒸馏 → 排空超时 → 整跑数据作废。
    #      当时**只改了 help 文案、没改实现**，把缺陷记录成了规格。
    #   ② 2026-08-22：同一个默认值打在生产上——`cli._start_consolidator()` 起
    #      子进程时只传 `--interval`，于是 `bladex start` / `bladex consolidator
    #      start` 拉起的守护进程并发恒为 1。实测 mean 23.2s/次 × 8–9 次/轮串行
    #      = 5–26 轮/小时，白天流入高于消费，Memory Index 积压 720 轮不降。
    #      而 `bladex sync` 那条路默认 4 —— 手工 sync 快、守护进程慢，
    #      两条路默认值不一致本身就是缺陷（ADR-0027 §5.4 同型腐化）。
    #
    # 取 4 的依据：`bladex sync` 已按 4 跑过全量重建（2845 轮 2.49h）无上游 429。
    # 上游限流时调小；显式 1 保留串行回滚通道（`--concurrency 1`）。
    # 彻底解法（批内合并调用、降低每轮 8–9 次的调用数）另立卡，本值只是止血。
    "BLADEX_DISTILL_CONCURRENCY": 4.0,
    # V-C1 / MQ-S47：每个 session_prefix 保留多少条前缀候选，供
    # `MemoryHub.check_prefix_changed` 遍历匹配。1 = 退回旧的单槽行为（回滚通道）。
    #
    # 🔴 为什么必须 >1：agent 会在**同一个 session_prefix 下混跑多路流量**。
    # claude-code 实测两路——主会话（msg_count 单调递增 2→173）与 3 条消息的
    # 小请求（标题生成/子 agent 之类，均匀插在主会话之间）。单槽被两路交替覆盖，
    # 于是双向假阳性：小请求撞 `len(3) < prev_msg_count` 直接 True；
    # 主会话拿小请求的 `(hash_of_3条, 3)` 当基准，前 3 条对不上也 True。
    # 两跑读数：36 次告警 35 次是 3↔N 交替、18 次告警 17 次是——
    # **54 次里 53 次假阳性，真阳性 0**，且第二跑里它是唯一的 warning
    # （18/18，error=0），warning 通道被它占满。
    #
    # 取 4 的依据：主会话 1 槽 + 小请求 1 槽 + 2 槽余量（未观测到第三路，
    # 但 agent 行为是既定事实、我们只能适配，见刚性原则 10）。溢出时最老的槽
    # 被挤出，那一路会**重报一次** changed —— 可接受，因为它自会重建自己的槽。
    #
    # 🔴 **不要**改成"msg_count 暴跌就不覆盖"那种判据：真压缩的信号形状与小请求
    # 一模一样（都是消息数骤降），那样只是把假阳性换成假阴性，而假阴性是静默的。
    "BLADEX_PREFIX_SLOTS": 4.0,
}

# 字符串型参数（同样集中默认值；与 `.env.example` 由 test_flag_defaults 对账）。
MEMORY_TEXT_DEFAULTS: dict[str, str] = {
    # ── 装配按 agent 开（2026-09-03 拍板 e，Fable 版；MQ-CA6）──
    # 逗号分隔的 agent_id 清单；只有清单内的 agent 跑 ContextAssembler（L1 证据
    # 降解 + L2 单元摘要 + 冷启裁剪），其余 agent 走 `assembler=None` 同款原样返回。
    # 🔴 默认只开 `claude-code`：08-28 止血读数只在 CC 上测过；live 其它 agent 全在
    # 掉字（hermes:default p50 0.508 / hermes:accept 0.749 / Pi 0.816，
    # `chars_after/chars_before`；codex 纯 L1 边际保留率 0.318，MQ-CA4）。
    # **代价已付**：非 CC 上游 token 回升 ~2.4×（hermes 44K→106K 字符）——北极星
    # "精确优先于省注入"有意付。正解 = per-agent 阈值（MQ-CA4，0.3.0）。
    # `*` = 对所有 agent 开（回到 09-03 之前的形态）；空 = 全关。
    # 匹配规则：完整 agent_id 或其 base（`hermes:default` 命中清单里的 `hermes`）。
    "BLADEX_ASSEMBLY_AGENTS": "claude-code",
}

#: 注入条目数上限（原 `bladex_core/prefetch.py` 的 `MAX_PREFETCH_K`，2026-09-03 S1 随该
#: 模块删除迁到这里）。15 = 2026-08-14 LongMemEval 基准标定值（TOPK=3: 48% / 15: 68%）。
#: S1 后主动注入检索路径已退役，本值只剩两个读点：`server._inject_top_k`（传给
#: InjectionSource.top_k，暂无消费）与 `admin_read` 的 `bladex config show`；
#: 0.3.0 重建检索时从这里取默认，不再拍脑袋。`BLADEX_INJECT_TOPK` 可覆盖。
MAX_PREFETCH_K = 15

# ── C1（2026-09-03 批 B）：三档登记 ────────────────────────────────────────
#
# 瘦身复核 v2 §2d：旋钮按"它回答什么问题"分三档，档位登记在这里、由测试对账：
#   mechanism   机制开关（bool）：开/关一个行为；默认值 = 生产形态。
#   calibration 标定值（数值/字符串）：阈值、窗口、并发——"待真实流量标定"的那一类。
#   instrument  仪器：默认值就是生产形态，存在只为 A/B 对照或止血；**不进 .env.example
#               正文**（只在文末「仪器」段登记，用户日常不该碰）。
# 不物理重排上面两张表——它们的注释是每个默认值的举证记录，搬动 = 制造无意义 diff。
FLAG_TIERS: dict[str, str] = {
    **{k: "mechanism" for k in MEMORY_FLAG_DEFAULTS},
    **{k: "calibration" for k in MEMORY_NUMERIC_DEFAULTS},
    **{k: "calibration" for k in MEMORY_TEXT_DEFAULTS},
    # 仪器（自己的注释里写着"保留开关是为了当仪器 / 单变量 A/B"）
    "BLADEX_LEDGER_GATE": "instrument",
    "BLADEX_ENTITY_CANON": "instrument",
}
FLAG_TIER_NAMES: tuple[str, ...] = ("mechanism", "calibration", "instrument")

_FALSEY = {"0", "false", "no", "off", ""}


def flag_text(name: str, default: str | None = None) -> str:
    """读一个字符串型参数：未设 → MEMORY_TEXT_DEFAULTS（或显式 default）。"""
    raw = os.environ.get(name)
    if raw is None:
        if default is not None:
            return default
        return MEMORY_TEXT_DEFAULTS.get(name, "")
    return raw.strip()


def flag_csv(name: str, default: str | None = None) -> frozenset[str]:
    """逗号分隔清单 → 去空白、去空项的集合（`BLADEX_ASSEMBLY_AGENTS` 这类）。"""
    return frozenset(p.strip() for p in flag_text(name, default).split(",") if p.strip())


def flag_enabled(name: str, default: bool | None = None) -> bool:
    """读一个记忆机制开关。

    未设环境变量 → 用 MEMORY_FLAG_DEFAULTS（或显式传入的 default）。
    设了 → `0/false/no/off/空` 为关，其余（含 `1/true/yes`）为开。
    """
    raw = os.environ.get(name)
    if raw is None:
        if default is not None:
            return default
        return MEMORY_FLAG_DEFAULTS.get(name, False)
    return raw.strip().lower() not in _FALSEY


def flag_number(name: str, default: float | None = None) -> float:
    """读一个数值型记忆参数（ADR-0028 E2.1）。

    未设环境变量 → 用 MEMORY_NUMERIC_DEFAULTS（或显式传入的 default）。
    设了但解析不出数 → 回落默认值（配置写坏不该让热路径崩）。
    """
    fallback = default if default is not None else MEMORY_NUMERIC_DEFAULTS.get(name, 0.0)
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(fallback)
    try:
        return float(raw.strip())
    except ValueError:
        return float(fallback)
