"""AgencyRuntime 的文件型输入：账本模板 / 系统自我介绍 / AGENTS 名册，以及末条 user 文本取值。

09-06 F0.1 自 `agency.py` 拆出，零行为（函数体逐字搬家）。门面 `bladex_proxy.agency` re-export。
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()


#: 「这段 user 文本像是机器写的」的**补充**形状——**只用于告警，不用于任何判定**。
#: 2026-08-27 Codex 病例的两条实测形态各贡献一条；`You are …` 那条覆盖面最广
#: （客户端把子任务的 system prompt 塞进 user 消息，是各家通用做法）。
#:
#: 🔴 MQ-L82（批 W W2，2026-09-22）两处改动：
#: ① **降级为补充项**——主判据改成复用取值侧的 `_clean_user_text`（见下）。
#:    这张表是"按一次采样写死的指纹"，MQ-S45 记过它的命：agent 换个变体就复发。
#:    实测拿 6 个 live 形态过它，**6/6 全漏** ⇒ 告警恒 0 是假绿。
#: ② 匹配从"开头 200 字符内 `in`"收紧为**剥完之后 `startswith`**。
#:    旧写法会把**提到**这句话的真人提问判成机器文本（"你觉得 You are a helpful
#:    assistant 这种 prompt 好吗"），而 MQ-L82 判据②把误判列为本条唯一红线：
#:    漏判只少一条告警，误判会让"用户明确说的这件事"被当噪声。
#:    这三条本来就来自**整条即模板**的病例（模板从第一行开始），锚定不丢覆盖。
#: 新增第一条 = codex 的 AGENTS.md 注入：`envelope` 成对剥掉 `<INSTRUCTIONS>` 后
#: 还剩标题行（≈50 字符，见 `envelope._ENVELOPE_PATTERNS` 的 codex_agents_md 注释）
#: ⇒ 它是 6 个形态里**唯一剥不空**的那个，靠主判据抓不住。
_MACHINE_TEXT_MARKS: tuple[tuple[str, str], ...] = (
    ("codex_agents_md_header", "# agents.md instructions for"),
    ("assistant_persona", "you are a "),      # "You are a helpful assistant…"
    ("md_overview", "# overview"),
    ("agent_persona", "you are an agent"),
)


def _machine_text_mark(text: str) -> str:
    """user 文本疑似机器模板 ⇒ 返回形态名；否则 ""。

    两级判据，**一张表两个消费方**（`agency/handlers.py` 建本告警 +
    `cli/ledger_cmds.py` doctor），不许各写一份指纹：

    | 级 | 判据 | 覆盖 |
    |---|---|---|
    | 主 | `_clean_user_text(text)` **剥完为空** ⇒ 机器文本，形态名取剥离 marks 的第一条 | 信封族（`<system-reminder>` / codex history / hermes 脚手架 / dsh runtime context…），随生产表自动跟进 |
    | 补充 | 剥完非空，但**剥完的文本以** `_MACHINE_TEXT_MARKS` 里某条**开头** | 剥不空的模板头（codex AGENTS.md 标题行、persona 句） |

    主判据复用的是**取值侧同一个函数**——`_last_user_text(for_goal=True)` 用它
    判"这条里有没有用户说的话"。同一个问题两处各写一套判据，早晚分叉
    （MQ-L82 就是分叉之后的样子：取值侧认得六种形态，告警侧一种都不认）。

    🔴 误判防线：空白 / 空串不判；补充项锚定开头，不在正文里搜。
    """
    if not (text or "").strip():
        return ""
    try:
        cleaned, marks = _clean_user_text(text)
    except Exception as e:  # noqa: BLE001 —— 告警侧不许因剥离失败而阻断建本
        logger.warning("agency_machine_text_probe_failed", error=str(e))
        cleaned, marks = text, []
    if not cleaned:
        # 剥完什么都不剩 = 这条里没有用户说的话 ⇒ 它整条就是机器文本。
        return marks[0] if marks else "envelope_only"
    head = cleaned.lstrip().lower()
    for name, mark in _MACHINE_TEXT_MARKS:
        if head.startswith(mark):
            return name
    return ""

def load_ledger_template(path: str = "") -> str:
    """读账本模板文件。**唯一来源 = `config/ledger-template.md`**
    （`BLADEX_LEDGER_TEMPLATE` 可指别处）；缺失/空 → 结构性兜底并告警可 grep。

    🔴 Jason 2026-08-25 拍板两条：①不许事出多头（core 侧不再存第二份模板正文）；
    ②**启动时预加载一次**，不在每个请求里读文件——`AgencyRuntime` 构建时取值
    存 `self.template`，改模板重启生效（与其它配置同语义）。
    """
    import os
    p = (path or os.environ.get("BLADEX_LEDGER_TEMPLATE", "").strip())
    if not p:
        # 🔴 **按部署根解析，不用相对路径**（2026-08-25 live 回归：首版写
        # `config/ledger-template.md` 相对 cwd，proxy 的工作目录不一定是仓库根 ⇒
        # 每次启动都静默回落到结构性兜底，那份精心写的模板从未被注入过。
        # 这正是 deployment.find_root 存在的理由——ADR-0027 §3 同型问题的既有解）。
        from bladex_proxy import deployment
        root = deployment.find_root() or os.getcwd()
        p = os.path.join(root, "config", "ledger-template.md")
    try:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            return text
        logger.warning("ledger_template_empty", path=p)
    except OSError as e:
        logger.warning("ledger_template_unreadable", path=p, error=str(e))
    from bladex_core.ledger import FALLBACK_TEMPLATE_MD
    return FALLBACK_TEMPLATE_MD


#: MQ-L77（2026-09-18）：自我介绍里**账本族**的段落用这对 HTML 注释包起来，
#: `render_system_notes(..., ledger=False)` 整段剥掉——账本工具面不注（模块关 /
#: 档位不在集合 / 零工具轮）时说明书也不能教模型 "call bladex_ledger_switch"，
#: 否则是「有令无器」：live n4 第 2 轮 codex 照着说明书发了一条工具列表里不存在的
#: 调用，白费一轮。与 08-30「工具面按族拆」是同一把尺。标记行本身**永远不进注入**。
LEDGER_SECTION_OPEN = "<!-- bladex:ledger -->"
LEDGER_SECTION_CLOSE = "<!-- /bladex:ledger -->"
_LEDGER_SECTION_RE = re.compile(
    re.escape(LEDGER_SECTION_OPEN) + r".*?" + re.escape(LEDGER_SECTION_CLOSE) + r"\n?",
    re.DOTALL,
)


def render_system_notes(text: str, *, ledger: bool) -> str:
    """按工具族装卸自我介绍：`ledger=False` 剥掉全部账本段；`True` 只去标记行。

    剥完把连续三个以上空行压成两个（段落边界照旧），避免留下一串空洞。
    未闭合的标记按原样保留——写错标记宁可让它可见，也不静默吞掉半篇正文。
    """
    if not text:
        return text
    if ledger:
        out = text.replace(LEDGER_SECTION_OPEN + "\n", "").replace(LEDGER_SECTION_CLOSE + "\n", "")
        out = out.replace(LEDGER_SECTION_OPEN, "").replace(LEDGER_SECTION_CLOSE, "")
    else:
        out = _LEDGER_SECTION_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", out).strip()


#: V-P6：自我介绍的三份来源，按此顺序拼接。缺文件跳过并告警（不阻断）。
#: `SKILLS.md` 现为占位（"No BladeX skills are available yet."）——**照注**：
#: 明确告诉模型"没有 skills"，好过让它去猜有没有。真有 skills 时换内容即可。
_SYSTEM_NOTE_FILES: tuple[str, ...] = ("AGENT.md", "TOOLS.md", "SKILLS.md")


def load_system_notes(dir_path: str = "") -> str:
    """读 `config/system/{AGENT,TOOLS,SKILLS}.md` 并拼成自我介绍正文（V-P6）。

    ## 来源与路径

    **唯一来源 = `<部署根>/config/system/`**（`BLADEX_SYSTEM_NOTES_DIR` 可指别处）。
    🔴 按**部署根**解析，不用相对 cwd 的路径——`load_ledger_template` 的
    docstring 记着这个教训：首版写相对路径，proxy 的工作目录不一定是仓库根，
    于是那份精心写的模板从未被注入过，只有一条 warning。同型问题不再犯第二次。

    与模板同语义：**启动时预加载一次**存 `self.about`，改文件重启生效。

    ## 缺失处置

    三份全缺 → 返回 `""`，调用方据此**整块不注**（不是注一个空壳）。
    部分缺 → 注已有的并告警。不设结构性兜底：自我介绍写错了不如不写——
    模板缺失还能靠 `FALLBACK_TEMPLATE_MD` 保住五段结构，而一段错的自我介绍
    会让模型对 BladeX 的能力边界产生错误预期，代价方向相反。
    """
    import os
    d = (dir_path or os.environ.get("BLADEX_SYSTEM_NOTES_DIR", "").strip())
    if not d:
        from bladex_proxy import deployment
        root = deployment.find_root() or os.getcwd()
        d = os.path.join(root, "config", "system")
    parts: list[str] = []
    missing: list[str] = []
    for name in _SYSTEM_NOTE_FILES:
        p = os.path.join(d, name)
        try:
            with open(p, encoding="utf-8") as f:
                text = f.read().strip()
        except OSError:
            missing.append(name)
            continue
        if text:
            parts.append(text)
        else:
            missing.append(name)
    if missing:
        logger.warning("system_notes_missing", dir=d, files=missing,
                       loaded=len(parts))
    if not parts:
        logger.warning("system_notes_empty", dir=d)
        return ""
    return "\n\n---\n\n".join(parts)


#: Goal 取值的长度上限（超长时截断并留标记——Goal 的价值是"用户一眼认出自己说的话"，
#: 一屏以上的机器文本进 Goal 只会毁掉这个价值。与已删的老渲染器 GOAL_INJECT_MAX_CHARS=600 同量级）。
_GOAL_MAX_CHARS = 600

#: Goal 专用：整条以成对包装标签开头（`<session>…</session>`、`<user_message>…` 等）
#: ⇒ 取标签内容，丢弃标签后的机器指令（CC 的 `<session>` 形态，2026-08-25 live）。
#: 标签名限字母数字下划线短横，避免误吃真实的 HTML/代码片段（`<div>` 这类也只在
#: **整条以它开头**时才命中，且内容会作为 Goal 让用户一眼看见——错了看得出来）。
_WRAPPER_TAG_RE = re.compile(
    r"^<(?P<tag>[A-Za-z][\w-]{2,30})>\s*(?P<inner>.*?)\s*</(?P=tag)>",
    re.DOTALL)

#: Goal 专用：**裸前缀句**形态的 agent 包装（既不是已知信封、也不是 `<tag>` 包裹，
#: 所以上面两道都漏）。2026-08-26 live 实证：jydesignhk 那 5 本账本的 Goal 全部以
#: `Fully describe and explain everything about this image, then answer the
#: following question:` 开头，用户真正说的话被压在 90 字符之后——5 个 Goal 的前缀
#: 逐字相同，区分度全在尾部。
#:
#: 只在**整条开头**匹配，且只剥这一句（后面照原样留给提炼）。Goal 会摆在用户
#: 眼前，剥错了看得出来——这也是敢用正则而不是模型判的原因。
#: 🔴 有意不动 `bladex_core.envelope.strip_envelopes`：它被蒸馏消费，改它要 bump
#: 台账版本并影响重建等价（同上一条注释的理由）。
_BARE_PREFIX_RES = (
    re.compile(r"^\s*Fully describe and explain everything about this image[^\n:]*:\s*",
               re.IGNORECASE),
)


#: Goal / 召回专用：**整条即 agent 运行时上下文**的裸文本形态——既没有成对标签、
#: 也不在 `envelope._WHOLE_MESSAGE_MARKERS` 里，所以现有两道全漏（MQ-L81）。
#:
#: 🔴 为什么落在这里而不是 `envelope.py`：那是蒸馏消费的模块，加一条模式要 bump
#: 台账版本并影响重建等价性（G19 边界明写"不碰 `strip_envelopes` 本体"）。
#: 与上面的 `_BARE_PREFIX_RES` / `_WRAPPER_TAG_RE` 同一条理由、同一个落点。
#: **正确归宿仍是 `envelope._WHOLE_MESSAGE_MARKERS`**，已登记 MQ-L81。
#:
#: live 依据（2026-09-20 Hub 全量，`scripts/probe_user_text_envelope.py`）：
#: dsh 每个请求都单发一条 `Current runtime context. This snapshot supersedes…`
#: 的 user 消息（沙盒策略/审批模式快照，475–487 字符，**485 次**），排在提问之后。
#: 它剥不掉 ⇒ 「往前找剥完非空的 user」会**停在它身上**（实测 `query_units=50`、
#: 召回 `ldg-5cc65426473e 0.62`，仍是噪声命中）——修一层只把错误往前挪一格。
#:
#: 🔴 判据 = **整条以它开头**（`lstrip` 后 `startswith`），不是 `in`：
#: 本仓的文档与结算稿会逐字引用这句话，用 `in` 就是 MQ-A19 自指陷阱的第 N 次复发。
_MACHINE_ONLY_PREFIXES: tuple[tuple[str, str], ...] = (
    ("dsh_runtime_context", "current runtime context. this snapshot supersedes"),
)


def _clean_user_text(raw: str) -> tuple[str, list[str]]:
    """一条 user 消息 → `(剥完的正文, marks)`。Goal 侧与召回侧的**唯一**清洗口径。

    三道，顺序固定：① `envelope.strip_envelopes`（生产表，不动它）；
    ② 整条即机器运行时上下文 ⇒ 判空（`_MACHINE_ONLY_PREFIXES`）；
    ③ 裸前缀句 / 通用包装标签解包（`_BARE_PREFIX_RES` / `_WRAPPER_TAG_RE`）。

    返回空串 = 「这条里没有用户说的话」，调用方据此往前找。
    """
    from bladex_core.envelope import strip_envelopes
    cleaned, marks = strip_envelopes(raw)
    head = cleaned.lstrip().lower()
    for kind, marker in _MACHINE_ONLY_PREFIXES:
        if head.startswith(marker):
            return "", [*marks, kind]
    for pat in _BARE_PREFIX_RES:
        stripped, n = pat.subn("", cleaned, count=1)
        if n:
            cleaned = stripped
            marks = [*marks, f"bare_prefix:{pat.pattern[:24]}"]
    m = _WRAPPER_TAG_RE.match(cleaned.strip())
    if m:
        cleaned = m.group("inner")
        marks = [*marks, f"wrapper:{m.group('tag')}"]
    return cleaned.strip(), marks


def _user_texts(messages: list[dict]) -> list[str]:
    """按原序取出所有 user 消息正文（多模态 list content 拼 text 块）。"""
    out: list[str] = []
    for m in messages or []:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        out.append(" ".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
                   if isinstance(c, list) else str(c or ""))
    return out


def _last_user_text(messages: list[dict], *, for_goal: bool = False) -> str:
    """末条 user 正文。`for_goal=True` 时取**最后一条剥完非空的 user** 并截断。

    🔴 2026-08-25 live 病例：CC 的 user 消息带 `<session>…</session>` 包装 +
    后续机器指令，原样取进 Goal ⇒ Goal 成了机器文本，"用户可核对"当场失效
    （ADR-0032 §4.3 goal「取」不是「炼」的前提是取到的是**用户说的那句**）。
    envelope 剥离是 V-R1 实证过的 CC 信封形态，现成复用（零 LLM、确定性）。

    🔴 2026-09-20 MQ-L78（执行卡 G19.0）：上面那段只看**最后一条** user，剥空了就
    `cleaned or raw` 回退原文。dsh 把注入上下文**以 user 角色单独发一条**排在提问
    之后（`core/session/src/surface.ts` 注释原文 "injected context project in user
    role"）⇒ 剥完为空 ⇒ 回退 ⇒ 拿 1446 字符的 AGENTS.md 当"用户这句话"去做账本召回。
    live 病例：目标账本相关性 0.80 排第一，实际召回到两本无关账本，模型 13 轮 / 21 次
    `bladex_memory_search` 才自己撞见 id（`docs/reviews/dsh-ledger-unreachable-20260920.md`）。

    修法 = **往前找**：最后一条剥完非空的 user 才是用户说的那句；全剥空才退回现状
    并打 `agency_user_text_all_envelopes`。**通用规则，不按 agent 名特判**（刚性原则 10）。

    仪器：`agency_goal_envelope_stripped` 带 `back=` —— 往前走了几条。
    `back>0` 就是本缺陷的**直接命中计数**（修前它恒为 0，因为根本不走）；
    旧的 `after==0` 读数不再产生，等价读数改看 `agency_user_text_all_envelopes`。
    """
    texts = _user_texts(messages)
    if not texts:
        return ""
    if not for_goal:
        return texts[-1]

    def _cap(s: str) -> str:
        return (s[:_GOAL_MAX_CHARS] + "…[bladex-goal-truncated]"
                if len(s) > _GOAL_MAX_CHARS else s)

    skipped: list[str] = []
    for back, raw in enumerate(reversed(texts)):
        try:
            cleaned, marks = _clean_user_text(raw)
        except Exception as e:  # noqa: BLE001 —— 剥离失败退回原文，不阻断建本
            logger.warning("agency_goal_envelope_failed", error=str(e))
            return _cap(raw.strip())
        if cleaned:
            if marks or back:
                logger.info("agency_goal_envelope_stripped", marks=marks[:4],
                            before=len(raw), after=len(cleaned),
                            back=back, skipped=skipped[:4])
            return _cap(cleaned)
        skipped.extend(marks or ["(无 marks)"])
    # 全部 user 消息剥完都是空 ⇒ 退回现状（末条原文），但**留可 grep 的告警**：
    # 这条日志的存在本身就是"我们取到的不是用户说的话"的自认（刚性原则 13 同族）。
    logger.warning("agency_user_text_all_envelopes", users=len(texts),
                   before=len(texts[-1]), skipped=skipped[:6])
    return _cap(texts[-1].strip())
