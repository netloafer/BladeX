"""AgencyRuntime 的文件型输入：账本模板 / 系统自我介绍 / AGENTS 名册，以及末条 user 文本取值。

09-06 F0.1 自 `agency.py` 拆出，零行为（函数体逐字搬家）。门面 `bladex_proxy.agency` re-export。
"""

from __future__ import annotations

import re

import structlog

logger = structlog.get_logger()


#: 「这段 user 文本像是机器写的」的形状——**只用于告警，不用于任何判定**。
#: 2026-08-27 Codex 病例的两条实测形态各贡献一条；`You are …` 那条覆盖面最广
#: （客户端把子任务的 system prompt 塞进 user 消息，是各家通用做法）。
_MACHINE_TEXT_MARKS: tuple[tuple[str, str], ...] = (
    ("assistant_persona", "you are a "),      # "You are a helpful assistant…"
    ("md_overview", "# overview"),
    ("agent_persona", "you are an agent"),
)


def _machine_text_mark(text: str) -> str:
    """user 文本疑似机器模板 ⇒ 返回形态名；否则 ""。**只在开头 200 字符内看**。

    位置门与 MQ-A19 同理：正文里**提到** "you are a helpful assistant"
    的真实用户消息不该被标记（这正是那次自指陷阱的教训）。
    """
    head = (text or "")[:200].lower()
    for name, mark in _MACHINE_TEXT_MARKS:
        if mark in head:
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


def load_agents_roster(max_rows: int = 15) -> str:
    """读 flash 树的 AGENTS.md 名册（V-F3；缺失 = 返回 ""，整段不注）。

    来源 = flash daemon 物化的机器文件（registry 缓存的确定性字段）。
    个人模式 principal 固定 "local"；企业形态的 per-principal 名册注入
    归后续（稳定层是进程级冻结，与 per-request principal 天然不合）。
    `max_rows` 封顶（ADR-0032 §3.1「一行一个封顶」）——名册是注意力预算的
    一部分，agent 多到溢出时裁最旧的（表按 first_seen 升序，尾部最新，保尾）。
    """
    import os as _os

    from bladex_core.flash_tree import agents_roster_path
    from bladex_proxy.config import resolve_flash_path
    path = agents_roster_path(resolve_flash_path(), "local")
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        return ""
    lines = [ln for ln in text.splitlines() if "<!--" not in ln]
    header, rows = [], []
    for ln in lines:
        (rows if ln.startswith("|") else header).append(ln)
    if len(rows) > 2 + max_rows:          # 表头 2 行 + 数据行封顶，保尾（最新）
        rows = rows[:2] + rows[-max_rows:]
    out = "\n".join([*header, *rows]).strip()
    return out if rows else ""


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


def _last_user_text(messages: list[dict], *, for_goal: bool = False) -> str:
    """末条 user 正文。`for_goal=True` 时**剥 agent 信封**并截断。

    🔴 2026-08-25 live 病例：CC 的 user 消息带 `<session>…</session>` 包装 +
    后续机器指令，原样取进 Goal ⇒ Goal 成了机器文本，"用户可核对"当场失效
    （ADR-0032 §4.3 goal「取」不是「炼」的前提是取到的是**用户说的那句**）。
    envelope 剥离是 V-R1 实证过的 CC 信封形态，现成复用（零 LLM、确定性）。
    """
    raw = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            raw = (" ".join(str(b.get("text", "")) for b in c if isinstance(b, dict))
                   if isinstance(c, list) else str(c or ""))
            break
    if not for_goal or not raw:
        return raw
    try:
        from bladex_core.envelope import strip_envelopes
        cleaned, marks = strip_envelopes(raw)
        # 🔴 `strip_envelopes` 只认已知信封种类（它被蒸馏消费，改它要 bump 台账版本
        # 并影响重建等价 ⇒ 不为 Goal 动它）。Goal 侧再加一条**通用包装标签解包**：
        # 整条即 `<tag>…</tag>`（可后接指令）时取标签内容——CC 的 `<session>` 属此形态。
        for pat in _BARE_PREFIX_RES:
            stripped, n = pat.subn("", cleaned, count=1)
            if n:
                cleaned = stripped
                marks = [*marks, f"bare_prefix:{pat.pattern[:24]}"]
        m = _WRAPPER_TAG_RE.match(cleaned.strip())
        if m:
            cleaned = m.group("inner")
            marks = [*marks, f"wrapper:{m.group('tag')}"]
        if marks:
            logger.info("agency_goal_envelope_stripped", marks=marks[:4],
                        before=len(raw), after=len(cleaned))
        raw = cleaned.strip() or raw.strip()
    except Exception as e:  # noqa: BLE001 —— 剥离失败退回原文，不阻断建本
        logger.warning("agency_goal_envelope_failed", error=str(e))
    if len(raw) > _GOAL_MAX_CHARS:
        raw = raw[:_GOAL_MAX_CHARS] + "…[bladex-goal-truncated]"
    return raw
