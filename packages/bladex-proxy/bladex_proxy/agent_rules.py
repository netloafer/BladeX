"""agent 识别规则样本库的加载与合并（执行卡 G11.10，台账接入域 MQ-A1/A6）。

规则此前是 `identity.py` 里的一段 Python 字面量：加一个 agent 要改代码、发版本，
用户无从参与。外置之后 = 预置随包发布 + 用户在部署根覆盖，配合 dashboard 认领
（G11.11）形成闭环：**新 agent 认领一次就永久认识**。

## 两份来源、一个顺序

======================  ==================================================
预置                    ``bladex_proxy/agent_rules.toml``（随包，升级会覆盖）
用户                    ``<部署根>/config/agent_rules.toml``（可选）
======================  ==================================================

用户规则**前置**（先匹配）。依据是 CLAUDE.md 刚性原则 9——配置文件里的静态策略
压过一切推断；预置表本质上是我们替用户做的推断，用户显式写的必须能盖过它。

## 🔴 加载是惰性的，import 无副作用

2026-08-04 踩过：consolidator 把 `_load_env_file()` 留在模块级，**任何 import 都会
把 live 配置灌进 `os.environ`**，实测污染同进程后续全部 `test_server` 用例。
本模块只在第一次真正需要规则时读盘，并且**只读、不碰环境变量**。

## 为什么不是纯 header 正则

Hermes 的 chat 主路径走 OpenAI SDK，UA 是 SDK 默认值（它自己的 ``hermes-cli/<ver>``
只用于 ``/v1/models`` 与 pricing catalog）。纯 header 规则永远匹配不上它。
所以规则保持多信号：system prompt 关键词 / tool 签名 / header 模式，任一命中即可。
header 维是给"确实自报家门且形态稳定"的 agent 的捷径，不是主力——理由见
:mod:`bladex_proxy.agent_bucket` 里 UA 不可作身份权威来源的四条实证。
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

__all__ = [
    "AgentFingerprintRule",
    "HeaderPattern",
    "PRESET_RULES_PATH",
    "USER_RULES_RELPATH",
    "append_user_rule",
    "load_agent_rules",
    "parse_rules",
    "effective_rules_for",
    "merge_with_preset",
    "render_rule_toml",
    "rule_as_dict",
    "suggest_rule",
    "reset_rules_cache",
]

#: 预置规则表（随包发布）。
PRESET_RULES_PATH = Path(__file__).with_name("agent_rules.toml")

#: 用户规则表相对部署根的位置。
USER_RULES_RELPATH = "config/agent_rules.toml"


@dataclass(frozen=True)
class HeaderPattern:
    """一条 header 维判据。

    :param name: header 名（小写比较）。
    :param pattern: 对 header **值**的正则；空串表示"只要这个 header 存在即命中"。
        对 ``x-<vendor>-session-id`` 这类**值每轮都变**的 header，必须留空——
        拿它的值做判定等于每轮换一次身份。
    """

    name: str
    pattern: str = ""

    def matches(self, headers: dict[str, str]) -> bool:
        value = headers.get(self.name.lower())
        if value is None:
            return False
        if not self.pattern:
            return True
        return re.search(self.pattern, value) is not None


@dataclass
class AgentFingerprintRule:
    """一条 agent 识别规则：system prompt 关键词 + tool 签名 + header 模式。

    三类信号**任一命中**即可识别 base agent；同时命中置信度更高。

    profile 区分：同一 agent 不同配置文件（如 Hermes accept vs default）从 agent
    自带的 profile 声明标记提取（如 ``Active Hermes profile: accept``）
    → ``agent_id = "hermes:accept"``，不 hash（ADR-0010 §3.3）。

    设计原则（CLAUDE.md 刚性原则 10）：agent 不需要主动配 header，proxy 从请求内容
    被动识别。规则可由用户在 ``config/agent_rules.toml`` 增补或覆盖。
    """

    agent_id: str
    #: 规则名（诊断用；用户表里同名规则视为覆盖预置的同名规则）
    name: str = ""
    system_prompt_keywords: list[str] = field(default_factory=list)
    tool_signatures: list[str] = field(default_factory=list)
    min_tool_match: int = 1
    header_patterns: list[HeaderPattern] = field(default_factory=list)
    #: ② 子代理声明 header（2026-08-26）：agent 自己说"这一轮是我的某个子代理"。
    #: 命中则 `agent_id = f"{agent_id}:{header 值}"` 并置 `Identity.subagent`。
    #: 实测来源：Codex 的 `x-openai-subagent: guardian`（12 轮全带）。
    #: 与 `profile_aware` 同一套命名设施（`hermes:accept` / `codex:guardian`），
    #: 只是信号从 system prompt 标记换成 header —— 子代理往往没有 system prompt。
    subagent_headers: list[str] = field(default_factory=list)
    #: 是否对该 agent 做 profile 级区分
    profile_aware: bool = False
    #: 是否为 agent 内部辅助调用（跳蒸馏；是否降路由档另见 DISTILL_ONLY_AUX_RULES）
    auxiliary: bool = False
    #: 规则来源，"preset" / "user"（诊断用）
    source: str = "preset"


def _rule_from_dict(raw: dict[str, Any], source: str) -> AgentFingerprintRule:
    """把一段 TOML 表转成规则；字段缺失走默认值，多余字段显式报错。

    **不静默忽略未知字段**：打错一个键名而规则悄悄不生效，是"配置写了没用"这类
    最难查的问题（CLAUDE.md：误配置必须响亮失败）。
    """
    known = {
        "agent_id", "name", "system_prompt_keywords", "tool_signatures",
        "min_tool_match", "header_patterns", "subagent_headers",
        "profile_aware", "auxiliary",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"agent 规则 {raw.get('name') or raw.get('agent_id')!r} 含未知字段 "
            f"{sorted(unknown)}；可用字段：{sorted(known)}"
        )
    agent_id = str(raw.get("agent_id", "")).strip()
    if not agent_id:
        raise ValueError(f"agent 规则缺少 agent_id：{raw!r}")

    headers: list[HeaderPattern] = []
    for hp in raw.get("header_patterns", []) or []:
        if not isinstance(hp, dict) or "name" not in hp:
            raise ValueError(f"{agent_id} 的 header_patterns 条目需要 name 字段：{hp!r}")
        pattern = str(hp.get("pattern", ""))
        if pattern:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(
                    f"{agent_id} 的 header_patterns 正则非法（{hp['name']}）：{exc}"
                ) from exc
        headers.append(HeaderPattern(name=str(hp["name"]).lower(), pattern=pattern))

    return AgentFingerprintRule(
        agent_id=agent_id,
        name=str(raw.get("name", "")) or agent_id,
        system_prompt_keywords=[str(k) for k in raw.get("system_prompt_keywords", [])],
        tool_signatures=[str(t) for t in raw.get("tool_signatures", [])],
        min_tool_match=int(raw.get("min_tool_match", 1)),
        header_patterns=headers,
        subagent_headers=[str(h).strip().lower()
                          for h in raw.get("subagent_headers", []) if str(h).strip()],
        profile_aware=bool(raw.get("profile_aware", False)),
        auxiliary=bool(raw.get("auxiliary", False)),
        source=source,
    )


def parse_rules(text: str, source: str = "preset") -> list[AgentFingerprintRule]:
    """解析一份规则 TOML 文本。纯函数，便于测试与 dashboard 校验用户输入。"""
    data = tomllib.loads(text)
    return [_rule_from_dict(raw, source) for raw in data.get("rules", [])]


_cache: list[AgentFingerprintRule] | None = None


def reset_rules_cache() -> None:
    """清空进程内缓存（测试与 dashboard 写入规则后调用）。"""
    global _cache
    _cache = None


def _generic_ua_rules(rules: list[AgentFingerprintRule]) -> list[tuple[str, str, str]]:
    """挑出「靠通用 SDK/运行时名认 agent」的 UA 规则。返回 `(agent, token, pattern)`。

    判据：规则里 `user-agent` 维的模式，能匹配上 `<token>/1.0` 这种最小 UA，
    而 `token` 在 :data:`GENERIC_CLIENT_TOKENS` 里。

    用**真的跑一遍正则**而不是对模式串做文本匹配——模式的写法千变万化
    （`^codex\\b` / `codex` / `(?:async)?openai`），只有让它对着真实形态跑，
    判据才不依赖某一种写法。纯函数，可单测。
    """
    from bladex_proxy.agent_bucket import GENERIC_CLIENT_TOKENS

    out: list[tuple[str, str, str]] = []
    for r in rules:
        for hp in r.header_patterns:
            if hp.name != "user-agent" or not hp.pattern:
                continue
            try:
                rx = re.compile(hp.pattern)
            except re.error:
                continue
            for token in GENERIC_CLIENT_TOKENS:
                if rx.search(f"{token}/1.0 (probe)"):
                    out.append((r.agent_id, token, hp.pattern))
                    break
    return out


def load_agent_rules(user_path: str | Path | None = None) -> list[AgentFingerprintRule]:
    """加载规则：用户规则（前置）+ 预置规则。惰性、带进程内缓存。

    :param user_path: 显式指定用户规则文件；``None`` 时按部署根发现。
        传了显式路径就**不走缓存**（测试与 dashboard 预览需要每次真读）。
    :returns: 按匹配优先级排好的规则列表。

    用户文件缺失是正常情况（多数部署没有），不报错；**存在但解析失败则抛出**——
    静默忽略一份写错的规则文件，用户会一直以为自己的规则生效了。
    """
    global _cache
    explicit = user_path is not None
    if _cache is not None and not explicit:
        return _cache

    preset = parse_rules(PRESET_RULES_PATH.read_text(encoding="utf-8"), source="preset")

    path = Path(user_path) if explicit else _discover_user_rules_path()
    user: list[AgentFingerprintRule] = []
    if path is not None and path.is_file():
        user = parse_rules(path.read_text(encoding="utf-8"), source="user")
        logger.info("agent_rules_user_loaded", path=str(path), count=len(user))

    # 用户表内**同名取最后一条**（append 即 update）。
    #
    # 🔴 这条语义是 dashboard 能编辑规则的前提：写盘走追加（不重写文件、不吃掉用户
    # 自己的注释），那么"改一条规则"就是再追加一条同名的。若保留第一条，用户改完
    # 会发现**改了没用**——旧的那条仍排在前面命中，是最难查的一类问题。
    deduped: dict[str, AgentFingerprintRule] = {}
    for r in user:
        deduped[r.name] = r          # 后写的覆盖先写的
    user_effective = list(deduped.values())

    # 用户表与预置同名同理视为**覆盖**而非叠加。
    overridden = set(deduped)

    # 🔴 覆盖导致**识别面缩小**时必须看得见（2026-08-26 事故）。
    #
    # 病例：dashboard 认领 codex 时写了一条 `name="codex"` 的用户规则，本意是
    # "补一条 header 规则"，实际效果是**用一条 UA 规则替换掉 codex 的全部识别面**
    # —— 预置的 2 条 system prompt 关键词、3 个工具签名、2 个 header 维、
    # 以及 `subagent_headers` 全部消失。后果不只是识别变脆：`subagent_headers` 没了，
    # `codex:guardian` 这层命名当场失效，②那整层白做。
    #
    # 最难受的是**它是静默的**：dashboard 上 codex 仍显示 "rule"、一切正常，
    # 只有把规则 dump 出来才看得见识别面缩了。
    #
    # 覆盖语义本身不改（刚性原则 9：用户显式写的必须能盖过我们替他做的推断）——
    # 但"盖掉了什么"要说出来。
    _by_name = {r.name: r for r in preset}
    for name, u in deduped.items():
        p = _by_name.get(name)
        if p is None:
            continue
        lost = [dim for dim, un, pn in (
            ("system_prompt_keywords", len(u.system_prompt_keywords),
             len(p.system_prompt_keywords)),
            ("tool_signatures", len(u.tool_signatures), len(p.tool_signatures)),
            ("header_patterns", len(u.header_patterns), len(p.header_patterns)),
            ("subagent_headers", len(u.subagent_headers), len(p.subagent_headers)),
        ) if pn > 0 and un == 0]
        if lost:
            logger.warning(
                "agent_rule_override_narrows_recognition",
                agent=name, lost_dimensions=lost,
                hint="a user rule with the same name REPLACES the preset (not merged); "
                     "these preset dimensions are now inactive. Copy them into the user "
                     "rule, or delete the user rule to fall back to the preset.",
            )

    # 🔴 对称检查：**过宽的 UA 规则**（2026-08-26，同日第二次同型事故）。
    #
    # 上面那条管"识别面缩小"，这条管"识别面过宽"——两个方向都会坏事，而且都静默。
    #
    # 病例（两次，同一形状）：`suggest_rule` 的缺陷让 dashboard 认领写出了一条
    # 匹配**通用 SDK 名**的 UA 规则（`(?i)^asyncopenai\b` 写进 hermes ⇒
    # "任何用 async-openai 的客户端都算 hermes"）。代码侧修好了，但认领**把当时
    # 进程内的代码版本固化成了磁盘上的配置** —— `gate_check` 只跑测试不重启服务，
    # 用户很自然会在 gate_check 通过后立刻操作，于是旧产物落了盘。
    #
    # 所以检查放在**加载时**：无论那条规则当初是怎么写进去的（认领生成、手工编辑、
    # 从别处拷来），下次启动都会喊出来。这是"凭 SDK 名认 agent"这个已被否掉的
    # 判据的最后一道门 —— 它被否的理由是：SDK 名标识**客户端库**，不标识 agent。
    _ua_generic = _generic_ua_rules(user_effective)
    for agent, token, pattern in _ua_generic:
        logger.warning(
            "agent_rule_ua_matches_generic_sdk",
            agent=agent, sdk_token=token, pattern=pattern,
            hint="this user rule identifies an agent by a generic SDK/runtime name; "
                 "any client using that library will be attributed to this agent. "
                 "SDK names identify the HTTP client, not the agent - remove this "
                 "header_pattern (other dimensions of the rule stay in effect).",
        )

    merged = user_effective + [r for r in preset if r.name not in overridden]

    if not explicit:
        _cache = merged
    return merged


#: 建议 system prompt 关键词时，优先挑这些形态的行（agent 的自我介绍）。
_SELF_INTRO_RE = re.compile(
    r"^\s*(?:#+\s*)?(you are\b.*|i am\b.*|.*\bpowered by\b.*)$", re.IGNORECASE
)

#: 建议关键词的长度带。太短没区分度（会误伤别的 agent），太长易随版本漂移。
_SUGGEST_KEYWORD_MIN = 12
_SUGGEST_KEYWORD_MAX = 60

#: 建议 tool 签名最多取几个。
_SUGGEST_MAX_TOOLS = 5


def rule_as_dict(rule: AgentFingerprintRule) -> dict[str, Any]:
    """把生效中的规则转回可编辑 / 可写盘的 dict（dashboard 预填用）。"""
    out: dict[str, Any] = {"name": rule.name, "agent_id": rule.agent_id}
    if rule.system_prompt_keywords:
        out["system_prompt_keywords"] = list(rule.system_prompt_keywords)
    if rule.tool_signatures:
        out["tool_signatures"] = list(rule.tool_signatures)
        out["min_tool_match"] = rule.min_tool_match
    if rule.subagent_headers:
        out["subagent_headers"] = list(rule.subagent_headers)
    if rule.header_patterns:
        out["header_patterns"] = [
            {"name": hp.name, "pattern": hp.pattern} for hp in rule.header_patterns
        ]
    if rule.profile_aware:
        out["profile_aware"] = True
    if rule.auxiliary:
        out["auxiliary"] = True
    return out


def effective_rules_for(agent_id: str,
                        rules: list[AgentFingerprintRule] | None = None,
                        ) -> list[AgentFingerprintRule]:
    """某个 agent_id 当前生效的规则（可能不止一条，如 hermes 的 aux + base）。"""
    rules = rules if rules is not None else load_agent_rules()
    return [r for r in rules if r.agent_id == agent_id]


def suggest_rule(
    agent_id: str,
    system_excerpt: str = "",
    tools: list[str] | None = None,
    headers: dict[str, str] | None = None,
    existing: list[AgentFingerprintRule] | None = None,
) -> dict[str, Any]:
    """从**已经抓到的会话内容**推一条候选识别规则，供 dashboard 预填。

    存在理由（Jason 2026-08-19）：认领对话里的关键词是选填的，不填就不会写规则，
    那个 agent 下次照样认不出来。而让用户自己手写关键词也不合理——**system prompt、
    工具集、header 我们全都抓到了**，系统本来就该给出建议，用户只需确认或改。

    产出是**建议**不是决定：dashboard 预填进可编辑的输入框，用户改完才写盘。

    :param existing: 现有规则表，用于**避开别家已经占用的工具名**。这一条是硬要求
        ——`session_search` 同时存在于 Hermes 与 deepseek-harness，当年就是它把 dsh
        误判成 hermes（台账 MQ-A6）。建议器绝不能再制造同型冲突。
    :returns: 可直接喂给 :func:`append_user_rule` 的 dict（可能只含部分维度）。
    """
    existing = existing if existing is not None else load_agent_rules()
    taken_tools = {t for r in existing for t in r.tool_signatures}
    taken_keywords = [k.lower() for r in existing for k in r.system_prompt_keywords]

    # agent_id 允许为空（未识别桶还没有名字，等用户填）；其余维度照常建议。
    rule: dict[str, Any] = {"name": agent_id, "agent_id": agent_id}

    # ① system prompt 自我介绍行 —— 辨识度最高
    keyword = ""
    for line in (system_excerpt or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _SELF_INTRO_RE.match(line)
        candidate = (m.group(1) if m else line).strip().rstrip(".。!！,，:：")
        if len(candidate) < _SUGGEST_KEYWORD_MIN:
            continue
        if len(candidate) > _SUGGEST_KEYWORD_MAX:
            # 在词边界截，别切出 "software engineer" 这种半截词（中文无空格 → 硬截）
            head = candidate[:_SUGGEST_KEYWORD_MAX]
            cut = head.rfind(" ")
            candidate = (head[:cut] if cut >= _SUGGEST_KEYWORD_MIN else head)
        candidate = candidate.strip().rstrip(".。!！,，:：")
        # 与别家关键词互为子串 = 会互相误命中，跳过
        low = candidate.lower()
        if any(low in k or k in low for k in taken_keywords):
            continue
        keyword = candidate
        if m:            # 命中自我介绍形态就不再往下找
            break
    if keyword:
        rule["system_prompt_keywords"] = [keyword]

    # ② 工具名 —— 先剔掉别家已占用的（MQ-A6 不得复发）
    own = sorted({t for t in (tools or []) if t and t not in taken_tools})

    # 🔴 再按**形状**判区分度。实测（2026-08-19 Pi）：它的工具是
    # `bash` / `edit` / `read` / `write` —— 没有任何规则占用，但全是通用单词。
    # 拿它们配 `min_tool_match=1`，等于"任何带 bash 的 agent 都是 Pi"，
    # 正是 `session_search` 当年把 dsh 误判成 hermes 的形状。
    #
    # 判据用形状而非名单：带命名空间分隔符的（`cordis_run` / `skill-manage`）
    # 是生态专名，裸单词是通用叫法。名单要维护、会漏；形状不用。
    distinctive = [t for t in own if ("_" in t or "-" in t or ":" in t or "." in t)]
    generic = [t for t in own if t not in distinctive]

    if distinctive:
        rule["tool_signatures"] = distinctive[:_SUGGEST_MAX_TOOLS]
        rule["min_tool_match"] = 1
    elif generic and not keyword:
        # 只有通用名可用、又没有 system prompt 关键词 —— 工具是唯一信号了。
        # 那就**要求同时命中多个**：单个 `bash` 谁都有，一整套才有点意思。
        picked = generic[:_SUGGEST_MAX_TOOLS]
        rule["tool_signatures"] = picked
        rule["min_tool_match"] = min(len(picked), 2)
    # 有强关键词时，通用工具名整个不进建议：一条强信号胜过三条弱信号，
    # 加进去只会扩大误命中面（规则是"任一命中即算"）。

    # ③ UA —— 只在它确实是 agent 自己的名字时才建议（SDK 名不算）
    ua = (headers or {}).get("user-agent", "")
    from bladex_proxy.agent_bucket import GENERIC_CLIENT_TOKENS, parse_ua_product

    product = parse_ua_product(ua)
    if product and product not in GENERIC_CLIENT_TOKENS:
        # 🔴 **不能硬拼斜杠**（2026-08-26 实测）：`parse_ua_product` 归一后是产品名的
        # 首 token，而真实 UA 的产品名可能带空格——
        #   Codex Desktop/0.149.0-alpha.4.3 (…)   → product='codex'
        #   `(?i)^codex/` 匹配 **False**（codex 后面是空格不是斜杠）
        # 单词产品名（`claude-cli/2.1.241`、`deepseek-harness/0.1.0`）恰好成立，
        # 所以这个模板一直没露馅——Codex 是我们接的五个 agent 里唯一的多词产品名。
        #
        # 生成一条**建议出来就该能用**的规则：只锚开头的 token 边界，
        # 后面是 `/`（单词名）还是空格（多词名）都放行。
        rule["header_patterns"] = [
            {"name": "user-agent", "pattern": f"(?i)^{re.escape(product)}\\b"}
        ]
        # 自证：建议器产出的规则必须匹配它自己采样的那条 UA。匹配不上说明模板
        # 与真实形态脱节——**报出来，不要静默给用户一条永远不命中的规则**
        # （用户会以为"下次自动认出来"已经成立，而它不会）。
        try:
            if not re.search(rule["header_patterns"][0]["pattern"], ua):
                logger.warning("suggest_rule_ua_pattern_misses_sample",
                               product=product, ua=ua[:120],
                               pattern=rule["header_patterns"][0]["pattern"])
                rule.pop("header_patterns")
        except re.error:
            rule.pop("header_patterns", None)

    return rule


def _toml_str(value: str) -> str:
    """把一个字符串渲染成安全的 TOML 标量。

    优先**字面量字符串**（`'…'`，不处理任何转义）——见 `render_rule_toml` 的立论：
    正则里的反斜杠在基本字符串里会被吃掉或炸掉。含单引号或控制字符时才退回基本
    字符串，并把反斜杠与双引号转义。
    """
    if "'" not in value and "\n" not in value and "\r" not in value:
        return f"'{value}'"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", "\\n").replace("\r", "\\r")
    return f'"{escaped}"'


def render_rule_toml(rule: dict[str, Any]) -> str:
    """把一条规则渲染成可追加进用户规则文件的 TOML 片段。

    手写渲染而不是引第三方 TOML writer：只有六个字段、且要**保留注释与人类可读的
    排版**（这个文件是给人改的）。渲染前先过 `_rule_from_dict` 校验，非法直接抛。

    🔴 **正则用 TOML 字面量字符串（单引号），不用基本字符串**（2026-08-26 事故）。

    基本字符串会处理转义，而正则里出现反斜杠是**常态不是例外**：

        原 `(?i)^codex\\b`  → 写成 `"…\\b"` → 读回 `(?i)^codex\\x08`（退格符！）
        原 `\\d+`           → 写成 `"\\d+"`  → **整份文件解析失败**

    第一种最难受：**静默**变成控制字符，规则从此永不命中，而界面上一切正常。
    live 实测就是这样——`(?i)^asyncopenai\\b` 落盘后其实是
    `(?i)^asyncopenai\\x08`，一个客户端都匹配不上。
    等于"dashboard 写不出带反斜杠的规则"，而那几乎是所有正则。

    字面量字符串（`'…'`）**不处理任何转义**，正是为正则准备的。
    代价是它不能包含单引号——正则里极少见，真出现就退回基本字符串并转义。
    """
    parsed = _rule_from_dict(rule, source="user")  # 校验；非法在写盘前就失败
    lines = ["", "[[rules]]", f'name = "{parsed.name}"', f'agent_id = "{parsed.agent_id}"']
    if parsed.system_prompt_keywords:
        kws = ", ".join(_toml_str(k) for k in parsed.system_prompt_keywords)
        lines.append(f"system_prompt_keywords = [{kws}]")
    if parsed.tool_signatures:
        ts = ", ".join(_toml_str(t) for t in parsed.tool_signatures)
        lines.append(f"tool_signatures = [{ts}]")
        lines.append(f"min_tool_match = {parsed.min_tool_match}")
    if parsed.header_patterns:
        items = ", ".join(
            f"{{ name = {_toml_str(hp.name)}, pattern = {_toml_str(hp.pattern)} }}"
            for hp in parsed.header_patterns
        )
        lines.append(f"header_patterns = [{items}]")
    if parsed.subagent_headers:
        subs = ", ".join(_toml_str(h) for h in parsed.subagent_headers)
        lines.append(f"subagent_headers = [{subs}]")
    if parsed.profile_aware:
        lines.append("profile_aware = true")
    if parsed.auxiliary:
        lines.append("auxiliary = true")
    return "\n".join(lines) + "\n"


def merge_with_preset(rule: dict[str, Any]) -> dict[str, Any]:
    """把一条待写入的规则与**同名预置规则**并起来（A，2026-08-26）。

    🔴 立论：**认领的意图是"补一条"，不是"重定义这个 agent"。**

    事故：dashboard 认领 codex 时写了 `name="codex"` 的用户规则（只有一条 UA
    header 维），而加载侧的语义是"用户表同名 ⇒ **覆盖**预置"，于是 codex 的
    2 条关键词 / 3 个工具签名 / 2 个 header 维 / `subagent_headers` **全部失效**
    ——`codex:guardian` 那层命名当场没了，而界面上一切正常。

    **覆盖语义本身不改**（刚性原则 9：用户显式写的必须能盖过我们替他做的推断）。
    收窄的只有**认领这一条写入路径**：它是系统代用户生成的补充，不是用户在说
    "我要重定义 codex"。用户手工编辑那份文件时，覆盖照旧。

    合并规则：**逐维取并集**，用户给的排在前（他的判断优先），预置的补在后；
    标量（`min_tool_match` / `profile_aware` / `auxiliary`）用户给了就用他的。
    """
    preset = {r.name: r for r in parse_rules(
        PRESET_RULES_PATH.read_text(encoding="utf-8"), source="preset")}
    p = preset.get(str(rule.get("name") or rule.get("agent_id") or ""))
    if p is None:
        return dict(rule)
    out = dict(rule)

    def _union(key: str, preset_vals: list) -> None:
        cur = list(out.get(key) or [])
        seen = {repr(x) for x in cur}
        out[key] = cur + [x for x in preset_vals if repr(x) not in seen]

    _union("system_prompt_keywords", list(p.system_prompt_keywords))
    _union("tool_signatures", list(p.tool_signatures))
    _union("header_patterns",
           [{"name": h.name, "pattern": h.pattern} for h in p.header_patterns])
    _union("subagent_headers", list(p.subagent_headers))
    for k, v in (("min_tool_match", p.min_tool_match),
                 ("profile_aware", p.profile_aware),
                 ("auxiliary", p.auxiliary)):
        out.setdefault(k, v)
    # 空维不落盘（render 侧也会跳过），保持文件可读
    return {k: v for k, v in out.items() if v not in ([], None)}


def append_user_rule(rule: dict[str, Any], path: str | Path | None = None,
                     *, merge_preset: bool = True) -> Path:
    """把 dashboard 认领生成的规则追加进用户规则文件，并清缓存使其立即生效。

    **追加而不是重写整个文件**：那个文件是用户自己在维护的，重写会丢掉他的注释与
    排版。同名规则的"覆盖"语义由加载侧处理（用户表内后写的同名条目在前——见
    :func:`load_agent_rules`），这里不做去重。

    :param merge_preset: 默认 True —— 与同名预置规则**逐维取并集**再写盘
        （见 :func:`merge_with_preset` 的立论：认领是"补"不是"替"）。
        手工路径若确实想覆盖预置，传 False。
    :returns: 实际写入的路径。
    :raises ValueError: 规则非法（在写盘**之前**校验，不留半个坏文件）。
    """
    target = Path(path) if path is not None else _discover_user_rules_path()
    if target is None:
        raise RuntimeError("找不到部署根，无法写用户规则文件；请显式传 path")
    if merge_preset:
        rule = merge_with_preset(rule)
    snippet = render_rule_toml(rule)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(
            "# BladeX 用户 agent 识别规则（由 dashboard 认领写入，也可手工编辑）\n"
            "# 模板与字段说明见 config/agent_rules.toml.example\n",
            encoding="utf-8",
        )
    with target.open("a", encoding="utf-8") as fh:
        fh.write(snippet)
    reset_rules_cache()
    logger.info("agent_rule_appended", path=str(target), agent_id=rule.get("agent_id"))
    return target


def _discover_user_rules_path() -> Path | None:
    """按部署根发现用户规则文件（复用 deployment 的唯一实现，不另起一套）。

    🔴 `BLADEX_AGENT_RULES_PATH` 可显式覆盖。**测试隔离靠它**——否则
    `load_agent_rules()` 会去读**真实部署**的 `config/agent_rules.toml`，
    测试结果随用户当天存了什么规则而变。实测炸过（2026-08-19）：Jason 在 live 存了
    一条 `Pi` 规则，三个用例当场变红——两个因为 `Pi` 突然"已存在"、一个因为建议的
    关键词与那条真规则冲突被滤掉。与 conftest 里 `BLADEX_ROCKSDB_PATH` 等路径隔离
    同一个理由：**测试不许读写 live 的任何东西**。
    """
    import os

    override = os.environ.get("BLADEX_AGENT_RULES_PATH", "")
    if override:
        return Path(override)
    try:
        from bladex_proxy.deployment import deployment_root

        root = deployment_root()
    except Exception as exc:  # pragma: no cover - 部署根发现失败不该拖垮识别
        logger.debug("agent_rules_root_discovery_failed", error=str(exc))
        return None
    return Path(root) / USER_RULES_RELPATH
