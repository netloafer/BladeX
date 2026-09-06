"""③ 漂移检测 —— 把"升级即失联"从静默变成可发现（2026-08-26，接入域）。

# 要解决什么

agent 升级导致识别信号漂移是**必然事件，不是意外**。实测：Codex 0.147 → 0.149
把 system prompt 措辞与工具集**同时**改了，两条内容维一起失效 ⇒ 落
`unknown-9d9bdd3e`，12 轮、10 条 fact 写进幻影命名空间，而告警只说了一句
泛泛的"未识别 agent"——**看不出这是老朋友换了皮**。

# 🔴 两件事必须分开，判据正好相反

| | 版本升级失联 | 子代理 |
|---|---|---|
| 现象 | 老 agent_id **停**，新桶**起** | 老 agent_id 与新桶**并存** |
| 处置 | **合并**（同一个东西换了皮） | **命名**（不同的东西，各自有身份） |
| 判据 | **接替** | **并存** |

搞反了两边都错：合并子代理是**误合并**，给升级后的自己起新名字是**身份分裂**。
所以 `succession` 这条证据不是锦上添花，它是**区分器**。

# 输出证据，不输出结论

三条证据齐了也只是"疑似"。BladeX 不替用户判定"这就是 codex"——那是认领动作的
语义，且误合并=0 是红线，"看起来像"永远不足以自动合并。本模块只把证据摆出来、
把认领动作和规则预填好，**按不按由人决定**。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DriftEvidence:
    """一个未识别桶"疑似某已知 agent 升级"的证据集。"""

    looks_like: str = ""                 # 疑似的已知 agent_id（空 = 无线索）
    #: 传输重叠：桶的 basis 里的 ua/vendor token 与该 agent 的 base 名相同
    transport_overlap: str = ""
    #: 时序接替：该 agent 有历史记录、但在本桶出现之后**没有再出现**
    succession: bool = False
    #: 会话接续：本桶的首轮消息尾部能接上该 agent 的某个会话
    session_continuation: bool = False

    @property
    def score(self) -> int:
        return (bool(self.transport_overlap) + bool(self.succession)
                + bool(self.session_continuation))

    def as_dict(self) -> dict:
        return {"looks_like": self.looks_like,
                "transport_overlap": self.transport_overlap,
                "succession": self.succession,
                "session_continuation": self.session_continuation,
                "score": self.score}


#: basis 里**可作为身份线索**的段前缀。
#:
#: 🔴 `tools:` 与 `none` 不在内：前者是工具集哈希（内容派生、正是升级会变的那一维），
#: 后者是"什么信号都没有"。拿它们找线索等于用最脆的信号做最重的判断。
_IDENTITY_SEGMENTS = ("ua:", "vendor:")


def _basis_tokens(basis: str) -> list[str]:
    out: list[str] = []
    for seg in (basis or "").split("|"):
        for prefix in _IDENTITY_SEGMENTS:
            if seg.startswith(prefix):
                tok = seg[len(prefix):].strip().lower()
                if tok:
                    out.append(tok)
    return out


def transport_overlap(basis: str, known_agent_ids: list[str]) -> str:
    """basis 的 ua/vendor token 与某个已知 agent 的 base 名对得上 ⇒ 返回那个 agent_id。

    匹配用**前缀**而非相等：Codex 的 vendor 段实测是 `codex-parent`
    （`x-codex-parent-thread-id` 被贪婪正则拆出来的），与 base 名 `codex` 不相等
    但显然同源。反过来 `codex` 也要能匹配 token `codex`。
    """
    for agent_id in known_agent_ids:
        base = (agent_id or "").split(":")[0].strip().lower()
        if not base or base.startswith("unknown"):
            continue
        for tok in _basis_tokens(basis):
            if tok == base or tok.startswith(base + "-") or base.startswith(tok + "-"):
                return agent_id
    return ""


def detect(
    *,
    basis: str,
    known_agent_ids: list[str],
    last_seen: dict[str, float] | None = None,
    bucket_first_seen: float = 0.0,
    session_match: str = "",
) -> DriftEvidence:
    """汇总三条证据。**纯函数**——调用方喂读数，本模块不碰任何存储。

    :param last_seen: `agent_id -> 最近一次被识别的时间戳`（注册表读数）。
    :param bucket_first_seen: 本桶首次出现的时间戳。
    :param session_match: 尾部接续命中的 agent_id（`_session_cache.find_match` 的产出，
        空 = 没接上）。**这是三条里唯一的内容级证据**，不依赖任何 header——
        升级常发生在会话中间：同一个对话，前半截 codex、后半截 unknown。
    """
    ev = DriftEvidence()
    ev.transport_overlap = transport_overlap(basis, known_agent_ids)
    candidate = ev.transport_overlap or session_match
    if not candidate:
        return ev
    ev.looks_like = candidate

    # 时序接替 vs 并存 —— 区分"升级"与"子代理"的那条。
    seen = (last_seen or {}).get(candidate)
    if seen is not None and bucket_first_seen > 0:
        # 老 agent 最后一次出现**不晚于**本桶首次出现 ⇒ 接替（它之后就没再说话）。
        # 反之（老 agent 在本桶出现之后还在跑）⇒ 并存 ⇒ 多半是子代理，不是升级。
        ev.succession = seen <= bucket_first_seen

    ev.session_continuation = bool(session_match and session_match == candidate)
    return ev


@dataclass
class ClaimSuggestion:
    """④ 给 dashboard 的一键认领建议。规则预填，用户改完才写盘。"""

    bucket_id: str
    to_agent_id: str
    merge: bool                          # 目标已存在 ⇒ 认领即合并，要二次确认
    evidence: dict = field(default_factory=dict)
    rule: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"bucket_id": self.bucket_id, "to_agent_id": self.to_agent_id,
                "merge": self.merge, "evidence": self.evidence, "rule": self.rule}


def suggest_claim(bucket_id: str, ev: DriftEvidence,
                  headers: dict[str, str] | None = None) -> ClaimSuggestion | None:
    """证据够强时给出认领建议；否则 None（让它继续以 unknown 待着）。

    门槛 **≥2 条证据**：单条太弱——只有传输重叠时，"另一个用同款 SDK 的新客户端"
    与"老朋友升级"长得一模一样，而误合并=0 是红线。
    """
    if not ev.looks_like or ev.score < 2:
        return None
    rule: dict = {"name": ev.looks_like, "agent_id": ev.looks_like}
    if headers:
        from bladex_proxy.agent_rules import suggest_rule
        rule = suggest_rule(ev.looks_like, headers=headers)
    return ClaimSuggestion(bucket_id=bucket_id, to_agent_id=ev.looks_like,
                           merge=True, evidence=ev.as_dict(), rule=rule)
