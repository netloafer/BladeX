"""接入体检（G17.22 / MQ-A69）：新 agent 接进来时，**三项能力是否真的就位**。

## 为什么要有这个模块

2026-09-11 一天之内在同一条路径上撞见三个**各自独立、表征完全相同**的缺陷：

| 号 | 静默失效点 | 表征 | 谁发现的 |
|---|---|---|---|
| A66 | 过期常量把 dsh 排除在工具面外 17 天 | 账本零活动（像"模型不调用"，已知 3%） | Jason 问了一句 |
| A67 | cwd 模式表只覆盖 6 个在产 agent 中的 2 个 | `project_id` 恒 0%（像"这个 agent 没项目"） | Jason 问了一句 |
| A68 | 规则库没有 opencode | `agent_id=unknown-…`（`fallback` 在日志里不刺眼） | Jason 让它连一次 |

**三次都是人发现的，机制一次都没发现。** 每一条单独看都有日志、都能事后查出来 ——
问题是**没有任何一行主动说"这个 agent 少了点什么"**，而三种失效的表征
都长得像"这个 agent 本来就这样"。

⇒ 本模块不发明新判据，只做一件事：**把已经存在的三个信号合成一行，
在新 agent 第一次真正跑起来时主动说一次。**

## 判据：三项，每项对着上面一条

1. **身份** —— `agent_source == "fallback"` 或 `agent_id` 以 `unknown-` 开头。
   A68 的形态：兜底分桶其实已经从 UA 里读出了名字，规则库里却没有它
   （"看得见却认不出"）。
2. **项目** —— 🔴 **判据是 `cwd_found`，不是 `project_id`**。
   这条区分 `project_identity.resolve_from_request` 的 docstring 早在 2026-08-29
   就写下了：「没抽到标记 ⇒ 改 pattern；抽到了不成项目 ⇒ 该 cwd 本就不是项目」。
   ⚠️ **本模块存在的直接原因之一，就是我 09-11 无视了这条区分**：把四个 agent 的
   `project_id` 0% 全当成缺陷，其中 hermes 两 profile 其实是"抽到了、是家目录、
   按三档正确判 Global"。**用 `project_id` 当判据会把正确行为报成缺口**，
   那种告警会被很快训练成噪声、然后被忽略——比没有告警更糟。
3. **工具面** —— `toolface_injected` 为假。A66 的形态。

## 报一次，不是每轮报

按 **agent_id / 进程** 去重：高流量 agent 每轮报一行就是噪声，噪声等于没有告警。
进程重启会重报 —— 这是**特意的**：重启后配置可能变了，那正是该重新体检的时刻。

**aux 轮整轮跳过**：子调用（标题生成 / 压缩）本来就不给工具面，
拿它体检会把正常行为记成缺口，而且有些 agent 的第一轮恰好是 aux（codex 实测）。
"跳过"意味着不记 seen —— 下一个正式轮仍会体检。
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger()

#: 进程内已体检过的 agent。上限防未认领指纹桶把它撑爆（`unknown-<sha>` 每个都不同）；
#: 撑满后停止体检而不是逐出——逐出会让同一个 agent 反复重报，那就成了噪声。
_MAX_SEEN = 200
_seen: set[str] = set()


def reset_seen() -> None:
    """仅供测试：清空进程内去重集合。"""
    _seen.clear()


def onboarding_gaps(*, agent_source: str, agent_id: str, cwd_found: bool,
                    toolface_injected: bool) -> list[str]:
    """三项能力里没就位的那些（纯函数，单测钉死）。

    返回稳定顺序的缺口名：`identity` / `project` / `toolface`。空列表 = 三项齐全。
    """
    gaps: list[str] = []
    # 🔴 调用方传的是 `AgentSource`（str 混入 Enum）：`str(AgentSource.FALLBACK)` 在 3.11+ 是
    # "AgentSource.FALLBACK" 而不是 "fallback"——live 日志 09-12 实证 `identity_source=AgentSource.FINGERPRINT`，
    # 与 `identity_resolved` 那行的 `agent_source=fingerprint` 同名不同形（批 J 顺手发现）。
    # 这里按 `.value` 归一，判据与日志字段都用短名。
    agent_source = str(getattr(agent_source, "value", agent_source))
    if agent_source == "fallback" or agent_id.startswith("unknown-"):
        gaps.append("identity")
    if not cwd_found:
        gaps.append("project")
    if not toolface_injected:
        gaps.append("toolface")
    return gaps


def _dedup_key(agent_id: str) -> str:
    """去重键的**唯一定义点**。

    G17.22 要把它改成 `(agent_id, project_id)`；届时只改这一处——调用方
    （`server/orchestration.py`）不许再复述这个判据（批 H 复核指出的形态：
    外面复制了一份 `agent_id not in _seen`，里面改了键、外面那句会静默挡住新行为）。
    """
    return agent_id


def pending(*, agent_id: str, auxiliary: bool) -> bool:
    """这个 agent 本进程是否**还需要**体检（不改状态，纯查询）。

    调用方用它决定要不要为体检付出 `extract_cwd` 那次重算——
    它与 `note_onboarding` 内部用的是同一个键、同一条 aux 规则、同一个上限。
    """
    if auxiliary or not agent_id:
        return False
    return _dedup_key(agent_id) not in _seen and len(_seen) < _MAX_SEEN


def note_onboarding(*, agent_id: str, agent_source: str, auxiliary: bool,
                    cwd_found: bool, project_id: str, project_source: str,
                    toolface_injected: bool) -> bool:
    """本进程首次见到这个 agent 时打一行接入体检；已报过则什么都不做。

    :returns: 这次是否真的报了（供测试与调用方判断，生产侧不依赖）。
    """
    if not pending(agent_id=agent_id, auxiliary=auxiliary):
        # aux 轮不体检、也不记 seen（见模块 docstring）；已报过 / 上限满同样跳过。
        return False
    _seen.add(_dedup_key(agent_id))
    agent_source = str(getattr(agent_source, "value", agent_source))  # 与 identity_resolved 同形（短名）
    gaps = onboarding_gaps(agent_source=agent_source, agent_id=agent_id,
                           cwd_found=cwd_found, toolface_injected=toolface_injected)
    emit = logger.warning if gaps else logger.info
    emit(
        "agent_onboarding_check",
        agent=agent_id,
        gaps=",".join(gaps),
        # 三项各自的原始读数一并落下：告警只说"哪项没就位"，
        # **修法要看原始值**（`identity_source=fallback` 改规则库；
        # `cwd_found=False` 改 pattern；`toolface=False` 查排除名单与档位）。
        identity_source=agent_source,
        cwd_found=cwd_found,
        # `project_id` 空而 `cwd_found` 真 = 抽到了但不成项目（家目录），
        # **不是缺口**——两个字段一起落，是为了让这句判断可以被复算。
        project_id=project_id,
        project_source=project_source or "global",
        toolface=toolface_injected,
    )
    return True
