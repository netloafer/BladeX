"""敏感度数据流原语（ADR-0021 §3）- agent 中立，stdlib logging。

三件事的纯逻辑底座（不含 toml 解析、不含模型调用）：

1. **暴露等级三级标签**（§3.3a）：`local` < `private` < `public`，对应企业三类算力
   （自建 vLLM/Ollama < 租赁裸显卡 < 公网 API）。模型 `[[models]].exposure` 与
   Fact `exposure_ceiling` 都用这套枚举；未标 exposure 按 `public`（最坏暴露假设，
   敏感流量自动避开）。
2. **多维敏感标签取最严**（§3.1）：agent / principal / team 三维命中取最严
   （allowed_exposure 序最低者 = 最严），fail-closed 到底（未知等级 -> 最严）。
3. **比较原语**：
   - `exposure_within(model_exposure, allowed_exposure)`：路由硬边界--候选模型 exposure
     不得超过请求 allowed_exposure（T4 选池最前裁剪）。
   - `exposure_allows(fact_ceiling, allowed_exposure)`：注入守卫--Fact.exposure_ceiling
     必须 >= 请求 allowed_exposure 才能注入（T5）。

敏感度解析是身份的**纯静态函数**（§3.3 核心洞察）：身份解析完成时三维全已知，
查表 O(1)，不需内容分类（内容级动态分类留 E2）。
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── 暴露等级（§3.3a）─────────────────────────────────────────────────────────

EXPOSURE_LOCAL = "local"      # 本地/私有网络内推理（自建 vLLM/Ollama）
EXPOSURE_PRIVATE = "private"  # 受控外部算力（租赁裸显卡 endpoint）
EXPOSURE_PUBLIC = "public"    # 公网 API（云厂商）

# 序：值越小越严（local 最严 = 暴露面最小）。比较一律走 rank。
_EXPOSURE_ORDER: dict[str, int] = {
    EXPOSURE_LOCAL: 0,
    EXPOSURE_PRIVATE: 1,
    EXPOSURE_PUBLIC: 2,
}

# 默认敏感等级 -> 允许暴露上限映射（个人模式 = 全 public，零回归）。
_DEFAULT_LEVELS: dict[str, str] = {"normal": EXPOSURE_PUBLIC}

DEFAULT_SENSITIVITY_LEVEL = "normal"


def exposure_rank(exposure: str) -> int:
    """暴露等级 -> 序号（越小越严）。

    未知值按 `public`（最坏暴露假设，§3.3a）：模型未标 exposure 视为公网暴露，
    敏感流量自动避开；Fact.exposure_ceiling 缺失视为 public（现状等价）。
    """
    return _EXPOSURE_ORDER.get(exposure, _EXPOSURE_ORDER[EXPOSURE_PUBLIC])


def exposure_within(model_exposure: str, allowed_exposure: str) -> bool:
    """路由硬边界（T4）：候选模型 exposure 不得超过请求 allowed_exposure。

    allowed_exposure 是请求允许到达的最高暴露等级（由敏感度解析产出）。
    模型 exposure 序 <= allowed_exposure 序 -> 在允许池内（不越界）。
    例：allowed=local(0)，模型 public(2) -> 2<=0 False（公网模型不服务敏感请求）。
    """
    return exposure_rank(model_exposure) <= exposure_rank(allowed_exposure)


def exposure_allows(fact_ceiling: str, allowed_exposure: str) -> bool:
    """注入守卫（T5）：Fact.exposure_ceiling 是否允许注入 allowed_exposure 的请求。

    exposure_ceiling 是该 Fact 允许到达的最高暴露等级（来源 Turn 敏感等级映射）。
    请求 allowed_exposure 是其目的地模型的最高暴露等级。
    ceiling 序 >= allowed 序 -> 不越界（Fact 能去的地方包含请求目的地）。
    例：ceiling=local(0)，allowed=public(2) -> 0>=2 False（本地 Fact 不注入公网请求）。
        ceiling=public(2)，allowed=local(0) -> 2>=0 True（公网 Fact 可注入本地请求）。
    """
    return exposure_rank(fact_ceiling) >= exposure_rank(allowed_exposure)


def strictest_exposure(levels: dict[str, str]) -> str:
    """一组 level->exposure 映射里最严的暴露等级（序最低者）。

    fail-closed 兜底：未知等级 / 解析异常 -> 最严。levels 为空 -> local（最严）。
    """
    if not levels:
        return EXPOSURE_LOCAL
    return min(levels.values(), key=exposure_rank)


# ── 敏感度配置 + 解析（§3.1 / §3.3b）─────────────────────────────────────────


class IdentitySensitivity(Protocol):
    """身份侧敏感度只读视图（T2 产出，T3 消费）。

    解耦 identity_registry 与 sensitivity：core 只依赖此 Protocol，
    proxy 侧 identity_registry 提供实现。个人模式（无 identity.toml）返回全 None。
    """

    @property
    def principal_level(self) -> str | None:
        """principal 的敏感等级（未标注 = None -> 用 default_level）。"""
        ...

    @property
    def team_levels(self) -> list[str]:
        """principal 所属各 team（含祖先链）的敏感等级（未标注不含）。"""
        ...


class SensitivityConfig(BaseModel):
    """敏感度策略数据（agent 中立；proxy 侧 routing_config 从 toml 产）。

    - enabled=False（默认）= 个人模式：resolve 恒返 (normal, public)，注入/路由逐字不变。
    - agents: agent base/id -> 敏感等级（如财务 agent 整体敏感）。
    - default_level: 未标注主体的默认等级（敏感部署建议配最严）。
    - levels: 敏感等级 -> 允许暴露上限（如 sensitive=local / internal=private / normal=public）。
    """

    enabled: bool = False
    agents: dict[str, str] = Field(default_factory=dict)
    default_level: str = DEFAULT_SENSITIVITY_LEVEL
    levels: dict[str, str] = Field(default_factory=lambda: dict(_DEFAULT_LEVELS))

    model_config = {"extra": "ignore"}

    def resolve(
        self,
        agent_id: str | None,
        identity: IdentitySensitivity | None,
    ) -> tuple[str, str]:
        """优先级 team > principal > agent -> (level, allowed_exposure)。

        关闭态恒返 (normal, public)（零回归）。开启态按层级覆盖：team 有标注
        （多 team 取最严）则 team 决定，否则 principal，否则 agent，否则 default_level。
        未知等级 -> 最严暴露（strictest_exposure）。
        """
        if not self.enabled:
            return DEFAULT_SENSITIVITY_LEVEL, self.levels.get(
                DEFAULT_SENSITIVITY_LEVEL, EXPOSURE_PUBLIC,
            )

        # ADR-0021 §3.1 修订：优先级 team > principal > agent（层级覆盖，非 max）。
        # team 维度（多个 team）内部取最严（fail-closed 在组织维度内保留）；
        # team 有标注则 team 决定，否则 principal，否则 agent，否则 default。
        # 注：层级覆盖下 team=normal 会覆盖 principal=sensitive（不取全局最严），
        # 这是组织优先级语义；team 未标注时才下放到 principal/agent。
        if identity is not None and identity.team_levels:
            return self._strictest_of(identity.team_levels)
        if identity is not None and identity.principal_level:
            lvl = identity.principal_level
            return lvl, self._level_exposure(lvl)
        if agent_id:
            lvl = self.agents.get(agent_id)
            if lvl is None and ":" in agent_id:
                lvl = self.agents.get(agent_id.split(":", 1)[0])
            if lvl:
                return lvl, self._level_exposure(lvl)
        return self.default_level, self._level_exposure(self.default_level)

    def _strictest_of(self, levels: list[str]) -> tuple[str, str]:
        """多等级取最严（exposure rank 最低）。team 维度多 team 合并用。"""
        best_lvl = levels[0]
        best_exp = self._level_exposure(best_lvl)
        for lvl in levels[1:]:
            exp = self._level_exposure(lvl)
            if exposure_rank(exp) < exposure_rank(best_exp):
                best_lvl, best_exp = lvl, exp
        return best_lvl, best_exp

    def _level_exposure(self, level: str) -> str:
        """等级 -> 允许暴露上限；未知等级 fail-closed 取最严。"""
        if level in self.levels:
            return self.levels[level]
        logger.warning(
            "sensitivity_unknown_level level=%s -> strictest fail-closed", level,
        )
        return strictest_exposure(self.levels)
