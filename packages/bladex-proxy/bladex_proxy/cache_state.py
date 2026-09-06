"""Prompt Cache TTL 感知的会话状态（ADR-0028 §2 / E4）。

## 为什么需要它

装配层（ADR-0019）与相关性裁剪（本卡 E4.3）都在**改写送给上游的 messages 前缀**。
改写前缀 = 上游 prompt cache 全部作废。所以"该不该改"取决于一件此前从未被问过的事：

    **这个会话的上游 cache 现在是温热的，还是已经过期了？**

- **温热**（同一模型 + 距上次转发 < 该模型 cache_ttl_s + agent 没自己压缩过）：
  全量转发，复用上一轮冻结的降解计划，**禁止**新增任何裁剪/降解——
  打碎前缀省的那点 token 远不够 cache 命中的损失。
- **冷**（模型换了 / TTL 过期 / 首轮）：cache 反正要重建，此时裁剪只赚不赔，
  ADR-0019 装配正常跑 + E4.3 相关性裁剪。

## 与会话粘性的关系

粘性表（`MemoryAwareRouter._sticky`）记的是 session→model，本模块记的是
session→(model, 上次转发时间, 冻结的降解计划, 单元向量缓存)。两者互补：
粘性让模型**别乱换**，本模块决定在模型没换的前提下**能不能改前缀**。

判定发生在装配时（路由之前），所以用的是"本会话上一次实际转发的模型"——
在粘性生效的前提下它就是本轮将要用的模型。若本轮路由真的换了模型，
`note_forward()` 会发现并丢弃冻结计划，下一轮自然按冷处理。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

# 会话状态表容量（LRU；与粘性表同量级）
_MAX_SESSIONS = 500


@dataclass
class SessionCacheState:
    """单会话的 cache 状态（ADR-0028 §2 数据结构）。"""

    model: str = ""
    last_forward_ts: float = 0.0
    # 单元 intent 向量缓存：unit_key -> vector。
    # unit_key 内容派生、跨轮稳定，故每轮通常只新增 1 个单元需要 embed。
    unit_vec_cache: dict[str, list[float]] = field(default_factory=dict)
    # 温热期冻结的装配/裁剪计划（原样复用，保证前缀逐字节一致）。
    frozen_degrade_plan: dict[str, Any] = field(default_factory=dict)


class SessionCacheRegistry:
    """会话 cache 状态注册表（进程内，LRU cap 500）。"""

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._states: OrderedDict[str, SessionCacheState] = OrderedDict()
        self._max = max_sessions

    def get(self, session_id: str) -> SessionCacheState | None:
        if not session_id:
            return None
        st = self._states.get(session_id)
        if st is not None:
            self._states.move_to_end(session_id)
        return st

    def ensure(self, session_id: str) -> SessionCacheState:
        st = self.get(session_id)
        if st is None:
            st = SessionCacheState()
            self._states[session_id] = st
            while len(self._states) > self._max:
                self._states.popitem(last=False)
        return st

    def is_warm(
        self,
        session_id: str,
        model: str,
        cache_ttl_s: int,
        *,
        prefix_changed: bool = False,
        now: float | None = None,
    ) -> bool:
        """ADR-0028 §2 判定：

            warm = (state.model == model)
                   and (now - last_forward_ts < cache_ttl_s)
                   and not prefix_changed

        `cache_ttl_s <= 0` → 该模型无 cache，恒冷。
        没有会话状态（首轮 / 进程刚起）→ 冷。
        `prefix_changed`（agent 自己压缩过历史，Memory Hub O(1) 缓存查）→ cache 本就已失效 → 冷。
        """
        if not session_id or not model or cache_ttl_s <= 0 or prefix_changed:
            return False
        st = self.get(session_id)
        if st is None or st.model != model or st.last_forward_ts <= 0:
            return False
        return (time.time() if now is None else now) - st.last_forward_ts < cache_ttl_s

    def note_forward(self, session_id: str, model: str, now: float | None = None) -> None:
        """记录"本会话刚往 model 转发了一轮"。

        模型变了 → 上一轮那份冻结计划对新模型的 cache 毫无意义，一并丢弃
        （下一轮自然走冷启路径重新规划）。
        """
        if not session_id or not model:
            return
        st = self.ensure(session_id)
        if st.model and st.model != model:
            st.frozen_degrade_plan = {}
            logger.info("cache_state_model_switched", session_id=session_id[:16],
                        old_model=st.model, new_model=model)
        st.model = model
        st.last_forward_ts = time.time() if now is None else now

    def freeze_plan(self, session_id: str, plan: dict[str, Any]) -> None:
        """冷启路径跑完后冻结本轮计划，供温热期原样复用。"""
        if not session_id:
            return
        self.ensure(session_id).frozen_degrade_plan = dict(plan)

    def frozen_plan(self, session_id: str) -> dict[str, Any]:
        st = self.get(session_id)
        return dict(st.frozen_degrade_plan) if st is not None else {}

    def clear(self) -> None:
        self._states.clear()


# 进程内单例（proxy 三端点共用；测试可 `registry.clear()`）
_REGISTRY = SessionCacheRegistry()


def registry() -> SessionCacheRegistry:
    return _REGISTRY
