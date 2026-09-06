"""条目重要性评分（ADR-0026 §5.4 机制4 / U5.4）。

取代恒 1.0 的 `trust`。公式形态已拍板（ADR-0026 §10.2）：

    importance = kind 基线 × 强度增益 × 引用增益 × 时间半衰减

- kind 基线：不同 item_kind 的先天重要性（hard_rule/preference 高、profile_obs 低）。
- 强度增益：strength（合并计数）越大越重要（对数，边际递减）。
- 引用增益：ref_count（注入命中回写，消费者=U7 注入侧接线）越多越重要（对数）。
- 时间半衰减：按 t_observed 年龄半衰期衰减（新记忆略重，旧记忆自然降权）。

参数留真实流量校准（守「验证驱动精度」）：默认值面向合理起点，`未标定` 标记见
`BLADEX_IMPORTANCE_CALIBRATED`。env 可覆盖（回滚/调参通道）。

消费者：检索排序（U6）+ 生命周期驱逐（U5.5）+ dashboard。纯函数，agent 中立。
"""

from __future__ import annotations

import math
import os

# kind 基线（ADR-0026 §4.2 六类 + hard_rule）。留校准。
_KIND_BASELINE: dict[str, float] = {
    "hard_rule": 1.00,   # MUST/NEVER，最高（虽然硬规则单独注入，排序里也置顶）
    "preference": 0.90,  # 用户偏好，跨会话高价值
    "procedure": 0.80,   # 怎么做有效
    "lesson": 0.80,      # 为什么错 + 怎么改（根因，高价值）
    "assertion": 0.70,   # 一般结论
    "file_ref": 0.60,    # 文件指针
    "profile_obs": 0.30,  # 画像原料，不直接注入
    "general": 0.60,     # 未分类兜底
}

_DEFAULT_BASELINE = 0.60


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def compute_importance(
    item_kind: str,
    *,
    strength: int = 1,
    ref_count: int = 0,
    age_seconds: float = 0.0,
    half_life_seconds: float | None = None,
    rating: int = 0,
) -> float:
    """算一条条目的 importance。

    - item_kind：六类之一（或 hard_rule/general），决定基线。
    - strength：合并计数（≥1）。
    - ref_count：注入命中回写次数（U7 接线；默认 0）。
    - age_seconds：t_observed 至今秒数（默认 0 = 刚产生）。
    - half_life_seconds：时间半衰期（默认取 env `BLADEX_IMPORTANCE_HALFLIFE_S` 或 30 天）。
    - rating：**内容内在分** 1–10（M1-5，蒸馏 prompt v4 产出）。>0 时取代 kind 基线。

    返回 [0, ∞) 的分数（通常 0.1–2.0），检索按此降序 + 其它信号融合。

    ## M1-5：为什么要 rating

    kind 基线只能表达"偏好类比一般结论重要"，表达不了**同类之间**的差别——
    「决定把存储换成 X」和「顺带提到今天天气」都是 assertion，基线一模一样 0.70。
    G4 说的正是这件事：kind 基线全盘继承了模块 1 的分类噪声，且无法表达真伪/时效。
    rating 让蒸馏器按内容判一次（Generative Agents 式，搭同一次调用零额外成本）。

    rating==0（模型没给分 / 给不出合法值 / 历史数据）→ 退回 kind 基线，
    即 v3 行为逐字不变。**不是"默认 5 分"**——那会把"未评分"伪装成"中等重要"，
    历史数据会被整体拉平到同一个分。
    """
    if rating and 1 <= rating <= 10:
        base = rating / 10.0
    else:
        base = _KIND_BASELINE.get((item_kind or "").strip().lower(), _DEFAULT_BASELINE)

    # 强度增益：1 + w_s * ln(strength)，strength=1 时增益 1.0
    w_s = _env_float("BLADEX_IMPORTANCE_W_STRENGTH", 0.30)
    strength_gain = 1.0 + w_s * math.log(max(1, strength))

    # 引用增益：1 + w_r * ln(1 + ref_count)
    w_r = _env_float("BLADEX_IMPORTANCE_W_REFCOUNT", 0.25)
    ref_gain = 1.0 + w_r * math.log(1 + max(0, ref_count))

    # 时间半衰减：0.5 ** (age / half_life)，age=0 时 1.0
    hl = half_life_seconds if half_life_seconds is not None else _env_float(
        "BLADEX_IMPORTANCE_HALFLIFE_S", 30 * 24 * 3600.0
    )
    decay = 0.5 ** (max(0.0, age_seconds) / hl) if hl > 0 else 1.0

    return base * strength_gain * ref_gain * decay


def is_calibrated() -> bool:
    """importance 参数是否已按真实流量标定（否则调用方应告警未标定）。"""
    return os.environ.get("BLADEX_IMPORTANCE_CALIBRATED", "false").lower() in ("true", "1", "yes")
