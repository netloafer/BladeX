"""ADR-0027 §5.4 / T36：记忆机制开关的默认值与配置模板对账。

背景（这组测试防的是什么）：ADR-0024/0025/0026 的五个机制 2026-08-03 完成、08-04 经
真实流量验收（重复注入 0、阴性误注 0），但**默认值仍是关、且五个开关名不在
config/.env.example 里**——只出现在一份 planning 文档的 export 段。也就是说验收是在
临时 shell 里跑出来的，仓库默认形态仍是旧注入。发出去的会是未修版本。

所以这里钉三件事：
1. 五个开关默认开（`flags.MEMORY_FLAG_DEFAULTS`）；
2. 三个消费点（inject / memory_index / prefetch）真的读这张表，不再是就地 `== "1"`；
3. `.env.example` 与这张表一致——任一侧改了另一侧不改就红。

第 3 条是本组的重点：机制与配置模板脱节是一类会重复发生的腐化，靠人记得改不行。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from bladex_core.flags import (
    MEMORY_FLAG_DEFAULTS,
    MEMORY_NUMERIC_DEFAULTS,
    flag_enabled,
    flag_number,
)

_ENV_EXAMPLE = Path(__file__).resolve().parents[3] / "config" / ".env.example"


def _repo_env_example() -> Path:
    assert _ENV_EXAMPLE.is_file(), f"找不到 {_ENV_EXAMPLE}（仓库结构变了？）"
    return _ENV_EXAMPLE


# ── 1. 默认值本身 ────────────────────────────────────────────────────────────


def test_all_memory_flags_and_defaults():
    """开关表整表钉死（第一批 ADR-0027 §5.4 五项 + 第二批 ADR-0028 §4 四项）。

    KEYWORD_CHANNEL 已删除（三段式卡 T3c，2026-08-10）：旧 substring 通道
    随 E6.2 删除后 flag 留成僵尸零消费方，`bladex status` 还把它打成 off
    误导观感——本表不再含它即是回归断言。
    """
    assert MEMORY_FLAG_DEFAULTS == {
        # 第一批（INJECT_INDEX_RECALL / INJECT_PLANES 随 2026-09-03 S1 删除；
        # 下面三个消费点不在注入路径——MemoryIndex.search 与写侧画像捕获——故保留）
        "BLADEX_SOFT_SCORING": True,
        "BLADEX_HOP_EXPAND": True,
        "BLADEX_PROFILE_CARDS": True,
        # MQ-S27 硬分流（2026-08-20 拍板）：lane:profile/tool 不进 Matter 归属。
        # 默认开：是排除不是合并，最坏=漏挂可补，零误合并风险（方向论证见 flags.py）。
        "BLADEX_LANE_HARD_SPLIT": True,
        # G12.2 D1（ADR-0031 §3）：默认延续 + 活跃窗口。
        # ✅ 2026-08-21 翻默认 False → True：误延续尺前后对照
        # （member-coherence pre-v11 vs post-20260821，同参数同索引）——
        # 新出现的 CONT 层判 116 条 **BOTH = 0.0%**（全表最干净，次好 L4 0.5%）、
        # R2 长链污染仍为「无」、L2/L3/C 画像三项逐字零回归；live 21 小时
        # 79 条层判决，两个失败计数在全部 20 个 consolidator 日志里均为 0。
        # 红线 2 要求的是"未验证机制默认关"，举证已履行。判据边界见 flags.py。
        "BLADEX_TASK_STATE_D1": True,
        # GM-1/GM-2（2026-08-18，MQ-S11/S13）：延续信号 + 按键反查进**候选池**。
        # 🔴 默认开，与（已删的）ATTRIB_L0 相反——两者举证责任方向不同：
        # L0 是**确定性合并**（成员跟着代表走、跳过 LLM），未验证就开 = 误合并直接落库；
        # 这两项只往候选池加卡、判定仍由 L4 做，最坏情况是多送一次裁决。
        # 反方向的教训同样在案（ADR-0027 §5.4：默认关且不进模板 = 仓库默认形态从未变过）。
        "BLADEX_ATTRIB_CONTINUATION": True,
        "BLADEX_ATTRIB_KEY_RECALL": True,
        # 第二批（ADR-0028 E2.1）
        "BLADEX_SUPERSEDE_ENABLED": True,
        "BLADEX_MATTER_LIFECYCLE": True,
        # E4：Prompt Cache TTL 感知 + 冷启相关性裁剪
        "BLADEX_CACHE_TTL_AWARE": True,
        # E7.1：文件内容索引
        "BLADEX_FILE_INDEX": True,
        # M1-2：蒸馏 v4 两级路由（关掉 = 回落 v3 三条独立流水线）
        "BLADEX_DISTILL_V4": True,
        # M2-2：写入时裁决（关掉 = 回落 supersede 精确键 + 冲突检测的旧形态）
        "BLADEX_ADJUDICATE_ENABLED": True,
        # 2026-08-31（MQ-S32 病 B 第五环）：全量重建重放取代判决。
        # **默认开**——恢复 ADR-0018 §4.1 明写声称过的行为，不是新功能。
        "BLADEX_SUPERSEDE_REPLAY_ENABLED": True,
        # V-I2（ADR-0032 批一）：实体键规范化。=0 退回 strip+lower 旧口径（A/B/止血）。
        "BLADEX_ENTITY_CANON": True,
        # V-L5：账本锚归属层。2026-09-02 翻默认开（4a/4b 口径签收；=0 是回滚通道）。
        "BLADEX_LEDGER_ANCHOR": True,
        # V-P3 硬点 8 播报：默认关。协议事件序列只能靠真实客户端证伪
        # （2026-08-27 第一版不合协议、Codex 一次请求即断），沙盒测不出来。
        "BLADEX_NARRATE_INTERCEPT": False,
        # MQ-L21：零自带工具不给账本面。默认开——只做排除、零误合并风险。
        "BLADEX_LEDGER_REQUIRE_TOOLS": True,
        # MQ-L32 / ADR-0032 §4.5.1：账本准入闸（DPL 落锚卡前过两道因果闸）。
        # 🔴 默认**开**，与紧邻两条默认关的相反——它修的是正确性不是加特性：
        # 关掉 = 回到 09-02 之前"内容相似压过账本边界"（把 ADR-0024 的开发过程
        # 吸进 08-28 才创建的「评估 v5 架构」卡）。红线 2「未验证机制默认关」
        # 针对的是新增判定；本闸只做**排除**，最坏=漏挂可补，零误合并风险
        # （与 LANE_HARD_SPLIT 同一方向的论证）。开关保留是为了当 A/B 仪器。
        "BLADEX_LEDGER_GATE": True,
    }
    assert "BLADEX_KEYWORD_CHANNEL" not in MEMORY_FLAG_DEFAULTS
    # 2026-09-03 S4：四个退役开关连代码一起删，表里不得再出现（僵尸回归断言）。
    for zombie in ("BLADEX_ITEM_BOUNDARY", "BLADEX_FILE_REF_ENABLED",
                   "BLADEX_CLARIFY_ENABLED", "BLADEX_ATTRIB_L0",
                   "BLADEX_FLASH_RENDER", "BLADEX_FLASH_MATTERS",
                   "BLADEX_INJECT_INDEX_RECALL", "BLADEX_INJECT_PLANES"):
        assert zombie not in MEMORY_FLAG_DEFAULTS


def test_numeric_defaults_table():
    """数值型参数同样集中默认值（ADR-0028 E2.1）。"""
    assert MEMORY_NUMERIC_DEFAULTS == {
        # Flash 保鲜四窗口（2026-08-29 Jason 拍板：贵在精；默认值暂定按真实
        # 流量再调，0=禁用；规则权威 docs/planning/flash-content-rules-20260829.md）
        # 账本并发标注窗口（2026-08-29「两 agent 同激活一本」批；0=关）
        "BLADEX_LEDGER_COAUTHOR_WINDOW_S": 1800,
        "BLADEX_FLASH_SESSIONS_PER_PROJECT": 10,
        "BLADEX_FLASH_PROJECTS_PER_AGENT": 10,
        "BLADEX_FLASH_PROJECT_ACTIVE_DAYS": 30,
        "BLADEX_FLASH_AGENT_RETIRE_DAYS": 90,
        # M0-8（复核 G3）：三时钟落地前不做重要性驱逐 —— 门槛回 0。
        "BLADEX_MIN_IMPORTANCE": 0.0,
        "BLADEX_LIFECYCLE_INTERVAL_S": 21600.0,
        # M0-5：bigram 影子表默认关（被评估集否掉，见 flags.py 注释）。
        "BLADEX_FTS_BIGRAM_LIMIT": 0.0,
        # （M0-6/T3-D 钉卡门三值随 2026-09-03 S1 删除。）
        # M2-2：novelty 降为"只拦完全重发"；0.90+ 邻居进裁决包；k=10（跨语言对在 top-8）。
        "BLADEX_NOVELTY_THRESHOLD": 0.98,
        "BLADEX_ADJUDICATE_MIN_SIM": 0.90,
        "BLADEX_ADJUDICATE_TOPK": 10.0,
        # 2026-08-08：一次裁决调用的候选上限。此前不分块，805 条一次调用
        # 被输出上限截断 -> 全量降级 ADD，整轮重建的裁决贡献为 0。
        "BLADEX_ADJUDICATE_CHUNK": 25.0,
        # 2026-08-31（MQ-S32 病 B）：裁决中断守卫。与蒸馏侧同型同默认值——
        # 裁决失败是降级不抛的（整块判 ADD），而 ADD 是合法判决，下游看不出。
        "BLADEX_ADJUDICATE_OUTAGE_ABORT_RATIO": 0.5,
        # 2026-08-31（MQ-A28/A29）：嵌入分块。整批调用让 IPC 必超时、期间零进度、
        # 且打不断；512 是保守起点，按 consolidate_embed_progress 的 rate 标定。
        "BLADEX_EMBED_BATCH": 512.0,
        # M3-2 三时钟（5.9 拍板 6：先立观测、真实数据标定后调 —— 起步值不是结论）
        "BLADEX_CLOCK_STALE_UPDATE_D": 14.0,
        "BLADEX_CLOCK_STALE_HIT_D": 30.0,
        "BLADEX_CLOCK_PROVISIONAL_MAX_D": 30.0,
        "BLADEX_CLOCK_MIN_COVERAGE": 0.5,
        # T3b（三段式卡）：融合 entity 通道热门实体衰减系数（mem0 起步值）。
        "BLADEX_ENTITY_DECAY_COEF": 0.001,
        # T3-A（检索侧收复卡，2026-08-15）：profile_obs 限额参与 + 排序折价。
        # T1 归因 11/18 错题时间锚困在 profile_obs（配额层整类丢弃=结构性不可达）。
        # GM-1：「无新主体」门槛。不承重（候选只送裁决）——探针扫 1→12 收益列全程
        # 高于风险列且不交叉，3 是那份 2×2 所用的值，换个数就对不上那份读数。
        "BLADEX_ATTRIB_CONT_NEW_SUBJECT_MAX": 3.0,
        # GM-2：按键反查的判别力门。🔴 分母是**库内可链接 Matter 数**，不是绝对张数
        # ——同一个"出现在 5 张卡上"在 20 张卡的库里是泛词、5000 张卡的库里是强标识符
        # （绝对阈值当门槛 = MQ-S1 在清查的设计模式）。0.05 [标定] 待真实读数复标。
        "BLADEX_ATTRIB_KEY_DF_RATIO": 0.05,
        "BLADEX_QUOTA_PROFILE_OBS": 2.0,
        "BLADEX_PROFILE_OBS_DISCOUNT": 0.85,
        # 同义折叠阈值。08-20 曾按读数提到 0.97（fold 吃掉 80.7% 的 e5 高位候选、
        # 0.92 埋在自己声明的噪声底 0.88–0.93 里）；08-21 共享库 A/B 终验
        # 0.92→72.0% / 0.97→68.0%，−4pp 不显著（McNemar p=0.727）但也无证据
        # 支持改动 ⇒ **回退 0.92**（无证据不改默认）。机制与 flag 全保留。
        "BLADEX_FOLD_COSINE": 0.92,
        # MQ-R1（2026-08-17）：索引碎片自维护阈值。由热路径预算推出
        # （800ms − 410ms 固定底 − 22ms ≈ 368ms 余量 ÷ 3.63ms/碎片 ≈ 100），
        # 所以 BLADEX_HOTPATH_BUDGET_MS 改了它要跟着重算。
        "BLADEX_INDEX_MAX_FRAGMENTS": 100.0,
        # MQ-S9 / G12.2-pre（2026-08-20 拍板 4h）：fp: 会话桶时间窗切分；0=关。
        "BLADEX_SESSION_FP_WINDOW_S": 14400.0,
        # 2026-08-22：写入侧蒸馏并发。收进本表的理由不是"它是记忆机制"（它不是），
        # 而是它犯的正是本组测试要防的那类腐化——同一个参数在两条调用路径上各带
        # 一个默认值（sync 4 / 守护进程 1），谁也不知道生产跑的是哪个。
        "BLADEX_DISTILL_CONCURRENCY": 4.0,
        # V-P3（agent-compat-survey D4 实测标定）：内循环墙钟与轮数硬顶。
        "BLADEX_INNER_LOOP_BUDGET_S": 120.0,
        "BLADEX_INNER_LOOP_MAX_ROUNDS": 6.0,
        # V-L2/L4：切换防抖与陈旧标记阈值。
        "BLADEX_LEDGER_SWITCH_DEBOUNCE": 3.0,
        "BLADEX_LEDGER_STALE_TURNS": 20.0,
        # V-P3：内循环 keepalive 间隔。收进本表的理由同 DISTILL_CONCURRENCY——
        # 它原本是 agency.py 里一个硬编码 `_KEEPALIVE_INTERVAL_S = 10.0`，
        # 而新接的两条协议路径差点各自再拍一个值（同一参数两个默认值，刚性原则 12）。
        "BLADEX_STREAM_KEEPALIVE_S": 10.0,
        "BLADEX_AGENT_REGISTRY_TTL_DAYS": 7.0,
        # V-P4 硬点 1：拼接条目老化轮数。0 = 不老化。
        # ADR-0032 §10-2 拍板"不预定默认值、真实流量标定"——这里登记的是
        # **旋钮的存在**，不是一个推荐值。它此前硬编码在
        # `SpliceLedger.__init__` 的默认参数里，想标定都没得转。
        "BLADEX_SPLICE_MAX_AGE_TURNS": 0,
        # V-C1 / MQ-S47：会话前缀候选槽数。1 = 退回旧单槽行为（回滚通道）。
        # 取 4 = 主会话 1 槽 + 小请求 1 槽 + 2 槽余量；理由与 live 读数在 flags.py。
        "BLADEX_PREFIX_SLOTS": 4.0,
    }


@pytest.mark.parametrize("name", sorted(MEMORY_FLAG_DEFAULTS))
def test_flag_enabled_without_env(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert flag_enabled(name) is MEMORY_FLAG_DEFAULTS[name]


@pytest.mark.parametrize("name", sorted(MEMORY_NUMERIC_DEFAULTS))
def test_flag_number_without_env(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    assert flag_number(name) == MEMORY_NUMERIC_DEFAULTS[name]


def test_flag_number_parses_env(monkeypatch):
    monkeypatch.setenv("BLADEX_MIN_IMPORTANCE", "0.45")
    assert flag_number("BLADEX_MIN_IMPORTANCE") == 0.45


def test_flag_number_bad_value_falls_back(monkeypatch):
    """配置写坏不该让热路径崩——回落默认值。"""
    monkeypatch.setenv("BLADEX_MIN_IMPORTANCE", "abc")
    assert flag_number("BLADEX_MIN_IMPORTANCE") == 0.0
    monkeypatch.setenv("BLADEX_MIN_IMPORTANCE", "")
    assert flag_number("BLADEX_MIN_IMPORTANCE") == 0.0


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "off", ""])
def test_rollback_channel_still_works(monkeypatch, raw):
    """关闭通道保留：设 0/false/no/off/空 退回旧行为（既有"逐字一致"测试仍有效）。"""
    monkeypatch.setenv("BLADEX_SOFT_SCORING", raw)
    assert flag_enabled("BLADEX_SOFT_SCORING") is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on"])
def test_explicit_on_values(monkeypatch, raw):
    monkeypatch.setenv("BLADEX_HOP_EXPAND", raw)
    assert flag_enabled("BLADEX_HOP_EXPAND") is True


def test_unknown_flag_defaults_off():
    """表外的名字默认关——避免拼错开关名时静默启用未知行为。"""
    assert flag_enabled("BLADEX_NOT_A_REAL_FLAG") is False


# ── 2. 消费点真的读这张表（防止有人加回就地 == "1"）────────────────────────


def test_assembly_agents_consumer_reads_flag_table(monkeypatch):
    """（原 `_planes_enabled` 用例随 S1 删除；换成 S1 后仍在的消费点。）"""
    from bladex_proxy.assembly import assembly_enabled_for

    monkeypatch.delenv("BLADEX_ASSEMBLY_AGENTS", raising=False)
    assert assembly_enabled_for("claude-code") is True
    monkeypatch.setenv("BLADEX_ASSEMBLY_AGENTS", "")
    assert assembly_enabled_for("claude-code") is False


def test_no_inline_equals_one_checks_remain():
    """三个消费点不得再出现 `os.environ.get("BLADEX_XXX", "") == "1"` 的就地判断。

    就地判断正是默认值散落三包、与模板脱节的成因；默认值必须只有一处。
    """
    pkg_root = Path(__file__).resolve().parents[3] / "packages"
    targets = [
        pkg_root / "bladex-proxy" / "bladex_proxy" / "inject.py",
        pkg_root / "bladex-proxy" / "bladex_proxy" / "storage" / "memory_index.py",
        pkg_root / "bladex-core" / "bladex_core" / "prefetch.py",
        # ADR-0028 E2.1：第二批开关的消费点一并纳入扫描
        pkg_root / "bladex-core" / "bladex_core" / "consolidation_proxy.py",
    ]
    offenders: list[str] = []
    for path in targets:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for flag in MEMORY_FLAG_DEFAULTS:
            # 匹配 environ.get("FLAG", ...) == "1" 这种就地默认关的写法
            pat = rf'environ\.get\(\s*["\']{flag}["\'][^)]*\)\s*==\s*["\']1["\']'
            if re.search(pat, text):
                offenders.append(f"{path.name}: {flag}")
    assert not offenders, (
        "这些位置仍在就地判断开关（默认值会与 flags.MEMORY_FLAG_DEFAULTS 脱节）："
        + ", ".join(offenders)
    )


# ── 3. 与 config/.env.example 对账 ──────────────────────────────────────────


def test_env_example_declares_every_flag():
    text = _repo_env_example().read_text(encoding="utf-8")
    missing = [f for f in MEMORY_FLAG_DEFAULTS if f not in text]
    assert not missing, (
        f"config/.env.example 缺少开关声明：{missing}。"
        "机制默认值改了、模板没改 —— 这正是 2026-08-04 发现的那类腐化。"
    )


def test_env_example_values_match_defaults():
    text = _repo_env_example().read_text(encoding="utf-8")
    for flag, default_on in MEMORY_FLAG_DEFAULTS.items():
        m = re.search(rf'^\s*export\s+{flag}="?([^"\n]*)"?\s*$', text, re.MULTILINE)
        assert m, f"config/.env.example 里 {flag} 不是 `export {flag}=\"...\"` 形态"
        declared = m.group(1).strip().lower() not in ("0", "false", "no", "off", "")
        assert declared is default_on, (
            f"{flag}: .env.example 写 {m.group(1)!r}，flags.py 默认 {default_on} —— 两侧必须一致"
        )


def test_env_example_hotpath_budget_matches_config_default():
    """热路径预算：ProxyConfig 默认与模板必须一致（200ms，ADR-0027 §5.4）。

    这个值定小了 = 注入静默降级只注硬规则，是 08-04 实测 18% 失效的直接成因，
    所以它和开关一样要对账，不能一处改一处忘。
    """
    from bladex_proxy.config import ProxyConfig

    text = _repo_env_example().read_text(encoding="utf-8")
    m = re.search(r'^\s*export\s+BLADEX_HOTPATH_BUDGET_MS="?(\d+)"?\s*$', text, re.MULTILINE)
    assert m, "config/.env.example 未声明 BLADEX_HOTPATH_BUDGET_MS"
    import os

    saved = os.environ.pop("BLADEX_HOTPATH_BUDGET_MS", None)
    try:
        assert ProxyConfig().hotpath_budget_ms == 200
    finally:
        if saved is not None:
            os.environ["BLADEX_HOTPATH_BUDGET_MS"] = saved
    assert int(m.group(1)) == 200


def test_env_example_declares_numeric_defaults():
    """数值项同样对账（ADR-0028 E2.1）——一处改一处忘就红。"""
    text = _repo_env_example().read_text(encoding="utf-8")
    for name, default in MEMORY_NUMERIC_DEFAULTS.items():
        m = re.search(rf'^\s*export\s+{name}="?([^"\n]*)"?\s*$', text, re.MULTILINE)
        assert m, f"config/.env.example 未声明 {name}"
        assert float(m.group(1)) == default, (
            f"{name}: .env.example 写 {m.group(1)!r}，flags.py 默认 {default} —— 两侧必须一致"
        )


# ── 4. C1 三档登记（2026-09-03 批 B）────────────────────────────────────────


def test_every_flag_has_a_tier():
    from bladex_core.flags import FLAG_TIER_NAMES, FLAG_TIERS, MEMORY_TEXT_DEFAULTS

    every = set(MEMORY_FLAG_DEFAULTS) | set(MEMORY_NUMERIC_DEFAULTS) | set(MEMORY_TEXT_DEFAULTS)
    assert set(FLAG_TIERS) == every, "三张表里的旋钮必须逐一登记档位（漏登记 = 没人回答它是哪类）"
    assert set(FLAG_TIERS.values()) <= set(FLAG_TIER_NAMES)


def test_instruments_live_only_in_the_env_example_tail():
    """仪器不进 .env.example 正文：只允许出现在文末「仪器」段之后。"""
    from bladex_core.flags import FLAG_TIERS

    text = _repo_env_example().read_text(encoding="utf-8")
    marker = "# 仪器（C1 2026-09-03"
    assert marker in text, ".env.example 缺「仪器」段"
    body, tail = text.split(marker, 1)
    for name, tier in FLAG_TIERS.items():
        exported = re.search(rf'^\s*export\s+{name}=', body, re.MULTILINE)
        if tier == "instrument":
            assert not exported, f"{name} 是仪器，不得出现在正文 export 里"
            assert re.search(rf'^\s*export\s+{name}=', tail, re.MULTILINE), f"{name} 该在仪器段登记"
        else:
            assert not re.search(rf'^\s*export\s+{name}=', tail, re.MULTILINE), \
                f"{name} 不是仪器，不该在仪器段"
