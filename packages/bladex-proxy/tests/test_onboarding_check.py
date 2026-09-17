"""接入体检（G17.22 / MQ-A69）：新 agent 的三项能力是否真的就位。

## 这张表的用例**就是今天那三条缺陷的真实形态**

A66 / A67 / A68 三条各自独立，但表征完全相同：agent 接进来了、某项能力静默
不生效、日志看起来一切正常。**三次都是人发现的，机制一次都没发现。**
所以这里的每条用例都照抄当时的 live 读数 —— 它们是这个机制的**回归集**：
若干周后哪一项又静默了，对应的用例必须变红。

🔴 **最要紧的一条是 `test_home_dir_project_is_not_a_gap`**：判据必须是
`cwd_found` 而不是 `project_id`。用后者会把 hermes 那种"抽到了、是家目录、
按三档正确判 Global"报成缺口 —— **把正确行为报成缺口的告警会被训练成噪声，
然后被忽略，那比没有告警更糟。** 而这正是我 09-11 当天犯的错。
"""

from __future__ import annotations

import pytest
import structlog.testing

from bladex_proxy.onboarding import _MAX_SEEN, note_onboarding, onboarding_gaps, reset_seen


@pytest.fixture(autouse=True)
def _clean():
    reset_seen()
    yield
    reset_seen()


def _run(**over) -> list[dict]:
    kw = dict(agent_id="dsh", agent_source="fingerprint", auxiliary=False,
              cwd_found=True, project_id="p-abc", project_source="git-root",
              toolface_injected=True)
    kw.update(over)
    with structlog.testing.capture_logs() as cap:
        note_onboarding(**kw)
    return [e for e in cap if e.get("event") == "agent_onboarding_check"]


# ── ① 三条真实缺陷各一条用例 ─────────────────────────────────────────────


def test_a68_unknown_bucket_is_an_identity_gap():
    """MQ-A68 的 live 形态：opencode 首连落 `unknown-e0ca7fa5`、`fallback`。

    兜底分桶其实已经从 UA 里读出了 "opencode" 这个名字，规则库里却没有它 ——
    **看得见却认不出**，而 `agent_source=fallback` 在日志里一点都不刺眼。
    """
    rows = _run(agent_id="unknown-e0ca7fa5", agent_source="fallback")
    assert rows and "identity" in rows[0]["gaps"]
    assert rows[0]["log_level"] == "warning", "有缺口必须是 warning，不能淹在 info 里"


def test_a67_cwd_not_found_is_a_project_gap():
    """MQ-A67 的 live 形态：`project_resolved cwd= cwd_found=False source=global`。

    dsh / Pi 两跑 159 轮无一例外 —— 模式表里没有它们的载体形态。
    """
    rows = _run(agent_id="dsh", cwd_found=False, project_id="", project_source="")
    assert rows and "project" in rows[0]["gaps"]


def test_a66_toolface_excluded_is_a_toolface_gap():
    """MQ-A66 的 live 形态：`agency_toolface_decision injected=False reason=agent_excluded`。

    过期常量把 dsh 挡在工具面外 17 天；症状是账本零活动，
    而那**看起来完全像**"模型拿到工具不调用"（已知形态，实测 3%）。
    """
    rows = _run(toolface_injected=False)
    assert rows and "toolface" in rows[0]["gaps"]


def test_all_three_gaps_reported_together_in_stable_order():
    rows = _run(agent_id="unknown-x", agent_source="fallback",
                cwd_found=False, toolface_injected=False)
    assert rows[0]["gaps"] == "identity,project,toolface"


# ── ② 🔴 不许把正确行为报成缺口 ──────────────────────────────────────────


def test_home_dir_project_is_not_a_gap():
    """🔴 判据是 `cwd_found`，**不是 `project_id`**。

    hermes 的 live 形态：载体抽到了（`Current working directory: /Users/jasonye`），
    值是纯家目录 ⇒ `resolve_project` 第三档**正确**判 Global、`project_id` 为空。
    这是设计上正确的行为，不是缺口。

    `project_identity.resolve_from_request` 的 docstring 2026-08-29 就写了这条区分：
    「没抽到标记 ⇒ 改 pattern；抽到了不成项目 ⇒ 该 cwd 本就不是项目」。
    09-11 我无视了它，把四个 agent 的 0% 全当缺陷 —— 本条就是那次的回归钉。
    """
    rows = _run(agent_id="hermes", cwd_found=True, project_id="", project_source="")
    assert rows and rows[0]["gaps"] == "", "家目录 ⇒ Global 是正确行为，不得报缺口"
    assert rows[0]["log_level"] == "info"


def test_clean_agent_still_reports_once_at_info():
    """三项齐全也报一行 —— 体检要能回答"报过了吗"，不能只在坏的时候出现。

    只在异常时打印的仪器分不开"正常"与"没跑过"（`feedback_instrument_reference_frame`
    的"读数为 0 先证明桶可达"同族）。
    """
    rows = _run()
    assert rows and rows[0]["gaps"] == "" and rows[0]["log_level"] == "info"


def test_raw_readings_travel_with_the_verdict():
    """告警只说"哪项没就位"，**修法要看原始值** —— 三项原料必须同行落下。"""
    rows = _run(agent_id="unknown-y", agent_source="fallback", cwd_found=False,
                project_id="", project_source="", toolface_injected=False)
    r = rows[0]
    for k in ("identity_source", "cwd_found", "project_id", "project_source", "toolface"):
        assert k in r, f"缺原始读数 {k}，判断无法被复算"
    assert r["project_source"] == "global", "空 source 要落成 global，不留空串歧义"


# ── ③ 报一次，且 aux 不算数 ──────────────────────────────────────────────


def test_reported_once_per_agent_per_process():
    """高流量 agent 每轮报一行就是噪声，噪声等于没有告警。"""
    assert _run(agent_id="codex")
    assert _run(agent_id="codex") == [], "同一 agent 第二次不得再报"
    assert _run(agent_id="Pi"), "别的 agent 不受影响"


def test_aux_turn_is_skipped_and_does_not_consume_the_slot():
    """🔴 aux 轮整轮跳过，**且不记 seen**。

    子调用（标题生成 / 压缩）本来就不给工具面，拿它体检会把正常行为记成缺口；
    而有些 agent 的**第一轮恰好是 aux**（codex 实测标题生成走在前面）——
    若 aux 占掉了名额，真正的首个正式轮就再也体检不到了。
    """
    assert _run(agent_id="codex", auxiliary=True, toolface_injected=False) == []
    rows = _run(agent_id="codex", auxiliary=False)
    assert rows, "aux 不得占用名额，下一个正式轮仍要体检"
    assert rows[0]["gaps"] == ""


def test_seen_set_is_bounded_against_unclaimed_fingerprint_buckets():
    """未认领指纹桶 `unknown-<sha>` 每个都不同，不设上限会把集合撑爆。

    撑满后**停止体检**而不是逐出 —— 逐出会让同一个 agent 反复重报，那就成了噪声。
    """
    for i in range(_MAX_SEEN):
        _run(agent_id=f"unknown-{i:08x}", agent_source="fallback")
    assert _run(agent_id="brand-new-agent") == [], "撑满后停止体检，不逐出"


# ── ④ 纯函数判据（与日志接线分开测）──────────────────────────────────────


@pytest.mark.parametrize("kw,expect", [
    (dict(agent_source="fingerprint", agent_id="dsh", cwd_found=True,
          toolface_injected=True), []),
    (dict(agent_source="fallback", agent_id="dsh", cwd_found=True,
          toolface_injected=True), ["identity"]),
    # 🔴 `unknown-` 前缀单独也要中：agent_source 可能是别的值，
    #    而挂在 unknown 命名空间上这件事本身就是缺口。
    (dict(agent_source="fingerprint", agent_id="unknown-abc", cwd_found=True,
          toolface_injected=True), ["identity"]),
])
def test_gaps_pure_function(kw, expect):
    assert onboarding_gaps(**kw) == expect


# ── ⑤ 批 H 复核两处小改（2026-09-12，批 J 顺延项）──────────────────────────


def test_pending_agrees_with_note_onboarding():
    """`pending` 与 `note_onboarding` 用同一个键、同一条 aux 规则、同一个上限。

    调用方靠 `pending` 决定要不要付 `extract_cwd` 的重算；若两者分叉，
    要么白算（pending 真、note 假），要么漏检（pending 假、note 本会报）。
    """
    from bladex_proxy.onboarding import pending

    assert pending(agent_id="dsh", auxiliary=False) is True
    assert _run(agent_id="dsh") != []
    assert pending(agent_id="dsh", auxiliary=False) is False, "报过一次后 pending 转假"
    assert pending(agent_id="codex", auxiliary=True) is False, "aux 轮不体检"
    assert pending(agent_id="", auxiliary=False) is False


def test_call_site_does_not_restate_dedup_key():
    """批 H 复核：`server/orchestration.py` 不许复述 `agent_id not in _seen`。

    G17.22 要把去重键改成 (agent_id, project_id)；键的定义只在 `onboarding._dedup_key`
    一处，调用方只问 `pending`。外面复制一份判据 = 改了里面、外面那句静默挡住新行为。
    """
    from _source_probe import package_source

    src = package_source("server")
    assert "_onb._seen" not in src, "调用点直接读 `_onb._seen` = 复述了去重键"
    assert "_onb.pending(" in src


def test_onboarding_failure_never_escapes_hot_path():
    """批 H 复核：体检是纯观测，`extract_cwd` 在 39 万字符消息上跑正则出错
    不许打进热路径（预算 200ms 内的一段副作用，出错只记一行）。

    只读源码不 import `orchestration`（它拖 fastapi / Router 网关；本守卫要在最轻的
    环境里也能跑）。断言的形状：`pending(` 之后紧跟 `try:`，`except Exception`
    里记 `agent_onboarding_check_failed`，且 `extract_cwd(` 落在 try 块内。
    """
    import re

    from _source_probe import package_source

    src = package_source("server")
    m = re.search(r"if _onb\.pending\([^)]*\):\s*\n\s*try:\n(.*?)except Exception as exc:", src, re.S)
    assert m, "调用点必须是 `if _onb.pending(...):` 紧跟 `try:` … `except Exception as exc:`"
    assert "extract_cwd(" in m.group(1), "`extract_cwd` 必须在 try 块内——它才是会抛的那一步"
    tail = src[m.end():m.end() + 400]
    assert "agent_onboarding_check_failed" in tail, "失败只许记一行，且事件名固定可 grep"


def test_agent_source_enum_is_normalised_to_short_name():
    """批 J 顺手发现（2026-09-12 live）：调用方传的是 `AgentSource` 枚举，
    `str(AgentSource.FALLBACK)` 在 3.11+ 是 "AgentSource.FALLBACK"，
    ⇒ `agent_source == "fallback"` 这条判据在 live 从未命中过（只剩 `unknown-` 前缀那条在扛），
    且日志 `identity_source=AgentSource.FINGERPRINT` 与 `identity_resolved` 的 `agent_source=fingerprint` 同名不同形。
    """
    from bladex_proxy.models import AgentSource

    assert onboarding_gaps(agent_source=AgentSource.FALLBACK, agent_id="opencode",
                           cwd_found=True, toolface_injected=True) == ["identity"]
    rows = _run(agent_id="opencode", agent_source=AgentSource.FALLBACK)
    assert rows and rows[0]["identity_source"] == "fallback"
    rows = _run(agent_id="dsh2", agent_source=AgentSource.FINGERPRINT)
    assert rows and rows[0]["identity_source"] == "fingerprint"
