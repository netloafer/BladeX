"""个人模式身份语义（ADR-0021 §2.3，2026-08-16 修订）的验收剧本。

修订内容：无 `identity.toml` 时 `user_id` 从「API Key 的 hash8」改为常量 `local`。

**为什么这条值得一整个测试文件**：原语义把凭证当身份，换一把 key / 换台设备就是换个人，
记忆静默劈成两半。live 实测 4 个 user_id 实为同一人，一件泰山啤酒尽调被切开，
而 P2 散点召回按 user_id 过滤 → 两半互相召不回；该分裂自 ADR-0021 上线起潜伏、
无任何告警。改回去的代价是全库身份漂移，所以要把语义钉死。

三条各自独立、不可互相替代：
  ① 个人模式：任何 key 都落同一个 `local`（凭证解耦）
  ② 无 key：落 `anonymous`，**任何模式都不并入** —— 它是 auth 开关的探测器
  ③ 企业模式：本次修订**一个字节都不许动**（回归保护）
"""

from __future__ import annotations

from bladex_proxy.identity import (
    ANONYMOUS_USER_ID,
    LOCAL_USER_ID,
    resolve_identity,
)
from bladex_proxy.models import ChatCompletionRequest


def _req() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="m", messages=[{"role": "user", "content": "你好"}])


def _resolve(api_key: str | None, registry=None):
    headers = {"authorization": f"Bearer {api_key}"} if api_key else {}
    ident, _src = resolve_identity(headers, _req(), registry=registry)
    return ident


# ── ① 个人模式：凭证解耦 ────────────────────────────────────────────────────


def test_personal_mode_gives_the_same_identity_for_any_key() -> None:
    """🔴 本次修订的核心：换 key 不换身份。

    原实现这里返回两个不同的 hash8 —— 那正是 live 上 24345848 / 53136499
    两半记忆互相召不回的根因。
    """
    a = _resolve("bladex-key-macbook")
    b = _resolve("bladex-key-codex")
    assert a.user_id == LOCAL_USER_ID
    assert b.user_id == LOCAL_USER_ID
    assert a.user_id == b.user_id, "个人模式换 key 必须仍是同一个人"


def test_local_id_is_a_constant_not_a_hash() -> None:
    """常量必须是可读常量。一旦变回 hash 形态，'换 key 不换身份'就失守了。"""
    assert LOCAL_USER_ID == "local"
    assert len(LOCAL_USER_ID) != 8 or not all(
        c in "0123456789abcdef" for c in LOCAL_USER_ID)


def test_personal_mode_treats_empty_registry_as_personal() -> None:
    """`registry.empty is True` 等价于没有 identity.toml —— 别把空注册表当企业模式。"""
    class _Empty:
        empty = True

    assert _resolve("bladex-key-x", registry=_Empty()).user_id == LOCAL_USER_ID


# ── ② 无 key：探测器，不并入 ────────────────────────────────────────────────


def test_no_api_key_never_folds_into_local() -> None:
    """🔴 anonymous 不是"另一个用户"，是 auth 开关正确性的探测器。

    并入 local 之后，auth 关了也永远发现不了 —— 唯一的信号被消解。
    """
    assert _resolve(None).user_id == ANONYMOUS_USER_ID
    assert _resolve(None).user_id != LOCAL_USER_ID


def test_no_api_key_stays_anonymous_in_enterprise_mode_too() -> None:
    """企业模式同样不并入：无 key 请求不该继承任何 principal。"""
    class _Reg:
        empty = False

        def resolve(self, api_key, fallback_user_id=""):    # noqa: ARG002
            return None

    assert _resolve(None, registry=_Reg()).user_id == ANONYMOUS_USER_ID


def test_no_api_key_emits_a_warning() -> None:
    """探测器要能被读到。告警名 `identity_no_api_key`（按 `not api_key` 触发，
    不含字面量 'anonymous' —— 2026-08-16 曾因 grep 错关键词而误判"没有告警"）。

    🔴 必须用 `structlog.testing.capture_logs`，**不能用 pytest 的 `caplog`**：
    本仓自有进程用 structlog 直接写 stdout，不经过 stdlib logging 的 handler，
    `caplog.records` 恒为空 —— 首跑就是这么假红的（日志明明打出来了）。
    """
    import structlog.testing

    with structlog.testing.capture_logs() as cap:
        _resolve(None)
    hit = [e for e in cap if e.get("event") == "identity_no_api_key"]
    assert hit, f"无 key 必须留下可 grep 的 warning，实得 {[e.get('event') for e in cap]}"
    assert hit[0].get("log_level") == "warning", "必须是 warning 级——info 会淹没在日志里"


def test_normal_key_does_not_emit_the_no_key_warning() -> None:
    """阴性对照：正常带 key 时不许报警。

    没有这条，上一条只证明"会打日志"，不能证明"只在该报的时候打"——
    一个恒报警的探测器与不报警的一样没用。
    """
    import structlog.testing

    with structlog.testing.capture_logs() as cap:
        _resolve("bladex-key-normal")
    assert not [e for e in cap if e.get("event") == "identity_no_api_key"]


# ── ③ 企业模式：零回归 ──────────────────────────────────────────────────────


def test_enterprise_mode_still_resolves_to_principal() -> None:
    """有 identity.toml 时走 principal，本次修订不得影响这一支。"""
    class _Resolved:
        principal_id = "jason"
        visibility = ["personal:jason", "team:rd"]
        principal_sensitivity = ""
        team_sensitivities: list[str] = []

    class _Reg:
        empty = False

        def resolve(self, api_key, fallback_user_id=""):    # noqa: ARG002
            return _Resolved()

    ident = _resolve("bladex-key-x", registry=_Reg())
    assert ident.user_id == "jason"
    assert ident.visibility == ["personal:jason", "team:rd"]
    assert ident.team_ids == ["rd"]


def test_fold_user_id_is_wired_at_every_production_rebuild_call_site() -> None:
    """🔴 参数存在 ≠ 生产上会传。

    本轮已撞见**六例**"机制存在但没接线"（assistant_progress / refresh_profile_docs /
    五开关默认关 / L0 / L4 recent_members / IDENTITY_MERGE 重放）。`fold_user_id`
    漏接一处的后果很具体：那条路径重建出来的 fact 仍带旧 hash8 user_id，
    与 `local` 分家 —— 正是本次修订要消除的两分裂，而且是安静的。

    故直接对源码断言：`rebuild_from_hub(` 每一个生产调用点都要带 `fold_user_id`。
    """
    import inspect

    from bladex_proxy import consolidator
    from bladex_proxy.cli import ops_cmds   # F0.1 拆包：sync 命令（rebuild_from_hub 调用点）在 cli/ops_cmds.py

    total_calls = 0
    for mod in (ops_cmds, consolidator):
        src = inspect.getsource(mod)
        total_calls += src.count("rebuild_from_hub(")
        # 每个调用点后面若干行内必须出现 fold_user_id
        for idx, line in enumerate(src.splitlines()):
            if "rebuild_from_hub(" in line and "def " not in line:
                window = "\n".join(src.splitlines()[idx:idx + 6])
                assert "fold_user_id" in window, (
                    f"{mod.__name__} 有一处 rebuild_from_hub 没传 fold_user_id：\n{window}")
    assert total_calls >= 3, "生产调用点少于 3 个，接线检查的分母不对，先核对是不是重构过"


def test_fold_user_id_defaults_to_off_for_enterprise() -> None:
    """企业模式必须**不**折叠 —— 折叠会把不同 principal 压成一个人，是隔离的反面。"""
    import inspect

    from bladex_proxy import consolidator

    src = inspect.getsource(consolidator)
    assert 'fold_user_id = "" if not cfg.identity_registry.empty else LOCAL_USER_ID' in src, (
        "折叠开关的判据必须是 registry.empty（= 无 identity.toml = 个人模式）")


def test_enterprise_unknown_key_falls_back_to_hash_not_local() -> None:
    """§2.3 回落规则 2：企业模式下未声明的 key 回落 hash8 隐式 principal。

    🔴 **不能**回落成 `local` —— 那会把陌生凭证并进本地身份，方向与隔离相反。
    """
    class _Reg:
        empty = False

        def resolve(self, api_key, fallback_user_id=""):    # noqa: ARG002
            return None

    uid = _resolve("bladex-unknown-key", registry=_Reg()).user_id
    assert uid != LOCAL_USER_ID
    assert len(uid) == 8 and all(c in "0123456789abcdef" for c in uid)
