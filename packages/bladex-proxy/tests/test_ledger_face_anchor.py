"""MQ-L26：没拿到账本面的轮次不参与锚定归属。

## 守的是什么

账本面 gating（proxy 侧，请求时）与锚定归属（Memory Index 侧，重建时）此前用
**两套完全不同的判据**，中间无对账：

    给不给账本面：is_ledgerless_auxiliary(aux) or subagent or brings_no_tools
                  or is_isolated_subcall
    锚不锚      ：只看 ledger_active_at(scope, ts)

⇒ 一轮没拿到账本面（模型看不到账本、也没有 `bladex_ledger_switch` 工具），
它的产出**照样**被锚到当时激活的旧账本上。

live 病例（`docs/reviews/t5-matter-signoff-20260828.md` §2.1）：
`[3] Codex ledger health 开发` 混入 10 条 BladeX 全局状态，全是 08-28 13:32–13:33
的零工具「建议生成」轮（`reason=aux`），而该卡其余成员是 08-27 20:xx。

**锚定一条模型从来没机会表态的产出，等于替它做了归属判断**——违反 ADR-0032
的作者边界。这是 T5 复验的净新增发现，也是「生产方与消费方判据不一致」
在同一天里的第五个实例（前四个：MQ-S46、注入标记无落点、
`session_resume_recap` 覆盖面更窄、`agent_excluded_or_already_present` 二义）。

## 三态是本次修法的关键

`None`（历史轮没有这个字段）**不能**当成 `False`，否则全量重建时所有历史锚定
一次性失效（ADR-0012 §3.6）。缺数就报缺数——这是"缺数长得像强证据"那条教训
的反面用法。
"""
from __future__ import annotations

import pathlib

from bladex_proxy.models import Turn

from _source_probe import package_source, source_of

#: 包源码目录。**按路径读文本，不 import** —— `memory_index` 需要 rocksdict、
#: `server` 需要 fastapi/litellm，而本组测的全是**源码写法**，不该被 native
#: 依赖挡在门外（`memory_index` 自己的注释也写着"延迟导入：纯函数可在无 native
#: 依赖环境测"）。顺带：这样也不受 stale `.pyc` 影响（MQ-V9）。
_PKG = pathlib.Path(__file__).resolve().parents[1] / "bladex_proxy"


def _src(*parts: str) -> str:
    p = _PKG.joinpath(*parts)
    assert p.exists(), f"源文件不在：{p} —— 挪位置了？本组会假绿，先修这里"
    return p.read_text(encoding="utf-8")


# ── ① schema：三态，且默认是 None ────────────────────────────────────────────

def test_ledger_face_is_tri_state_defaulting_to_none() -> None:
    """🔴 默认必须是 `None`，不是 `False`。

    默认 `False` = 把「记录里没有这个信息」说成「明确没给账本面」，
    全量重建时所有历史轮会一次性失去锚定。
    """
    f = Turn.model_fields["ledger_face"]
    assert f.default is None, (
        f"默认值是 {f.default!r} —— 必须是 None。见本文件 docstring 末段。")
    t = Turn(identity=_ident(), model="m", request_messages=[])
    assert t.ledger_face is None


def test_ledger_face_accepts_all_three_states() -> None:
    for v in (None, True, False):
        t = Turn(identity=_ident(), model="m", request_messages=[], ledger_face=v)
        assert t.ledger_face is v


def _ident():
    from bladex_proxy.identity import Identity
    return Identity(user_id="u", agent_id="a", session_id="s")


# ── ② 锚定判据：只有明确 False 才拦 ──────────────────────────────────────────

def test_anchor_skips_only_explicit_false() -> None:
    """🔴 判据必须是 `is False`，不是 falsy。

    `None` 是 falsy —— 写成 `if not ledger_key_to_face.get(lk)` 会把历史轮
    （None）和缺键（None）一起拦掉，那正是上面那条要防的事故。
    本条读源码钉住这个写法，因为它是**一个字符的差别**、且错了不会报错。
    """
    src = _anchor_src()
    assert "ledger_key_to_face.get(lk) is False" in src, (
        "锚定判据不是 `is False` —— None（历史轮/缺键）会被误当成"
        "『明确没给账本面』而丢掉锚定"
    )


def _anchor_src() -> str:
    """取 `_la_matter_for` 的源码。它是嵌套函数，`source_of` 只找顶层/类内，
    故这里按文本切片——**带前置断言**，切不到就报错而不是返回空串。"""
    text = _src("storage", "memory_index.py")
    head = text.find("def _la_matter_for(")
    assert head > 0, "找不到 _la_matter_for —— 改名了？本条会假绿，先修这里"
    tail = text.find("\n        # ", head)      # 下一个同级注释块
    assert tail > head, "切不到函数尾"
    return text[head:tail]


def test_anchor_counts_the_skips() -> None:
    """跳过要有计数器，否则修完了也不知道它有没有在工作。

    恒 0 有两种成因（live 没这类轮 / 字段没接上），日志注释里写了怎么分。
    """
    src = _anchor_src()
    assert '_la_counters["no_ledger_face"]' in src


def test_skip_counter_reaches_the_log() -> None:
    """计数器必须打进 `index_ledger_anchor` —— 只加不报等于没有观测。"""
    text = _src("storage", "memory_index.py")
    i = text.find('logger.info("index_ledger_anchor"')
    assert i > 0, "找不到 index_ledger_anchor 日志"
    block = text[i:i + 900]
    assert "no_ledger_face=" in block, "计数器没进日志"


# ── ③ 生产接线：proxy 侧真的记了这个决定 ─────────────────────────────────────

def test_augment_tools_records_the_decision() -> None:
    """`augment_tools` 必须把决定记到 runtime 上，供 `_enqueue_turn` 取。"""
    from bladex_proxy.agency import AgencyRuntime

    src = source_of(AgencyRuntime, "augment_tools")
    assert "self.last_ledger_face = injected" in src


def test_enqueue_turn_reads_it_without_defaulting_to_false() -> None:
    """🔴 `_enqueue_turn` 取不到时必须写 `None`，不能写 `False`。

    与 ① 同一条理由：取不到 ≠ 明确没给。这里单独钉，是因为
    `getattr(x, 'y', False)` 是极顺手的写法，而它会静默毁掉历史锚定。

    🔴 F1.3 / MQ-A49：取值处从 `_ag.last_ledger_face`（进程级"最近一次"，
    await 之后读 ⇒ 并发串台）改成 `request.state.bladex_ledger_face`
    （请求作用域，由 `_apply_agency_surfaces` 在同步链里停下）。
    **三态语义与本条的判据一字未动**——变的是从哪儿取，不是取不到时写什么。
    """
    src = package_source("server")   # F0.1 拆包：server.py → server/ 包，按包拼接读源码
    assert '"bladex_ledger_face", None)' in src, (
        "兜底值不是 None —— 见本条 docstring")
    assert 'request.state.bladex_ledger_face = getattr(agency, "last_ledger_face", None)' \
        in src, "生产侧的停放点没了 —— 那样 request.state 上恒为 None（静默退化）"
    assert "ledger_face=_ledger_face" in src, "没传进 Turn"


def test_runtime_field_starts_as_none() -> None:
    """进程刚起、还没处理过请求时是 `None`（不是 False）。"""
    from bladex_proxy.agency import AgencyRuntime

    src = source_of(AgencyRuntime, "__init__")
    assert "self.last_ledger_face: bool | None = None" in src


# ── ④ 映射建立：扫描阶段把 ledger_face 收进 dict ────────────────────────────

def test_scan_phase_builds_the_face_map() -> None:
    """归属循环只有 `fact.source_ledger_key`、拿不到 Turn，故映射必须在扫描阶段建。

    漏了这一步，`ledger_key_to_face` 恒空 ⇒ `.get(lk)` 恒 None ⇒ **修法静默失效**
    （测试全绿、行为照旧）。这是本组最容易假绿的一环，故单独钉。
    """
    text = _src("storage", "memory_index.py")
    assert 'ledger_key_to_face[key_str] = getattr(turn, "ledger_face", None)' in text, (
        "扫描阶段没建映射 —— 锚定判据会恒拿到 None，修法静默失效"
    )
