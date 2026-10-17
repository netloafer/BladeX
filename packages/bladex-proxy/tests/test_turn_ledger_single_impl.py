"""批 Y · Y1a（MQ-L88）：「这一轮发生时激活的是哪本账本」只有一个实现点，且按该轮自己的 project 查。

钉住：
  ① 重建锚层按**该轮的** project scope 查（阳性）：Global 与项目键各绑一本 ⇒ 带 project 的轮次
     锚到项目键那本。把 `_la_ledger_at` 改回写死 `""` ⇒ 本条红（它会锚到 Global 那本 = Y0 的 59 条形状）；
  ② 阴性：不带 project 的轮次仍查 Global（修法不是"一律查项目键"）；
  ③ `LEDGER_SCOPE_MIGRATE` 被时间轴重放：迁移之前查 Global、之后查项目键，Global 在迁移时刻变空；
  ④ 墓碑 × 迁移：墓碑前的迁移原样保留，项目键窗口在墓碑时刻闭合（不换受害者）；
  ⑤ 静态守卫：生产代码里除 `turn_active_ledger` 外，任何函数同时取 turn key 身份
     （`agent_of_turn_key` / `turn_key_ts_ms`）又查 scope / 时间轴（`activation_scope` /
     `ledger_active_at` / `ledger_activation_windows`）⇒ 红。自带阳性样本（守卫能抓到第二份写法）。
"""

from __future__ import annotations

import ast
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from bladex_core.ledger import (
    EVENT_LEDGER_SCOPE_MIGRATE,
    EVENT_LEDGER_SWITCH,
    apply_ledger_tombstones,
    ledger_activation_windows,
    switch_timeline,
)
from bladex_core.ledger_runtime import (
    activation_scope,
    ledger_anchor_matter_id,
    turn_active_ledger,
)
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    SessionIdSource,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex
from test_ledger_anchor_wiring import MockEmbedder, StubDistiller, _anchor_edges

_A = "claude-code"
_P = "p-8aba3f61da56"
_LG = "ldg-468debdba521"     # Global 上那本（Y0：「评估 v5 架构」）
_LP = "ldg-a0439a790bdb"     # 项目键上模型切对的那本

_ROOT = Path(__file__).resolve().parents[3]


def _key(ms: float, agent: str = _A) -> str:
    return f"u1/{agent}/s1/{int(ms)}-0"


def _sw(ts_ms: float, to: str, project: str = "", agent: str = _A) -> dict:
    return {"type": EVENT_LEDGER_SWITCH, "ts": ts_ms,
            "payload": {"agent_id": agent, "project_id": project, "to_ledger_id": to}}


def _mig(ts_ms: float, lid: str, to_project: str = _P, agent: str = _A) -> dict:
    return {"type": EVENT_LEDGER_SCOPE_MIGRATE, "ts": ts_ms,
            "payload": {"agent_id": agent, "from_project": "", "to_project": to_project,
                        "ledger_id": lid}}


# ── ③ 迁移重放（纯函数） ──────────────────────────────────────────────────────

T0 = 1_790_000_000_000.0


def test_migrate_replayed_on_timeline() -> None:
    tl = switch_timeline([_sw(T0, _LG), _mig(T0 + 10_000, _LG)])
    # 迁移前：Global 轮查到它；项目键轮什么都没有（运行时迁移就发生在第一条带 project 的请求上）
    assert turn_active_ledger(tl, _key(T0 + 5_000), "") == _LG
    # 迁移后：项目键接手，Global 变空（运行时 `ActivationTable.migrate` 是 pop src）
    assert turn_active_ledger(tl, _key(T0 + 20_000), _P) == _LG
    assert turn_active_ledger(tl, _key(T0 + 20_000), "") == "", \
        "迁移之后 Global 仍查到迁走的那本 ⇒ 时间轴没重放 LEDGER_SCOPE_MIGRATE"


def test_migrate_does_not_backfill_project_history() -> None:
    """迁移之前的项目键轮次不得被终态兜底塞给迁移来的那本（MQ-L15 反向形状）。

    `T0-1s` 另一个项目上的切换 = project 维已生效（见 `project_scope_since`），
    否则 T0+5s 那轮会按"project 维尚未生效"查 Global——那是另一条用例的事。"""
    events = [_sw(T0 - 1_000, "ldg-other", "p-other"), _sw(T0, _LG), _mig(T0 + 10_000, _LG)]
    tl = switch_timeline(events)
    bindings = {activation_scope(_A, _P): _LG}
    assert turn_active_ledger(tl, _key(T0 + 5_000), _P, bindings=bindings) == ""


def test_project_dimension_before_first_project_event_reads_global() -> None:
    """原则 13 层 4：08-29 `identity.project_id` 先于"绑定按 project 分键"接通。

    live 病例：codex 轮次 04:28 起带 `p-10ad1f61d0c2`，09:23 仍在 `codex@Global` 上切账本，
    第一条带 project 的 switch 10:12:49。那之前按轮次 project 查 = 把轮次从它当时真正所在的
    账本上摘掉。生效时刻取事件流里第一条非 Global 激活事件（运行时自己的记录，不是日期常量）。"""
    tl = switch_timeline([_sw(T0, _LG), _sw(T0 + 60_000, _LP, _P)])
    assert turn_active_ledger(tl, _key(T0 + 30_000), _P) == _LG, \
        "project 维生效之前的轮次应按 Global 查（运行时当时就是这么查的）"
    assert turn_active_ledger(tl, _key(T0 + 90_000), _P) == _LP
    assert turn_active_ledger(tl, _key(T0 + 90_000), "") == _LG


def test_windows_follow_migrate() -> None:
    tl = switch_timeline([_sw(T0, _LG), _mig(T0 + 10_000, _LG), _sw(T0 + 30_000, "ldg-x", _P)])
    w = ledger_activation_windows(tl, _LG)
    assert w == {activation_scope(_A, ""): [(T0, T0 + 10_000)],
                 activation_scope(_A, _P): [(T0 + 10_000, T0 + 30_000)]}


# ── ④ 墓碑 × 迁移 ────────────────────────────────────────────────────────────

def test_tombstone_after_migrate_closes_project_window() -> None:
    events = [_sw(T0, _LG), _mig(T0 + 10_000, _LG)]
    out = apply_ledger_tombstones(events, {_LG}, tombstone_ts_ms={_LG: T0 + 50_000})
    tl = switch_timeline(out)
    assert turn_active_ledger(tl, _key(T0 + 5_000), "") == _LG         # 墓碑前 · 迁移前
    assert turn_active_ledger(tl, _key(T0 + 20_000), "") == ""          # 迁移后 Global 空
    assert turn_active_ledger(tl, _key(T0 + 20_000), _P) == _LG         # 墓碑前 · 迁移后
    assert turn_active_ledger(tl, _key(T0 + 60_000), _P) == ""          # 墓碑后
    assert turn_active_ledger(tl, _key(T0 + 60_000), "") == ""


def test_tombstone_unknown_time_drops_migrate() -> None:
    """墓碑时刻未知 ⇒ 旧行为（切入改写为空、迁移丢弃）：该账本在任何轴上都查不到。"""
    out = apply_ledger_tombstones([_sw(T0, _LG), _mig(T0 + 10_000, _LG)], {_LG})
    tl = switch_timeline(out)
    for ms in (T0 + 5_000, T0 + 20_000):
        for p in ("", _P):
            assert turn_active_ledger(tl, _key(ms), p) == ""


# ── ①② 重建锚层（生产接线） ─────────────────────────────────────────────────

def _now_ms() -> int:
    return int(datetime.now(UTC).timestamp() * 1000)


def _tick() -> None:
    time.sleep(0.02)


def _put(hub: MemoryHub, seq: int, user: str, *, project: str) -> str:
    identity = Identity(user_id="u1", agent_id=_A, session_id=f"s{seq}", turn_index=seq,
                        session_id_source=SessionIdSource.FINGERPRINT, project_id=project)
    turn = Turn(identity=identity, model="test",
                request_messages=[{"role": "user", "content": user}],
                response_text="好的。", status=TurnStatus.OK,
                ts=datetime.now(UTC) + timedelta(milliseconds=0),
                session_id_source=SessionIdSource.FINGERPRINT)
    key = identity.storage_key(f"{_now_ms()}-0")
    hub.put(key, turn)
    _tick()
    return key


def _open(tmp_path: Path) -> tuple[MemoryHub, MemoryIndex]:
    hub = MemoryHub(tmp_path / "rocksdb")
    hub.open()
    index = MemoryIndex(tmp_path / "index", embedder=MockEmbedder(),
                        read_only=False, distiller=StubDistiller())
    index.open()
    return hub, index


def test_anchor_uses_turn_project_scope(tmp_path, monkeypatch) -> None:
    """① 阳性 + ② 阴性，同一次 rebuild（分两次 rebuild 会让错法蒙对，见 anchor_wiring ③）。"""
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    hub, index = _open(tmp_path)
    try:
        hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id=_A, project_id="",
                               session_id="s0", to_ledger_id=_LG)
        _tick()
        hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id=_A, project_id=_P,
                               session_id="s1", to_ledger_id=_LP)
        _tick()
        kp = _put(hub, 2, "在 bench worktree 里：把 handoff 基准的第三组任务跑完并记录每组的轮次", project=_P)
        kg = _put(hub, 3, "家目录会话：另一件和项目无关的事，整理一下本周的阅读清单和笔记", project="")
        index.rebuild_from_hub(hub)

        by_turn: dict[str, set[str]] = {}
        facts = {f.id: f for f in index.all_facts()}
        for e in _anchor_edges(index):
            f = facts.get(e.target_key)
            if f is not None:
                by_turn.setdefault(f.source_ledger_key, set()).add(e.matter_id)
        assert by_turn.get(kp) == {ledger_anchor_matter_id(_LP)}, \
            f"带 project 的轮次没锚到项目键那本（MQ-L88 形状：锚层只查 Global）：{by_turn.get(kp)}"
        assert by_turn.get(kg) == {ledger_anchor_matter_id(_LG)}, \
            f"不带 project 的轮次应仍查 Global：{by_turn.get(kg)}"
    finally:
        index.close()
        hub.close()


def test_anchor_replays_scope_migrate(tmp_path, monkeypatch) -> None:
    """③ 生产接线：迁移之后 Global 上的轮次不再锚到迁走的那本。"""
    monkeypatch.setenv("BLADEX_LEDGER_ANCHOR", "1")
    hub, index = _open(tmp_path)
    try:
        hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id=_A, project_id="",
                               session_id="s0", to_ledger_id=_LG)
        _tick()
        k_before = _put(hub, 1, "迁移之前的 Global 会话：评估 v5 架构的账本注入面和工具面拆分方案", project="")
        hub.append_admin_event(AdminEventType.LEDGER_SCOPE_MIGRATE, "", agent_id=_A,
                               from_project="", to_project=_P, ledger_id=_LG)
        _tick()
        k_proj = _put(hub, 2, "迁移之后在项目里接着做：评估 v5 架构的拦截协议剥离拼接部分", project=_P)
        k_glob = _put(hub, 3, "迁移之后回到家目录：查一下明天的天气和出行安排，和项目无关", project="")
        index.rebuild_from_hub(hub)

        facts = {f.id: f for f in index.all_facts()}
        anchored = {facts[e.target_key].source_ledger_key for e in _anchor_edges(index)
                    if e.target_key in facts}
        assert k_before in anchored and k_proj in anchored
        assert k_glob not in anchored, "迁移之后 Global 轮次仍锚到迁走的那本 ⇒ 时间轴没重放迁移"
    finally:
        index.close()
        hub.close()


# ── ⑤ 静态守卫：单实现点 ────────────────────────────────────────────────────

_TURN_ID = frozenset({"agent_of_turn_key", "turn_key_ts_ms"})
_SCOPE_LOOKUP = frozenset({"activation_scope", "ledger_active_at", "ledger_activation_windows"})
#: 唯一允许的实现点（模块相对路径, 函数名）。
_ALLOWED = frozenset({("packages/bladex-core/bladex_core/ledger_runtime.py", "turn_active_ledger"),
                      ("packages/bladex-core/bladex_core/ledger_runtime.py", "turn_activation_scope")})


def _called_names(fn: ast.AST) -> set[str]:
    """函数**自身**体内调用的名字（不下钻嵌套 def——嵌套函数单独判）。"""
    out: set[str] = set()
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
        stack.extend(ast.iter_child_nodes(n))
    return out


def turn_ledger_mapping_violations(source: str, rel: str) -> list[str]:
    bad: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        calls = _called_names(node)
        if calls & _TURN_ID and calls & _SCOPE_LOOKUP and (rel, node.name) not in _ALLOWED:
            bad.append(f"{rel}:{node.lineno} {node.name}()")
    return bad


def _production_files() -> list[Path]:
    out: list[Path] = []
    for pkg in (_ROOT / "packages").iterdir():
        for mod in pkg.glob("bladex_*"):
            if mod.is_dir():
                out.extend(p for p in mod.rglob("*.py") if "tests" not in p.parts)
    return sorted(out)


def test_single_implementation_point() -> None:
    files = _production_files()
    assert any(p.name == "memory_index.py" for p in files), "扫描范围不对：没扫到 memory_index.py"
    bad: list[str] = []
    for p in files:
        rel = str(p.relative_to(_ROOT))
        bad += turn_ledger_mapping_violations(p.read_text(encoding="utf-8"), rel)
    assert not bad, (
        "「这一轮激活的是哪本账本」出现了第二种写法（原则 13 层 2 / MQ-L88）——"
        f"改调 `ledger_runtime.turn_active_ledger`：{bad}")


def test_guard_catches_second_mapping() -> None:
    """守卫阳性样本：MQ-L88 修前 `_la_ledger_at` 的原样写法 / MQ-I22 修前的窗口查表 ⇒ 都必须被抓到。"""
    la_old = (
        "def _la_ledger_at(lk):\n"
        "    agent = agent_of_turn_key(lk)\n"
        "    scope = activation_scope(agent, '')\n"
        "    return ledger_active_at(_tl, scope, turn_key_ts_ms(lk))\n")
    window_old = (
        "def ledger_window_turn_keys(hub, lid):\n"
        "    for key, meta in rows:\n"
        "        spans = windows.get(activation_scope(agent_of_turn_key(key), meta['p']))\n")
    assert turn_ledger_mapping_violations(la_old, "x.py") == ["x.py:1 _la_ledger_at()"]
    assert turn_ledger_mapping_violations(window_old, "y.py") == ["y.py:1 ledger_window_turn_keys()"]
    # 阴性：只调单实现点的写法不报
    ok = "def f(lk):\n    return turn_active_ledger(tl, lk, p)\n"
    assert turn_ledger_mapping_violations(ok, "z.py") == []
