"""批 T0（2026-09-22）：`--with-memory` 的窗口 scope（MQ-I22）+ `--dry-run`（MQ-I21）。

# MQ-I22 —— 又一个"0 长得像成功"

`ledger_window_turn_keys` 此前按 `activation_scope(agent, "")` 查窗口表，而表的键是真实
scope `agent␟project` ⇒ **凡有 project 身份的账本一律查不中** ⇒ `turn_keys=[]` ⇒ CLI 打印
"✓ tombstoned … 0 turn tombstones"。真跑证据（2026-09-22 生产日志）：`p-e479f97c661a`
下四本各 `window_turns=0`，而墓碑前快照重算分别是 14 / 45 / 19 / 8。

它一直被掩盖是因为 project 识别此前恒空、全落 Global，窗口"碰巧"对得上——**project
识别修好之后这条反而失效了**（原则 13 层 4：字段恒有值、值还好看，但值不来自它声称
的来源）。

# 本文件钉死

  ① 有 project 身份的轮次进得了窗口（主体修复）；
  ② 🔴 阴性对照：Global（空 project）那条路一字不改；
  ③ 🔴 反向变异：把 project 写回空串 ⇒ ① 必须红（判别力自证）；
  ④ 跨 project 不串台：同一 agent 在别的 project 的轮次不进本窗口；
  ⑤ `--dry-run` 一条墓碑不写，且 `window_turns` 与真跑**逐字相等**（尺子与扳机同源）；
  ⑥ 不加 `--dry-run` 时行为一字不变（阴性对照：别把默认路径改成只读）；
  ⑦ 备份软闸只告警不阻断，没有备份 ⇒ `None`（NO-DATA ≠ 0）。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import ActivationTable, activation_scope
from bladex_proxy import ledger_memory as _lm
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    SessionIdSource,
    TombstoneTargetType,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub

_L = "ldg-t0scope0001"
_AGENT = "claude-code"
_PROJ = "p-t0e479f97c66"
_OTHER_PROJ = "p-t0otherproj"


def _put(hub: MemoryHub, seq: int, text: str, *, agent: str = _AGENT, project: str = "") -> str:
    ident = Identity(user_id="u1", agent_id=agent, session_id="s1", turn_index=seq,
                     session_id_source=SessionIdSource.FINGERPRINT,
                     project_id=project, project_name=project)
    turn = Turn(identity=ident, model="test", request_messages=[{"role": "user", "content": text}],
                response_text="好的。", status=TurnStatus.OK, ts=datetime.now(UTC),
                session_id_source=SessionIdSource.FINGERPRINT)
    key = ident.storage_key(f"{int(datetime.now(UTC).timestamp() * 1000)}-0")
    hub.put(key, turn)
    time.sleep(0.02)
    return key


def _seed(hub: MemoryHub, *, project: str) -> dict[str, list[str]]:
    """建本 + 切入（scope = agent␟project），窗口内三类轮次各一条。"""
    led = new_ledger(ledger_id=_L, title="T0 scope 账本", goal="g")
    led.created_at = datetime.now(UTC).isoformat()
    hub.append_admin_event(AdminEventType.LEDGER_CREATE, "", ledger=led.model_dump(mode="json"))
    time.sleep(0.02)
    hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id=_AGENT,
                           project_id=project, session_id="s1", turn_index=1, to_ledger_id=_L)
    time.sleep(0.02)
    inside = [_put(hub, 1, "窗口内第一轮", project=project),
              _put(hub, 2, "窗口内第二轮", project=project)]
    cross = _put(hub, 3, "同一 agent，另一个 project", project=_OTHER_PROJ)
    other_agent = _put(hub, 4, "另一个 agent", agent="codex", project=project)
    return {"inside": inside, "cross": [cross], "other_agent": [other_agent]}


@pytest.fixture
def hub_with_project(tmp_path: Path):
    hub = MemoryHub(tmp_path / "hub")
    hub.open()
    keys = _seed(hub, project=_PROJ)
    yield hub, keys
    hub.close()


@pytest.fixture
def hub_global(tmp_path: Path):
    """阴性对照夹具：本账本在 Global 上（= project 识别接通之前的存量形态）。

    🔴 批 Y · Y1a：先写一条**别的 agent 在别的 project 上**的切换，表示"运行时已按
    (agent, project) 分键"。`turn_active_ledger` 以事件流里第一条非 Global 激活事件为
    project 维生效时刻（原则 13 层 4，08-29 live 病例见 `project_scope_since`）；没有这条，
    `cross` 那轮的 project 会被按"尚未生效"读成 Global——而运行时一旦分键，同一 agent
    带 project 来就会触发作用域迁移、留下带 project 的事件，"分键了却一条 project 事件都没有"
    在今天的运行时里不可能出现。"""
    hub = MemoryHub(tmp_path / "hub_global")
    hub.open()
    hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id="pi-cutover",
                           project_id="p-t0cutover00", session_id="s0", to_ledger_id="ldg-t0cutover0")
    time.sleep(0.02)
    keys = _seed(hub, project="")
    yield hub, keys
    hub.close()


# ── ①④ 主体修复 ─────────────────────────────────────────────────────────────

def test_1_window_finds_turns_with_project_identity(hub_with_project) -> None:
    hub, keys = hub_with_project
    w = _lm.ledger_window_turn_keys(hub, _L)
    assert w["scopes"] == [activation_scope(_AGENT, _PROJ)]
    assert set(w["turn_keys"]) == set(keys["inside"]), "🔴 MQ-I22：有 project 身份的轮次必须进窗口"


def test_4_other_project_and_other_agent_stay_out(hub_with_project) -> None:
    hub, keys = hub_with_project
    got = set(_lm.ledger_window_turn_keys(hub, _L)["turn_keys"])
    assert not (got & set(keys["cross"])), "同一 agent 的别的 project 不得串台"
    assert not (got & set(keys["other_agent"])), "别的 agent 不得串台"


# ── ② 阴性对照：Global 那条路不许改坏 ────────────────────────────────────────

def test_2_global_scope_path_unchanged(hub_global) -> None:
    """轮次无 project ⇒ `activation_scope` 落 Global，与历史窗口对得上（正确行为，不是兜底）。

    live 复算佐证（墓碑前快照）：Global 窗口的两本 `ldg-5cc65426473e` 仍是 **37**，
    `ldg-37696c99a521` 由 264 → **219**。
    """
    hub, keys = hub_global
    w = _lm.ledger_window_turn_keys(hub, _L)
    assert w["scopes"] == [activation_scope(_AGENT, "")]
    assert set(w["turn_keys"]) == set(keys["inside"]), \
        "project 落空的轮次照旧全进 Global 窗口（存量形态零变化）"


def test_2b_global_window_no_longer_swallows_other_projects(hub_global) -> None:
    """🔴 Global 窗口**不再**吃掉同 agent 在别的 project 的轮次 —— 这是修复的一部分。

    live 读数：`ldg-37696c99a521`（codex␟Global）264 → 219，少掉的 **45 条与
    `ldg-88a44fea09c8`（codex␟p-e479f97c661a）的窗口逐条相等**（差集 == 该窗口，
    project 分布 100% `p-e479f97c661a`）。修复前那 45 条被记在**错的账本**名下。
    """
    hub, keys = hub_global
    got = set(_lm.ledger_window_turn_keys(hub, _L)["turn_keys"])
    assert not (got & set(keys["cross"])), \
        "有 project 身份的轮次落进 Global 窗口 = 修复前的误归属"


# ── ③ 反向变异：判别力自证 ───────────────────────────────────────────────────

def test_3_reverse_mutation_blank_project_breaks_it(hub_with_project, monkeypatch) -> None:
    """把 project 改回空串（= 修复前的写法）⇒ ① 必须红。

    批 Y · Y1a 起 `ledger_memory` 不再自己拼 scope，改调 `turn_active_ledger`（单实现点）
    ⇒ 变异打在它的 project 参数上，语义与原变异逐字相同。"""
    real = _lm.turn_active_ledger
    monkeypatch.setattr(_lm, "turn_active_ledger",
                        lambda tl, key, project, **kw: real(tl, key, "", **kw))
    assert _lm.ledger_window_turn_keys(hub_with_project[0], _L)["turn_keys"] == [], \
        "变异后仍能查中 ⇒ 判据没有判别力"


# ── ⑤⑥ dry-run ──────────────────────────────────────────────────────────────

class _StubAgency:
    def __init__(self, pool):
        self.pool = pool
        self.activation = ActivationTable()
        self.activation.restore({activation_scope(_AGENT, _PROJ): _L})


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    hub_path = tmp_path / "hub"
    hub = MemoryHub(hub_path)
    hub.open()
    keys = _seed(hub, project=_PROJ)
    hub.close()
    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(hub_path))
    monkeypatch.setenv("BLADEX_INDEX_PATH", str(tmp_path / "index"))
    app = create_app(ProxyConfig())
    with TestClient(app) as c:
        c.app.state.agency = _StubAgency({_L: new_ledger(ledger_id=_L, title="t", goal="g")})
        yield c, hub_path, keys


def _tombstone_count(hub_path: Path) -> int:
    hub = MemoryHub(hub_path, read_only=True)
    hub.open()
    try:
        return sum(1 for _k, _t in hub.scan_tombstones())
    finally:
        hub.close()


def test_5_dry_run_writes_nothing_and_matches_the_real_run(client) -> None:
    c, hub_path, keys = client
    before = _tombstone_count(hub_path)
    r = c.request("DELETE", f"/admin/ledgers/{_L}",
                  params={"with_memory": "true", "dry_run": "true"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["status"] == "dry_run" and d["dry_run"] is True
    assert d["window_turns"] == len(keys["inside"]) == 2
    assert "memory" not in d and "tombstone_key" not in d
    assert _tombstone_count(hub_path) == before, "🔴 dry-run 写了墓碑"
    assert _StubAgency and _L in c.app.state.agency.pool, "dry-run 不许动池"
    # 同一本账本再来一次 ⇒ 逐字相等（幂等 + 只读自证）
    d2 = c.request("DELETE", f"/admin/ledgers/{_L}",
                   params={"with_memory": "true", "dry_run": "true"}).json()
    assert d2["window_turns"] == d["window_turns"]
    # 🔴 尺子与扳机同源：真跑的 window_turns 必须等于 dry-run 的
    real = c.request("DELETE", f"/admin/ledgers/{_L}", params={"with_memory": "true"}).json()
    assert real["window_turns"] == d["window_turns"], "尺子与扳机不同源，比没有 dry-run 更糟"
    assert real["memory"]["turn_tombstones"] == d["window_turns"]


def test_6_default_path_unchanged_without_dry_run(client) -> None:
    """阴性对照：不加 `--dry-run` 时照常写墓碑（别把默认路径改成只读）。"""
    c, hub_path, keys = client
    before = _tombstone_count(hub_path)
    d = c.request("DELETE", f"/admin/ledgers/{_L}", params={"with_memory": "true"}).json()
    assert d["status"] == "tombstoned" and d["tombstone_key"]
    assert d["memory"]["turn_tombstones"] == 2
    assert _tombstone_count(hub_path) == before + 3, "1 条 LEDGER 墓碑 + 2 条 TURN 墓碑"
    hub = MemoryHub(hub_path, read_only=True)
    hub.open()
    try:
        turn_tombs = {t.target_key for _k, t in hub.scan_tombstones()
                      if t.target_type == TombstoneTargetType.TURN}
    finally:
        hub.close()
    assert turn_tombs == set(keys["inside"])


# ── ⑦ 备份软闸 ──────────────────────────────────────────────────────────────

def test_7_backup_age_is_none_when_there_is_no_backup(client) -> None:
    c, hub_path, _keys = client
    d = c.request("DELETE", f"/admin/ledgers/{_L}", params={"dry_run": "true"}).json()
    assert d["backup_age_days"] is None, "没有备份 ⇒ NO-DATA，不许当 0"


def test_7b_backup_age_reads_the_newest_backup_dir(client) -> None:
    c, hub_path, _keys = client
    (hub_path.parent / "backup_old").mkdir()
    fresh = hub_path.parent / "backup_fresh"
    fresh.mkdir()
    import os
    os.utime(hub_path.parent / "backup_old", (time.time() - 40 * 86400,) * 2)
    d = c.request("DELETE", f"/admin/ledgers/{_L}", params={"dry_run": "true"}).json()
    assert d["backup_age_days"] is not None and d["backup_age_days"] < 1.0, \
        "取最近一次，不是最早一次"


def test_7c_soft_gate_never_blocks(client) -> None:
    """软闸 = 只告警不阻断：备份再旧也照样执行。"""
    c, hub_path, _keys = client
    old = hub_path.parent / "backup_ancient"
    old.mkdir()
    import os
    os.utime(old, (time.time() - 400 * 86400,) * 2)
    r = c.request("DELETE", f"/admin/ledgers/{_L}", params={"with_memory": "true"})
    assert r.status_code == 200 and r.json()["status"] == "tombstoned"
    assert r.json()["backup_age_days"] > 300


# ── ⑧ `facts_deleted` 桶可达性（判据 ⑦）────────────────────────────────────

def _fact(turn_key: str, content: str):
    from bladex_core.fact import Fact
    f = Fact(content=content, source_ledger_key=turn_key)
    f.id = f"t0-{abs(hash(content)) % 10**12}"
    return f


def test_8_facts_deleted_bucket_is_reachable_on_a_writable_index(tmp_path: Path) -> None:
    """🔴 `index_cascade_delete_done.facts_deleted` 曾**从无非零读数** ⇒ 先证明桶可达。

    live 佐证（2026-09-22 隔离 Index 快照 + 真 `tombstone_ledger_memory`）：
    一条 27 条 fact 的轮次 ⇒ `facts_deleted=27  total_removed=44`。
    """
    from bladex_proxy.storage.memory_index import MemoryIndex

    index = MemoryIndex(tmp_path / "idx", read_only=False)
    index.open()
    hub = MemoryHub(tmp_path / "hub")
    hub.open()
    try:
        key = "u1/claude-code/s1/1790000000000-0"
        for i in range(3):
            index.add_fact(_fact(key, f"事实 {i}"))
        assert sum(1 for f in index.all_facts() if f.source_ledger_key == key) == 3
        res = _lm.tombstone_ledger_memory(hub, index, [key])
        assert res == {"turn_tombstones": 1, "index_cascaded": 1, "index_deferred": 0}
        assert sum(1 for f in index.all_facts() if f.source_ledger_key == key) == 0, \
            "级联报成功却一条没删 ⇒ 又一个'0 长得像成功'"
    finally:
        index.close()
        hub.close()


def test_8b_readonly_index_defers_instead_of_reporting_a_silent_zero(tmp_path: Path) -> None:
    """阴性对照 = 生产形态：proxy 里的 Index 只读 ⇒ 有 fact 的轮次必须走 `deferred`。

    这解释了生产日志里 210 条 `facts_deleted=0`：它们是**没有 fact** 的轮次；
    真有 fact 的 91 条全部报 `index_cascade_delete_failed`（"Not supported operation in
    secondary mode."）。⇒ proxy 路径上 `facts_deleted` 结构上只能是 0，非零读数只可能
    来自可写 Index（重建 / 离线），见上一条。
    """
    from bladex_proxy.storage.memory_index import MemoryIndex

    key = "u1/claude-code/s1/1790000000001-0"
    writer = MemoryIndex(tmp_path / "idx2", read_only=False)
    writer.open()
    writer.add_fact(_fact(key, "只读对照用的事实"))
    writer.close()

    index = MemoryIndex(tmp_path / "idx2", read_only=True)
    index.open()
    hub = MemoryHub(tmp_path / "hub2")
    hub.open()
    try:
        res = _lm.tombstone_ledger_memory(hub, index, [key])
        assert res["turn_tombstones"] == 1
        assert res["index_deferred"] == 1 and res["index_cascaded"] == 0, \
            "只读 Index 上有 fact 的轮次必须是 deferred，不许静默算成功"
    finally:
        index.close()
        hub.close()


# ── ⑨ 池事件不得进时间轴（T0.2 的承重注释，`load_ledger_events`）───────────

def test_9_kept_pool_create_events_carry_no_scope(tmp_path: Path) -> None:
    """🔴 `keep_tombstoned_pool=True` 把被墓碑账本的 create 补回池，**不得**顺手把窗口开回来。

    `switch_timeline` 现在也认「建本即激活」，所以"池事件与时间轴无关"这句话从本批起
    是承重的：补回时剥掉 `agent_id` / `project_id`，时间轴的判据仍只在
    `apply_ledger_tombstones` 一处。
    """
    from bladex_core.ledger import ledger_activation_windows, switch_timeline
    from bladex_proxy.ledger_events import load_ledger_events
    from bladex_proxy.models import TombstoneSource, TombstoneTargetType

    hub = MemoryHub(tmp_path / "hub9")
    hub.open()
    try:
        led = new_ledger(ledger_id=_L, title="T0 池事件", goal="g")
        hub.append_admin_event(AdminEventType.LEDGER_CREATE, "",
                               ledger=led.model_dump(mode="json"),
                               agent_id=_AGENT, project_id=_PROJ)
        time.sleep(0.02)
        hub.append_tombstone(target_type=TombstoneTargetType.LEDGER, target_key=_L,
                             source=TombstoneSource.USER)
        events = load_ledger_events(hub, keep_tombstoned_pool=True)
        creates = [e for e in events if e["type"] == "ledger_create"]
        assert len(creates) == 1, "池里要留着它（重建锚层取标题）"
        assert "agent_id" not in creates[0]["payload"], "补回池的 create 不许带 scope"
        assert ledger_activation_windows(switch_timeline(events), _L) == {}, \
            "被墓碑账本的窗口被 create 顶回来了"
        # 判别力：不走 keep_tombstoned_pool 时本来就没有这条 create
        assert [e for e in load_ledger_events(hub) if e["type"] == "ledger_create"] == []
    finally:
        hub.close()
