"""Z4b：consolidator 退避 sleep 的第二个唤醒源——外部回流目标（Obsidian vault / Postgres 收件表）有变化。

判据（prompt-batch-Z4b）在本文件的落点：
  ① 退到 1800 s 上限后改 vault 一行 ⇒ 一个 1 s 节拍内跑回流、POST 出 correct
     —— `test_vault_edit_at_cap_is_backflowed_within_one_tick`；拆掉唤醒源 ⇒ 红
     （`test_without_the_wake_source_the_edit_waits_the_full_sleep` 是同一场景的阴性对照）
  ② 导出器自写不唤醒自己 —— `test_exporter_self_writes_do_not_wake`（基线取早 ⇒ 红）
     + `test_one_edit_is_one_external_round`（外部一轮不取基线 ⇒ 每秒都被唤醒 ⇒ 红）
  ③ Postgres：`inbox` 关零 SQL；开时按旋钮限频；自己处理收件行不唤醒自己
  ④ 主循环接线守卫（批 X 续棒同款）—— `test_wake_source_is_wired_*`
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.exporters import ExportBatch
from bladex_core.fact import Fact
from bladex_core.matter import Matter, MatterOrigin, MatterStatus
from bladex_proxy.consolidator import IdleBackoff, sleep_until_work
from bladex_proxy.export_backflow import AdminBackflowClient
from bladex_proxy.export_sync import ExporterConfig, ExportSyncWorker, ExternalChangeProbe
from bladex_proxy.exporters.obsidian import ObsidianExporter
from test_export_backflow import _pg, _PgState

_REPO = Path(__file__).resolve().parents[3]
_CONSOLIDATOR = _REPO / "packages/bladex-proxy/bladex_proxy/consolidator.py"


def _fact(fid: str, content: str, mid: str = "m1") -> Fact:
    return Fact(id=fid, content=content, source_user_id="u1", matter_id=mid,
                created_at=datetime.now(UTC))


def _vault(tmp_path: Path, **opts) -> tuple[ObsidianExporter, Path]:
    root = tmp_path / "vault" / "BladeX"
    exp = ObsidianExporter("v", "private", {"vault_dir": str(root), **opts})
    exp.sync_incremental(ExportBatch(
        facts=[_fact("f1", "用户偏好中文"), _fact("f2", "部署在 arm64")],
        matters=[Matter(matter_id="m1", title="发布", status=MatterStatus.ACTIVE,
                        origin=MatterOrigin.AUTO)],
        hard_rules=["MUST be polite"]), "")
    return exp, root


def _clock():
    t = {"now": 0.0}

    def now() -> float:
        return t["now"]

    def sleep(s: float) -> None:
        t["now"] += max(s, 0.001)

    return now, sleep, t


def _at_cap() -> IdleBackoff:
    b = IdleBackoff(60.0, enabled=True, after=0, max_s=1800.0)
    for _ in range(10):
        b.observe(0)
    assert b.interval == 1800.0
    return b


def _edit(root: Path, old: str, new: str) -> None:
    p = root / "Matters/m1.md"
    p.write_text(p.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")


class _Idx:
    def all_facts(self):
        return []

    def all_matters(self):
        return []

    def get_edges(self, _):
        return []

    def get_matter(self, _):
        return None


# ═══ ① vault 改动在一个节拍内回流 ═══════════════════════════════════


def _scenario(tmp_path, *, wire_external: bool):
    exp, root = _vault(tmp_path)
    posts: list[tuple] = []
    client = AdminBackflowClient(lambda m, p, b: (posts.append((m, p, b)), (200, {}))[1])
    w = ExportSyncWorker(_Idx(), None, [(ExporterConfig("obsidian", "v"), exp)],
                         tmp_path / "st.json", backflow=client, max_deletes=5)
    w.run_once()
    probe = w.change_probe(pg_every_s=30)
    assert probe is not None
    probe.mark()
    now, sleep, t = _clock()
    b = _at_cap()
    rounds: list[tuple[float, str]] = []

    def external_round(src: str) -> None:
        rounds.append((now(), src))
        w.run_once(force=True)
        probe.mark()

    def poll_cmd():
        if 100.0 <= now() < 101.0 and not posts:
            _edit(root, "- 用户偏好中文 [[", "- 用户偏好简体中文 [[")
        return None

    out = sleep_until_work(b.interval, poll_cmd=poll_cmd, idle=b, sleep=sleep, now=now,
                           external=probe if wire_external else None,
                           external_round=external_round)
    return out, posts, rounds, b, t


def test_vault_edit_at_cap_is_backflowed_within_one_tick(tmp_path):
    """①：退到 1800 s 上限后第 100 s 改一行 ⇒ ≤1 个节拍内回流出 correct；
    sleep 照常睡完、退避阶梯不动（外部唤醒不改 interval）。"""
    out, posts, rounds, b, t = _scenario(tmp_path, wire_external=True)
    assert out is None
    assert len(rounds) == 1 and rounds[0][1] == "v"
    assert 100.0 <= rounds[0][0] <= 102.0, f"唤醒时刻 {rounds[0][0]}"
    assert [(m, p) for m, p, _ in posts] == [("POST", "/admin/facts/f1/correct")]
    assert posts[0][2]["content"] == "用户偏好简体中文"
    assert t["now"] >= 1800.0, "外部唤醒不该结束 sleep（整轮由 Hub 水位去叫）"
    assert b.interval == 1800.0, "外部唤醒动了退避阶梯"


def test_without_the_wake_source_the_edit_waits_the_full_sleep(tmp_path):
    """① 阴性对照：拆掉唤醒源 ⇒ 1800 s 里零回流（这就是 Z4 §10-3 的 30 分钟）。"""
    _, posts, rounds, _, t = _scenario(tmp_path, wire_external=False)
    assert rounds == [] and posts == []
    assert t["now"] >= 1800.0


def test_backoff_switch_off_also_disables_external_wake(tmp_path):
    """关 `IDLE_BACKOFF` = 逐字旧行为：外部唤醒一起关（批 X 判据④(a) 同款）。"""
    exp, root = _vault(tmp_path)
    probe = ExternalChangeProbe([(ExporterConfig("obsidian", "v"), exp)], pg_every_s=30)
    probe.mark()
    _edit(root, "- 用户偏好中文 [[", "- 改了 [[")
    now, sleep, _ = _clock()
    calls: list[str] = []
    sleep_until_work(60.0, idle=IdleBackoff(60.0, enabled=False, after=3, max_s=600),
                     external=probe, external_round=calls.append, sleep=sleep, now=now)
    assert calls == []


# ═══ ② 自写回声不唤醒 ══════════════════════════════════════════════


def test_exporter_self_writes_do_not_wake(tmp_path):
    """基线在导出写完之后取 ⇒ 导出器重写页面不算变化；用户改一行才算。"""
    exp, root = _vault(tmp_path)
    probe = ExternalChangeProbe([(ExporterConfig("obsidian", "v"), exp)], pg_every_s=30)
    probe.mark()
    assert probe.has_change() is None
    exp.sync_incremental(ExportBatch(facts=[_fact("f2", "部署在 arm64（Index 更新了）")]), "c")
    probe.mark()                      # ← 主循环：导出后 finally 里取基线
    assert probe.has_change() is None
    _edit(root, "- 用户偏好中文 [[", "- 用户改了 [[")
    assert probe.has_change() == "v"


def test_base_taken_before_export_would_wake_on_own_writes(tmp_path):
    """② 变异形状：基线取在导出**之前** ⇒ 自己写的页面把自己叫醒（这就是要防的事）。"""
    exp, _ = _vault(tmp_path)
    probe = ExternalChangeProbe([(ExporterConfig("obsidian", "v"), exp)], pg_every_s=30)
    probe.mark()
    exp.sync_incremental(ExportBatch(facts=[_fact("f2", "部署在 arm64（又更新了）")]), "c")
    assert probe.has_change() == "v"


def test_one_edit_is_one_external_round(tmp_path):
    """② 占空比守卫：一次编辑 = 一次外部轮，且外部轮自己的导出写不再触发下一次。

    外部轮若不重取基线（或导出后没取），1800 s 里会每秒被自己唤醒一次 ——
    等于退避退不上去。本例把那种形状钉成 `len(rounds) > 1` 的红。
    """
    exp, root = _vault(tmp_path)
    probe = ExternalChangeProbe([(ExporterConfig("obsidian", "v"), exp)], pg_every_s=30)
    probe.mark()
    now, sleep, _ = _clock()
    rounds: list[float] = []

    def external_round(_src: str) -> None:
        rounds.append(now())
        # 模拟一次导出重写（Index 里那条被更正了 ⇒ 页面被重写）
        exp.sync_incremental(ExportBatch(facts=[_fact("f1", f"更正 {len(rounds)}")]), "c")
        probe.mark()

    def poll_cmd():
        if 10.0 <= now() < 11.0 and not rounds:
            _edit(root, "- 用户偏好中文 [[", "- 改一次 [[")
        return None

    sleep_until_work(1800.0, poll_cmd=poll_cmd, idle=_at_cap(), sleep=sleep, now=now,
                     external=probe, external_round=external_round)
    assert len(rounds) == 1, f"一次编辑被唤醒了 {len(rounds)} 次"


def test_broken_base_is_disarmed_not_spun(tmp_path):
    """② 兜底：外部轮之后探针仍报变化（基线取不上）⇒ 本次 sleep 内只跑一轮、随即解除，
    不许每拍跑一轮（更不许不睡直接空转——反向变异 W2b 曾把测试进程卡死在那个形状上）。"""
    exp, root = _vault(tmp_path)
    probe = ExternalChangeProbe([(ExporterConfig("obsidian", "v"), exp)], pg_every_s=30)
    probe.mark()
    probe.mark = lambda: None                 # 基线永远取不上
    now, sleep, t = _clock()
    rounds: list[float] = []

    def external_round(_src: str) -> None:
        rounds.append(now())
        exp.sync_incremental(ExportBatch(facts=[_fact("f1", f"重写 {len(rounds)}")]), "c")

    _edit(root, "- 用户偏好中文 [[", "- 改 [[")
    sleep_until_work(1800.0, idle=_at_cap(), sleep=sleep, now=now,
                     external=probe, external_round=external_round)
    assert len(rounds) == 1 and t["now"] >= 1800.0


def _only_matters(root: Path) -> None:
    _edit(root, "- 用户偏好中文 [[", "- 只改 Matter 页 [[")


def _only_inbox(root: Path) -> None:
    p = root / "Inbox.md"
    p.write_text(p.read_text(encoding="utf-8") + "\n- 用户在 Inbox 里记了一笔\n", encoding="utf-8")


def _only_hard_rules(root: Path) -> None:
    p = root / "Hard Rules.md"
    p.write_text(p.read_text(encoding="utf-8") + "- NEVER use emojis\n", encoding="utf-8")


@pytest.mark.parametrize("touch", [_only_matters, _only_inbox, _only_hard_rules],
                         ids=["Matters/*.md", "Inbox.md", "Hard Rules.md"])
def test_each_managed_file_kind_wakes_on_its_own(tmp_path, touch):
    """三类受管文件各自独立地能叫醒退避 sleep——任何一类从探针清单里掉了都要红。

    复核方 09-23 的变异：把受管清单里的 `Hard Rules.md` 拿掉，此前 15 条全绿 ——
    而 Z1.2「Obsidian 改硬规则随热加载生效」恰好靠这一条；拆了它，改硬规则又回到
    最坏 30 分钟，测试不报。所以每一类只改它自己、其余两类一字不动。
    """
    exp, root = _vault(tmp_path)
    inbox = root / "Inbox.md"
    if not inbox.is_file():  # 夹具的 fact 全归 m1，导出器不会生成 Inbox —— 先有它再取基线
        inbox.write_text("# BladeX Inbox\n\n## Facts\n\n", encoding="utf-8")
    probe = ExternalChangeProbe([(ExporterConfig("obsidian", "v"), exp)], pg_every_s=30)
    probe.mark()
    now, sleep, _ = _clock()
    rounds: list[float] = []

    def external_round(_src: str) -> None:
        rounds.append(now())
        probe.mark()

    def poll_cmd():
        if 200.0 <= now() < 201.0 and not rounds:
            touch(root)
        return None

    sleep_until_work(1800.0, poll_cmd=poll_cmd, idle=_at_cap(), sleep=sleep, now=now,
                     external=probe, external_round=external_round)
    assert len(rounds) == 1 and 200.0 <= rounds[0] <= 202.0, rounds


def test_touch_without_content_change_wakes_once_but_emits_nothing(tmp_path):
    """取舍的代价（结算稿 §11）：stat 签名 ⇒ 只 touch 也唤醒一次，但回流 0 条编辑。"""
    exp, root = _vault(tmp_path)
    probe = ExternalChangeProbe([(ExporterConfig("obsidian", "v"), exp)], pg_every_s=30)
    probe.mark()
    p = root / "Matters/m1.md"
    st = p.stat()
    import os
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert probe.has_change() == "v"
    assert exp.collect_edits(5) == []


def test_backflow_off_means_no_probe(tmp_path):
    exp, _ = _vault(tmp_path, backflow=False)
    w = ExportSyncWorker(_Idx(), None, [(ExporterConfig("obsidian", "v"), exp)],
                         tmp_path / "st.json", backflow=AdminBackflowClient(
                             lambda m, p, b: (200, {})), max_deletes=5)
    assert w.change_probe(pg_every_s=30) is None


# ═══ ③ Postgres 收件表 ══════════════════════════════════════════════


def _pg_probe(st: _PgState, **opts):
    exp = _pg(st, **opts)
    probe = ExternalChangeProbe([(ExporterConfig("postgres", "pg"), exp)], pg_every_s=30)
    return exp, probe


def test_pg_inbox_off_is_zero_sql_across_a_full_sleep():
    st = _PgState()
    exp = _pg(st)
    w = ExportSyncWorker(_Idx(), None, [(ExporterConfig("postgres", "pg"), exp)],
                         "/nonexistent/st.json", backflow=AdminBackflowClient(
                             lambda m, p, b: (200, {})), max_deletes=5)
    assert w.change_probe(pg_every_s=30) is None
    assert exp.inbox_has_new() is False
    assert st.sql == []


def test_pg_probe_is_rate_limited_and_wakes_within_the_period():
    """开着时：1800 s 退避里最多 1800/30+1 次 `SELECT max(id)`；插一行 ⇒ ≤30 s 内被叫醒。"""
    st = _PgState()
    exp, probe = _pg_probe(st, inbox=True)
    exp.collect_edits(5)                      # 首轮 collect：建表 + 取水位
    base_sql = len(st.sql)
    now, sleep, _ = _clock()
    probe._now = now
    woke: list[float] = []

    def poll_cmd():
        if 700.0 <= now() < 701.0 and not st.rows:
            st.insert(target_type="fact", target_id="f1", op="update", new_value="x")
        return None

    def external_round(_src: str) -> None:
        woke.append(now())
        exp.collect_edits(5)

    sleep_until_work(1800.0, poll_cmd=poll_cmd, idle=_at_cap(), sleep=sleep, now=now,
                     external=probe, external_round=external_round)
    probes = [s for s in st.sql[base_sql:] if s.startswith("SELECT COALESCE(MAX(id), 0)")
              and "pending" not in s]
    assert len(woke) == 1 and 700.0 <= woke[0] <= 731.0, woke
    assert len(probes) <= 1800 // 30 + 1 + 1, f"{len(probes)} 次探测——限频没生效"


def test_pg_own_processing_and_history_do_not_wake():
    """BladeX 回写收件行不改 max(id) ⇒ 不唤醒；只有历史已处理行、没有 pending ⇒ 不唤醒。"""
    st = _PgState()
    rid = st.insert(target_type="fact", target_id="f1", op="update", new_value="x")
    st.rows[rid - 1]["status"] = "accepted"          # 历史行，早已处理
    exp, _ = _pg_probe(st, inbox=True)
    assert exp.collect_edits(5) == []
    assert exp.inbox_has_new() is False
    st.insert(target_type="fact", target_id="f2", op="update", new_value="y")
    assert exp.inbox_has_new() is True
    (edit,) = exp.collect_edits(5)
    from bladex_proxy.export_backflow import BackflowResult
    exp.ack(edit, BackflowResult("accepted"))
    assert exp.inbox_has_new() is False


def test_pg_probe_before_first_collect_is_zero_sql():
    st = _PgState()
    exp = _pg(st, inbox=True)
    assert exp.inbox_has_new() is False and st.sql == []


# ═══ ④ 主循环接线守卫 ═══════════════════════════════════════════════


def test_wake_source_is_wired_into_the_sleep():
    src = _CONSOLIDATOR.read_text("utf-8")
    assert "external = export_worker.change_probe()" in src
    assert "external=external," in src
    assert "external_round=_external_round," in src


def test_wake_source_base_is_taken_after_the_export():
    """🔴 基线必须在 `export_worker.run_once()` 之后取（主循环 finally 里）。"""
    src = _CONSOLIDATOR.read_text("utf-8")
    loop = src[src.index("# T13：主管线之后跑导出"):]
    assert loop.index("export_worker.run_once()") < loop.index("external.mark()")
    ext = src[src.index("def _external_round"):]
    ext = ext[:ext.index("\n\n")]
    assert ext.index("export_worker.run_once(force=True)") < ext.index("external.mark()")
