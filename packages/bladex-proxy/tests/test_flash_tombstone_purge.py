"""MQ-F3（2026-09-18）：墓碑之后 Flash 投影文件必须不可读——删除语义兑现到文件面。

现场（09-11，FLASH 开着）：墓碑两本 T3 账本，`bladex ledger list` 回到 19 本（Hub 正确），
但 `data/bladex_flash/.../ledgers/c1/f1/ldg-c1f1fb7cc44c.md` 仍在 ⇒ 已删的本仍可被模型与
用户读到；`bladex status` 的 "N ledgers" 数投影文件，把已删的也数进去（09-14 再现）。
ADR-0009 Tombstone 承诺「P2 重建不得复活已删数据」，文件投影层同理。

DoD 四条各一用例：① 墓碑之后投影文件不可读；② status 计数不含已删；③ 阴性对照
（去掉 `run_once` 里的 `_purge_tombstoned` 调用 ⇒ ①② 红）；④ 不误删活着的本（正向用例，
含「池装载降级为空」这个最危险的形态）。外加红线 2：用户直编过的投影宁留勿删。
"""

from __future__ import annotations

import os

from bladex_core.flash import summarize_tree
from bladex_core.ledger import ledger_local_path, ledger_md_path, new_ledger

from bladex_proxy.flash_daemon import FlashDaemon, hub_tombstoned_source


def _led(lid: str, title: str):
    return new_ledger(ledger_id=lid, title=title, goal=f"goal of {title}",
                      goal_source="user", created_at="2026-09-18T00:00:00Z")


def _mk(tmp_path, holder: dict):
    return FlashDaemon(
        root=str(tmp_path), principal="u1",
        ledger_source=lambda: (holder["pool"], {}),
        tombstoned_source=lambda: holder["tomb"],
        emit_admin_event=lambda t, p: None)


def _p(tmp_path, lid: str) -> str:
    return ledger_md_path(str(tmp_path), "u1", lid)


def test_tombstoned_projection_becomes_unreadable(tmp_path):
    """DoD ①：墓碑之后投影文件不存在，分桶目录也不留空壳。"""
    holder = {"pool": {"ldg-a1": _led("ldg-a1", "甲"), "ldg-b2": _led("ldg-b2", "乙")},
              "tomb": set()}
    d = _mk(tmp_path, holder)
    assert d.run_once()["written"] == 2
    assert os.path.isfile(_p(tmp_path, "ldg-b2"))

    holder["pool"] = {"ldg-a1": holder["pool"]["ldg-a1"]}     # Hub 过滤后的池
    holder["tomb"] = {"ldg-b2"}
    d.push()
    out = d.run_once()

    assert out["removed"] == 1
    assert not os.path.exists(_p(tmp_path, "ldg-b2")), "墓碑之后投影仍可读（MQ-F3）"
    assert not os.path.isdir(os.path.dirname(_p(tmp_path, "ldg-b2")))
    assert os.path.isfile(_p(tmp_path, "ldg-a1")), "活着的本被误删"


def test_status_count_excludes_tombstoned(tmp_path):
    """DoD ②：`bladex status` 那行数的是 `summarize_tree(...).ledgers`——删了文件它就对了。"""
    holder = {"pool": {"ldg-a1": _led("ldg-a1", "甲"), "ldg-b2": _led("ldg-b2", "乙")},
              "tomb": set()}
    d = _mk(tmp_path, holder)
    d.run_once()
    assert summarize_tree(str(tmp_path)).ledgers == 2
    holder["pool"] = {"ldg-a1": holder["pool"]["ldg-a1"]}
    holder["tomb"] = {"ldg-b2"}
    d.push()
    d.run_once()
    assert summarize_tree(str(tmp_path)).ledgers == 1


def test_live_ledgers_survive_even_when_pool_degrades_to_empty(tmp_path):
    """DoD ④：判据是墓碑集合，不是「不在池里」——池装载降级为空**不许**变成全删。"""
    holder = {"pool": {"ldg-a1": _led("ldg-a1", "甲"), "ldg-b2": _led("ldg-b2", "乙")},
              "tomb": set()}
    d = _mk(tmp_path, holder)
    d.run_once()
    holder["pool"] = {}            # 装载降级（catch_up 失败 / Hub 短暂读不到）
    d.push()
    out = d.run_once()
    assert out["removed"] == 0
    assert os.path.isfile(_p(tmp_path, "ldg-a1")) and os.path.isfile(_p(tmp_path, "ldg-b2"))


def test_tombstone_source_failure_removes_nothing(tmp_path):
    holder = {"pool": {"ldg-a1": _led("ldg-a1", "甲")}, "tomb": set()}

    def _boom():
        raise RuntimeError("hub down")

    d = FlashDaemon(root=str(tmp_path), principal="u1",
                    ledger_source=lambda: (holder["pool"], {}),
                    tombstoned_source=_boom, emit_admin_event=lambda t, p: None)
    d.run_once()
    holder["pool"] = {}
    d.push()
    assert d.run_once()["removed"] == 0
    assert os.path.isfile(_p(tmp_path, "ldg-a1"))


def test_user_edited_projection_is_kept_and_local_override_untouched(tmp_path):
    """红线 2：用户直编过的投影宁留勿删；`.local.md` 永不被机器碰。"""
    holder = {"pool": {"ldg-a1": _led("ldg-a1", "甲")}, "tomb": set()}
    d = _mk(tmp_path, holder)
    d.run_once()
    path = _p(tmp_path, "ldg-a1")
    local = ledger_local_path(str(tmp_path), "u1", "ldg-a1")
    with open(local, "w", encoding="utf-8") as f:
        f.write("# my notes\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n- 用户手加的一行\n")
    holder["pool"] = {}
    holder["tomb"] = {"ldg-a1"}
    d.push()
    # 直编先被捕获进 pending（emit 是 no-op ⇒ 永不收敛），随后 purge 必须跳过它
    out = d.run_once()
    assert out["removed"] == 0
    assert os.path.isfile(path), "用户直编过的投影被删了（红线 2）"
    assert os.path.isfile(local)


def test_hub_tombstoned_source_reads_hub_tombstones():
    """接线：源用的是 `ledger_events.tombstoned_ledger_ids`（与账本装载同一判据）。"""
    from bladex_proxy.ledger_events import LEDGER_TOMBSTONE_TYPE

    class _T:
        def __init__(self, k, tt):
            self.target_key, self.target_type = k, tt

    class _Hub:
        def catch_up(self):
            pass

        def scan_tombstones(self):
            yield "k1", _T("ldg-dead", LEDGER_TOMBSTONE_TYPE)
            yield "k2", _T("fact-x", "fact")

    assert hub_tombstoned_source(_Hub())() == {"ldg-dead"}


def test_main_wires_tombstoned_source():
    """接线守卫：进程入口必须把墓碑源传给 daemon，否则机制在 live 上不存在。"""
    from _source_probe import source_of

    from bladex_proxy import flash_daemon
    src = source_of(flash_daemon.main)
    assert "tombstoned_source=hub_tombstoned_source(hub)" in src
