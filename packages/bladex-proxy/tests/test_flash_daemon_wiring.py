"""V-F2 启用卡：flash 维护进程的 Hub 接线件。

骨架（单写者物化/直编捕获/收敛）已有 test_flash_daemon.py 守着；本文件钉接线：
① `hub_ledger_source`：追新 + 按 core `LEDGER_EVENT_TYPES` 过滤 + replay；
② emit 失败**不吞**——快照不动、下轮重发（用户直编绝不静默丢）；
③ `admin_emit`：POST 正门形态（路径/鉴权头/payload；非 200 抛出）；
④ 事件集单一真相源（agency 与 daemon 同一份，§3.2b）。
"""

from __future__ import annotations

import os

from bladex_core.ledger import Ledger
from bladex_proxy.flash_daemon import FlashDaemon, admin_emit, hub_ledger_source


class _Ev:
    def __init__(self, etype, payload):
        self.event_type = etype
        self.payload = payload


class _FakeHub:
    def __init__(self, events):
        self.events = events
        self.catch_ups = 0

    def catch_up(self):
        self.catch_ups += 1

    def scan_admin_events(self):
        yield from (("k", e) for e in self.events)


def _led(lid="ldg-wire0000001", title="接线测试"):
    return Ledger(ledger_id=lid, title=title)


class TestHubLedgerSource:
    def test_filters_and_replays(self):
        led = _led()
        hub = _FakeHub([
            _Ev("ledger_create", {"ledger": led.model_dump()}),
            _Ev("scope_promote", {"whatever": 1}),      # 非账本事件必须被滤掉
        ])
        source = hub_ledger_source(hub)
        pool, _bindings = source()
        assert list(pool) == ["ldg-wire0000001"]
        assert pool["ldg-wire0000001"].title == "接线测试"
        assert hub.catch_ups == 1, "每次取投影前必须先追新（secondary 语义）"
        source()
        assert hub.catch_ups == 2

    def test_catch_up_failure_degrades_to_stale(self):
        led = _led()

        class _Flaky(_FakeHub):
            def catch_up(self):
                raise RuntimeError("primary busy")

        pool, _ = hub_ledger_source(_Flaky([
            _Ev("ledger_create", {"ledger": led.model_dump()})]))()
        assert "ldg-wire0000001" in pool, "追新失败要退到旧快照，不许炸"


class TestEmitFailureRetry:
    def test_failed_emit_is_retried_next_round(self, tmp_path):
        led = _led()
        emitted: list[dict] = []
        fail = {"on": True}

        def emit(etype, payload):
            if fail["on"]:
                raise RuntimeError("proxy down")
            emitted.append(payload)

        d = FlashDaemon(root=str(tmp_path), principal="local",
                        ledger_source=lambda: ({led.ledger_id: led}, {}),
                        emit_admin_event=emit)
        d.run_once()                        # 首轮物化
        [path] = [os.path.join(r, f) for r, _dirs, fs in os.walk(tmp_path)
                  for f in fs if f.endswith(".md")]
        with open(path, encoding="utf-8") as f:
            content = f.read()
        with open(path, "w", encoding="utf-8") as f:
            f.write(content.replace("接线测试", "接线测试（用户改）"))

        stats = d.run_once()                # emit 失败
        assert stats["user_edits"] == 0 and not emitted, "失败不算捕获成功"
        fail["on"] = False
        stats = d.run_once()                # 重试成功
        assert stats["user_edits"] == 1 and len(emitted) == 1
        assert emitted[0]["ledger"]["ledger_id"] == led.ledger_id


class TestAdminEmit:
    def _capture(self, monkeypatch, status=200):
        seen = {}

        class _Resp:
            def __init__(self):
                self.status = status

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            seen["headers"] = dict(req.header_items())
            seen["body"] = req.data
            return _Resp()

        import urllib.request
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        return seen

    def test_posts_to_the_front_door_with_auth(self, monkeypatch):
        seen = self._capture(monkeypatch)
        admin_emit("http://127.0.0.1:38080", "adm-key")(
            "ledger_user_edit", {"ledger": {"ledger_id": "ldg-x"}})
        assert seen["url"].endswith("/admin/ledgers/user-edit")
        assert seen["headers"].get("Authorization") == "Bearer adm-key"
        assert b"ldg-x" in seen["body"]

    def test_personal_mode_no_key_no_header(self, monkeypatch):
        seen = self._capture(monkeypatch)
        admin_emit("http://127.0.0.1:38080", "")("ledger_user_edit", {"ledger": {}})
        assert "Authorization" not in seen["headers"], \
            "未配 admin key 不带鉴权头（require_admin_key 个人模式回落）"

    def test_non_200_raises(self, monkeypatch):
        self._capture(monkeypatch, status=503)
        import pytest
        with pytest.raises(RuntimeError):
            admin_emit("http://x", "")("ledger_user_edit", {"ledger": {}})


def test_event_types_single_source():
    """§3.2b：账本事件的装载必须只有一处。

    🔴 **2026-09-01 判据加强**（`task-ledger-tombstone-cleanup-20260901.md` §2.3）：
    原断言是 `agency._LEDGER_EVENT_TYPES is LEDGER_EVENT_TYPES` —— 它只钉了
    agency 与 daemon 用同一份**事件类型集**，钉不住**第三处**：
    重建锚层当时自己写着 `_et.startswith("ledger_")`，两处注释都写着
    "各抄一份迟早分叉"，而分叉早就发生了、守卫看不见。

    现在三处都走 `ledger_events.load_ledger_events`（类型判据 + 账本墓碑 + ts
    各一处），故判据改为 **fail-closed 的接线断言**：谁扫 Hub 取账本事件，
    谁就必须经这个入口；新增第四个消费方默认红。
    """
    import pathlib

    from bladex_core.ledger import LEDGER_EVENT_TYPES
    from bladex_proxy import ledger_events

    # ① 唯一入口自己用 core 的单一真相源
    assert ledger_events.LEDGER_EVENT_TYPES is LEDGER_EVENT_TYPES

    # ② 三处装载点都调它（**按源码断言**，不靠运行期覆盖）
    root = pathlib.Path(ledger_events.__file__).parent
    call_sites = {
        "agency.py": root / "agency.py",
        "flash_daemon.py": root / "flash_daemon.py",
        "storage/memory_index.py": root / "storage" / "memory_index.py",
    }
    for name, path in call_sites.items():
        src = path.read_text(encoding="utf-8")
        assert "load_ledger_events" in src, f"{name} 没有经唯一入口装载账本事件"

    # ③ fail-closed：除唯一入口外，没有第二处自己扫 `ledger_` 事件
    #    （下界断言同时防守卫自身漂移——扫不到文件时不能静默通过）
    scanned = 0
    for path in root.rglob("*.py"):
        if path.name == "ledger_events.py":
            continue
        scanned += 1
        src = path.read_text(encoding="utf-8")
        assert 'startswith("ledger_")' not in src, (
            f"{path.name} 自己判定账本事件 —— 判据必须取 core 的 "
            f"LEDGER_EVENT_TYPES，装载必须经 load_ledger_events")
    assert scanned >= 10, f"只扫到 {scanned} 个文件，守卫自己漂了"
