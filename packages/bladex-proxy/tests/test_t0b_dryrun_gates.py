"""批 T0b · 三道闸（MQ-I24，2026-09-22）：把 `--dry-run` 这把**假的只读尺子**修可信。

# 事故回放（2026-09-22 16:37）

新 CLI（源码树，刚提交的 T0）打旧 proxy（15:39 启动，T0 之前的代码）。CLI 发
`?with_memory=true&dry_run=true`；那版 `delete_ledger` 没有 `dry_run` 这个参数，
**FastAPI 对未声明的 query 参数静默忽略** ⇒ 退化成普通 `with_memory=true` **真删**，
回 HTTP 200；CLI 只看了状态码，照 dry-run 的格式打出 `DRY RUN -- nothing written.`。
两本账本连派生记忆一起不可逆地没了，而终端说什么都没写。

🔴 刚性原则 13 层 4 的最纯形态：`"nothing written"` 这个值**不来自它声称的来源**
（服务端的 dry-run 分支），而来自 CLI 的一厢情愿。

# 三道闸，各杀一环

| 闸 | 杀的那一环 | 本文件 |
|---|---|---|
| **A** | CLI 不看服务端是否自认走了 dry-run 分支 | `test_a_*` |
| **C** | 服务端静默忽略未知参数（**根因**） | `test_c_*` |
| **B** | 没人问过"跑着的进程是不是这份代码" | `test_b_*` |

# 闸 A 的守卫经**真端点**，不是只测函数（MQ-W13 / CA15 的教训）

`test_a_negative_control_old_server_shape` 的阴性对照不是"造一个假响应"：它把
`dry_run` 参数**从请求里拿掉**再打真端点 —— 这正是旧 FastAPI 对新 CLI 做的事，
拿到的是真端点**真的执行了删除**之后的真实响应。CLI 必须在这个形状下红。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from bladex_core.ledger import new_ledger
from bladex_core.ledger_runtime import ActivationTable, activation_scope
from bladex_proxy.admin_contract import ADMIN_CONTRACT_FIELD, ADMIN_CONTRACT_VERSION
from bladex_proxy.models import (
    AdminEventType,
    Identity,
    SessionIdSource,
    Turn,
    TurnStatus,
)
from bladex_proxy.storage.memory_hub import MemoryHub

_L = "ldg-t0bgate00001"
_AGENT = "claude-code"
_PROJ = "p-t0bgate0001"


def _put(hub: MemoryHub, seq: int, text: str) -> str:
    ident = Identity(user_id="u1", agent_id=_AGENT, session_id="s1", turn_index=seq,
                     session_id_source=SessionIdSource.FINGERPRINT,
                     project_id=_PROJ, project_name="gate")
    turn = Turn(identity=ident, model="test", request_messages=[{"role": "user", "content": text}],
                response_text="好的。", status=TurnStatus.OK, ts=datetime.now(UTC),
                session_id_source=SessionIdSource.FINGERPRINT)
    key = ident.storage_key(f"{int(datetime.now(UTC).timestamp() * 1000)}-{seq}")
    hub.put(key, turn)
    time.sleep(0.02)
    return key


def _seed(hub: MemoryHub) -> list[str]:
    led = new_ledger(ledger_id=_L, title="T0b 闸测试账本", goal="g")
    led.created_at = datetime.now(UTC).isoformat()
    hub.append_admin_event(AdminEventType.LEDGER_CREATE, "", ledger=led.model_dump(mode="json"))
    time.sleep(0.02)
    hub.append_admin_event(AdminEventType.LEDGER_SWITCH, "", agent_id=_AGENT,
                           project_id=_PROJ, session_id="s1", turn_index=1, to_ledger_id=_L)
    time.sleep(0.02)
    return [_put(hub, 1, "窗口内第一轮"), _put(hub, 2, "窗口内第二轮")]


class _StubAgency:
    def __init__(self, pool):
        self.pool = pool
        self.activation = ActivationTable()
        self.activation.restore({activation_scope(_AGENT, _PROJ): _L})


@pytest.fixture
def live(tmp_path: Path, monkeypatch, isolated_home):
    """真端点 + 真 CLI：`_admin_call` 转发到 TestClient，`_probe_ready` 读真 /ready。"""
    import os

    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    hub_path = tmp_path / "hub"
    hub = MemoryHub(hub_path)
    hub.open()
    keys = _seed(hub)
    hub.close()
    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    monkeypatch.setenv("BLADEX_ROCKSDB_PATH", str(hub_path))
    monkeypatch.setenv("BLADEX_INDEX_PATH", str(tmp_path / "index"))
    app = create_app(ProxyConfig())
    with TestClient(app) as c:
        c.app.state.agency = _StubAgency({_L: new_ledger(ledger_id=_L, title="t", goal="g")})

        def transport(method: str, path: str, body: dict | None = None,
                      params: dict | None = None):
            r = c.request(method, path, params={k: v for k, v in (params or {}).items()
                                                if v not in ("", None)}, json=body)
            try:
                return r.status_code, r.json()
            except ValueError:
                return r.status_code, {}

        # CLI 跑在 tmp 目录、无 BLADEX_* 环境（同 test_cli_ledger 的隔离）。
        monkeypatch.chdir(tmp_path)
        for k in list(os.environ):
            if k.startswith("BLADEX_"):
                monkeypatch.delenv(k, raising=False)
        from bladex_proxy import cli as _cli
        from bladex_proxy.cli import ledger_cmds as _ldg
        from bladex_proxy.cli import memory_cmds as _mem
        monkeypatch.setattr(_ldg, "_admin_call", transport)
        monkeypatch.setattr(_mem, "_admin_call", transport)
        monkeypatch.setattr(_cli, "_probe_ready", lambda *_a, **_k: c.get("/ready").json())
        yield c, hub_path, keys, transport


def _tombstone_count(hub_path: Path) -> int:
    hub = MemoryHub(hub_path, read_only=True)
    hub.open()
    try:
        return sum(1 for _k, _t in hub.scan_tombstones())
    finally:
        hub.close()


# ══ 闸 A ═════════════════════════════════════════════════════════════════════

def test_a_happy_path_real_endpoint_acknowledges_dry_run(live, capsys) -> None:
    """正面：真端点自认 `status=dry_run` ⇒ CLI 照常渲染预览、退出码 0、零墓碑。"""
    from bladex_proxy import cli
    c, hub_path, keys, _t = live
    before = _tombstone_count(hub_path)
    assert cli.main(["ledger", "tombstone", _L, "--with-memory", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN -- nothing written." in out
    assert f"turns in window: {len(keys)}" in out
    assert _tombstone_count(hub_path) == before


def test_a_negative_control_old_server_shape(live, monkeypatch, capsys) -> None:
    """🔴 阴性对照 = 09-22 那个事故形态，**经真端点**复现。

    把 `dry_run` 从请求里拿掉（旧 FastAPI 对新 CLI 做的事就是这个）⇒ 真端点
    **真的删了**并回 `status=tombstoned`。CLI 必须：
      ① 不打印 "nothing written"；② 非零退出；③ 说清"真删可能已经发生了，去查"。
    """
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds as _ldg
    c, hub_path, keys, transport = live
    before = _tombstone_count(hub_path)

    def old_server(method: str, path: str, body=None, params=None):
        # 旧服务端不认识 dry_run ⇒ 静默忽略它。
        return transport(method, path, body, {k: v for k, v in (params or {}).items()
                                              if k != "dry_run"})

    monkeypatch.setattr(_ldg, "_admin_call", old_server)
    rc = cli.main(["ledger", "tombstone", _L, "--with-memory", "--dry-run"])
    assert rc == 2, "标记缺失必须非零退出（2 = 我不知道服务端做了什么）"
    out = capsys.readouterr().out
    assert "nothing written" not in out, "🔴 这正是 09-22 打出来的那句话"
    assert "REFUSING to report a dry run" in out
    assert "IRREVERSIBLE OPERATION MAY HAVE JUST RUN" in out
    # 而删除**确实**发生了 —— 阴性对照的判别力自证：闸 A 挡住的是"报告"，不是"删除"，
    # 删除本身要靠闸 B/C 在发出请求前就拦住。
    assert _tombstone_count(hub_path) > before


def test_a_marker_must_be_both_status_and_flag(capsys) -> None:
    """半个标记不算数：只有 `status` 或只有 `dry_run` 都必须红。"""
    import typer
    from bladex_proxy.cli import _require_dry_run_ack
    for payload in ({"status": "dry_run"}, {"dry_run": True}, {"status": "tombstoned"}, {}):
        with pytest.raises(typer.Exit) as exc:
            _require_dry_run_ack(payload, "x")
        assert exc.value.exit_code == 2
    _require_dry_run_ack({"status": "dry_run", "dry_run": True}, "x")  # 完整标记 ⇒ 不抛


# ══ 闸 C ═════════════════════════════════════════════════════════════════════

def test_c_unknown_query_param_is_400_and_names_it(live) -> None:
    """未知 query 参数 ⇒ 400，措辞**点名那个参数**，且什么都没写。"""
    c, hub_path, _keys, _t = live
    before = _tombstone_count(hub_path)
    r = c.request("DELETE", f"/admin/ledgers/{_L}",
                  params={"with_memory": "true", "dyr_run": "true"})
    assert r.status_code == 400, r.text
    assert "dyr_run" in r.text, "不点名就等于让人继续猜"
    assert _tombstone_count(hub_path) == before, "🔴 被拒的请求写了墓碑"


def test_c_negative_control_known_params_still_200(live) -> None:
    """阴性对照：合法参数照常 200（别把闸修成拒绝一切）。"""
    c, _hub_path, _keys, _t = live
    r = c.request("DELETE", f"/admin/ledgers/{_L}",
                  params={"with_memory": "true", "dry_run": "true"})
    assert r.status_code == 200 and r.json()["status"] == "dry_run"
    assert c.request("DELETE", f"/admin/ledgers/{_L}").status_code == 200


def test_c_covers_the_new_irreversible_endpoints(live) -> None:
    """路径①② 两条新端点同样不许静默忽略。"""
    c, _hub_path, _keys, _t = live
    r = c.request("DELETE", f"/admin/ledgers/{_L}/memory", params={"dry_runn": "true"})
    assert r.status_code == 400 and "dry_runn" in r.text
    r = c.post("/admin/turns/tombstone-by-project", params={"dry_run": "true"},
               json={"project_ids": [_PROJ]})
    assert r.status_code == 400 and "dry_run" in r.text, \
        "这条端点的 dry_run 在 body 里；放进 query 就是调用方记错了协议，必须报错"


# ══ 闸 B ═════════════════════════════════════════════════════════════════════

def test_b_ready_declares_the_contract(live) -> None:
    c, _hub_path, _keys, _t = live
    body = c.get("/ready").json()
    assert body[ADMIN_CONTRACT_FIELD] == ADMIN_CONTRACT_VERSION
    assert body["build"]


def test_b_negative_control_mismatched_identifier_refuses(live, monkeypatch, capsys) -> None:
    """🔴 伪造一个不匹配的标识 ⇒ **拒绝执行**（不是告警后照发），且一个请求都没发出去。"""
    from bladex_proxy import cli
    from bladex_proxy.cli import ledger_cmds as _ldg
    c, hub_path, _keys, _t = live
    sent: list = []
    monkeypatch.setattr(_ldg, "_admin_call", lambda *a, **k: sent.append(a) or (200, {}))
    for fake in ({"ready": True, ADMIN_CONTRACT_FIELD: ADMIN_CONTRACT_VERSION + 1},
                 {"ready": True, ADMIN_CONTRACT_FIELD: ADMIN_CONTRACT_VERSION - 1},
                 {"ready": True}):  # 旧 proxy 根本不声明它 —— 缺席与不等同罪
        monkeypatch.setattr(cli, "_probe_ready", lambda *_a, _f=fake, **_k: _f)
        assert cli.main(["ledger", "tombstone", _L, "--with-memory", "--dry-run"]) == 2
    assert sent == [], "🔴 闸 B 必须在发请求**之前**拦住，否则拦了个寂寞"
    out = capsys.readouterr().out
    assert "the running proxy is not this code" in out


def test_b_unreachable_proxy_refuses_too(live, monkeypatch) -> None:
    from bladex_proxy import cli
    monkeypatch.setattr(cli, "_probe_ready", lambda *_a, **_k: None)
    assert cli.main(["ledger", "tombstone", _L, "--dry-run"]) == 2


def test_b_gate_guards_all_three_irreversible_commands(live, monkeypatch) -> None:
    """三条不可逆命令都过闸 B —— 少挂一条就是留一个后门。"""
    from bladex_proxy import cli
    monkeypatch.setattr(cli, "_probe_ready", lambda *_a, **_k: {"ready": True})
    for argv in (["ledger", "tombstone", _L, "--dry-run"],
                 ["ledger", "tombstone-memory", _L, "--dry-run"],
                 ["memory", "tombstone-project", _PROJ, "--dry-run"]):
        assert cli.main(argv) == 2, argv


def test_b_contract_constant_is_single_sourced() -> None:
    """常量只有一个定义点：服务端与 CLI 都从 `admin_contract` 取（原则 12 的默认值判据同款）。"""
    import re
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[3] / "packages"
    pat = re.compile(r"^ADMIN_CONTRACT_VERSION\s*=", re.M)
    hits = sorted(str(f.relative_to(root)) for f in root.rglob("*.py")
                  if pat.search(f.read_text(encoding="utf-8", errors="ignore")))
    assert hits == ["bladex-proxy/bladex_proxy/admin_contract.py"], \
        f"ADMIN_CONTRACT_VERSION 不是单一定义点：{hits}"
