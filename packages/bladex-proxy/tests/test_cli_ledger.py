"""`bladex ledger` CLI 命令族验收（task-cc-ledger-cli-20260827）。

与 test_cli_memory_matter.py 同款做法：_admin_call monkeypatch 成录制桩，
CLI 不直捅存储、只打 admin HTTP API 的纪律由此结构性保证。数据面语义
（池来自 Hub 事件重放 / active_in 拆分 / 排序）由 test_admin_ledgers.py
钉住；本文件只验证 CLI 层：路径正确、--json 原样透传、404 detail 透传、
空态有用提示、只读（命令族不存在写动词）。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from bladex_proxy import cli


@pytest.fixture(autouse=True)
def _pin_tz(monkeypatch):
    """UPDATED 列是本地时区（MQ-L46）：钉 Asia/Shanghai，端点给的 UTC 10:11:12 应显示 18:11:12。
    不钉的话断言随测试机时区变。"""
    import time
    monkeypatch.setenv("TZ", "Asia/Shanghai")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path: Path, monkeypatch, isolated_home):
    """隔离：cli.main 会把 cwd 下 config/.env 灌进 os.environ（且不回收）--
    在仓库根跑会吸入真实配置（同 test_cli_memory_matter 的坑）。"""
    monkeypatch.chdir(tmp_path)
    for k in list(os.environ):
        if k.startswith("BLADEX_"):
            monkeypatch.delenv(k, raising=False)


class _Recorder:
    def __init__(self, responses: dict | None = None):
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []
        self._responses = responses or {}

    def __call__(self, method: str, path: str, body: dict | None = None,
                 params: dict | None = None):
        self.calls.append((method, path, body, params))
        return self._responses.get((method, path), (200, {"status": "ok"}))


@pytest.fixture
def rec(monkeypatch):
    r = _Recorder()
    from bladex_proxy.cli import ledger_cmds as _ldg  # F0.1 拆包：消费方在 cli/ledger_cmds.py
    monkeypatch.setattr(_ldg, "_admin_call", r)
    return r


# 一行 /admin/ledgers 的真实形状。**逐字段对照 server.py `list_ledgers` 的
# rows.append 生成**（复核 2026-08-27 打回的教训：fixture 声称对照端点而
# 实际没有，凭印象造了不存在的 `sections` 键）。此后每次端点改字段，这里
# 应随之更新并重跑本文件。
# 端点字段：ledger_id / title / status / matter_id / parent_ledger_id /
# goal / goal_source / created_at / updated_at / active_in / entry_counts
# （entry_counts 键来自 Ledger.section_order，非闭集；goal 恒不在其中）。
_LEDGER_ROW = {
    "ledger_id": "ldg-aaba12345001",
    "title": "proxy 502 修复",
    "status": "active",
    "matter_id": "matter-77",
    "parent_ledger_id": "",
    "goal": "把 proxy 502 修好",
    "goal_source": "user",
    "created_at": "2026-08-26T09:00:00",
    "updated_at": "2026-08-27T10:11:12",
    "entry_counts": {"core": 1, "verified": 0, "open": 1, "next": 1},
    "active_in": [{"agent": "hermes:default", "project": "Global"}],
}


def test_ledger_list_maps_to_admin_get(rec):
    rec._responses[("GET", "/admin/ledgers")] = (
        200, {"enabled": True, "ledgers": [_LEDGER_ROW], "total": 1})
    assert cli.main(["ledger", "list"]) == 0
    assert rec.calls[0][:2] == ("GET", "/admin/ledgers")
    assert rec.calls[0][3] is None  # 无查询参数


def test_ledger_list_renders_row_fields(capsys, rec):
    rec._responses[("GET", "/admin/ledgers")] = (
        200, {"enabled": True, "ledgers": [_LEDGER_ROW], "total": 1})
    cli.main(["ledger", "list"])
    out = capsys.readouterr().out
    assert "ldg-aaba12345001" in out          # id
    assert "proxy 502 修复" in out             # 标题
    assert "[u] 把 proxy 502 修好" in out      # Goal + user 来源角标
    assert "hermes:default@Global" in out     # 激活 scope
    assert "matter-77" in out                 # Matter 锚
    assert "2026-08-27T18:11" in out          # 更新时间：UTC 10:11 → CST 18:11（MQ-L46）
    assert "2026-08-27T10:11" not in out      # 面上不再出现 UTC
    assert "active" in out                    # status
    # 五段条目数：goal 有 + c/v/o/n（entry_counts，goal 不在其中）
    assert "g1/c1/v0/o1/n1" in out


def test_ledger_list_columns_aligned(capsys, rec):
    """缺陷 2 的回归剧本：表头列数必须与数据行列数一致、各列起始位置对齐。

    复核打回的教训：实现里表头 7 列、数据行 8 列（status 列没有表头），
    整行从第一列起错位。这里按"每列起始列号"断言，改列不改两边就会炸。
    """
    rec._responses[("GET", "/admin/ledgers")] = (
        200, {"enabled": True, "ledgers": [_LEDGER_ROW], "total": 1})
    cli.main(["ledger", "list"])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln]
    # 第一行是 "N task ledger(s):"，随后表头、数据行。
    header, row = lines[1], lines[2]
    import re
    header_pos = [m.start() for m in re.finditer(r"\S+", header)]
    header_words = re.findall(r"\S+", header)
    # 表头 8 列点名（防列语义漂移）。
    assert header_words == ["ID", "TITLE", "GOAL", "ENTRIES",
                            "ACTIVE-IN", "MATTER", "UPDATED", "STATUS"], header_words
    # 列对齐的本质不变量（缺陷 2 的直接对偶）：以每个表头词的起始位置为
    # 切点把数据行切片，8 列**每列都非空**。不按空隙 split--列内容填满
    # 列宽时间隙可能只剩 1 空格（如中文标题占满列宽），split 不可靠。
    bounds = header_pos + [len(row)]
    cells = [row[i:j].strip() for i, j in zip(bounds, bounds[1:], strict=False)]
    assert len(cells) == 8 and all(cells), cells
    # 逐列抽查内容（错位在这里冒出来：列语义与列内容一一对应）。
    assert cells[0] == "ldg-aaba12345001"
    assert cells[1] == "proxy 502 修复"
    assert cells[2].startswith("[u]")
    assert cells[3] == "g1/c1/v0/o1/n1"
    assert cells[4] == "hermes:default@Global"
    assert cells[5] == "matter-77"
    assert cells[6] == "2026-08-27T18:11:12"   # 本地时区（+08:00 被 [:19] 截掉）
    assert cells[7] == "active"


def test_ledger_list_custom_section_entry_counts(capsys, rec):
    """section_order 非闭集：自定义段出现在 entry_counts 里，渲染不丢。"""
    row = {**_LEDGER_ROW, "entry_counts":
           {"core": 2, "verified": 1, "open": 0, "next": 2, "risks": 1}}
    rec._responses[("GET", "/admin/ledgers")] = (
        200, {"enabled": True, "ledgers": [row], "total": 1})
    cli.main(["ledger", "list"])
    out = capsys.readouterr().out
    assert "g1/c2/v1/o0/n2" in out
    assert "risks=1" in out


def test_ledger_list_goal_source_tags(capsys, rec):
    """goal 来源角标：user/model/model_revised 各自渲染，空 goal 打 '-'。"""
    rows = [
        {**_LEDGER_ROW, "ledger_id": "ldg-src00000001", "goal": "G1",
         "goal_source": "model"},
        {**_LEDGER_ROW, "ledger_id": "ldg-src00000002", "goal": "G2",
         "goal_source": "model_revised"},
        {**_LEDGER_ROW, "ledger_id": "ldg-src00000003", "goal": "",
         "goal_source": ""},
    ]
    rec._responses[("GET", "/admin/ledgers")] = (
        200, {"enabled": True, "ledgers": rows, "total": 3})
    cli.main(["ledger", "list"])
    out = capsys.readouterr().out
    assert "[m] G1" in out
    assert "[r] G2" in out
    assert "[?] G" not in out  # 空 goal_source 只在 goal 非空时才渲染


def test_ledger_list_empty_goal_cell(capsys, rec):
    row = {**_LEDGER_ROW, "goal": "", "goal_source": ""}
    rec._responses[("GET", "/admin/ledgers")] = (
        200, {"enabled": True, "ledgers": [row], "total": 1})
    cli.main(["ledger", "list"])
    out = capsys.readouterr().out
    assert "g0/" in out          # goal 空 -> g0
    assert "[?]" not in out      # 不渲染空来源角标


def test_ledger_list_json_passthrough(capsys, rec):
    payload = {"enabled": True, "ledgers": [_LEDGER_ROW], "total": 1}
    rec._responses[("GET", "/admin/ledgers")] = (200, payload)
    assert cli.main(["ledger", "list", "--json"]) == 0
    out = capsys.readouterr().out
    assert "ldg-aaba12345001" in out
    # 原样透传：非表格渲染（无表头）
    assert "TITLE" not in out
    import json
    assert json.loads(out) == payload


def test_ledger_list_empty_state(capsys, rec):
    rec._responses[("GET", "/admin/ledgers")] = (
        200, {"enabled": True, "ledgers": [], "total": 0})
    assert cli.main(["ledger", "list"]) == 0
    out = capsys.readouterr().out
    assert "No task ledgers" in out
    assert "LEDGER" not in out  # 空态不打印表头


def test_ledger_list_disabled_state(capsys, rec):
    # agency 未启用：端点返回 {"enabled": false, "ledgers": [], "total": 0}
    rec._responses[("GET", "/admin/ledgers")] = (
        200, {"enabled": False, "ledgers": [], "total": 0})
    assert cli.main(["ledger", "list"]) == 0
    out = capsys.readouterr().out
    assert "not enabled" in out


# GET /admin/ledgers/{id} 的真实形状。**逐字段对照 server.py `get_ledger`
# 的 JSONResponse(content=...)**：ledger（Ledger model_dump）/ markdown
# （render_ledger_md 产物）/ children（池内子账本行）/ active_in。
_LEDGER_DETAIL = {
    "ledger": {**_LEDGER_ROW},
    "markdown": "# proxy 502 修复\n\n## Goal\nuser asks to fix 502\n",
    "children": [{"ledger_id": "ldg-child000001", "title": "子任务"}],
    "active_in": [{"agent": "hermes:default", "project": "Global"}],
}


def test_ledger_show_maps_to_admin_get(rec):
    rec._responses[("GET", "/admin/ledgers/ldg-aaba12345001")] = (200, _LEDGER_DETAIL)
    assert cli.main(["ledger", "show", "ldg-aaba12345001"]) == 0
    assert rec.calls[0][:2] == ("GET", "/admin/ledgers/ldg-aaba12345001")


def test_ledger_show_prints_markdown_and_children(capsys, rec):
    rec._responses[("GET", "/admin/ledgers/ldg-aaba12345001")] = (200, _LEDGER_DETAIL)
    cli.main(["ledger", "show", "ldg-aaba12345001"])
    out = capsys.readouterr().out
    assert "# proxy 502 修复" in out          # 端点渲染好的 markdown 直接打印
    assert "ldg-child000001" in out           # 子账本
    assert "子任务" in out


def test_ledger_show_json_passthrough(capsys, rec):
    rec._responses[("GET", "/admin/ledgers/ldg-aaba12345001")] = (200, _LEDGER_DETAIL)
    assert cli.main(["ledger", "show", "ldg-aaba12345001", "--json"]) == 0
    out = capsys.readouterr().out
    import json
    assert json.loads(out) == _LEDGER_DETAIL


def test_ledger_show_404_prints_endpoint_detail(capsys, rec):
    # 404 时端点返回 {"status": "not_found", "detail": "unknown ledger ..."}
    # -- 必须照它的 detail 打印，不许吞成 "HTTP 404"。
    rec._responses[("GET", "/admin/ledgers/ldg-nope000000")] = (
        404, {"status": "not_found",
              "detail": "unknown ledger: no ledger ldg-nope000000 in the pool"})
    assert cli.main(["ledger", "show", "ldg-nope000000"]) == 1
    out = capsys.readouterr().out
    assert "no ledger ldg-nope000000 in the pool" in out
    assert "404" not in out  # 不许吐 "HTTP 404" 这种吞掉 detail 的形态


def test_ledger_family_is_read_only(rec):
    """账本的作者是 agent 侧的模型，Goal 只有用户能改（ADR-0032 三红线）--
    CLI 命令族不得出现任何写动词。"""
    write_verbs = ("create", "update", "delete", "forget", "write", "revise",
                   "set", "edit", "add", "remove", "merge", "assign", "rename")
    command_names = set()
    for cmd in cli.ledger_app.registered_commands:
        if isinstance(cmd.name, str):
            command_names.add(cmd.name)
        elif cmd.name:
            command_names.update(n for n in cmd.name if isinstance(n, str))
    # 白名单。新增命令必须在这里显式登记 —— 这条**按设计拦住过一次**：
    # 2026-08-28 加 `doctor` 时当场变红，迫使加的人回来确认它确实只读。
    # `doctor` = 账本健康检查，只报告不修改（守卫见 test_cli_ledger_doctor.py
    # 的 `test_is_read_only`：全程只有 GET + 输出明写 "Nothing was changed"）。
    assert set(command_names) <= {"list", "show", "doctor"}, command_names
    # 以及防御性断言：将来有人加写命令时，这里先炸。
    assert not (set(command_names) & set(write_verbs))
    # 全程只有 GET：跑一遍两个命令，录制桩里不允许出现非 GET。
    rec._responses[("GET", "/admin/ledgers")] = (200, {"ledgers": [], "total": 0})
    cli.main(["ledger", "list"])
    assert all(m == "GET" for m, *_ in rec.calls)


# ═════════════════════════════════════════════════════════════════════════
# 契约测试：起真实 /admin/ledgers 端点，把端点的真实响应喂给 CLI。
# 复核打回的根因是"fixture 声称对照端点、实际凭印象造"（造出不存在的
# `sections` 键）。下面这组测试从结构上杜绝同类缺陷：CLI 读的每个字段
# 都来自 create_app(...) + 真实 list_ledgers/get_ledger 端点代码的产出，
# 端点改字段时这里会直接 KeyError/断言失败，逼着两边同步。
# 与 test_admin_ledgers.py 的分工：那边钉端点语义（重放存活/排序/拆分），
# 这边钉"CLI 渲染所依赖的字段确实存在于端点响应里"。
# ═════════════════════════════════════════════════════════════════════════


@pytest.fixture
def live_endpoint(monkeypatch):
    """真实 TestClient 端点 + `_admin_call` 转发。返回 (client, transport)。

    transport 挂进 cli._admin_call，CLI 的 HTTP 调用全部落到本端点。
    """
    from bladex_core.ledger import new_ledger
    from bladex_core.ledger_runtime import (
        ActivationTable,
        activation_scope,
        bind_matter,
        ledger_anchor_matter_id,
    )
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.server import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setenv("BLADEX_AUTH_ENABLED", "false")
    app = create_app(ProxyConfig())

    parent = new_ledger(ledger_id="ldg-live0000001", title="端点契约任务",
                        goal="goal 原话", goal_source="user",
                        created_at="2026-08-26T10:00:00Z")
    parent = bind_matter(parent, ledger_anchor_matter_id("ldg-live0000001"),
                         updated_at="2026-08-26T11:00:00Z")
    pool = {parent.ledger_id: parent}
    bindings = {activation_scope("hermes:default", ""): parent.ledger_id}

    class _StubAgency:
        def __init__(self):
            self.pool = pool
            self.activation = ActivationTable(debounce_turns=0)
            self.activation.restore(bindings)

    with TestClient(app) as c:
        c.app.state.agency = _StubAgency()

        def transport(method: str, path: str, body: dict | None = None,
                      params: dict | None = None):
            if method == "GET":
                r = c.get(path)
            else:  # 只读命令族不该走到这里；fail loud。
                raise AssertionError(f"read-only family sent {method} {path}")
            try:
                payload = r.json()
            except ValueError:
                payload = {}
            return r.status_code, payload

        from bladex_proxy.cli import ledger_cmds as _ldg  # F0.1 拆包：消费方在 cli/ledger_cmds.py
        monkeypatch.setattr(_ldg, "_admin_call", transport)
        yield c, transport


def test_contract_list_fields_from_real_endpoint(capsys, live_endpoint):
    """CLI list 渲染读的每个键，必须出现在真实端点响应里。"""
    _, transport = live_endpoint
    code, data = transport("GET", "/admin/ledgers")
    assert code == 200
    # 端点真实键集（新键无害，缺键=契约破坏）。
    row = data["ledgers"][0]
    for key in ("ledger_id", "title", "status", "matter_id", "goal",
                "goal_source", "created_at", "updated_at", "active_in",
                "entry_counts"):
        assert key in row, f"endpoint row missing {key!r}"
    # CLI 全流程跑通（表格渲染）。
    assert cli.main(["ledger", "list"]) == 0
    out = capsys.readouterr().out
    assert "ldg-live0000001" in out
    assert "端点契约任务" in out
    assert "[u] goal 原话" in out


def test_contract_show_fields_from_real_endpoint(capsys, live_endpoint):
    """CLI show 渲染读的每个键（markdown/children），来自真实端点。"""
    _, transport = live_endpoint
    code, data = transport("GET", "/admin/ledgers/ldg-live0000001")
    assert code == 200
    for key in ("ledger", "markdown", "children", "active_in"):
        assert key in data, f"endpoint detail missing {key!r}"
    assert cli.main(["ledger", "show", "ldg-live0000001"]) == 0
    out = capsys.readouterr().out
    assert "# 端点契约任务" in out      # render_ledger_md 的标题
    assert "## Goal" in out             # 渲染正文


def test_contract_404_detail_from_real_endpoint(capsys, live_endpoint):
    """404 的 detail 文本来自真实端点，CLI 照印。"""
    _, transport = live_endpoint
    code, data = transport("GET", "/admin/ledgers/ldg-none0000000")
    assert code == 404
    assert data.get("detail")
    rc = cli.main(["ledger", "show", "ldg-none0000000"])
    assert rc == 1
    out = capsys.readouterr().out
    assert data["detail"] in out
