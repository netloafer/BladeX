"""`bladex ledger doctor` —— 账本健康检查（只读）。

## 来历

2026-08-27：用户只发了**一句话**，BladeX 建了**四本账本**（两本来自 Codex
客户端的内部功能调用），靠肉眼翻列表才发现，排查三小时。根因已修（MQ-L21），
但 V-L5 §1c 记着一条没修的：**反方向「多本账本本该是一本」零检测**。

本命令补那个检测。**只报告不自动修** —— 账本的作者是 agent 侧的模型，
Goal 只有用户能改；自动合并会踩「误合并=0」第一红线。

## 🔴 fixture 与端点对账

本仓两次栽在「fixture 自造了端点不返回的字段，测试全绿但功能是坏的」
（CC 的 `sections` / 我自己的 dict-vs-属性 tool_calls）。所以第一条测试就是
**把端点源码的键集与 fixture 键集比一次**，其余测试才有意义。
"""
from __future__ import annotations

import json

import pytest

from bladex_proxy import cli


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch, isolated_home):
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))


# ── fixture：字段**必须**与 server.list_ledgers 的 rows.append 一致 ────────

def _row(lid, *, title="", goal="", matter_id="m-x", counts=None,
         created="2026-08-27T10:00:00Z", updated="2026-08-27T11:00:00Z",
         active=(("codex", "Global"),), status="active"):
    return {
        "ledger_id": lid, "title": title, "status": status,
        "matter_id": matter_id, "parent_ledger_id": "",
        "goal": goal, "goal_source": "model",
        "created_at": created, "updated_at": updated,
        "active_in": [{"agent": a, "project": p} for a, p in active],
        "entry_counts": counts if counts is not None
        else {"core": 1, "verified": 2, "open": 0, "next": 1},
    }


def _detail(lid, *, verbatim="", revisions=0):
    return {"ledger": {"ledger_id": lid, "goal_verbatim": verbatim,
                       "goal_revisions": revisions},
            "markdown": "# x", "children": [], "active_in": []}


class _Rec:
    """录制 `_admin_call`。list 一次 + 每本 detail 一次。"""

    def __init__(self, rows, details=None, list_code=200, list_body=None):
        self.rows, self.details = rows, details or {}
        self.list_code = list_code
        self.list_body = list_body
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, path, body=None, params=None, timeout=15.0):
        self.calls.append((method, path))
        if path == "/admin/ledgers":
            if self.list_body is not None:
                return self.list_code, self.list_body
            return self.list_code, {"ledgers": self.rows, "total": len(self.rows)}
        lid = path.rsplit("/", 1)[-1]
        if lid in self.details:
            return 200, self.details[lid]
        return 404, {"status": "not_found", "detail": f"unknown ledger {lid}"}


def _run(monkeypatch, rec, argv=("ledger", "doctor")):
    from _source_probe import consumer_module
    _ldg = consumer_module("bladex_proxy.cli.ledger_cmds", "bladex_proxy.cli")   # F0.1 拆包：消费方在 cli/ledger_cmds.py
    monkeypatch.setattr(_ldg, "_admin_call", rec)
    return cli.main(list(argv))


# ══════════════════════════════════════════════════════════════════════════

def test_fixture_keys_match_the_real_endpoint():
    """🔴 前置对账：fixture 的键集必须 ⊇ 端点实际产出的键集。

    端点改字段而 fixture 没跟上 ⇒ 这条先炸，而不是让整个文件继续绿着
    去测一个不存在的形态。
    """
    import pathlib
    import re

    # 🔴 读**源码文本**，不 import `bladex_proxy.server`。
    # 本测试做的是源码级对账，import 只会拖进 rocksdict / litellm 这些
    # native 依赖 —— 那会让它在缺依赖的环境里根本跑不起来，
    # 而这条恰恰是本文件里最该处处都跑的一条（其余测试的前提）。
    from _source_probe import package_source   # 只读文件，不 import 生产模块
    src = package_source("server")   # F0.1 拆包：server.py → server/ 包，按包拼接读源码
    i = src.index("rows.append({")
    block = src[i:src.index("})", i)]
    endpoint_keys = set(re.findall(r'"(\w+)":', block))
    assert endpoint_keys, "没解析出端点键集——server.py 的形状变了，先看那边"
    missing = endpoint_keys - set(_row("ldg-1"))
    assert not missing, f"fixture 缺端点真实返回的键：{missing}"


class TestDuplicateDetection:
    def test_same_title_in_same_scope_is_flagged(self, monkeypatch, capsys):
        rec = _Rec([_row("ldg-a", title="Codex ledger health 开发"),
                    _row("ldg-b", title="codex   ledger health 开发")])
        rc = _run(monkeypatch, rec)
        out = capsys.readouterr().out
        assert rc == 1
        assert "duplicate_title" in out
        assert "ldg-a" in out and "ldg-b" in out

    def test_different_scopes_are_not_duplicates(self, monkeypatch, capsys):
        """🔴 跨 agent 同名是**北极星要的形态**（交接），不是重复。"""
        rec = _Rec([_row("ldg-a", title="同一件事", active=(("codex", "Global"),)),
                    _row("ldg-b", title="同一件事", active=(("claude-code", "Global"),))])
        _run(monkeypatch, rec)
        assert "duplicate_title" not in capsys.readouterr().out

    def test_similar_but_different_titles_are_not_flagged(self, monkeypatch, capsys):
        """归一化只做小写+压空白。宁可漏报，不误报。"""
        rec = _Rec([_row("ldg-a", title="Codex ledger health 开发"),
                    _row("ldg-b", title="Codex ledger health 复核")])
        assert _run(monkeypatch, rec) == 0
        assert "no problems found" in capsys.readouterr().out

    def test_same_goal_prefix_is_flagged(self, monkeypatch, capsys):
        long_goal = "Generate 0 to 3 hyperpersonalized suggestions for this project"
        rec = _Rec([_row("ldg-a", title="A", goal=long_goal + " one"),
                    _row("ldg-b", title="B", goal=long_goal + " two")])
        _run(monkeypatch, rec)
        assert "duplicate_goal" in capsys.readouterr().out

    def test_title_and_goal_duplicate_reported_once(self, monkeypatch, capsys):
        """同一对账本不该既报标题又报 Goal。

        ⚠️ 判据数的是 **finding 条数**，不是 id 在输出里出现几次——
        每条 finding 里 id 本来就出现两次（ids 行 + action 行）。
        初版按 id 计数，红了，而代码是对的：**尺子数错了对象**
        （今天同型错误第 N 次，记在这儿）。
        """
        rec = _Rec([_row("ldg-a", title="T", goal="G" * 80),
                    _row("ldg-b", title="T", goal="G" * 80)])
        _run(monkeypatch, rec, ("ledger", "doctor", "--json", "--no-details"))
        payload = json.loads(capsys.readouterr().out)
        kinds = [f["kind"] for f in payload["findings"]]
        assert kinds == ["duplicate_title"], kinds


class TestMachineTextGoal:
    def test_uses_goal_verbatim_not_refined_goal(self, monkeypatch, capsys):
        """🔴 判据对象是**用户原话**，不是模型提炼过的 goal。

        提炼后的 goal 早就不像模板了；查它等于关掉这条检查。
        """
        rec = _Rec([_row("ldg-a", title="T", goal="给 Codex 生成任务建议")],
                   {"ldg-a": _detail("ldg-a",
                                     verbatim="# Overview\n\nGenerate 0 to 3 ...")})
        _run(monkeypatch, rec)
        assert "machine_text_goal" in capsys.readouterr().out

    def test_real_user_goal_is_not_flagged(self, monkeypatch, capsys):
        rec = _Rec([_row("ldg-a", title="T")],
                   {"ldg-a": _detail("ldg-a", verbatim="按任务卡执行开发任务")})
        assert _run(monkeypatch, rec) == 0

    def test_reuses_agency_machine_text_mark(self):
        """判据必须复用 `agency._machine_text_mark` —— 两份判据早晚分叉。

        任务卡明确写了"直接复用那个函数，不要再写一份"。
        """
        import inspect
        src = inspect.getsource(cli._doctor_findings)
        assert "_machine_text_mark" in src
        assert "you are a" not in src.lower(), "别在 doctor 里另抄一份形态表"

    def test_position_gate_inherited(self, monkeypatch, capsys):
        """MQ-A19 的位置门（只看前 200 字符）随复用一起继承。

        正文里**提到** "you are a helpful assistant" 的真实目标不该被标记。
        """
        rec = _Rec([_row("ldg-a", title="T")],
                   {"ldg-a": _detail("ldg-a", verbatim="x" * 250 + "you are a helpful")})
        assert _run(monkeypatch, rec) == 0


class TestNeverUsed:
    def test_empty_and_old_is_flagged(self, monkeypatch, capsys):
        rec = _Rec([_row("ldg-a", title="T", counts={"core": 0, "verified": 0},
                         created="2026-08-01T00:00:00Z")],
                   {"ldg-a": _detail("ldg-a")})
        _run(monkeypatch, rec)
        assert "never_used" in capsys.readouterr().out

    def test_empty_but_fresh_is_not_flagged(self, monkeypatch, capsys):
        import datetime as dt
        today = dt.datetime.now(dt.UTC).isoformat()
        rec = _Rec([_row("ldg-a", title="T", counts={"core": 0}, created=today)],
                   {"ldg-a": _detail("ldg-a")})
        assert _run(monkeypatch, rec) == 0

    def test_has_entries_is_not_flagged(self, monkeypatch, capsys):
        rec = _Rec([_row("ldg-a", title="T", counts={"core": 1},
                         created="2026-08-01T00:00:00Z")],
                   {"ldg-a": _detail("ldg-a")})
        assert _run(monkeypatch, rec) == 0

    def test_goal_revised_counts_as_used(self, monkeypatch, capsys):
        rec = _Rec([_row("ldg-a", title="T", counts={"core": 0},
                         created="2026-08-01T00:00:00Z")],
                   {"ldg-a": _detail("ldg-a", revisions=1)})
        assert _run(monkeypatch, rec) == 0

    def test_unparseable_date_is_not_guessed(self, monkeypatch, capsys):
        """时间解析不了 ⇒ 不报，也不当成 0 天。不猜。"""
        rec = _Rec([_row("ldg-a", title="T", counts={"core": 0}, created="???")],
                   {"ldg-a": _detail("ldg-a")})
        assert _run(monkeypatch, rec) == 0


class TestNoMatterAnchor:
    """🔴 只报**锚点机制上线之后**建的无锚账本。

    初版报全部：live 实测 20 本报 12 本，信噪比 25%（2026-08-28）。
    锚点 08-26 才启用，之前的账本天然没锚 —— **它们不是问题**，
    而一条必然大面积误报的检查会让人不再看这个命令。

    判据不写死日期，改**数据自证**：已锚账本里最早的 `created_at` = 机制上线时刻。
    """

    def test_flags_only_ledgers_born_after_anchoring(self, monkeypatch, capsys):
        rec = _Rec([
            _row("old", title="A", matter_id="", created="2026-08-20T00:00:00Z"),
            _row("first-anchored", title="B", matter_id="m-1",
                 created="2026-08-26T00:00:00Z"),
            _row("new-unanchored", title="C", matter_id="",
                 created="2026-08-27T00:00:00Z"),
        ], {k: _detail(k) for k in ("old", "first-anchored", "new-unanchored")})
        _run(monkeypatch, rec, ("ledger", "doctor", "--json", "--no-details"))
        payload = __import__("json").loads(capsys.readouterr().out)
        flagged = [f["ledger_ids"][0] for f in payload["findings"]
                   if f["kind"] == "no_matter_anchor"]
        assert flagged == ["new-unanchored"], flagged

    def test_no_anchored_ledger_means_check_is_skipped(self, monkeypatch, capsys):
        """一本已锚的都没有 ⇒ 机制没跑过 ⇒ 无从判断，整条跳过。"""
        rec = _Rec([_row("ldg-a", title="T", matter_id=""),
                    _row("ldg-b", title="U", matter_id="")],
                   {"ldg-a": _detail("ldg-a"), "ldg-b": _detail("ldg-b")})
        _run(monkeypatch, rec, ("ledger", "doctor", "--no-details"))
        assert "no_matter_anchor" not in capsys.readouterr().out

    def test_anchored_is_clean(self, monkeypatch, capsys):
        rec = _Rec([_row("ldg-a", title="T", matter_id="m-1")],
                   {"ldg-a": _detail("ldg-a")})
        assert _run(monkeypatch, rec) == 0


class TestOutputAndBoundaries:
    def test_healthy_pool_says_so(self, monkeypatch, capsys):
        """阴性对照：干净的池子说"没发现问题"，不打空表。"""
        rec = _Rec([_row("ldg-a", title="A"), _row("ldg-b", title="B")],
                   {"ldg-a": _detail("ldg-a"), "ldg-b": _detail("ldg-b")})
        assert _run(monkeypatch, rec) == 0
        assert "no problems found" in capsys.readouterr().out

    def test_json_shape(self, monkeypatch, capsys):
        # 用「机器文本 Goal」当样本 —— 它不依赖 ④ 的锚点纪元前提。
        rec = _Rec([_row("ldg-a", title="T")],
                   {"ldg-a": _detail("ldg-a", verbatim="# Overview\n\nGenerate ...")})
        _run(monkeypatch, rec, ("ledger", "doctor", "--json"))
        payload = json.loads(capsys.readouterr().out)
        assert payload["checked"] == 1
        f = payload["findings"][0]
        assert set(f) == {"kind", "ledger_ids", "detail", "action"}

    def test_empty_pool(self, monkeypatch, capsys):
        assert _run(monkeypatch, _Rec([])) == 0
        assert "nothing to check" in capsys.readouterr().out.lower()

    def test_disabled_feature(self, monkeypatch, capsys):
        rec = _Rec([], list_body={"enabled": False, "ledgers": [], "total": 0})
        assert _run(monkeypatch, rec) == 0
        assert "not enabled" in capsys.readouterr().out

    def test_endpoint_error_is_surfaced(self, monkeypatch):
        rec = _Rec([], list_code=500,
                   list_body={"error": {"message": "boom"}})
        assert _run(monkeypatch, rec) != 0

    def test_is_read_only(self, monkeypatch, capsys):
        """🔴 只读边界：全程只有 GET。"""
        rec = _Rec([_row("ldg-a", title="T")],
                   {"ldg-a": _detail("ldg-a", verbatim="# Overview\n\nGenerate ...")})
        _run(monkeypatch, rec)
        assert all(m == "GET" for m, _ in rec.calls), rec.calls
        assert "Nothing was changed" in capsys.readouterr().out

    def test_no_details_skips_the_extra_calls(self, monkeypatch):
        rec = _Rec([_row("ldg-a", title="T"), _row("ldg-b", title="U")],
                   {"ldg-a": _detail("ldg-a"), "ldg-b": _detail("ldg-b")})
        _run(monkeypatch, rec, ("ledger", "doctor", "--no-details"))
        assert rec.calls == [("GET", "/admin/ledgers")]

    def test_detail_fetch_failure_does_not_crash(self, monkeypatch):
        """某本账本详情取不到 ⇒ 降级继续，不炸整条命令。"""
        rec = _Rec([_row("ldg-a", title="T")], {})     # detail 全 404
        assert _run(monkeypatch, rec) == 0
