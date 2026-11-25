"""批 AC2 步骤 4（MQ-P35 过渡）：Router 网关出口一处 `router_call_usage`，带调用方 `purpose=`。

- 守卫：全仓（生产包 + scripts/，不含测试）直接调 `router_sdk.(a)completion / embedding` 的地方
  都显式传了 `purpose`，取值在闭集 `router_sdk.PURPOSES` 里；7 个 BladeX 自身调用点的 purpose
  恰好是那 7 个，agent 主请求（`route.call_model`）标 `PURPOSE_AGENT`、网关不重复记；
- 网关：purpose 先剥掉再转发（上游看不到它）；未知取值 ⇒ ValueError；没传 ⇒ `unknown`；
  抛异常的调用也记一行（`ok=False`）；
- 判据：带桩上游跑一次蒸馏 + 裁决，日志按 purpose 汇总出的调用次数 = 桩记录的调用次数。
"""

from __future__ import annotations

import ast
import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from bladex_core.adjudication import AdjudicationInput
from bladex_proxy import router_sdk
from bladex_proxy.adjudicator import LLMAdjudicator
from bladex_proxy.distillation import LLMDistiller

_REPO = Path(__file__).resolve().parents[3]
_ENTRY = {"acompletion", "completion", "embedding"}
_SELF_PURPOSES = {"distill", "file_summary", "adjudicate", "link_judge", "assembly_review",
                  "route_judge", "embedding", "consistency_judge"}


def _sources() -> list[Path]:
    out = []
    for base in (_REPO / "packages", _REPO / "scripts"):
        for p in base.rglob("*.py"):
            parts = set(p.relative_to(_REPO).parts)
            if parts & {"tests", "archive", "__pycache__", ".venv"}:
                continue
            out.append(p)
    return out


def _purpose_value(kw: ast.keyword) -> str | None:
    v = kw.value
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return v.value
    if (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name)
            and v.value.id == "router_sdk" and v.attr.startswith("PURPOSE_")):
        return getattr(router_sdk, v.attr)
    return None


def router_call_sites(paths: list[Path]) -> list[tuple[str, int, str | None]]:
    """(文件, 行, purpose 取值 / None=没传或不是字面量) —— 每个直接调用点一条。"""
    sites = []
    for p in paths:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _ENTRY and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "router_sdk"):
                kws = [k for k in node.keywords if k.arg == "purpose"]
                sites.append((str(p.relative_to(_REPO)), node.lineno,
                              _purpose_value(kws[0]) if kws else None))
    return sites


def test_every_direct_router_call_passes_a_known_purpose() -> None:
    sites = router_call_sites(_sources())
    # 下限从被扫目录推出，不写死总数：生产包 11 处恒在；scripts/ 里凡是写了 router_sdk 调用形态的文件
    # 都必须至少扫出一处（公开树不带私有脚本时，这一项自然为空，总数天然变少）。
    script_files = [p for p in (_REPO / "scripts").rglob("*.py")
                    if re.search(r"router_sdk\.(?:a?completion|embedding)\(", p.read_text(encoding="utf-8"))]
    scanned_scripts = {s[0] for s in sites if s[0].startswith("scripts/")}
    assert len([s for s in sites if s[0].startswith("packages/")]) >= 11, \
        f"scanned too few call sites ({len(sites)})"
    assert len(scanned_scripts) >= len(script_files), (scanned_scripts, script_files)
    bad = [s for s in sites if s[2] not in router_sdk.PURPOSES]
    assert bad == [], f"router_sdk calls without an explicit known purpose=: {bad}"
    prod = {s[2] for s in sites if s[0].startswith("packages/")}
    assert prod == _SELF_PURPOSES | {router_sdk.PURPOSE_AGENT}
    assert {s[2] for s in sites if s[0].startswith("scripts/")} == ({"script"} if script_files else set())


def test_no_aliased_imports_of_the_call_surface() -> None:
    """`from bladex_proxy.router_sdk import completion` 会绕过上面的守卫 ⇒ 不许。"""
    offenders = []
    for p in _sources():
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.ImportFrom) and (node.module or "").endswith("router_sdk")
                    and {a.name for a in node.names} & _ENTRY):
                offenders.append(str(p.relative_to(_REPO)))
    assert offenders == []


def test_guard_flags_a_call_without_purpose(tmp_path: Path) -> None:
    f = tmp_path / "x.py"
    f.write_text("router_sdk.completion(model='m')\n"
                 "router_sdk.embedding(purpose='nope', model='m')\n"
                 "router_sdk.completion(purpose='distill', model='m')\n")
    got = [(ln, p) for _, ln, p in router_call_sites_rel(f)]
    assert got == [(1, None), (2, "nope"), (3, "distill")]


def router_call_sites_rel(f: Path) -> list[tuple[str, int, str | None]]:
    tree = ast.parse(f.read_text())
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in _ENTRY and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "router_sdk"):
            kws = [k for k in node.keywords if k.arg == "purpose"]
            out.append((str(f), node.lineno, _purpose_value(kws[0]) if kws else None))
    return sorted(out, key=lambda x: x[1])


# ── 网关 ─────────────────────────────────────────────────────────────────────


class _Rec:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def info(self, event: str, **kw: Any) -> None:
        self.events.append((event, kw))

    warning = info


@pytest.fixture()
def rec(monkeypatch) -> _Rec:  # noqa: ANN001
    r = _Rec()
    monkeypatch.setattr(router_sdk, "_usage_log", r)
    return r


def _resp(text: str) -> SimpleNamespace:
    msg = SimpleNamespace(content=text, reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="stop")],
        usage={"prompt_tokens": 120, "completion_tokens": 9, "total_tokens": 129,
               "prompt_tokens_details": {"cached_tokens": 64}})


def test_purpose_is_stripped_and_logged(rec: _Rec, monkeypatch) -> None:  # noqa: ANN001
    seen: list[dict] = []
    monkeypatch.setattr(router_sdk.litellm, "completion",
                        lambda **kw: (seen.append(kw), _resp("ok"))[1])
    router_sdk.completion(purpose="link_judge", model="openai/m", messages=[])
    assert "purpose" not in seen[0]
    [(ev, kw)] = rec.events
    assert ev == "router_call_usage" and kw["purpose"] == "link_judge"
    assert kw["model"] == "openai/m" and kw["ok"] is True
    assert (kw["input"], kw["output"], kw["cache_read"]) == (120, 9, 64)
    assert isinstance(kw["ms"], float)


def test_missing_purpose_is_unknown_and_bad_purpose_raises(rec: _Rec, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(router_sdk.litellm, "embedding", lambda **kw: {"data": []})
    router_sdk.embedding(model="m", input=["x"])
    assert rec.events[0][1]["purpose"] == router_sdk.PURPOSE_UNKNOWN
    assert rec.events[0][1]["input"] is None  # 没报 usage ⇒ NO-DATA，不是 0
    with pytest.raises(ValueError):
        router_sdk.embedding(purpose="distil", model="m", input=["x"])


def test_agent_purpose_not_logged_and_errors_logged(rec: _Rec, monkeypatch) -> None:  # noqa: ANN001
    async def _ok(**kw: Any) -> Any:
        return _resp("hi")

    async def _boom(**kw: Any) -> Any:
        raise TimeoutError("upstream")

    monkeypatch.setattr(router_sdk.litellm, "acompletion", _ok)
    asyncio.run(router_sdk.acompletion(purpose=router_sdk.PURPOSE_AGENT, model="m"))
    assert rec.events == []
    monkeypatch.setattr(router_sdk.litellm, "acompletion", _boom)
    with pytest.raises(TimeoutError):
        asyncio.run(router_sdk.acompletion(purpose="route_judge", model="m"))
    [(ev, kw)] = rec.events
    assert kw["purpose"] == "route_judge" and kw["ok"] is False and kw["error"] == "TimeoutError"


# ── 判据：按日志复算调用次数 = 桩记录 ─────────────────────────────────────────

_GOOD = json.dumps({"facts": [{"content": "用户偏好简洁回复", "item_kind": "preference",
                               "subject": "回复", "attribute": "风格", "entities": []}],
                    "matter_proposals": []}, ensure_ascii=False)


def test_distill_and_adjudicate_call_counts_recomputed_from_log(rec: _Rec, monkeypatch) -> None:  # noqa: ANN001
    stub_calls: list[str] = []

    def _stub(**kw: Any) -> Any:
        assert "purpose" not in kw  # 网关已剥掉
        stub_calls.append(kw["model"])
        body = kw["messages"][-1]["content"]
        if "### NEW " in body:
            ids = [ln.split()[-1] for ln in body.splitlines() if ln.startswith("### NEW ")]
            return _resp(json.dumps([{"candidate_id": i, "op": "add", "reason": "new"}
                                     for i in ids]))
        # 第一次蒸馏回空 ⇒ 走一次重试（重试也是一次调用，也要记）
        return _resp("" if len(stub_calls) == 1 else _GOOD)

    monkeypatch.setattr(router_sdk.litellm, "completion", _stub)
    monkeypatch.setenv("BLADEX_ADJUDICATE_CHUNK", "4")

    d = LLMDistiller(model="stub/distill", max_tokens=512)
    for text in ("回复请简洁一些，这是足够长的输入。", "我习惯用中文写提交信息，这是另一条足够长的输入。"):
        d.distill(text)
    n_distill = len(stub_calls)
    LLMAdjudicator(model="stub/adj").adjudicate(
        [AdjudicationInput(candidate_id=f"c{i}", content=f"fact {i}") for i in range(10)])
    n_adj = len(stub_calls) - n_distill

    by_purpose = Counter(kw["purpose"] for ev, kw in rec.events if ev == "router_call_usage")
    assert n_distill >= 3 and n_adj == 3  # 2 次蒸馏（其中一次重试）+ 10 条按 4 分块
    assert by_purpose == {"distill": n_distill, "adjudicate": n_adj}
    assert sum(by_purpose.values()) == len(stub_calls)
    models = Counter(kw["model"] for _, kw in rec.events)
    assert models == {"stub/distill": n_distill, "stub/adj": n_adj}
