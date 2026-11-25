"""批 AC · AC1 —— 用量聚合单一实现点（`bladex_proxy.usage`）。

判据：
- 口径：原文走 `capture.usage_norm`；旧轮次只取三键里的 input / output，**cache_read 记 NO-DATA 不记 0**；
- lane 三分 main / aux / bladex（内循环）；
- 窗口 `[since, until)` + 按给定时区切天；墓碑覆盖的轮次不计；
- **U2 A1 复算**：09-28 09:50:40–11:00:20（+08:00）的 input 合计 = **20,850,495**（`report-batch-U2-20260928.md` §1.1）。
  A1 跑在隔离实例上、其 Hub 已随 scratch 清除，生产 Hub 该窗口 0 轮（批 AC 在生产快照上复核过）⇒
  夹具 `fixtures/ac_u2_a1_capture_rows.json` 是从 A1 proxy 日志的 208 条 `capture_done` 行抽出的
  每轮 usage，这里按生产写法构造 Turn、`MemoryHub.put` 进 tmpdir，再用 `iter_hub_turns` + `aggregate`
  从 Hub 读回来算。它验的是"Hub 轮次 → 汇总"这一段的口径（时区、窗口切法、是否含 aux），
  不是 A1 当年那个 Hub 本身。
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from bladex_proxy.capture import usage_norm
from bladex_proxy.usage import (
    LANE_AUX,
    LANE_BLADEX,
    LANE_MAIN,
    aggregate,
    iter_hub_turns,
    turn_usage,
)

FIX = Path(__file__).parent / "fixtures"
CST = timezone(timedelta(hours=8))
SHAPES = json.loads((FIX / "ac_upstream_usage_shapes.json").read_text())["cells"]


def _turn(ts: datetime, *, agent: str = "a", model: str = "m", session: str = "s",
          usage: dict | None = None, raw: dict | None = None, aux: bool = False,
          aux_source: str = "", failed: bool = False) -> dict:
    """Hub 原始值的最小形状（与 `Turn.model_dump(mode="json")` 同键）。"""
    return {"ts": ts.astimezone(UTC).isoformat().replace("+00:00", "Z"), "model": model,
            "identity": {"agent_id": agent, "session_id": session},
            "auxiliary": aux, "aux_source": aux_source, "status": "failed" if failed else "ok",
            "response_meta": {"usage": usage or {}, **({"usage_raw": raw} if raw is not None else {})}}


T0 = datetime(2026, 10, 24, 23, 30, tzinfo=CST)


class TestNorm:
    def test_openai_shape(self):
        raw = next(c["usage"] for c in SHAPES if c["model"].startswith("openai/"))
        n = usage_norm(raw)
        assert n["input"] == raw["prompt_tokens"] and n["output"] == raw["completion_tokens"]
        assert n["cache_read"] == raw["prompt_tokens_details"]["cached_tokens"]
        assert n["reasoning"] == raw["completion_tokens_details"]["reasoning_tokens"]
        assert n["cache_creation"] is None, "OpenAI 形状不报 cache 写入 ⇒ None，不是 0"

    def test_anthropic_shape(self):
        raw = dict(next(c["usage"] for c in SHAPES if "cache_creation_input_tokens" in c["usage"]))
        raw.update(cache_read_input_tokens=77, cache_creation_input_tokens=5)
        n = usage_norm(raw)
        assert n["cache_read"] == 77 and n["cache_creation"] == 5

    def test_native_input_output_names(self):
        n = usage_norm({"input_tokens": 3, "output_tokens": 4,
                        "output_tokens_details": {"reasoning_tokens": 2}})
        assert (n["input"], n["output"], n["reasoning"]) == (3, 4, 2)

    def test_empty_is_all_none(self):
        assert set(usage_norm({}).values()) == {None}


class TestTurnUsage:
    def test_raw_wins_over_legacy(self):
        meta = {"usage": {"prompt": 1, "completion": 1, "total": 2},
                "usage_raw": {"prompt_tokens": 10, "completion_tokens": 2,
                              "prompt_tokens_details": {"cached_tokens": 8}}}
        assert turn_usage(meta) == {"input": 10, "output": 2, "cache_read": 8}

    def test_legacy_cache_read_is_nodata(self):
        assert turn_usage({"usage": {"prompt": 9, "completion": 2, "total": 11}}) == \
            {"input": 9, "output": 2, "cache_read": None}

    def test_empty_usage_is_all_nodata(self):
        assert turn_usage({"usage": {}}) == {"input": None, "output": None, "cache_read": None}
        assert turn_usage(None) == {"input": None, "output": None, "cache_read": None}


class TestAggregate:
    def test_nodata_is_counted_not_zeroed(self):
        rows = [("k1", _turn(T0, raw={"prompt_tokens": 100, "completion_tokens": 5,
                                      "prompt_tokens_details": {"cached_tokens": 60}})),
                ("k2", _turn(T0, usage={"prompt": 40, "completion": 3, "total": 43})),
                ("k3", _turn(T0, failed=True))]
        t = aggregate(rows, tz=CST)["totals"]["all"]
        assert (t["turns"], t["failed_turns"], t["raw_turns"]) == (3, 1, 1)
        assert (t["input"], t["input_nodata"]) == (140, 1)
        assert (t["output"], t["output_nodata"]) == (8, 1)
        assert (t["cache_read"], t["cache_read_nodata"]) == (60, 2)

    def test_lanes_are_separate(self):
        u = {"prompt": 10, "completion": 1, "total": 11}
        rows = [("k1", _turn(T0, usage=u)),
                ("k2", _turn(T0, usage=u, aux=True, aux_source="haiku_title")),
                ("k3", _turn(T0, usage=u, aux=True, aux_source="inner_loop"))]
        r = aggregate(rows, tz=CST)
        assert {ln: r["totals"][ln]["turns"] for ln in (LANE_MAIN, LANE_AUX, LANE_BLADEX)} == \
            {LANE_MAIN: 1, LANE_AUX: 1, LANE_BLADEX: 1}
        assert sorted(g["lane"] for g in r["groups"]) == [LANE_AUX, LANE_BLADEX, LANE_MAIN]
        assert r["totals"]["all"]["input"] == 30

    def test_inner_loop_source_value_matches_models(self):
        from bladex_proxy.models import INNER_LOOP_AUX_SOURCE
        from bladex_proxy.usage import BLADEX_SELF_AUX_SOURCES
        assert INNER_LOOP_AUX_SOURCE in BLADEX_SELF_AUX_SOURCES

    def test_groups_by_agent_model_session_day(self):
        u = {"prompt": 1, "completion": 1, "total": 2}
        rows = [("k1", _turn(T0, usage=u)),                                  # 10-24 23:30 CST
                ("k2", _turn(T0 + timedelta(hours=1), usage=u)),            # 10-25 00:30 CST
                ("k3", _turn(T0, usage=u, agent="b")),
                ("k4", _turn(T0, usage=u, model="m2")),
                ("k5", _turn(T0, usage=u, session="s2"))]
        r = aggregate(rows, tz=CST)
        keys = [(g["agent"], g["model"], g["session"], g["day"]) for g in r["groups"]]
        assert keys == sorted(keys) and len(keys) == 5
        assert ("a", "m", "s", "2026-10-24") in keys and ("a", "m", "s", "2026-10-25") in keys
        assert r["window"]["tz"] == "+08:00"
        # 同一批轮次按 UTC 切天：两轮都落在 10-24
        days_utc = {g["day"] for g in aggregate(rows, tz=UTC)["groups"] if g["agent"] == "a"
                    and g["model"] == "m" and g["session"] == "s"}
        assert days_utc == {"2026-10-24"}

    def test_window_is_half_open_and_filters(self):
        u = {"prompt": 1, "completion": 0, "total": 1}
        rows = [("k1", _turn(T0, usage=u)), ("k2", _turn(T0 + timedelta(minutes=10), usage=u)),
                ("k3", _turn(T0 + timedelta(minutes=10), usage=u, agent="b"))]
        r = aggregate(rows, since=T0, until=T0 + timedelta(minutes=10), tz=CST)
        assert r["totals"]["all"]["turns"] == 1
        assert aggregate(rows, agent="b", tz=CST)["totals"]["all"]["turns"] == 1
        assert aggregate(rows, model="nope", tz=CST)["totals"]["all"]["turns"] == 0
        assert aggregate(rows, session="s", tz=CST)["totals"]["all"]["turns"] == 3

    def test_pure_same_input_same_output(self):
        rows = [("k1", _turn(T0, usage={"prompt": 3, "completion": 1, "total": 4}))]
        assert aggregate(rows, tz=CST) == aggregate(list(rows), tz=CST)


# ── 从 Hub 读：iter_hub_turns + 墓碑 + U2 A1 复算 ───────────────────────────

@pytest.fixture
def hub(tmp_path):  # noqa: ANN001, ANN201
    from bladex_proxy.storage.memory_hub import MemoryHub
    h = MemoryHub(tmp_path / "hub")
    h.open()
    yield h
    h.close()


def _put(hub, key: str, *, agent: str, session: str, model: str, ts: datetime,  # noqa: ANN001
         usage: dict, failed: bool = False) -> None:
    from bladex_proxy.models import Identity, ResponseMeta, Turn, TurnStatus
    hub.put(key, Turn(identity=Identity(user_id="u", agent_id=agent, session_id=session),
                      model=model, ts=ts, status=TurnStatus.FAILED if failed else TurnStatus.OK,
                      response_meta=ResponseMeta(usage=usage)))


class TestFromHub:
    def test_tombstoned_turn_is_skipped(self, hub):
        from bladex_proxy.models import TombstoneTargetType
        ts = datetime(2026, 10, 25, 1, 0, tzinfo=UTC)
        _put(hub, "u/a/s/1-0", agent="a", session="s", model="m", ts=ts, usage={"prompt": 5})
        _put(hub, "u/a/s/2-0", agent="a", session="s", model="m", ts=ts, usage={"prompt": 7})
        hub.append_tombstone(TombstoneTargetType.TURN, "u/a/s/2-0")
        t = aggregate(iter_hub_turns(hub), tz=CST)["totals"]["all"]
        assert (t["turns"], t["input"]) == (1, 5)

    def test_u2_a1_input_recomputed_from_hub(self, hub):
        doc = json.loads((FIX / "ac_u2_a1_capture_rows.json").read_text())
        rows = doc["rows"]
        assert len(rows) == 208
        for i, r in enumerate(rows):
            _put(hub, f"u/{r['agent']}/{r['session_id']}/{i:04d}-0", agent=r["agent"],
                 session=r["session_id"], model=r["model"],
                 ts=datetime.fromtimestamp(r["t_done_epoch"], UTC), usage=r["usage"],
                 failed=bool(r["error"]) or not r["done"])
        out = aggregate(iter_hub_turns(hub), since=datetime(2026, 9, 28, 9, 50, 40, tzinfo=CST),
                        until=datetime(2026, 9, 28, 11, 0, 20, tzinfo=CST), tz=CST)
        t = out["totals"]["all"]
        assert t["input"] == 20_850_495, "U2 §1.1 A1 Σ input"
        assert t["output"] == 358_693, "U2 §1.1 A1 Σ output"
        assert (t["turns"], t["failed_turns"]) == (208, 14), "U2 §1.1：上游轮 208、限流 14"
        assert t["input_nodata"] == 14, "限流轮没有 usage ⇒ NO-DATA，不计 0"
        assert t["cache_read_nodata"] == 208, "批 AC 之前的轮次没有原文 ⇒ cache_read 全 NO-DATA"
        assert out["totals"][LANE_AUX]["turns"] == 0 and out["totals"][LANE_BLADEX]["turns"] == 0
        assert len({g["session"] for g in out["groups"]}) == 3, "U2 §1.1：会话 id 3"
