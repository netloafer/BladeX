"""V-I3：`bladex_memory_search` 四模式后端 + keyword 跨 fact/matter 关联。

模式优先级 fact_id > matter_id > keyword > query；全部过 exposure 守卫
（工具结果也是注入，ADR-0032 §3.2-7）；不可见与不存在同文案（不泄露存在性）。
keyword 的键空间 = V-I2 `canonical_entity_key`（fact.entities 与 matter.topic_keys
同一把尺）。
"""

from __future__ import annotations

import asyncio

import pytest
from bladex_proxy.agency import AgencyRuntime


class _F:
    def __init__(self, content, entities=(), ceiling="public", t_invalid=""):
        self.content = content
        self.entities = list(entities)
        self.exposure_ceiling = ceiling
        self.t_invalid = t_invalid


class _M:
    def __init__(self, matter_id, title, topic_keys=None, status="active"):
        self.matter_id = matter_id
        self.title = title
        self.topic_keys = topic_keys or {}
        self.status = status


class _FakeIndex:
    """duck-typed Index：记录调了哪条路（模式分流的行为断言用）。"""

    def __init__(self):
        self.calls: list[str] = []
        self.fact = _F("deepseek 无 id delta 病例", entities=["deepseek-v4-flash"])
        self.secret = _F("内网网关地址", ceiling="private")
        self.matter = _M("m-abc123", "拼接锚修复", {"splice": 3, "anchor": 2})

    def get_fact(self, fid):
        self.calls.append(f"get_fact:{fid}")
        return {"f1": self.fact, "sec": self.secret}.get(fid)

    def get_matter(self, mid):
        self.calls.append(f"get_matter:{mid}")
        return self.matter if mid == "m-abc123" else None

    def get_facts_for_matter(self, mid, top_k=5):
        self.calls.append("facts_for_matter")
        return [self.fact, self.secret]

    def keyword_lookup(self, key, **kw):
        self.calls.append(f"keyword:{key}")
        return [self.matter], [self.fact, self.secret]

    def search(self, query, top_k=10, session_id=""):
        self.calls.append("dense")
        return [self.fact]


def _run(idx, args, exposure="public"):
    rt = AgencyRuntime(index=idx, hub=None)
    return asyncio.run(rt._h_memory_search(
        args, allowed_exposure=exposure, context={"session_id": "s"}))


class TestModes:
    def test_query_mode_unchanged(self):
        idx = _FakeIndex()
        out = _run(idx, {"query": "锚"})
        assert "deepseek 无 id delta 病例" in out and "dense" in idx.calls

    def test_keyword_mode_returns_matters_and_facts(self):
        idx = _FakeIndex()
        out = _run(idx, {"keyword": "Splice"})
        assert "(matter m-abc123)" in out and "拼接锚修复" in out
        assert "deepseek 无 id delta 病例" in out
        # 规范化的单一权威在关联层（keyword_lookup 内部），handler 传原词——
        # 折叠行为由下方 TestKeywordMapIntegration 用真表证。
        assert "keyword:Splice" in idx.calls, idx.calls

    def test_matter_mode_lists_facts_and_topic_keys(self):
        idx = _FakeIndex()
        out = _run(idx, {"matter_id": "m-abc123"})
        assert "拼接锚修复" in out and "splice" in out
        assert "deepseek 无 id delta 病例" in out

    def test_fact_mode_returns_canonical_keywords(self):
        idx = _FakeIndex()
        out = _run(idx, {"fact_id": "f1"})
        assert "deepseek 无 id delta 病例" in out
        assert "deepseek-v4-flash" in out  # canonical 后的键

    def test_mode_priority_explicit_id_wins(self):
        idx = _FakeIndex()
        _run(idx, {"fact_id": "f1", "keyword": "x", "query": "y"})
        assert idx.calls == ["get_fact:f1"], idx.calls

    def test_no_args_is_a_loud_error(self):
        out = _run(_FakeIndex(), {})
        assert out.startswith("Error:") and "keyword" in out


class TestExposureGuard:
    """阴性对照：private 条目对 public 目的地在四模式下都不可见。"""

    def test_fact_mode_hides_private(self):
        out = _run(_FakeIndex(), {"fact_id": "sec"})
        assert "内网网关地址" not in out
        # 不泄露存在性：与不存在同文案
        assert out == _run(_FakeIndex(), {"fact_id": "nope"}).replace("'nope'", "'sec'")

    def test_matter_and_keyword_modes_filter_private(self):
        for args in ({"matter_id": "m-abc123"}, {"keyword": "splice"}):
            out = _run(_FakeIndex(), args)
            assert "内网网关地址" not in out, args

    def test_private_destination_sees_private(self):
        out = _run(_FakeIndex(), {"fact_id": "sec"}, exposure="private")
        assert "内网网关地址" in out


class TestKeywordMapIntegration:
    """倒排真表：canonical 折叠（qwen 分隔符病例同族）+ 行数戳失效。"""

    def test_map_over_real_lancedb(self, tmp_path):
        pytest.importorskip("lancedb")
        import lancedb
        import pyarrow as pa
        from bladex_proxy.storage.memory_index import MemoryIndex

        idx = MemoryIndex(str(tmp_path / "index"))
        db = lancedb.connect(str(tmp_path / "index"))
        schema = pa.schema([("id", pa.string()), ("entities", pa.list_(pa.string()))])
        tbl = db.create_table("facts", schema=schema)
        tbl.add([{"id": "fa", "entities": ["qwen3.8:27b"]},
                 {"id": "fb", "entities": ["qwen3.8-27B"]},
                 {"id": "fc", "entities": ["别的"]}])
        idx._lancedb = db
        idx._table = tbl
        m = idx._keyword_fact_map()
        from bladex_core.topic_keys import canonical_entity_key
        key = canonical_entity_key("qwen3.8:27b")
        assert sorted(m.get(key, [])) == ["fa", "fb"], (
            "分隔符变体必须折到同一 keyword（V-I2 病例）")
        # 行数戳：不变则复用缓存对象
        assert idx._keyword_fact_map() is m
        tbl.add([{"id": "fd", "entities": ["qwen3.8:27b"]}])
        m2 = idx._keyword_fact_map()
        assert m2 is not m and "fd" in m2.get(key, [])
