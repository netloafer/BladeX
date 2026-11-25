"""批 DS1b（MQ-I41）：`GET /admin/matters?include=members` 的批量路径**一次**全扫 meta 按 Matter 分组。

判据按**全扫次数**计，不按秒计（夹具上秒数说明不了生产）：给 meta RocksDB 句柄包一层计数，
数 `items()` / `keys()` / `values()` 的调用。逐个取（DS1a）每张卡 `get_edges` + `get_facts_for_matter`
各全扫一遍 ⇒ N 张卡 2N 遍；修后批量取成员只扫 1 遍，与卡数无关。
逐字相同由 `test_ds1a_matters_members.py` 钉（同一个序列化函数，两条路逐字相同），这里再对一遍。
"""

from __future__ import annotations

import tempfile

from bladex_proxy.admin_read import _matter_members_payload
from bladex_proxy.config import ProxyConfig
from bladex_proxy.server import create_app
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from test_ds1a_matters_members import _seeded_index

_SCANS = ("items", "keys", "values")


class _CountingDb:
    """meta RocksDB 句柄的计数代理：只数全库迭代，其余照转。"""

    def __init__(self, db) -> None:  # noqa: ANN001
        self._db = db
        self.scans = 0

    def __getattr__(self, name):  # noqa: ANN001, ANN204
        attr = getattr(self._db, name)
        if name in _SCANS:
            def counted(*a, **kw):  # noqa: ANN002, ANN003, ANN202
                self.scans += 1
                return attr(*a, **kw)
            return counted
        return attr

    def __getitem__(self, k):  # noqa: ANN001, ANN204
        return self._db[k]

    def __contains__(self, k):  # noqa: ANN001, ANN204
        return k in self._db


def test_members_bulk_read_is_one_meta_scan_regardless_of_matter_count() -> None:
    idx = _seeded_index()
    try:
        counting = _CountingDb(idx._meta_db)
        idx._meta_db = counting
        ids = [f"m{m:02d}" for m in range(6)]
        members = _matter_members_payload(idx, ids, 30)
        assert set(members) == set(ids)
        assert counting.scans == 1, counting.scans
        # 逐个取仍是每张卡两遍（单个详情的取法本批不改）——对照组证明这把尺子量得到差别
        counting.scans = 0
        for mid in ids:
            idx.get_edges(mid)
            idx.get_facts_for_matter(mid, top_k=30)
        assert counting.scans == 2 * len(ids)
    finally:
        idx._meta_db = counting._db
        idx.close()


def test_members_endpoint_scans_twice_total_and_matches_details() -> None:
    """整条请求：列表 `all_matters` 一遍 + 成员一遍 = 2；逐个取的 6 张卡是 1 + 12 = 13。"""
    cfg = ProxyConfig(upstream_model="openai/test", upstream_api_key="sk-fake",
                      rocksdb_path=f"{tempfile.mkdtemp()}/rocksdb",
                      index_path=f"{tempfile.mkdtemp()}/index")
    app = create_app(cfg)
    idx = _seeded_index()
    counting = _CountingDb(idx._meta_db)
    idx._meta_db = counting
    with TestClient(app) as c:
        started = app.state.index
        app.state.index = idx
        try:
            r = c.get("/admin/matters?include=members&facts_top_k=30&limit=500")
            assert r.status_code == 200, r.text
            assert counting.scans == 2, counting.scans
            body = r.json()
            assert len(body["members"]) == 6
            for mid, payload in body["members"].items():
                d = c.get(f"/admin/matters/{mid}?facts_top_k=30")
                assert JSONResponse(content=payload).body == d.content, mid
        finally:
            app.state.index = started
            idx._meta_db = counting._db
            idx.close()
