"""批 AC2 步骤 0.5：两件前置小修的钉住用例。

1. **MQ-R16 尾巴（批 DS1b 复核 Y2）**：三条读原始 JSON 的 admin 写端点，`_strict_body(...)` 的允许名单
   没有参照物——给 bind-legacy 名单多加一个 `dryrun`，DS1b 的守卫照样全绿（有模型的端点拿模型字段对账，
   原始 JSON 入口没有模型）。这里把参照物定为「端点实际读取的键」：用 AST 从端点函数体里读出
   `body.get("…")` / `(body or {}).get("…")` / `body["…"]` 的字面量，要求允许名单**等于**它，
   并钉死三条的取值，不多不少。
2. **MQ-W26**：`embedding._QueryLRU` 的命中（查在不在 → `move_to_end`）与另一线程的淘汰交错 ⇒ `KeyError`。
   用确定性构造复现：命中线程停在 `move_to_end` 入口，另一线程在这段时间里写入并淘汰同一个 key。
"""

from __future__ import annotations

import ast
import inspect
import tempfile
import textwrap
import threading
from collections import OrderedDict

import pytest
from bladex_proxy.config import ProxyConfig
from bladex_proxy.embedding import _QueryLRU
from bladex_proxy.server import create_app
from fastapi.routing import APIRoute
from pydantic import BaseModel

# ── 1. 原始 JSON 入口的允许名单 ─────────────────────────────────────────────

_RAW_JSON_EXPECTED = {
    "/admin/ledgers/user-edit": frozenset({"ledger", "expected_rev"}),
    "/admin/audience/set": frozenset({"fact_id", "audience"}),
    "/admin/ledgers/bind-legacy-matters": frozenset({"dry_run"}),
    # 批 BB · 2b：矛盾标记 / 裁定（只追加事件；载荷校验在 consistency.parse_payload / build_verdict_payload）
    "/admin/contradictions": frozenset({"pair_key", "payload"}),
    "/admin/contradictions/{pair_id}/settle": frozenset({"verdict", "origin", "note"}),
}
_WRITE = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _config() -> ProxyConfig:
    tmpdir = tempfile.mkdtemp(prefix="ac2_prereq_")
    return ProxyConfig(
        upstream_model="openai/test", upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb", index_path=f"{tmpdir}/index",
    )


def _strict_allowed_of(route: APIRoute) -> frozenset[str] | None:
    for dep in route.dependant.dependencies:
        allowed = getattr(dep.call, "strict_body_allowed", None)
        if allowed is not None:
            return allowed
    return None


def _has_body_model(route: APIRoute) -> bool:
    for p in route.dependant.body_params:
        t = p.field_info.annotation
        if isinstance(t, type) and issubclass(t, BaseModel):
            return True
    return False


def _is_body(node: ast.AST) -> bool:
    """`body` 或 `body or {...}`。"""
    if isinstance(node, ast.Name) and node.id == "body":
        return True
    return (isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or)
            and isinstance(node.values[0], ast.Name) and node.values[0].id == "body")


def body_keys_read(endpoint: object) -> frozenset[str]:
    """端点函数体里从 `body` 读出的字面量键（`.get("k")` 与 `["k"]`）。"""
    tree = ast.parse(textwrap.dedent(inspect.getsource(endpoint)))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and _is_body(node.func.value)
                and node.args and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.add(node.args[0].value)
        elif (isinstance(node, ast.Subscript) and _is_body(node.value)
              and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            keys.add(node.slice.value)
    return frozenset(keys)


def _raw_json_routes() -> dict[str, APIRoute]:
    """写端点里「没有 body 模型、允许名单非空」的 = 原始 JSON 入口（从路由表取，不另抄一份）。"""
    app = create_app(_config())
    out: dict[str, APIRoute] = {}
    for r in app.routes:
        if not (isinstance(r, APIRoute) and r.path.startswith("/admin")
                and set(r.methods) & _WRITE):
            continue
        allowed = _strict_allowed_of(r)
        if allowed and not _has_body_model(r):
            out[r.path] = r
    return out


def test_raw_json_entries_are_exactly_the_three_known() -> None:
    assert set(_raw_json_routes()) == set(_RAW_JSON_EXPECTED)


@pytest.mark.parametrize("path", sorted(_RAW_JSON_EXPECTED))
def test_raw_json_allowlist_equals_keys_the_endpoint_reads(path: str) -> None:
    """允许名单 == 端点实际读取的键 == 钉死的取值。名单宽了（Y2：多一个 `dryrun`）或端点多读一个键都红。"""
    route = _raw_json_routes()[path]
    allowed = _strict_allowed_of(route)
    read = body_keys_read(route.endpoint)
    assert read == _RAW_JSON_EXPECTED[path], f"{path} reads {sorted(read)}"
    assert allowed == read, (f"{path}: _strict_body allows {sorted(allowed or ())} "
                             f"but the endpoint reads {sorted(read)}")


def test_ast_reader_sees_the_three_access_forms() -> None:
    """尺子自检：三种读法都认，别的变量不认。"""
    def _ep(request: object) -> None:  # pragma: no cover —— 只供 AST 读
        body = {}
        other = {}
        body.get("a")
        (body or {}).get("b", 1)
        body["c"]
        other.get("x")

    assert body_keys_read(_ep) == {"a", "b", "c"}


# ── 2. MQ-W26：_QueryLRU 并发命中 × 淘汰 ─────────────────────────────────────


class _PausingDict(OrderedDict):
    """命中线程进 `move_to_end` 时停住，等主线程放行（其它线程不停）。"""

    def __init__(self, pause_thread: str) -> None:
        super().__init__()
        self.pause_thread = pause_thread
        self.paused = threading.Event()
        self.release = threading.Event()

    def move_to_end(self, key: str, last: bool = True) -> None:
        if threading.current_thread().name == self.pause_thread:
            self.paused.set()
            self.release.wait(5)
        super().move_to_end(key, last)


def test_query_lru_hit_interleaved_with_eviction_does_not_raise() -> None:
    lru = _QueryLRU(capacity=1)
    d = _PausingDict(pause_thread="hit")
    lru._data = d
    lru.put("a", [1.0])  # 主线程写，不停
    errors: list[BaseException] = []
    got: list[object] = []

    def _hit() -> None:
        try:
            got.append(lru.get("a"))
        except BaseException as e:  # noqa: BLE001 —— 记下来给断言
            errors.append(e)

    def _evict() -> None:
        try:
            lru.put("b", [2.0])  # capacity=1 ⇒ 淘汰 "a"
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    t_hit = threading.Thread(target=_hit, name="hit")
    t_hit.start()
    assert d.paused.wait(5), "hit thread never reached move_to_end"
    # 命中线程已判定 "a" 在、停在 move_to_end 入口。无锁 ⇒ 淘汰线程此刻跑完、把 "a" 删掉；
    # 有锁 ⇒ 淘汰线程卡在锁上，join 超时后放行命中线程，两者先后完成。
    t_ev = threading.Thread(target=_evict, name="evict")
    t_ev.start()
    t_ev.join(0.3)
    d.release.set()
    t_hit.join(5)
    t_ev.join(5)
    assert not t_hit.is_alive() and not t_ev.is_alive()
    assert errors == [], f"interleaved hit/evict raised: {errors!r}"
    assert got == [[1.0]]
    assert list(d) == ["b"]
