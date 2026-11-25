"""批 DS1a 守卫（MQ-W24）：`/admin/*` 路由体里不许直接调 Hub / Index 句柄。

判据（AST，不跑代码）：在 `async def` 路由函数体内——
  ① 调用 Hub / Index 句柄的方法（`hub.count()`、`request.app.state.index.get_fact(...)`）；
  ② 把句柄当参数直接传给一个同步函数（`build_agent_claim_map(hub)`）。
都算"直接调用"，除非它在**交给 `run_blocking` 的**内层函数 / lambda 里，或者就是
`run_blocking(...)` 本身的参数（`run_blocking(index.get_fact, fact_id)`）。

扫描面 = `bladex_proxy` 包里**所有**注册 `/admin/...` 路径的路由，不限于 `admin_read.py` /
`server/admin_api.py`——以后加在别处的新路由（如批 AC 的 `/admin/usage`）同样受约束。

写路由：DS1a 时 18 条整条豁免；批 DS1b（MQ-W25）移走后，只剩**逐个调用**登记在
`_ON_LOOP_WITH_REASON`（每条带理由），并要求登记项**仍然**存在——修好一个就必须从清单里删掉，
清单只能变短。GET 路由一律零容忍。

已知盲区（写进结算稿）：只认名字叫 hub / index（及从 `.hub` / `.index` / `getattr(…, "hub")`
赋值而来）的变量；句柄的**属性读取**（非调用，如 `index.last_rebuild_backlog`）不算。
"""

from __future__ import annotations

import ast
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1] / "bladex_proxy"
_HANDLE_ATTRS = frozenset({"hub", "index"})
_ROUTE_METHODS = frozenset({"get", "post", "put", "delete", "patch"})
_OFFLOAD = "run_blocking"

#: 写路由里**有理由**留在事件循环上的直接调用：`{路由: {调用: 理由}}`。只许删、不许加；登记的
#: 调用必须仍然存在（修好了就删）。批 DS1b（MQ-W25）把 DS1a 的 18 条"整条路由豁免"收成
#: "逐个调用豁免"——同一路由里新加的全扫调用照样红。理由的全文见 DS1b 结算稿 §3。
_ON_LOOP_WITH_REASON: dict[str, dict[str, str]] = {
    "server/admin_api.py::delete_ledger": {
        "hub.append_tombstone()": (
            "账本墓碑单条 put（亚毫秒）与随后的摘池 / 解激活之间不能让出 await："
            "让出一次，热路径就可能在'Hub 已墓碑、池里还在'的窗口里写这本账本"),
    },
}


def _call_of(violation: str) -> str:
    """`'L2045 hub.append_tombstone()'` → `'hub.append_tombstone()'`（行号会漂，不进登记）。"""
    return violation.split(" ", 1)[1] if violation[:1] == "L" else violation


def _route_info(fn: ast.AsyncFunctionDef) -> tuple[str, str] | None:
    """(method, path) if `fn` is decorated `@app.<method>("/admin/...")`."""
    for dec in fn.decorator_list:
        if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                and dec.func.attr in _ROUTE_METHODS and dec.args
                and isinstance(dec.args[0], ast.Constant)
                and str(dec.args[0].value).startswith("/admin")):
            return dec.func.attr, str(dec.args[0].value)
    return None


def _is_handle_expr(node: ast.AST, handles: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in handles
    if isinstance(node, ast.Attribute):
        return node.attr in _HANDLE_ATTRS
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "getattr" and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant) and node.args[1].value in _HANDLE_ATTRS):
        return True
    return False


def _handle_names(fn: ast.AsyncFunctionDef) -> set[str]:
    handles = set(_HANDLE_ATTRS)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(fn):
            targets: list[ast.expr] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            if value is None:
                continue
            if _is_handle_expr(value, handles):
                for t in targets:
                    if isinstance(t, ast.Name) and t.id not in handles:
                        handles.add(t.id)
                        changed = True
    return handles


def _is_offload_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == _OFFLOAD)


def _violations_in(fn: ast.AsyncFunctionDef) -> list[str]:
    handles = _handle_names(fn)
    offloaded_names: set[str] = set()
    offloaded_nodes: set[int] = set()
    for node in ast.walk(fn):
        if _is_offload_call(node):
            for a in [*node.args, *(k.value for k in node.keywords)]:
                offloaded_nodes.add(id(a))
                if isinstance(a, ast.Name):
                    offloaded_names.add(a.id)
    out: list[str] = []

    def visit(node: ast.AST) -> None:
        if id(node) in offloaded_nodes:
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            if node.name in offloaded_names:
                return
        if isinstance(node, ast.Call) and not _is_offload_call(node):
            f = node.func
            if isinstance(f, ast.Attribute) and _is_handle_expr(f.value, handles):
                out.append(f"L{node.lineno} {ast.unparse(f)}()")
            else:
                for a in [*node.args, *(k.value for k in node.keywords)]:
                    if _is_handle_expr(a, handles) and not (
                            isinstance(f, ast.Name) and f.id == "getattr"):
                        out.append(f"L{node.lineno} {ast.unparse(f)}(<handle>)")
                        break
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in fn.body:
        visit(stmt)
    return out


def scan_source(src: str, rel: str) -> dict[str, tuple[str, list[str]]]:
    """{'<rel>::<fn>': (method, [violations])} for every /admin route in `src`."""
    found: dict[str, tuple[str, list[str]]] = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.AsyncFunctionDef):
            info = _route_info(node)
            if info is not None:
                found[f"{rel}::{node.name}"] = (info[0], _violations_in(node))
    return found


def scan_package() -> dict[str, tuple[str, list[str]]]:
    out: dict[str, tuple[str, list[str]]] = {}
    for f in sorted(_PKG.rglob("*.py")):
        src = f.read_text(encoding="utf-8")
        if "/admin" not in src:
            continue
        out.update(scan_source(src, f.relative_to(_PKG).as_posix()))
    return out


def test_admin_routes_do_not_touch_hub_or_index_on_the_event_loop() -> None:
    routes = scan_package()
    assert len(routes) >= 50, f"scanned too few /admin routes ({len(routes)}) -- scanner broken?"
    bad = {k: left for k, (_m, v) in routes.items()
           if (left := [c for c in v if _call_of(c) not in _ON_LOOP_WITH_REASON.get(k, {})])}
    assert not bad, (
        "these /admin routes call Hub / Index directly inside `async def` -- that runs on the "
        "event loop and stalls every agent request (MQ-W24). Wrap the sync part and call it "
        f"through `bladex_proxy.server.blocking.run_blocking`:\n{bad}")


def test_get_routes_are_never_allowlisted() -> None:
    routes = scan_package()
    gets = {k for k, (m, _v) in routes.items() if m == "get"}
    assert not gets & set(_ON_LOOP_WITH_REASON)


def test_allowlist_only_shrinks() -> None:
    """登记的调用必须仍然存在且仍然违规——修好了就删掉登记，不许留着当豁免；每条都要写理由。"""
    routes = scan_package()
    stale = sorted(f"{k} :: {c}" for k, calls in _ON_LOOP_WITH_REASON.items() for c in calls
                   if k not in routes or c not in {_call_of(v) for v in routes[k][1]})
    assert not stale, f"no longer violating / no longer exists -- remove from allowlist: {stale}"
    assert all(r.strip() for calls in _ON_LOOP_WITH_REASON.values() for r in calls.values())


_INJECTED_ROUTE = '''

def _register_extra(app, admin_dep):
    @app.get("/admin/usage")
    async def admin_usage(request, cfg=admin_dep):
        hub = request.app.state.hub
        return {"turns": hub.count()}

    @app.get("/admin/usage2")
    async def admin_usage2(request, cfg=admin_dep):
        index = getattr(request.app.state, "index", None)
        return {"n": len(summarize(index))}

    @app.get("/admin/usage_ok")
    async def admin_usage_ok(request, cfg=admin_dep):
        hub = request.app.state.hub

        def _read():
            return hub.count()

        return {"turns": await run_blocking(_read), "n": await run_blocking(hub.count)}
'''


def test_guard_reports_an_injected_violating_route() -> None:
    """往 admin_read 源码里加违规路由 ⇒ 守卫必须报出它（反向验证守卫不是摆设）。"""
    rel = "admin_read.py"
    src = (_PKG / rel).read_text(encoding="utf-8") + _INJECTED_ROUTE
    routes = scan_source(src, rel)
    assert routes[f"{rel}::admin_usage"][1], routes[f"{rel}::admin_usage"]
    assert routes[f"{rel}::admin_usage2"][1], routes[f"{rel}::admin_usage2"]
    assert routes[f"{rel}::admin_usage_ok"][1] == []
    bad = {k for k, (_m, v) in routes.items() if v}
    assert bad == {f"{rel}::admin_usage", f"{rel}::admin_usage2"}
