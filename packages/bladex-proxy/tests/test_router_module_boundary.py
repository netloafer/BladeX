"""V-A2 守卫：Router 模块边界（ADR-0032 §2.1，Router = 必选模块）。

剥离的形态是**边界收窄 + 守卫**，不是物理搬文件——编译期组合不要求搬迁，
搬迁只会留下 shim 垃圾。模块文件集：

    route.py（门面：init_router / resolve_route / call_model / make_success_hook）
    router_sdk.py（网关：全仓唯一 vendor import 点，守卫在 test_router_sdk.py）
    routing_config.py / model_health.py（内部件）
    bladex_core/routing.py（策略引擎）

两条边界（都按路径读文本断言，不 import——沙盒无 native 依赖也能跑）：
① 内部件不外泄：`model_health` 只许 Router 模块文件 import。
② 编排面只经门面：server.py 不得直接 import `bladex_core.routing`
   （StickyOrigin/NoCapableCandidateError 类需求一律走 route.py 门面）。

已知合法的跨模块读（不在守卫范围，列出防误解）：
    inject.py / toolface.py / cli.py（调试命令）/ routing_config 消费方
    读 bladex_core.routing 的**类型**——类型共享不是边界泄漏；
    router_sdk 的 import 面本就开放（它是全仓调大模型的唯一入口）。
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_PROXY_SRC = _REPO / "packages/bladex-proxy/bladex_proxy"

#: ① 允许 import model_health 的生产文件（模块内部件的合法持有者）
_MODEL_HEALTH_ALLOWED = {"route.py", "model_health.py"}

_MH_IMPORT = re.compile(
    r"from\s+bladex_proxy\.model_health\s+import|import\s+bladex_proxy\.model_health")
_CORE_ROUTING_IMPORT = re.compile(
    r"from\s+bladex_core\.routing\s+import|import\s+bladex_core\.routing"
    r"|from\s+bladex_core\s+import\s+.*\brouting\b")


def _prod_py_files():
    for p in _PROXY_SRC.rglob("*.py"):
        if "__pycache__" in p.parts:
            continue
        yield p


def test_model_health_only_imported_by_router_module():
    offenders = [
        str(p.relative_to(_REPO))
        for p in _prod_py_files()
        if p.name not in _MODEL_HEALTH_ALLOWED
        and _MH_IMPORT.search(p.read_text(encoding="utf-8"))
    ]
    assert not offenders, (
        f"model_health 是 Router 模块内部件，只许 {sorted(_MODEL_HEALTH_ALLOWED)} import；"
        f"其余代码经 route.init_router 拿实例（V-A2）。违例: {offenders}")


def test_server_does_not_import_core_routing_directly():
    text = (_PROXY_SRC / "server.py").read_text(encoding="utf-8")
    hits = [m.group(0) for m in _CORE_ROUTING_IMPORT.finditer(text)]
    assert not hits, (
        "server.py（编排面）只许经 route.py 门面用路由能力——"
        f"直引 bladex_core.routing 违反 V-A2 边界: {hits}")


def test_facade_exports_declared_in_route_source():
    """门面完整性：server 需要的符号必须在 route.py 里有定义/再导出。

    按文本断言（不 import，绕 native 依赖）；符号真正可用性由
    test_route.py 的行为测试与全量回归守。
    """
    text = (_PROXY_SRC / "route.py").read_text(encoding="utf-8")
    for needle in ("def init_router(", "def make_success_hook(",
                   "NoCapableCandidateError"):
        assert needle in text, f"route.py 门面缺 {needle!r}"
