"""模块注册表 —— core 的模块清单/开关/依赖校验（ADR-0032 §2.1；批一卡 V-A1）。

v5 代码结构 = core(gateway) + Router(必选) + 插件模块，**编译期组合**：
模块是代码里的显式调用点（本表声明），不是运行期事件总线——热路径预算吃不起
动态分发（ADR-0030 否决 Cordis 移植的物理约束）。用户面只有开/关：
`BLADEX_MODULE_<NAME>=0/1`（默认值集中本表，与 `config/.env.example` 由测试对账，
刚性原则 12：同一参数不许在两条路径上各有默认值）。

# 🔴 三条 fails-loud（misconfiguration 拒启动，不静默跳过）

1. **Router 必选、无开关**：`BLADEX_MODULE_ROUTER` 配了关也拒启动——
   "关掉 Router" 不是一个合法形态，配置里写出来就是错误，要在启动时炸给用户看。
2. **未知模块名拒启动**：`BLADEX_MODULE_*` 前缀下出现本表没有的名字（多半是拼错）
   → 拒启动。静默忽略拼错的开关 = 用户以为关了实际开着（或反之）。
3. **依赖不满足拒启动**：开着的模块依赖的模块被关 → 拒启动，报错点名缺哪个。

导入无副作用（ADR-0027 踩坑）：本模块只有纯函数，校验由 server lifespan 显式调用。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

ENV_PREFIX = "BLADEX_MODULE_"

_FALSY = {"0", "false", "no", "off"}
_TRUTHY = {"1", "true", "yes", "on"}


class ModuleConfigError(RuntimeError):
    """模块配置违例。lifespan 捕到即拒启动（与 identity registry 校验同一形态）。"""


@dataclass(frozen=True)
class ModuleSpec:
    """一个模块的编译期声明。`required=True` ⇒ 恒开且不许配开关（fails-loud 1）。"""

    name: str
    default_enabled: bool
    required: bool = False
    depends: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


#: 模块清单（编译期唯一真相源）。新增模块 = 加一行 + 在调用点接 `module_enabled`。
#: 默认形态（一个 BLADEX_MODULE_* 都不配）必须等于**生产形态**。
#: v5 四模块（账本/Flash/工具面/拦截）2026-09-02 翻默认开（Jason 拍板选项①，
#: `docs/planning/task-modules-default-on-20260902.md`）：live `.env` 四个已 =1 跑了 8 天，
#: 五 agent 应开者 4/4 有 live 记录、N1a 过渡读数 19.4%、播报回带零泄漏三层证明——
#: 翻默认改的是"默认形态 = 生产形态"，不改行为。`=0` 是各模块的回滚通道。
#: 仍默认关的只剩 embed（举证 = MQ-W8 复测，触发式）。
MODULE_SPECS: dict[str, ModuleSpec] = {s.name: s for s in (
    ModuleSpec("router", default_enabled=True, required=True,
               note="代理转发与路由策略（ADR-0032 §2.1：必选，无开关）"),
    ModuleSpec("inject", default_enabled=True,
               note="记忆注入平面（prefetch/硬规则/卡）"),
    ModuleSpec("embed", default_enabled=False,
               note="V-A6 embedding 独立模块（IPC+hot/bulk 队列）；开=proxy 与 "
                    "consolidator 都当客户端。默认关=沿用既有 local/proxy 档位，"
                    "翻默认的举证 = MQ-W8 两个病例复测通过"),
    ModuleSpec("sensitivity", default_enabled=True,
               note="敏感度解析与守卫（ADR-0021；细粒度仍由 [strategies.sensitivity] 配）"),
    ModuleSpec("assembly", default_enabled=True,
               note="上下文装配（ADR-0019；细粒度仍由 BLADEX_ASSEMBLY_* 配）"),
    ModuleSpec("ledger", default_enabled=True,
               note="五段账本（ADR-0032 §4）；2026-09-02 翻默认开"),
    ModuleSpec("flash", default_enabled=True,
               note="Memory Flash 注入投影 / 维护进程；2026-09-02 翻默认开。"
                    "（consolidator 侧老渲染器旋钮 BLADEX_FLASH_RENDER 已于 09-03 S5 删除，"
                    "本模块门是 Flash 的唯一开关。）"),
    ModuleSpec("toolface", default_enabled=True,
               note="bladex_* 工具面（V-P1）；2026-09-02 翻默认开"),
    ModuleSpec("interception", default_enabled=True, depends=("toolface",),
               note="拦截协议：内循环/剥离-拼接（V-P2–P4）；2026-09-02 翻默认开"),
    # ── V-A3（09-06 批 F0.2）：两个"边缘"模块补进注册表，默认全开 = 现状。
    #    `mcp`（bladex-mcp 读写面）**没有**加：它与 proxy 进程零耦合点（proxy 代码不 import
    #    bladex_mcp、也不为它登记任何路由），一个没有调用点的开关就是假开关。
    ModuleSpec("export", default_enabled=True,
               note="Obsidian/Postgres 连续导出 worker（consolidator 内）+ CLI connector/export/import；"
                    "关=consolidator 不构造 worker、三组命令退出 2（sync_control 心跳不受影响）"),
    ModuleSpec("admin_read", default_enabled=True,
               note="只读 admin API（/admin/status /admin/matters …）+ /dashboard；"
                    "关=只读面 404、写端点（/admin/ledgers/user-edit 等）照常"),
)}


def _env_key(name: str) -> str:
    return f"{ENV_PREFIX}{name.upper()}"


def _parse_bool(raw: str, *, key: str) -> bool:
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    raise ModuleConfigError(f"{key}={raw!r} is not a boolean (use 1/0/true/false)")


def resolve_module_switches(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """env 覆盖后的开关表。**只解析不校验**——校验归 `validate_modules`。"""
    e = os.environ if env is None else env
    out: dict[str, bool] = {}
    for name, spec in MODULE_SPECS.items():
        raw = e.get(_env_key(name))
        out[name] = spec.default_enabled if raw is None else _parse_bool(raw, key=_env_key(name))
    return out


def validate_modules(env: Mapping[str, str] | None = None) -> dict[str, bool]:
    """启动校验（fails-loud 三条）。通过则返回开关表，违例抛 `ModuleConfigError`。"""
    e = os.environ if env is None else env
    # 2) 未知模块名（拼错的开关）
    known = {_env_key(n) for n in MODULE_SPECS}
    unknown = sorted(k for k in e if k.startswith(ENV_PREFIX) and k not in known)
    if unknown:
        raise ModuleConfigError(
            f"unknown module switch(es): {', '.join(unknown)} "
            f"(known modules: {', '.join(sorted(MODULE_SPECS))})")
    switches = resolve_module_switches(e)
    # 1) 必选模块：不许配开关（配了开也不行——那个开关不该存在）
    for name, spec in MODULE_SPECS.items():
        if spec.required and e.get(_env_key(name)) is not None:
            raise ModuleConfigError(
                f"module '{name}' is mandatory and has no switch; "
                f"remove {_env_key(name)} from the environment")
    # 3) 依赖闭包
    for name, spec in MODULE_SPECS.items():
        if not switches[name]:
            continue
        missing = [d for d in spec.depends if not switches.get(d)]
        if missing:
            raise ModuleConfigError(
                f"module '{name}' is enabled but depends on disabled module(s): "
                f"{', '.join(missing)} (enable them or disable '{name}')")
    return switches


def module_enabled(name: str, env: Mapping[str, str] | None = None) -> bool:
    """调用点查询（每请求 O(1) env 读，微秒级；不做缓存——测试与热重载语义更简单）。

    未声明的名字直接抛错：调用点拼错模块名必须在第一次执行就炸，
    而不是永远返回 False 假装模块关着（静默失效是本仓的高发病）。
    """
    if name not in MODULE_SPECS:
        raise ModuleConfigError(f"module '{name}' is not declared in MODULE_SPECS")
    e = os.environ if env is None else env
    spec = MODULE_SPECS[name]
    raw = e.get(_env_key(name))
    if raw is None:
        return spec.default_enabled
    return _parse_bool(raw, key=_env_key(name))
