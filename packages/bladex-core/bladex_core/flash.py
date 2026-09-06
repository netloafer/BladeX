"""Memory Flash 文件层的路径 / 落盘助手（ADR-0032 §5；原 ADR-0031 G12.4/G12.5 模块）。

2026-09-03 S5：ADR-0031 时代的老渲染器段（`MatterFlash` 六槽卡 / `render_matter_md` /
`assemble_injection` / `find_hub_entities` / `core_entities` / `filter_matter_by_exposure` /
`migrate_legacy_layout` / `prune_orphan_machine_files` / `prune_empty_dirs` 及 `matters/`
路径族）连同 `BLADEX_FLASH_RENDER` / `BLADEX_FLASH_MATTERS` 一起删除——ADR-0031 已废、
两开关默认关且 live 未设、`memory_index._render_flash` 是唯一调用点。素材在 git 历史。
本模块只剩 **被 `flash_daemon` / `ledger` / `flash_tree` 共用的路径与落盘助手**。

# 🔴 仍然有效的红线

1. **文件是投影，不是存储。** 真相在 Hub 事件流（ADR-0032 §5.2）；单写者 = `flash_daemon`。
2. **渲染必须是纯函数**（无墙上时钟），删树重跑逐字一致。
3. **用户手写区永不被覆盖**：`X.md` 机器整体重写，`X.local.md` 一个字不碰。
4. 路径成分来自外部输入，一律过 `safe_component`（单射清洗，防穿越、防撞名）。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass

# 本模块被 consolidator / flash_daemon 进程加载，但也可能被脚本/测试直接 import：
# 按工程规范"随宿主"，库侧用 stdlib logging（structlog 归 BladeX 自有进程的入口）。
logger = logging.getLogger(__name__)

# ── 文件布局 ────────────────────────────────────────────────────────────────

#: 用户手写覆盖文件的后缀。`X.md` 是机器渲染（每次整体重写），`X.local.md` 是用户的。
LOCAL_SUFFIX = ".local.md"

#: 文件名里不安全的成分一律替换（principal / agent / scope 来自外部输入，
#: 未经清洗直接拼路径 = 目录穿越）。matter_id 形如 `m-<hash12>` 本就安全，
#: 但清洗对它是恒等式，所以统一走同一个函数、不留"这个不用洗"的例外。
_UNSAFE = re.compile(r"[^0-9A-Za-z._\-]+")
_MAX_COMPONENT = 80
_FP_LEN = 8


def _fingerprint(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()[:_FP_LEN]


def safe_component(raw: str) -> str:
    """把一个 id 洗成安全的单层路径成分。**清洗有损时补指纹，保证单射。**

    三条清洗：① 非 `[0-9A-Za-z._-]` 一律折成 `_`（路径分隔符因此消失，穿越无从谈起）；
    ② **不许以 `.` 开头**——既挡掉 `.` / `..`，也顺手挡掉隐藏文件
    （用户打不开的目录等于没有"能打开看的文件夹"这条收益）；③ 截到 80 字符。
    空输入返回 `_`——**不返回空串**，否则 `root//USER.md` 会把文件写到上一层去。

    🔴 **单射（G12.12 ①）**：只做上面三条的话它是**多对一**的——`team:a-b` 与
    `team:a:b` 同折成 `team_a_b`，两个不同 team 的目录会合成一个。而 §6.9 选"文件级
    边界"的全部理由就是它比行级过滤更难写错；一个会撞名的目录名把这条理由直接抵消，
    且失效方式是静默的（两边都能读写，看不出异常）。故：**清洗结果与原串不逐字相等时，
    追加原串指纹**。真实 id（principal 的 hash8、`m-<hex>`、`claude-code`、`personal`）
    清洗是恒等式 ⇒ 一个字符都不变、存量目录零改名；只有本来就该被区分的病态输入才带尾巴。
    """
    s0 = raw or ""
    s = _UNSAFE.sub("_", s0.strip())[:_MAX_COMPONENT]
    if not s:
        s = "_"
    if s.startswith("."):
        s = f"_{s}"
    if s == s0:
        return s
    return f"{s[: _MAX_COMPONENT - _FP_LEN - 1]}-{_fingerprint(s0)}"


# ── 作用域目录（§6.9；G12.12 ①）────────────────────────────────────────────

#: 个人作用域的目录名。个人模式下**恒为**它 ⇒ 旧布局（`<principal>/…`）是新布局
#: （`<principal>/personal/…`）的特例，语义不变、只多一层。
SCOPE_PERSONAL = "personal"

#: `Fact.scope` / `Matter.scope` 的前缀（ADR-0021 §2.4）。从这里读，不在别处抄字面量。
SCOPE_PREFIX_PERSONAL = "personal:"
SCOPE_SHARED_PREFIXES: tuple[str, ...] = ("team:", "org:")

#: 拒绝落盘的信号。`scope_dir_of` 返回它 = "这条数据放哪儿我说不准" ⇒ 调用方**不写**。
SCOPE_DIR_REFUSED = ""


def scope_dir_of(scope: str, *, principal: str) -> str:
    """`Fact.scope` / `Matter.scope` → 作用域目录名；说不准就返回 `SCOPE_DIR_REFUSED`。

    ==============================  ===================  =============================
    scope                           目录                 说明
    ==============================  ===================  =============================
    ``""``（未标注，存量全是它）    ``personal``         它就在这个 principal 的树下
    ``personal:<principal>``        ``personal``
    ``personal:<别人>``             **拒绝**             异常数据，不落盘 + 告警
    ``team:x`` / ``org:y``          ``team-x`` / ``org-y``
    其它                            **拒绝**             未知形态一律 fail-closed
    ==============================  ===================  =============================

    🔴 **空 scope 在这里比行层更严，是有意的**：`_matter_visible` 对空 scope
    fail-open（"存量 120/120 都是空，过滤掉等于把②平面清零"，是迁移兼容）。
    文件层不需要那个兼容——它本来就按 principal 分根，未标注的数据只可能是这个
    principal 自己的。**两处不一致的方向是"文件层更严"**，即安全侧；
    `test_flash_scope_parity.py` 把这条差异当**具名例外**钉住，而不是让它悄悄存在。
    """
    s = (scope or "").strip()
    if not s:
        return SCOPE_PERSONAL
    if s.startswith(SCOPE_PREFIX_PERSONAL):
        pid = s[len(SCOPE_PREFIX_PERSONAL):]
        return SCOPE_PERSONAL if pid == principal else SCOPE_DIR_REFUSED
    for prefix in SCOPE_SHARED_PREFIXES:
        if s.startswith(prefix):
            head = prefix[:-1]
            rest = s[len(prefix):]
            return f"{head}-{safe_component(rest)}" if rest else head
    return SCOPE_DIR_REFUSED


def readable_scope_dirs(
    pids: list[str] | None,
    visibility: list[str] | None,
    *,
    principal: str,
) -> list[str]:
    """请求者能读这个 principal 树下的哪几个作用域目录（注入侧选目录，§6.9）。

    🔴 **不在这里重写三态判定**：`pids` 必须由 `MemoryIndex._visibility_pids` 算好传进来
    ——`[]` / `None` / `["team:x"]` 三种"空"的语义差别害过一次（2026-07-28，Fact 散点
    召回 fail-closed 锁死 13/13 轮、潜伏 6 天），那套语义**只允许有一个实现**。
    本函数只做"pids/visibility → 目录名"的映射，且与 `_matter_visible` 的比对结果
    由 `test_flash_scope_parity.py` 逐格对账。

    `pids is None`（legacy 不过滤）时只给 `personal`：那条路径连 principal 都不确定，
    给不出更多目录也不该猜——同样是**更严的一侧**。
    """
    if pids is None:
        return [SCOPE_PERSONAL] if principal else []
    out: list[str] = []
    if principal and principal in pids:
        out.append(SCOPE_PERSONAL)
    for v in visibility or []:
        if v.startswith(SCOPE_SHARED_PREFIXES):
            d = scope_dir_of(v, principal=principal)
            if d and d not in out:
                out.append(d)
    return out


def principal_dir(root: str, principal: str) -> str:
    """`<flash-root>/<principal>/`——**作用域目录的父级**，本身不再直接放文件。"""
    return f"{_norm_root(root)}/{safe_component(principal)}"


def scope_dir(root: str, principal: str, scope_dir_name: str = SCOPE_PERSONAL) -> str:
    """`<flash-root>/<principal>/<scope>/`。所有文件路径都从这里长出来。"""
    return f"{principal_dir(root, principal)}/{safe_component(scope_dir_name)}"


def _norm_root(root: str) -> str:
    return (root or ".").rstrip("/") or "/"


def write_if_changed(path: str, content: str) -> int:
    """内容变了才写；返回 1 = 真写了。

    两个理由不做无脑覆盖：① 渲染是纯函数，绝大多数轮次内容一字不变，无脑写会让
    整棵树的 mtime 每两分钟跳一次——用户看目录时分不出"哪张卡真的动了"，
    编辑器与同步盘也跟着抖；② 少一次写就少一次被中断撕裂的机会。

    先写同目录临时文件再 `os.replace`：**替换是原子的**，读侧永远读到完整的一份。
    Flash 的读侧将来是注入热路径，**撕裂的注入包比没有更糟**。
    """
    try:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                if f.read() == content:
                    return 0
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
        return 1
    except OSError as e:
        logger.warning("flash_write_failed path=%s error=%s", path, e)
        return 0


@dataclass
class FlashTreeSummary:
    """一棵 Flash 树的体检读数（`bladex status` 与探针共用）。

    🔴 `newest_mtime` 是"**最近一次内容真的变了**"，**不是"最近一次渲染"**。
    两者不同，因为 `write_if_changed` 内容没变就不写——把它读成渲染龄，
    会把"这段时间没有新东西"误报成"渲染器停了"（尺子与被测对象的参照系脱钩，
    本仓库的高发病）。真正的渲染心跳要另立，且**不能放进 flash-root**：
    一个内容是时间戳的文件会让"删掉重渲染逐字一致"当场失效。

    2026-09-03 S5 起按 ADR-0032 布局计数：`ledgers`（`ledgers/<桶>/<id>.md` 账本真身）
    取代旧的 `cards`/`archived`（`matters/` 目录族已随老渲染器删除）。
    """

    principals: int = 0
    ledgers: int = 0      # ledgers/<桶>/<id>.md（账本池真身）
    user_files: int = 0   # *.local.md —— 用户手写的那些
    newest_mtime: float = 0.0


def summarize_tree(root: str) -> FlashTreeSummary:
    """扫一棵 flash-root，纯统计、不改任何东西。目录不存在返回全 0。"""
    if not os.path.isdir(root):
        return FlashTreeSummary()
    principals = ledgers = user_files = 0
    newest = 0.0
    try:
        principals = len([d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d))])
        for dirpath, _dirnames, filenames in os.walk(root):
            in_ledgers = "/ledgers/" in (dirpath.replace(os.sep, "/") + "/")
            for name in filenames:
                if name.endswith(LOCAL_SUFFIX):
                    user_files += 1
                elif name.endswith(".md") and in_ledgers:
                    ledgers += 1
                else:
                    continue
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(dirpath, name)))
                except OSError:
                    pass
    except OSError as e:
        logger.warning("flash_summary_failed root=%s error=%s", root, e)
    return FlashTreeSummary(principals=principals, ledgers=ledgers,
                            user_files=user_files, newest_mtime=newest)
