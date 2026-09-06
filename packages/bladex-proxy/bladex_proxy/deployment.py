"""部署根发现 + `config/.env` 加载 —— 全仓库唯一实现（2026-08-06 立）。

## 为什么要有这个模块

两个用户报的问题追到同一处根因：**"部署根在哪"和"`.env` 说了什么"此前有三份
互不相同的实现**，谁也不知道另外两份存在。

| 位置 | 根怎么找 | 症状 |
|---|---|---|
| `cli.py:_repo_root` | `BLADEX_HOME` → cwd | 换个目录（甚至仓库子目录）就找不到配置 |
| `consolidator.py:_load_env_file` | 只认 cwd，**连 `BLADEX_HOME` 都不认** | 全局安装后 consolidator 永远读不到 `.env` |
| `admin_read.py:_declared_env` | `Path(__file__).parents[3]` | 装成 wheel 后指向 site-packages，漂移检测恒空 |

三份实现漂移的那一侧会在用户机上悄悄失效——这正是 CLAUDE.md 里"通用品借用、
护城河自建"之外的第三条：**同一件事只许有一处实现**。

## 两条语义（都被真实事故推着定）

**① 根发现**：`BLADEX_HOME` → cwd 及其**祖先目录**（同 git 的做法）→ `~/.bladex`。
此前只有前两个锚点里的 cwd 本身，所以 `cd packages && bladex status` 就会看到
一个空库——**症状不是报错而是"数据看起来消失了"**，比报错更吓人。

**② 环境变量优先于 `.env`**（默认，可关）。这是 12 因子应用的常规语义，也是本仓库
`_env(key, default)` 一直以来的口径：编排器（compose/systemd/CI）注入的值必须能压过
镜像里带的文件，否则容器化部署根本没法覆盖路径。

这条曾在 2026-08-06 早些时候被翻成"文件优先"，动机是两次事故：

  - 2026-08-05：`.env` 写 `BLADEX_HOTPATH_BUDGET_MS=200`、进程跑 150，
    每轮注入超时 1–3ms 被整段丢弃，记忆静默失效。
  - 2026-08-06：用户改了上游 base URL 与 API key、重启 consolidator，
    进程仍读残留 export → 大量 `distill_failed`（认证失败），提炼整段停摆。

**同日复盘后翻回来**（用户拍板）：两次事故的真正病根是「**冲突当时完全不可见**」，
不是优先级方向本身——症状都是"配置文件写对了、进程用的是别的值"，而唯一痕迹
要么没有、要么混在 info 流里。既然 `format_conflicts` 现在无条件把每个被遮蔽的键
连值一起 WARNING 出来，翻转优先级的收益就没了，只剩下"文件悄悄改写编排器注入的值"
这个新风险。

需要文件压过环境时用 `BLADEX_ENV_PRECEDENCE=file` 显式声明；只想保住个别键用
`BLADEX_ENV_KEEP=A,B`。

**无论哪个方向，冲突都必须可见**：`load_env_file` 把每个"文件与环境不一致"的键
原样返回，调用方负责报出来（密钥掩码）。静默旁路才是那两次事故真正的共犯。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: 部署根的判据：有 `config/.env` 就是它。
CONFIG_RELPATH = os.path.join("config", ".env")

#: 全局安装（`uv tool install` / `pip install`）时的默认部署根。
DEFAULT_HOME = "~/.bladex"

#: 这几个键决定"配置怎么加载"，不能由被加载的文件自己改写（会自指）。
#: `BLADEX_HOME` 更是决定了文件从哪来的——文件不许搬动自己。
_NEVER_OVERRIDE = frozenset({"BLADEX_HOME", "BLADEX_ENV_PRECEDENCE", "BLADEX_ENV_KEEP"})

#: 掩码触发词（大小写不敏感的子串匹配）。宁可多掩不可漏。
_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


@dataclass(frozen=True)
class EnvConflict:
    """`config/.env` 声明的值与进程环境里已有的值不一致。

    `applied=True` 表示文件赢了（`precedence=file`），`False` 表示环境赢了（默认）。
    两种都要报给用户看——"谁赢了"比"有没有冲突"更容易被误判，而那正是
    2026-08-05/06 两次事故里唯一缺的东西。
    """

    key: str
    env_value: str
    file_value: str
    applied: bool

    @property
    def winner(self) -> str:
        return self.file_value if self.applied else self.env_value

    @property
    def loser(self) -> str:
        return self.env_value if self.applied else self.file_value


def mask(key: str, value: str) -> str:
    """密钥掩码 —— 冲突报告会进日志和终端，不能把 API key 原样吐出来。"""
    upper = key.upper()
    if not any(hint in upper for hint in _SECRET_HINTS):
        return value
    if not value:
        return "(empty)"
    if len(value) <= 8:
        return f"***({len(value)} chars)"
    return f"{value[:4]}***({len(value)} chars)"


# ── 根发现 ──────────────────────────────────────────────────────────────────


def _expand(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path.strip()))


def _has_config(root: str) -> bool:
    return os.path.isfile(os.path.join(root, CONFIG_RELPATH))


def _ancestors(start: str) -> list[str]:
    """cwd 及其所有祖先（含自身），近的在前。

    向上走是 git/npm/uv 的通行做法：在仓库任何子目录里跑命令都该正常工作。
    """
    out: list[str] = []
    current = Path(_expand(start))
    for path in (current, *current.parents):
        out.append(str(path))
    return out


def anchors() -> list[str]:
    """根候选的三个锚点（顺序即优先级），不含祖先展开。"""
    out: list[str] = []
    home = os.environ.get("BLADEX_HOME", "").strip()
    if home:
        out.append(_expand(home))
    out.append(_expand(os.getcwd()))
    out.append(_expand(DEFAULT_HOME))
    # 去重保序（BLADEX_HOME 指向 cwd 时不重复列）
    seen: set[str] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def find_root() -> str | None:
    """返回含 `config/.env` 的部署根；找不到返回 None（调用方负责报错）。

    顺序：`BLADEX_HOME` → cwd 及其祖先 → `~/.bladex`。

    `BLADEX_HOME` 设了就**到此为止**，哪怕那里还没有 `config/.env`（返回 None）——
    不往下找。曾想过"过期路径就 fall through 回当前仓库"，但 `bladex init` 写的正是
    `deployment_root()`：fall through 会让"我指定了部署目录、在仓库里跑 init"
    悄悄把配置写进那个仓库。显式声明必须是终局答案，错了就报错并把它列在首位。
    """
    home = os.environ.get("BLADEX_HOME", "").strip()
    if home:
        expanded = _expand(home)
        return expanded if _has_config(expanded) else None
    for candidate in _ancestors(os.getcwd()):
        if _has_config(candidate):
            return candidate
    fallback = _expand(DEFAULT_HOME)
    if _has_config(fallback):
        return fallback
    return None


#: 用户敲命令时所在的目录。CLI 会 chdir 到部署根（否则 `data/`、`logs/`、PID 文件
#: 这些相对路径会落在用户当前目录，`bladex status` 在别处就报一个空库），但**用户
#: 在命令行上给的相对路径必须仍相对他敲命令的地方** —— `bladex snapshot export out.jsonl`
#: 写到部署根去是彻头彻尾的意外。由 `remember_invocation_cwd()` 在 chdir 前记下。
_invocation_cwd: str | None = None


def remember_invocation_cwd() -> str:
    """在任何 chdir 之前记住用户的当前目录（入口点调用，import 时不做）。"""
    global _invocation_cwd
    _invocation_cwd = _expand(os.getcwd())
    return _invocation_cwd


def invocation_cwd() -> str:
    """用户敲命令时的目录；没记过就退回当前 cwd（库调用场景）。"""
    return _invocation_cwd or _expand(os.getcwd())


def resolve_user_path(path: str) -> str:
    """把用户在命令行上给的相对路径钉回他敲命令的目录。空串原样返回（= 用默认值）。"""
    if not path:
        return path
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.join(invocation_cwd(), expanded)


def deployment_root() -> str:
    """部署根，找不到就退回 cwd（日志目录等相对路径要有个落点，不能是 None）。

    与 `find_root()` 的区别是问题不同：`find_root` 问"**现有**配置在哪"（用来读），
    这里问"东西该放哪"（用来写，`bladex init` 就靠它）。所以 `BLADEX_HOME` 指向
    一个还是空的目录时，这里照样返回它——init 正该在那儿建配置。

    **调用时求值**：import 时冻结会让 chdir 后仍读旧位置（2026-08-03 环境污染 bug）。
    """
    home = os.environ.get("BLADEX_HOME", "").strip()
    if home:
        return _expand(home)
    return find_root() or _expand(os.getcwd())


def search_paths() -> list[str]:
    """找不到配置时原样列给用户的候选位置（不静默用空目录）。

    只列三个锚点的 `config/.env`；祖先目录由调用方另加一行说明，
    否则在深路径下会刷屏（tmp 目录能有十几层）。
    """
    return [os.path.join(root, CONFIG_RELPATH) for root in anchors()]


# ── .env 解析与加载 ─────────────────────────────────────────────────────────


def unbalanced_quote_lines(path: str | os.PathLike[str]) -> list[tuple[int, str]]:
    """`config/.env` 里引号错配的行 → `[(行号, 键名)]`。

    🔴 为什么需要它（2026-09-01 live 病例）：`config/.env:105` 写成
    `BLADEX_MODULE_FLASH="1'` —— 开 `"` 收 `'`。后果**只在 shell 侧显形**：

    - Python 侧 `parse_env_file` 走 `.strip('"').strip("'")`，`"1'` → `1`，
      读出来正是本意，运行时毫无异常；
    - zsh `source config/.env` 从该行一路吞到 EOF，实测丢掉三个
      `BLADEX_ASSEMBLY_*` 键，而且**只报一句 `unmatched "`**。

    ⇒ **同一个文件，两个加载器读出两个不同的环境**。这正是刚性原则 12
    「同一个参数在两条调用路径上各有一个默认值 = 缺陷」的形状，与哪边更
    合理无关。宽容的 strip 不是稳健，是把错误藏起来（原则 13 同族）。

    只报不改：修值是用户的事，程序的责任是**别让它静默**。
    """
    bad: list[tuple[int, str]] = []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return bad
    for n, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        val = val.strip()
        if not val:
            continue
        # 只判首尾配对：`"…"` / `'…'` / 无引号都合法；一头有一头没有 = 错配。
        opens_d, closes_d = val.startswith('"'), val.endswith('"')
        opens_s, closes_s = val.startswith("'"), val.endswith("'")
        if len(val) < 2:
            if opens_d or opens_s:
                bad.append((n, key.strip()))
            continue
        if (opens_d and not closes_d) or (opens_s and not closes_s) \
                or (closes_d and not opens_d) or (closes_s and not opens_s):
            bad.append((n, key.strip()))
    return bad


def parse_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """解析 `config/.env`（**不**注入 os.environ——只读来比对时也用这个）。

    🔴 本函数保持**纯**：引号错配不在这里报。它在只读比对场景也被调用
    （`declared_env` / drift 测试 / 探针），在这里打日志会刷屏，而且
    "解析"与"报告"混在一起会让下一个人不敢改任一边。报告接在
    `cli._load_env_file()` —— 那条路本来就负责把配置问题打到 stderr。
    检出用 `unbalanced_quote_lines`。
    """
    out: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, val = line.partition("=")
        key = key.strip()
        if key:
            out[key] = val.strip().strip('"').strip("'")
    return out


def declared_env(root: str | None = None) -> dict[str, str]:
    """部署根的 `config/.env` 声明了什么（只读，不注入）。"""
    base = root if root is not None else deployment_root()
    return parse_env_file(os.path.join(base, CONFIG_RELPATH))


def _precedence(file_vals: dict[str, str]) -> str:
    """`env`（默认）| `file`。

    2026-08-06 用户拍板：**回到环境变量优先**（12 因子应用的常规语义，也是本仓库
    `_env(key, default)` 一直以来的口径）。当天早些时候曾把默认翻成 `file`，
    动机是两次"配置文件写对了、进程用的是别的值"的事故（见模块顶部）——但那两次
    真正的病根是**冲突当时完全不可见**，而不是优先级方向本身：一旦
    `format_conflicts` 把每个被遮蔽的键连值一起 WARNING 出来（本模块现在无条件做），
    翻转优先级带来的收益就没了，剩下的只有"文件悄悄改写编排器注入的值"这个新风险。

    想让文件压过环境时用 `BLADEX_ENV_PRECEDENCE=file` 显式声明，
    或用 `BLADEX_ENV_KEEP=A,B` 保住个别键。
    """
    raw = (os.environ.get("BLADEX_ENV_PRECEDENCE")
           or file_vals.get("BLADEX_ENV_PRECEDENCE") or "env")
    mode = raw.strip().lower()
    return mode if mode in ("file", "env") else "env"


def _keep_keys(file_vals: dict[str, str]) -> frozenset[str]:
    raw = (os.environ.get("BLADEX_ENV_KEEP") or file_vals.get("BLADEX_ENV_KEEP") or "")
    return frozenset(k.strip() for k in raw.split(",") if k.strip())


def load_env_file(root: str | None = None) -> list[EnvConflict]:
    """把部署根的 `config/.env` 加载进 os.environ，返回冲突清单。

    - 环境里没有的键：直接设上（两种 precedence 下都一样）。
    - 值相同的键：不是冲突，不报。
    - 值不同的键：按 precedence 决定谁赢，**无论谁赢都进返回值**。

    返回空列表既可能是"没冲突"也可能是"没有 .env"——调用方不该用它判断
    配置是否存在，那是 `find_root()` 的活。
    """
    base = root if root is not None else deployment_root()
    file_vals = parse_env_file(os.path.join(base, CONFIG_RELPATH))
    if not file_vals:
        return []

    mode = _precedence(file_vals)
    keep = _keep_keys(file_vals)
    conflicts: list[EnvConflict] = []

    for key, file_value in file_vals.items():
        env_value = os.environ.get(key)
        if env_value is None:
            os.environ[key] = file_value
            continue
        if env_value == file_value:
            continue
        applied = mode == "file" and key not in keep and key not in _NEVER_OVERRIDE
        if applied:
            os.environ[key] = file_value
        conflicts.append(EnvConflict(key=key, env_value=env_value,
                                     file_value=file_value, applied=applied))
    return conflicts


def format_conflicts(conflicts: list[EnvConflict], env_path: str) -> list[str]:
    """冲突清单 → 给人看的行（英文，ADR-0027 §5.1；密钥已掩码）。

    分两段是因为两种冲突的处置完全不同：被文件覆盖的只是"提醒你环境里有残留"，
    而被环境遮蔽的是**配置文件没有生效**——后者正是两次事故的形状。
    """
    if not conflicts:
        return []
    lines: list[str] = []
    applied = [c for c in conflicts if c.applied]
    shadowed = [c for c in conflicts if not c.applied]

    if applied:
        lines.append(f"  note: {len(applied)} setting(s) from {env_path} overrode a stale "
                     f"value inherited from the environment:")
        for c in applied:
            lines.append(f"    {c.key}: using {mask(c.key, c.file_value)} "
                         f"(environment had {mask(c.key, c.env_value)})")
    if shadowed:
        lines.append(f"  WARNING: {len(shadowed)} setting(s) in {env_path} are NOT in effect -- "
                     f"the environment wins (BLADEX_ENV_PRECEDENCE=env):")
        for c in shadowed:
            lines.append(f"    {c.key}: running with {mask(c.key, c.env_value)}, "
                         f"but the file declares {mask(c.key, c.file_value)}")
        lines.append("    Unset those variables in your shell, or drop BLADEX_ENV_PRECEDENCE, "
                     "to let config/.env win.")
    return lines
