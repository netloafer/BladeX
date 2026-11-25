"""全量重建的暂存目录与原子换名（批 AA · MQ-I34）。

此前 `rebuild_from_hub(full=True)` 在**生产目录原地** `clear()` 再逐条写回：写入期
（T1 实测 ≈ 44 min）任何中止都留下半截 Index（实测：33 条的库在第 5 次写入时被
kill ⇒ 剩 4 条），只能靠备份恢复。修法是把 T2a / U1b 在操作层用过的形状搬进代码：

    生产目录 ──checkpoint/拷贝──▶ `<index>.staging` ──同一个重建引擎──▶ 自检 ──原子换名──▶ 生产
                                                                          旧生产 ──▶ `<index>.bak-<ts>`（不自动删）

本模块只管**目录**：建暂存、收拾上次中断留下的暂存、原子交换。重建引擎与自检在
`MemoryIndex` 里（它们要读库内容）。

🔴 三条纪律：
1. **换名必须原子**（`renamex_np(RENAME_SWAP)` / `renameat2(RENAME_EXCHANGE)`）。
   做不到 ⇒ `AtomicSwapError`，调用方不换名、生产保持原样。**绝不退化成"拷贝 + 删除"**
   ——那正是要消灭的半截窗口。
2. **暂存与生产同目录（同一文件系统）**；跨设备 ⇒ 同上，拒绝。
3. **只收割纯缓存**：上一次中断的暂存里只取 `distill/`、`adjcache/` 与 `embedcache/`（键 =
   输入哈希的模型产出缓存，命中与否不改变语义，只改变要不要再算一次）。其余状态（话语 memo、
   规则文件登记、判决台账…）一律不带——它们会改变重建的输入，续跑就不再等价于不中断那一跑。
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import shutil
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

STAGING_SUFFIX = ".staging"
PREV_INFIX = ".staging-prev-"
BAK_INFIX = ".bak-"
#: 暂存目录的身份标记。有它 = 我们建的暂存（可收割后删）；没有 = 换名中途崩溃留下的旧生产（只改名为 bak，不删）。
MARKER = ".bladex_staging"
#: 跨中断可收割的前缀：纯 LLM 产出缓存（见模块 docstring 纪律 3）。
HARVEST_PREFIXES: tuple[bytes, ...] = (b"distill/", b"adjcache/", b"embedcache/")
#: 不进暂存的条目：proxy 的 secondary 实例目录（属于读端进程）、Finder 垃圾、标记本身。
_SKIP = {"meta_rocksdb_secondary", ".DS_Store", MARKER}
_FTS_FILE = "fts.sqlite"


class AtomicSwapError(RuntimeError):
    """原子换名做不到（平台不支持 / 文件系统不支持 / 跨设备）。生产保持原样。"""


def staging_path(prod: Path) -> Path:
    return prod.with_name(prod.name + STAGING_SUFFIX)


def _ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")


def _write_marker(d: Path, state: str) -> None:
    (d / MARKER).write_text(json.dumps({"state": state, "ts": _ts(), "pid": os.getpid()}),
                            encoding="utf-8")


def read_marker(d: Path) -> dict[str, Any] | None:
    try:
        return json.loads((d / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ── 原子交换 ─────────────────────────────────────────────────────────────

_RENAME_SWAP = 0x2         # macOS <stdio.h>
_RENAME_EXCHANGE = 1 << 1  # Linux <linux/fs.h>
_AT_FDCWD = -100


def _libc() -> Any:
    return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)


def exchange(a: Path, b: Path) -> None:
    """原子交换两个目录项（两者都必须存在）。做不到 ⇒ `AtomicSwapError`，什么都没动。"""
    if os.stat(a).st_dev != os.stat(b).st_dev:
        raise AtomicSwapError(f"cross-device: {a} vs {b}")
    fa, fb = os.fsencode(str(a)), os.fsencode(str(b))
    if sys.platform == "darwin":
        fn = getattr(_libc(), "renamex_np", None)
        if fn is None:
            raise AtomicSwapError("renamex_np unavailable")
        rc = fn(fa, fb, ctypes.c_uint(_RENAME_SWAP))
    elif sys.platform.startswith("linux"):
        fn = getattr(_libc(), "renameat2", None)
        if fn is None:
            raise AtomicSwapError("renameat2 unavailable (glibc < 2.28)")
        rc = fn(_AT_FDCWD, fa, _AT_FDCWD, fb, ctypes.c_uint(_RENAME_EXCHANGE))
    else:
        raise AtomicSwapError(f"no atomic directory exchange on {sys.platform}")
    if rc != 0:
        err = ctypes.get_errno()
        raise AtomicSwapError(f"exchange failed: {os.strerror(err)} (errno {err})")


# ── 收拾上次中断留下的暂存 ───────────────────────────────────────────────

def retire_leftovers(prod: Path) -> list[Path]:
    """把上次中断留下的 `<index>.staging` 挪开，返回待收割的暂存目录列表。

    - 有 MARKER：我们建的暂存 ⇒ 改名 `.staging-prev-<ts>`，交收割（收割完删）。
    - 无 MARKER：换名后、改 bak 前崩溃留下的**旧生产** ⇒ 改名 `.bak-<ts>`，不删。
    """
    st = staging_path(prod)
    if st.exists():
        if read_marker(st) is not None:
            dst = prod.with_name(prod.name + PREV_INFIX + _ts())
            os.rename(st, dst)
            logger.warning("index_rebuild_staging_leftover", path=str(dst),
                           hint="previous full rebuild was interrupted; its LLM caches "
                                "will be harvested into the new staging, then it is removed")
        else:
            dst = prod.with_name(prod.name + BAK_INFIX + _ts())
            os.rename(st, dst)
            logger.warning("index_rebuild_swap_leftover_retired", path=str(dst),
                           hint="a previous swap was interrupted after the exchange; "
                                "the old production Index is kept as a backup")
    return sorted(p for p in prod.parent.glob(prod.name + PREV_INFIX + "*") if p.is_dir())


# ── 建暂存 ───────────────────────────────────────────────────────────────

def build_staging(prod: Path, meta_db: Any) -> Path:
    """生产 → `<index>.staging`。meta 走 RocksDB checkpoint（一致快照、同盘硬链接），
    FTS 走 SQLite backup API，其余原样复制。调用方须持有生产 meta 的写句柄（= 锁）。"""
    import rocksdict

    st = staging_path(prod)
    st.mkdir()
    _write_marker(st, "copying")
    rocksdict.Checkpoint(meta_db).create_checkpoint(str(st / "meta_rocksdb"))
    for entry in sorted(prod.iterdir()):
        name = entry.name
        if name in _SKIP or name == "meta_rocksdb" or name.startswith(_FTS_FILE + "-"):
            continue
        dst = st / name
        if name == _FTS_FILE:
            src_db = sqlite3.connect(str(entry))
            dst_db = sqlite3.connect(str(dst))
            try:
                src_db.backup(dst_db)
            finally:
                dst_db.close()
                src_db.close()
        elif entry.is_dir():
            shutil.copytree(entry, dst, symlinks=True)
        else:
            shutil.copy2(entry, dst)
    _write_marker(st, "built")
    logger.info("index_rebuild_staging_built", staging=str(st), prod=str(prod))
    return st


def harvest(prev_dirs: list[Path], meta_db: Any) -> int:
    """把中断暂存里的纯缓存（`HARVEST_PREFIXES`）写进新暂存（write-if-absent），再删掉它们。"""
    import rocksdict

    copied = 0
    for p in prev_dirs:
        try:
            src = rocksdict.Rdict(str(p / "meta_rocksdb"), rocksdict.Options(), None,
                                  rocksdict.AccessType.read_only())
            try:
                for prefix in HARVEST_PREFIXES:
                    for k, v in src.items(from_key=prefix):
                        kb = bytes(k) if not isinstance(k, bytes) else k
                        if not kb.startswith(prefix):
                            break
                        if meta_db.get(kb) is None:
                            meta_db[kb] = v
                            copied += 1
            finally:
                src.close()
        except Exception as e:  # noqa: BLE001 —— 半截拷贝的暂存打不开：没得收，删掉即可
            logger.warning("index_rebuild_harvest_failed", path=str(p), error=str(e))
        shutil.rmtree(p, ignore_errors=True)
    if prev_dirs:
        logger.info("index_rebuild_harvested", dirs=len(prev_dirs), keys=copied,
                    prefixes=[x.decode() for x in HARVEST_PREFIXES])
    return copied


# ── 换入 ─────────────────────────────────────────────────────────────────

def swap_in(staging: Path, prod: Path) -> Path:
    """暂存与生产原子交换，旧生产改名 `.bak-<ts>` 留着。返回 bak 路径。"""
    exchange(staging, prod)
    # 此刻 prod = 新库、staging 路径 = 旧生产。先摘新库的标记（它已是生产），
    # 再把旧生产改成 bak —— 两步之间崩溃由 `retire_leftovers` 的"无标记 ⇒ bak"兜住。
    try:
        (prod / MARKER).unlink()
    except OSError:
        pass
    bak = prod.with_name(prod.name + BAK_INFIX + _ts())
    os.rename(staging, bak)
    return bak
