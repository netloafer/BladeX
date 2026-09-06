"""Embedding 模块的传输决断 —— **启动期一次，不每请求判**（V-A6）。

🔴 Jason 2026-08-25 拍板：「解析顺序不要每次请求决断，放在启动的时候决断，
后续 embedding 请求就按当前状态运行，需要改动就再重启一次」。

理由与仓库里另外两条同源（账本模板改启动预加载、`flags.py` 启动期解析）：
**每请求自适应 = 事后无法回答"这次走的哪条路"**。可诊断性比动态灵活值钱——
决断结果打进启动日志（`embed_transport_resolved`），要改就重启。

# 平台约束（为什么不能只有 uds）

| 传输 | 用于 | 平台 |
|---|---|---|
| `uds` | 单机同主机：最快、无端口、无鉴权面 | Linux/macOS |
| `tcp` | **Windows**（Python 在 Win 上无可用 `AF_UNIX`）、**Docker 跨容器**（现 compose 是 proxy/consolidator 两个容器，靠 network 通信） | 全平台 |
| `local` | 兜底：各进程自己加载模型（模块没起时不拒服务，同 ADR-0017 Redis 懒恢复原则） | 全平台 |

`auto` 判定序：显式配置 > 非 POSIX 或容器内 → `tcp` > `uds`。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

TRANSPORT_UDS = "uds"
TRANSPORT_TCP = "tcp"
TRANSPORT_LOCAL = "local"
TRANSPORT_AUTO = "auto"

#: 默认地址（uds 落部署根的 run/ 下——与数据同根，便于容器挂载与清理）。
DEFAULT_UDS_NAME = "embed.sock"
DEFAULT_TCP_ADDRESS = "127.0.0.1:38081"


@dataclass(frozen=True)
class TransportDecision:
    transport: str
    address: str
    reason: str            # 可 grep 的决断依据（打进启动日志）


def _in_container() -> bool:
    """容器检测：`/.dockerenv` 或 cgroup 里有 docker/kubepods。
    宁可误判为容器（→ tcp，全平台可用）也不误判为同机（→ uds，跨容器连不上）。"""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as f:
            blob = f.read()
        return "docker" in blob or "kubepods" in blob or "containerd" in blob
    except OSError:
        return False


def _uds_supported() -> bool:
    import socket
    return hasattr(socket, "AF_UNIX") and os.name == "posix"


#: 🔴 `sockaddr_un.sun_path` 的长度上限（**含结尾 NUL**）：macOS/BSD 104、Linux 108。
#: 这是内核 ABI 常量，不是可调参数——超了 `bind()` 直接抛
#: `OSError: AF_UNIX path too long`，**服务端进程当场死掉**，而 `bladex start`
#: 那侧只会打 "started"（live 病例 1/3 的形状）。
#: 2026-08-26 用户机 gate 撞到：pytest 的 tmp_path 就超了 104——沙盒里 `/tmp/...`
#: 短所以全绿，**测试宿主路径长度差异又是一类盲区**（与"测试宿主无事件循环"、
#: "测试规模 vs 生产规模"同族，这是第三次）。生产上部署根较深的用户会一样炸。
_SUN_PATH_MAX = 104 if sys.platform == "darwin" else 108


def uds_path_too_long(path: str) -> bool:
    """按字节算（路径含非 ASCII 时字符数会骗人）。留 1 字节给结尾 NUL。"""
    return len(os.fsencode(path)) >= _SUN_PATH_MAX


def default_uds_path() -> str:
    from bladex_proxy import deployment
    root = deployment.find_root() or os.getcwd()
    return os.path.join(root, "run", DEFAULT_UDS_NAME)


def resolve(*, explicit: str = "", explicit_address: str = "",
            env: dict | None = None) -> TransportDecision:
    """决断传输与地址。**只在进程启动时调一次**，结果存进程状态。"""
    e = os.environ if env is None else env
    want = (explicit or e.get("BLADEX_EMBED_TRANSPORT", "") or TRANSPORT_AUTO).strip().lower()
    addr = (explicit_address or e.get("BLADEX_EMBED_ADDRESS", "")).strip()

    if want == TRANSPORT_LOCAL:
        return TransportDecision(TRANSPORT_LOCAL, "", "explicit:local")
    if want == TRANSPORT_TCP:
        return TransportDecision(TRANSPORT_TCP, addr or DEFAULT_TCP_ADDRESS,
                                 "explicit:tcp")
    if want == TRANSPORT_UDS:
        if not _uds_supported():
            # 显式要 uds 但平台不支持：**报错式降级**（打 reason，调用方会记日志），
            # 不静默——用户明确指定过的东西被换掉必须看得见。
            return TransportDecision(TRANSPORT_TCP, addr or DEFAULT_TCP_ADDRESS,
                                     "explicit:uds_unsupported_on_platform")
        path = addr or default_uds_path()
        if uds_path_too_long(path):
            return TransportDecision(
                TRANSPORT_TCP, DEFAULT_TCP_ADDRESS,
                f"explicit:uds_path_too_long({len(os.fsencode(path))}"
                f">={_SUN_PATH_MAX})")
        return TransportDecision(TRANSPORT_UDS, path, "explicit:uds")

    # auto
    if not _uds_supported():
        return TransportDecision(TRANSPORT_TCP, addr or DEFAULT_TCP_ADDRESS,
                                 f"auto:no_af_unix(os={os.name},plat={sys.platform})")
    if _in_container():
        return TransportDecision(TRANSPORT_TCP, addr or DEFAULT_TCP_ADDRESS,
                                 "auto:container_detected")
    path = addr or default_uds_path()
    if uds_path_too_long(path):
        # 部署根太深 ⇒ socket 路径超 sun_path ⇒ bind 必抛、服务端秒死。
        # 降 tcp 而不是让它去死——理由与"平台不支持"那条完全一样，reason 写明长度。
        return TransportDecision(
            TRANSPORT_TCP, DEFAULT_TCP_ADDRESS,
            f"auto:uds_path_too_long({len(os.fsencode(path))}>={_SUN_PATH_MAX})")
    return TransportDecision(TRANSPORT_UDS, path, "auto:posix_host")
