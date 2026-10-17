"""Z4（MQ-P33）：外部工具回流——Obsidian vault 直编 / Postgres 收件表 ⇒ Hub 管理事件。

README 公开承诺「Bidirectional sync with external tools」的兑现件。本模块只做两件事：

1. **编辑模型**（`BackflowEdit`）：两条通道（`exporters/obsidian.py` 的
   `collect_edits`、`exporters/postgres.py` 的 `collect_edits`）各自把"用户在外面
   改了什么"翻译成同一种编辑；
2. **落账客户端**（`AdminBackflowClient`）：把编辑经 proxy 的 `/admin/*` **正门**
   写进 Memory Hub。

# 🔴 单写者（ADR-0009；ADR-0032 §2.3 写权限矩阵）

Memory Hub 的唯一写者是 proxy。回流跑在 consolidator 进程（`export_sync` 的宿主），
而 consolidator 对 Hub 是**只读 secondary**——所以本模块**不开任何 Hub / Index 写句柄**，
一律 HTTP POST 到 proxy，与 `flash_daemon.admin_emit`（MQ-L51 直编采纳）同一条路。
Index 的生效 = proxy 即时应用（拿得到写锁时）或 consolidator 下一 pass 增量重放
管理事件（`MemoryIndex._replay_new_admin_events`）。守卫：`test_export_backflow.py`
的静态扫描（`exporters/`、本模块、`export_sync.py` 里零 Index 写调用、零 Hub 句柄）。

# 删除语义（两条通道一致，结算稿 report-batch-Z4 §0）

- 删**单条** fact ⇒ 墓碑（`DELETE /admin/facts/{id}?origin=…`，墓碑带 origin）；
- 删**整页 / 整文件**，或一次扫描里要删的 fact **超过** `BLADEX_BACKFLOW_MAX_DELETES`
  ⇒ **一条都不删，只告警**（挪文件夹、换 vault、同步盘冲突都长这样）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()

#: 编辑种类。`reject` = 通道侧已判不合法（如 Postgres 收件里的硬规则改动），不发 HTTP。
EDIT_KINDS = frozenset({"correct", "assign", "detach", "delete", "hard_rules", "reject"})


@dataclass
class BackflowEdit:
    """一条外部编辑（通道无关）。`ref` 是通道私有的回执数据（Obsidian 快照 / PG 行 id）。"""

    kind: str
    origin: str                      # "obsidian" | "postgres"
    submitter: str = ""
    fact_id: str = ""
    content: str = ""                # correct
    matter_id: str = ""              # assign：目标 Matter
    from_matter_id: str = ""         # assign / detach：原 Matter（可空 = 原先未归属）
    rules: list[str] = field(default_factory=list)      # hard_rules
    previous: list[str] = field(default_factory=list)   # hard_rules
    reason: str = ""                 # reject
    ref: Any = None

    def __post_init__(self) -> None:
        if self.kind not in EDIT_KINDS:
            raise ValueError(f"unknown backflow edit kind {self.kind!r}")


@dataclass
class BackflowResult:
    """`accepted` = Hub 已落事件（200 或 202 deferred 都算）；`rejected` = 永久拒绝
    （不重试，回执给用户看原因）；`retry` = 暂时失败（proxy 不在 / 5xx），下轮重来。"""

    status: str
    reason: str = ""
    event_ref: str = ""


#: (method, path, json_body | None) -> (http_status, json_response)。网络失败以抛出表达。
Requester = Callable[[str, str, dict | None], tuple[int, dict]]


def http_requester(base_url: str, admin_key: str = "", timeout: float = 10.0) -> Requester:
    """本机 proxy 的 admin 调用（与 `flash_daemon.admin_emit` 同一扇门、同一套鉴权回落）。"""
    base = base_url.rstrip("/")

    def _req(method: str, path: str, body: dict | None) -> tuple[int, dict]:
        headers = {"Content-Type": "application/json"}
        if admin_key:
            headers["Authorization"] = f"Bearer {admin_key}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = urllib.request.Request(base + path, data=data, headers=headers,  # noqa: S310
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 —— 本机 proxy
                raw = resp.read()
                status = int(resp.status)
        except urllib.error.HTTPError as e:  # 4xx/5xx 也要读出 body（拒绝原因在里面）
            raw = e.read() or b""
            status = int(e.code)
        try:
            parsed = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            parsed = {}
        return status, parsed if isinstance(parsed, dict) else {}

    return _req


def _q(s: str) -> str:
    return urllib.parse.quote(s, safe="")


def _reason(status: int, body: dict) -> str:
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict) and err.get("message"):
        return f"HTTP {status}: {err['message']}"
    if isinstance(body, dict) and body.get("detail"):
        return f"HTTP {status}: {body['detail']}"
    return f"HTTP {status}"


class AdminBackflowClient:
    """把 `BackflowEdit` 落成 Hub 管理事件（经 proxy 正门）。永不抛出：失败折成 `retry`。"""

    def __init__(self, requester: Requester) -> None:
        self._req = requester

    def _call(self, method: str, path: str, body: dict | None) -> BackflowResult:
        try:
            status, resp = self._req(method, path, body)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            return BackflowResult("retry", reason=f"proxy unreachable: {e}")
        if status in (200, 202):
            return BackflowResult("accepted",
                                  event_ref=str(resp.get("event_key")
                                                or resp.get("tombstone_key") or ""),
                                  reason=str(resp.get("status", "")))
        if 400 <= status < 500 and status not in (401, 408, 429):
            return BackflowResult("rejected", reason=_reason(status, resp))
        # 401 = admin key 配错（修配置后应能重试成功，不是编辑本身的错）；5xx / 限流同理
        return BackflowResult("retry", reason=_reason(status, resp))

    def apply(self, edit: BackflowEdit) -> BackflowResult:
        who = {"origin": edit.origin, "submitter": edit.submitter}
        if edit.kind == "reject":
            return BackflowResult("rejected", reason=edit.reason)
        if edit.kind == "correct":
            return self._call("POST", f"/admin/facts/{_q(edit.fact_id)}/correct",
                              {"content": edit.content, **who})
        if edit.kind == "delete":
            return self._call("DELETE", f"/admin/facts/{_q(edit.fact_id)}"
                                        f"?origin={_q(edit.origin)}", None)
        if edit.kind in ("assign", "detach"):
            # 换归属 = 先摘该 fact 的全部归属边、再挂新卡。两条事件都只点名这一条 fact
            # （判据⑦：不触发任何自动合并）；摘边幂等，挂卡失败重试时重摘无害。
            detach = self._call(
                "POST", f"/admin/matters/{_q(edit.from_matter_id or '_unassigned')}/detach",
                {"target_key": edit.fact_id, "target_type": "fact", **who})
            if edit.kind == "detach" or detach.status != "accepted":
                return detach
            return self._call("POST", f"/admin/matters/{_q(edit.matter_id)}/assign",
                              {"target_type": "fact", "target_key": edit.fact_id, **who})
        if edit.kind == "hard_rules":
            return self._call("POST", "/admin/hard-rules/set",
                              {"rules": edit.rules, "previous": edit.previous, **who})
        return BackflowResult("rejected", reason=f"unsupported edit kind {edit.kind!r}")
