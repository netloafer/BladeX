"""BladeX MCP server（Beta T16/T17 只读工具 + remember 写工具）。

stdio MCP server。数据面**只走 proxy HTTP admin API**（同 client key 鉴权），
不直接打开 Memory Index/Memory Hub——存储写者纪律（consolidator 独占写）与可见性/敏感度过滤
全部复用 proxy 侧的实现。

连接配置（env）：
  BLADEX_PROXY_URL   proxy 地址（默认 http://127.0.0.1:38080）
  BLADEX_MCP_KEY     client key（缺省取 BLADEX_CLIENT_KEYS 第一把；auth 关则不需要）

Claude Code 配置示例（.mcp.json / `claude mcp add`）：
  { "mcpServers": { "bladex": { "command": "bladex-mcp" } } }

proxy 不可达时工具返回明确错误文本（不挂起：所有请求带超时）。
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import quote, urlencode

# mcp 1.x（FastMCP）与 2.x（MCPServer）双兼容——两者同有 .tool() 装饰器与 run("stdio")
try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as _ServerCls
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _ServerCls  # type: ignore[no-redef]

_TIMEOUT_S = 15.0


def _base_url() -> str:
    return os.environ.get("BLADEX_PROXY_URL", "http://127.0.0.1:38080").rstrip("/")


def _first_key_of(raw: str) -> str:
    """取 `key||label||key2||label2` 里的第一把 key。

    ADR-0027 §5.2：格式权威是 `bladex_proxy.auth.KeyStore.parse`——**纯 `||` 分隔**。
    此前先按 `,` split，label 含逗号即截断出错误的 key。
    """
    parts = [p.strip() for p in raw.split("||") if p.strip()]
    return parts[0] if parts else ""


def _client_key() -> str:
    """MCP 打的是 admin 只读面，故优先 admin key（ADR-0027 §2.2）。

    优先级：BLADEX_MCP_KEY（显式）> BLADEX_ADMIN_KEYS > BLADEX_CLIENT_KEYS（回落）。
    """
    key = os.environ.get("BLADEX_MCP_KEY", "")
    if key:
        return key
    return (_first_key_of(os.environ.get("BLADEX_ADMIN_KEYS", ""))
            or _first_key_of(os.environ.get("BLADEX_CLIENT_KEYS", "")))


def _http_json(method: str, path: str, params: dict | None = None,
               body: dict | None = None) -> dict:
    """admin API 调用。失败抛 RuntimeError（工具层转为明确错误文本）。"""
    url = _base_url() + path
    if params:
        url += "?" + urlencode({k: v for k, v in params.items()
                                if v not in ("", None)})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)  # noqa: S310
    req.add_header("Content-Type", "application/json")
    key = _client_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode("utf-8"))
        except ValueError:
            detail = {}
        msg = (detail.get("error", {}).get("message")
               or detail.get("detail")
               or str(e))
        raise RuntimeError(f"BladeX proxy returned HTTP {e.code}: {msg}") from e
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(
            f"BladeX proxy unreachable at {_base_url()} ({e}). "
            "Start it with `bladex start` or set BLADEX_PROXY_URL.") from e


def _fmt_fact(f: dict) -> str:
    parts = [f"[{f.get('kind', '?')}] {f.get('content', '')}"]
    meta = []
    if f.get("id"):
        meta.append(f"id={f['id']}")
    if f.get("source_session"):
        meta.append(f"session={f['source_session']}")
    if f.get("matter_id"):
        meta.append(f"matter={f['matter_id']}")
    if meta:
        parts.append("  (" + ", ".join(meta) + ")")
    return "".join(parts)


server = _ServerCls(
    "bladex",
    instructions=(
        "BladeX personal memory layer. Use memory_search to recall facts the "
        "user has accumulated across agents and sessions; matter_list/matter_get "
        "for ongoing matters (tasks/projects); ledger_list/ledger_read for "
        "task ledgers; hard_rules for the user's MUST/NEVER rules; "
        "session_recall to find past sessions about a topic; remember to save "
        "an explicit new fact."),
)


@server.tool()
def memory_search(query: str, k: int = 10) -> str:
    """Search the user's BladeX memory (semantic + keyword) for relevant facts.

    Args:
        query: what to look for (natural language, any language)
        k: max results (default 10)
    """
    try:
        data = _http_json("GET", "/admin/facts", params={"q": query, "limit": k})
    except RuntimeError as e:
        return f"ERROR: {e}"
    facts = data.get("facts", [])
    if not facts:
        return "No matching facts in BladeX memory."
    return "\n".join(_fmt_fact(f) for f in facts)


@server.tool()
def matter_list(status: str = "") -> str:
    """List the user's Matters (ongoing tasks/projects tracked by BladeX).

    Args:
        status: optional filter (active / closed / provisional)
    """
    try:
        data = _http_json("GET", "/admin/matters",
                          params={"status": status, "limit": 100})
    except RuntimeError as e:
        return f"ERROR: {e}"
    matters = data.get("matters", [])
    if not matters:
        return "No matters."
    lines = []
    for m in matters:
        lines.append(f"{m.get('matter_id')} [{m.get('status')}] {m.get('title')}"
                     + (f" — {m.get('summary')}" if m.get("summary") else ""))
    return "\n".join(lines)


@server.tool()
def matter_get(matter_id: str) -> str:
    """Get one Matter in full: summary, member facts, sessions, participants.

    Args:
        matter_id: the matter id (from matter_list)
    """
    try:
        data = _http_json("GET", f"/admin/matters/{quote(matter_id, safe='')}")
    except RuntimeError as e:
        return f"ERROR: {e}"
    m = data.get("matter", {})
    lines = [f"{m.get('matter_id')} [{m.get('status')}] {m.get('title')}"]
    if m.get("summary"):
        lines.append(f"Summary: {m['summary']}")
    if m.get("participants"):
        ags = ", ".join(p.get("agent_id", "?") for p in m["participants"])
        lines.append(f"Participants: {ags}")
    facts = data.get("facts", [])
    if facts:
        lines.append("Facts:")
        lines += ["  - " + _fmt_fact(f) for f in facts]
    if data.get("session_keys"):
        lines.append("Sessions: " + ", ".join(data["session_keys"]))
    return "\n".join(lines)


@server.tool()
def hard_rules() -> str:
    """Get the user's MUST/NEVER hard rules (always-on constraints)."""
    try:
        data = _http_json("GET", "/admin/hard_rules")
    except RuntimeError as e:
        return f"ERROR: {e}"
    rules = data.get("hard_rules", [])
    if not rules:
        return "No hard rules configured."
    return "\n".join(f"- {r}" for r in rules)


@server.tool()
def session_recall(query: str, k: int = 20) -> str:
    """Find past sessions related to a topic (which sessions discussed X).

    Args:
        query: the topic to look for
        k: max facts to inspect (default 20)
    """
    try:
        data = _http_json("GET", "/admin/facts", params={"q": query, "limit": k})
    except RuntimeError as e:
        return f"ERROR: {e}"
    sessions: dict[str, list[str]] = {}
    for f in data.get("facts", []):
        sess = f.get("source_session", "") or "(unknown session)"
        sessions.setdefault(sess, []).append(f.get("content", ""))
    if not sessions:
        return "No past sessions found for this topic."
    lines = []
    for sess, contents in sessions.items():
        lines.append(f"session {sess}:")
        lines += [f"  - {c}" for c in contents[:5]]
    return "\n".join(lines)


@server.tool()
def ledger_list(active_only: bool = False) -> str:
    """List task ledgers (active first, most recently updated first).

    Args:
        active_only: when true, list only ledgers currently activated by an agent
    """
    try:
        data = _http_json("GET", "/admin/ledgers")
    except RuntimeError as e:
        return f"ERROR: {e}"
    if not data.get("enabled", False):
        return "Ledger agency is not enabled."
    ledgers = data.get("ledgers", [])
    if active_only:
        ledgers = [led for led in ledgers if led.get("active_in")]
    if not ledgers:
        return "No task ledgers."
    lines = []
    for led in ledgers:
        lines.append(
            f"{led.get('ledger_id')} [{led.get('status', 'active')}] "
            f"{led.get('title') or '(untitled)'}")
        if led.get("goal"):
            lines.append(f"  Goal: {led['goal']}")
        if led.get("matter_id"):
            lines.append(f"  Matter: {led['matter_id']}")
        entry_counts = led.get("entry_counts") or {}
        if entry_counts:
            counts = ", ".join(
                f"{name}: {count}" for name, count in entry_counts.items())
            lines.append(f"  Entries: {counts}")
        if led.get("updated_at"):
            lines.append(f"  Updated: {led['updated_at']}")
        active_in = led.get("active_in") or []
        if active_in:
            scopes = ", ".join(
                f"{scope.get('agent', '?')}@{scope.get('project', '?')}"
                for scope in active_in)
            lines.append(f"  Active in: {scopes}")
    return "\n".join(lines)


@server.tool()
def ledger_read(ledger_id: str) -> str:
    """Read one task ledger, including its goal and all saved sections.

    Args:
        ledger_id: the ledger id (from ledger_list)
    """
    try:
        data = _http_json(
            "GET", f"/admin/ledgers/{quote(ledger_id, safe='')}")
    except RuntimeError as e:
        return f"ERROR: {e}"
    markdown = data.get("markdown", "")
    if markdown:
        return markdown
    ledger = data.get("ledger")
    if ledger is None:
        return "No task ledger content available."
    return str(ledger)


@server.tool()
def remember(text: str, scope: str = "") -> str:
    """Save an explicit fact into the user's BladeX memory (manual, high trust).

    Args:
        text: the fact to remember (one atomic statement)
        scope: optional visibility scope (personal / team:<id> / org:<id>).
               Leave empty on a personal deployment -- team:/org: scopes require
               an identity.toml and are rejected with 400 otherwise.
    """
    try:
        data = _http_json("POST", "/admin/facts",
                          body={"content": text, "scope": scope})
    except RuntimeError as e:
        return f"ERROR: {e}"
    return (f"Remembered (fact_id={data.get('fact_id', '?')}, "
            f"status={data.get('status', '?')}).")


def main() -> None:
    """stdio 入口（[project.scripts] bladex-mcp）。"""
    server.run("stdio")


if __name__ == "__main__":
    main()
