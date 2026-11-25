"""交接包（handoff package）v1 —— 纯函数（批 CB，plan-0.4.0 §2.2 F1；拍板 dispatch-0.4.0 (41)）。

交接包是 Memory Hub 已有事实的**只读投影**，不新增真相。本模块只做四件事，全部无 IO、无墙钟：

1. `sanitize`：路径改写 + 凭证替换，**一处实现**（(41)-a-④⑤、(41)-b-2）；
2. `build_package`：把取好的数据组装成包，按 ③ → ② 的顺序裁到上限，① 永不裁；
3. `canonical_json` / `content_hash`：内容哈希；
4. `parse_package` / `ledger_from_package`：导入校验与落地形态。

取数归 `bladex_proxy.ledger_handoff`（core 不依赖 proxy）。

# 「逐字」的口径（(41)-a-④）

包内文本 == `sanitize(原文)`。没有路径和凭证的文本就是原文相等；有的，改写次数如实进
`rewrites`，只有次数，**没有原值**。

# 「其余绝对路径」的口径（(41)-a-⑤）

只认已知根（`_KNOWN_ROOTS`）与带常见文件扩展名的路径。字面的"一律替换"会毁掉真实账本里的
148 处正文（`/扣非…/经营现金流…` 这种中文斜杠枚举、`/admin/ledgers` 这种路由），而那些字符串
一处本地路径都没有。留存面写成白名单用例，见 `test_cb_handoff_core.py`。

# 边界字符为什么不用 `\\w`

`\\w` 匹配中文。第一版原型因此漏掉了 `存到/Users/x/…` 这种中文紧贴路径的写法，
是 74 本真样本抓到的（步骤 1 结算稿 §0-7）。下面的边界全部用显式 ASCII 字符类。

# 哈希防什么

防损坏和误改，**不防伪造**：谁都能重算。JSON 字符串之外的空白不算内容。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bladex_core.ledger import (
    ENTRY_SOURCE_IMPORT,
    ENTRY_SOURCES,
    SECTION_GOAL,
    STATUS_ACTIVE,
    Ledger,
    LedgerEntry,
)
from bladex_core.ledger_runtime import bind_matter, ledger_anchor_matter_id

FORMAT_VERSION = 1
#: 导入白名单。未知版本拒绝并点名——读不懂就不猜。
SUPPORTED_FORMAT_VERSIONS: tuple[int, ...] = (1,)

#: 显式截断标记（与 agency.handlers.TRUNCATED_MARK 同一记法；core 不 import proxy）。
TRUNCATED_MARK = "…<bladex:truncated>"

PLACEHOLDER_PROJECT = "<project>"
PLACEHOLDER_HOME = "~"
PLACEHOLDER_PATH = "<path>"
PLACEHOLDER_CREDENTIAL = "<redacted>"


class HandoffImportError(ValueError):
    """导入校验失败。`code` 给调用方分支，消息给人看（英文，会原样到终端）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# ── 包结构（字段表 = 步骤 1 结算稿 §2.1）──────────────────────────────────────


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HandoffEntry(_Strict):
    text: str
    source: str = "user"
    ref: str = ""
    writer: str = ""


class HandoffLedger(_Strict):
    """① 五段（逐字口径见模块 docstring）。`goal_verbatim` / `goal_origin` 刻意不在这里。"""

    title: str = ""
    status: str = STATUS_ACTIVE
    rev: int = 0
    goal: str = ""
    goal_source: str = ""
    section_order: list[str] = Field(default_factory=list)
    sections: dict[str, list[HandoffEntry]] = Field(default_factory=dict)


class HandoffMatter(_Strict):
    """② 只摘要，不带成员 fact。"""

    title: str = ""
    summary: str = ""


class HandoffRoundEntry(_Strict):
    section: str
    text: str


class HandoffRound(_Strict):
    """③ 一个用户轮写进账本的条目增删 + 用户话首段。没有 assistant 正文和工具结果。"""

    at: str = ""
    agent: str = ""
    added: list[HandoffRoundEntry] = Field(default_factory=list)
    removed: list[HandoffRoundEntry] = Field(default_factory=list)
    user_text_head: str = ""
    user_text_truncated: bool = False


class HandoffOmitted(_Strict):
    matters_exposure: int = 0
    rounds_exposure: int = 0
    matters_unavailable: int = 0
    rounds_without_user_text: int = 0


class HandoffTrimmed(_Strict):
    rounds: int = 0
    matters: int = 0
    round_entries: int = 0
    over_limit: bool = False


class HandoffRewrites(_Strict):
    """只有次数，没有原值（(41)-b-2）。"""

    project: int = 0
    home: int = 0
    path: int = 0
    credential: int = 0


class HandoffPackage(_Strict):
    format_version: int = FORMAT_VERSION
    bladex_version: str = ""
    exported_at: str = ""
    source_ledger_id: str = ""
    source_project: str = ""
    target_exposure: str = ""
    content_hash: str = ""
    ledger: HandoffLedger = Field(default_factory=HandoffLedger)
    matters: list[HandoffMatter] = Field(default_factory=list)
    recent_rounds: list[HandoffRound] = Field(default_factory=list)
    omitted: HandoffOmitted = Field(default_factory=HandoffOmitted)
    trimmed: HandoffTrimmed = Field(default_factory=HandoffTrimmed)
    rewrites: HandoffRewrites = Field(default_factory=HandoffRewrites)


@dataclass(frozen=True)
class HandoffLimits:
    max_chars: int = 20000
    max_rounds: int = 5
    user_text_chars: int = 500
    round_entries: int = 8


def limits_from_flags() -> HandoffLimits:
    from bladex_core.flags import flag_number
    return HandoffLimits(
        max_chars=int(flag_number("BLADEX_HANDOFF_MAX_CHARS")),
        max_rounds=int(flag_number("BLADEX_HANDOFF_MAX_ROUNDS")),
        user_text_chars=int(flag_number("BLADEX_HANDOFF_USER_TEXT_CHARS")),
        round_entries=int(flag_number("BLADEX_HANDOFF_ROUND_ENTRIES")),
    )


# ── sanitize：路径改写 + 凭证替换（一处实现）──────────────────────────────────

#: 路径字符的左边界：前面不能紧贴这些（否则是相对路径或 URL 的一段）。
_LB = r"(?<![A-Za-z0-9_.~:/\\-])"
#: 家目录左边界放行 `:` 与 `=`：`PATH=a:/Users/x/bin` 里的 `/Users/x` 也是本地路径。
_LB_HOME = r"(?<![A-Za-z0-9_.~/\\-])"
#: 一段路径的合法字符（中文标点与括号都是终止符）。
_SEG = r"[^\s`'\"<>()\[\]{},;:，。；：、）（】【]+"
#: 路径的右边界。
_RB_ALT = r"$|[\s`'\"<>()\[\]{},;:，。；：、）（】【]"
_RB = rf"(?={_RB_ALT})"

_KNOWN_ROOTS = ("tmp", "private", "var", "etc", "opt", "usr", "bin", "sbin", "dev", "Volumes",
                "mnt", "data", "srv", "Library", "Applications", "System", "nix", "proc", "sys",
                "workspace")
_FILE_EXTS = ("py", "sh", "md", "json", "jsonl", "toml", "yml", "yaml", "log", "txt", "ts", "js",
              "rs", "go", "c", "h", "cpp", "html", "csv", "db", "lock", "plist", "env")

_RE_FILE_URL = re.compile(r"file://(?=/)")
_RE_HOME = re.compile(_LB_HOME + r"(?:/Users/[^/\s`'\"<>)\]]+|/home/[^/\s`'\"<>)\]]+"
                      + rf"|/root(?=/|{_RB_ALT}))")
_RE_WIN_HOME = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\Users\\[^\\\s`'\"<>]+")
_RE_WIN_PATH = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\[^\s`'\"<>]*")
_RE_EXT_PATH = re.compile(_LB + rf"/{_SEG}/{_SEG}\.(?:{'|'.join(_FILE_EXTS)})" + _RB, re.IGNORECASE)
_RE_ROOT_PATH = re.compile(_LB + rf"/(?:{'|'.join(_KNOWN_ROOTS)})(?:/{_SEG})?" + _RB, re.IGNORECASE)

#: 高置信凭证形态（带边界：`task-codex-…` 这种前面紧贴字母的不算）。
_RE_CRED_FORM = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_\w{20,}"
    r"|xox[bpas]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|(?<=Bearer )[A-Za-z0-9._-]{16,})")
#: 键值形态：值只取 ASCII 连续串（中文正文不是值）。占位写法（`<…>`、`${…}`）不算。键不分大小写。
_RE_CRED_KV = re.compile(
    r"(?i)((?:api[_-]?key)[\"'`]?\s*[=:]\s*[\"'`]?)"
    r"(?![<$])([A-Za-z0-9_./+=:@-]{8,})")
#: 密码类（拍板 (43)-a-②）：密码常带 `!` `#` 这类符号、也可能很短，值放宽为
#: 「非空白、非引号的可打印 ASCII，长度 ≥ 6」，`<` `$` 开头的是占位写法。
_RE_CRED_PASSWORD = re.compile(
    r"(?i)((?:password|passwd|secret(?:[_-]?key)?)[\"'`]?\s*[=:]\s*[\"'`]?)"
    r"(?![<$])([!#-&(-_a-~]{6,})")
#: `token` 误报多（步骤 1 §1.2 里 3/4 是正文或计数），值要同时含字母和数字且 ≥16 字符。
_RE_CRED_TOKEN = re.compile(r"(?i)(token[\"'`]?\s*[=:]\s*[\"'`]?)([A-Za-z0-9_./+=-]{16,})")


def _redact_token(m: re.Match[str], counts: HandoffRewrites) -> str:
    value = m.group(2)
    if re.search(r"[A-Za-z]", value) and re.search(r"\d", value):
        counts.credential += 1
        return m.group(1) + PLACEHOLDER_CREDENTIAL
    return m.group(0)


def sanitize(text: str, project_roots: list[str] | tuple[str, ...] = (),
             counts: HandoffRewrites | None = None) -> str:
    """路径改写 + 凭证替换。**全仓库唯一实现**；`counts` 就地累加，只记次数。

    顺序有意义：项目根先于家目录（`/Users/x/dev/BladeX/a.py` → `<project>/a.py`，
    不是 `~/dev/BladeX/a.py`）；长根先于短根；凭证放最后。
    """
    if not text:
        return text
    c = counts if counts is not None else HandoffRewrites()
    out = _RE_FILE_URL.sub("", text)
    for root in sorted({r.rstrip("/") for r in project_roots if r and r.strip("/")},
                       key=len, reverse=True):
        out, n = re.subn(rf"(?<![A-Za-z0-9_.~/\\-]){re.escape(root)}(?![A-Za-z0-9_.-])",
                         PLACEHOLDER_PROJECT, out)
        c.project += n
    out, n = _RE_HOME.subn(PLACEHOLDER_HOME, out)
    c.home += n
    out, n = _RE_WIN_HOME.subn(PLACEHOLDER_HOME, out)
    c.home += n
    out, n = _RE_WIN_PATH.subn(PLACEHOLDER_PATH, out)
    c.path += n
    out, n = _RE_EXT_PATH.subn(PLACEHOLDER_PATH, out)
    c.path += n
    out, n = _RE_ROOT_PATH.subn(PLACEHOLDER_PATH, out)
    c.path += n
    out, n = _RE_CRED_FORM.subn(PLACEHOLDER_CREDENTIAL, out)
    c.credential += n
    out, n = _RE_CRED_KV.subn(lambda m: m.group(1) + PLACEHOLDER_CREDENTIAL, out)
    c.credential += n
    out, n = _RE_CRED_PASSWORD.subn(lambda m: m.group(1) + PLACEHOLDER_CREDENTIAL, out)
    c.credential += n
    out = _RE_CRED_TOKEN.sub(lambda m: _redact_token(m, c), out)
    return out


# ── 哈希 ──────────────────────────────────────────────────────────────────


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def content_hash(package: dict[str, Any]) -> str:
    """包内除 `content_hash` 之外**全部字段值**的 sha256。"""
    body = {k: v for k, v in package.items() if k != "content_hash"}
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


# ── ③ 最近轮次：从账本事件算条目增删（纯函数）──────────────────────────────────


@dataclass
class RawRound:
    """一个用户轮的条目增删（未 sanitize）。`session_id` / `turn_index` 空 = 历史事件没带写者。"""

    at: str = ""
    agent: str = ""
    session_id: str = ""
    turn_index: int | None = None
    added: list[tuple[str, str]] = field(default_factory=list)
    removed: list[tuple[str, str]] = field(default_factory=list)
    user_text: str = ""
    exposure: str = ""


def _entry_texts(dump: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for sec, items in (dump.get("sections") or {}).items():
        out[sec] = [str((e or {}).get("text", "")) for e in (items or [])]
    return out


def _multiset_diff(a: list[str], b: list[str]) -> list[str]:
    """b 里比 a 多出来的（按出现次数，保持 b 的顺序）。"""
    budget: dict[str, int] = {}
    for x in a:
        budget[x] = budget.get(x, 0) + 1
    out: list[str] = []
    for x in b:
        if budget.get(x, 0) > 0:
            budget[x] -= 1
        else:
            out.append(x)
    return out


def rounds_from_events(events: list[dict[str, Any]], ledger_id: str) -> list[RawRound]:
    """这本账本的事件流 → 按用户轮聚合的条目增删，**新到旧**。

    只有 `ledger_update` 产生轮次（F1 ③ 的定义）；create / user_edit 只推进比较基线。
    同一 `(session_id, turn_index)` 的多次 update 合成一轮；历史事件没带写者 ⇒ 各自一轮。
    没有条目变化的 update（如 bind-legacy-matters）不算一轮。
    """
    prev: dict[str, list[str]] = {}
    rounds: list[RawRound] = []
    index: dict[tuple[str, int], RawRound] = {}
    for ev in events:
        etype = ev.get("type", "")
        payload = ev.get("payload") or {}
        dump = payload.get("ledger") or {}
        if etype not in ("ledger_create", "ledger_update", "ledger_user_edit"):
            continue
        if dump.get("ledger_id") != ledger_id:
            continue
        cur = _entry_texts(dump)
        if etype == "ledger_update":
            added = [(s, t) for s in cur for t in _multiset_diff(prev.get(s, []), cur[s])]
            removed = [(s, t) for s in prev for t in _multiset_diff(cur.get(s, []), prev[s])]
            if added or removed:
                sid = str(payload.get("session_id", "") or "")
                ti = payload.get("turn_index")
                ti_ok = isinstance(ti, int) and not isinstance(ti, bool)
                key = (sid, ti) if sid and ti_ok else None
                at = _iso_of(ev.get("ts"))
                rnd = index.get(key) if key else None
                if rnd is None:
                    rnd = RawRound(at=at, agent=str(payload.get("agent_id", "") or ""),
                                   session_id=sid if key else "",
                                   turn_index=ti if key else None)
                    rounds.append(rnd)
                    if key:
                        index[key] = rnd
                rnd.at = at or rnd.at
                rnd.added.extend(added)
                rnd.removed.extend(removed)
        prev = cur
    rounds.sort(key=lambda r: r.at, reverse=True)
    return rounds


def _iso_of(ts: Any) -> str:
    if ts is None:
        return ""
    if hasattr(ts, "isoformat"):
        return ts.isoformat(timespec="seconds")
    return str(ts)


def first_paragraph(text: str) -> str:
    return (text or "").strip().split("\n\n", 1)[0].strip()


# ── 组装 + 裁剪 ─────────────────────────────────────────────────────────────


def build_package(
    ledger: Ledger, *,
    exported_at: str,
    bladex_version: str = "",
    source_project: str = "",
    target_exposure: str = "public",
    project_roots: list[str] | tuple[str, ...] = (),
    matters: list[tuple[str, str]] | None = None,
    rounds: list[RawRound] | None = None,
    omitted: HandoffOmitted | None = None,
    limits: HandoffLimits | None = None,
) -> HandoffPackage:
    """组装交接包。同输入逐字节相同（`exported_at` 由调用方传，函数内没有墙钟）。

    超 `max_chars` 时：从最旧的 ③ 一轮轮去掉 → 再去 ② → ① 永不裁；① 本身超限照常出包，
    `trimmed.over_limit=True`。
    """
    lim = limits or HandoffLimits()
    c = HandoffRewrites()

    def s(t: str) -> str:
        return sanitize(t, project_roots, c)

    sections: dict[str, list[HandoffEntry]] = {}
    for key in ledger.section_order:
        if key == SECTION_GOAL:
            continue
        sections[key] = [HandoffEntry(text=s(e.text), source=e.source, ref=s(e.ref),
                                      writer=e.writer)
                         for e in ledger.entries(key)]
    led = HandoffLedger(title=s(ledger.title), status=ledger.status, rev=ledger.rev,
                        goal=s(ledger.goal), goal_source=ledger.goal_source,
                        section_order=list(ledger.section_order), sections=sections)
    pkg_matters = [HandoffMatter(title=s(t), summary=s(sm)) for t, sm in (matters or [])]

    trimmed = HandoffTrimmed()
    om = (omitted or HandoffOmitted()).model_copy()
    pkg_rounds: list[HandoffRound] = []
    all_rounds = list(rounds or [])
    trimmed.rounds += max(0, len(all_rounds) - lim.max_rounds)
    for r in all_rounds[:lim.max_rounds]:
        head = first_paragraph(r.user_text)
        truncated = len(head) > lim.user_text_chars
        if truncated:
            head = head[:lim.user_text_chars] + TRUNCATED_MARK
        if not r.user_text:
            om.rounds_without_user_text += 1
        trimmed.round_entries += (max(0, len(r.added) - lim.round_entries)
                                  + max(0, len(r.removed) - lim.round_entries))
        pkg_rounds.append(HandoffRound(
            at=r.at, agent=r.agent,
            added=[HandoffRoundEntry(section=a, text=s(b)) for a, b in r.added[:lim.round_entries]],
            removed=[HandoffRoundEntry(section=a, text=s(b)) for a, b in r.removed[:lim.round_entries]],
            user_text_head=s(head), user_text_truncated=truncated))

    pkg = HandoffPackage(
        bladex_version=bladex_version, exported_at=exported_at,
        source_ledger_id=ledger.ledger_id, source_project=source_project,
        target_exposure=target_exposure, ledger=led, matters=pkg_matters,
        recent_rounds=pkg_rounds, omitted=om, trimmed=trimmed, rewrites=c)

    def size() -> int:
        return len(canonical_json(pkg.model_dump(mode="json")))

    while size() > lim.max_chars and pkg.recent_rounds:
        pkg.recent_rounds.pop()          # 新到旧排列，末尾最旧
        pkg.trimmed.rounds += 1
    while size() > lim.max_chars and pkg.matters:
        pkg.matters.pop()
        pkg.trimmed.matters += 1
    pkg.trimmed.over_limit = size() > lim.max_chars
    pkg.content_hash = content_hash(pkg.model_dump(mode="json"))
    return pkg


# ── Markdown（只读渲染，不可导入）─────────────────────────────────────────────


def _indent_continuation(text: str, prefix: str) -> str:
    lines = (text or "").split("\n")
    return "\n".join([lines[0], *[prefix + ln for ln in lines[1:]]])


def render_markdown(pkg: HandoffPackage) -> str:
    """一页纸。Goal 用引用块、条目续行缩进 ⇒ 正文不会以 `/` 开头。"""
    led = pkg.ledger
    out = [f"# Handoff: {led.title or pkg.source_ledger_id}", "",
           f"- ledger: `{pkg.source_ledger_id}` (rev {led.rev}, {led.status})",
           f"- exported: {pkg.exported_at} by BladeX {pkg.bladex_version or '?'}"
           f" (format {pkg.format_version}, exposure {pkg.target_exposure})",
           f"- content hash: `{pkg.content_hash}`",
           "- This page is for reading or pasting. To import on another machine, "
           "export with `--format json`.", ""]
    out += ["## Goal", ""]
    out += [f"> {ln}" for ln in (led.goal or "(empty)").split("\n")]
    for key in led.section_order:
        if key == SECTION_GOAL:
            continue
        out += ["", f"## {key.capitalize()}", ""]
        items = led.sections.get(key) or []
        if not items:
            out.append("_(empty)_")
        for e in items:
            tail = " · ".join(x for x in (e.source, e.ref, f"@{e.writer}" if e.writer else "") if x)
            out.append(f"- {_indent_continuation(e.text, '  ')}  [{tail}]")
    if pkg.matters or pkg.omitted.matters_exposure or pkg.omitted.matters_unavailable:
        out += ["", "## Matter", ""]
        for m in pkg.matters:
            out.append(f"- **{m.title}**: {_indent_continuation(m.summary, '  ')}")
        if pkg.omitted.matters_exposure:
            out.append(f"- ({pkg.omitted.matters_exposure} omitted: above the target exposure)")
        if pkg.omitted.matters_unavailable:
            out.append(f"- ({pkg.omitted.matters_unavailable} unavailable: exposure could not be read)")
    if pkg.recent_rounds:
        out += ["", "## Recent rounds", ""]
        for r in pkg.recent_rounds:
            out.append(f"- {r.at}" + (f" · {r.agent}" if r.agent else ""))
            if r.user_text_head:
                out.append(f"  - user: {_indent_continuation(r.user_text_head, '    ')}")
            for a in r.added:
                out.append(f"  - + [{a.section}] {_indent_continuation(a.text, '    ')}")
            for a in r.removed:
                out.append(f"  - - [{a.section}] {_indent_continuation(a.text, '    ')}")
    notes = []
    if pkg.omitted.rounds_exposure:
        notes.append(f"{pkg.omitted.rounds_exposure} rounds omitted (exposure)")
    if pkg.trimmed.rounds or pkg.trimmed.matters:
        notes.append(f"trimmed for size: {pkg.trimmed.rounds} rounds, {pkg.trimmed.matters} matters")
    if pkg.trimmed.over_limit:
        notes.append("the five sections alone exceed the size limit (they are never trimmed)")
    rw = pkg.rewrites
    if rw.project or rw.home or rw.path or rw.credential:
        notes.append(f"rewritten: {rw.project} project paths, {rw.home} home paths, "
                     f"{rw.path} other paths, {rw.credential} credentials")
    if notes:
        out += ["", "---", "", *[f"- {n}" for n in notes]]
    return "\n".join(out) + "\n"


# ── 导入 ──────────────────────────────────────────────────────────────────


def parse_package(raw: Any) -> HandoffPackage:
    """校验顺序：版本 → 形状 → 哈希。版本放最前：版本不认识，后面的形状校验没有意义。"""
    if not isinstance(raw, dict):
        raise HandoffImportError("shape", "a handoff package must be a JSON object")
    version = raw.get("format_version")
    if version not in SUPPORTED_FORMAT_VERSIONS or isinstance(version, bool):
        known = ", ".join(str(v) for v in SUPPORTED_FORMAT_VERSIONS)
        raise HandoffImportError(
            "version", f"unsupported format_version {version!r} (this BladeX reads: {known})")
    try:
        pkg = HandoffPackage.model_validate(raw)
    except ValidationError as e:
        first = e.errors()[0]
        where = ".".join(str(x) for x in first.get("loc", ()))
        raise HandoffImportError("shape", f"invalid handoff package at {where}: "
                                          f"{first.get('msg', 'invalid')}") from e
    if content_hash(raw) != raw.get("content_hash"):
        raise HandoffImportError(
            "hash", "content hash mismatch: the package was changed or damaged after export; "
                    "nothing was imported")
    for key in pkg.ledger.sections:
        if key == SECTION_GOAL or not re.fullmatch(r"[a-z][a-z0-9_-]*", key):
            raise HandoffImportError("shape", f"invalid section name {key!r}")
    for key, items in pkg.ledger.sections.items():
        for e in items:
            if e.source not in ENTRY_SOURCES:
                raise HandoffImportError("shape", f"invalid entry source {e.source!r} in {key!r}")
    if not pkg.ledger.goal.strip():
        raise HandoffImportError("shape", "the package has no Goal")
    return pkg


def ledger_from_package(pkg: HandoffPackage, *, ledger_id: str, now_iso: str) -> Ledger:
    """导入落地形态。②③ **不进五段**（(41)-a-⑨），Goal 身份 = 用户（红线：Goal 仅用户可改）。"""
    led = pkg.ledger
    order = list(led.section_order) or [SECTION_GOAL, *led.sections]
    if SECTION_GOAL not in order:
        order.insert(0, SECTION_GOAL)
    for key in led.sections:
        if key not in order:
            order.append(key)
    sections = {key: [LedgerEntry(text=e.text, source=ENTRY_SOURCE_IMPORT, ref=e.ref, writer=e.writer)
                      for e in led.sections.get(key, [])]
                for key in order if key != SECTION_GOAL}
    out = Ledger(
        ledger_id=ledger_id, title=led.title, status=STATUS_ACTIVE,
        created_at=now_iso, updated_at=now_iso, rev=1, last_writer="user",
        goal=led.goal, goal_source="user",
        goal_origin=f"handoff:{pkg.source_ledger_id}@{pkg.content_hash[:12]}",
        goal_verbatim="", goal_revisions=0,
        sections=sections, section_order=order,
    )
    return bind_matter(out, ledger_anchor_matter_id(ledger_id), updated_at=now_iso)
