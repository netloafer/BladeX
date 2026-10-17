"""Beta T14 Obsidian connector——把 MatterGraph 同步成 vault 里的双链笔记。

布局（options.vault_dir 之下，通常是 vault 内一个子目录）：

    Matters/<matter_id>.md   一个 Matter 一页：frontmatter（bladex id/状态/scope）
                             + 摘要 + "## Facts" 成员条目（每条带 [[bx-fact-<id>]]
                             wikilink 与 <!--bladex-fact:<id>--> 定位标记）
    Inbox.md                 未归属 fact 的收集页
    Hard Rules.md            硬规则单独页（整页重写）

回流（Z4，MQ-P33；编辑模型与落账见 `bladex_proxy.export_backflow`）：

    .bladex/sync_snapshot.json   本导出器**自己**上次写下的每条 fact 行 + 所在页、
                                 Hard Rules 列表。只有与快照对不上的才算用户改过
                                 （自写回声 0 事件）；用户改动不会被吸进快照。

    可回流：改 fact 行文字 ⇒ 更正；把行挪到另一页 ⇒ 换归属（挪进 Inbox = 摘除）；
    删 fact 行 ⇒ 墓碑；改 Hard Rules.md ⇒ 硬规则整表。
    不回流 + 告警：页 frontmatter 的 bladex_id 对不上文件名、行里 wikilink 还在但
    定位标记没了、同一标记出现两次（认不出是哪条，宁可不动）；整页消失 / 一次要删
    超过阈值 ⇒ 一条都不删。快照不存在（升级前的老 vault）⇒ 现状收作基线、零事件。

语义：
  - 幂等重写：实体按 id 定位（文件名 = matter_id；fact 行 = HTML 注释标记），
    重复同步不产生重复条目；fact 换归属 = 从旧页摘行、写入新页。
  - 墓碑：fact → 删所在行；matter → 删整个文件。
  - [[wikilink]] 表达归属边：Matter 页 → bx-fact-<id> 节点，Obsidian 关系图
    可见 MatterGraph 结构（fact 节点为 unresolved link，无需为每条 fact 建页）。
  - 纯 stdlib，无第三方依赖；默认 exposure=private（config 层）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import structlog
from bladex_core.exporters import ExportBatch

logger = structlog.get_logger()

_FACT_MARK = "<!--bladex-fact:{fid}-->"
_FACT_MARK_RE = re.compile(r"<!--bladex-fact:([^>]+?)-->")
_SESSION_MARK = "<!--bladex-session:{key}-->"
_WIKILINK_RE = re.compile(r"\[\[bx-fact-([^\]|]+?)\]\]")
_FRONTMATTER_ID_RE = re.compile(r"^bladex_id:\s*(.+?)\s*$", re.MULTILINE)
_SNAPSHOT_REL = ".bladex/sync_snapshot.json"
_HARD_RULES_ID = "hard-rules"
_INBOX_REL = "Inbox.md"


def _line_content(line: str) -> str:
    """fact 行 → 正文（去列表符、wikilink、定位标记）。导出与回流共用这一个解析点。"""
    text = _FACT_MARK_RE.sub("", line)
    text = _WIKILINK_RE.sub("", text).strip()
    for bullet in ("- ", "* ", "+ "):
        if text.startswith(bullet):
            text = text[len(bullet):]
            break
    return text.strip()


def _frontmatter_id(text: str) -> str:
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    if end < 0:
        return ""
    m = _FRONTMATTER_ID_RE.search(text[3:end])
    return m.group(1) if m else ""


def parse_hard_rules_page(text: str) -> list[str]:
    """`Hard Rules.md` → 规则列表（列表项；frontmatter / 标题 / 空行不算）。

    🔴 Z1.2（MQ-W18 硬规则热加载）合入后，**规则的归一化与校验**（去空、去重、坏输入
    保留旧规则）以它的单一实现点为准；这里只负责"Markdown 列表 → 字符串列表"这一步
    页面格式解析（`.env` 的 `||` 格式归 Z1）。结算稿 report-batch-Z4 §4 写明该依赖。
    """
    body = text
    if body.startswith("---"):
        end = body.find("\n---", 3)
        body = body[end + 4:] if end >= 0 else body
    rules = []
    for line in body.splitlines():
        s = line.strip()
        for bullet in ("- ", "* ", "+ "):
            if s.startswith(bullet):
                rule = s[len(bullet):].strip()
                if rule:
                    rules.append(rule)
                break
    return rules


def _sanitize(name: str) -> str:
    """文件名安全化（matter_id 通常已安全，防御特殊字符）。"""
    return re.sub(r"[^\w\-.]+", "_", name) or "_"


class ObsidianExporter:
    def __init__(self, name: str, exposure: str, options: dict[str, Any]) -> None:
        self.name = name
        self.exposure = exposure
        vault = options.get("vault_dir") or options.get("dir")
        if not vault:
            raise ValueError("obsidian exporter needs options.vault_dir")
        self._root = Path(os.path.expanduser(str(vault)))
        self._matters_dir = self._root / "Matters"
        # Z4 回流开关：默认开（README「改了就回流」）；`options.backflow = false` 关。
        self.backflow_enabled = bool(options.get("backflow", True))
        self._snap_path = self._root / _SNAPSHOT_REL
        self._snap: dict[str, Any] = self._load_snapshot()

    # ── 自写快照（Z4）────────────────────────────────────────────

    def _load_snapshot(self) -> dict[str, Any]:
        if self._snap_path.is_file():
            try:
                data = json.loads(self._snap_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("facts"), dict):
                    data.setdefault("hard_rules", None)
                    return data
            except ValueError as e:
                logger.warning("obsidian_snapshot_corrupt_rebaseline", error=str(e))
        return {"facts": {}, "hard_rules": None, "baseline": False}

    def _save_snapshot(self) -> None:
        self._snap_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._snap_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._snap, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self._snap_path)

    def _rel(self, page: Path) -> str:
        return page.relative_to(self._root).as_posix()

    # ── Exporter 协议 ─────────────────────────────────────────────

    def sync_incremental(self, batch: ExportBatch, since_cursor: str) -> None:
        self._matters_dir.mkdir(parents=True, exist_ok=True)
        for matter in batch.matters:
            self._upsert_matter(matter)
        # 边先于 fact 应用：edge 携带的归属让 fact 落对页
        assign: dict[str, str] = {}
        for edge in batch.edges:
            tt = str(getattr(edge.target_type, "value", edge.target_type))
            if tt == "fact":
                assign[edge.target_key] = edge.matter_id
            elif tt == "session":
                self._append_session(edge.matter_id, edge.target_key)
        for fact in batch.facts:
            matter_id = getattr(fact, "matter_id", "") or assign.get(fact.id, "")
            self._upsert_fact(fact, matter_id)
        # 已有 fact 的改归属（fact 本体不在本批但边是新的）
        for fid, mid in assign.items():
            if not any(f.id == fid for f in batch.facts):
                self._move_fact_line(fid, mid)
        if batch.hard_rules:
            self._write_hard_rules(batch.hard_rules)
        self._snap["baseline"] = True  # 导出器自己写过 ⇒ 快照即权威基线
        self._save_snapshot()

    def handle_tombstone(self, target_type: str, target_key: str) -> None:
        if target_type == "fact":
            removed = self._remove_fact_line(target_key)
            self._snap["facts"].pop(target_key, None)
            logger.debug("obsidian_tombstone_fact", fact_id=target_key,
                         removed=removed)
        elif target_type == "matter":
            path = self._matter_path(target_key)
            if path.is_file():
                path.unlink()
                rel = self._rel(path)
                for fid in [f for f, e in self._snap["facts"].items() if e.get("page") == rel]:
                    self._snap["facts"].pop(fid, None)
                logger.info("obsidian_tombstone_matter", matter_id=target_key)
        # turn/edge/distill 墓碑与 vault 无对应物：忽略
        self._save_snapshot()

    # ── Matter 页 ─────────────────────────────────────────────────

    def _matter_path(self, matter_id: str) -> Path:
        return self._matters_dir / f"{_sanitize(matter_id)}.md"

    def _upsert_matter(self, matter: Any) -> None:
        """重写头部（frontmatter + 摘要），保留已有 Facts/Sessions 条目行。"""
        path = self._matter_path(matter.matter_id)
        fact_lines, session_lines = [], []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                if _FACT_MARK_RE.search(line):
                    fact_lines.append(line)
                elif "<!--bladex-session:" in line:
                    session_lines.append(line)
        status = str(getattr(matter.status, "value", matter.status))
        origin = str(getattr(matter.origin, "value", matter.origin))
        head = [
            "---",
            f"bladex_id: {matter.matter_id}",
            f"status: {status}",
            f"origin: {origin}",
            f"scope: {getattr(matter, 'scope', '') or 'personal'}",
            "---",
            "",
            f"# {matter.title or matter.matter_id}",
            "",
        ]
        if getattr(matter, "summary", ""):
            head += [matter.summary, ""]
        body = head + ["## Facts", ""] + fact_lines + [""]
        if session_lines:
            body += ["## Sessions", ""] + session_lines + [""]
        path.write_text("\n".join(body), encoding="utf-8")

    def _ensure_matter_stub(self, matter_id: str) -> Path:
        path = self._matter_path(matter_id)
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"---\nbladex_id: {matter_id}\n---\n\n# {matter_id}\n\n## Facts\n\n",
                encoding="utf-8")
        return path

    # ── Fact 行 ───────────────────────────────────────────────────

    def _fact_line(self, fact: Any) -> str:
        content = str(fact.content).replace("\n", " ").strip()
        return (f"- {content} [[bx-fact-{fact.id}]] "
                + _FACT_MARK.format(fid=fact.id))

    def _iter_pages(self):
        if self._matters_dir.is_dir():
            yield from sorted(self._matters_dir.glob("*.md"))
        inbox = self._root / "Inbox.md"
        if inbox.is_file():
            yield inbox

    def _remove_fact_line(self, fact_id: str) -> bool:
        mark = _FACT_MARK.format(fid=fact_id)
        removed = False
        for page in self._iter_pages():
            lines = page.read_text(encoding="utf-8").splitlines()
            kept = [ln for ln in lines if mark not in ln]
            if len(kept) != len(lines):
                page.write_text("\n".join(kept), encoding="utf-8")
                removed = True
        return removed

    def _append_fact_line(self, line: str, matter_id: str) -> Path:
        if matter_id:
            page = self._ensure_matter_stub(matter_id)
        else:
            page = self._root / "Inbox.md"
            if not page.is_file():
                self._root.mkdir(parents=True, exist_ok=True)
                page.write_text("# BladeX Inbox\n\n## Facts\n\n", encoding="utf-8")
        text = page.read_text(encoding="utf-8")
        lines = text.splitlines()
        # 插到 "## Facts" 节末尾（找该节最后一条标记行；无则节头之后）
        try:
            idx = lines.index("## Facts")
        except ValueError:
            lines += ["", "## Facts", ""]
            idx = len(lines) - 1
        insert_at = idx + 1
        for i in range(idx + 1, len(lines)):
            if lines[i].startswith("## "):
                break
            if _FACT_MARK_RE.search(lines[i]) or lines[i].strip() == "":
                insert_at = i + 1
        lines.insert(insert_at, line)
        page.write_text("\n".join(lines), encoding="utf-8")
        return page

    def _upsert_fact(self, fact: Any, matter_id: str) -> None:
        # 幂等：先摘旧行（含改归属场景），再写新行
        self._remove_fact_line(fact.id)
        line = self._fact_line(fact)
        page = self._append_fact_line(line, matter_id)
        self._snap["facts"][fact.id] = {"line": line, "page": self._rel(page)}

    def _move_fact_line(self, fact_id: str, matter_id: str) -> None:
        """已存在的 fact 行改归属（只有边、无本体的批次）。"""
        mark = _FACT_MARK.format(fid=fact_id)
        found = None
        for page in self._iter_pages():
            for ln in page.read_text(encoding="utf-8").splitlines():
                if mark in ln:
                    found = ln
                    break
            if found:
                break
        if found is None:
            return
        self._remove_fact_line(fact_id)
        page = self._append_fact_line(found, matter_id)
        # 挪的是磁盘上那一行（可能被用户改过）：快照只改页，不改行——
        # 行文字的"上次自写版本"仍是导出器写的那份，用户改动照样认得出来。
        entry = self._snap["facts"].get(fact_id)
        if entry is not None:
            entry["page"] = self._rel(page)

    # ── Z4 回流：扫描 vault，与自写快照对账 ─────────────────────────

    def _scan_vault(self) -> tuple[dict[str, tuple[str, str]], set[str], set[str], dict[str, str]]:
        """→ (observed {fid: (page_rel, line)}, unreadable fids, pages present, page→matter_id)。

        认不出的（frontmatter id 被改 / 标记被删而 wikilink 还在 / 同标记出现两次）
        一律进 unreadable 并告警——不更正、不挪、不删。
        """
        observed: dict[str, tuple[str, str]] = {}
        unreadable: set[str] = set()
        pages: set[str] = set()
        page_matter: dict[str, str] = {}
        for page in self._iter_pages():
            rel = self._rel(page)
            pages.add(rel)
            text = page.read_text(encoding="utf-8")
            lines = text.splitlines()
            marked = [(ln, _FACT_MARK_RE.search(ln)) for ln in lines]
            if rel != _INBOX_REL:
                fm_id = _frontmatter_id(text)
                if not fm_id or _sanitize(fm_id) != page.stem:
                    # 标记与 wikilink 都算：只剩 wikilink 的行同样不许被读成"已删"
                    ids = {m.group(1) for _, m in marked if m}
                    ids |= set(_WIKILINK_RE.findall(text))
                    unreadable |= ids
                    logger.warning("obsidian_backflow_page_id_broken", page=rel,
                                   frontmatter_id=fm_id or None, facts=len(ids),
                                   hint="bladex_id in the frontmatter must match the file "
                                        "name; this page's edits are ignored until fixed")
                    continue
                page_matter[rel] = fm_id
            for ln, m in marked:
                if m is None:
                    for wl in _WIKILINK_RE.findall(ln):
                        unreadable.add(wl)
                        logger.warning("obsidian_backflow_marker_missing", page=rel,
                                       fact_id=wl,
                                       hint="the <!--bladex-fact:...--> marker was removed; "
                                            "this line is ignored (not updated, not deleted)")
                    continue
                fid = m.group(1)
                if fid in observed:
                    unreadable.add(fid)
                    logger.warning("obsidian_backflow_duplicate_marker", fact_id=fid,
                                   pages=sorted({observed[fid][0], rel}))
                    continue
                observed[fid] = (rel, ln)
        for fid in unreadable:
            observed.pop(fid, None)
        return observed, unreadable, pages, page_matter

    def collect_edits(self, max_deletes: int) -> list:
        """vault 与快照对账 ⇒ `BackflowEdit` 列表（不写任何东西；落账后调用 `ack`）。"""
        from bladex_proxy.export_backflow import BackflowEdit
        submitter = str(self._root)
        observed, unreadable, pages, page_matter = self._scan_vault()
        snap_facts: dict[str, dict] = self._snap["facts"]
        if not self._snap.get("baseline"):
            # 升级前的老 vault：认不出哪些是用户改的 ⇒ 现状收作基线，零事件
            for fid, (rel, ln) in observed.items():
                snap_facts[fid] = {"line": ln, "page": rel}
            hr = self._root / "Hard Rules.md"
            if hr.is_file() and self._snap.get("hard_rules") is None:
                self._snap["hard_rules"] = parse_hard_rules_page(hr.read_text(encoding="utf-8"))
            self._snap["baseline"] = True
            self._save_snapshot()
            logger.info("obsidian_backflow_baseline", facts=len(observed), vault=submitter)
            return []

        edits: list = []
        vanished: list[str] = []
        page_gone = 0
        for fid, entry in snap_facts.items():
            if fid in unreadable:
                continue
            if fid not in observed:
                if entry.get("page") not in pages:
                    page_gone += 1          # 整页没了：不删（挪目录 / 换 vault）
                else:
                    vanished.append(fid)
                continue
            rel, ln = observed[fid]
            if ln != entry.get("line"):
                new = _line_content(ln)
                if not new:
                    logger.warning("obsidian_backflow_empty_line_ignored", fact_id=fid,
                                   page=rel, hint="delete the whole line to forget a fact")
                elif new != _line_content(entry.get("line", "")):
                    edits.append(BackflowEdit("correct", "obsidian", submitter,
                                              fact_id=fid, content=new,
                                              ref=("line", fid, ln)))
            if rel != entry.get("page"):
                old_mid = page_matter.get(entry.get("page", ""), "")
                if not old_mid and entry.get("page", "").startswith("Matters/"):
                    old_mid = Path(entry["page"]).stem
                if rel == _INBOX_REL:
                    edits.append(BackflowEdit("detach", "obsidian", submitter, fact_id=fid,
                                              from_matter_id=old_mid, ref=("page", fid, rel)))
                else:
                    edits.append(BackflowEdit("assign", "obsidian", submitter, fact_id=fid,
                                              matter_id=page_matter[rel],
                                              from_matter_id=old_mid,
                                              ref=("page", fid, rel)))
        if page_gone:
            logger.warning("obsidian_backflow_pages_missing_no_delete", facts=page_gone,
                           vault=submitter,
                           hint="whole pages disappeared (moved folder / switched vault / "
                                "sync conflict?) -- nothing is forgotten for them")
        if len(vanished) > max_deletes:
            logger.warning("obsidian_backflow_mass_delete_refused", vanished=len(vanished),
                           threshold=max_deletes, vault=submitter,
                           hint="more fact lines vanished in one scan than "
                                "BLADEX_BACKFLOW_MAX_DELETES allows -- nothing is "
                                "forgotten; delete in smaller steps or raise the knob")
        else:
            edits += [BackflowEdit("delete", "obsidian", submitter, fact_id=fid,
                                   ref=("delete", fid, None)) for fid in vanished]

        hr = self._root / "Hard Rules.md"
        snap_rules = self._snap.get("hard_rules")
        if hr.is_file() and snap_rules is not None:
            text = hr.read_text(encoding="utf-8")
            if _frontmatter_id(text) not in ("", _HARD_RULES_ID):
                logger.warning("obsidian_backflow_page_id_broken", page="Hard Rules.md",
                               frontmatter_id=_frontmatter_id(text))
            else:
                rules = parse_hard_rules_page(text)
                if not rules:
                    logger.warning("obsidian_backflow_hard_rules_empty_ignored",
                                   hint="an empty Hard Rules page would drop every "
                                        "MUST/NEVER rule; ignored")
                elif rules != list(snap_rules):
                    edits.append(BackflowEdit("hard_rules", "obsidian", submitter,
                                              rules=rules, previous=list(snap_rules),
                                              ref=("rules", "", rules)))
        return edits

    def change_signature(self) -> tuple | None:
        """Z4b 唤醒探针：受管文件的 `(相对路径, mtime_ns, size)` 目录签名；回流关 ⇒ None。

        取舍（结算稿 §11）：用 stat 不用内容哈希——逐秒探，stat 只读 inode 不读正文；
        代价是"只 touch 不改内容"也会唤醒一次（那一轮 `collect_edits` 得 0 条编辑，白跑一次
        回流 → 导出，不写 Hub）。基线由调用方在**导出写完之后**取，导出器自写不算变化。
        """
        if not self.backflow_enabled:
            return None
        # os.scandir + DirEntry.stat：逐秒跑，Path.glob/relative_to 的开销在大 vault 上
        # 占大头（2000 页实测 39.7 ms/次 → 见结算稿 §11 的读数）。
        sig: list[tuple] = []
        for d, only in ((self._matters_dir, None), (self._root, ("Inbox.md", "Hard Rules.md"))):
            try:
                with os.scandir(d) as it:
                    for e in it:
                        if (e.name in only) if only else e.name.endswith(".md"):
                            try:
                                st = e.stat()
                            except OSError:
                                continue  # 列目录与 stat 之间被删：下一秒的签名里自然少了它
                            sig.append((d is self._root, e.name, st.st_mtime_ns, st.st_size))
            except OSError:
                sig.append((d is self._root, None, 0, 0))  # 目录不在（挪走 / 换 vault）也是一种签名
        sig.sort()
        return tuple(sig)

    def ack(self, edit: Any, result: Any = None) -> None:
        """落账（accepted）或永久拒绝（rejected）后推进快照——同一编辑不重复回流。"""
        kind, fid, value = edit.ref
        facts = self._snap["facts"]
        if kind == "line" and fid in facts:
            facts[fid]["line"] = value
        elif kind == "page" and fid in facts:
            facts[fid]["page"] = value
        elif kind == "delete":
            facts.pop(fid, None)
        elif kind == "rules":
            self._snap["hard_rules"] = list(value)
        self._save_snapshot()

    # ── Sessions / Hard rules ─────────────────────────────────────

    def _append_session(self, matter_id: str, session_key: str) -> None:
        page = self._ensure_matter_stub(matter_id)
        mark = _SESSION_MARK.format(key=session_key)
        text = page.read_text(encoding="utf-8")
        if mark in text:
            return
        line = f"- session `{session_key}` {mark}"
        if "## Sessions" not in text:
            text = text.rstrip("\n") + "\n\n## Sessions\n\n" + line + "\n"
        else:
            text = text.rstrip("\n") + "\n" + line + "\n"
        page.write_text(text, encoding="utf-8")

    def _write_hard_rules(self, rules: list[str]) -> None:
        # Z1.2 ⑦：生效规则一变 worker 就重导；页面内容已经是这套规则（用户自己在 Obsidian 改的、
        # 刚回流生效）⇒ 不重写，免得把用户的排版刷掉。快照照样对齐（自写快照 = 回声判据）。
        hr = self._root / "Hard Rules.md"
        if hr.is_file():
            try:
                if parse_hard_rules_page(hr.read_text(encoding="utf-8")) == list(rules):
                    self._snap["hard_rules"] = list(rules)
                    return
            except OSError:
                pass
        self._root.mkdir(parents=True, exist_ok=True)
        lines = ["---", "bladex_id: hard-rules", "---", "", "# Hard Rules", ""]
        lines += [f"- {r}" for r in rules]
        (self._root / "Hard Rules.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
        self._snap["hard_rules"] = list(rules)
