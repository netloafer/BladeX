"""Beta T14 Obsidian connector——把 MatterGraph 同步成 vault 里的双链笔记。

布局（options.vault_dir 之下，通常是 vault 内一个子目录）：

    Matters/<matter_id>.md   一个 Matter 一页：frontmatter（bladex id/状态/scope）
                             + 摘要 + "## Facts" 成员条目（每条带 [[bx-fact-<id>]]
                             wikilink 与 <!--bladex-fact:<id>--> 定位标记）
    Inbox.md                 未归属 fact 的收集页
    Hard Rules.md            硬规则单独页（整页重写）

语义：
  - 幂等重写：实体按 id 定位（文件名 = matter_id；fact 行 = HTML 注释标记），
    重复同步不产生重复条目；fact 换归属 = 从旧页摘行、写入新页。
  - 墓碑：fact → 删所在行；matter → 删整个文件。
  - [[wikilink]] 表达归属边：Matter 页 → bx-fact-<id> 节点，Obsidian 关系图
    可见 MatterGraph 结构（fact 节点为 unresolved link，无需为每条 fact 建页）。
  - 纯 stdlib，无第三方依赖；默认 exposure=private（config 层）。
"""

from __future__ import annotations

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

    def handle_tombstone(self, target_type: str, target_key: str) -> None:
        if target_type == "fact":
            removed = self._remove_fact_line(target_key)
            logger.debug("obsidian_tombstone_fact", fact_id=target_key,
                         removed=removed)
        elif target_type == "matter":
            path = self._matter_path(target_key)
            if path.is_file():
                path.unlink()
                logger.info("obsidian_tombstone_matter", matter_id=target_key)
        # turn/edge/distill 墓碑与 vault 无对应物：忽略

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

    def _append_fact_line(self, line: str, matter_id: str) -> None:
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

    def _upsert_fact(self, fact: Any, matter_id: str) -> None:
        # 幂等：先摘旧行（含改归属场景），再写新行
        self._remove_fact_line(fact.id)
        self._append_fact_line(self._fact_line(fact), matter_id)

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
        self._append_fact_line(found, matter_id)

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
        self._root.mkdir(parents=True, exist_ok=True)
        lines = ["---", "bladex_id: hard-rules", "---", "", "# Hard Rules", ""]
        lines += [f"- {r}" for r in rules]
        (self._root / "Hard Rules.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
