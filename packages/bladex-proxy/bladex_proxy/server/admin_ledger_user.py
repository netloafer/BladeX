"""账本的用户写入面与交接包端点（批 CB，README ②；拍板 dispatch-0.4.0 (41)）。

`GET /admin/ledgers/{id}/handoff` · `POST /admin/ledgers/create` · `POST /admin/ledgers/{id}/edit` ·
`POST /admin/ledgers/import`。

🔴 全部只落已有事件类型 `ledger_user_edit`，新信息放 payload（`origin`、`handoff`）：0.3.0 读到未知事件
类型会抛错，回滚即坏（dispatch §3.0 第 13 条）。模型的写路径仍只有工具面，这里的 `origin` 不收 `model`。
"""

from __future__ import annotations

from typing import Any

import structlog
from bladex_core.sensitivity import EXPOSURE_LOCAL, EXPOSURE_PRIVATE, EXPOSURE_PUBLIC
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import (
    LEDGER_USER_WRITE_ORIGINS,
    AdminEventType,
    LedgerCreateBody,
    LedgerEditBody,
    LedgerImportBody,
)
from bladex_proxy.server.admin_api import (
    _admin_key_dep,
    _admin_response,
    _strict_body,
    _strict_query,
)
from bladex_proxy.server.blocking import run_blocking

logger = structlog.get_logger(__name__)


def register_ledger_user_routes(app: FastAPI) -> None:
    @app.get("/admin/ledgers/{ledger_id}/handoff")
    async def get_ledger_handoff(
        ledger_id: str,
        request: Request,
        exposure: str = EXPOSURE_PUBLIC,
        cfg: ProxyConfig = _admin_key_dep,
        _query: None = Depends(_strict_query("exposure")),
    ) -> JSONResponse:
        """交接包：预览与导出是同一个端点（批 CB）。只读。

        `exposure` 默认 public：包要离开本机，目的地未知按最坏算。凭证与路径的原值
        不进响应也不进日志，`ledger_handoff_exported` 只记次数。
        """
        if exposure not in (EXPOSURE_LOCAL, EXPOSURE_PRIVATE, EXPOSURE_PUBLIC):
            return JSONResponse(status_code=400, content={"error": {
                "message": f"unknown exposure {exposure!r} (use local, private or public)",
                "type": "validation_error"}})
        agency = getattr(request.app.state, "agency", None)
        led = None if agency is None else agency.pool.get(ledger_id)
        if led is None:
            return JSONResponse(status_code=404, content={
                "status": "not_found", "detail": f"unknown ledger {ledger_id}"})
        hub = getattr(request.app.state, "hub", None)
        if hub is None:
            return JSONResponse(status_code=503, content={"error": {
                "message": "Memory Hub unavailable", "type": "service_unavailable"}})
        from datetime import UTC, datetime

        from bladex_core.handoff import render_markdown

        import bladex_proxy
        from bladex_proxy.ledger_handoff import collect_package
        index = getattr(request.app.state, "index", None)
        pkg = await run_blocking(
            collect_package, hub, index, led,
            exported_at=datetime.now(UTC).isoformat(timespec="seconds"),
            target_exposure=exposure,
            sensitivity=cfg.routing_config.sensitivity_config(),
            bladex_version=bladex_proxy.__version__)
        logger.info("ledger_handoff_exported", ledger=ledger_id, exposure=exposure,
                    rounds=len(pkg.recent_rounds), matters=len(pkg.matters),
                    omitted=pkg.omitted.model_dump(), trimmed=pkg.trimmed.model_dump(),
                    rewrites=pkg.rewrites.model_dump())
        return JSONResponse(content={"package": pkg.model_dump(mode="json"),
                                     "markdown": render_markdown(pkg)})

    @app.post("/admin/ledgers/create")
    async def create_ledger_endpoint(
        body: LedgerCreateBody,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
        _body_keys: None = Depends(_strict_body(LedgerCreateBody)),
    ) -> JSONResponse:
        """用户建本（批 CB，F6）。**不激活**：激活键是 `(agent, project)`，这里没有 agent 身份，
        而激活只有 switch 一条路（拍板 (41)-a-③）；新本在候选段里，模型自己切。

        事件用 `ledger_user_edit` 而不是 `ledger_create`：现有探针把后者当「模型建本」计数，
        载荷也不带 agent，进不了时间轴。
        """
        from datetime import UTC, datetime

        from bladex_core.ledger import (
            ACTOR_DASHBOARD,
            ACTOR_USER,
            ENTRY_SOURCE_USER,
            LedgerEntry,
            LedgerError,
            add_entry,
            new_ledger,
            template_section_order,
        )
        from bladex_core.ledger_runtime import bind_matter, ledger_anchor_matter_id
        if body.origin not in LEDGER_USER_WRITE_ORIGINS:
            return JSONResponse(status_code=400, content={"error": {
                "message": f"origin must be one of {sorted(LEDGER_USER_WRITE_ORIGINS)}",
                "type": "validation_error"}})
        if not body.goal.strip():
            return JSONResponse(status_code=400, content={"error": {
                "message": "goal: must not be empty", "type": "validation_error"}})
        agency = getattr(request.app.state, "agency", None)
        if agency is None:
            return JSONResponse(status_code=503, content={"error": {
                "message": "agency runtime unavailable; cannot create",
                "type": "service_unavailable"}})
        now_iso = datetime.now(UTC).isoformat(timespec="seconds")
        title = body.title.strip() or body.goal.strip().split("\n", 1)[0][:60]
        led = new_ledger(title=title, goal=body.goal, goal_source=ACTOR_USER,
                         goal_verbatim=body.goal, created_at=now_iso,
                         section_order=template_section_order(agency.template))
        actor = ACTOR_DASHBOARD if body.origin == "dashboard" else ACTOR_USER
        try:
            for section, texts in (("core", body.core), ("verified", body.verified),
                                   ("open", body.open), ("next", body.next)):
                for text in texts:
                    if text.strip():
                        led = add_entry(led, section,
                                        LedgerEntry(text=text, source=ENTRY_SOURCE_USER),
                                        actor=actor, updated_at=now_iso)
        except LedgerError as e:
            return JSONResponse(status_code=400, content={"error": {
                "message": str(e), "type": "validation_error"}})
        led = bind_matter(led, ledger_anchor_matter_id(led.ledger_id), updated_at=now_iso)
        led = led.model_copy(update={"rev": 1, "last_writer": "user"})
        agency.pool[led.ledger_id] = led
        agency._emit(AdminEventType.LEDGER_USER_EDIT,
                     {"ledger": led.model_dump(), "origin": f"{body.origin}_create"})
        logger.info("ledger_user_created", ledger=led.ledger_id, origin=body.origin,
                    entries=sum(len(v) for v in led.sections.values()))
        return _admin_response(200, "ok", ledger_id=led.ledger_id, rev=led.rev,
                               ledger=led.model_dump(mode="json"))

    @app.post("/admin/ledgers/{ledger_id}/edit")
    async def edit_ledger_endpoint(
        ledger_id: str,
        body: LedgerEditBody,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
        _body_keys: None = Depends(_strict_body(LedgerEditBody)),
    ) -> JSONResponse:
        """用户改本（批 CB，F6）。`expected_rev` 必填，不符 ⇒ 409，池与 Hub 一字不动。

        Goal 有变化 ⇒ 只经 `update_goal`（`goal_source=user`、`goal_revisions+1`），这里不另赋值。
        已关的本可改，不重开；墓碑本不在池里 ⇒ 404。什么都没变 ⇒ `noop`，不落事件。
        """
        from datetime import UTC, datetime

        from bladex_core.ledger import (
            ACTOR_DASHBOARD,
            ACTOR_USER,
            ENTRY_SOURCES,
            SECTION_GOAL,
            LedgerError,
            add_section,
            update_goal,
        )

        def _bad(message: str) -> JSONResponse:
            return JSONResponse(status_code=400, content={"error": {
                "message": message, "type": "validation_error"}})

        if body.origin not in LEDGER_USER_WRITE_ORIGINS:
            return _bad(f"origin must be one of {sorted(LEDGER_USER_WRITE_ORIGINS)}")
        if body.expected_rev is None:
            return _bad("expected_rev: required (the rev you read this ledger at)")
        if body.goal is not None and not body.goal.strip():
            return _bad("goal: must not be empty")
        agency = getattr(request.app.state, "agency", None)
        if agency is None:
            return JSONResponse(status_code=503, content={"error": {
                "message": "agency runtime unavailable; cannot edit",
                "type": "service_unavailable"}})
        prev = agency.pool.get(ledger_id)
        if prev is None:
            return JSONResponse(status_code=404, content={
                "status": "not_found", "detail": f"unknown ledger {ledger_id}"})
        if prev.rev != body.expected_rev:
            logger.warning("admin_ledger_edit_rev_conflict", ledger=ledger_id,
                           expected_rev=body.expected_rev, current_rev=prev.rev,
                           last_writer=prev.last_writer)
            return JSONResponse(status_code=409, content={
                "status": "conflict", "ledger_id": ledger_id,
                "expected_rev": body.expected_rev, "current_rev": prev.rev,
                "ledger": prev.model_dump(mode="json"),
                "detail": ("ledger changed since you read it; nothing was written. "
                           "Read it again and redo the edit.")})
        now_iso = datetime.now(UTC).isoformat(timespec="seconds")
        led = prev
        try:
            for key in body.new_sections:
                led = add_section(led, key)
        except LedgerError as e:
            return _bad(str(e))
        goal_changed = body.goal is not None and body.goal != prev.goal
        if goal_changed:
            actor = ACTOR_DASHBOARD if body.origin == "dashboard" else ACTOR_USER
            led = update_goal(led, body.goal, actor=actor, updated_at=now_iso)
        updates: dict[str, Any] = {}
        if body.title is not None and body.title != prev.title:
            updates["title"] = body.title
        if body.sections:
            sections = {k: list(v) for k, v in led.sections.items()}
            for key, items in body.sections.items():
                if key == SECTION_GOAL:
                    return _bad("goal is a text block, not an entry section; use the goal field")
                if key not in led.section_order:
                    return _bad(f"unknown section {key!r} (this ledger has: "
                                f"{', '.join(s for s in led.section_order if s != SECTION_GOAL)})")
                for e in items:
                    if e.source not in ENTRY_SOURCES:
                        return _bad(f"invalid entry source {e.source!r} in {key!r}")
                if items != led.entries(key):
                    sections[key] = list(items)
            if sections != led.sections:
                updates["sections"] = sections
        if not goal_changed and not updates and led.section_order == prev.section_order:
            return _admin_response(200, "noop", ledger_id=ledger_id, rev=prev.rev,
                                   ledger=prev.model_dump(mode="json"))
        led = led.model_copy(update={**updates, "rev": prev.rev + 1, "last_writer": "user",
                                     "updated_at": now_iso})
        agency.pool[ledger_id] = led
        agency._emit(AdminEventType.LEDGER_USER_EDIT,
                     {"ledger": led.model_dump(), "origin": f"{body.origin}_edit"})
        logger.info("ledger_user_edited", ledger=ledger_id, origin=body.origin, rev=led.rev,
                    goal_changed=goal_changed, sections=sorted(body.sections or {}))
        return _admin_response(200, "ok", ledger_id=ledger_id, rev=led.rev,
                               ledger=led.model_dump(mode="json"))

    @app.post("/admin/ledgers/import")
    async def import_ledger_handoff(
        body: LedgerImportBody,
        request: Request,
        cfg: ProxyConfig = _admin_key_dep,
        _body_keys: None = Depends(_strict_body(LedgerImportBody)),
    ) -> JSONResponse:
        """导入交接包（批 CB）。校验顺序：大小 → 版本 → 形状 → 哈希；任一不过 ⇒ 400，什么都不写。

        同 id 不覆盖：**已知集合 = 活池 ∪ 墓碑账本**——沿用墓碑 id 的话事件会被
        `apply_ledger_tombstones` 丢弃，导入后立刻消失。Goal 身份 = 用户（Goal 仅用户可改）。
        不绑定、不激活、不落项目 scope：账本池没有项目维度（拍板 (41)-a-②③）。
        全包存进事件（原则 13 无损），②③ **只存不注入**（(41)-a-⑨）。
        🔴 只用已有事件类型 `ledger_user_edit`：0.3.0 读到未知类型会抛错（§3.0 第 13 条）。
        """
        from datetime import UTC, datetime

        from bladex_core.flags import flag_number
        from bladex_core.handoff import HandoffImportError, ledger_from_package, parse_package
        from bladex_core.ledger import new_ledger_id

        from bladex_proxy.ledger_events import tombstoned_ledger_ids
        agency = getattr(request.app.state, "agency", None)
        if agency is None:
            return JSONResponse(status_code=503, content={"error": {
                "message": "agency runtime unavailable; cannot import",
                "type": "service_unavailable"}})
        max_bytes = int(flag_number("BLADEX_HANDOFF_IMPORT_MAX_BYTES"))
        if len(await request.body()) > max_bytes:
            return JSONResponse(status_code=400, content={"error": {
                "message": f"package larger than {max_bytes} bytes; nothing was imported",
                "type": "validation_error", "code": "size"}})
        try:
            pkg = parse_package(body.package)
        except HandoffImportError as e:
            return JSONResponse(status_code=400, content={"error": {
                "message": str(e), "type": "validation_error", "code": e.code}})
        hub = getattr(request.app.state, "hub", None)
        tombstoned = (await run_blocking(tombstoned_ledger_ids, hub)) if hub is not None else set()
        # 上面的 await 之后到写池之间不再让出：判 id 与占 id 必须是一步。
        wanted = pkg.source_ledger_id
        conflict = (not wanted) or wanted in agency.pool or wanted in tombstoned
        ledger_id = new_ledger_id() if conflict else wanted
        now_iso = datetime.now(UTC).isoformat(timespec="seconds")
        led = ledger_from_package(pkg, ledger_id=ledger_id, now_iso=now_iso)
        counts = {k: len(v) for k, v in led.sections.items()}
        if body.dry_run:
            return _admin_response(200, "dry_run", dry_run=True, ledger_id=ledger_id,
                                   id_conflict=conflict, title=led.title, goal=led.goal,
                                   entry_counts=counts)
        agency.pool[ledger_id] = led
        agency._emit(AdminEventType.LEDGER_USER_EDIT, {
            "ledger": led.model_dump(), "origin": "handoff_import",
            "handoff": {"package": body.package}})
        logger.info("ledger_handoff_imported", source_ledger=pkg.source_ledger_id,
                    ledger=ledger_id, id_conflict=conflict,
                    format_version=pkg.format_version, entries=sum(counts.values()))
        return _admin_response(200, "ok", ledger_id=ledger_id, id_conflict=conflict,
                               ledger=led.model_dump(mode="json"))
