"""Beta T13（B3.2）：export sync worker——增量同步 Memory Index 记忆到外部目标。

跑在 consolidator 进程（独占写者旁的只读消费），主管线之后：
  - 水位游标持久化（JSON 状态文件，按 exporter 隔离）；
  - 失败指数退避（目标不可达只降级不阻塞 rebuild 主管线）；
  - scope / exposure 过滤（ADR-0021：越级 Fact 不出境）；
  - 墓碑同步（done 集合去重，目标侧删除）；
  - Z4 回流（MQ-P33）：**先回流、后导出**——外部编辑经 proxy admin 正门落 Hub 管理事件
    （本进程对 Hub 只读，不开写句柄，见 `export_backflow`）。有暂时失败（proxy 不在）
    ⇒ 本轮**不导出**该目标并退避：导出会重写 fact 行，先导出就可能把还没落账的用户
    改动盖掉。

配置 `config/export.toml`（凭证只写 env 变量名，见 export.toml.example）。
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from bladex_core.exporters import ExportBatch, Exporter
from bladex_core.sensitivity import exposure_allows

from bladex_proxy.export_backflow import AdminBackflowClient

logger = structlog.get_logger()

_BACKOFF_BASE_S = 30.0
_BACKOFF_MAX_S = 3600.0


# ── 配置 ──────────────────────────────────────────────────────────


@dataclass
class ExporterConfig:
    type: str
    name: str
    enabled: bool = True
    exposure: str = "private"
    interval_s: float = 300.0
    scopes: list[str] = field(default_factory=list)  # 空 = 不过滤
    options: dict[str, Any] = field(default_factory=dict)


def load_export_config(path: str | Path) -> list[ExporterConfig]:
    """读 config/export.toml。文件不存在 → 空列表（功能关闭，零回归）。"""
    import tomllib
    p = Path(path)
    if not p.is_file():
        return []
    with p.open("rb") as f:
        data = tomllib.load(f)
    out: list[ExporterConfig] = []
    for raw in data.get("exporters", []):
        if not raw.get("type") or not raw.get("name"):
            raise ValueError("export.toml: each [[exporters]] needs `type` and `name`")
        out.append(ExporterConfig(
            type=str(raw["type"]), name=str(raw["name"]),
            enabled=bool(raw.get("enabled", True)),
            exposure=str(raw.get("exposure", "private")),
            interval_s=float(raw.get("interval_s", 300)),
            scopes=[str(s) for s in raw.get("scopes", [])],
            options=dict(raw.get("options", {})),
        ))
    names = [c.name for c in out]
    if len(names) != len(set(names)):
        raise ValueError("export.toml: exporter names must be unique (cursor isolation)")
    return out


# ── 状态（游标 + 退避 + 墓碑 done 集合）───────────────────────────


class ExportStateStore:
    """JSON 文件状态：{name: {cursor, tombstones_done, failures, next_attempt, last_ok}}。"""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._state: dict[str, dict[str, Any]] = {}
        if self._path.is_file():
            try:
                self._state = json.loads(self._path.read_text(encoding="utf-8"))
            except ValueError as e:
                logger.warning("export_state_corrupt_reset", error=str(e))
                self._state = {}

    def get(self, name: str) -> dict[str, Any]:
        return self._state.setdefault(name, {
            "cursor": "", "tombstones_done": [], "failures": 0,
            "next_attempt": 0.0, "last_ok": "", "last_synced": 0,
        })

    def reset(self, name: str | None = None) -> None:
        if name is None:
            self._state = {}
        else:
            self._state.pop(name, None)
        self.save()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(self._path)

    @property
    def all(self) -> dict[str, dict[str, Any]]:
        return self._state


# ── worker ────────────────────────────────────────────────────────


def _ts_of(obj: Any, attr: str) -> str:
    v = getattr(obj, attr, None)
    if isinstance(v, datetime):
        return v.isoformat()
    return str(v or "")


def _fact_change_ts(fact: Any) -> str:
    """Fact 的"最后变更时间" = max(created_at, updated_at)（ADR-0027 §5.3）。

    `updated_at` 是加法式新增字段，历史 Fact 为 None → 退化成 created_at（零迁移）。
    """
    created = _ts_of(fact, "created_at")
    updated = _ts_of(fact, "updated_at")
    return max(created, updated) if updated else created


def _rules_fingerprint(rules: list[str]) -> str:
    """硬规则集合的指纹（顺序敏感：页面按序渲染，换序也算变化）。"""
    import hashlib
    return hashlib.sha256("\n".join(rules).encode("utf-8")).hexdigest()[:16]


class ExportSyncWorker:
    """增量同步执行器。consolidator 循环里每轮 `run_once()`；异常永不外抛。"""

    def __init__(
        self,
        index: Any,
        ledger: Any,
        bindings: list[tuple[ExporterConfig, Exporter]],
        state_path: str | Path,
        hard_rules: list[str] | Callable[[], list[str]] | None = None,
        now: Any = time.time,
        backflow: AdminBackflowClient | None = None,
        max_deletes: int | None = None,
    ) -> None:
        self._index = index
        self._hub = ledger
        self._bindings = bindings
        self._state = ExportStateStore(state_path)
        # Z1.2（MQ-W18）：可传可调用对象——consolidator 传 `lambda: cfg.effective_hard_rules`，
        # 每轮现取热加载后的值（传列表 = 旧形态，测试 / 一次性脚本）。
        self._hard_rules = hard_rules if hard_rules is not None else []
        self._now = now
        # Z4：None = 回流整体不接线（无 proxy 地址 / 测试）；每目标另有自己的开关
        self._backflow = backflow
        if max_deletes is None:
            from bladex_core.flags import flag_number
            max_deletes = int(flag_number("BLADEX_BACKFLOW_MAX_DELETES"))
        self._max_deletes = max_deletes

    # -- Z4 回流 --

    def backflow_status(self) -> dict[str, str]:
        """{exporter 名: "on" | "off"}——启动行与 `run_once` 结果共用的口径。"""
        out: dict[str, str] = {}
        for cfg, exp in self._bindings:
            flag = getattr(exp, "backflow_enabled", None)
            if flag is None:
                flag = getattr(exp, "inbox_enabled", None)
            if flag is None:
                continue  # 该类型没有回流通道（jsonl）
            out[cfg.name] = "on" if (flag and self._backflow is not None) else "off"
        return out

    def _run_backflow(self, cfg: ExporterConfig, exporter: Any) -> dict[str, int]:
        """一个目标的回流：收编辑 ⇒ 落账 ⇒ 回执。有 retry ⇒ 抛出（调用方退避、不导出）。"""
        counts = {"accepted": 0, "rejected": 0, "retry": 0}
        collect = getattr(exporter, "collect_edits", None)
        if self._backflow is None or collect is None:
            return counts
        if not (getattr(exporter, "backflow_enabled", False)
                or getattr(exporter, "inbox_enabled", False)):
            return counts
        from bladex_proxy.export_backflow import BackflowEdit
        for edit in collect(self._max_deletes):
            if edit.kind == "assign" and not self._matter_exists(edit.matter_id):
                edit = BackflowEdit("reject", edit.origin, edit.submitter, ref=edit.ref,
                                    reason=f"matter {edit.matter_id!r} does not exist")
            result = self._backflow.apply(edit)
            counts[result.status] = counts.get(result.status, 0) + 1
            log = logger.info if result.status == "accepted" else logger.warning
            log("export_backflow_edit", exporter=cfg.name, origin=edit.origin,
                kind=edit.kind, fact_id=edit.fact_id or None, status=result.status,
                reason=result.reason or None, event_ref=result.event_ref or None)
            if result.status != "retry":
                exporter.ack(edit, result)
        if counts["retry"]:
            raise RuntimeError(f"backflow: {counts['retry']} edit(s) not recorded "
                               "(proxy unreachable?); export skipped this round so the "
                               "user's edits are not overwritten")
        return counts

    def _matter_exists(self, matter_id: str) -> bool:
        try:
            return bool(matter_id) and self._index.get_matter(matter_id) is not None
        except Exception as e:  # noqa: BLE001 —— 读不到就别挂（宁可拒绝）
            logger.warning("export_backflow_matter_lookup_failed", error=str(e))
            return False

    # -- 数据收集（游标 = 实体时间戳 ISO 水位）--

    def current_hard_rules(self) -> list[str]:
        src = self._hard_rules
        return list(src() if callable(src) else src)

    def _collect_since(self, cursor: str, cfg: ExporterConfig) -> tuple[ExportBatch, str]:
        batch = ExportBatch()
        max_ts = cursor
        for fact in self._index.all_facts():
            # ADR-0027 §5.3：取 max(created_at, updated_at)——Fact 会被非破坏取代、
            # scope 提升、生命周期更新，只看 created_at 会让这些变更永不重导，
            # 目标端把被取代的事实当现行显示。
            ts = _fact_change_ts(fact)
            if cursor and ts <= cursor:
                continue
            if cfg.scopes and (getattr(fact, "scope", "") or "personal") not in cfg.scopes:
                continue
            # 敏感联动：Fact 血统上限低于目标暴露等级 → 不出境（fail-closed）
            if not exposure_allows(getattr(fact, "exposure_ceiling", "public"),
                                   cfg.exposure):
                continue
            batch.facts.append(fact)
            max_ts = max(max_ts, ts)
        for matter in self._index.all_matters():
            ts = _ts_of(matter, "updated_at")
            if cursor and ts <= cursor:
                continue
            if cfg.scopes and (getattr(matter, "scope", "") or "personal") not in cfg.scopes:
                continue
            batch.matters.append(matter)
            max_ts = max(max_ts, ts)
        for matter in self._index.all_matters():
            for edge in self._index.get_edges(matter.matter_id):
                ts = _ts_of(edge, "created_at")
                if cursor and ts <= cursor:
                    continue
                batch.edges.append(edge)
                max_ts = max(max_ts, ts)
        return batch, max_ts

    def _pending_tombstones(self, done: set[str]) -> list[tuple[str, str, str]]:
        """未处理墓碑 [(ledger_key, target_type, target_key)]。Memory Hub 缺席 → 空。"""
        if self._hub is None:
            return []
        out = []
        try:
            for key, tomb in self._hub.scan_tombstones():
                if key in done:
                    continue
                tt = str(getattr(tomb.target_type, "value", tomb.target_type))
                out.append((key, tt, tomb.target_key))
        except Exception as e:  # noqa: BLE001
            logger.warning("export_tombstone_scan_failed", error=str(e))
        return out

    # -- 主入口 --

    def run_once(self, force: bool = False) -> dict[str, dict[str, Any]]:
        """每 exporter 一次尝试。返回 {name: {synced, tombstones, skipped, error}}。"""
        results: dict[str, dict[str, Any]] = {}
        now = float(self._now())
        for cfg, exporter in self._bindings:
            r: dict[str, Any] = {"synced": 0, "tombstones": 0, "skipped": False,
                                 "backflow": {}}
            results[cfg.name] = r
            if not cfg.enabled:
                r["skipped"] = "disabled"
                continue
            st = self._state.get(cfg.name)
            if not force and now < float(st.get("next_attempt", 0)):
                r["skipped"] = "backoff"
                continue
            try:
                r["backflow"] = self._run_backflow(cfg, exporter)
                batch, new_cursor = self._collect_since(st.get("cursor", ""), cfg)
                # Z1.2 ⑦：硬规则按"上次导出的指纹"判断要不要重导——首轮照旧带；之后生效规则
                # 一变（`.env` 热加载 / Obsidian 回流落 Hub）就重写一次目标页 / 表，否则 vault 里
                # 留着旧规则，用户下次随手改那一页就把 `.env` 的改动悄悄盖回去。
                rules = self.current_hard_rules()
                rules_fp = _rules_fingerprint(rules)
                # （首轮没有指纹 ⇒ 带；升级前导出过的目标同样没有指纹 ⇒ 对齐一次。此前"游标为空
                #  就带"在空库上每轮都推一遍，指纹判据顺带把它收掉。）
                if rules and st.get("hard_rules_fp") != rules_fp:
                    batch.hard_rules = rules
                    if st.get("hard_rules_fp"):
                        logger.info("export_hard_rules_resync", exporter=cfg.name,
                                    count=len(rules), fingerprint=rules_fp,
                                    previous=st.get("hard_rules_fp"))
                if not batch.empty:
                    exporter.sync_incremental(batch, st.get("cursor", ""))
                    r["synced"] = (len(batch.facts) + len(batch.matters)
                                   + len(batch.edges))
                done = set(st.get("tombstones_done", []))
                for key, tt, tk in self._pending_tombstones(done):
                    exporter.handle_tombstone(tt, tk)
                    done.add(key)
                    r["tombstones"] += 1
                # 成功：推进游标 + 清退避
                st["cursor"] = new_cursor
                if rules:
                    st["hard_rules_fp"] = rules_fp
                st["tombstones_done"] = sorted(done)
                st["failures"] = 0
                st["next_attempt"] = 0.0
                st["last_ok"] = datetime.now().astimezone().isoformat()
                st["last_synced"] = r["synced"]
                if r["synced"] or r["tombstones"]:
                    logger.info("export_sync_done", exporter=cfg.name,
                                synced=r["synced"], tombstones=r["tombstones"],
                                cursor=st["cursor"])
            except Exception as e:  # noqa: BLE001 —— 目标不可达只降级不阻塞
                failures = int(st.get("failures", 0)) + 1
                backoff = min(_BACKOFF_BASE_S * (2 ** (failures - 1)), _BACKOFF_MAX_S)
                st["failures"] = failures
                st["next_attempt"] = now + backoff
                r["error"] = str(e)
                logger.warning("export_sync_failed", exporter=cfg.name,
                               failures=failures, backoff_s=backoff, error=str(e))
        self._state.save()
        return results

    @property
    def state(self) -> ExportStateStore:
        return self._state

    def change_probe(self, *, pg_every_s: float | None = None,
                     now: Any = time.monotonic) -> ExternalChangeProbe | None:
        """Z4b：给 consolidator 退避 sleep 用的外部变化探针；没有任何开着的回流目标 ⇒ None。"""
        if self._backflow is None:
            return None
        if pg_every_s is None:
            from bladex_core.flags import flag_number
            pg_every_s = flag_number("BLADEX_BACKFLOW_PG_PROBE_S")
        probe = ExternalChangeProbe(self._bindings, pg_every_s=pg_every_s, now=now)
        return probe if probe.armed else None


class ExternalChangeProbe:
    """Z4b：退避 sleep 期间的第二个唤醒源——外部回流目标有变化（第一个是 `_HubWatermark`）。

    - Obsidian：`change_signature()`（受管文件 stat 签名）**逐次**探，与 `mark()` 取的基线比；
    - Postgres：`inbox_has_new()`（一条 `SELECT max(id)`）按 `pg_every_s` 限频——
      sleep 循环 1 秒一跳，**不许每秒打一次库**；`inbox` 关着零 SQL（探针本身短路）。

    🔴 `mark()` 必须在**导出写完之后**调：每轮导出都可能重写页面，基线取早了，
    导出器自写就会被读成"用户改了"⇒ 每轮都被自己唤醒，退避永远退不上去
    （把批 X 的空载占空比 52.9% → 4.55% 吃回去）。接线守卫见 test_export_wake.py。

    任何异常都按"没变化"处理（退避照常，只是不提前唤醒），并告警一次。
    """

    def __init__(self, bindings: list[tuple[ExporterConfig, Any]], *, pg_every_s: float,
                 now: Any = time.monotonic) -> None:
        self._now = now
        self._pg_every = max(1.0, float(pg_every_s))
        self._sig: list[tuple[str, Any]] = []
        self._pg: list[tuple[str, Any]] = []
        for cfg, exp in bindings:
            if not cfg.enabled:
                continue
            if getattr(exp, "backflow_enabled", False) and hasattr(exp, "change_signature"):
                self._sig.append((cfg.name, exp))
            if getattr(exp, "inbox_enabled", False) and hasattr(exp, "inbox_has_new"):
                self._pg.append((cfg.name, exp))
        self._base: dict[str, Any] = {}
        self._pg_next = 0.0
        self._warned: set[str] = set()

    @property
    def armed(self) -> bool:
        return bool(self._sig or self._pg)

    def _safe(self, name: str, fn: Any) -> Any:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if name not in self._warned:
                self._warned.add(name)
                logger.warning("export_wake_probe_failed", exporter=name, error=str(e),
                               hint="no early wake from this target; normal rounds unaffected")
            return None

    def mark(self) -> None:
        """取基线（导出写完之后）。Postgres 不需要：它以上次取走的 id 为基线。"""
        for name, exp in self._sig:
            self._base[name] = self._safe(name, exp.change_signature)

    def has_change(self) -> str | None:
        """有变化 ⇒ 返回目标名（唤醒日志用），否则 None。"""
        for name, exp in self._sig:
            cur = self._safe(name, exp.change_signature)
            if cur is not None and name in self._base and cur != self._base[name]:
                return name
        if self._pg and self._now() >= self._pg_next:
            self._pg_next = self._now() + self._pg_every
            for name, exp in self._pg:
                if self._safe(name, exp.inbox_has_new):
                    return name
        return None


# ── 内置参考实现：JSONL 目录 exporter（也作 mock/调试用）──────────


class JsonlDirExporter:
    """把增量写成目录下的 JSONL 追加文件；墓碑记 tombstones.jsonl。

    参考实现（验证 worker 协议闭环）；生产 connector 见 Obsidian/PostgreSQL。
    options: {"dir": 目标目录}
    """

    def __init__(self, name: str, exposure: str, options: dict[str, Any]) -> None:
        self.name = name
        self.exposure = exposure
        self._dir = Path(os.path.expanduser(str(options.get("dir", "data/export_out"))))

    def sync_incremental(self, batch: ExportBatch, since_cursor: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with (self._dir / "entities.jsonl").open("a", encoding="utf-8") as f:
            for kind, items in (("fact", batch.facts), ("matter", batch.matters),
                                ("edge", batch.edges)):
                for it in items:
                    rec = it.model_dump(mode="json",
                                        exclude={"embedding", "centroid"})
                    f.write(json.dumps({"type": kind, **rec}, ensure_ascii=False) + "\n")
            for rule in batch.hard_rules:
                f.write(json.dumps({"type": "hard_rule", "content": rule},
                                   ensure_ascii=False) + "\n")

    def handle_tombstone(self, target_type: str, target_key: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        with (self._dir / "tombstones.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"target_type": target_type,
                                "target_key": target_key}) + "\n")


# ── 工厂/注册表（T14 Obsidian、T15 PostgreSQL 在此登记）──────────

_EXPORTER_FACTORIES: dict[str, Any] = {
    "jsonl": JsonlDirExporter,
}


def register_exporter(type_name: str, factory: Any) -> None:
    _EXPORTER_FACTORIES[type_name] = factory


def _lazy_register(type_name: str) -> None:
    """按需 import 具体 connector（未用到的不引其依赖：postgres 需 psycopg）。"""
    if type_name in _EXPORTER_FACTORIES:
        return
    if type_name == "obsidian":
        from bladex_proxy.exporters.obsidian import ObsidianExporter
        _EXPORTER_FACTORIES["obsidian"] = ObsidianExporter
    elif type_name == "postgres":
        from bladex_proxy.exporters.postgres import PostgresExporter
        _EXPORTER_FACTORIES["postgres"] = PostgresExporter


def build_exporters(configs: list[ExporterConfig]) -> list[tuple[ExporterConfig, Exporter]]:
    bindings: list[tuple[ExporterConfig, Exporter]] = []
    for cfg in configs:
        _lazy_register(cfg.type)
        factory = _EXPORTER_FACTORIES.get(cfg.type)
        if factory is None:
            raise ValueError(
                f"unknown exporter type {cfg.type!r}; "
                f"available: {sorted(_EXPORTER_FACTORIES) + ['obsidian', 'postgres']}")
        bindings.append((cfg, factory(cfg.name, cfg.exposure, cfg.options)))
    return bindings


def build_worker_from_config(
    index: Any, ledger: Any, config_path: str | Path,
    state_path: str | Path = "data/export_state.json",
    hard_rules: list[str] | Callable[[], list[str]] | None = None,
    backflow: AdminBackflowClient | None = None,
) -> ExportSyncWorker | None:
    """从 export.toml 建 worker；无配置/全禁用 → None（功能关闭）。

    建好后逐目标打一行 `export_backflow_status`（回流 on/off）——Postgres 收件默认关，
    打开时这一行就是"它真的开了"的证据。
    """
    configs = load_export_config(config_path)
    configs = [c for c in configs if c.enabled]
    if not configs:
        return None
    worker = ExportSyncWorker(index, ledger, build_exporters(configs), state_path,
                              hard_rules=hard_rules, backflow=backflow)
    for name, state in worker.backflow_status().items():
        logger.info("export_backflow_status", exporter=name, backflow=state,
                    max_deletes=worker._max_deletes,
                    wired=backflow is not None)
    return worker
