"""Beta T13（B3.2）：export sync worker——增量同步 Memory Index 记忆到外部目标。

跑在 consolidator 进程（独占写者旁的只读消费），主管线之后：
  - 水位游标持久化（JSON 状态文件，按 exporter 隔离）；
  - 失败指数退避（目标不可达只降级不阻塞 rebuild 主管线）；
  - scope / exposure 过滤（ADR-0021：越级 Fact 不出境）；
  - 墓碑同步（done 集合去重，目标侧删除）。

配置 `config/export.toml`（凭证只写 env 变量名，见 export.toml.example）。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from bladex_core.exporters import ExportBatch, Exporter
from bladex_core.sensitivity import exposure_allows

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


class ExportSyncWorker:
    """增量同步执行器。consolidator 循环里每轮 `run_once()`；异常永不外抛。"""

    def __init__(
        self,
        index: Any,
        ledger: Any,
        bindings: list[tuple[ExporterConfig, Exporter]],
        state_path: str | Path,
        hard_rules: list[str] | None = None,
        now: Any = time.time,
    ) -> None:
        self._index = index
        self._hub = ledger
        self._bindings = bindings
        self._state = ExportStateStore(state_path)
        self._hard_rules = hard_rules or []
        self._now = now

    # -- 数据收集（游标 = 实体时间戳 ISO 水位）--

    def _collect_since(self, cursor: str, cfg: ExporterConfig) -> tuple[ExportBatch, str]:
        batch = ExportBatch(hard_rules=list(self._hard_rules) if not cursor else [])
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
            r: dict[str, Any] = {"synced": 0, "tombstones": 0, "skipped": False}
            results[cfg.name] = r
            if not cfg.enabled:
                r["skipped"] = "disabled"
                continue
            st = self._state.get(cfg.name)
            if not force and now < float(st.get("next_attempt", 0)):
                r["skipped"] = "backoff"
                continue
            try:
                batch, new_cursor = self._collect_since(st.get("cursor", ""), cfg)
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
    hard_rules: list[str] | None = None,
) -> ExportSyncWorker | None:
    """从 export.toml 建 worker；无配置/全禁用 → None（功能关闭）。"""
    configs = load_export_config(config_path)
    configs = [c for c in configs if c.enabled]
    if not configs:
        return None
    return ExportSyncWorker(index, ledger, build_exporters(configs), state_path,
                            hard_rules=hard_rules)
