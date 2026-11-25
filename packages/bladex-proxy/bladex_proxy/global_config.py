"""全局配置 `config/bladex.toml` 的单一读取点（批 AE，拍板 (7) 甲）。

embedding 配置不是路由配置，所以从 `routing.toml` 搬出来；`[distill]` 等其它段以后按需迁。
embed_server / proxy / consolidator / CLI 都经 `embedding.resolve_embed_settings` 走到这里。

优先级（逐键）：env > `bladex.toml` > `routing.toml [embedding]` > 默认。本模块只管中间两级，
env 与默认在 `resolve_embed_settings` 的 `pick` 里。

🔴 文件内容**每次调用都重读**：这里不在热路径上（启动期与 CLI），而 `api_base_env` 的解析依赖
当时的环境。只有日志按进程去重——一个进程会合并好几次（见结算稿 §1.3）。

依赖方向：本模块 → `routing_config`（schema 与 api_base 校验沿用同一份），不反向。
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import structlog

from bladex_proxy.routing_config import EmbeddingConfig, RoutingConfig, RoutingConfigError

logger = structlog.get_logger()

#: 指定全局配置路径的 env；不设 = 与 `routing.toml` 同目录（隔离实例只改 routing 路径时也跟着走）。
GLOBAL_CONFIG_ENV = "BLADEX_GLOBAL_CONFIG"
GLOBAL_CONFIG_NAME = "bladex.toml"
DEPRECATION_EVENT = "config_embedding_section_deprecated"

ORIGIN_GLOBAL = "bladex.toml"
ORIGIN_LEGACY = "routing.toml"

#: 逻辑键。🔴 `api_base` 含 `api_base_env` 两种写法：它们是同一个键，`api_base_env` 又优先，
#: 按字段名分开合并会让旧段的 `api_base_env` 压过新文件的 `api_base`（结算稿 §1.5）。
#: 这里不列 `api_base_env`：两个文件读进来时它都已展开进 `api_base`。
_SCALAR_KEYS = ("backend", "model", "api_base", "api_key_env",
                "proxy_url", "proxy_key_env", "query_cache_size")
_KNOWN_TOML_KEYS = frozenset(_SCALAR_KEYS) | {"api_base_env", "profiles"}


class GlobalConfigError(RoutingConfigError):
    """`bladex.toml` 写错了。继承 `RoutingConfigError`：lifespan 对它**不降级、拒绝启动**（MQ-L61）。"""


@dataclass
class EmbeddingSection:
    """合并后的 `[embedding]`：值 + 每个键来自哪个文件。"""

    config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    #: 逻辑键 → `bladex.toml` | `routing.toml`；没写的键不在表里。
    origins: dict[str, str] = field(default_factory=dict)
    #: 旧段里写了、但被新文件压掉的键。
    ignored_legacy_keys: list[str] = field(default_factory=list)
    #: 两个文件里各自出现过的键（`resolve_embed_settings` 判「多处出现」用）。
    global_keys: frozenset[str] = frozenset()
    legacy_keys: frozenset[str] = frozenset()


def global_config_path(cfg: Any = None) -> Path:
    """全局配置路径：`BLADEX_GLOBAL_CONFIG` > 与 `routing.toml` 同目录。"""
    explicit = os.environ.get(GLOBAL_CONFIG_ENV, "").strip()
    if explicit:
        return Path(explicit)
    routing = getattr(cfg, "routing_config_path", "") or "config/routing.toml"
    return Path(str(routing)).parent / GLOBAL_CONFIG_NAME


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise GlobalConfigError(f"{path}: TOML syntax error: {e}") from e


def _unknown_keys(data: dict[str, Any], section: dict[str, Any]) -> tuple[list[str], list[str]]:
    return sorted(set(data) - {"embedding"}), sorted(set(section) - _KNOWN_TOML_KEYS)


def unknown_keys(cfg: Any = None) -> tuple[list[str], list[str]]:  # noqa: ANN401
    """`bladex.toml` 里不会被读取的 (段, `[embedding]` 下的键)（给 `bladex config check`，批 DS2）。

    语法错误抛 `GlobalConfigError`。
    """
    data = _read_toml(global_config_path(cfg))
    raw = data.get("embedding")
    return _unknown_keys(data, raw if isinstance(raw, dict) else {})


def _warn_unknown_keys(path: Path, data: dict[str, Any], section: dict[str, Any]) -> None:
    """写错键名、或把还没迁的段写进来，都要看得见（否则就是一条静默不生效的配置）。"""
    unknown_sections, unknown_embedding = _unknown_keys(data, section)
    if unknown_sections or unknown_embedding:
        _log_once(("unknown", str(path), tuple(unknown_sections), tuple(unknown_embedding)),
                  logger.warning, "global_config_unknown_keys", path=str(path),
                  sections=unknown_sections, embedding_keys=unknown_embedding,
                  hint="these are not read from bladex.toml; check the spelling, or keep "
                       "sections that have not moved yet in routing.toml")


def load_global_embedding(cfg: Any = None) -> EmbeddingConfig | None:
    """读 `bladex.toml [embedding]`；文件或段不存在返回 None，写错抛 `GlobalConfigError`。"""
    path = global_config_path(cfg)
    data = _read_toml(path)
    if "embedding" not in data:
        if data:
            _warn_unknown_keys(path, data, {})
        return None
    raw = data["embedding"]
    if not isinstance(raw, dict):
        raise GlobalConfigError(f"{path}: [embedding] must be a table")
    _warn_unknown_keys(path, data, raw)
    try:
        section = EmbeddingConfig.model_validate(raw)
    except ValueError as e:
        raise GlobalConfigError(f"{path}: invalid [embedding]: {e}") from e
    try:
        section.api_base = RoutingConfig._resolve_one_api_base(
            what=f"{path} [embedding]", literal=section.api_base,
            env_name=section.api_base_env, urlparse=urlparse)
    except RoutingConfigError as e:
        raise GlobalConfigError(str(e)) from e
    return section


def _set_keys(section: Any) -> frozenset[str]:
    """这一段写了哪些逻辑键（空串 / None 与「没写」同义，沿用旧 `pick` 的判据）。"""
    if section is None:
        return frozenset()
    found = {k for k in _SCALAR_KEYS if getattr(section, k, None) not in (None, "")}
    found |= {f"profiles.{name}" for name in (getattr(section, "profiles", None) or {})}
    return frozenset(found)


def embedding_section(cfg: Any = None) -> EmbeddingSection:
    """合并 `bladex.toml` 与旧 `routing.toml [embedding]`（**逐键**，新文件优先）。

    逐键而不是整段：只搬了一半（比如只搬了 `backend`）时，还留在旧段的 `model` 继续生效。
    整段忽略的话 `model` 会静默丢掉，而它决定向量空间。
    """
    legacy = getattr(getattr(cfg, "routing_config", None), "embedding", None)
    new = load_global_embedding(cfg)
    legacy_keys, new_keys = _set_keys(legacy), _set_keys(new)

    merged = EmbeddingConfig()
    origins: dict[str, str] = {}
    for key in _SCALAR_KEYS:
        src, origin = (new, ORIGIN_GLOBAL) if key in new_keys else (legacy, ORIGIN_LEGACY)
        if key not in new_keys and key not in legacy_keys:
            continue
        setattr(merged, key, getattr(src, key, None))
        origins[key] = origin
    profiles: dict[str, Any] = dict(getattr(legacy, "profiles", None) or {})
    for name in profiles:
        origins[f"profiles.{name}"] = ORIGIN_LEGACY
    for name, prof in (getattr(new, "profiles", None) or {}).items():
        profiles[name] = prof
        origins[f"profiles.{name}"] = ORIGIN_GLOBAL
    merged.profiles = profiles

    result = EmbeddingSection(
        config=merged, origins=origins,
        ignored_legacy_keys=sorted(legacy_keys & new_keys),
        global_keys=new_keys, legacy_keys=legacy_keys)
    if legacy_keys:
        _warn_legacy_section(cfg, result)
    return result


# ── 日志去重：一个进程只说一次 ────────────────────────────────────────────────

_logged: set[tuple[Any, ...]] = set()


def _log_once(once_key: tuple[Any, ...], emit: Any, event: str, **fields: Any) -> None:
    if once_key in _logged:
        return
    _logged.add(once_key)
    emit(event, **fields)


def reset_log_once() -> None:
    """测试用：清掉去重记录。"""
    _logged.clear()


def _warn_legacy_section(cfg: Any, section: EmbeddingSection) -> None:
    keys = sorted(section.legacy_keys)
    target = str(global_config_path(cfg))
    _log_once(("legacy", tuple(keys), tuple(section.ignored_legacy_keys)),
              logger.warning, DEPRECATION_EVENT,
              routing_config=str(getattr(cfg, "routing_config_path", "") or ""),
              keys=keys, ignored_keys=section.ignored_legacy_keys,
              hint=f"routing.toml [embedding] is deprecated: move it to {target}"
                   + (f"; ignored there because {target} sets them: "
                      f"{', '.join(section.ignored_legacy_keys)}"
                      if section.ignored_legacy_keys else ""))


def log_key_resolution(key: str, winner: str, present_in: list[str]) -> None:
    """同一个键在多处出现时说明最终取自哪里（一行 info；不带值，值可能是地址或 env 名）。"""
    _log_once(("key", key, winner, tuple(present_in)), logger.info,
              "embed_config_key_resolved", key=key, winner=winner, present_in=present_in)


# ── doctor：只提示，不改文件 ──────────────────────────────────────────────────


def _toml_line(key: str, value: Any) -> str:
    if isinstance(value, bool):
        return f"{key} = {'true' if value else 'false'}"
    if isinstance(value, (int, float)):
        return f"{key} = {value}"
    return f'{key} = "{value}"'


def migration_hint(cfg: Any) -> list[str]:
    """旧段的迁移提示（给 `bladex doctor`）。没有旧段返回空表。

    读**原始文件**而不是解析后的对象：`api_base_env` 解析后 `api_base` 已是展开值，
    照那个提示用户搬，配置里就多了一个写死的地址。
    """
    routing_path = Path(str(getattr(cfg, "routing_config_path", "") or ""))
    if not routing_path.is_file():
        return []
    try:
        with routing_path.open("rb") as f:
            raw = tomllib.load(f).get("embedding")
    except tomllib.TOMLDecodeError:
        return []  # routing.toml 的语法错误有自己那一条检查
    if not isinstance(raw, dict) or not raw:
        return []

    target = global_config_path(cfg)
    new_keys = _set_keys(load_global_embedding(cfg))

    def _logical(k: str) -> str:
        return "api_base" if k == "api_base_env" else k

    scalars = {k: v for k, v in raw.items() if k != "profiles"}
    profiles = raw.get("profiles") or {}
    move = {k: v for k, v in scalars.items() if _logical(k) not in new_keys}
    move_profiles = sorted(n for n in profiles if f"profiles.{n}" not in new_keys)
    ignored = sorted({k for k in scalars if _logical(k) in new_keys}
                     | {f"profiles.{n}" for n in profiles if f"profiles.{n}" in new_keys})

    out = [f"routing.toml [embedding] is deprecated; embedding settings now live in {target}."]
    if ignored:
        out.append(f"Ignored in routing.toml because {target} sets them: {', '.join(ignored)}. "
                   "Delete them from routing.toml.")
    if move or move_profiles:
        out.append(f"Still read from routing.toml. Move these lines to [embedding] in {target}:")
        out.extend(f"  {_toml_line(k, v)}" for k, v in move.items())
        out.extend(f'  [embedding.profiles."{n}"] (the whole table)' for n in move_profiles)
    out.append("Then delete the [embedding] section from routing.toml "
               "(keep it if you may roll back to 0.3.0, which only reads routing.toml). "
               "BladeX does not edit your config files.")
    return out

