"""embedding 预检（`start` 与 `doctor` 共用）。批 LC 自 `lifecycle.py` 零行为搬出，不含 `@app.command`，故不影响命令登记顺序。"""

from __future__ import annotations

import time
from pathlib import (  # noqa: E402
    Path,
)


def _embed_api_probe(settings) -> tuple[int, float]:  # noqa: ANN001
    """一次真实 api 探活（doctor 与 start 预检共用一份实现）。

    返回 (dim, rtt_ms)；失败原样抛，措辞与去留由调用方决定。
    """
    from bladex_proxy.embedding import RouterEmbedAdapter
    adapter = RouterEmbedAdapter(model=settings.model, api_base=settings.api_base,
                                 api_key=settings.api_key, query_cache_size=0)
    t0 = time.perf_counter()
    vecs = adapter.embed_query(["probe"])
    rtt_ms = (time.perf_counter() - t0) * 1000.0
    dim = len(vecs[0]) if vecs and vecs[0] else 0
    return dim, rtt_ms


def _preflight_embedding(cfg, force: bool = False) -> bool:  # noqa: ANN001
    """G18.5（MQ-W10，Jason 2026-09-18 拍板）：embedding 配置预检，错误中止启动。

    记忆是护城河——配置错误不许降级成"proxy 活着但记忆死了"（MQ-L61 同族）。
    预检只判**配置错误** + api 档一次真探活；运行时故障的降级路径
    （app_factory 的 except Exception）保持不动，那条线是 MQ-L61 划的。

    - ①模型名空 / ②key 声明未解析：**恒中止**（这两种配置下记忆必死，--force 无意义）；
    - ③引擎与盖章不符：中止 + 三条出路（回退配置 / reembed+重标定 / --force 强启）；
    - api 探活失败：中止，--force 可放行（网络临时不通时允许带病起）。
    ③只对 local/api 档判（proxy/ipc 档身份要握手后才知道，预检不做负面判定——
    与 memory_index 「空身份不作负面判定」同则）。
    """
    from bladex_proxy import router_sdk
    from bladex_proxy.embedding import (
        _sensitivity_enabled,
        embed_model_identity,
        resolve_embed_settings,
    )
    from bladex_proxy.global_config import embedding_section
    s = resolve_embed_settings(cfg)
    # 敏感层开着时 api 档在运行时被强制回本地（fail-closed 既有行为）——
    # 预检跟随**生效**形态：不按声明的 api 检查、不探活，只做③的本地身份对账。
    api_effective = s.backend == "api" and not _sensitivity_enabled(cfg)

    if api_effective:
        if not s.model:
            print("✗ Embedding preflight: backend=api but no embedding model configured.")
            print('  fix: set [embedding].model in config/bladex.toml (e.g. '
                  '"openai/text-embedding-3-small") or BLADEX_EMBED_MODEL')
            return False
        key_env = embedding_section(cfg).config.api_key_env
        if key_env and not s.api_key:
            print(f"✗ Embedding preflight: api key env {key_env!r} is declared in "
                  "the [embedding] config but resolves to empty.")
            print(f"  fix: export {key_env}=<key> in the environment that runs bladex "
                  "(keys never go into config files themselves)")
            return False

    if s.backend in ("local", "api"):
        expected = embed_model_identity(cfg, role="proxy")
        sidecar = Path(getattr(cfg, "index_path", "") or "") / "embed_model_id"
        stored = sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else ""
        if stored and stored != expected:
            if not force:
                print(f"✗ Embedding preflight: engine changed -- Memory Index is stamped "
                      f"{stored!r} but current config resolves to {expected!r}.")
                print("  Starting like this would silently disable vector search (proxy) "
                      "and make the consolidator refuse to write. Pick one:")
                print("  1) keep your data: revert the [embedding] config (backend/model) "
                      "to the stamped engine;")
                print("  2) switch engines for real: rebuild the vector index with "
                      "`python scripts/reembed_index.py` (full re-embed; with backend=api "
                      "this sends ALL memory content to the provider) and re-calibrate "
                      "thresholds on real traffic (non-e5 profiles ship uncalibrated defaults);")
                print("  3) start anyway: `bladex start --force` (memory degraded as above).")
                return False
            print(f"⚠ Embedding preflight: engine mismatch (store={stored!r} "
                  f"current={expected!r}) -- continuing due to --force; vector search "
                  "disabled, consolidator will refuse to write.")

    if api_effective:
        try:
            dim, rtt_ms = _embed_api_probe(s)
        except Exception as exc:  # noqa: BLE001 — 探活失败=配置/网络问题，按 --force 决定去留
            print(f"✗ Embedding preflight: api probe failed: {router_sdk.error_text(exc)}")
            print("  fix: check [embedding] model / api_base and the api key; the model "
                  "name must be a Router embedding id like provider/model")
            if not force:
                print("  (start anyway with `bladex start --force`; memory writes will "
                      "fail until this is fixed)")
                return False
            print("  -- continuing due to --force")
            return True
        print(f"  Embedding:    embed_api_probe_ok dim={dim} rtt_ms={rtt_ms:.0f}")
    return True
