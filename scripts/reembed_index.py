"""Memory Index 全量重嵌迁移 — 切换 embedding 后端/模型后运行（embedding 后端可选化，2026-07-26）。

向量空间全局唯一（ADR-0009）：换 embedding 后端/模型 = 换空间，Memory Index 的 model_id
不变量会拒绝混写。本脚本用**当前 env 配置的新 embedder** 重算全部向量：

  1. 读全部 Fact metadata（RocksDB，内容不动 —— 不动 LLM 蒸馏台账、不动 Memory Hub）；
  2. 批量重嵌 fact.content（passage 语义）；
  3. drop + 重建 LanceDB facts / matters 表（新 dim）；
  4. 重算全部 Matter 质心（复用 recompute_centroid 加权逻辑）+ 重建质心向量表；
  5. 盖新 model_id 章。

用法（先停 proxy 和 consolidator，操作独占 Memory Index 写锁）：
  .venv/bin/python scripts/reembed_index.py          # 交互确认
  .venv/bin/python scripts/reembed_index.py --yes    # 免确认

耗时：本地 e5-large ~20ms/条；api 档取决于 RTT（存量数百条 = 秒级~分钟级）。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("packages/bladex-core", "packages/bladex-proxy"):
    _full = os.path.join(_ROOT, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)


def _load_env_file() -> None:
    env_path = os.path.join(_ROOT, "config", ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


_load_env_file()

_BATCH = 64


def main() -> int:
    parser = argparse.ArgumentParser(description="BladeX Memory Index full re-embed migration")
    parser.add_argument("--yes", action="store_true", help="跳过交互确认")
    args = parser.parse_args()

    from bladex_core.consolidation_proxy import embed_passage_compat
    from bladex_proxy.config import ProxyConfig
    from bladex_proxy.embedding import build_embedder, validate_embed_sensitivity
    from bladex_proxy.storage.memory_index import _EMBED_MODEL_KEY, MemoryIndex

    cfg = ProxyConfig()
    validate_embed_sensitivity(cfg)
    embedder = build_embedder(cfg, role="consolidator")
    embedder.embed(["warmup"])  # 触发加载/probe，拿到 model_identity
    new_identity = getattr(embedder, "model_identity", None)
    if not new_identity:
        print("ERROR: embedder has no model_identity; abort")
        return 1

    # embed_model_id=None：迁移操作自身绕过不变量校验
    index = MemoryIndex(cfg.index_path, embedder=embedder, embed_model_id=None,
                   novelty_threshold=cfg.novelty_threshold,
                   entity_aware_novelty=cfg.entity_aware_novelty,
                   entity_overlap_threshold=cfg.entity_overlap_threshold,
                   novelty_topk=cfg.novelty_topk,
                   entity_aware_rerank=cfg.entity_aware_rerank,
                   entity_rerank_alpha=cfg.entity_rerank_alpha,
                   entity_rerank_expand_k=cfg.entity_rerank_expand_k)
    index.open()

    raw = None
    if index.meta_db is not None:
        raw = index.meta_db.get(_EMBED_MODEL_KEY.encode())
    old_identity = raw.decode() if isinstance(raw, bytes) else raw

    facts = index.all_facts()
    matters = index.all_matters()
    print(f"Memory Index: {len(facts)} facts, {len(matters)} matters")
    print(f"vector space: {old_identity or '(unstamped)'} -> {new_identity}")
    if old_identity == new_identity:
        print("model identity unchanged; re-embed is only needed after a backend/model switch.")
    if not args.yes:
        ans = input("proceed with full re-embed? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("aborted")
            index.close()
            return 1

    t0 = time.perf_counter()

    # 1) 批量重嵌 fact contents（passage 语义，与 consolidation 写入路径一致）
    for i in range(0, len(facts), _BATCH):
        batch = facts[i:i + _BATCH]
        vecs = embed_passage_compat(embedder, [f.content for f in batch])
        for f, v in zip(batch, vecs, strict=True):
            f.embedding = v
        print(f"  embedded {min(i + _BATCH, len(facts))}/{len(facts)}")

    # 2) drop 旧向量表（dim 可能已变，不能就地更新）
    if index._lancedb is not None:
        for tbl in ("facts", "matters"):
            try:
                index._lancedb.drop_table(tbl)
                print(f"  dropped lancedb table: {tbl}")
            except Exception:
                pass
        index._table = None
        index._matter_table = None

    # 3) 重建 facts 向量表
    for f in facts:
        index._add_to_lancedb(f)
    print(f"  facts table rebuilt: {len(facts)} rows")

    # 4) Matter 质心：从归属边 + 新 fact 向量重算（复用加权逻辑），再重建质心向量表
    rebuilt_matters = 0
    for m in matters:
        index.recompute_centroid(m.matter_id)  # 更新 metadata 里的 centroid
        fresh = index.get_matter(m.matter_id)
        if fresh is not None and fresh.centroid:
            index._add_matter_to_lancedb(fresh)
            rebuilt_matters += 1
    print(f"  matters centroid rebuilt: {rebuilt_matters}/{len(matters)}")

    # 5) 盖新章
    if index.meta_db is not None:
        index.meta_db[_EMBED_MODEL_KEY.encode()] = new_identity.encode()
    index.close()

    elapsed = time.perf_counter() - t0
    print(f"done in {elapsed:.1f}s. model_id stamped: {new_identity}")
    print("restart proxy + consolidator to pick up the new vector space.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
