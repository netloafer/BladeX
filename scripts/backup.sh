#!/usr/bin/env bash
# BladeX 备份（ADR-0021 §4）-- RocksDB checkpoint（在线，不停服）+ LanceDB 目录 + Redis AOF。
#
# 用法：bash scripts/backup.sh [备份目标目录]（默认 ./data/backup_<ts>）
# 依赖：redis-cli（可选，无则跳过 Redis 备份）。RocksDB checkpoint 用 Python rocksdict 做。
#
# 恢复见 scripts/restore.sh（恢复后触发 Memory Index 重建）。
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 加载 .env（取存储路径）--但保留调用方已设的 BLADEX_* 覆盖（.env 的 export 会覆盖）。
_SAVE_ROCKS="${BLADEX_ROCKSDB_PATH:-}"
_SAVE_INDEX="${BLADEX_INDEX_PATH:-}"
_SAVE_REDIS="${BLADEX_REDIS_URL:-}"
[ -f config/.env ] && source config/.env
[ -n "$_SAVE_ROCKS" ] && BLADEX_ROCKSDB_PATH="$_SAVE_ROCKS"
[ -n "$_SAVE_INDEX" ] && BLADEX_INDEX_PATH="$_SAVE_INDEX"
[ -n "$_SAVE_REDIS" ] && BLADEX_REDIS_URL="$_SAVE_REDIS"

ROCKS_PATH="${BLADEX_ROCKSDB_PATH:-data/bladex_hub}"
INDEX_PATH="${BLADEX_INDEX_PATH:-data/bladex_index}"
REDIS_URL="${BLADEX_REDIS_URL:-redis://127.0.0.1:6379/0}"
DEST="${1:-data/backup_$(date +%Y%m%d_%H%M%S)}"

echo "=== BladeX 备份 -> $DEST ==="
mkdir -p "$DEST"

# 1. RocksDB checkpoint（在线，不停服--checkpoint 是 RocksDB 原生一致快照）
echo "[1/3] RocksDB checkpoint..."
DEST_CKPT="${DEST}/bladex_hub"
rm -rf "$DEST_CKPT"
PYTHONPATH=packages/bladex-core:packages/bladex-proxy .venv/bin/python - <<PY
import os, rocksdict
opts = rocksdict.Options()
# read_only 打开避免与运行中 proxy 抢写锁；Checkpoint API 在 read_only 句柄上可用。
db = rocksdict.Rdict("${ROCKS_PATH}", opts, None, rocksdict.AccessType.read_only())
ckpt = rocksdict.Checkpoint(db)
ckpt.create_checkpoint("${DEST_CKPT}")  # 路径须不存在，create_checkpoint 创建之
db.close()
print("  RocksDB checkpoint done")
PY

# 2. LanceDB + Memory Index meta（目录拷贝；consolidator 周期写，短暂不一致可接受--Memory Index 可从 Memory Hub 重建）
echo "[2/3] Memory Index (LanceDB + meta) 拷贝..."
if [ -d "$INDEX_PATH" ]; then
    cp -R "$INDEX_PATH" "$DEST/bladex_index"
    echo "  Memory Index copied"
else
    echo "  Memory Index 目录不存在（consolidator 未跑过），跳过"
fi

# 3. Redis AOF（可选）
echo "[3/3] Redis BGSAVE..."
_REDIS_HOST=$(echo "$REDIS_URL" | sed -n 's|^redis://\([^:/]*\).*|\1|p')
_REDIS_PORT=$(echo "$REDIS_URL" | sed -n 's|.*:\([0-9]*\)/.*|\1|p')
_REDIS_HOST="${_REDIS_HOST:-127.0.0.1}"
_REDIS_PORT="${_REDIS_PORT:-6379}"
if command -v redis-cli &>/dev/null; then
    redis-cli -h "$_REDIS_HOST" -p "$_REDIS_PORT" BGSAVE >/dev/null 2>&1 || true
    # 等待 BGSAVE 完成
    sleep 1
    echo "  Redis BGSAVE 触发（AOF 文件见 redis 数据目录）"
else
    echo "  redis-cli 不可用，跳过 Redis 备份（Pipeline 是缓冲，掉电由 AOF + 磁盘溢出兜底）"
fi

echo "=== 备份完成：$DEST ==="
echo "恢复：bash scripts/restore.sh $DEST"
