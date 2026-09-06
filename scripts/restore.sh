#!/usr/bin/env bash
# BladeX 恢复（ADR-0021 §4）-- 从备份目录恢复 Memory Hub RocksDB + Memory Index + Redis，触发 Memory Index 重建。
#
# 用法：bash scripts/restore.sh <备份目录>
# 注意：恢复前请停止 proxy + consolidator（避免写冲突）。恢复后启动 consolidator
# 会自动重建 Memory Index（从 Memory Hub 重放），确保 Memory Index 与 Memory Hub 一致。
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# 加载 .env --保留调用方覆盖（同 backup.sh）。
_SAVE_ROCKS="${BLADEX_ROCKSDB_PATH:-}"
_SAVE_INDEX="${BLADEX_INDEX_PATH:-}"
[ -f config/.env ] && source config/.env
[ -n "$_SAVE_ROCKS" ] && BLADEX_ROCKSDB_PATH="$_SAVE_ROCKS"
[ -n "$_SAVE_INDEX" ] && BLADEX_INDEX_PATH="$_SAVE_INDEX"

ROCKS_PATH="${BLADEX_ROCKSDB_PATH:-data/bladex_hub}"
INDEX_PATH="${BLADEX_INDEX_PATH:-data/bladex_index}"
SRC="${1:-}"

if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
    echo "用法：bash scripts/restore.sh <备份目录>"
    exit 1
fi

echo "=== BladeX 恢复 <- $SRC ==="
echo "警告：恢复前请确保 proxy + consolidator 已停止（避免写冲突）。"
echo "现有数据将被覆盖。5 秒后开始（Ctrl-C 中止）..."
sleep 5

# 1. Memory Hub RocksDB（删除现有 + 拷贝 checkpoint）
echo "[1/3] 恢复 Memory Hub RocksDB..."
# 🔴 备份里的子目录名新旧都认（2026-08-07 bladex_rocksdb -> bladex_ledger；2026-09-05 -> bladex_hub）。
# 只认新名 = 改名之前做的备份**一个都恢复不了**，而备份的全部意义就是出事时能回去。
SRC_LEDGER=""
for _n in bladex_hub bladex_ledger bladex_rocksdb; do
    [ -d "$SRC/$_n" ] && { SRC_LEDGER="$SRC/$_n"; break; }
done
if [ -n "$SRC_LEDGER" ]; then
    rm -rf "$ROCKS_PATH"
    cp -R "$SRC_LEDGER" "$ROCKS_PATH"
    echo "  Memory Hub restored (from $(basename "$SRC_LEDGER"))"
else
    echo "  备份无 bladex_hub / bladex_ledger / bladex_rocksdb，跳过（保留现有 Memory Hub）"
fi

# 2. Memory Index（可选--恢复后会被重建覆盖，但先恢复可减少重建时间）
echo "[2/3] 恢复 Memory Index..."
SRC_INDEX=""
for _n in bladex_index bladex_p2; do
    [ -d "$SRC/$_n" ] && { SRC_INDEX="$SRC/$_n"; break; }
done
if [ -n "$SRC_INDEX" ]; then
    rm -rf "$INDEX_PATH"
    cp -R "$SRC_INDEX" "$INDEX_PATH"
    echo "  Memory Index restored (from $(basename "$SRC_INDEX"))"
else
    echo "  备份无 bladex_index / bladex_p2，跳过（启动后 consolidator 会从 Memory Hub 重建）"
fi

# 3. Redis（AOF 由 redis-server 启动时从 appendonly.aof 重放；这里提示）
echo "[3/3] Redis AOF..."
echo "  Redis 由其数据目录的 appendonly.aof 自动重放（见 redis 配置 --dir）。"
echo "  若用容器，redis-data 卷已含 AOF，重启 redis 服务即恢复。"

echo "=== 恢复完成 ==="
echo "下一步："
echo "  1. 启动 proxy + redis（bash config/run_proxy.sh）"
echo "  2. 启动 consolidator（bash config/run_consolidator.sh start）--自动从 Memory Hub 重建 Memory Index"
echo "  3. 可选：全量重建核对一致性：.venv/bin/python scripts/run_index_consolidator.py --full"
