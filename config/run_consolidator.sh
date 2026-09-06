#!/usr/bin/env bash
# 兼容薄壳（ADR-0027 §4.4 / Y4）—— 真实逻辑已收编进 `bladex consolidator`。
# 用法不变：bash config/run_consolidator.sh [start|stop|status|restart]
#
# 推荐直接用 CLI（在任意安装形态下都成立，不依赖仓库目录）：
#   bladex consolidator start|stop|status|restart
#
# 注意：CLI 启动的是**包内入口** `python -m bladex_proxy.consolidator`
# （ADR-0027 §4.2），不再是 scripts/run_index_consolidator.py——后者已是薄壳。

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Error: config/.env not found. Run 'bladex init' first, or copy config/.env.example." >&2
    exit 1
fi

ACTION="${1:-start}"
case "$ACTION" in
    start|stop|status|restart) ;;
    *) echo "Usage: bash config/run_consolidator.sh [start|stop|status|restart]" >&2; exit 1 ;;
esac

if [ -x ".venv/bin/bladex" ]; then
    exec .venv/bin/bladex consolidator "$ACTION"
elif [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python -m bladex_proxy.cli consolidator "$ACTION"
elif command -v bladex >/dev/null 2>&1; then
    exec bladex consolidator "$ACTION"
else
    echo "Error: no bladex entry point found (.venv/bin/bladex, .venv/bin/python, or bladex on PATH)." >&2
    exit 1
fi
