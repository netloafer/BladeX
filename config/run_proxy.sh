#!/usr/bin/env bash
# 兼容薄壳（ADR-0027 §4.4 / Y4）—— 真实启动逻辑已收编进 `bladex start`。
#
# 为什么薄壳化：这个脚本与 CLI 曾各自实现一遍 Redis 自启、consolidator 拉起、
# 日志落盘，两边会漂移（而漂移的那一侧会在用户机上悄悄失效）。现在只有一处实现。
#
# 推荐直接用 CLI：
#   bladex start                后台启动（redis + consolidator + proxy），并报 ready/degraded
#   bladex start --foreground   前台（本脚本等价物）
#   bladex stop / status / status --traffic
#
# 本脚本保留的额外动作：开发形态的 editable install（CLI 面向已安装的用户，不做这件事）。
# 跳过安装：BLADEX_SKIP_INSTALL=1 bash config/run_proxy.sh

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo "Error: config/.env not found. Run 'bladex init' first, or copy config/.env.example." >&2
    exit 1
fi

echo "note: run_proxy.sh is now a thin wrapper around 'bladex start --foreground'."

if [ "${BLADEX_SKIP_INSTALL:-0}" != "1" ] && [ -f "pyproject.toml" ]; then
    echo "installing packages (editable)..."
    uv pip install -e packages/bladex-proxy -e packages/bladex-core >/dev/null || {
        echo "editable install failed; run manually:" >&2
        echo "  uv pip install -e packages/bladex-proxy -e packages/bladex-core" >&2
        exit 1
    }
fi

if [ -x ".venv/bin/bladex" ]; then
    exec .venv/bin/bladex start --foreground "$@"
elif [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python -m bladex_proxy.cli start --foreground "$@"
elif command -v bladex >/dev/null 2>&1; then
    exec bladex start --foreground "$@"
else
    echo "Error: no bladex entry point found (.venv/bin/bladex, .venv/bin/python, or bladex on PATH)." >&2
    exit 1
fi
