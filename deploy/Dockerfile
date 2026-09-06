# BladeX 自部署镜像（proxy + consolidator 共用，ADR-0021 §4 单机基线）。
# Python 3.11+，uv 管理依赖。embedding 模型首次启动自动下载（默认 bge-small-en
# 约 0.07GB；pin multilingual-e5-large 则约 2.24GB。挂 /app/data 卷缓存）。
#
# 🔴 凭证纪律（ADR-0027 §2.1）：本镜像只带 config/*.example 与 run_*.sh。
# 真实 config/.env、live routing.toml/identity.toml 一律走 volume / env_file
# 注入（compose 已挂 ../config），绝不 COPY——仓库根 .dockerignore 是第二道闸。
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 系统依赖：Redis（P1 可选本地实例）、构建工具（rocksdict/lancedb 编译）。
RUN apt-get update && apt-get install -y --no-install-recommends \
        redis-server build-essential gcc git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv（与本地一致的工作流）
RUN pip install --no-cache-dir uv

WORKDIR /app

# 先装依赖（利用层缓存：pyproject 改动少时跳过）
COPY pyproject.toml ./
COPY packages/bladex-core packages/bladex-core
COPY packages/bladex-proxy packages/bladex-proxy
RUN uv pip install --system -e packages/bladex-proxy -e packages/bladex-core

# 应用代码 + 脚本。配置只带模板与薄壳脚本（凭证不进镜像层，见文件头）。
COPY scripts ./scripts
COPY config/*.example ./config/
COPY config/run_proxy.sh config/run_consolidator.sh ./config/

# 数据卷（P1 overflow / P3 RocksDB / P2 LanceDB+meta / e5 缓存）
RUN mkdir -p data
VOLUME ["/app/data"]

EXPOSE 38080

# 默认起 proxy；consolidator 服务在 compose 里 override command。
CMD ["python", "-m", "bladex_proxy.server"]
