# BladeX 自部署指南（ADR-0021 §4 单机企业就绪）

> 形态：v1 = 单机，水平扩展与 HA 显式不做（类 GitLab omnibus，先把单机做好）。
> 并行赛道，不改主线。

## 1. 最小起服（Docker Compose）

```bash
cp config/.env.example config/.env          # 填上游模型 + API key
cp config/routing.toml.example config/routing.toml   # 路由模型池
cp config/identity.toml.example config/identity.toml # 可选：身份两体系（不做则个人模式）
# 编辑 .env：BLADEX_UPSTREAM_API_KEY / BLADEX_UPSTREAM_MODEL / BLADEX_ROUTE_ENABLED=true

cd deploy && docker compose up -d           # 起 proxy + redis(AOF) + consolidator
curl http://127.0.0.1:38080/health          # 冒烟
```

三服务：`proxy`（FastAPI 热路径）+ `redis`（P1 接入缓冲，AOF 防掉电）+ `consolidator`（P2 后台提炼，独立进程）。数据卷挂 `/app/data`（P3 RocksDB / P2 LanceDB+meta / P1 overflow / e5 缓存）。

一条请求走通：注入（`<bladex-memory>`）-> 路由（routing.toml 流水线）-> P3 落库（Redis Streams -> RocksDB）。consolidator 60s 一轮把 P3 蒸馏进 P2。

## 2. 绑定地址与安全（BLADEX_HOST）

- **默认 `BLADEX_HOST=127.0.0.1`**（仅本地）--生产暴露请用反代，不要直接 `0.0.0.0` 裸听。
- compose 里 `BLADEX_HOST=0.0.0.0` 是**容器内**监听所有接口；外部可达性由 `ports: 127.0.0.1:38080:38080`（仅宿主本地）+ 反代控制。**不要**把 38080 直接映射到 `0.0.0.0:38080` 公网裸 HTTP。
- `BLADEX_AUTH_ENABLED=true` + `BLADEX_CLIENT_KEYS` 开启客户端 key 校验（恒定时间比较）。企业部署**必须**开。
- **启动安全警告（T8）**：若绑非回环地址（非 `127.0.0.1`/`localhost`/`::1`）且 `BLADEX_AUTH_ENABLED=false`，启动会打横幅 + `bind_non_loopback_no_auth` warning 日志。这是"任何能到达该地址者都可无鉴权调用"的高风险配置，按提示修正（绑回环 / 开 auth / 上反代）。

### 2.1 端点暴露口径

| 端点 | 鉴权 | 暴露内容 |
|---|---|---|
| `/v1/chat/completions`、`/v1/messages`、`/v1/models`、`/v1/models/{model}` | `BLADEX_AUTH_ENABLED=true` 时验 client key；`false` 时放行。`/v1/chat/completions` 读 `Authorization: Bearer`；`/v1/messages` 兼容 `x-api-key`（Claude Code 默认发此头）与 `Authorization: Bearer` | 上游模型调用入口（转发 + 记忆注入）；`/v1/models{,/{model}}` 按 `anthropic-version` header 分流 OpenAI/Anthropic 形态 |
| `/v1/embeddings` | 同上（`Authorization: Bearer`） | 本地 e5 嵌入（exposure=local，输入不出境；`model` 字段恒走本地 e5）；与 P2 fact 向量同源 |
| `/v1/messages/count_tokens` | 同 `/v1/messages`（`x-api-key` / `Bearer`） | 本地近似 input_tokens 计数（Claude Code 每轮请求前调用；不引重依赖，标注 approximate） |
| `/admin/*` | 同 client key 体系（bearer，`require_admin_key` 复用 `auth_check`）；`false` 时**同样放行** | 记忆管理写操作（turn 墓碑 / Matter assign/merge/detach/split/close/rename / scope promote）；`/admin/status` 全景只读（T9，待实现） |
| `/metrics` | **无鉴权** | Prometheus 文本指标（请求计数/延迟 by agent·sensitivity、路由 source 分布、P1 水位、注入过滤命中） |
| `/health` | **无鉴权** | liveness 存活探针 + P1 溢出/spill 计数 |
| `/ready` | **无鉴权** | readiness 探针--仅布尔就绪位（`ready` + `checks:{redis,p3,p2,upstream}`），不泄运行数据；`ready=redis∧p3`，否则 503。供 compose/k8s 探针与 LB 用 |

- `/admin/*` **有独立 key**：`BLADEX_ADMIN_KEYS`（格式同 `BLADEX_CLIENT_KEYS` 的 `key||label`；`bladex init` 会同时生成两把）。三级语义（ADR-0027 §2.2）：
  - 配了 → `/admin/*` 只认 admin key，数据面 key 一律 401。
  - 未配 + 个人模式（无 `identity.toml`）→ 回落共用 client key（与旧行为逐字一致），启动打 `admin_key_shared_with_data_plane` warning。
  - 未配 + 存在 `identity.toml`（多 principal）→ **拒绝启动**：共用 key 意味着任一员工的 agent key 拥有全库管理权（删 turn / 改写合并 Matter / 提升 scope），可见性隔离拦得住读、拦不住写。
- `BLADEX_AUTH_ENABLED=false`（默认）时 `/admin/*` 与 `/v1/*` 一并开放。**这是绑非回环 + 默认配置下最危险的暴露面**，T8 启动横幅会强警告。
- `/metrics`、`/health`、`/ready` 不带鉴权，便于反代/监控探针直连。**绑非回环时**这些端点会向任何能到达者暴露运行数据（`/metrics` 最多：请求量、路由分布、队列水位；`/health` 次之：溢出/spill 计数；`/ready` 最少：仅布尔位）。公网部署务必：反代 + 网络层 ACL 限制 `/metrics`、`/health`、`/ready` 仅内网/监控网可达，或在反代层加 basic-auth 网关保护。
- `/health`（liveness：进程活即 200）与 `/ready`（readiness：子系统就绪才 200）分层--`/ready` 探 Redis 连通 / P3 / P2 / 上游熔断缓存标记；P2 与上游为 informational 不 gate（fresh install 或单模型熔断仍有 failover 时不致 503），`ready = redis ∧ p3`（写路径必需）。
- `/admin/*` 持 client key，但 key 一旦泄漏则记忆可被改写/删除。`.env` 里设强随机 `BLADEX_CLIENT_KEYS`，勿入仓（`.env` 已 gitignore）。
- e5 嵌入推理**全本地**（fastembed，无外呼）；`/v1/embeddings` 与 P2 内部 e5 同源、输入不出境；上游模型调用才出境--数据流向见 README「数据流向诚实义务」。

#### base_url 怎么填（两协议不同，别照抄接口路径）

上表列的是 proxy 暴露的**接口路径**，不是 agent 该填的 `base_url`。SDK 在 `base_url` 后自动拼接接口路径，故 `base_url` 填根、留接口路径给 SDK 拼：

| 协议 | agent 端 `base_url` | SDK 自拼的接口 |
|---|---|---|
| OpenAI（Hermes、Codex 等） | `http://<host>:<port>/v1` | `/chat/completions`、`/models` |
| Anthropic（Claude Code） | `http://<host>:<port>`（**根**，不带 `/v1/messages`） | `/v1/messages`、`/v1/messages/count_tokens`、`/v1/models` |

常见错误：把 Anthropic 的 `base_url` 填成 `http://<host>:<port>/v1/messages`（误把接口路径当 base_url），SDK 再拼一次变成 `/v1/messages/v1/messages` -> 404，列模型即失败、消息发不出。**Anthropic 协议填根即可**，`/v1/messages` 由 SDK 自动调用。


## 3. TLS 反向代理（公网暴露必前置）

BladeX 自身只起 HTTP。TLS 在反代层终止。下面给 nginx / caddy 两例。

### nginx

```nginx
server {
    listen 443 ssl http2;
    server_name bladex.example.com;
    ssl_certificate     /etc/ssl/bladex.crt;
    ssl_certificate_key /etc/ssl/bladex.key;

    # 流式响应必须关缓冲，否则 SSE 分块卡到上游结束才下发
    proxy_buffering off;
    proxy_read_timeout 600s;

    location / {
        proxy_pass http://127.0.0.1:38080;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        # 透传 agent 身份 header（可选，agent 也可自己在请求里带）
        proxy_pass_request_headers on;
    }
}
```

### caddy（自动 TLS）

```caddy
bladex.example.com {
    reverse_proxy 127.0.0.1:38080 {
        flush_interval -1   # 流式：立即下发 SSE
    }
}
```

## 4. 身份两体系（identity.toml）与敏感度路由

- **个人模式**：不配 `identity.toml` = 现状（`user_id = key hash8`）。配两 key 一 principal 可修复多设备记忆分裂（见 `config/identity.toml.example` 示例一）。
- **企业**：配 `[[orgs]]`/`[[teams]]` + principal 挂载（示例二）。凭证独立轮换、组织独立调整。
- **敏感度**：`routing.toml` 加 `[strategies.sensitivity]`（enabled + levels 映射）+ `[[models]]` 标 `exposure`（local=自建 vLLM / private=租赁裸卡 / public=公网 API）。identity.toml 的 principal/team 可带 `sensitivity`。敏感流量自动只走 `exposure <= allowed` 的模型池，裁判钉本地（绝不外发敏感 query）。
- 启动校验：敏感层开 + 裁判非本地 + `BLADEX_ROUTE_STRICT=true` -> 拒启动。

## 5. 备份与恢复

```bash
bash scripts/backup.sh              # 在线 RocksDB checkpoint + LanceDB + Redis AOF -> data/backup_<ts>/
bash scripts/restore.sh <备份目录>   # 恢复后启动 consolidator 自动从 P3 重建 P2
```

- RocksDB checkpoint 在线、不停服（read_only 句柄 + Checkpoint API，原生一致快照）。
- 恢复后 P2 由 consolidator 从 P3 重建（P2 坏了能从 P3 重建，ADR-0009）。
- 定期备份建议 cron：`0 3 * * * cd /app && bash scripts/backup.sh /backups/$(date +\%F)`。

### 5.1 磁盘增长预期与归档（ADR-0027 §4.4）

P3 是完整总账、**设计上无限增长**（ADR-0009：它是唯一真相源，P2 可从它重建）。所以磁盘
是需要主动管理的资源，不是"够用就行"：

| 目录 | 增长来源 | 处置 |
|---|---|---|
| `data/bladex_ledger`（P3） | 每轮对话一条追加记录（含 request_messages / 响应元数据） | 只增不减；容量吃紧时整段归档到冷盘（**不要删**——删了 P2 就不可重建） |
| `data/bladex_index`（P2） | 派生 fact/matter/向量 + 蒸馏台账 | 可删可重建（`bladex storage rebuild`），但重建要花 LLM 钱；台账在这里，删了台账重建成本会上去 |
| `data/overflow` | 只在 Redis 不可达时产生的溢出文件 | 正常应为空；非空说明曾经掉过 P1，回灌后自动清 |
| `logs/` | 每次启动一个时间戳文件 | `bladex start` 按 `BLADEX_LOG_KEEP`（默认保留最近 20 个）清理 |

查看当前占用：`bladex status`（P3 turn 数 / P2 fact 数）与 dashboard 状态页。
把整段历史挪走的做法参考 ADR-0024 §5.1 的归档流程——**三层要同时处理**（P3/P2/P1），
只挪 P3 会让 `data/overflow` 与 Redis AOF 里的残留在下次启动时被回灌。

## 6. 数据流向（诚实义务）

BladeX 让**算力节点无状态化**（留存主权）：租赁裸卡只跑推理、可即插即拔，全部会话记录与记忆资产留在企业自己控制的 BladeX 节点上。但以下边界必须明确：

1. **推理时 prompt 仍到达算力节点的显存/内存**--这是推理的物理前提，任何中间件都无法消除。BladeX 解决**留存**风险（算力节点关机即无痕），不解决**传输/驻留**暴露。对外表述统一用"算力节点无状态化/留存主权"，不用"数据不出企业"。
2. **公网 API 上游**：发往 `exposure=public` 模型的请求（含注入的记忆 Fact）会到达云厂商。敏感数据应配 `exposure=local/private` 模型 + 敏感标签，敏感 Fact 经血统继承不会注入公网请求（ADR-0021 §3.3c）。
3. **雇主留存员工会话**的合规与知情责任在部署方--代码提供 scope/可见性原语（personal/team/org），但"雇主留存并共享员工会话"的治理由部署方决定。默认边界：personal scope 仅本人可见；team/org 共享需显式 scope 提升操作（记管理事件，可审计）。

## 7. 不在 v1 范围

SaaS 多租户/计费、SSO/OIDC（反代层自解）、水平扩展/HA、应用层静态加密（维持 ADR-0009 决定，服务器场景用 LUKS/FileVault 全盘加密）。详见 ADR-0021 §2.6/§4。
