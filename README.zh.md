<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/bladex-lockup-horizontal-dark.svg">
    <img src="assets/brand/bladex-lockup-horizontal.svg" alt="BladeX" width="340">
  </picture>
</p>

> **Build your private data assets.**
> **Connect all your agents and LLMs like a blade.**
>
> 把你的数据变成自己的资产；像一把刀刃一样，薄薄地连起你所有的 Agent 与大模型。

BladeX 是一个**自部署、以记忆为核心的代理**，工作在你的 AI Agent 与模型供应商之间。把任意 Agent 的
API 地址指向 BladeX，它就获得跨会话、**跨 Agent** 的记忆，以及一份按任务维护的账本——**不需要修改
Agent 的任何代码**。所有数据都留在你自己的机器上：记忆是你拥有的私有数据资产，不是租来的功能。

English version: [README.md](README.md)

## 4S

| | 含义 |
|---|---|
| **Save all Memory** | 每一轮**无损**进 Memory Hub——正文与元数据同等对待，唯一真相源。不做静默截断，也不靠删除腾地方：容量紧张时走信任衰减与归档。 |
| **Saving your Token** | 注入面不再是「把记忆全倒一遍」。每轮只带硬规则、一段稳定的自我说明和当前任务账本，其余由模型经 `bladex_memory_search` **按需取**。而账本活过上下文压缩之后，省掉的是反复解释的**轮次**——真正的开销在那里。 |
| **Smart LLM Router** | 一个入口，多个模型。配置驱动的确定性路由（显式指定的模型 → team → principal → agent）、可选 LLM 裁判、会话粘性、按健康度 failover。**配置里的静态策略压过一切运行时推断。** |
| **Sift out experience** | 原始对话不是记忆。后台蒸馏把它变成结构化事实，归并进 Matter（一件跨会话、跨 Agent 的「事儿」），模型则把真正确立下来的结论写进账本的 `Verified` 段。 |

## 它解决了什么问题？

Agent 在跨会话时会忘事，内置记忆普遍有四个老毛病：

1. **Token 膨胀** —— 每轮把记忆全倒一遍，上下文越长越贵越慢。
2. **记忆丢失** —— 上下文压缩丢掉旧事实：「你不记得上周聊了什么？」
3. **矛盾冲突** —— 不同会话的信息互相打架，没有统一的事实来源。
4. **强制删除** —— 上下文满了就把旧事实删掉。

还有一个只在你同时用多个 Agent 时才浮现的问题：**它们之间什么都传不过去。** 你在一个编码 Agent 里
确立的东西，下一个 Agent 完全不知道。**跨 Agent 交接正是 BladeX 要解决的那件事**，也是记忆必须住在
代理层、而不是住在某一个 Agent 里的原因。

## 它怎么工作：账本 + 记忆 + 代理

### 1. 账本 —— 让模型的注意力落在任务上

长任务会漂。上下文被压缩、会话重启、你从一个 Agent 换到另一个——每一次模型都丢线索，你都得重讲一遍。
一旦是多个 Agent 配多个模型，这笔税就叠起来：每一个都要知道目标、约束、以及哪些事已经定了，
而每一个都要你**写进它自己的配置文件、按它自己的格式、遵它自己的约定**。同一件任务被描述四遍、
四套标准，从写下的那一刻起就开始各自漂移。

BladeX 的做法是：**一份任务状态、一种格式，放在代理层**。它后面的每个 Agent、每个模型读的是同一本账本——
不需要逐 Agent 配置，不需要逐模型适配，也没有第二份要同步：

| 段 | 放什么 |
|---|---|
| **Goal** | 这件任务是干什么的。**只有用户能改**——见下面的三红线。 |
| **Core** | 整个任务都成立的约束与前提。 |
| **Verified** | 已经确立的结论，最好有工具或测试结果佐证。 |
| **Open** | 已知的未知。 |
| **Next** | 约定的下一步。 |

它换来什么：

- **注意力落在没定的那部分。** 五段天然是短的，而块追加在消息数组**真正的末尾**——长工具循环里
  模型实际读的位置。它一上来就知道这件任务是什么、哪些已经定了，注意力就花在还没定的部分。
- **聚焦但不失真。** 账本不是对话摘要——摘要是有损的、会漂。它是模型**在确立结论的当下**写下的
  结构化记录，所以上下文可以被压缩掉，任务状态不会跟着掉。集中与保真在这里不是一个换一个。
- **先对，再省。** `Verified` 只放真正确立的东西、最好有工具或测试佐证——这才是长任务不会悄悄
  建在一个猜测上的原因。而省下来的是**反复重建上下文的那些轮次**，比从注入面上抠掉几段正文值钱得多。
- **交接不要钱。** 一件任务一本，一个会话绑一本激活的，两者都与「现在是哪个 Agent 在开」无关。
  在一个 Agent 里干完一段，换一个接着干，状态已经在那儿了。
- **「切回去」是个真选项。** 模型考虑换任务时，BladeX 同时给它两段候选——**最近在干什么**（recency）
  与**看起来是同一件事**（相关性）。于是回到三周前那件任务是它可以选的，而不是默默地为一件
  已经有四本账本的事再开第五本。

### 2. 记忆 —— 四层，一个方向

写入方向恒为 **Pipeline → Hub → Index → Flash**；Hub 下游的每一层都能从 Hub 重建。

| 层 | 后端 | 职责 |
|---|---|---|
| **Pipeline** | Redis（AOF） | 接入缓冲。落库成功才 ack；Redis 掉线走磁盘兜底，轮次不丢。 |
| **Memory Hub** | RocksDB | 完整总账，**唯一真相源**，长期留存。其它层都能从它重建。 |
| **Memory Index** | LanceDB + FTS | 派生层：蒸馏事实、向量、Matter 关联网络。可从 Hub 增量重建。 |
| **Memory Flash** | 文件 | 当前状态的文件投影，`User → Agent → Project → Session`。你自己也读得懂，不只给代理读。 |

### 3. 代理 —— BladeX 对模型是**可见**的

纯被动的向量召回有天花板：做决定的那一方根本不知道有东西可查。所以 BladeX 会自我介绍
（`config/system/AGENT.md`、`TOOLS.md`、`SKILLS.md`），并给出一小组模型可以主动调用的工具：

- `bladex_memory_search` —— 按需取历史事实（取记忆的唯一通道，不再硬塞）
- `bladex_ledger_read` / `bladex_ledger_update` / `bladex_ledger_switch` —— 读、记、切换任务账本

这些调用由 BladeX 拦截，不会当成 Agent 的工作打到你的上游账单上：**纯 bladex 调用**的轮次走内循环，
**混合调用**的轮次被剥离、处理、再拼接回去。每一次都能在 `bladex status --traffic` 里看到。

### 三红线

1. **BladeX 的工具只读写记忆与账本对象**：不改文件、不执行命令、不联网。工具面是「文具」不是「秘书」。
2. **只有用户能改 Goal**：模型能改自己的目标，就等于自我拆台。
3. **模型不调用时零行为差异**：把工具面整个撤掉，系统退回纯被动注入形态。这条同时是测量通道——
   工具面到底值不值，可以直接 A/B 量出来。

## 和其他方案相比

| | BladeX | Agent 内置记忆 | mem0 |
|---|---|---|---|
| **无需修改 Agent 代码** | ✅ 只改一个 base URL | — | 需要 Agent 侧集成 |
| **跨不同 Agent 生效** | ✅ 一份记忆放在所有 Agent 背后 | ❌ 各自为政 | 取决于集成方式 |
| **有任务状态，不只有事实** | ✅ 五段账本 | ❌ | ❌ |
| **数据完全由你掌控** | ✅ 100% 在你自己的基础设施 | 取决于 Agent | 默认云端 |
| **记忆驱动的模型路由** | ✅ 可选 | ❌ | ❌ |

## 快速开始（v0.1.0）

### 环境要求

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) 包管理器
- Redis 服务（已安装但未运行时，`bladex start` 会自动拉起）

**不需要**系统级 RocksDB —— 存储层用的是 `rocksdict` 预编译 wheel。

### 安装运行

```bash
git clone https://github.com/YOUR-ORG/bladex.git
cd bladex
uv sync

# 把 `bladex` 命令放进 PATH。`uv sync` 只是把它装进 .venv、不会自动激活，
# 少了这步在新终端里会直接 "command not found"。
source .venv/bin/activate          # Windows: .venv\Scripts\activate
# 不想激活？每条命令前面加 `uv run`，例如 `uv run bladex init`。

# 交互式初始化：写 config/.env（chmod 600）+ config/routing.toml，
# 生成客户端 key 与 admin key，并默认开启鉴权。
bladex init

# 按提示填上游 API key（也可以事后改 config/.env）。

bladex doctor          # 环境体检
bladex start           # Redis + 提炼进程 + proxy
bladex doctor --e2e    # 端到端：记忆到底有没有被提炼出来、能不能被召回
```

`bladex start` 会打印 dashboard 地址；如果记忆管线有任何一环没起来，它会打印 `degraded` 并**写明后果**
（例如「Redis 不可达 → 对话不会入库，不会产生任何记忆」）。绿色的 `ready` 表示闭环成立，
`bladex doctor --e2e` 会发一条探针对话真正证明它。

> **首次启动会先下载 embedding 模型**（约 70 MB，来自 Hugging Face），下完 proxy 才应答 `/ready`。
> 网络慢时会超过 30 秒的就绪窗口，`bladex start` 可能报 `failed -- /ready did not answer within 30s`，
> 但进程其实是好的：看 `logs/proxy-*.log` 里 `embed_model_download_start … server_ready`，再跑
> `bladex doctor` 即可；第二次启动是秒起。（已登记 MQ-A42，0.2.0 让等待窗口感知下载。）

### 配置你的 Agent

在 Agent 设置里，把 **API 基础地址**从默认（例如 `https://api.openai.com/v1`）改成：

```
http://localhost:38080/v1
```

把 **BladeX 客户端 key**（`bladex init` 打印，存在 `config/.env`）填到「API 密钥」位置。

**原生支持的接入格式**：

- OpenAI `/v1/chat/completions` —— 兼容所有 OpenAI 格式客户端
- Anthropic `/v1/messages` —— 原生兼容 Claude Code
- OpenAI Responses `/v1/responses` —— 原生兼容 Codex CLI

三步接入对照表：

| Agent | 改什么 | 怎么确认接上了 |
|---|---|---|
| Hermes | `base_url` → `http://localhost:38080/v1`，API key → BladeX 客户端 key | `bladex status --traffic` 出现 `agent=hermes:*` |
| Claude Code | `export ANTHROPIC_BASE_URL=http://localhost:38080` 与 `ANTHROPIC_API_KEY=<bladex client key>` | `bladex status --traffic` 出现 `agent=claude-code` |
| Codex CLI | OpenAI base URL 指向 `http://localhost:38080/v1`（Responses API） | `bladex status --traffic` 出现 `agent=codex` |

每个 agent 的详版接入页（含已知行为与排查表）：
[Hermes](docs/agent-hermes.md) · [Claude Code](docs/agent-claude-code.md) · [Codex CLI](docs/agent-codex.md)

除了那个地址，Agent 侧不需要做任何改动。**BladeX 去适配每个 Agent 的行为**——不要求你加 header、
不要求你改 system prompt、不要求你改请求格式。

### 它到底在不在工作？

```bash
bladex status --traffic     # 最近每轮一行：agent / 注入了什么 / 路由 / 是否入库
bladex doctor --e2e         # 完整闭环：转发 → 入库 → 提炼 → 召回
bladex ledger list          # 任务账本，最近的在前
bladex ledger show <id>     # 某一本的五段
bladex memory search "..."  # BladeX 到底记住了什么
```

`bladex status` 顶部还会在记忆**安静降级**时给出黄色横幅——注入超时、有轮次没能入库、
或提炼进程落后太多。看不见的失效等于没有记忆，所以这些不只写在日志里。

### 万一 BladeX 出问题

BladeX 挡在你所有 Agent 流量前面，所以**先知道退路**：把 Agent 的 base URL 改回原来的地址。
这就是全部的恢复步骤——Agent 立刻恢复正常，`data/` 里的东西一点不丢，重启 BladeX 后接着用。

```bash
bladex stop      # 停 proxy 与 consolidator（Redis 保持运行）
bladex start     # 起回来
bladex consolidator restart   # 只重启提炼进程（改配置后最常用，不打断正在跑的会话）
```

### 额外的 LLM 调用与成本

BladeX 会在你的上游账户里产生 Agent 之外的调用，请一并预算：

| 调用 | 何时发生 | 怎么控制 |
|---|---|---|
| 事实蒸馏 | 后台，每条入库的 turn（在 consolidator 里，不在热路径） | `routing.toml` 的 `[distill].model` 指一个便宜模型 |
| 结论 / 产出蒸馏 | 后台，产生结果的轮次 | 同上 |
| 工具面内循环 | 模型调了 `bladex_*` 工具、BladeX 需要把这一轮续下去时 | 关掉工具面（`BLADEX_MODULE_TOOLFACE=0`） |
| LLM 路由裁判 | 只在启用 `[strategies.filter]` 时 | 不开路由，或收窄候选池 |

蒸馏跑在请求路径之外，不会拖慢 Agent，**但会体现在账单上**。被识别为内部调用（auxiliary）的请求
会自动走便宜档。注入与检索本身**不产生**任何 LLM 调用。

**关于「Saving your Token」这句话**：我们公布机制，不公布百分比。目前实测到的那个数（约省 87%）
是**记忆注入 token 口径**——端到端请求 token 的基线还在测；在那个数出来之前，
BladeX 不会对外说「端到端省了多少」。

### 小上下文窗口

如果你跑的是小窗口本地模型（32K 及以下），要盯住注入面：账本块与系统说明是实打实的 token，
在小窗口上会挤压输出预算。`bladex status --traffic` 与日志里的 `upstream_output_truncated`
告警会告诉你什么时候发生了。窗口感知路由与按预算裁剪排在 0.2.0。

### 非英语用户请注意

默认 embedding 模型是 `BAAI/bge-small-en-v1.5`（英语优先，0.07 GB）。如果你的对话主要是其它语言，
请在**积累记忆之前**改成多语模型——事后再换需要全量重新嵌入：

```toml
# config/routing.toml
[embedding]
model = "intfloat/multilingual-e5-large"   # 2.24 GB，BladeX 的阈值就是在这个模型上标定的
```

## 我的数据存在哪里？

全部本地存储在你机器的 `data/`：

- **Pipeline**：Redis AOF 持久化
- **Memory Hub**：RocksDB —— 完整对话总账，唯一真相源
- **Memory Index**：LanceDB —— 向量、蒸馏事实、Matter 关联网络（可从 Hub 重建）
- **Memory Flash**：`data/bladex_flash` 下的普通文件 —— 当前状态投影为
  `User → Agent → Project → Session`。想让它出现在你顺手能打开的地方（比如笔记库），
  改 `BLADEX_FLASH_PATH`。

BladeX **绝不会把你的记忆发送给第三方服务器**。唯一的出站请求是发往你自己配置的 LLM API 供应商的
模型推理请求。静态加密交给磁盘（FileVault / LUKS），应用层不另做一套。

## 功能特性

### 核心（v0.1.0）

- 🧾 **五段任务账本** —— Goal / Core / Verified / Open / Next；活过压缩、重启与跨 Agent 交接，
  Goal 归用户所有。
- 🧠 **记忆质量就是产品** —— 相关性优先于数量、压缩之前就完成增量提取、写入时做矛盾校验、
  容量满走信任衰减与归档而不是删除。
- 🛠️ **工具面** —— `bladex_memory_search` 加三个账本工具，边界严格（只碰记忆与账本对象），
  不调用时零行为差异。
- 🔌 **零改造接入** —— 任何说 OpenAI Chat Completions / Anthropic Messages / OpenAI Responses 的 Agent。
- 🚦 **Router** —— 配置驱动的确定性路由 + 可选 LLM 裁判 + 会话粘性 + 按健康度 failover；
  静态配置压过一切运行时推断。
- 🔒 **隐私优先** —— 全数据自托管，完全由你掌控。
- 👥 **多 Agent、多用户** —— 多 API key，可选的团队 / 身份隔离（企业自部署）。
- 🖥️ **Web 管理面板** `http://localhost:38080/dashboard` —— 状态、事实检索、Matter 关联图、
  账本、会话回放、硬规则。
- 🔌 **MCP Server**（`bladex-mcp`）—— 给说 MCP 的客户端的第二个读写面。
- 📤 **导出 / 导入** —— 带版本的 JSONL 快照，以及到 Obsidian 与 PostgreSQL 的持续单向同步。

### 下一步（0.2.0）

- 上下文窗口感知的路由 + 按预算裁剪注入面
- 已完结账本的 `close` / `supersede` 原语
- 与外部工具的双向同步
- 硬规则热加载（当前写在 `config/.env`，改完要重启）

## 已知限制（v0.1.0）

- 当前面向**个人自部署**优化 —— 企业多租户能力仍在开发中。
- 账本工具面默认只对 `medium` / `strong` 档模型开（`BLADEX_LEDGER_TIERS`），或者本会话已经绑着
  激活账本时开。弱档模型仍**看得见**账本，只是不被要求去维护它。
- 小窗口本地模型上，注入面会与输出预算竞争（见上文「小上下文窗口」）；按窗口裁剪排在 0.2.0。
- LanceDB 的 ANN 索引在表超过 256 行后才建；低于这个规模是暴力扫描（该规模下很快，
  但确实意味着小库不走索引）。
- 目前只原生支持 OpenAI Chat Completions、Anthropic Messages、OpenAI Responses 三种端点，
  其它格式按需增加。
- **导出到外部的记忆可能是旧的**：到 Obsidian / PostgreSQL 的持续同步会推送新增与更新的事实，
  但 Matter 卡的 `summary` / `open_issues` **没有**像单条事实那样按敏感度过滤。
  同步到共享位置前请先自查。
- 硬规则在 `config/.env` 里编辑（`||` 分隔）且需要重启 proxy；dashboard 只读展示。
- CLI 按当前目录（或 `BLADEX_HOME`）解析 `config/`、`data/`、`logs/`。在无关目录下跑
  `bladex status` 会看到一个空库——**数据没丢，只是你指错了地方**。

## 许可证

Apache 2.0 —— 详见 [LICENSE](LICENSE)。

## 致谢

- 记忆质量设计参考了对 [Hermes](https://hermes-agent.nousresearch.com/) 记忆架构的分析
- 多模型路由使用 [LiteLLM](https://www.litellm.ai/) SDK 实现多供应商兼容
- 基于 FastAPI、Redis、RocksDB、LanceDB 构建
