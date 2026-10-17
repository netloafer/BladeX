# Release process

How a BladeX version gets from a commit to a git tag, a GitHub Release and (eventually)
PyPI — and which step has a gate that stops a wrong release.

This page exists because the first two releases (0.1.0, 0.2.0) shipped with **no** gate
between "tag pushed" and "wheel uploaded": 0.2.0 was tagged on a tree whose
`pyproject.toml` files still said `0.1.0`, and the only reason a wrong package was not
uploaded was that the PyPI publisher had not been configured yet. PyPI version numbers
cannot be overwritten or reused, so that upload would have been irreversible.

## 1. Before tagging

| Step | Command | Gate |
|---|---|---|
| Bump the version in **all four** `pyproject.toml` files (root + `packages/*`) | edit `version = "X.Y.Z"` | `test_release_readiness.py::test_pyproject_versions_are_identical` — any one of the four drifting turns red |
| Re-sync the editable install so `bladex --version` reports the new number | `uv sync` | `test_dunder_version_derives_from_pyproject` — stale dist metadata turns red |
| Update the version labels in `README.md` / `README.zh.md` (`(vX.Y.Z)` headings) | edit | `test_readme_version_labels_match_package_version` |
| Lint + tests | `ruff check .` then `pytest -q` (private tree: `bash scripts/gate_check.sh`, which runs both) | CI runs the same two commands on every push to `main` |
| Dry-run the tag gate locally | `python scripts/check_release_version.py --tag vX.Y.Z` | exit 0 = the tag you are about to push matches the package version (incl. the four workspace entries inside `uv.lock` — batch P) |
| Check the **public repo** CI is green | `gh run list --workflow ci.yml --limit 1` (in the public repo checkout) | the latest run must say `success` — **do not tag if it is not `success`**. CI was red for 11 straight days after 0.1.0 and nobody noticed (MQ-V37); the README badge makes red visible in passing, this line makes it a release blocker |

`__version__` in `bladex_proxy` / `bladex_mcp` is **derived** from `pyproject.toml`
(installed metadata first, the adjacent `pyproject.toml` when running from a source tree).
Do not add a second literal.

## 2. Tag push → `release.yml`

```
git tag -a vX.Y.Z -m "..." && git push origin vX.Y.Z
```

The workflow runs, in order:

1. **`Check tag == package version`** — `scripts/check_release_version.py --tag <tag>`.
   The tag minus its leading `v` must equal, byte for byte, the version in all four
   `pyproject.toml` files. Mismatch ⇒ the job fails **before** anything is built.
2. `uv build --all-packages`.
3. **`Publish to PyPI`** — runs **only** when the repository variable
   `BLADEX_PYPI_PUBLISH` is exactly `true`. Otherwise the step is skipped with a notice.
   Publishing is irreversible, so it is opt-in, never the default.
4. `GitHub Release (draft)` with the built distributions attached. Publish the draft by hand.

Until the trusted publisher below is configured, leave `BLADEX_PYPI_PUBLISH` unset.

## 3. Configuring PyPI trusted publishing (one-time, done on pypi.org)

BladeX publishes three distributions: `bladex-core`, `bladex-proxy`, `bladex-mcp`.
Each needs its own publisher entry. No API token is involved; PyPI verifies the GitHub
OIDC token minted by the workflow. The claims it checks (as seen in the failed 0.2.0 run):

| PyPI field | Value |
|---|---|
| Owner | `netloafer` |
| Repository name | `BladeX` |
| Workflow name | `release.yml` |
| Environment name | *(leave blank — the job does not use a GitHub environment)* |

Steps:

1. Sign in to pypi.org → **Your account → Publishing** (`https://pypi.org/manage/account/publishing/`).
2. For a distribution that does **not** exist on PyPI yet, add a **pending publisher** with the
   project name (`bladex-core`, then `bladex-proxy`, then `bladex-mcp`) and the four fields above.
   The first successful upload creates the project and converts the pending publisher into a
   normal one. For a project that already exists, do the same under
   *Project → Manage → Publishing*.
3. Repeat on **test.pypi.org** for the rehearsal below (separate account, separate publishers).
4. Set the repository variable: GitHub → Settings → Secrets and variables → Actions →
   **Variables** → `BLADEX_PYPI_PUBLISH` = `true`. Do this only after the rehearsal passes.

## 4. TestPyPI rehearsal (do this before the first real upload)

The workflow has a manual trigger for exactly this:

1. Push the release tag as usual (the tag-push run will build and skip publishing as long as
   `BLADEX_PYPI_PUBLISH` is not `true`).
2. GitHub → Actions → **Release** → *Run workflow* → choose the **tag** `vX.Y.Z` as the ref,
   `target = testpypi`, run.
3. The run must pass the version gate, build, and upload to `https://test.pypi.org/`.
   Verify with `pip install --index-url https://test.pypi.org/simple/ --no-deps bladex-proxy==X.Y.Z`.
4. A TestPyPI version, like a PyPI one, is consumed forever — if the rehearsal must be repeated,
   use a post-release tag (`vX.Y.Z.post1` with matching `pyproject.toml` versions), not the same one.
5. Only after the rehearsal passes: configure the pypi.org publishers (§3) and set
   `BLADEX_PYPI_PUBLISH=true`. The next tag push publishes for real; `target = pypi` on the
   manual trigger does the same for a tag that was pushed while the variable was unset.

## 5. 待写进发版说明的 behavior changes（长期收集点）

**每批改了用户可见行为就往这里加一行，发版棒统一整理成 CHANGELOG。**

### 5.1 0.3.0（✅ 已整理进定稿 `docs/release-notes-0.3.0.md`，2026-10-17 批 REL；本节只读）

CHANGELOG 不是发版那天写出来的，是一路攒的（MQ-V37 ⑤ 的真修法：仓库至今无 CHANGELOG，
而 0.2.0 已公开发布、用户可自行打开的开关下已经有行为变了）。一行 = 改了什么 + 谁受影响 + 取代它的是什么。

- 🔴 **上下文装配 L1 的公开默认改为「只观测不裁」：`BLADEX_ASSEMBLY_AGENTS` 默认值 `claude-code` → 空**（批 U · U0，MQ-CA19，2026-09-23）。
  **公开用户可见**：v0.2.0 的默认会对 Claude Code 每轮按位置把已闭合任务的大块工具结果换成占位摘录（L1）；MQ-CA10 实证它在防一个不存在的问题（上游窗口 1M、未裁前只占 10–15%），代价是重读循环让轮次翻倍，作者本机 09-09 起已关，但默认值当时没改。
  升级后 CC 用户的 agent 历史**不再被 L1 降解**、原样转发；`assembly_done` 日志照打（`skipped=agent_not_enabled`）。想要旧行为：`BLADEX_ASSEMBLY_AGENTS=claude-code`。取代者 = ADR-0033（批 U1 机制，标定后批 U3 再翻 `all`）。
- **codex 的会话分组改为读 codex 自报的 `session-id` header**（批 U · U1.1，MQ-A76，2026-09-23）。codex 每条请求都带裸 `session-id`（无 `x-` 前缀），BladeX 此前一条没读、退到指纹 / 尾部接续去猜 ⇒ Hub 存量 111 个自报会话被塞进 66 个会话桶（最坏一桶 13 个）。**受影响：codex 用户**——升级后 codex 的 Memory Hub key 与 `session_id` 从 `fp:<hash>` 变为 codex 的会话 UUID，`session_id_source=explicit`；按会话聚合的读数（`bladex status` / dashboard / `SESSIONS` 类统计）上 codex 的会话数会变多（离线回放：66 → 111 桶，混装桶 12 → 0）。优先级在 `X-Session-ID` 与 `x-{vendor}-session-id` 之后；`thread-id` 不收。
- **轮次口径统一为 ADR-0033 §2.2**（批 U · U1.1）：工具循环中的用户插话、连续 user 不再各算一轮（此前每条 user 消息算一轮）。**用户可见的两处**：① 账本切换防抖（`BLADEX_LEDGER_SWITCH_DEBOUNCE`）与陈旧提示（`BLADEX_LEDGER_STALE_TURNS`）的分母 `turn_index` 变小 ⇒ 两者都变宽松（插话不再推进计数）；② `Turn.logical_turn` / `TaskUnit` / `unit_key` 口径改变——**新旧 unit_key 不互认、存量不迁移**，2026-09-23 之前的 `turn_index` / `unit_key` 读数与之后的不可直接比较。codex 首轮的 unit_key 改取真人那条消息（此前取 codex 灌进来的 `# AGENTS.md instructions` / `<environment_context>`，同仓库跨会话逐字撞车，MQ-CA18）。
- **上下文装配换成 ADR-0033 机制**（批 U · U1.2–U1.4）：只在「会话轮次数 > `BLADEX_ASSEMBLY_MIN_UNITS` ∧ 上一轮上游 `cache_read_tokens == 0`」时对已闭合、带工具结果的轮次做关联复核，无关的把 tool 结果换成可见标记 `<bladex:evidence_trimmed tool=… chars=… unit=…>`（user 原文与结论保留、tool_call_id 配对不动、当前轮次永不裁），判定会话内复用、热轮原样重放。**默认不生效**（`BLADEX_ASSEMBLY_AGENTS` 默认空，见上一条 CA19）；自行把它设成非空的用户：L1 按位置降解与 E4.3 冷启整单元裁剪不再发生，改为上述行为。
  - **新旋钮**（全部待批 U2 标定）：`BLADEX_ASSEMBLY_MIN_UNITS`（默认 `10`）· `BLADEX_ASSEMBLY_REVIEW_EMBED`（`1`，一级 embedding 复核）· `BLADEX_ASSEMBLY_REVIEW_MODEL`（空 = 不开二级轻量模型复核）· `BLADEX_ASSEMBLY_SCORE_HI` / `_LO`（`0.82` / `0.72`）· `BLADEX_ASSEMBLY_COLD_BUDGET_MS`（`5000`，冷轮复核总预算）。
  - **删除的旋钮**（设了也不再读）：`BLADEX_ASSEMBLY_EVIDENCE_MIN_CHARS` · `BLADEX_ASSEMBLY_EVIDENCE_EXCERPT_CHARS` · `BLADEX_ASSEMBLY_KEEP_RECENT_TOOL_RESULTS` · `BLADEX_ASSEMBLY_ENABLED` · 旧名 `BLADEX_CAP_ENABLED`（装配总开关只剩模块开关 `BLADEX_MODULE_ASSEMBLY`）。
  - **Hub 记录**：`Turn.reconstruction` 新增 `verdicts`（逐单元判定：kept/trimmed、来源 embed/model/frozen、锚点）与 `review`（本轮复核摘要）；`degrade_plan` 的值从裸字符串（`excerpt` / `minimal`）变为 `{tier, unit, source}`，历史记录照常可读。`assembly_done` 日志对开了装配的 agent 每轮都打，新增 `triggered` / `reviewed` / `trimmed` / `turns` / `cold_reason` 字段。
- **L2 上下文摘要与 B 层 minimal 档已删除**（ADR-0033 取代；批 Q · Q5 MQ-CA9 清账，2026-09-18）。
  `BLADEX_ASSEMBLY_AGENTS=""`（默认 live 形态）的用户无影响；自行打开该开关的用户，这两项不再生效。
  连带 `Turn.decision_meta.cap` 不再写 `summary_hash` / `kept_recent_n` 两键（批 R · R0 MQ-CA14 退役；历史值仍可读）。
- **账本墓碑从墓碑时刻起生效，不再倒填到切入时刻**（MQ-I19，批 S · S2.1，2026-09-18）。影响所有用过 `DELETE /admin/ledgers/{id}` 的用户：下一次全量重建时，被墓碑账本存活期内的轮次仍锚到它的锚卡（此前散成内容卡 / 卡消失）。取代者 = ADR-0034 §1.2。
- **新增 `bladex ledger tombstone <id> [--with-memory]`**（拍板 0.3.0-(2)，批 S · S3）：默认只墓碑账本并提示"派生记忆保留 N 条"；`--with-memory` 对激活窗口内源轮次追加 TURN 墓碑（Hub 层，不物理删）。`DELETE /admin/ledgers/{id}` 响应多了 `derived_facts_retained` / `window_turns` / `hint` 字段。
- **裁决快路径开始对库内同键邻居生效**（MQ-I20，批 S · S2.1）：跨 pass 的同键**重述**折叠为 NOOP（旧条 strength+1）；库内同键**新值**交裁决器，不强制取代；裁决器判的 UPDATE/NOOP 现在真正落到库内目标（此前一律降级 ADD）。开关 `BLADEX_ADJUDICATE_LIBRARY_HYDRATE`（默认开，=0 回旧形态）。
- **删除三张常驻卡的注入渲染与 Flash `MATTERS.md` 设计**（拍板 0.3.0-(4)(5)，ADR-0034 §3）：`MemoryIndex.get_profile_cards` / `project_card_for`、`rulefile.render_profile_card`、`project.render_project_card`、`flash_tree.render_matters_list` 等入口不再存在；注入行为零变化（它们 09-03 起已无调用点），只影响自行 import 这些函数的脚本。
- **consolidator 空闲时不再每分钟轮询一次**（MQ-W17 修法①，批 X，2026-09-22）：连续 3 轮没有新对话后，轮询间隔按 60s→120→…→**封顶 1800s** 退避；**记忆库一有新数据 ≤1 秒内唤醒并立刻回到 60s 节奏**，所以消费延迟不受影响（实测优于退避前：新对话被消费的时延 133s → 25s）。修的是一个**自 v0.1.0 起就存在**的缺陷——空载时 consolidator 仍跑满全套例行作业，两万轮量级实测占空比约 54%、约 13 小时 CPU/天。新增三个旋钮 `BLADEX_CONSOLIDATOR_IDLE_BACKOFF` / `_IDLE_AFTER` / `_IDLE_MAX_S`（默认 `1` / `3` / `1800`），第一个设 `0` 逐字回到旧行为。
- **consolidator 心跳多了两个字段**（同批）：`bladex:consolidator:heartbeat` 新增 `interval_s` 与 `sleep_remaining_s`，`bladex status` 因此能区分"在退避、还剩 N 秒醒"与"卡住了"。读侧不认这两个键的旧版本不受影响。
- **与外部工具的双向同步（批 Z · Z4，MQ-P33，README 公开承诺第 3 条兑现）**：外部编辑一律经 proxy 的 admin 正门落成 **Memory Hub 管理事件**（Hub 唯一写者仍是 proxy；回流跑在 consolidator 进程，但不开任何 Hub / Index 写句柄）。
  - **Obsidian 默认开**（`config/export.toml` 该 exporter 的 `options.backflow`，默认 `true`）：改 fact 行 = 更正、挪到别的 Matter 页 = 换归属（挪进 `Inbox.md` = 摘除）、删行 = 墓碑、改 `Hard Rules.md` = 硬规则整表事件。导出器在 vault 里新增 `.bladex/sync_snapshot.json`（自写快照，判"是不是用户改的"）。**升级前已导出的 vault 首次扫描 = 收作基线、零事件**（认不出升级前哪些行被改过）。受影响：已配置 Obsidian 导出的用户——升级后在 vault 里的改动**会写回记忆**，不想要就设 `backflow = false`。
  - **Postgres 收件表默认关**（`options.inbox = true` 才生效，否则连表都不建）；新表 `bladex_edit_inbox`（只追加；BladeX 只回写 `status` / `result_reason` / `event_ref` / `processed_at`，从不删行）；**硬规则改动一律 rejected**。原导出表 `facts` / `matters` / `matter_edges` / `hard_rules` 行为一字不变。启动时每个目标打一行 `export_backflow_status`。
  - **新事件类型** `fact_correct`（状态式完整载荷，重放 = 覆盖；被更正的 fact 从此不被自动蒸馏 / 裁决 / 取代改动）与 `hard_rules_set`（仅 Obsidian；**生效依赖批 Z1.2 硬规则热加载读它作覆盖层**，Z1 未合入前只入账不生效）。墓碑新增加法式字段 `origin`（历史墓碑为空）；`matter_assign` / `matter_detach` 载荷在回流时多带 `origin` / `submitter`。
  - **新端点** `POST /admin/facts/{id}/correct`、`POST /admin/hard-rules/set`；`DELETE /admin/facts/{id}` 新增可选 `?origin=`，并改为**未知 query 参数 ⇒ 400**（MQ-I24 同款严格校验）。
  - **新旋钮** `BLADEX_BACKFLOW_MAX_DELETES`（默认 `5`）：一次扫描 / 一次轮询要删的 fact 超过此数 ⇒ 一条都不删、只告警；整页消失无论多少都不删。
  - **回流不再等 consolidator 的空载退避**（Z4b）：退避 sleep 期间 consolidator 逐秒探 Obsidian 受管文件（`Matters/*.md`、`Inbox.md`、`Hard Rules.md` 的 stat 签名），有变化就地跑一次回流 → 导出（实测改一行到 Hub 落 `fact_correct` ≈1 s，此前最坏 30 分钟）；导出器自己写的页面不算变化。Postgres 收件表按新旋钮 **`BLADEX_BACKFLOW_PG_PROBE_S`（默认 `30` 秒）** 探一次 `SELECT max(id)`，`inbox` 关时零 SQL。`BLADEX_CONSOLIDATOR_IDLE_BACKOFF=0` 时这条唤醒也一起关（逐字旧行为）。探针开销随 vault 页数线性：2000 页约 4.9 ms/次（1 Hz 下占 0.49% CPU）。
  - **手动换归属现在同步改 `fact.matter_id`**（`assign_manual` / `detach_edges_for_target`，与 `merge_matters` 同款）。此前 dashboard 手动挂卡后该字段仍指旧卡；受影响的是读 `fact.matter_id` 的消费方（Obsidian 导出页、同 Matter 兄弟跳）——现在与归属边一致。
- 🔴 **回收一个公开承诺过的用户面：Memory Flash 不再写 `User → Agent → Project → Session` 目录树**（批 F4，2026-09-23，ADR-0034 §3）。
  0.2.0 的 README 教过用户"把 `BLADEX_FLASH_PATH` 指到笔记库里读这棵树"——**这不是内部清理，是一项对外行为变更**。
  受影响：把笔记库 / 同步盘指向过 `data/bladex_flash`（或自定义 `BLADEX_FLASH_PATH`）的用户，升级后会看到
  `<principal>/personal/` 下的 `AGENTS.md`、`USER.md`、`RULES.md`、各 agent 子目录（`PROJECTS.md` / `SESSIONS.md` /
  `LEDGERS.md` / `PROJECT.md` / `AGENT.md`）**停在升级那一刻、不再更新**。**BladeX 不会替你删它们**（可能被你的
  笔记工具引用），确认不需要后请自行删除这些路径（`ledgers/` 与 `behavior.json` 除外）。取代者：账本仍在
  `ledgers/**/<id>.md` 可读可编辑；按 agent / 项目的活动情况改由 `behavior.json`（运维经 `/admin/behavior`；模型入口
  `bladex_flash_read` 已由 F5 判决退出注册，10-17 拍板 (45)）。同批删除四个旋钮 `BLADEX_FLASH_{SESSIONS_PER_PROJECT,PROJECTS_PER_AGENT,
  PROJECT_ACTIVE_DAYS,AGENT_RETIRE_DAYS}`——`.env` 里设了也不再有人读。
- **Flash 维护进程不再调用 LLM：agent / 项目一句话简介的蒸馏退役**（批 F4，拍板 (18)）。此前 `bladex-flash`
  每当某 agent 的 system prompt 或某项目的 `CLAUDE.md`/`AGENTS.md`/`README.md` 头部变化就经 Router 调一次便宜档模型，
  日志 `flash_daemon_agent_summarized` 自此消失。受影响：按调用次数计费的上游用户（调用归零）；读过名册
  `about` 列或 `PROJECT.md` 的用户（这些文件不再生成）。旧缓存 `.agent_summaries.json` / `.project_summaries.json`
  留在 Flash 根目录、不再读写，可自行删除。
- **AGENTS.md 名册退出注入稳定层 ⇒ 升级后每个会话的前缀 cache 冷启一次**（批 F4，拍板 (16)）。BladeX 的自我介绍块
  此前在末尾附一张"你用过哪些 agent"的名册（生产机实测 974 字符）；它没有注入以外的消费方；模型需要 agent / 项目
  活动情况时改经 `bladex_flash_read` 按需读取（该工具 F5 判决退出注册，0.3.0 正式版不提供）。升级后该块变短（`agency_about_injected chars` 9703 → 8729）、日志字段
  `roster_chars` 退役（`Turn.ledger_anchor.roster_chars` 字段保留，历史值可读、新轮恒 0）。**一次性代价**：
  稳定层内容变了，每个进行中会话的第一轮上游 prompt cache 不命中一次，之后恢复。简介蒸馏退役与名册退出
  同批落地（拍板 (18)），稳定层因本批只变这一次；批 Z2 的首轮定格若再改稳定层内容，另记一条。
- **`bladex start` 首启会等 embedding 模型下完，不再报假失败**（批 Z · Z1.1，MQ-A42，README 承诺兑现）。本地档首启要在 proxy 里下模型，此前 30 秒 `/ready` 窗口一到就报 `failed -- /ready did not answer within 30s`（每个新装用户首启一次）。现在：识别到"正在下载"就打 `downloading embedding model <名> (~N GB) -- first start only` + 每 10 秒进度，下完才重新计 30 秒窗口；下载失败或连续 120 秒无进度 ⇒ `state: failed -- the embedding model download failed: <原因>` 并**停掉刚起的 proxy**（此前它会带着"没有 Memory Index"降级跑起来）。embedding 独立模块（`BLADEX_MODULE_EMBED=1`）的 20 秒等待同样感知下载。新增文件：`<BLADEX_FASTEMBED_CACHE_PATH>/.bladex-download/<pid>.json`（下载期间存在，成功即删，失败留作诊断）。不在下载的真失败窗口不变。
  - 连带修两处 CLI 缺陷（批 Z · Z1.3 顺修）：`bladex start` 拉起的 proxy 若在启动期退出，此前被判成"活着"（僵尸进程）⇒ 白等 30 秒且报 `the proxy process is alive but /ready did not answer`，现在立刻停等并把日志里的 **traceback** 原样打到终端（此前只打 uvicorn 那句 `Application startup failed. Exiting.`）。受影响：所有遇到启动失败的用户——报错更早、更准。
- **硬规则热加载**（批 Z · Z1.2，MQ-W18，README 承诺兑现）：改 `config/.env` 的 `BLADEX_HARD_RULES` **不用重启**，proxy / consolidator / Flash 维护进程各自在 `BLADEX_HARD_RULES_RELOAD_S`（新旋钮，默认 `5` 秒，`0` = 回到"只在启动时读"）内生效，各打一行 `hard_rules_reloaded process=… source=env|hub old=N new=M`。Obsidian `Hard Rules.md` 的改动（Z4 落 Hub 的 `hard_rules_set` 事件）**自此真正生效**；两边都改过时**后改的赢**（按"`.env` 里这个值最近一次变化的时刻"比，改 `.env` 的别的键不算）。清空那一行 / 引号错配 / Hub 坏载荷 ⇒ **保留原规则**并 WARNING `hard_rules_reload_rejected`（此前清空 = 重启后回落两条默认规则）。shell 里另设了 `BLADEX_HARD_RULES` 时 `.env` 的改动不热加载（与启动语义一致），打 `hard_rules_env_shadowed`。新增文件 `data/hard_rules_env_seen.json`（记"值最近一次变化的时刻"）。
  - **导出目标跟着改**：生效规则一变，consolidator 就地重写 Obsidian `Hard Rules.md`（内容已一致则不动，保住你的排版）；**Postgres `hard_rules` 表改为与生效规则整表一致**——此前只在首次导出插入、只插不删，规则改过的库里会新旧并存，升级后第一轮导出会删掉不在当前规则集里的行。JSONL 目标追加新的 `hard_rule` 记录。`export_state.json` 每个目标多一个 `hard_rules_fp` 键。
- **同一条 Redis 流上第二个实例启动即被拒绝**（批 Z · Z1.3，MQ-A38）：proxy 连上 Redis 后在 `bladex:instance:<stream>` 登记自己的 `BLADEX_HOME`（TTL 30 秒、每 10 秒续，正常退出即删）。另一个 `BLADEX_HOME` 的实例连到**同一条 stream**（不论消费组是否相同）⇒ `refusing to start`，报错点名 stream、两边的消费组、对方的 `BLADEX_HOME` / pid / 端口，以及三旋钮 + 换 db 的隔离方法；此前它会静默吞掉对方的轮次、写进另一个 Hub。**升级窗口**：对方是 0.3.0 之前的版本（不登记）但正在读这条流 ⇒ 同样拒绝（判活：等 4 秒看消费者 idle 是否被重置），所以**升级时请先停旧进程再起新的**；刚停完就起只会多等 4 秒。同实例重启、崩溃后重启、用三旋钮隔离的实例都不受影响；异机崩溃留下的登记最多挡 30 秒。懒恢复（运行中 Redis 重连）遇到冲突 ⇒ 不接这条流、轮次继续落盘溢出 + ERROR `pipeline_redis_recover_refused`。
- 🔴 **账本切换不再按轮数拒绝：删除切换防抖，改为「举证」（观测阶段只记不拦）**（批 Y · Y1c，MQ-L91，2026-09-24，ADR-0032 §4.3 修订）。
  此前同一会话里距上次切换不足 3 轮再调 `bladex_ledger_switch` 会被拒（工具回文 `NOT switched (debounce)…`）——
  实测它一直在拦正当切换（09-23：距上次切换 13 小时的换题被拒两次），从未被确认拦对过一次。
  **受影响：所有用账本的 agent**——换题当轮即可切换，不再需要"再说两遍"；同一轮里 A→B→A 也会放行（可观测，不拦）。
  - **删除的旋钮**：`BLADEX_LEDGER_SWITCH_DEBOUNCE`（`.env` 里设了也不再读，请删掉该行）。取代上面「轮次口径统一」一条里关于防抖的那半句。
  - **工具参数新增（可选）**：`bladex_ledger_switch` 的 `evidence`——切到已有账本时由模型摘录本轮用户原话，BladeX 对照剥信封后的用户文本校验。不给也照常切换。
  - **Hub 记录**：`ledger_switch` 管理事件载荷新增 `evidence` / `evidence_ok`（`true` / `false` / 空）/ `evidence_source`（`user_text` / `new_ledger` / `child_writeback` / 空），加法式，历史事件无这三键照常可读；`turn_index` 仍写但不再被任何判定使用。新日志 `agency_switch_evidence`（每次落定切换一条）与 `agency_switch_evidence_mismatch`；**删除的日志** `agency_switch_debounced` / `agency_switch_user_requested`。
  - **注入面内容变更 ⇒ 前缀 cache 冷启一次**：工具 schema `bladex_ledger_switch` **1,815 → 2,066 字符（+251）**（全部工具 schema 7,277 → 7,528）；`TOOLS.md` **6,203 → 5,896 字符（−307）**（删防抖回文那一行、「恢复旧账本」一条加 evidence 用法；两处均在账本族段内）——注入渲染后（`render_system_notes`）账本面开 **6,014 → 5,707（−307）**、账本面关 **1,803 不变**。口径 = `json.dumps(schema, ensure_ascii=False)` / 字符数。
    🔴 **与批 Z · Z3.3（close 原语）合并描述**：同一版本内工具说明会再改一次，发版说明里 cache 冷启**只说一次**（升级后每个会话首轮冷一次），但两次改动的字符数**分开报**（本条 + Z3.3 那条）。
- **账本可以关闭 / 标记被取代 / 重开**（批 Z · Z3，MQ-L39 / MQ-L86，README 公开承诺第 2 条兑现，2026-09-24）。
  此前账本做完了只能一直挂着（`status` 字段恒为 `active`，从未被使用）——它继续出现在切换候选里，之后换题的轮次继续被锚进来。
  **受影响：所有用账本的 agent 与用户**；不调用新原语时行为逐字不变（见下「停锚」一条的默认值）。
  - **新工具** `bladex_ledger_close`（账本工具族，受 `BLADEX_LEDGER_TIERS` / 本会话绑定同一道门）：关激活账本或指定 `ledger_id`；可带 `superseded_by` = 标记被哪本取代。**不改 Goal、不改任何条目**。关的若是某些 scope 的激活账本 ⇒ 这些 scope 的激活随之解除（等于"切到空"，不回退到上一本）。**没有单独的 reopen 工具**：模型对一本已关的账本调 `bladex_ledger_switch` = 先重开再切（回文带 `(It was closed; switching reopened it.)`）。
  - **新 CLI** `bladex ledger close <id> [--reason …]` · `bladex ledger reopen <id>` · `bladex ledger supersede <id> --by <id>`；`bladex ledger show` 新增「Superseded by (follow the chain)」与「Supersedes」两段。三者都是追加式、可逆的管理事件，不同于墓碑（不问确认、不走备份软闸）。
  - **新端点** `POST /admin/ledgers/{id}/close | reopen | supersede`（body `{origin: "user"|"cli", reason, superseded_by}`；`origin="model"` ⇒ 400——模型只能经工具关本）；`GET /admin/ledgers/{id}` 响应新增 `superseded_chain` / `supersedes`。
  - **新事件类型** `ledger_close` / `ledger_reopen` / `ledger_supersede`（增量式载荷 `{ledger_id, origin, reason, …}`，模型来源另带 agent / project / session / turn_index）；全量重建后状态保持。`Ledger` 新增字段 `superseded_by`（缺省空；账本 `.md` 投影在有值时多一行 `- superseded by: \`ldg-…\``，可读可编辑、往返不丢）。
  - **切换候选**：已关的账本不再进 recency top-5 与相关性候选两段（相关性打分仍按全池算，没关的账本得分不变）。
  - **`ledger_update` 事件载荷**新增写者四格 `agent_id` / `project_id` / `session_id` / `turn_index`（加法式，历史事件无这四键照常可读）。
  - **注入面 ⇒ 前缀 cache 冷启一次**（与上面 Y1c 那条合并成一次）：新增 `bladex_ledger_close` 的 schema **459 字符**（OpenAI 形态；Anthropic 形态 427），全部工具 schema **7,518 → 7,977**（Anthropic 7,358 → 7,785；基线 7,518 = 在 Y1c 提交 `c3e65c9` 上按同一口径复算——上面 Y1c 那条写的 7,528 多了 10 字符，switch 单项 2,066 两边一致），只在账本族放行时注入；`TOOLS.md` 与账本块 **0 变化**。口径同 Y1c 条：`json.dumps(schema, ensure_ascii=False)` 字符数。Z3.3 的模型侧引导（"Verified 回答了 Goal、Open/Next 为空时关本；换到无关任务前先关"）就写在这份 schema 描述里，没有另加注入。
- **新开关 `BLADEX_LEDGER_DONE_PAUSE`（默认 `0`）：账本自述完结即停锚**（批 Z · Z3.2 / Y1b）。打开后：某次写入之后 Open 与 Next 都空、Verified 非空 ⇒ 之后的轮次在**全量重建**时不再锚到这本（live 绑定不解除；写入让 Open/Next 重新非空或显式切回 ⇒ 恢复），被停锚的轮次仍不能挂进别的账本锚卡（准入闸② 按原窗口判）。
  🔴 **默认关，读数驱动**：它清掉 Y0/Y1a 的 25 条目标错边中的 17 条，但让 **793 轮 / 774 条 fact** 离开 ANCHOR，分层抽 60 轮判读 **56 轮本就属于那本账本**（report-batch-Z3-20260923.md §⑤）。默认值下行为逐字不变；打开归 Jason 拍。

- **窗口感知路由 + BladeX 注入面按预算裁剪**（批 Z · Z2，MQ-A34 ①②，README 公开承诺第 1 条兑现，2026-09-24）。
  **默认零行为**：`routing.toml [[models]]` 新增可选字段 `context_window`（token）与 `chars_per_token`，不写 = 一切照旧。
  写了 `context_window` 的模型：① 动态选池排除"估算 prompt + 输出下限 > 窗口"的（估算 = 正文 + agent 工具 schema + 硬规则，
  按 chars/token 换算）；没配窗口的模型照常入池；全被排除 ⇒ 保留原池 + `route_context_window_no_fit`；粘性沿用同样要放得下
  （`route_sticky_no_fit`）；② 静态策略（req_model / team / principal / agent）钉死的放不下 ⇒ **不换模型**，WARNING
  `route_static_context_window_exceeded` 点名 source / agent / model / context_window / prompt_tokens_est 指向 routing.toml；
  ③ BladeX **自己的**注入面按 `(窗口 − 输出下限) × chars/token − agent prompt − 硬规则` 裁剪，顺序写死在
  `bladex_proxy/surface_budget.py::TRIM_ORDER`：切换候选段 → 账本块 → 账本族（工具 + 说明书账本段整体）→ 自我介绍 → 记忆族工具；
  **硬规则永不裁；agent 正文、工具与 `max_tokens` 一字不动**；每次裁剪 INFO `agency_surface_trimmed step=… chars=… budget_chars=…`，
  每轮 INFO `agency_surface_budget`。稳定层（工具 schema + 自我介绍）的裁剪**只在会话首轮决定一次并冻结**，之后每轮只裁动态层（账本块）。
  - **新旋钮** `BLADEX_CHARS_PER_TOKEN`（默认 `2.5`，Z2.0 读数 p10 一侧）· `BLADEX_CONTEXT_OUTPUT_FLOOR_TOKENS`（默认 `2048`）。
  - `route_decision` 日志新增 `context_window` / `prompt_tokens_est` 两列；`agency_toolface_decision` 新增 `stable_trimmed`。
  - `bladex doctor` 新增一行提示：看起来是本地模型（exposure=local 或 api_base 指本机）却没配 `context_window` 的条目（只提示，不填值）。
  - 受影响：给本地小窗口模型配了 `context_window` 的用户（MQ-A34 病例：hermes + ollama 32K）；其余用户无变化。
- **注入稳定层会话首轮定格 ⇒ 升级后每个进行中会话的前缀 cache 冷启一次**（批 Z · Z2.4，MQ-A58，ADR-0032 §3.1 兑现）。
  BladeX 的自我介绍块此前每轮按"最后一条 agent `system` 之后"重新定位，Claude Code 往数组末尾追加 `role=system` 的 system-reminder
  ⇒ 它在一个会话里漂十几次（Z2.0 读数：CC 67 次注入变了 32 次；hermes / codex / dsh 0 次）。现在首轮定格位置与内容，之后按**内容锚点**复位，
  agent 新追加的 system 一律排在它之后；锚点找不到（agent 压缩了历史）⇒ 回落老规则重新定格 + INFO `agency_about_anchor_lost`；
  账本面翻转 / proxy 升级改了 TOOLS.md ⇒ 内容合法地变一次 + INFO `agency_about_refrozen reason=ledger_face|notes_changed`。
  `agency_about_injected` 新增 `freeze=frozen|anchored|anchor_lost|none` 列。**一次性代价与 F4 / Y1c / Z3 合并描述**：
  升级后每个进行中会话首轮 cache 冷一次（CC 会话此后不再因位置漂移冷启）。🔴 修它的理由是实现要做到 ADR-0032 §3.1 写的那件事，
  **不是性能**——09-09 读数已证伪"漂移让会话变慢"，本批未测出新的性能差异（`cache_read_tokens` 对照 NO-DATA，见结算稿 §⑨）。

- **首次进入一个项目时不再自动沿用该 agent 在 Global 上的激活账本**（批 Y · Y1d，拍板 (30)，MQ-L93，2026-09-28）。
  此前（08-29 起）「项目键无绑定 ∧ 同 agent Global 有绑定 ∧ 本请求带项目信号」会把 Global 绑定**静默搬**到项目键
  （`ledger_scope_migrate` 事件、日志 `agency_scope_binding_migrated`）；实测全库 4 次里 3 次把项目专属账本搬进了别的项目。
  **受影响：所有带项目信号的 agent**（codex / claude-code / opencode / hermes 在仓库里跑时）——升级后首次进项目那一轮
  没有激活账本，账本块走「无激活本」分支：Global 上那本**置顶出现在 "Other recent ledgers" 首行**、带
  `(your active ledger outside any project - switch to it if this is the same task)` 标注，切不切归模型；
  该轮注入面比"直接沿用"多约 750 字符（隔离实测 1,457 → 2,211，其中置顶行 +83）。无项目信号的 agent（Pi / hermes 在家目录）零变化。
  已写入 Hub 的历史迁移事件照旧重放（重建结果不变）；`ActivationTable.migrate` 从库里删除，自行 import 它的脚本受影响。

- **09-28 之后直接写进发版说明、未在本节留行的条目**（批 REL 补索引，正文以定稿为准）：
  批 E · E4 关本后返工可找回 + 关本举证（MQ-L98，`be29191`）· 批 HO 已关账本按内容进相关性段、E4 小节并入（MQ-L99，`312d453`）·
  批 L100 子任务回写不重开已关父本（MQ-L100，`c6b631c`）· 批 F5 `bladex_flash_read` 退出注册（`0790033`）·
  内部子调用判 aux（MQ-A82 `5cbfed1` · A84–A86 `920459e` · A92 `605d971` · A93 `35ed2b0`）· codex 压缩后不丢项目（MQ-A83 `5cbfed1` `4532909` · A88 `3002107`）·
  Hermes 后台进程通知判 aux（MQ-S44 `e5e4323`）· `/v1/responses` 空 input 400（MQ-P24 `62e68ab`）·
  `bladex start` 降级退出码 3（`4a40b0a`）· 积压趋势 + `backlog_rising`（MQ-W4 `a3c1184`，W20 `4193003` `ca36a6a`）·
  CLI 进程发现不跨实例误认（MQ-A78 `21bb68a` `2fb0f58`）· 空库首启一次全起（MQ-A81 `4222564` · A89 `a429d11`）·
  同一 `data/` 拒第二个 Redis + `bladex repair-aof`（批 RD `b2771ce`）· `tombstone-project` 后台任务与客户端超时（MQ-I30 `31f0c3e` `b7875ce` `1a6d282`）·
  `consolidator --full` 中止报非零（MQ-I33 `edb6e02`）· `sync run` 进度与参数互斥（MQ-I25 `4809908`）· 日志保留（MQ-V21 `6845581` · `e17fc0c`）·
  consolidator 空载提速 / Hub 计数改读索引（MQ-W19 `4b10509` · `63b927e`）· dashboard 最近轮次不全扫（MQ-A91 `13e9776`）·
  CLI 文案英文化（MQ-L63 `4858d1f`）· doctor 跨 scope 同名账本（MQ-L94 `6fd4bc8`）· 写者四格 / gate 留痕 / 日志退役（MQ-L85 `8bdaa2c` · L84 `9cbfae1` · R10 `d4615c9`）·
  产品点名的修复脚本随发行包（`e5feff1` `a9bb124`，自检 `26cf4fe` `6a8294f`）。

### 5.2 0.4.0 收集点

（0.4.0 起每批改了用户可见行为就在这里加一行；格式同 5.1。）

- **适配 fastembed 0.9**（MQ-W21，拍板 (47)）：0.3.0 依赖限定 `fastembed>=0.3,<0.9`；0.9 把模型仓库名大小写改了（`qdrant/…` → `Qdrant/…`）⇒ 缓存目录大小写。**放开上限前先在 Linux 上实测老缓存是否被认出**（认不出 = 首启重下约 2 GB，要写进 0.4.0 发版说明「升级前必读」）。

## 6. Known gaps (recorded, not fixed here)

### 6.1 0.3.0 Known issues（批 REL 2026-10-17 与定稿同步，权威正文 = `docs/release-notes-0.3.0.md` §7）

- **MQ-L96** CONT 延续直挂账本锚卡沿上一轮投票选锚、不核内容 ⇒ 少量跨题挂卡（现存他题 15 + face=True 5）；T5 第一层具名未闭合项，修法排 0.4.0。
- **MQ-A80** agent 自压缩历史后 BladeX 开新会话 id ⇒ 轮次归零、原账本不再自动绑定（只在内容对得上时作切换候选）；排 0.4.0（拍板 (34)(46)）。
- **MQ-L95** Matter `topic_keys` 只增不减、L3 自我强化；排 0.4.0（拍板 (32)）。
- **MQ-P18** `/v1/responses` 推理参数不转发 ⇒ 推理内联进正文、`apply_patch` 可能失效；ADR-0023 保持 Proposed，排 0.4.0。
- **MQ-CA21** `BLADEX_ASSEMBLY_MIN_UNITS=10` 对会自压缩的 agent 不可达（默认关、不撞）；排 0.4.0，前置 MQ-A80。
- **MQ-I36** dashboard Projects 页只有一个假项目 `AGENTS.md`；🔴 未修，排 0.4.0 与 MQ-A75 同批（拍板 (46) 甲）。
- **MQ-L97** 同项目并发会话共用激活账本、后切覆盖先切；登记不修（只告警，全库写错本 0 次）。
- **MQ-S51** 无矛盾检测；排 0.4.0（README Coming next 第 3 条）。
- **MQ-I34 / I35** 全量重建无暂存切换（崩溃即 Index 损坏）、断网降级整批 ADD；README Coming next 第 5 条。
- **MQ-A57** 上游长输出 × 客户端 ~185 s 超时 ⇒ 重试风暴；排 0.4.0。
- **MQ-A79** 配了 `context_window` 时稳定层首轮定格后不再让位；排 0.4.0。
- **MQ-L74 / MQ-CA12 / MQ-CA13** 账本面每轮成本 ≈3× · 注入面在场时上游 prompt cache 失效 · glm 上 18–20% 整轮零缓存未解释；登记不修 / 拆分中。
- **F4 名册退出 + F5 `bladex_flash_read` 退出注册**：模型面不再有 agent / 项目活动情况入口（画像只留 dashboard / `/admin/behavior`）；已定（拍板 (45)），不修。
- （批 H 版本里的 MQ-A81 / MQ-A78 已于 09-29 闭合，移出；见定稿 §5「启动 / 运维」。）

### 6.2 流程侧

- `actions/checkout@v4`, `actions/setup-python@v5`, `astral-sh/setup-uv@v3` emit a
  *Node.js 20 deprecated* warning. Bumping them is a separate change (major-version bumps of
  actions can break the workflow) and should not ride along with a release.
- The GitHub Release is created as a draft with auto-generated notes; the hand-written release
  notes are pasted in when publishing the draft. Nothing checks that they exist.
