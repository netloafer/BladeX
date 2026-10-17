"""不可逆 admin 操作的 **CLI↔服务端协议版本**（MQ-I24，批 T0b 2026-09-22）。

# 为什么需要它

2026-09-22 16:37 的事故：新 CLI（源码树）打旧 proxy（15:39 启动，T0 之前的代码）。
CLI 发 `?with_memory=true&dry_run=true`，而那版 `delete_ledger` **根本没有 `dry_run`
这个参数** —— FastAPI 对未声明的 query 参数**静默忽略** ⇒ 被当成普通 `with_memory=true`
**真删**执行；CLI 只看了 HTTP 200，照 dry-run 的格式打印了 "DRY RUN -- nothing written."。
两本账本连记忆一起不可逆地没了，终端说"什么都没写"。

⇒ **"跑着的进程是不是这份代码"必须是一个可以问出来的事实**，而不是靠"我刚提交过"的记忆。

# 为什么是一个整数，而不是 git commit

commit 哈希会因为**任何**提交（改一行文档也算）而变，⇒ 每次提交都会把所有不可逆操作
拒掉，直到重启 proxy。那样的闸会被绕过去（人会加 `--force`），等于没有。

本常量只在**不可逆 admin 操作的线上协议**发生变化时 +1：新增/删除/改名 query 参数、
改响应里那些 CLI 用来判断"服务端到底做了什么"的标记、新增一条不可逆端点。
⇒ 它精确地回答"我要用的那条协议，对面认不认"，没有假阳性。

# 判定 = 相等，不是 `>=`

不相等一律拒绝执行（不是告警后照发）。CLI 与服务端本来就来自同一个仓库检出，
**不相等只有一个成因：跑着的进程不是这份代码**——那正是要拦的那件事。

🔴 **改动纪律**：动了任何一条不可逆 admin 端点的线上协议 ⇒ 本常量 +1，
并在收口清单里写明"生产 proxy 必须重启"（dispatch §5）。守卫见
`tests/test_t0b_dryrun_gates.py::test_b_contract_constant_is_single_sourced`。
"""

from __future__ import annotations

#: 不可逆 admin 操作的线上协议版本。1 = 批 T0b（2026-09-22）首次声明：
#: `DELETE /admin/ledgers/{id}` 的 `dry_run` 标记 + `DELETE /admin/ledgers/{id}/memory`
#: + `POST /admin/turns/tombstone-by-project` + 未知 query 参数改报 400。
#: 2 = 批 Q3A（2026-09-29，MQ-I30）：`POST /admin/turns/tombstone-by-project` 真跑改为
#: 后台 job——回 `status="submitted"` + `job_id`，结果经 `GET /admin/jobs/{job_id}` 取。
#: 旧 CLI 会把 "submitted" 当失败渲染 ⇒ 必须 +1，生产 proxy 必须重启。
ADMIN_CONTRACT_VERSION = 2

#: `/ready` 里承载它的键名（CLI 与服务端共用，避免两处各写一个字符串）。
ADMIN_CONTRACT_FIELD = "admin_contract"
