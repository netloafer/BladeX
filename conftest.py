"""测试环境隔离 —— ①不让测试流量写进生产 Pipeline/Memory Ledger/Memory Index；②不让生产配置渗进测试。

两件事都在 conftest.py 顶层做，因为它们都必须发生在**任何测试模块导入之前**。

────────────────────────────────────────────────────────────────────────
## 二、环境卫生：清空进程内的 BLADEX_*（2026-08-05，C6 实测驱动）

跑 BladeX 的终端通常 source 过 `config/.env`，于是 `BLADEX_AUTH_ENABLED=true` /
`CLIENT_KEYS` / `UPSTREAM_API_BASE` 等运行时值会压过测试预期的默认值，制造成片
**假失败**：所有 TestClient 端点 401、`test_auth_disabled_by_default` 直接红。

这套清洗原先只写在 `scripts/gate_check.sh` 里——**包装脚本级的修法**。C6 在公开树上
实测：`cd /tmp/bladex-public && pytest` 一次 19 条假失败，因为那里没有 gate_check。
公开仓的贡献者恰恰最可能设了 BLADEX_*（那是运行 BladeX 的前提），却拿不到任何解释。

所以搬进 conftest：任何调用方式（裸 pytest / IDE / CI / gate_check）都自动干净。
确需保留真实 env（调试特殊场景）用 `BLADEX_TEST_KEEP_ENV=1`。

────────────────────────────────────────────────────────────────────────
## 一、存储隔离 —— 禁止测试流量写进生产 Pipeline / Memory Ledger / Memory Index

## 为什么需要它（2026-07-28 真实事故）

ADR-0024 §5.1 刚把三层数据整体归档、起了全新库。重启后第一次检查发现新 Memory Ledger 里
已经有 13 条 `model=openai/test` 的 turn（query 是 `Hi` / `Say hello` / `Bye`）——
**是 pytest 的流量**。

根因：各测试**正确地**把 `rocksdb_path` 指到了 tmpdir，但**没有覆盖
`redis_stream`**，于是用了默认的生产流 `bladex:turns`：

    测试 Turn → 生产 Pipeline stream → 正在运行的 proxy 的 pipeline worker 消费
              → 写进生产 Memory Ledger

测试以为自己隔离了，其实只隔离了终点、没隔离入口。归档库里的
`test-agent` 207 轮 / `a` 190 轮就是这个机制的历史沉积。

## 做法

在**任何测试模块导入之前**（conftest.py 顶层执行）把全部存储类 env 指向
本次 run 专属的临时位置。`ProxyConfig` 的字段是 `field(default_factory=lambda:
_env(...))`，构造时才读 env，所以这里的覆盖对所有未显式传参的配置生效。

`BLADEX_REDIS_URL` **不覆盖**——集成测试需要连真实 Redis；隔离靠 stream/group
名字唯一，不靠断网。

## 边界

已显式传参的测试（如 `rocksdb_path=f"{tmpdir}/rocksdb"`）不受影响，行为不变；
`tests/integration/test_proxy_storage_pipeline.py` 自带 `_unique_stream()`，
与本文件叠加也安全（它显式传 stream 给 `PipelineRedis`）。
"""

from __future__ import annotations

import atexit
import itertools
import os
import shutil
import tempfile
import uuid

import pytest

# ── 环境卫生：先把继承来的 BLADEX_* 全部清掉（见模块 docstring 第二节）──
# 顺序要紧：必须在下面写隔离值**之前**清，否则会把刚设好的值一并删掉。
KEEP_ENV_FLAGS = ("BLADEX_TEST_KEEP_ENV", "BLADEX_GATE_KEEP_ENV")
_keep_env = any(os.environ.get(f) == "1" for f in KEEP_ENV_FLAGS)
SCRUBBED_ENV: dict[str, str] = {}
if not _keep_env:
    for _k in [k for k in os.environ if k.startswith("BLADEX_")]:
        SCRUBBED_ENV[_k] = os.environ.pop(_k)

# ── 本次 pytest run 专属命名空间 ──
_RUN_ID = uuid.uuid4().hex[:8]
_TMP_ROOT = tempfile.mkdtemp(prefix=f"bladex-test-{_RUN_ID}-")

# 生产默认值（守卫测试用来断言"没被用到"）
PRODUCTION_DEFAULTS: dict[str, str] = {
    "BLADEX_ROCKSDB_PATH": "data/bladex_hub",
    "BLADEX_INDEX_PATH": "data/bladex_index",
    "BLADEX_REDIS_STREAM": "bladex:turns",
    "BLADEX_REDIS_GROUP": "bladex-workers",
    "BLADEX_OVERFLOW_DIR": "data/overflow",
    # G11.10：用户 agent 规则文件。不隔离的话测试会读**真实部署**的那份，
    # 结果随用户当天在 dashboard 存了什么规则而变（2026-08-19 实测炸过三条）。
    "BLADEX_AGENT_RULES_PATH": "config/agent_rules.toml",
    # MQ-A10：同上，注册表缓存也必须隔离。
    "BLADEX_AGENT_REGISTRY_CACHE": "data/agent_registry.json",
}

# 无条件覆盖（不是 setdefault）——开发机上 config/.env 可能已经导出了生产值，
# setdefault 会让隔离静默失效，正是本次事故的形状。
_ISOLATED = {
    # 🔴 测试注入的 env 必须赢过 config/.env（2026-08-06）。
    # 那天把默认优先级翻成了"config/.env 压过环境变量"（终端里的残留 export 曾
    # 静默劫持配置文件，两次事故见 bladex_proxy/deployment.py）。但**测试**属于
    # 与编排器同一类的场景：env 是本进程刻意注入的隔离值，不是残留。不钉死这一条，
    # 下面那几行 tmp 路径会被仓库根的真实 .env 直接盖掉 —— 测试流量重新写进生产库，
    # 正是 2026-07-28 事故的形状。
    "BLADEX_ENV_PRECEDENCE": "env",
    "BLADEX_ROCKSDB_PATH": os.path.join(_TMP_ROOT, "ledger"),
    "BLADEX_INDEX_PATH": os.path.join(_TMP_ROOT, "index"),
    "BLADEX_OVERFLOW_DIR": os.path.join(_TMP_ROOT, "overflow"),
    "BLADEX_REDIS_STREAM": f"bladex:test:{_RUN_ID}",
    "BLADEX_REDIS_GROUP": f"bladex-test-workers:{_RUN_ID}",
    "BLADEX_REDIS_CONSUMER": f"test-worker-{_RUN_ID}",
    # 指向一个**不存在**的临时路径：默认形态就是"没有用户规则"，
    # 需要用户规则的用例自己写文件到 tmp_path 并显式传 user_path。
    "BLADEX_AGENT_RULES_PATH": os.path.join(_TMP_ROOT, "agent_rules.toml"),
    # MQ-A10：agent 注册表缓存（跨重启的 origin_key→agent 绑定 + 待认领桶）。
    # 与上面那条同型同因：server 启动会 `load_cache()`，不隔离就会读写**真实部署**
    # 的 `data/agent_registry.json`，用例之间互相污染（2026-08-26 当场踩到：
    # 一个全新用例启动时打出 `pending=4`，那 4 条是上一个用例留下的）。
    "BLADEX_AGENT_REGISTRY_CACHE": os.path.join(_TMP_ROOT, "agent_registry.json"),
    # Z1.2（MQ-W18）：硬规则热加载会读**部署根**的 config/.env 与 Hub 里的
    # `hard_rules_set`，并压过 `ProxyConfig(hard_rules=[...])` 的显式值——不关的话，
    # 每个跑 lifespan 的用例拿到的硬规则随开发机当天的 .env 而变（与上面两条同型）。
    # 热加载自己的用例显式传 `interval_s` / monkeypatch 这个变量。
    "BLADEX_HARD_RULES_RELOAD_S": "0",
}
os.environ.update(_ISOLATED)

# 🔴 批 Z1b（MQ-A77）：测试进程闸——任何测试连 live 默认 Redis stream 一律抛错。
# 放模块属性不放 env：要拦的缺陷形状正是"夹具删光 BLADEX_* 环境变量"（见 pipeline_redis 注释）。
try:
    from bladex_proxy.storage import pipeline_redis as _pipeline_redis

    _pipeline_redis.TEST_PROCESS_GUARD = True
except ImportError:  # 只装了 bladex-core 的环境：没有 Redis 管线可守
    pass

# ── 🔴 测试面禁止联网下载 embedding 模型（2026-09-01，gate 挂死 driven）──
#
# 病灶：任何 `create_app()` + `TestClient` 的用例都会跑真 lifespan →
# `build_embedder` → `ensure_local_model`；缓存缺文件时它**发起 HF 下载**。
# 而 `ensure_local_model` 抛异常时 proxy 是「捕获后降级只注硬规则」的 ⇒
#
#   网络**快速失败** ⇒ 降级、app 正常起、用例通过
#   网络**可达但慢** ⇒ 无限等，gate 整个挂死
#
# **同一段代码，结果由当天网速决定**——比失败更糟的不是慢，是"有时过有时挂"。
# 2026-09-01 实测：`test_cli_ledger::test_contract_list_fields_from_real_endpoint`
# 卡在 `snapshot_download`，faulthandler 打出四个并发下载线程；下的是
# `DEFAULT_LOCAL_MODEL`（bge-small），它在 `data/fastembed_cache` 里的 blob
# 自 2026-07-26 起就是 `.incomplete`，一直没人发现。
#
# `HF_HUB_OFFLINE=1` 的语义是**只读缓存、绝不发请求**：
#   命中（e5-large 完整）⇒ 照常加载，行为不变；
#   未命中 ⇒ 立刻抛 `LocalEntryNotFoundError` ⇒ 走既有降级路径 ⇒ 用例秒过。
# 把一个网速赌局换成一条确定性分支，与仓库"静默降级比失败更糟"同一条纪律。
#
# 🔴 用 setdefault：本机确实要下模型时（首次装机 / 换模型）可以
# `HF_HUB_OFFLINE=0 bash scripts/gate_check.sh` 显式放行，逃生门不焊死。
# 非 BLADEX_ 前缀 ⇒ 不受上面那轮清洗管辖，也不进 POST_SCRUB 快照。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

#: 清洗 + 写隔离值之后，进程里应当**只**剩这些 BLADEX_*。
#: 守卫测试断言的是这个快照，不是"运行期间 os.environ 始终干净"——后者会被任何
#: 合法地 monkeypatch.setenv 的测试打破，是个按执行顺序抽风的断言。
POST_SCRUB_BLADEX_ENV: tuple[str, ...] = tuple(
    sorted(k for k in os.environ if k.startswith("BLADEX_"))
)


@atexit.register
def _cleanup_tmp_root() -> None:
    """run 结束清临时目录；失败不致命（临时目录由 OS 兜底回收）。"""
    shutil.rmtree(_TMP_ROOT, ignore_errors=True)


# ── 三、每个用例之后把 BLADEX_* 恢复到隔离基线（2026-08-06，gate 实测驱动）──
#
# 上面第二节的清洗只发生**一次**（conftest 导入时）。但污染可以在 run 中途注入：
# `cli.main()` / `consolidator._load_env_file()` → `deployment.load_env_file()`
# 会把部署根的 `config/.env` **整份灌进 os.environ 且不回收**（monkeypatch 也管不着，
# 它没记录这些键）。于是任何调 CLI 的用例都会把开发机上的真实配置留给后面所有用例。
#
# 实测（gate_check 的 NEW_TESTS 顺序）：`test_sync_cli.py` 之后进程里多出 28 个
# BLADEX_*，含 `BLADEX_ADMIN_KEYS` / `BLADEX_UPSTREAM_API_KEY`；随后
# `test_admin_read_api::test_admin_read_requires_key_when_auth_enabled` 拿合法数据面 key
# 却收到 401 —— 因为 `BLADEX_ADMIN_KEYS` 一旦存在，管理面就要求管理 key（ADR-0027 §2.2）。
# 全量 pytest 里同一条用例是绿的：collection 顺序把 admin 排在 sync 之前。
# **一条按执行顺序抽风的假失败**，而且指向的地方（鉴权）与真因（环境污染）毫无关系。
#
# 这个洞 2026-08-03 就出现过一次，当时的修法是在 `test_cli_lifecycle.py` 里加一个
# autouse fixture 清 BLADEX_*——**修在了单个文件层**。后来又出现第二个调 CLI 的测试文件
# （`test_sync_cli.py`），没人记得照抄那段，洞就回来了。
# 所以按"护栏要随被保护对象走"的原则搬到这里：**任何**用例、**任何**调用方式都自动恢复。
_BASELINE_BLADEX_ENV: dict[str, str] = {
    k: v for k, v in os.environ.items() if k.startswith("BLADEX_")
}


@pytest.fixture(autouse=True)
def _restore_bladex_env_baseline():
    """每个用例结束后把 BLADEX_* 恢复成隔离基线。

    autouse 且无依赖 → 最先 setup、最后 teardown，因此在 monkeypatch 自己的
    undo **之后**运行：基线是最终态，不会被 monkeypatch 恢复的旧值再盖回去。

    合法的 `monkeypatch.setenv` 不受影响（用例执行期间照常生效）；被清掉的只有
    "用例结束后还赖着不走"的那些——那正是污染的定义。
    """
    yield
    for k in [k for k in os.environ if k.startswith("BLADEX_")]:
        if k not in _BASELINE_BLADEX_ENV:
            del os.environ[k]
    for k, v in _BASELINE_BLADEX_ENV.items():
        if os.environ.get(k) != v:
            os.environ[k] = v


@pytest.fixture(autouse=True)
def _reset_agent_registry():
    """每个用例之间清 agent 注册表单例 + 它的落盘缓存（MQ-A10，2026-08-26）。

    🔴 为什么必须有：`_agent_registry` 是**进程级单例**，`note_unrecognized`
    的待认领登记会一路累加；2026-08-26 给它加了**跨重启落盘缓存**之后更糟——
    server 启动会 `load_cache()`，于是 A 用例写盘的 pending 会被 B 用例读回来。
    症状是**只在整套里失败、单跑通过**（order-dependent），最难查的那一类。

    与上面那条 env 基线 fixture 同款理由：护栏要随被保护对象走，
    修在单个测试文件里迟早会漏（`test_cli_lifecycle` 那次的教训）。
    """
    from bladex_proxy import agent_registry as _ar
    _ar._agent_registry.clear()
    _ar._autosave_path = None
    try:
        _ar.cache_path().unlink(missing_ok=True)
    except OSError:
        pass
    yield
    _ar._agent_registry.clear()
    _ar._autosave_path = None
    try:
        _ar.cache_path().unlink(missing_ok=True)
    except OSError:
        pass


# ── 四、每个用例一条自己的 Redis stream（批 GS，2026-09-29，gate 变慢 driven）──
#
# 病灶：上面第二节给整个 run 只分了**一条** stream / 一个组 / 一个消费者名。
# Z1.3（`612a20e`，09-23）起 lifespan 启动先跑 `InstanceClaim.acquire()`：登记已被
# 上一个用例的 app 在关停时删掉 ⇒ 组里却有一个 4 s 内刚动过的消费者（就是上一个用例的
# worker）⇒ 按"同一实例刚重启"的设计**等满一个判活窗口 `LIVENESS_WINDOW_S`（4 s）**
# 再采样。生产上一次重启付一次；测试里每个起 TestClient 的用例都付一次。
# 实测：同一文件 `test_admin_memory_classes.py` 在 `612a20e~1` 2.29 s、在 `612a20e`
# 26.10 s（每例 0.03 s → 4.08 s）；HEAD 全量 5,532 例里 ≥2 s 的有 259 例、合计 1,279 s
# （09-18 只有 16 例），gate 两段各吃一遍。
#
# 修法：共享的是测试之间的状态，就在测试之间断开——每个用例一条新 stream（消费组挂在
# stream 之下、登记键由 stream 名派生，随之全新），用例结束删掉。**不改 `LIVENESS_WINDOW_S`**：用例内
# 真的重启两次 app 的场景照旧走完整判活，那正是它要测的东西。
# 顺带：此前每个 run 在 Redis 里留一条 `bladex:test:<run>` 从不回收（09-29 数到 1,547 条）。
_STREAM_SEQ = itertools.count()
_stream_janitor: list = []  # [redis.Redis | None]；首次用到时建，连不上就记 None 不再试


def _drop_test_stream(stream: str) -> None:
    """删掉本用例的 stream 及其实例登记键；Redis 不在 / 出错都不影响用例结论。"""
    if not stream.startswith("bladex:test:"):  # 只清测试命名空间，别的一概不碰
        return
    if not _stream_janitor:
        try:
            import redis
            from bladex_proxy.config import ProxyConfig

            client = redis.Redis.from_url(ProxyConfig().redis_url,
                                          socket_connect_timeout=0.5, socket_timeout=1.0)
            client.ping()
            _stream_janitor.append(client)
        except Exception:  # noqa: BLE001 —— 没装 redis / 没起 Redis：没有东西要清
            _stream_janitor.append(None)
    client = _stream_janitor[0]
    if client is None:
        return
    try:
        client.delete(stream, f"bladex:instance:{stream}")
    except Exception:  # noqa: BLE001 —— 清理失败不许把一个绿用例染红
        pass


@pytest.fixture(autouse=True)
def _per_test_redis_stream():
    """本用例的 `BLADEX_REDIS_STREAM` = `bladex:test:t<序号>:<run id>`；结束后删 stream。

    run id 仍是最后一段（`test_stream_and_group_unique_per_run` 按这个约定解析）。
    排在 `_restore_bladex_env_baseline` 之后 setup、之前 teardown ⇒ 用例结束时
    env 仍由那个 fixture 恢复成 run 级基线（守卫断言的基线不变）。
    """
    stream = _next_test_stream()
    os.environ["BLADEX_REDIS_STREAM"] = stream
    yield
    _drop_test_stream(stream)


def _next_test_stream() -> str:
    return f"bladex:test:t{next(_STREAM_SEQ)}:{_RUN_ID}"


@pytest.fixture
def fresh_redis_stream():
    """给"一个用例里先后起两个 app"的场景再分一条 stream（用例结束一并删除）。

    用法：两次 `create_app` 之间 `monkeypatch.setenv("BLADEX_REDIS_STREAM", fresh_redis_stream())`。
    第二个 app 若沿用第一个的 stream，会被当成"同一实例刚重启"而等满 4 s 判活窗口——
    被测的若不是重启语义，这 4 s 就是纯等待。**真在测重启的用例别用它。**
    """
    made: list[str] = []

    def _make() -> str:
        made.append(_next_test_stream())
        return made[-1]

    yield _make
    for stream in made:
        _drop_test_stream(stream)


# ── 五、测试不许连 live 的 embed 服务，默认也不加载真实模型（批 GS2 · MQ-A87，2026-09-29）──
#
# 病灶：tracked 的 `config/routing.toml` 写着 `[embedding] backend = "ipc"`，而 socket 地址由
# `embed_transport.default_uds_path()` 经 `deployment.find_root()` 解析——主仓有 `config/.env`
# ⇒ 解析到**部署根的 `run/embed.sock`，也就是 live 的 embed_server 进程**。每个跑 lifespan 的用例
# 都会 `embedder.embed(["warmup"])` 打过去（修前实测：一次全量 507 次连接、243 个用例）。
# 快照 / 公开树没有那个 socket ⇒ `embed_ipc_fallback_local` 在 pytest 进程里加载 e5-large
# （RSS 15–16 GB）。**同一套测试，行为随"在哪个目录跑"而变，而且碰到了生产进程。**
#
# 修法（测试侧，产品代码不动）：
#   1. 默认：`build_embedder` 在它的两个消费方模块上换成确定性假 embedder（不连 socket、不加载模型）。
#      与 `test_responses_endpoint` / `test_sync_cli` 早就在用的 patch 点同一处；用例自己再 patch 的照旧后写者赢。
#      测工厂本身的 `test_embedding_backends` 在模块顶层按名字导入了原函数，不受影响。
#   2. 守卫：任何 IPC adapter 的地址解析到部署根 `run/embed.sock` ⇒ 抛错，并在用例 teardown
#      `pytest.fail` 点名（lifespan 会把 embedder 构建异常降级吞掉，只抛不记就是静默）。
#      与 Z1b「测试进程连 live Redis 命名空间即抛」同一形状。
#   批 GS2 盘点：修前打 live 的 243 个用例换成**每次返回随机向量**的 embedder 仍全绿，关键词候选的
#   30 个文件都用各自的 mock ——**没有用例依赖真语义**，所以不设"真模型"夹具。将来真要语义：
#   直接构造 `FastEmbedAdapter(model_name=..., cache_dir=<仓库>/data/fastembed_cache)`（HF 离线，
#   不走 `build_embedder` / IPC）。

class LiveEmbedInTestError(RuntimeError):
    """测试进程试图连部署根下的 live embed socket。"""


def _live_embed_sockets() -> frozenset[str]:
    """清洗前后都算：部署根（cwd 祖先 / `~/.bladex`）+ 被清洗掉的 `BLADEX_HOME`。"""
    roots: set[str] = set()
    try:
        from bladex_proxy import deployment

        found = deployment.find_root()
        if found:
            roots.add(found)
        roots.add(deployment._expand(deployment.DEFAULT_HOME))
    except ImportError:  # 只装了 bladex-core：没有 embed 客户端可守
        pass
    if SCRUBBED_ENV.get("BLADEX_HOME"):
        roots.add(os.path.expanduser(SCRUBBED_ENV["BLADEX_HOME"]))
    return frozenset(os.path.realpath(os.path.join(r, "run", "embed.sock")) for r in roots)


LIVE_EMBED_SOCKETS = _live_embed_sockets()

try:
    from bladex_proxy import embed_client as _embed_client

    # 🔴 状态挂在 embed_client 模块对象上、包装只做一次：有用例 `import conftest`（importlib 模式下
    # 是第二份模块副本，会把本文件顶层再跑一遍）——若各副本各包一层、各记各的列表，最外层那份
    # 记下的违规 pytest 这份 teardown 看不见，守卫就静默了。
    if not hasattr(_embed_client, "_TEST_LIVE_VIOLATIONS"):
        _embed_client._TEST_LIVE_VIOLATIONS = []
        _embed_client._TEST_LIVE_SOCKETS = LIVE_EMBED_SOCKETS
        _orig_ipc_init = _embed_client.IPCEmbedAdapter.__init__

        def _guarded_ipc_init(self, transport, address, *a, **k):  # type: ignore[no-untyped-def]
            if transport == "uds" and os.path.realpath(str(address)) in LIVE_EMBED_SOCKETS:
                who = os.environ.get("PYTEST_CURRENT_TEST", "?").split(" ")[0]
                _embed_client._TEST_LIVE_VIOLATIONS.append(f"{who} → {address}")
                raise LiveEmbedInTestError(
                    f"测试进程试图连 live embed socket {address}（{who}）。"
                    "默认应走 conftest 的假 embedder（FakeEmbedder）。")
            _orig_ipc_init(self, transport, address, *a, **k)

        _embed_client.IPCEmbedAdapter.__init__ = _guarded_ipc_init
    _live_embed_violations: list[str] = _embed_client._TEST_LIVE_VIOLATIONS
except ImportError:
    _live_embed_violations = []


class FakeEmbedder:
    """确定性假 embedder：词 + 字二元组哈希到 64 维、L2 归一化。

    同一文本 ⇒ 同一向量；共享词越多余弦越高（够"去重 / 召回自身"类用例用，不冒充真语义）。
    `hash()` 受 PYTHONHASHSEED 影响，这里用 blake2b 保证跨进程稳定。
    """

    DIM = 64
    model_identity = "test:fake-hash-64"
    model_name = "test/fake-hash-64"
    available = True

    def _vec(self, text: str) -> list[float]:
        import hashlib
        import math
        import re

        v = [0.0] * self.DIM
        s = text.lower()
        feats = re.findall(r"\w+", s) + [s[i:i + 2] for i in range(len(s) - 1)]
        for f in feats or [s]:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.DIM] += 1.0 if (h >> 32) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    embed_passage = embed
    embed_query = embed


@pytest.fixture(autouse=True)
def _default_fake_embedder(monkeypatch):
    try:
        from bladex_proxy import embedding as _emb
        from bladex_proxy.server import app_factory as _af

        fake = lambda cfg, role="proxy": FakeEmbedder()  # noqa: E731
        monkeypatch.setattr(_emb, "build_embedder", fake)
        monkeypatch.setattr(_af, "build_embedder", fake)
    except ImportError:
        pass
    start = len(_live_embed_violations)
    yield
    if len(_live_embed_violations) > start:
        hits = _live_embed_violations[start:]
        pytest.fail(f"用例连了 live embed socket（{len(hits)} 次）：{hits[0]}", pytrace=False)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """把 `HOME` 指到临时目录，隔断 `~/.bladex` 兜底（2026-08-06）。

    部署根发现现在是 `BLADEX_HOME → cwd 及祖先 → ~/.bladex`。最后那条兜底是给
    全局安装（`pip` / `uv tool`）的用户的，但它也意味着：**开发机上真有一个
    `~/.bladex/config/.env` 时，任何 chdir 到 tmp_path 再调 `cli.main()` 的用例
    都会跑去加载那份真实配置**——正是本文件开头那两段隔离要防的事，只是换了个入口。

    不做成 autouse：整包重定向 HOME 会让真实 e5 用例把模型重下一遍到临时目录
    （fastembed 缓存路径虽已隔离，HF/onnx 那层的默认位置仍挂在 HOME 下）。
    所以按需注入——凡是"chdir 到临时目录再跑 CLI"的模块显式请求它。
    """
    fake = tmp_path / "fake_home"
    fake.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake))
    monkeypatch.setenv("USERPROFILE", str(fake))  # Windows 上 expanduser 认它
    return fake


# ── 六、起过的 PipelineWorker 用例结束时必须已停（批 CI1，2026-10-17，公开仓 CI 卡死 driven）──
#
# 病灶：lifespan 关停只停 `app.state.pipeline_worker` 上**当时**挂着的那个。用例在
# `with TestClient(app)` 里把它换成替身（或 None）不换回 ⇒ 真 worker 的任务没人停，
# 留到 anyio 关事件循环时由 `_cancel_all_tasks` 收尾。macOS 上收得回、照绿；
# Linux + 真 Redis 下整跑到第 ~1,080 条时收不回，CI 在 `TestClient.__exit__` 挂 29 分钟
# （`test_admin_read_api.py::test_queue_flush_defaults_to_dry_run`）。
#
# 判据看 `_running` 而不是任务是否结束：漏停的任务最后也会被关循环取消掉（所以 macOS
# 才绿），只有 `stop()` 会把 `_running` 置回 False ⇒ 这一位在任何平台上都能红。
@pytest.fixture(autouse=True)
def _pipeline_workers_stopped():
    from bladex_proxy.storage.pipeline_worker import PipelineWorker

    started: list = []
    orig_start = PipelineWorker.start

    def _tracking_start(self):  # type: ignore[no-untyped-def]
        started.append(self)
        return orig_start(self)

    PipelineWorker.start = _tracking_start  # type: ignore[method-assign]
    try:
        yield
    finally:
        PipelineWorker.start = orig_start  # type: ignore[method-assign]
    leaked = [w for w in started if w._running]
    if leaked:
        pytest.fail(
            f"{len(leaked)} 个起过的 PipelineWorker 在用例结束时没被 stop()——"
            "多半是在 `with TestClient(app)` 里替换了 app.state.pipeline_worker 却没换回，"
            "lifespan 关停只停替身。改法：记下原对象，finally 换回（见 test_agent_claim）。",
            pytrace=False,
        )
