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
}
os.environ.update(_ISOLATED)

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
