"""`bladex doctor` 与它的探针。批 LC 自 `lifecycle.py` / `cli/__init__.py` 零行为搬出。"""

from __future__ import annotations

import errno
import json
import secrets
import socket
import time
import urllib.request
from pathlib import (  # noqa: E402
    Path,
)

import typer

from bladex_proxy.cli import (  # noqa: E402
    _PROXY_PIDFILE,
    _UNDETERMINED_TEXT,
    RC_UNDETERMINED,
    _admin_key,
    _consolidator_pid,
    _first_client_key,
    _http_json,
    _http_request,
    _probe_ready,
    _read_pidfile,
    app,
)
from bladex_proxy.cli.embed_preflight import (  # noqa: E402
    _embed_api_probe,
)
from bladex_proxy.cli.init_cmd import (  # noqa: E402
    _ASSET_TARGETS,
)
from bladex_proxy.cli.lifecycle import (  # noqa: E402
    _AOF_REPAIR_HINT,
    _flash_refused_text,
    _redis_hostport,
    _redis_ping,
    _shell_restriction,
)
from bladex_proxy.storage.redis_launch import (  # noqa: E402
    aof_manifest_problem,
    redis_servers_on_data_dir,
)


def _hidden_pth_files() -> list[str]:
    """site-packages 里带 macOS `UF_HIDDEN` 的 `.pth` 文件。

    为什么值得单独查（2026-08-05 真实事故）：Python 3.11+ 的 `site.addpackage()`
    对带 hidden 标志的 `.pth` **静默 return**——不报错、不警告。后果是
    editable 安装的包全部导不进来，症状是 `ModuleNotFoundError`，而
    `pip list` / dist-info / `.pth` 内容**全都看起来正常**，指不到真因。

    触发场景：项目放在 iCloud Drive / OneDrive 等同步目录里（macOS 会给
    这些文件打 hidden）。非 macOS 上 `st_flags` 不存在，本检查恒为空。
    """
    import site
    import stat

    hidden: list[str] = []
    try:
        dirs = list(site.getsitepackages())
    except Exception:  # noqa: BLE001 —— 非常规解释器布局，跳过即可
        return hidden
    for d in dirs:
        try:
            entries = list(Path(d).glob("*.pth"))
        except OSError:
            continue
        for p in entries:
            try:
                st = p.lstat()
            except OSError:
                continue
            if getattr(st, "st_flags", 0) & getattr(stat, "UF_HIDDEN", 0):
                hidden.append(str(p))
    return hidden


def _probe_upstream(cfg, timeout_s: float = 5.0) -> tuple[str, str, str]:
    """带凭证探一次上游（ADR-0027 §3.1）。

    返回 `(state, detail, fix)`，state ∈:
      ok          — 拿到 2xx，凭证与网络都成立
      auth_failed — 拿到 401/403，网络通但 key 不对/为空
      unreachable — 网络层不通
      not_probed  — 没有可探的地址（走 Router 默认 endpoint），如实说未探测

    只打 `GET {api_base}/models`（OpenAI 兼容面通用、只读、无 token 消耗）。
    """
    api_base = (cfg.upstream_api_base or "").rstrip("/")
    api_key = (cfg.upstream_api_key or "").strip()
    if not api_base:
        return ("not_probed",
                "no BLADEX_UPSTREAM_API_BASE (Router default endpoint); cannot probe from here",
                "")
    if not api_key:
        return ("auth_failed", f"{api_base} -- BLADEX_UPSTREAM_API_KEY is empty",
                "set BLADEX_UPSTREAM_API_KEY in config/.env")

    url = f"{api_base}/models"
    req = urllib.request.Request(url)  # noqa: S310 —— 用户自配的上游地址
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return ("ok", f"{api_base} (HTTP {resp.status})", "")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return ("auth_failed", f"{api_base} -- HTTP {e.code} (key rejected)",
                    "check BLADEX_UPSTREAM_API_KEY (and that the key matches this api_base)")
        # 404/405 等：说明服务在、只是没有 /models 这个路径 —— 凭证无法断言，但网络与
        # 端点都成立，不误报为失败。
        return ("ok", f"{api_base} (HTTP {e.code} on /models; endpoint reachable)", "")
    except (urllib.error.URLError, OSError, ValueError) as e:
        return ("unreachable", f"{api_base} -- {e}",
                "check BLADEX_UPSTREAM_API_BASE, network and proxy settings")


def _probe_port(host: str, port: int) -> tuple[str, str]:
    """端口状态 -> (state, detail)。

    state: free / ours / foreign / undetermined / error

    为什么不是简单的"占用就警告"（2026-08-05 C7 实测）：`bladex start` 之后跑 doctor
    必然看到端口被占——那是**健康态**，旧实现却无条件打 ⚠。假红灯与假绿灯一样有害，
    它训练用户忽略警告，而 ADR-0027 §3 的整条契约就是"灯要说真话"。所以占用时先探
    `/ready`：应答的是我们自己的 proxy（正常），不应答才是真冲突（start 会失败）。
    """
    bind_host = host if host != "0.0.0.0" else "127.0.0.1"  # noqa: S104
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        rc = s.connect_ex((bind_host, port))
        s.close()
    except OSError as e:
        return "error", str(e)
    # 批 W33：只看 == 0 时，沙箱里的 EPERM 也算「没人占」，11-05 打出 `Port (38080 free)`。
    if rc in (errno.EPERM, errno.EACCES):
        return "undetermined", f"{port}: {_UNDETERMINED_TEXT}"
    if rc != 0:
        return "free", f"{port} free"
    if _probe_ready(f"http://{bind_host}:{port}/ready") is not None:
        pid = _read_pidfile(_PROXY_PIDFILE)
        return "ours", f"{port} serving this proxy" + (f" (PID {pid})" if pid else "")
    return ("foreign",
            f"port {port} is in use by something that is not a BladeX proxy "
            f"(`bladex start` will fail -- free the port or set BLADEX_PORT)")


@app.command()
def doctor(
    warmup: bool = typer.Option(False, "--warmup", help="Pre-download / load the embedding model (first run can be slow)"),
    e2e: bool = typer.Option(False, "--e2e",
                             help="End-to-end memory loop check: send a probe turn, wait for it "
                                  "to be distilled, then recall it semantically"),
    e2e_timeout: float = typer.Option(180.0, "--e2e-timeout",
                                      help="Seconds to wait for distillation (--e2e)"),
) -> int:
    """Health check: redis / RocksDB / embedding model / upstream / port."""
    from bladex_proxy.config import ProxyConfig
    cfg = ProxyConfig()
    failures = 0
    # 批 W33：受限环境里探测给的不是读数。Redis / 上游 / 端口标 ?，不打 ✗，也不打 ✓。
    restriction = _shell_restriction()
    if restriction is not None:
        print(f"  ? Runtime checks: {_UNDETERMINED_TEXT}: {restriction}")

    def _check(label: str, ok: bool, detail: str = "", fix: str = "") -> None:
        nonlocal failures
        mark = "✓" if ok else "✗"
        print(f"  {mark} {label}" + (f" ({detail})" if detail else ""))
        if not ok:
            failures += 1
            if fix:
                print(f"    fix: {fix}")

    _check("config/.env", Path("config/.env").is_file(),
           fix="bladex init")

    # 批 BD · 步骤 1：账本块等长填充是 bench 专用——开着就标 `!`（不计失败，但不许被带进真实使用）。
    from bladex_core.flags import bench_ledger_filler_chars
    _bench_n = bench_ledger_filler_chars()
    if _bench_n > 0:
        print(f"  ! BENCH switch active: ledger block is replaced by {_bench_n} chars of constant "
              "filler text.")
        print("    fix: remove it from config/.env (and your shell) and restart; "
              "it is for the BD control-arm rehearsal only.")

    # 批 Z · Z2.1（MQ-A34 ①）：本地模型没配 context_window 只**提示**，不替用户填——
    # 窗口是硬件常量（`ollama show <model>` 的 num_ctx），BladeX 不猜、不探测。
    _rc = getattr(cfg, "routing_config", None)
    _no_window = (_rc.local_models_without_context_window()
                  if _rc is not None and hasattr(_rc, "local_models_without_context_window") else [])
    if _no_window:
        print(f"  ! Local models without context_window: {', '.join(_no_window)}")
        print("    hint: add `context_window = <tokens>` (e.g. from `ollama show <model>` num_ctx) to "
              "those [[models]] in config/routing.toml so BladeX can route around them when a "
              "request does not fit and trim its own injected surface to the budget; "
              "without it nothing changes for that model.")

    # 批 AE：旧 `routing.toml [embedding]` 的迁移提示。只说要搬哪几行，不改用户的文件；
    # 新文件写错时不在这里报，下面 embedding 那一条会以失败项报出来。
    from bladex_proxy.global_config import migration_hint
    from bladex_proxy.routing_config import RoutingConfigError
    try:
        _migration = migration_hint(cfg)
    except RoutingConfigError:
        _migration = []
    if _migration:
        print(f"  ! {_migration[0]}")
        for _line in _migration[1:]:
            print(f"    {_line}")

    # 🔴 MQ-A41：模型面资产在位（system notes 三份 + 账本模板）。
    # 缺了不致命但是**降级**：`load_system_notes()` 返回 "" ⇒ 自我介绍整块不注 ⇒
    # 模型不知道 BladeX 存在、不知道有工具面。G7 冷安装实跑时这件事只在启动日志里
    # 有两条 warning，doctor 全绿 —— 判据比承诺窄，所以补这一条。
    _missing_assets = [dst for _src, dst in _ASSET_TARGETS if not Path(dst).is_file()]
    _check("Model-facing assets", not _missing_assets,
           "all present" if not _missing_assets
           else f"{len(_missing_assets)} missing: {', '.join(_missing_assets)}",
           fix=("bladex init copies them from the package. Without config/system/*.md the "
                "self-introduction is not injected at all -- the model will not know BladeX "
                "exists or that it has tools."))

    hidden_pth = _hidden_pth_files()
    _check("Python path files (.pth)", not hidden_pth,
           f"{len(hidden_pth)} hidden .pth in site-packages" if hidden_pth else "",
           fix=("macOS marked them hidden and Python 3.11+ SILENTLY skips hidden .pth files, "
                "so installed packages become unimportable. Run: chflags nohidden "
                "\"$(python -c 'import site;print(site.getsitepackages()[0])')\"/*.pth "
                "-- if the flag comes back, the folder is on iCloud/OneDrive: move the "
                "project (and its virtualenv) outside the synced folder."))

    _refused_n, _refused_detail = _flash_refused_text()
    _check("Flash direct edits", not _refused_n, _refused_detail,
           fix="fix those files under the flash path, or undo the edit; "
               "BladeX keeps your file and pauses that ledger's projection until then")

    if restriction is not None:
        print(f"  ? Redis ({cfg.redis_url}): {_UNDETERMINED_TEXT}")
    else:
        _check("Redis", _redis_ping(cfg.redis_url), cfg.redis_url,
               fix="`bladex start` launches a local Redis; or run redis-server --appendonly yes yourself")

    # AOF manifest 对账（2026-09-19 事故）：manifest 列的文件缺一个 ⇒ redis 加载 AOF
    # 失败即退出，而 start 只说得出 "failed to start"。这条在 redis 起不来**之前**
    # 就点名是哪个文件不在——判据不能窄于被检查的对象。
    _aof_problem = aof_manifest_problem()
    _check("Redis AOF manifest", _aof_problem is None,
           _aof_problem or "consistent with data/appendonlydir/",
           fix=" / ".join(_AOF_REPAIR_HINT))

    # 🔴 MQ-A90：data/ 上只许有一个 redis-server，且就是配置里那个端口。
    # 第二个实例平时不出声，关机时才把自己那份过期 manifest 写回盘。
    _owners = redis_servers_on_data_dir()
    _cfg_port = _redis_hostport(cfg.redis_url)[1]
    # 🔴 批 W33（MQ-W37）：读不到端口 ≠ 端口不同。11-05 沙箱里 `port ?` 被判成游离实例，
    # 修复提示是 `kill 17932`，而 17932 正是生产 Redis。读不到端口的进程不算游离，也不给 kill。
    _stray = [o for o in _owners if o.port is not None and o.port != _cfg_port]
    _unread = [o for o in _owners if o.port is None]
    if len(_owners) == 1 and _unread:
        print(f"  ? Redis servers on data/ ({_owners[0].describe()}): port could not be read, "
              f"so this cannot be checked. Check it with: "
              f"lsof -nP -a -p {_owners[0].pid} -iTCP -sTCP:LISTEN")
    else:
        _steps = [f"redis-cli -p {o.port} shutdown" for o in _stray or _owners[1:]
                  if o.port is not None]
        _steps += [f"lsof -nP -a -p {o.pid} -iTCP -sTCP:LISTEN (port unreadable; check it first)"
                   for o in _unread]
        _check("Redis servers on data/", len(_owners) <= 1 and not _stray,
               ", ".join(o.describe() for o in _owners) or "none running",
               fix=("more than one redis-server (or one on a port BladeX is not configured for) is "
                    "using data/; they overwrite each other's AOF manifest on shutdown. Next: "
                    + "; ".join(_steps)))

    # Memory Hub RocksDB：secondary 只读探测（不与运行中的 proxy 抢主锁）
    ledger_dir = Path(cfg.rocksdb_path)
    if not ledger_dir.exists():
        _check("Memory Hub RocksDB", True, "no data yet (created on first start)")
    else:
        try:
            from bladex_proxy.storage.memory_hub import MemoryHub
            hub = MemoryHub(cfg.rocksdb_path, secondary=True)
            hub.open()
            n = hub.count()
            hub.close()
            _check("Memory Hub RocksDB", True, f"{n} turns")
        except Exception as e:  # noqa: BLE001
            _check("Memory Hub RocksDB", False, str(e),
                   fix="directory corrupted or version mismatch; back up, then check permissions under data/")

    # embedding 模型（盖章对账 + 按档分流，见 _doctor_embedding）
    try:
        _doctor_embedding(cfg, warmup, _check)
    except Exception as e:  # noqa: BLE001
        _check("Embedding", False, str(e),
               fix="check BLADEX_EMBED_* and the [embedding] section of config/bladex.toml; set HF_ENDPOINT for a mirror if the network is blocked")

    # 上游（ADR-0027 §3.1 契约 A）：带 key 做一次真实鉴权探测，不再只测网络层。
    # 旧行为是 urlopen(api_base) 且 api_base 为空直接打 ✓ —— 空 key 也是绿灯，
    # 而这正是新用户最常见的失败原因（冷启动实测）。
    state, detail, fix = (("not_probed", _UNDETERMINED_TEXT, "") if restriction is not None
                          else _probe_upstream(cfg))
    if state == "ok":
        _check("Upstream", True, detail)
    elif state == "not_probed":
        # 探不了就如实说"未探测"，不打 ✓ —— 绿灯不许骗人。
        print(f"  · Upstream: not probed ({detail})")
    else:
        _check("Upstream", False, detail, fix=fix)

    # 端口
    state, detail = _probe_port(cfg.host, cfg.port)
    if state == "free":
        _check("Port", True, detail)
    elif state == "ours":
        _check("Port", True, detail)
    elif state == "foreign":
        print(f"  ⚠ {detail}")
    elif state == "undetermined":
        print(f"  ? Port ({detail})")
    else:
        _check("Port", False, detail)

    if restriction is not None:
        print(f"Health check incomplete: {restriction}. Run `bladex doctor` from a normal terminal.")
        return RC_UNDETERMINED
    if failures:
        print(f"Health check found {failures} problem(s).")
        if e2e:
            print("  (fix the problems above before running --e2e)")
        return 1
    print("Health check passed.")

    if e2e:
        return _run_e2e_check(cfg, timeout_s=e2e_timeout)
    return 0


def _doctor_embedding(cfg, warmup: bool, check) -> None:  # noqa: ANN001
    """doctor 的 embedding 段：**先**盖章对账，**再**按档分流（MQ-W14）。

    对账放在 api/local 分支**之外**、且在分流**之前**，与 `_preflight_embedding` 同序同判据：
    `expected = embed_model_identity(cfg, role="proxy")`（唯一来源——内部已处理敏感层
    fail-closed 强制回本地与 proxy 档的 role 折算），对 `<index>/embed_model_id` 侧车。
    两者都非空且不等 ⇒ 报，并**跳过分流**（与 preflight 一样，章不对时不去探活/查缓存）。

    为什么必须在分支之外（2026-09-18 复核真跑）：local 分支从来不对账（只 `_model_cached`），
    ⇒ 纯 local + 章不一致 = doctor ✓ / start ✗；W12 把"敏感层 + api"挪进 local 分支后，
    该形态也从"报错误的警"变成"不报警"。**doctor 说健康、start 说起不来** ——
    ADR-0027 §2「红灯不许骗人」的镜像面：绿灯也不许骗人。

    🔴 只对 local / api 两档对账；proxy / ipc 的身份要握手后才知道，不作负面判定
    （与 preflight「空身份不作负面判定」同则）。
    """
    from bladex_proxy.embedding import (
        _local_model_for,
        _model_cached,
        _sensitivity_enabled,
        embed_model_identity,
        resolve_embed_settings,
    )
    settings = resolve_embed_settings(cfg)
    if settings.backend == "api" and _sensitivity_enabled(cfg):
        # MQ-W12：敏感层 fail-closed 在运行时把 api 档强制回本地
        # （build_embedder / embed_model_identity 同判据）。doctor 按**生效**
        # 形态走 local 分支：不外呼探活（敏感层的语义就是不外发）。
        print("  · Embedding: backend=api is overridden by the sensitivity layer "
              "(fail-closed forces local) -- checking the effective local backend")
        settings.backend = "local"
        settings.model = _local_model_for(settings, forced=True)

    if settings.backend in ("local", "api"):
        expected = embed_model_identity(cfg, role="proxy")
        sidecar = Path(getattr(cfg, "index_path", "") or "") / "embed_model_id"
        stored = sidecar.read_text(encoding="utf-8").strip() if sidecar.is_file() else ""
        if stored and stored != expected:
            check("Embedding", False,
                  f"Memory Index embed model mismatch: store={stored!r} current={expected!r} "
                  "-- `bladex start` will refuse to start like this",
                  fix="keep your data: revert the [embedding] config (backend/model) to the "
                      "stamped engine; or switch engines for real: `python scripts/reembed_index.py` "
                      "(full re-embed; with backend=api this sends ALL memory content to the provider)")
            return

    if settings.backend == "local":
        model = settings.model or "(default)"
        cached = _model_cached(settings.model, cfg.fastembed_cache_path) if settings.model else False
        if warmup:
            from bladex_proxy.embedding import build_embedder
            print(f"  … loading embedding model {model} (downloaded automatically if missing; please wait)")
            emb = build_embedder(cfg, role="proxy")
            emb.embed(["warmup"])
            check("Embedding", True, f"local/{model} ready")
        else:
            check("Embedding", True,
                  f"local/{model} {'cached' if cached else 'not cached (downloaded on first start, or run doctor --warmup)'}")
    elif settings.backend == "api":
        _doctor_embed_api(cfg, settings, warmup, check)
    else:
        check("Embedding", True, f"{settings.backend} backend (remote; no local cache check)")


def _doctor_embed_api(cfg, settings, warmup: bool, check) -> None:  # noqa: ANN001
    """G18.4：`backend=api` 的 doctor 路径（MQ-W9 断点④）。

    三条失败各自点名配置（模型名空 / key 声明了但解析为空 / 与已盖章库不同源），
    成功路径报一次真实探活的维度与 RTT。`--warmup` 在本档下没有东西可下载，
    明说跳过而不是装作做了。
    """
    from bladex_proxy import router_sdk
    from bladex_proxy.global_config import embedding_section

    if warmup:
        print("  · Embedding warmup: nothing to download for backend=api "
              "(the probe below hits the remote endpoint instead)")
    if not settings.model:
        check("Embedding", False, "backend=api but no embedding model configured",
              fix='set [embedding].model in config/bladex.toml (e.g. '
                  '"openai/text-embedding-3-small") or BLADEX_EMBED_MODEL')
        return
    key_env = embedding_section(cfg).config.api_key_env
    if key_env and not settings.api_key:
        check("Embedding", False,
              f"api key env {key_env!r} is declared in the [embedding] config but resolves to empty",
              fix=f"export {key_env}=<key> in the environment that runs bladex "
                  "(keys never go into config files themselves)")
        return

    try:
        dim, rtt_ms = _embed_api_probe(settings)
    except Exception as exc:  # noqa: BLE001 — 探活失败要指配置，不能让整段 doctor 崩
        check("Embedding", False, router_sdk.error_text(exc),
              fix="check [embedding] model / api_base and the api key; the model name "
                  "must be a Router embedding id like provider/model")
        return

    # 盖章对账不在这里做——已提到 `_doctor_embedding` 的分流之前（MQ-W14），
    # 全 doctor 只许有那一处判定；能走到这里说明章已对上（或还没盖章）。
    check("Embedding", True, f"embed_api_probe_ok dim={dim} rtt_ms={rtt_ms:.0f}")


# 探针事实：内容要足够特异，语义检索能唯一命中；带随机串避免与历史探针混淆。
#
# 🔴 三条硬约束，全是 2026-08-05 首次实跑（G4）用失败换来的。探针要穿过一个 LLM 改写
#    和一层判重，能活下来的形状比看上去窄得多：
#
# ① **token 必须是句子的语义载荷，不能是修饰语**。"my probe passphrase is {token}"
#    里 token 是被陈述的值，蒸馏成中文仍保留；改成 "the probe named {token} has the
#    favourite colour chartreuse" 后 token 降级为定语，模型直接把它省掉了
#    （实测：该轮 new_facts=1，但库里搜不到任何含 token 的 fact）。
# ② **整句每轮必须不同**，否则被 novelty 判重吃掉。原模板的颜色子句是常量，
#    第二次跑就 `candidates=2 new_facts=1 existing=1`，库里积着一堆无 token 的
#    "favourite probe colour is chartreuse"。token 在句中 → 每轮天然新颖。
# ③ **只写一句、只说一件事**。蒸馏器按"原子事实"拆分，两个子句会被拆成两条 fact，
#    token 只跟着其中一条走——而 step 4 问的是另一条，出的题和判分标准就对不上了。
#    颜色那半句纯属多余，这一连串麻烦全是它引出来的。
_E2E_TEMPLATE = (
    "Please remember this for later: my BladeX end-to-end probe passphrase is {token}."
)


# step 4 的提问：fact 的**语义改写**，且**不含答案**——问句里带 token 就退化成关键词
# 查找，测不到检索这条腿。历史探针事实的干扰由"通过后自动清理"消掉（见 step 5），
# 不靠问句去规避。
_E2E_RECALL_QUERY = "what is my BladeX end-to-end probe passphrase?"


def _run_e2e_check(cfg, timeout_s: float = 180.0) -> int:
    """端到端记忆闭环自检。

    为什么需要这一枪（ADR-0027 §3.2）：记忆从产生到可感知要穿过
    `入库（Redis）→ consolidator 活着 → 蒸馏 LLM 调用成功 → fact 落 Memory Index → 语义命中`
    这条长链，**每一环都可能安静断掉，而 agent 界面上看到的都是"正常回答"**。
    分项体检各自都能绿，链路仍可能是断的。这里真的走一遍，把静默失效变成显式失败。
    """
    token = f"bx{secrets.token_hex(4)}"
    base = f"http://{'127.0.0.1' if cfg.host == '0.0.0.0' else cfg.host}:{cfg.port}"  # noqa: S104
    client_key = _first_client_key()
    print("\n=== Memory loop check (--e2e) ===")
    print(f"  probe token: {token}")

    # ① 发一轮真实对话过完整管线
    t0 = time.monotonic()
    status_code, body = _http_request(
        "POST", f"{base}/v1/chat/completions", key=client_key,
        body={"model": cfg.upstream_model or "default",
              "messages": [{"role": "user", "content": _E2E_TEMPLATE.format(token=token)}],
              "stream": False},
        timeout=120.0)
    if status_code != 200:
        msg = (body.get("error") or {}).get("message") or body
        print(f"  ✗ step 1 forwarding failed (HTTP {status_code}):{msg}")
        print("    Stuck at: proxy -> upstream. Check the upstream key / quota / model name (`bladex config check`).")
        return 1
    print(f"  ✓ step 1 forwarded ({time.monotonic() - t0:.1f}s)")

    # ② turn 是否入库（Pipeline → Memory Hub）。入不了库，后面全都不可能发生。
    stored = False
    for _ in range(15):
        # MQ-A91：/admin/turns 是全 Hub scan_meta 再排序，17.7k 轮 ≈ 4 s；默认 3 s 超时
        # 会把"已入库"读成 None ⇒ 本步假红。超时放宽只修判读，端点的 O(N) 另账。
        idx = _http_json(f"{base}/admin/turns?limit=20", key=_admin_key(), timeout=30.0)
        if idx:
            for item in idx.get("turns", []):
                from urllib.parse import quote
                d = _http_json(f"{base}/admin/turns/{quote(item['key'])}?include_messages=true",
                               key=_admin_key())
                if d and token in json.dumps(d.get("request_messages", ""), ensure_ascii=False):
                    stored = True
                    break
        if stored:
            break
        time.sleep(2.0)
    if not stored:
        print("  ✗ step 2 not stored -- this turn will produce no memory")
        print("    Stuck at: Pipeline/Memory Hub write. Is Redis reachable (`bladex status`)? "
              "Grep the log for enqueue_skipped_no_redis / pipeline_skipped_no_redis.")
        return 1
    print(f"  ✓ step 2 stored in Memory Hub ({time.monotonic() - t0:.1f}s)")

    # ③ consolidator 是否把它蒸成 fact（这一步最慢，也最容易静默停滞）
    deadline = time.monotonic() + timeout_s
    hit = None
    while time.monotonic() < deadline:
        facts = _http_json(f"{base}/admin/facts?q={token}&limit=5", key=_admin_key())
        for f in (facts or {}).get("facts", []):
            if token in (f.get("content") or ""):
                hit = f
                break
        if hit:
            break
        time.sleep(5.0)

    if hit is None:
        elapsed = time.monotonic() - t0
        print(f"  ✗ step 3 no fact distilled within {int(timeout_s)}s (waited {elapsed:.0f}s)")
        cons_pid = _consolidator_pid()
        if cons_pid is None:
            print("    Stuck at: the consolidator is not running -- `bladex start` launches it; "
                  "see logs/consolidator-*.log for why it failed.")
        else:
            print(f"    Stuck at: the consolidator (PID {cons_pid}) is running but produced nothing. "
                  "Common causes: the distillation model is unavailable or out of quota (grep the log for distill), "
                  "the catch-up lag is too large (see lag_turns in `bladex status`), "
                  "or this turn was judged to carry no memory value.")
        return 1
    print(f"  ✓ step 3 distilled into a Fact (id={hit.get('id', '?')[:12]}, "
          f"{time.monotonic() - t0:.1f}s)")

    # ④ 语义召回：用自然语言问句（不是拿 fact 原文精确匹配），验证的是"检索这条腿"
    from urllib.parse import quote as _quote
    recall = _http_json(
        f"{base}/admin/facts?q={_quote(_E2E_RECALL_QUERY)}&limit=10",
        key=_admin_key())
    recalled = any(token in (f.get("content") or "")
                   for f in (recall or {}).get("facts", []))
    if not recalled:
        print("  ✗ step 4 semantic recall missed it (the fact exists but the query did not return it)")
        print("    Stuck at: retrieval. Check, in this order: "
              "(a) is the embedding backend the same model that wrote these vectors? "
              "(changing models requires a full re-embed via scripts/reembed_index.py); "
              "(b) is the fact's content actually about what was asked -- "
              f"`bladex memory show {hit.get('id', '?')}`; "
              "(c) is the top-10 crowded out by low-value facts -- "
              f"`bladex memory search \"{_E2E_RECALL_QUERY}\" --k 10`.")
        return 1

    total = time.monotonic() - t0
    print(f"\n  ✓ Memory loop OK: the probe fact was distilled and semantically recalled ({total:.1f}s)")

    # ⑤ 清理：探针是测试数据，不是用户的记忆。留着会一轮轮堆积、互相竞争排名，
    #    还会挤占真实记忆的召回位（C13 冷安装那一轮会有人反复跑 doctor）。
    #    只在**通过后**清理——失败时留着让人能 `bladex memory show` 现场取证。
    fact_id = hit.get("id", "")
    status_code, _ = _http_request("DELETE", f"{base}/admin/facts/{fact_id}",
                                   key=_admin_key(), timeout=10.0)
    if status_code in (200, 202):
        print(f"    Cleaned up the probe fact ({fact_id[:12]}, tombstoned -- never revived on rebuild)")
    else:
        print(f"    Note: could not clean up the probe fact ({fact_id}) -- "
              f"HTTP {status_code}; remove it with `bladex memory forget {fact_id}`")
    return 0
