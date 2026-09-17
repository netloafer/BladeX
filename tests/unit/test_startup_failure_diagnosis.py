"""启动失败要**当场说清、并且不留孤儿**（MQ-L61 / MQ-L62，2026-09-08）。

## 三条各自的来历（同一次实测，Jason 故意把 glm-5.3-flash 配成 glm-5.4-flash 验收 MQ-L58）

    [warning] index_init_failed error='routing.toml 里有策略引用了…'
              hint='proxy will run without Memory Index derived layer'
    ...
    state: failed -- the proxy process is up but /ready did not answer within 30s
    $ bladex status
      Proxy: ✗ not running   Consolidator: ✗   Embedding: ✗
      Memory Flash: ✓ running (PID 60572)          ← 孤儿

- **MQ-L61**：配置错误从 `build_embedder` 冒出来，被 Index 初始化那层
  `except Exception` **降级**成"索引层不可用"。要不是 `init_router` 也读同一份配置，
  proxy 会带着"没有 Memory Index"正常起来 —— 与 MQ-L58 同形：告警有、严重性没有。
- **MQ-L62 ①**：进程 3 秒就退出了，CLI 却轮询满 30 秒，最后说 `the proxy process is up`
  —— 这句是错的，而且把人引向就绪/网络而不是配置。
- **MQ-L62 ②**：proxy 没起来，flash daemon 却留着继续写 Flash 投影。
"""

from __future__ import annotations

from pathlib import Path

import bladex_proxy.cli.lifecycle as L

# ── MQ-L61：配置错误不许被降级成"索引不可用" ────────────────────────────────


def test_routing_config_error_is_reraised_before_the_index_degrade_path():
    """Index 初始化那块必须**先**放行 `RoutingConfigError`，再谈降级。

    🔴 **这是一条源码顺序守卫，不是行为断言** —— 说清楚免得下一棒高估它
    （A53 的教训：判别力对照证明"我写的那条路会红"，证明不了"我没写的那条路不会静默"）。
    要做成行为版得跑整个 `lifespan`（Redis + Hub + 池加载都在里面），
    代价远超它守的东西；真正的行为证据是 09-08 那次实测本身。

    守两件：① `except RoutingConfigError` 出现在 `except Exception` **之前**
    （Python 按书写顺序匹配，顺序颠倒 = 修法失效且看不出来）；
    ② 那一支是 `raise`，不是又一个 `logger.warning`。
    """
    import inspect

    import bladex_proxy.server.app_factory as A
    from bladex_proxy.routing_config import RoutingConfigError

    assert A.RoutingConfigError is RoutingConfigError, "app_factory 没引入这个类型"
    src = inspect.getsource(A.lifespan)
    i_cfg = src.find("except RoutingConfigError")
    i_any = src.find("except Exception as e:\n        logger.warning(\"index_init_failed\"")
    assert i_cfg != -1, "配置错误没被单独接住 ⇒ 仍会掉进 except Exception 降级"
    assert i_any != -1, "Index 降级路径不见了 —— 它是有意的（运行时故障仍要降级），别一起删"
    assert i_cfg < i_any, "顺序反了：Python 按书写顺序匹配，Exception 在前就永远轮不到它"
    branch = src[i_cfg:i_any]
    assert "raise" in branch and "logger.warning" not in branch, \
        "这一支必须原样上抛 —— 再打一条 warning 就是把同一个缺陷换了个名字"


# ── MQ-L62 ①：进程死了就别再等就绪 ─────────────────────────────────────────


def _no_ready(*_a, **_k):
    return None


def test_dead_proxy_stops_polling_immediately(monkeypatch, tmp_path, capsys):
    """`proxy_pid` 已经不在 ⇒ 立刻收工，**不等满 timeout**，且不说 "process is up"。"""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "proxy-20260908-150216.log").write_text(
        "INFO:     Started server process [60574]\n"
        "ERROR:    Traceback (most recent call last):\n"
        "bladex_proxy.routing_config.RoutingConfigError: unresolved model ref\n",
        encoding="utf-8")
    monkeypatch.setattr(L, "_probe_ready", _no_ready)
    monkeypatch.setattr(L, "_pid_alive", lambda _pid: False)
    slept: list[float] = []
    monkeypatch.setattr(L.time, "sleep", lambda s: slept.append(s))

    rc = L._report_startup_state("127.0.0.1", 38080, True, True,
                                 timeout_s=30.0, proxy_pid=60574)
    out = capsys.readouterr().out
    assert rc == 1
    assert slept == [], "进程已死还在 sleep 轮询 —— 30 秒白等就是从这里来的"
    assert "exited during startup" in out
    assert "process is up" not in out, "这句是错的：进程并没有 up"
    # 报错原样摆到终端，用户不必再去翻日志
    assert "RoutingConfigError" in out
    assert out.isascii(), "面向用户的输出必须是英文（2026-09-08 Jason 定的标准）"


def test_alive_but_not_ready_keeps_the_timeout_wording(monkeypatch, tmp_path, capsys):
    """阳性对照：进程**活着**但不就绪 ⇒ 仍是超时语义（这条路没被改坏）。

    没有它，上一条也能靠"永远走 died 分支"通过 —— 那会把"真·卡住"误报成"退出了"。
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    monkeypatch.setattr(L, "_probe_ready", _no_ready)
    monkeypatch.setattr(L, "_pid_alive", lambda _pid: True)
    ticks = {"n": 0}

    def _fake_sleep(_s):
        ticks["n"] += 1
        if ticks["n"] > 3:                       # 快进到 deadline
            monkeypatch.setattr(L.time, "monotonic", lambda: 1e9)

    monkeypatch.setattr(L.time, "sleep", _fake_sleep)
    rc = L._report_startup_state("127.0.0.1", 38080, True, True,
                                 timeout_s=3.0, proxy_pid=60574)
    out = capsys.readouterr().out
    assert rc == 1 and "alive but /ready did not answer" in out
    assert "exited during startup" not in out


# ── MQ-L62 ②：只回收**本次拉起的**旁挂进程 ─────────────────────────────────


def test_rollback_stops_only_what_this_run_started(monkeypatch, capsys):
    """本次新起的停掉；**进来之前就在跑的一律不碰**。

    判据是 PID 文件的值变了，不是"我调用过 _start_x" —— 那三个 helper 在进程已在跑时
    是空操作，按调用判会去杀用户自己起的守护进程。停错东西没有再说的机会
    （与 `cleanup --ledgers` 的 fail-closed 安全阀同一条立论）。
    """
    sidecars = (("Embedding", "run/embed.pid"),
                ("Consolidator", "run/cons.pid"),
                ("Flash daemon", "run/flash.pid"))
    before = {"Embedding": 111, "Consolidator": None, "Flash daemon": None}
    now = {"run/embed.pid": 111,      # 之前就在跑，没变 ⇒ 不碰
           "run/cons.pid": 222,       # 本次新起 ⇒ 停
           "run/flash.pid": 333}      # 本次新起 ⇒ 停
    stopped: list[str] = []
    monkeypatch.setattr(L, "_read_pidfile", lambda p: now.get(p))
    monkeypatch.setattr(L, "_stop_by_pidfile", lambda p, name: stopped.append(name.strip()))

    L._rollback_sidecars(before, sidecars)
    assert stopped == ["Consolidator", "Flash daemon"], \
        "Embedding 是进来之前就在跑的，不该被回收"
    out = capsys.readouterr().out
    assert "stopping the sidecars this run started" in out
    assert out.isascii(), "面向用户的输出必须是英文（2026-09-08 Jason 定的标准）"


def test_rollback_is_silent_when_nothing_new_was_started(monkeypatch, capsys):
    """全都是"之前就在跑" ⇒ 一句话都不打（别在失败输出里加噪声）。"""
    sidecars = (("Embedding", "run/embed.pid"),)
    monkeypatch.setattr(L, "_read_pidfile", lambda _p: 111)
    monkeypatch.setattr(L, "_stop_by_pidfile",
                        lambda *_a: pytest_fail("不该停任何东西"))
    L._rollback_sidecars({"Embedding": 111}, sidecars)
    assert capsys.readouterr().out == ""


def pytest_fail(msg: str):
    raise AssertionError(msg)


# ── 日志尾巴抽取 ───────────────────────────────────────────────────────────


def test_error_tail_starts_at_the_last_error_marker(tmp_path):
    p = tmp_path / "proxy.log"
    p.write_text("INFO: a\nERROR:    first\nboom1\nINFO: b\nERROR:    second\nboom2\n",
                 encoding="utf-8")
    tail = L._log_error_tail(str(p))
    assert "second" in tail and "boom2" in tail
    assert "first" not in tail, "取的是**最后**一段错误，不是第一段"


def test_error_tail_falls_back_to_the_last_lines(tmp_path):
    """没有错误标记时不许返回空 —— 空 = 用户什么线索都拿不到。"""
    p = tmp_path / "proxy.log"
    p.write_text("\n".join(f"line{i}" for i in range(10)), encoding="utf-8")
    assert "line9" in L._log_error_tail(str(p))


def test_error_tail_on_a_missing_file_is_empty_not_an_exception(tmp_path):
    """诊断代码自己不许炸 —— 它跑在"已经失败了"的路径上。"""
    assert L._log_error_tail(str(tmp_path / "nope.log")) == ""


def test_newest_proxy_log_picks_the_latest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    old = tmp_path / "logs" / "proxy-1.log"
    new = tmp_path / "logs" / "proxy-2.log"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    import os
    os.utime(old, (1, 1))
    assert Path(L._newest_proxy_log()).name == "proxy-2.log"


def test_newest_proxy_log_without_logs_dir_is_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert L._newest_proxy_log() == ""
