"""M4-2 + MS-8：记忆漏斗指标 + 库存健康度（两个独立端点）。

对应主卡 M4-2（D7/E8/F6/G7/H7/I5 六连）与补充卡 MS-8。

## 这组测试守什么

漏斗存在的理由是复核里反复出现的同一句话：**机制有没有在跑，此前只能靠翻日志猜**。
判重杀了多少、放行了多少（E8）、注入命中回写有没有信号（G7/I5）——全部无指标，
于是 E1/E2 那种"门被焊开"的状态在观测面上完全不可见。

所以这里钉的不是"某个数字对不对"，而是三件结构性的事：

1. **指标名只有一处定义**（`bladex_core.funnel`）——两个进程各自暴露端点，
   名字对不上就拼不出一条漏斗，而"两边名字不一样"没人会主动发现；
2. **默认无操作**——没接指标时调用点不该崩，也不该逼调用方到处判空；
3. **写入侧端点默认关且只听回环**——它是不鉴权的观测面。
"""

from __future__ import annotations

import pytest
from bladex_core import funnel as F
from bladex_proxy.metrics import Metrics


# ── 1. 共享词汇表：名字只许有一处定义 ─────────────────────────────────────


def test_funnel_covers_the_whole_chain():
    """漏斗六段齐全，且顺序就是判读顺序（哪段为 0 而上一段非 0 = 那段断了）。

    2026-09-03 S1：读取侧 RECALL / INJECT 随主动注入检索路径删除（八段 → 六段）。"""
    assert F.FUNNEL_ORDER == F.ALL_FUNNEL_METRICS
    assert len(F.FUNNEL_ORDER) == 6
    # 写入侧五段 + 读取侧一段（HIT）
    assert F.FUNNEL_CANDIDATES in F.FUNNEL_ORDER[:5]
    assert F.FUNNEL_HIT in F.FUNNEL_ORDER[5:]


def test_metric_names_are_defined_in_core_only():
    """两端不得各自造名字——共享常量只许有一处定义（同 flags.py 纪律）。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    for rel in ("packages/bladex-proxy/bladex_proxy/inject.py",
                "packages/bladex-core/bladex_core/consolidation_proxy.py"):
        src = (root / rel).read_text(encoding="utf-8")
        assert "bladex_memory_" not in src, (
            f"{rel} 里出现了字面量指标名 —— 必须从 bladex_core.funnel 引用"
        )


def test_all_names_share_one_prefix():
    for name in F.ALL_FUNNEL_METRICS + F.ALL_LIBRARY_METRICS:
        assert name.startswith("bladex_memory_")


# ── 2. NullFunnel：没接指标时不崩、调用方不必判空 ─────────────────────────


def test_null_funnel_is_a_noop():
    F.NULL_FUNNEL.inc(F.FUNNEL_DISTILL, labels={"outcome": "ok"})
    F.NULL_FUNNEL.set_gauge(F.LIB_FACTS, 1.0)


def test_consolidator_without_funnel_still_works():
    """不传 funnel 的构造路径（proxy / 测试 / 第三方）行为不变。"""
    from bladex_core.consolidation_proxy import ProxyConsolidator

    c = ProxyConsolidator(embedder=None)
    assert c._funnel is F.NULL_FUNNEL
    assert c.consolidate_turns([]) == []


# ── 3. 写入侧漏斗真的被打点 ───────────────────────────────────────────────


class _Emb:
    """确定性 one-hot 桩 embedder。

    🔴 2026-08-06 修 flaky：原实现是 `v[abs(hash(t)) % 16] = 1.0`。
    Python 的**字符串 hash 每进程随机化**（PYTHONHASHSEED），于是两条不同文本
    有 1/16 ≈ 6% 的概率落进同一个桶 → cosine 1.0 → 第二条被判重 →
    `dedup{novel}` 少一条、断言随机翻红。实测全量跑两次挂一次。

    危害不在这条测试本身，在于它是 M1/M2/M5 三个 gate 依赖的**验收仪器**：
    一条 6% 概率假红的测试会让后续每一次"漏斗有数"的结论都不可信。
    改用 sha256（跨进程稳定）+ 16 维扩到 64 维降低碰撞面。
    """

    _DIM = 64

    def embed(self, texts):
        import hashlib

        out = []
        for t in texts:
            v = [0.0] * self._DIM
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            v[int.from_bytes(digest[:4], "big") % self._DIM] = 1.0
            out.append(v)
        return out


class _Distiller:
    """按内容决定产出：ok / 真空 / 失败——三种 outcome 各来一条。"""

    model_name = "stub"

    def _out(self, text):
        from bladex_core.distillation import DistillFact, DistillOutput

        if "失败" in text:
            return DistillOutput(model_name="stub", failed=True, failure_kind="parse")
        if "无事实" in text:
            return DistillOutput(model_name="stub")
        return DistillOutput(
            facts=[DistillFact(content=text[:80], kind="general")],
            matter_proposals=[], model_name="stub")

    def distill(self, text):
        return self._out(text)

    def distill_conclusion(self, text):
        return self._out(text)


#: 候选装配的最短长度是 20 字符（`_MIN_CONTENT_CHARS`）——短于它的消息
#: 在装配阶段就被滤掉，根本走不到蒸馏，测出来的会是"机制没打点"的假象。
#: 这个坑本轮已经踩过两次（M0-1、MS-1），所以这里显式断言而不是靠数字数。
def _turn(text: str, key: str):
    from bladex_core.consolidation_proxy import _MIN_CONTENT_CHARS
    from bladex_core.fact import ConversationTurn

    assert len(text) >= _MIN_CONTENT_CHARS, (
        f"测试输入 {len(text)} 字符 < 装配下限 {_MIN_CONTENT_CHARS}，会被静默滤掉"
    )
    return ConversationTurn(session_id="s", user_id="u", user_messages=[text],
                            ledger_key=key, logical_turn=1)


def _counter(m: Metrics, metric: str, **labels) -> float:
    """从 snapshot 取一个 counter 值。snapshot 形态：{metric: [(labels, value), ...]}。"""
    for lbls, value in m.snapshot()["counters"].get(metric, []):
        if all(lbls.get(k) == v for k, v in labels.items()):
            return value
    return 0.0


def test_distill_outcomes_are_counted_separately():
    """🔴 `empty` 与 `*_fail` 必须分开。

    "这轮真没事实" 与 "上游断了" 在降级实现里长得一模一样——那正是 2026-08-05
    事故里唯一缺的那条信息（consolidator 把积压烧成"已消费且零事实"，
    日志里没有一行说这些轮次是白烧的）。
    """
    from bladex_core.consolidation_proxy import ProxyConsolidator

    m = Metrics()
    c = ProxyConsolidator(embedder=_Emb(), distiller=_Distiller(), funnel=m)
    c.consolidate_turns([
        _turn("这是一条有内容的用户陈述，需要被长期记住并在后续复用", "k1"),
        _turn("这一条无事实，完全没有任何值得记下来的东西在里面呢", "k2"),
        _turn("这一条会失败，因为上游这次返回的是一段无法解析的垃圾内容", "k3"),
    ])
    assert _counter(m, F.FUNNEL_DISTILL, outcome="ok") == 1
    assert _counter(m, F.FUNNEL_DISTILL, outcome="empty") == 1
    assert _counter(m, F.FUNNEL_DISTILL, outcome="parse_fail") == 1


def test_candidates_and_dedup_and_stored_are_counted():
    """三段都要有数——任何一段为 0 而上一段非 0，就是那一段断了。"""
    from bladex_core.consolidation_proxy import ProxyConsolidator

    m = Metrics()
    c = ProxyConsolidator(embedder=_Emb(), distiller=_Distiller(), funnel=m)
    facts = c.consolidate_turns([
        _turn("用户决定把向量库换成新的引擎以便支持更大规模", "k1"),
        _turn("用户还决定把蒸馏模型换成更便宜的那一档以控制成本", "k2"),
    ])
    assert facts
    assert _counter(m, F.FUNNEL_CANDIDATES, channel="consolidation") >= 2
    assert _counter(m, F.FUNNEL_DEDUP, verdict="novel") >= 2
    assert _counter(m, F.FUNNEL_STORED, item_kind="assertion") >= 1


def test_duplicate_candidate_counts_as_dedup_kill():
    """判重杀掉的那条要留下痕迹（E8：门被焊开时观测面必须看得见）。"""
    from bladex_core.consolidation_proxy import ProxyConsolidator

    m = Metrics()
    c = ProxyConsolidator(embedder=_Emb(), distiller=_Distiller(), funnel=m)
    same = "用户决定把向量库换成新的引擎以便支持更大规模"
    c.consolidate_turns([_turn(same, "k1"), _turn(same, "k2")])
    assert _counter(m, F.FUNNEL_DEDUP, verdict="cosine_kill") >= 1


# ── 4. MS-8 库存健康度 ────────────────────────────────────────────────────


def test_library_kpis_shape(tmp_path):
    from bladex_core.fact import Fact, ItemKind
    from bladex_proxy.storage.memory_index import MemoryIndex

    idx = MemoryIndex(tmp_path / "idx", embedder=None)
    idx.open()
    idx.add_fact(Fact(id="f1", content="一条断言", item_kind=ItemKind.ASSERTION,
                      source_user_id="u"))
    idx.add_fact(Fact(id="f2", content="一条偏好", item_kind=ItemKind.PREFERENCE,
                      source_user_id="u"))
    k = idx.library_kpis()
    idx.close()

    assert k["facts_by_kind"]["assertion"] == 1
    assert k["facts_by_kind"]["preference"] == 1
    assert k["invalidated"] == 0
    assert 0.0 <= k["promotion_rate"] <= 1.0
    for key in ("provenance", "matters_by_status",
                "file_ref_in_vectors", "zombie_vectors"):
        assert key in k


def test_library_kpis_are_published_as_gauges(tmp_path):
    from bladex_core.fact import Fact, ItemKind
    from bladex_proxy.storage.memory_index import MemoryIndex

    m = Metrics()
    idx = MemoryIndex(tmp_path / "idx2", embedder=None)
    idx.open()
    idx.attach_funnel(m)
    idx.add_fact(Fact(id="f1", content="一条断言", item_kind=ItemKind.ASSERTION,
                      source_user_id="u"))
    idx.publish_library_kpis()
    idx.close()

    rendered = m.render()
    assert F.LIB_FACTS in rendered
    assert F.LIB_PROMOTION_RATE in rendered


# ── 5. consolidator 独立端点：默认关 + 只听回环 ──────────────────────────


def test_consolidator_metrics_disabled_by_default(monkeypatch):
    from bladex_proxy import consolidator_metrics as cm

    monkeypatch.delenv("BLADEX_CONSOLIDATOR_METRICS_PORT", raising=False)
    assert cm.metrics_port() == 0
    assert cm.start_metrics_server(Metrics()) is None


@pytest.mark.parametrize("raw,want", [
    ("0", 0), ("", 0), ("abc", 0), ("70000", 0), ("-1", 0), ("39090", 39090),
])
def test_consolidator_metrics_port_parsing(monkeypatch, raw, want):
    from bladex_proxy import consolidator_metrics as cm

    monkeypatch.setenv("BLADEX_CONSOLIDATOR_METRICS_PORT", raw)
    assert cm.metrics_port() == want


def test_consolidator_metrics_binds_loopback_only():
    """🔴 不鉴权的观测面绝不能听在 0.0.0.0（与 proxy bind 安全同一条纪律）。"""
    from bladex_proxy import consolidator_metrics as cm

    assert cm._BIND_HOST == "127.0.0.1"
    # 只看代码，不看注释——注释里出现 0.0.0.0 是在解释为什么**不能**用它
    src = __import__("pathlib").Path(cm.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    code = code.split('"""', 2)[-1]          # 去掉模块 docstring
    assert "0.0.0.0" not in code


def test_consolidator_metrics_serves_and_shuts_down():
    """真起一次服务并抓一次——端点本身也要被测，不能只测端口解析。"""
    import urllib.request

    from bladex_proxy import consolidator_metrics as cm

    m = Metrics()
    m.inc(F.FUNNEL_STORED, labels={"item_kind": "assertion"})
    server = cm.start_metrics_server(m, port=0)   # 0 → 本测试直接指定，见下
    assert server is None, "port<=0 必须不启动"

    # 用系统分配的端口真起一次
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    server = cm.start_metrics_server(m, port=port)
    assert server is not None
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            body = r.read().decode()
        assert F.FUNNEL_STORED in body
        with pytest.raises(Exception):
            urllib.request.urlopen(f"http://127.0.0.1:{port}/nope", timeout=5)
    finally:
        server.shutdown()


def test_bind_failure_does_not_raise(monkeypatch):
    """观测面起不来绝不能拖垮提炼管线（端口被占是最常见的情形）。"""
    from bladex_proxy import consolidator_metrics as cm

    def boom(*a, **k):
        raise OSError("address already in use")

    monkeypatch.setattr(cm, "ThreadingHTTPServer", boom)
    assert cm.start_metrics_server(Metrics(), port=39099) is None
