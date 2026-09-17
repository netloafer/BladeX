"""Memory Index 派生层单元测试（T3 — consolidation + LanceDB + 增量重建）。

嵌入/LLM 全部 mock，不跑真模型。
"""

import tempfile
from pathlib import Path

from bladex_core.consolidation_proxy import ProxyConsolidator
from bladex_core.fact import ConversationTurn, Fact
from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import Identity, Turn, TurnStatus
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class MockEmbedder:
    """固定向量 mock embedder — 用简单 hash 映射到固定维度向量。"""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            # 用文本 hash 生成确定性向量
            h = hash(text)
            vec = [((h >> i) & 1) * 1.0 for i in range(self._dim)]
            results.append(vec)
        return results


def _make_turn(session_id: str, user_msg: str, ledger_key: str = "") -> Turn:
    return Turn(
        identity=Identity(user_id="u1", agent_id="a1", session_id=session_id),
        model="test",
        request_messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": user_msg},
        ],
        response_text="OK",
        status=TurnStatus.OK,
    )


def test_consolidation_produces_facts():
    """一段多轮会话经 Memory Hub → 后台提炼出 Fact 并入 Memory Index（嵌入 mock）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # 写几条 Memory Hub turns
        turn1 = _make_turn("s1", "My name is Jason and I work on外贸 exports.", "k1")
        turn2 = _make_turn("s1", "I prefer communicating in Chinese for business emails.", "k2")
        ledger.put("u1/a1/s1/k1", turn1)
        ledger.put("u1/a1/s1/k2", turn2)

        # 从 Memory Hub 增量重建 Memory Index
        new_count = index.rebuild_from_hub(ledger)

        # 应该提炼出至少 1 条 fact（两条不同的 user 消息）
        assert new_count >= 1
        assert index.fact_count() >= 1

        # 验证 fact 内容
        facts = index.all_facts()
        assert len(facts) >= 1
        assert any("Jason" in f.content or "外贸" in f.content for f in facts)

        index.close()
        ledger.close()


def test_index_rebuild_from_hub():
    """删掉 Memory Index、从 Memory Hub 增量重建 → Fact/索引一致。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # 写 Memory Hub
        turn = _make_turn("s1", "I am a developer building a memory layer project called BladeX.", "k1")
        ledger.put("u1/a1/s1/k1", turn)

        # 第一次重建
        count1 = index.rebuild_from_hub(ledger)
        assert count1 >= 1
        facts1 = index.all_facts()
        assert len(facts1) >= 1

        # 记住 fact 内容
        original_contents = sorted(f.content for f in facts1)

        # 清空 Memory Index
        index.clear()
        assert index.fact_count() == 0

        # 重新重建
        count2 = index.rebuild_from_hub(ledger)
        assert count2 >= 1
        facts2 = index.all_facts()
        rebuilt_contents = sorted(f.content for f in facts2)

        # 内容一致
        assert original_contents == rebuilt_contents

        index.close()
        ledger.close()


def test_index_rebuild_incremental_consumed():
    """增量重建：第二次只处理新增 turns（consumed-key 集合）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # 写第一批
        turn1 = _make_turn("s1", "I like using Python for data analysis tasks.", "100-0")
        ledger.put("u1/a1/s1/100-0", turn1)

        count1 = index.rebuild_from_hub(ledger)
        assert count1 >= 1
        assert index.is_consumed("u1/a1/s1/100-0")

        # 写第二批
        turn2 = _make_turn("s1", "My favorite database for vector search is LanceDB.", "101-0")
        ledger.put("u1/a1/s1/101-0", turn2)

        count2 = index.rebuild_from_hub(ledger)
        assert count2 >= 1
        assert index.is_consumed("u1/a1/s1/101-0")

        # 第三次（无新数据）
        count3 = index.rebuild_from_hub(ledger)
        assert count3 == 0

        index.close()


def test_index_rebuild_watermark_poisoning_not_skipped():
    """水位游标中毒防护：高字典序脏 key 不应挡住低字典序真实 key。

    复现 review 20260707 场景：测试/压测流量（高字典序 user_id 段）把
    字典序水位顶到最大值后，真实数据（低字典序 user_id 段）每轮被跳过。
    consumed-key 集合替代字典序单游标后，每条 key 独立标记，互不影响。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # 先写一条高字典序 user_id 的 turn（模拟测试/压测脏流量）
        dirty_turn = Turn(
            identity=Identity(
                user_id="f3abf2a6", agent_id="a1", session_id="s_dirty",
            ),
            model="test",
            request_messages=[
                {"role": "user", "content": "This is test traffic with enough length for consolidation processing."},
            ],
            response_text="OK",
            status=TurnStatus.OK,
        )
        ledger.put("f3abf2a6/a1/s_dirty/100-0", dirty_turn)

        # 第一次重建：消费脏 key
        index.rebuild_from_hub(ledger)
        assert index.is_consumed("f3abf2a6/a1/s_dirty/100-0")

        # 再写一条低字典序 user_id 的 turn（模拟真实用户数据）
        real_turn = Turn(
            identity=Identity(
                user_id="53136499", agent_id="a1", session_id="s_real",
            ),
            model="test",
            request_messages=[
                {"role": "user", "content": "I am a real user discussing the World Cup 2026 in detail here."},
            ],
            response_text="OK",
            status=TurnStatus.OK,
        )
        ledger.put("53136499/a1/s_real/100-0", real_turn)

        # 第二次增量重建：真实数据不应被脏 key 挡住
        count = index.rebuild_from_hub(ledger)
        assert count >= 1, "real user turn must not be skipped by dirty key poisoning"

        # 验证真实数据的 fact 在
        facts = index.all_facts()
        assert any("World Cup" in f.content for f in facts)

        # 验证两条 key 都已消费
        assert index.is_consumed("f3abf2a6/a1/s_dirty/100-0")
        assert index.is_consumed("53136499/a1/s_real/100-0")

        index.close()
        ledger.close()
        ledger.close()


def test_lancedb_semantic_lookup():
    """LanceDB 向量可按语义查询召回（用固定小向量 mock 嵌入）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # 直接添加 facts（带嵌入）
        fact1 = Fact(
            id="fact_001",
            content="User prefers Chinese for business emails",
            category="general",
            embedding=embedder.embed(["passage: User prefers Chinese for business emails"])[0],
        )
        fact2 = Fact(
            id="fact_002",
            content="User works on foreign trade exports",
            category="general",
            embedding=embedder.embed(["passage: User works on foreign trade exports"])[0],
        )
        index.add_fact(fact1)
        index.add_fact(fact2)

        # 搜索
        results = index.search("query: Chinese emails", top_k=2)
        assert len(results) >= 1

        # 搜索结果中应有至少一条 fact
        result_ids = [f.id for f in results]
        assert "fact_001" in result_ids or "fact_002" in result_ids

        index.close()


def test_consolidation_core_directly():
    """直接测 ProxyConsolidator（不经过 Memory Index/Memory Hub）。"""
    embedder = MockEmbedder()
    consolidator = ProxyConsolidator(embedder=embedder)

    turns = [
        ConversationTurn(
            session_id="s1",
            user_messages=["I am a software engineer working on AI memory systems."],
            ledger_key="k1",
            logical_turn=1,
        ),
        ConversationTurn(
            session_id="s1",
            user_messages=["I prefer dark mode in all my development tools."],
            ledger_key="k2",
            logical_turn=2,
        ),
    ]

    facts = consolidator.consolidate_turns(turns, existing_facts=None)
    assert len(facts) >= 1
    assert all(f.content for f in facts)
    assert all(f.embedding is not None for f in facts)
    assert all(f.source_session == "s1" for f in facts)


def test_consolidation_novelty_dedup():
    """novelty 检测：相似消息不重复写入。"""
    embedder = MockEmbedder()
    consolidator = ProxyConsolidator(embedder=embedder)

    # 两条完全相同的消息
    turns = [
        ConversationTurn(
            session_id="s1",
            user_messages=["I am a software engineer working on AI memory systems."],
            ledger_key="k1",
        ),
        ConversationTurn(
            session_id="s1",
            user_messages=["I am a software engineer working on AI memory systems."],
            ledger_key="k2",
        ),
    ]

    facts = consolidator.consolidate_turns(turns, existing_facts=None)
    # 相同消息 → 只写入 1 条（第二条 novelty 检测跳过）
    assert len(facts) == 1


def test_index_no_embedder_returns_empty():
    """无 embedder 时 consolidate 返回空列表（宁缺毋滥）。"""
    consolidator = ProxyConsolidator(embedder=None)
    turns = [ConversationTurn(session_id="s1", user_messages=["test message here"])]
    facts = consolidator.consolidate_turns(turns)
    assert facts == []


def test_core_does_not_import_proxy():
    """core 与 proxy 依赖方向正确（core 不 import proxy）。"""
    import inspect

    import bladex_core.consolidation_proxy
    import bladex_core.fact

    # 检查 core 模块的源码不含 proxy import
    for module in [bladex_core.consolidation_proxy, bladex_core.fact]:
        source = inspect.getsource(module)
        assert "bladex_proxy" not in source, \
            f"{module.__name__} should not import bladex_proxy"


# ── T2: 跨 session 召回 + user 隔离 ──

def test_cross_session_recall():
    """会话 A 写入的事实，在同 user 会话 B 的相关 query 下被召回。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # 会话 A 写入事实（user=u1, session=sA）
        turn_a = Turn(
            identity=Identity(user_id="u1", agent_id="a1", session_id="sA"),
            model="test",
            request_messages=[
                {"role": "user", "content": "My favorite programming language is Rust for systems work."},
            ],
            response_text="OK",
            status=TurnStatus.OK,
        )
        ledger.put("u1/a1/sA/100-0", turn_a)

        # 从 Memory Hub 重建 Memory Index（提炼 Fact）
        index.rebuild_from_hub(ledger)
        assert index.fact_count() >= 1

        # 会话 B 查询（同 user=u1, 不同 session=sB）应能召回
        results = index.search("favorite programming language", top_k=5, user_id="u1")
        assert len(results) >= 1
        assert any("Rust" in f.content for f in results)

        index.close()
        ledger.close()


def test_recall_user_isolation():
    """跨 session 召回不串到别的 user。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        ledger = MemoryHub(Path(tmpdir) / "ledger")
        ledger.open()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # user=u1 的事实
        turn_u1 = Turn(
            identity=Identity(user_id="u1", agent_id="a1", session_id="sA"),
            model="test",
            request_messages=[
                {"role": "user", "content": "User u1 likes Python for data analysis tasks."},
            ],
            response_text="OK",
            status=TurnStatus.OK,
        )
        ledger.put("u1/a1/sA/100-0", turn_u1)

        # user=u2 的事实
        turn_u2 = Turn(
            identity=Identity(user_id="u2", agent_id="a1", session_id="sB"),
            model="test",
            request_messages=[
                {"role": "user", "content": "User u2 prefers Java for enterprise development."},
            ],
            response_text="OK",
            status=TurnStatus.OK,
        )
        ledger.put("u2/a1/sB/100-0", turn_u2)

        index.rebuild_from_hub(ledger)
        assert index.fact_count() >= 2

        # user=u1 查询 → 只看到自己的事实
        results_u1 = index.search("programming language", top_k=10, user_id="u1")
        assert len(results_u1) >= 1
        assert all("u1" in f.content for f in results_u1)
        assert not any("u2" in f.content for f in results_u1)

        # user=u2 查询 → 只看到自己的事实
        results_u2 = index.search("programming language", top_k=10, user_id="u2")
        assert len(results_u2) >= 1
        assert all("u2" in f.content for f in results_u2)
        assert not any("u1" in f.content for f in results_u2)

        index.close()
        ledger.close()


def test_search_without_user_id_returns_all():
    """不带 user_id 的搜索返回所有 Fact（向后兼容）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        fact1 = Fact(
            id="fact_001",
            content="User likes Python",
            category="general",
            source_user_id="u1",
            embedding=embedder.embed(["passage: User likes Python"])[0],
        )
        fact2 = Fact(
            id="fact_002",
            content="User likes Java",
            category="general",
            source_user_id="u2",
            embedding=embedder.embed(["passage: User likes Java"])[0],
        )
        index.add_fact(fact1)
        index.add_fact(fact2)

        # 不带 user_id → 返回所有
        all_results = index.search("programming", top_k=10)
        assert len(all_results) >= 2

        index.close()


# ── F1: 跨进程只读路径测试（read_only Memory Index 实例也能 search）──

def test_read_only_index_can_search_after_writes():
    """F1 验收：进程 A 可写 add_fact 落盘 → 新开 read_only=True 的 Memory Index 实例 open() 后直接 search 能召回。

    这是复现生产读路径的测试——proxy 进程只读 Memory Index、从不 add_fact。
    修复前此测试会失败（_table=None → search 返回 []）。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        # 写路径：可写 Memory Index 实例，add_fact 落盘
        index_write = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index_write.open()
        fact1 = Fact(
            id="fact_001",
            content="User prefers Chinese for business emails",
            category="general",
            source_user_id="u1",
            embedding=embedder.embed(["passage: User prefers Chinese for business emails"])[0],
        )
        index_write.add_fact(fact1)
        index_write.close()

        # 读路径：新开 read_only=True 的 Memory Index 实例（模拟 proxy 进程）
        index_read = MemoryIndex(Path(tmpdir) / "index", embedder=embedder, read_only=True)
        index_read.open()

        # 直接 search（不经过 add_fact）——修复前 _table=None，返回 []
        results = index_read.search("Chinese emails", top_k=5, user_id="u1")
        assert len(results) >= 1, "read_only Memory Index search should return results"
        assert any("Chinese" in f.content for f in results)

        index_read.close()


def test_read_only_index_empty_database_returns_empty():
    """空 Memory Index（无 facts 表）read_only 打开后 search 返回 []（合理行为）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder, read_only=True)
        index.open()
        # 表不存在 → search 返回 []
        results = index.search("query", top_k=5)
        assert results == []
        index.close()


# ── F2: 相关性阈值过滤 ──


# ── F3: 真超时降级 ──


# ── F4: user_id 特殊字符 ──

def test_recall_user_isolation_special_chars():
    """F4: user_id 含单引号不串号/不报错。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        index = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index.open()

        # user_id 含单引号
        fact = Fact(
            id="f1",
            content="User with special id likes Python",
            category="general",
            source_user_id="user'with'quotes",
            embedding=embedder.embed(["passage: User with special id likes Python"])[0],
        )
        index.add_fact(fact)

        # 搜索不报错
        results = index.search("Python", top_k=5, user_id="user'with'quotes")
        assert len(results) >= 1

        # 隔离：不同 user_id 不串
        results_other = index.search("Python", top_k=5, user_id="other_user")
        assert all(f.source_user_id != "user'with'quotes" for f in results_other)

        index.close()


# ── ADR-0014 L0: 阈值倒挂修复 ──

def test_threshold_invariant_novelty_above_semantic():
    """ADR-0014 §2.1 不变式：T_去重 > T_聚类（否则 Matter 结构性锁死成单例）。

    若 novelty_threshold <= semantic_threshold，则一条 Fact B 能被存下
    (sim < novelty) 但无法并入已有 Matter (sim < semantic <= novelty)，
    使每个 Matter 永远只有 1 个成员。
    """
    from bladex_core.attribution import _DEFAULT_SEMANTIC_THRESHOLD
    from bladex_core.consolidation_proxy import _DEFAULT_NOVELTY_THRESHOLD

    assert _DEFAULT_NOVELTY_THRESHOLD > _DEFAULT_SEMANTIC_THRESHOLD, (
        f"不变式违反：novelty({_DEFAULT_NOVELTY_THRESHOLD}) <= "
        f"semantic({_DEFAULT_SEMANTIC_THRESHOLD}) → Matter 锁死单例"
    )


def test_index_derived_thresholds_wired():
    """ADR-0014 L0: MemoryIndex 接受并传递阈值配置。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        index = MemoryIndex(
            Path(tmpdir) / "index", embedder=embedder,
            novelty_threshold=0.97, semantic_threshold=0.80,
            digestion_threshold=0.80,
        )
        assert index._novelty_threshold == 0.97
        assert index._semantic_threshold == 0.80
        assert index._digestion_threshold == 0.80
        index.open()
        index.close()


def test_config_has_consolidation_thresholds():
    """ADR-0014 L0: ProxyConfig 暴露三个阈值配置项。"""

    cfg = ProxyConfig()
    assert cfg.novelty_threshold > cfg.semantic_threshold, (
        "config 不变式违反：novelty <= semantic"
    )
    # M2-2：0.95 → 0.98。**语义变了，不只是数变了**：novelty 不再独自决定
    # "这条该不该入库"（e5 在本库噪声底 0.816–0.93，它注定要么漏网要么误杀），
    # 降为"只拦完全重发"；入不入库交给写入时裁决（看文本不看阈值）。
    # 默认值单一真相源在 `bladex_core.flags.MEMORY_NUMERIC_DEFAULTS`。
    from bladex_core.flags import MEMORY_NUMERIC_DEFAULTS

    assert cfg.novelty_threshold == MEMORY_NUMERIC_DEFAULTS["BLADEX_NOVELTY_THRESHOLD"]
    assert cfg.novelty_threshold == 0.98
    assert cfg.semantic_threshold == 0.85
    assert cfg.digestion_threshold == 0.85


# ── T3(ADR-0018 A1): Memory Index 读端追新 ──

def test_read_only_sees_writes_after_open_without_restart():
    """A1/T3 核心：read_only Memory Index 打开后，consolidator 再写入，不重启 proxy 即可见。

    修复前（read_only 静态快照）：proxy 启动后 consolidator 写的新 fact 永远读不到
    -> 记忆闭环随 proxy 寿命退化。修复后（secondary + catch_up + checkout_latest）：可见。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()
        # consolidator（read_write primary）
        index_write = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index_write.open()
        fact1 = Fact(
            id="fact_a1_1", content="User likes e5 embeddings",
            category="general", source_user_id="u1",
            embedding=embedder.embed(["passage: User likes e5 embeddings"])[0],
        )
        index_write.add_fact(fact1)

        # proxy（read_only secondary）打开，能看见 fact1
        index_read = MemoryIndex(
            Path(tmpdir) / "index", embedder=embedder, read_only=True,
            catchup_throttle_s=0.0,  # 测试不节流，每次 search 都 catch-up
        )
        index_read.open()
        r1 = index_read.search("e5", top_k=5, user_id="u1")
        assert any("e5" in f.content for f in r1), "should see fact written before open"

        # consolidator 再写一条新 fact（proxy 不重启）
        fact2 = Fact(
            id="fact_a1_2", content="User prefers multilingual-e5-large model",
            category="general", source_user_id="u1",
            embedding=embedder.embed(["passage: User prefers multilingual-e5-large model"])[0],
        )
        index_write.add_fact(fact2)

        # proxy 不重启 -> search 应能看见 fact2（A1 修复点）
        r2 = index_read.search("multilingual", top_k=5, user_id="u1")
        assert any("multilingual" in f.content for f in r2), \
            "read_only Memory Index must see writes after open without restart (A1)"

        index_read.close()
        index_write.close()


def test_read_only_lazy_open_after_empty_start():
    """A1/T3: 空库启动（meta_db=None, _table=None）后 consolidator 首轮写入，proxy 惰性重开可见。

    修复前：启动时 meta_rocksdb 不存在 -> meta_db 终生 None -> 全新安装永远 Memory Index 空库。
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        embedder = MockEmbedder()

        # proxy 先打开（空库）-- meta_db=None, _table=None
        index_read = MemoryIndex(
            Path(tmpdir) / "index", embedder=embedder, read_only=True,
            catchup_throttle_s=0.0,
        )
        index_read.open()
        assert index_read._meta_db is None, "empty start -> meta_db None"
        assert index_read._table is None, "empty start -> _table None"
        assert index_read.search("anything") == [], "empty -> no results"

        # consolidator 首轮写入
        index_write = MemoryIndex(Path(tmpdir) / "index", embedder=embedder)
        index_write.open()
        index_write.add_fact(Fact(
            id="fact_lazy_1", content="BladeX uses RocksDB for Memory Hub",
            category="general", source_user_id="u1",
            embedding=embedder.embed(["passage: BladeX uses RocksDB for Memory Hub"])[0],
        ))

        # proxy 不重启 -> 惰性重开 + catch_up -> 可见
        r = index_read.search("RocksDB", top_k=5, user_id="u1")
        assert any("RocksDB" in f.content for f in r), \
            "lazy reopen must see first write after empty start (A1)"

        index_read.close()
        index_write.close()
