"""ADR-0028 E6：检索五层重构（core 侧——query 理解 / 标识符 / 融合 / 蒸馏保真）。

五层各自在修一个**有实测证据**的毛病：

    E6.1  信封当 query（claude-code 末条 user p50 = 70,261 字符，98.21% 是信封）
          + 短指代（「继续」「为什么」，向量近乎无信息）
    E6.2  dense 对标识符是瞎的（`bxe3663a38` vs `bx3a984bcd` cosine = 0.9890）
    E6.3  分值不可比（cosine 噪声底 0.88–0.93、top-15 极差 0.024–0.060，
          而 importance 差分 ≈0.007）→ 改用秩融合 RRF
    E6.4  层从"硬过滤 + 5% 权重"改成边界与配额（同 8 条过程叙述重复注入 171 次）
    E6.5  写入侧丢标识符 / 语言漂移 / novelty 误杀

（E6.2 的 FTS5 实现与 E6.3 在 Memory Index 上的接线见 proxy 侧
 `test_adr0028_e6_fts_fusion.py`；本文件只测 agent 中立的纯函数层。）
"""

from __future__ import annotations

import pytest
from bladex_core.distill_fidelity import (
    attach_missing_identifiers,
    check_identifiers,
    check_language,
    detect_lang,
    identifier_novelty_override,
    mark_language_drift,
)
from bladex_core.fact import Fact, ItemKind
from bladex_core.fusion import (
    Channel,
    apply_type_quota,
    fold_synonyms,
    fuse_and_rerank,
    mmr_select,
    rrf_fuse,
)
from bladex_core.identifiers import extract_identifiers, missing_identifiers
from bladex_core.query_understanding import (
    MAX_QUERY_CHARS,
    is_pronoun_query,
    understand_query,
)


# ── E6.1 Query 理解层 ───────────────────────────────────────────────────


def test_envelope_stripped_from_query():
    """信封不该进 query 向量——否则向量指的是信封的语义，不是用户想问的事。"""
    raw = ("<system-reminder>项目规范一大段……</system-reminder>\n"
           "T8b 的 ANN 索引 recall 阈值定成多少了？")
    q, _ids = understand_query(raw)
    assert "<system-reminder>" not in q
    assert q == "T8b 的 ANN 索引 recall 阈值定成多少了？"


def test_pure_envelope_query_falls_back_to_open_intent():
    raw = "<transcript>子 agent 的会话转录……</transcript>"
    q, _ = understand_query(raw, open_unit_intent="把 T8b 的基准脚本补进仓库")
    assert q == "把 T8b 的基准脚本补进仓库"


@pytest.mark.parametrize("word", ["继续", "这个", "为什么", "why", "and then", "ok"])
def test_pronoun_queries_detected(word):
    assert is_pronoun_query(word)
    assert is_pronoun_query(word + "？")


def test_pronoun_query_expanded_with_open_unit_intent():
    q, _ = understand_query("继续", open_unit_intent="修 Memory Index 的 ref_count 回写")
    assert q == "修 Memory Index 的 ref_count 回写 继续"


def test_query_with_substance_not_expanded():
    """有信息量的 query 不扩写——扩写是给"指代"用的，不是给所有短句用的。"""
    raw = "Memory Index 的 ref_count 为什么恒为 0"
    q, _ = understand_query(raw, open_unit_intent="无关的上文")
    assert q == raw


def test_short_query_with_identifier_not_expanded():
    """短但带标识符 → 信息量足够（词法通道能用），不扩写。"""
    q, ids = understand_query("bxe3663a38", open_unit_intent="无关上文")
    assert q == "bxe3663a38"
    assert "bxe3663a38" in ids


def test_query_truncated_to_limit():
    q, _ = understand_query("あ" * 5000)
    assert len(q) == MAX_QUERY_CHARS


def test_identifiers_extracted_from_raw_not_stripped():
    """标识符从**原始**文本抽：信封里也可能带用户提到的文件名/错误码。

    剥离只是为了让**向量**别跑偏，词法通道该拿的还得拿。
    """
    raw = "<system-reminder>见 packages/bladex_core/fusion.py</system-reminder>\n继续"
    _q, ids = understand_query(raw, open_unit_intent="修检索")
    assert any("fusion.py" in i for i in ids)


def test_understand_query_is_pure_function():
    a = understand_query("同一个输入", open_unit_intent="同一个上文")
    b = understand_query("同一个输入", open_unit_intent="同一个上文")
    assert a == b


# ── 标识符抽取（E6.1/E6.5 单一事实源）────────────────────────────────


def test_extract_probe_tokens_and_paths_and_consts():
    text = ("探针代码 bxe3663a38，改的是 packages/bladex_core/fusion.py，"
            "开关 BLADEX_MIN_IMPORTANCE，版本 v0.1.0，见 `IndexType`")
    ids = extract_identifiers(text)
    assert "bxe3663a38" in ids
    assert any("fusion.py" in i for i in ids)
    assert "BLADEX_MIN_IMPORTANCE" in ids
    assert "v0.1.0" in ids
    assert "IndexType" in ids


def test_extract_ignores_plain_english_words():
    """纯字母的"hex"多半只是英文单词（decade/faced/added…），不该当标识符。"""
    ids = extract_identifiers("the decade faced added deface")
    assert ids == []


def test_extract_dedupes_and_preserves_order():
    ids = extract_identifiers("bxe3663a38 再说一次 bxe3663a38")
    assert ids.count("bxe3663a38") == 1


def test_missing_identifiers_is_case_insensitive():
    """模型改了大小写但 token 没丢 → 不算丢。"""
    assert missing_identifiers("错误码 ROUTE_ENABLED", ["route_enabled 被打开了"]) == []


# ── E6.3 融合与重排 ────────────────────────────────────────────────────


def _f(fid: str, kind: ItemKind = ItemKind.ASSERTION, *, importance: float = 0.7,
       entities: list[str] | None = None, subject: str = "", attribute: str = "") -> Fact:
    return Fact(id=fid, content=f"内容 {fid}", item_kind=kind, importance=importance,
                entities=entities or [], subject=subject, attribute=attribute,
                source_user_id="u", source_session="s")


def test_rrf_uses_rank_not_score():
    """RRF 只看秩：两路都排第一的那条必然赢过"某一路分值极高"的。"""
    scores = rrf_fuse([
        Channel("dense", ["a", "b", "c"]),
        Channel("lexical", ["a", "d"]),
    ])
    assert scores["a"] > scores["b"] > scores["c"]
    assert scores["a"] > scores["d"]


def test_rrf_constant_makes_top_ranks_close():
    """RRF_K=60 是平滑项：前几名差距小，正是"别让单路分值噪声主导"的设计。"""
    s = rrf_fuse([Channel("dense", ["a", "b"])])
    assert 0 < s["a"] - s["b"] < 0.001


def test_fold_synonyms_by_supersede_key():
    """同取代键的两条只留高分那条（跨语言双份在此收拢）。"""
    zh = _f("zh", ItemKind.ASSERTION, subject="探针", attribute="颜色")
    en = _f("en", ItemKind.ASSERTION, subject="探针", attribute="颜色")
    kept, folded = fold_synonyms([zh, en])
    assert [f.id for f in kept] == ["zh"] and folded == 1


def test_fold_synonyms_by_vector_and_entities():
    a = _f("a", entities=["chartreuse"])
    b = _f("b", entities=["chartreuse"])
    vecs = {"a": [1.0, 0.0], "b": [0.99, 0.01]}
    kept, folded = fold_synonyms([a, b], vectors=vecs)
    assert [f.id for f in kept] == ["a"] and folded == 1


def test_fold_keeps_different_topics():
    a = _f("a", entities=["阿根廷"], subject="阿根廷", attribute="状态")
    b = _f("b", entities=["开票"], subject="开票", attribute="状态")
    kept, folded = fold_synonyms([a, b], vectors={"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert len(kept) == 2 and folded == 0


def test_type_quota_drops_profile_obs_and_caps_file_ref():
    ranked = [
        _f("pipeline", ItemKind.PROFILE_OBS),
        _f("fr1", ItemKind.FILE_REF),
        _f("fr2", ItemKind.FILE_REF),
        _f("a1"), _f("a2"),
    ]
    out = [f.id for f in apply_type_quota(ranked, 5)]
    assert "pipeline" not in out                      # 画像原料不直接注入
    assert out.count("fr1") == 1 and "fr2" not in out   # file_ref ≤1


def test_type_quota_floor_for_lesson_and_procedure():
    """修的是"记住了发生什么、没记住为什么错"的**展示面**：

    lesson / procedure 在纯相关性排序里经常被一堆 assertion 挤掉，
    而它们恰恰是同型错误第三次躲过的那部分记忆。
    """
    ranked = [_f(f"a{i}") for i in range(5)] + [
        _f("lesson1", ItemKind.LESSON), _f("proc1", ItemKind.PROCEDURE)]
    out = [f.id for f in apply_type_quota(ranked, 3)]
    assert "lesson1" in out and "proc1" in out


def test_mmr_prefers_diverse_candidates():
    a, b, c = _f("a"), _f("b"), _f("c")
    scores = {"a": 1.0, "b": 0.99, "c": 0.9}
    vecs = {"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]}  # b 与 a 重复
    got = [f.id for f in mmr_select([a, b, c], scores, 2, vectors=vecs)]
    assert got == ["a", "c"]


def test_mmr_without_vectors_degrades_to_score_order():
    a, b = _f("a"), _f("b")
    got = mmr_select([a, b], {"a": 1.0, "b": 0.5}, 2)
    assert [f.id for f in got] == ["a", "b"]


def test_fuse_and_rerank_end_to_end_is_deterministic():
    facts = {f.id: f for f in (_f("a"), _f("b"), _f("c"))}
    channels = [Channel("dense", ["a", "b", "c"]), Channel("lexical", ["c", "a"])]
    out1, info1 = fuse_and_rerank(facts, channels, 3)
    out2, info2 = fuse_and_rerank(facts, channels, 3)
    assert [f.id for f in out1] == [f.id for f in out2]
    assert info1["channels"] == info2["channels"] == {"dense": 3, "lexical": 2}


def test_importance_is_a_correction_not_the_main_signal():
    """importance 是**修正项**：同秩时它决定顺序，但不该压过秩本身。"""
    hi = _f("hi", importance=1.5)
    lo = _f("lo", importance=0.1)
    # 同一路、同秩不可能，所以给两路对称的秩 → RRF 相同 → importance 决胜
    out, _ = fuse_and_rerank(
        {"hi": hi, "lo": lo},
        [Channel("dense", ["lo", "hi"]), Channel("lexical", ["hi", "lo"])], 2)
    assert [f.id for f in out] == ["hi", "lo"]


# ── E6.5 蒸馏保真 ──────────────────────────────────────────────────────


class _DFact:
    """⚠ 测试替身 —— **不要拿它验证字段是否存在**。

    2026-08-08 事故：`test_language_drift_tagged` 用这个替身断言
    `"lang:drift" in f.tags`，一直绿；而生产走的是 pydantic `DistillFact`，
    当时**根本没有 `tags` 字段**，赋值直接抛 ValueError，
    全量重建 784 次蒸馏里死了 163 次、那些轮的事实全丢。

    普通 class 允许任意属性赋值，pydantic 模型不允许 —— 替身比真模型宽松，
    于是测试通过而生产崩溃。凡是断言"某字段可写"的用例，
    必须喂真模型（见 `packages/bladex-proxy/tests/test_distill_tags_and_adjudicate_chunk.py`）。
    """

    def __init__(self, content: str, entities: list[str] | None = None) -> None:
        self.content = content
        self.entities = entities or []
        self.tags = ""


def test_identifier_loss_detected():
    src = "端到端探针代码是 bxe3663a38"
    facts = [_DFact("用户的探针代码已记录")]        # token 被改写掉了
    assert "bxe3663a38" in check_identifiers(src, facts)


def test_identifier_kept_in_entities_counts_as_present():
    src = "端到端探针代码是 bxe3663a38"
    facts = [_DFact("用户的探针代码已记录", ["bxe3663a38"])]
    assert check_identifiers(src, facts) == []


def test_attach_missing_identifiers_recovers_token():
    facts = [_DFact("用户的探针代码已记录")]
    assert attach_missing_identifiers(facts, ["bxe3663a38"]) == 1
    assert "bxe3663a38" in facts[0].entities


@pytest.mark.parametrize("text,lang", [
    ("用户希望回复简洁一些", "zh"),
    ("the user prefers concise replies", "en"),
    ("热路径预算定成了 200ms，超时只注硬规则", "zh"),
])
def test_language_detection(text, lang):
    """判定规则（任务卡 E6.5 写死）：CJK 字符占比 > 30% → zh，否则 en。"""
    assert detect_lang(text) == lang


def test_language_detection_known_edge_case_ascii_heavy_chinese():
    """已知边界（照规格实现，不擅自改）：ASCII 标识符很多的中文技术句会判成 en。

    "BladeX 的 proxy 热路径预算是 200ms" 里 CJK 只占 21% < 30%。
    后果可控——语言判定只用于"产出语言是否跟随输入"，判偏最多多花一次重蒸；
    真要改阈值应先在 E5 评估集上出数，而不是照着一两个例子调（ADR-0013 教训）。
    """
    assert detect_lang("BladeX 的 proxy 热路径预算是 200ms") == "en"


def test_language_drift_detected():
    """同一事实中英两份的根因（实测 ZH/EN cosine 0.8639 < 判重阈值 0.95）。"""
    want, got = check_language("用户希望回复简洁", [_DFact("the user prefers concise replies")])
    assert want == "zh" and got == "en"


def test_language_drift_tagged():
    facts = [_DFact("x"), _DFact("y")]
    mark_language_drift(facts)
    assert all("lang:drift" in f.tags for f in facts)


def test_identifier_novelty_override_lets_new_token_through():
    """新旧探针 token 的 cosine 实测 0.9890 —— 判重会把新的吃掉。

    标识符抽取是确定性的，比"指望模型这次记得把 token 写进 entities"可靠。
    """
    assert identifier_novelty_override(
        "端到端探针代码是 bxe3663a38", ["端到端探针代码是 bx3a984bcd"]) is True


def test_identifier_novelty_override_does_not_fire_without_identifiers():
    assert identifier_novelty_override("用户喜欢简洁的回复", ["用户希望回复简洁"]) is False


def test_identifier_novelty_override_same_token_is_not_novel():
    assert identifier_novelty_override(
        "探针代码 bxe3663a38", ["探针代码是 bxe3663a38，已记录"]) is False
