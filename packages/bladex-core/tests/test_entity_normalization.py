"""V-I2 验收剧本：分隔符变体折叠（08-20 病例回归）/ CJK 粘连修复 / 误折叠阴性对照。"""

from __future__ import annotations

from types import SimpleNamespace

from bladex_core.fusion import _entity_jaccard, build_entity_channel
from bladex_core.identifiers import extract_identifiers, split_cjk_boundary
from bladex_core.topic_keys import canonical_entity_key

# ── canonical_entity_key ────────────────────────────────────────────────────

class TestCanonicalKey:
    def test_known_case_qwen_separator_variants_unify(self):
        # 2026-08-20 注入归因的原病例
        variants = ["qwen3.8:27b", "qwen3.8-27B", "qwen3.8_27b", "qwen3.8 27b"]
        assert {canonical_entity_key(v) for v in variants} == {"qwen3.8-27b"}

    def test_dot_is_preserved_negative_control(self):
        # 误折叠阴性对照：5.2 与 52 是两个版本
        assert canonical_entity_key("glm-5.2") != canonical_entity_key("glm-52")
        assert canonical_entity_key("qwen3.8-27b") != canonical_entity_key("qwen3.8-72b")

    def test_distinct_names_stay_distinct(self):
        assert canonical_entity_key("泰山啤酒") != canonical_entity_key("泰安仁信")

    def test_fullwidth_and_case_fold(self):
        assert canonical_entity_key("ＱＷＥＮ３.８：２７Ｂ") == "qwen3.8-27b"

    def test_idempotent_and_identity_on_plain(self):
        assert canonical_entity_key("redis") == "redis"
        assert canonical_entity_key(canonical_entity_key("a:b_c")) == \
            canonical_entity_key("a:b_c")

    def test_empty(self):
        assert canonical_entity_key("") == ""
        assert canonical_entity_key(" : ") == ""


# ── CJK 粘连（identifiers）─────────────────────────────────────────────────

class TestCjkBoundary:
    def test_glued_version_token_is_clean(self):
        toks = extract_identifiers("已下载qwen3.8模型并解压")
        assert any(t.startswith("qwen3.8") for t in toks)
        assert all(not any("一" <= c <= "鿿" for c in t) for t in toks)

    def test_glued_path_token_is_clean(self):
        toks = extract_identifiers("改了routing.toml之后重启")
        assert "routing.toml" in toks

    def test_ascii_only_text_unchanged(self):
        text = "see fact_270d485c1a78 and BLADEX_MIN_IMPORTANCE in memory_index.py"
        assert split_cjk_boundary(text) == text
        assert extract_identifiers(text) == extract_identifiers(split_cjk_boundary(text))

    def test_split_is_idempotent(self):
        s = "已下载qwen3.8模型"
        assert split_cjk_boundary(split_cjk_boundary(s)) == split_cjk_boundary(s)


# ── entity 通道（fusion）────────────────────────────────────────────────────

def _fact(entities):
    return SimpleNamespace(entities=entities)


class TestEntityChannel:
    def test_separator_variant_now_hits(self):
        # 修复前：`qwen3.8:27b`.lower() not in query → 通道全灭
        facts = {"f1": _fact(["qwen3.8:27b"])}
        ch = build_entity_channel("帮我看 qwen3.8-27B 下载好了没", facts)
        assert ch is not None and ch.ranked_ids == ["f1"]
        assert facts["f1"]._entity_hit == 1.0

    def test_plain_hit_still_hits(self):
        facts = {"f1": _fact(["redis"])}
        ch = build_entity_channel("redis 掉线了怎么办", facts)
        assert ch is not None and ch.ranked_ids == ["f1"]

    def test_negative_no_false_fold(self):
        facts = {"f1": _fact(["glm-52"])}
        assert build_entity_channel("升级 glm-5.2 的配置", facts) is None

    def test_jaccard_folds_variants(self):
        assert _entity_jaccard(["qwen3.8:27b"], ["qwen3.8-27B"]) == 1.0
        assert _entity_jaccard(["glm-5.2"], ["glm-52"]) == 0.0
