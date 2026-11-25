"""批 BB · 2a：提示词 sha 载入、判定解析、候选预筛（S4a ∪ top-3）。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from _bb_support import load_mutant, mk_fact
from bladex_core import consistency as c

_BENCH_PROMPT = Path(__file__).resolve().parents[3] / "docs/benchmarks/bb/judge-prompt-v3.txt"


def test_prompt_loads_and_sha_matches_step2_0() -> None:
    text = c.load_prompt()
    assert hashlib.sha256(text.encode()).hexdigest()[:16] == "596c18755b4093e8" == c.PROMPT_SHA16
    if _BENCH_PROMPT.is_file():  # 步骤 2-0 的出处文件；包内副本必须逐字相同
        assert _BENCH_PROMPT.read_bytes() == c.PROMPT_FILE.read_bytes()


def test_prompt_edited_without_sha_update_fails_to_load(tmp_path: Path) -> None:
    p = tmp_path / "p.txt"
    p.write_bytes(c.PROMPT_FILE.read_bytes() + b"\n")
    with pytest.raises(c.PromptShaMismatch):
        c.load_prompt(p)


def _tampered_load_raises(mod, p: Path) -> bool:
    try:
        mod.load_prompt(p)
    except mod.PromptShaMismatch:
        return True
    return False


def test_prompt_sha_check_mutant_is_red(tmp_path: Path) -> None:
    p = tmp_path / "p.txt"
    p.write_bytes(c.PROMPT_FILE.read_bytes() + b"\n")
    assert _tampered_load_raises(c, p)
    m = load_mutant(("    if got != PROMPT_SHA256:", "    if False:"))
    with pytest.raises(AssertionError):
        assert _tampered_load_raises(m, p), "去掉 sha 校验，改过的提示词也载入了，必须红"


def test_render_prompt_fills_all_fields() -> None:
    a, b = mk_fact("new", "新内容甲", day=9), mk_fact("old", "旧内容乙")
    out = c.render_prompt(a, b)
    assert "新内容甲" in out and "旧内容乙" in out and "{ta}" not in out and "{cb}" not in out


@pytest.mark.parametrize(("text", "verdict", "flagged"), [
    ('{"verdict": "矛盾", "reason": "r"}', "矛盾", True),
    ('前缀 {"verdict": "不矛盾", "reason": "r"} 后缀', "不矛盾", False),
    ('{"verdict": "说不清", "reason": "r"}', "说不清", False),   # 说不清映射为不标
    ('{"contradiction": true}', None, False),                     # 旧提示词形状 ⇒ 解析失败
    ("not json", None, False),
    ('{"verdict": "也许", "reason": "r"}', None, False),
])
def test_parse_verdict(text: str, verdict: str | None, flagged: bool) -> None:
    p = c.parse_verdict(text)
    assert p.verdict == verdict and p.flagged is flagged


def test_select_candidates_is_s4a_union_top3() -> None:
    new = mk_fact("new", "新", subject="项目甲", attribute="状态", entities=["e1", "e2"], day=9)
    ns = [
        (mk_fact("n1", "a", subject="无关", attribute="x", entities=["z"]), 0.99),               # top-3 靠前，无关
        (mk_fact("n2", "b", subject="无关", attribute="x", entities=["z"]), 0.98),               # top-3
        (mk_fact("n3", "c", subject="无关", attribute="x", entities=["z"]), 0.97),               # top-3
        (mk_fact("n4", "d", subject="项目甲", attribute="其它", entities=[]), 0.90),             # 同 subject ⇒ S4a
        (mk_fact("n5", "e", subject="别的", attribute="y", entities=["e1", "e2"]), 0.88),        # 共享 2 个 entities ⇒ S4a
        (mk_fact("n6", "f", subject="别的", attribute="y", entities=["e1"]), 0.86),              # 只共享 1 个 ⇒ 不选
        (mk_fact("n7", "g", subject="项目甲", attribute="状态", entities=[]), 0.70),             # 余弦 < 0.75 ⇒ 截断
    ]
    got = c.select_candidates(new, ns)
    assert [x.fact.id for x in got] == ["n1", "n2", "n3", "n4", "n5"]
    assert [x.via for x in got] == ["top", "top", "top", "related", "related"]
    assert [x.rank for x in got] == [1, 2, 3, 4, 5]


def test_select_candidates_excludes_self_and_dedups() -> None:
    new = mk_fact("new", "新", subject="s", day=9)
    ns = [(new, 1.0), (mk_fact("o1", "a", subject="s"), 0.9)]
    got = c.select_candidates(new, ns)
    assert [x.fact.id for x in got] == ["o1"] and got[0].via == "both"
