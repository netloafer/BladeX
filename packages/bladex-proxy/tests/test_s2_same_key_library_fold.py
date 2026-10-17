"""批 S · S2.1（MQ-I20）最小复现：跨 pass 的同键重述 —— 修前并存、修后折叠。

隔离环境（真 Hub + 真 Index + MockEmbedder），裁决器是"全判 ADD"的桩：这样库里的折叠
**只能**来自快路径 —— 快路径对库内邻居命中与否就是被测对象。两个 pass：
  pass 1：轮 A 蒸出 (BladeX, 状态) = "0.2.0 已发布。"
  pass 2：轮 B 蒸出同键**重述** "0.2.0 已发布！"（同实体、正文只差一个标点 ⇒ `should_merge` 重述）
修后：current 只剩 1 条（NOOP 丢弃候选）；阴性对照（hydrate 拆掉）⇒ 2 条并存 = 缺陷本体。
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

from bladex_core.adjudication import AdjudicationOp, AdjudicationVerdict
from bladex_core.consolidation_proxy import InRunNeighborSource
from bladex_core.distillation import DistillFact, DistillOutput
from bladex_proxy.models import Identity, SessionIdSource, Turn, TurnStatus
from bladex_proxy.storage import memory_index as _mi
from bladex_proxy.storage.memory_hub import MemoryHub
from bladex_proxy.storage.memory_index import MemoryIndex


class _KeyedDistiller:
    """按用户消息原样出一条 assertion（subject/attribute 固定 = 同键）。"""

    @property
    def model_name(self) -> str:
        return "s2-keyed"

    @property
    def prompt_ver(self) -> str:
        return "s2-keyed-001"

    def distill(self, user_message: str) -> DistillOutput:
        # 实体 = BladeX + 正文里的版本号：重述（同版本）实体重叠 1.0；新值（换版本）重叠 1/3
        # ⇒ `should_merge` 前者判重述、后者判新值，两条路径都能在本夹具里走到。
        import re
        version = re.findall(r"\d+\.\d+\.\d+", user_message)
        return DistillOutput(
            facts=[DistillFact(content=user_message, kind="event", item_kind="assertion",
                               subject="BladeX", attribute="状态", entities=["BladeX", *version])],
            model_name="s2-keyed")


class _AddAllAdjudicator:
    """LLM 裁决桩：一律 ADD（把折叠的机会只留给快路径）。"""

    @property
    def model_name(self) -> str:
        return "add-all"

    @property
    def prompt_ver(self) -> str:
        return "add-all-001"

    def adjudicate(self, items):  # noqa: ANN001, ANN201
        return [AdjudicationVerdict(candidate_id=i.candidate_id, op=AdjudicationOp.ADD)
                for i in items]


class MockEmbedder:
    def embed(self, texts):  # noqa: ANN001, ANN201
        out = []
        for t in texts:
            v = [0.0] * 64
            for i, ch in enumerate(t[:32]):
                v[(ord(ch) + i) % 64] += 0.1
            out.append(v)
        return out

    @property
    def available(self) -> bool:
        return True


def _put(hub: MemoryHub, seq: int, text: str) -> None:
    ident = Identity(user_id="u1", agent_id="claude-code", session_id="s1", turn_index=seq,
                     session_id_source=SessionIdSource.FINGERPRINT)
    turn = Turn(identity=ident, model="test", request_messages=[{"role": "user", "content": text}],
                response_text="好的。", status=TurnStatus.OK, ts=datetime.now(UTC),
                session_id_source=SessionIdSource.FINGERPRINT)
    hub.put(ident.storage_key(f"{int(datetime.now(UTC).timestamp() * 1000)}-0"), turn)
    time.sleep(0.02)


class _UpdateLibraryAdjudicator(_AddAllAdjudicator):
    """LLM 裁决桩：候选取代篮子里的全部邻居（模拟"库内旧值被新值取代"的判决）。"""

    def adjudicate(self, items):  # noqa: ANN001, ANN201
        return [AdjudicationVerdict(candidate_id=i.candidate_id, op=AdjudicationOp.UPDATE,
                                    target_fact_ids=[n.fact_id for n in i.neighbors])
                for i in items]


def _run(tmp_path: Path, *, second: str = "BladeX 状态：0.2.0 已发布！",
         adjudicator=None) -> list:  # noqa: ANN001
    hub = MemoryHub(tmp_path / "hub")
    hub.open()
    # novelty 阈值抬到 0.999：正文只差一个标点的重述向量很近，别让判重先把它吞了
    index = MemoryIndex(tmp_path / "index", embedder=MockEmbedder(), read_only=False,
                        distiller=_KeyedDistiller(),
                        adjudicator=adjudicator or _AddAllAdjudicator(),
                        novelty_threshold=0.999)
    index.open()
    try:
        _put(hub, 1, "BladeX 状态：0.2.0 已发布。")
        index.rebuild_from_hub(hub)          # pass 1 → 库里 1 条
        _put(hub, 2, second)
        index.rebuild_from_hub(hub)          # pass 2 → 候选与库内同键
        return index.all_facts()
    finally:
        index.close()
        hub.close()


def _current(facts: list) -> list:
    return [f for f in facts if f.t_invalid is None]


def test_same_key_restatement_across_passes_folds_to_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BLADEX_ADJUDICATE_ENABLED", "1")
    facts = _run(tmp_path)
    current = _current(facts)
    assert len(current) == 1, [f.content for f in current]
    assert current[0].content.endswith("。"), "留下的是库里先入的那条，候选被 NOOP 丢弃"
    assert current[0].strength >= 2, "NOOP 强化落到了库内那条（判决真的应用了，不再降级 ADD）"


def test_llm_update_against_library_target_now_applies(tmp_path, monkeypatch) -> None:
    """裁决器判 UPDATE（新值取代库内旧值）⇒ 旧条 t_invalid + superseded_by，不再降级 ADD。"""
    monkeypatch.setenv("BLADEX_ADJUDICATE_ENABLED", "1")
    facts = _run(tmp_path, second="BladeX 状态：0.3.0 已发布。",
                 adjudicator=_UpdateLibraryAdjudicator())
    current = _current(facts)
    assert len(current) == 1 and "0.3.0" in current[0].content, [f.content for f in facts]
    old = [f for f in facts if f.t_invalid is not None]
    assert len(old) == 1 and old[0].superseded_by == current[0].id


def test_negative_control_without_hydrate_coexists(tmp_path, monkeypatch) -> None:
    """阴性对照 = 缺陷本体：库内邻居没坐标 ⇒ 同键不命中 ⇒ 两条并存。"""
    monkeypatch.setenv("BLADEX_ADJUDICATE_ENABLED", "1")
    monkeypatch.setattr(_mi, "InRunNeighborSource",
                        lambda checker=None, hydrate=None: InRunNeighborSource(checker=checker))
    current = _current(_run(tmp_path))
    assert len(current) == 2, [f.content for f in current]


def test_flag_off_restores_old_behaviour(tmp_path, monkeypatch) -> None:
    """`BLADEX_ADJUDICATE_LIBRARY_HYDRATE=0` = 旧形态：重述并存、UPDATE 降级 ADD。"""
    monkeypatch.setenv("BLADEX_ADJUDICATE_ENABLED", "1")
    monkeypatch.setenv("BLADEX_ADJUDICATE_LIBRARY_HYDRATE", "0")
    assert len(_current(_run(tmp_path))) == 2
    facts = _run(tmp_path / "b", second="BladeX 状态：0.3.0 已发布。",
                 adjudicator=_UpdateLibraryAdjudicator())
    assert len(_current(facts)) == 2, "旧形态下库内目标查不到 ⇒ UPDATE 降级 ADD"
