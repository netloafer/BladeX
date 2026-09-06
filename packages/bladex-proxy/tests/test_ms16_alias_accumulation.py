"""MS-16 雪球源头回归（T4a，2026-08-10）：aliases 停止自动累积。

复现（scripts/audit_l2l3_gate_replay.py，live 库 08-10）：
`_accumulate_matter_metadata` 把成员 fact 的 proposal_titles 无界追加进
matter.aliases（README matter 145 边累到 95 条 alias），一条错边混入后
它的提案标题即成合法匹配键——L2=466 里 415 条走此路径。

修后语义：
  - aliases = 创建时原生键 + manual 追加（rename / 手动 enrich 保留，
    ADR-0012 手动映射主权——test_index_matter.py::
    test_manual_correction_improves_later_auto_attribution 原样守护）。
  - entities 仍累积（封顶 FIFO），但仅展示——匹配面见 core 侧
    test_ms16_attribution_regate.py。
"""

import tempfile
from pathlib import Path

from bladex_core.fact import Fact
from bladex_core.matter import Matter, MatterOrigin, MatterStatus
from bladex_proxy.storage.memory_index import MemoryIndex


def _make_index(tmpdir: str) -> MemoryIndex:
    index = MemoryIndex(Path(tmpdir) / "index", embedder=None, read_only=False)
    index.open()
    return index


def _matter() -> Matter:
    return Matter(
        matter_id="m-native", title="BladeX 双语 README 编写任务",
        aliases=["BladeX 双语 README 编写任务", "README", "beta-b9"],
        status=MatterStatus.ACTIVE, origin=MatterOrigin.AUTO,
    )


def test_accumulate_does_not_grow_aliases():
    """成员 fact 的 proposal_titles 不再灌入 aliases（雪球源头掐断）。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)
        index.add_matter(_matter())

        fact = Fact(
            id="f-1", content="unabyssapp one memory 研究",
            entities=["unabyssapp"],
            proposal_titles=["深入研究 unabyssapp 架构并与 BladeX 对比"],
        )
        index._accumulate_matter_metadata("m-native", fact)

        loaded = index.get_matter("m-native")
        assert loaded.aliases == ["BladeX 双语 README 编写任务", "README", "beta-b9"]
        index.close()


def test_accumulate_entities_still_capped_union():
    """entities 照旧累积（展示/观察用），封顶 FIFO 语义不变。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        index = _make_index(tmpdir)
        index.add_matter(_matter())

        fact = Fact(id="f-1", content="x", entities=["unabyssapp", "one-memory"])
        index._accumulate_matter_metadata("m-native", fact)

        loaded = index.get_matter("m-native")
        assert "unabyssapp" in loaded.entities
        assert "one-memory" in loaded.entities
        # aliases 不动
        assert loaded.aliases == ["BladeX 双语 README 编写任务", "README", "beta-b9"]
        index.close()
