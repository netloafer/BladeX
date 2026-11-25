"""MQ-W42：模型池为空走 PassthroughDistiller 时，consolidator 收尾读 stats() 不许炸。

consolidator 在有产出的批次后调 `distiller.stats()`；透传类缺该方法会抛
AttributeError，被外层 except 记成 `index_consolidator_error`（心跳 state=error + 退避）。
"""

from __future__ import annotations

from bladex_core.distillation import PassthroughDistiller
from bladex_proxy.distillation import LLMDistiller


def test_passthrough_stats_exists_and_is_empty_counts() -> None:
    stats = PassthroughDistiller().stats()
    for key in ("calls", "fails", "parse_fails", "length_fails",
                "length_salvaged", "ledger_hits"):
        assert stats[key] == 0


def test_passthrough_stats_keys_match_llm_distiller() -> None:
    llm_keys = set(LLMDistiller(model="test/model").stats())
    assert set(PassthroughDistiller().stats()) == llm_keys
