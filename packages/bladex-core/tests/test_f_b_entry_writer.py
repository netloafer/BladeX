"""F-B3：条目级 `writer`（轴 C 数据面）—— 渲染 / 解析 / 向后兼容。

**要能回答什么**：一本账本上"**这一条**是谁写的"。账本级 `last_writer` 只回答
"谁最后动过这本"——而跨 agent 交接的常态形状恰恰是 CC 写的 Verified 与 Hermes
写的 Next 并存在同一本里，账本级那一个字段把它们压成了同一个答案。

三条最容易安静失效的地方，各钉一条：

1. **旧行照读**：池里存量全是两段形态 `[model · ref]` / `[user]`。解析改坏了
   不会报错，只会让 ref 或 writer 悄悄变成空/错值，而 render→parse→render
   往返等价**照样成立**（两边一起错）。故必须直接断言字段值。
2. **`@` 前缀的 ref 前瞻**：没有 `(?!@)`，`[model · @cc]`（无 ref 的新行）会把
   `@cc` 读成 ref、writer 恒空。判别力对照 `test_writer_only_line_...`。
3. **空 writer 不渲染**：历史条目与用户直编的裸行要逐字保持原样。
"""

from __future__ import annotations

from bladex_core.ledger import (
    ACTOR_MODEL, LedgerEntry, add_entry, new_ledger, parse_ledger_md,
    render_ledger_md,
)


def _with(entries: list[LedgerEntry]):
    led = new_ledger(ledger_id="ldg-fb3", title="写者标记",
                     created_at="2026-09-07T02:00:00+00:00")
    for e in entries:
        led = add_entry(led, "verified", e, actor=ACTOR_MODEL)
    return led


class TestRender:
    def test_writer_renders_after_source_and_ref(self):
        md = render_ledger_md(_with([LedgerEntry(
            text="gate 全绿", source="tool", ref="gate#1", writer="claude-code")]))
        assert "- gate 全绿  <sub>[tool · gate#1 · @claude-code]</sub>" in md

    def test_writer_without_ref(self):
        md = render_ledger_md(_with([LedgerEntry(
            text="没有证据的一条", source="model", writer="hermes:default")]))
        assert "<sub>[model · @hermes:default]</sub>" in md

    def test_empty_writer_renders_nothing(self):
        """历史条目与用户直编的裸行逐字不变——`@` 不许以空值形态漏出来。"""
        md = render_ledger_md(_with([LedgerEntry(text="老条目", source="model",
                                                 ref="r1")]))
        assert "- 老条目  <sub>[model · r1]</sub>" in md
        assert "@" not in md.split("## Verified")[1]

    def test_render_is_still_deterministic(self):
        led = _with([LedgerEntry(text="x", source="model", writer="codex")])
        assert render_ledger_md(led) == render_ledger_md(led)


class TestParse:
    def test_round_trip_keeps_writer(self):
        led = _with([
            LedgerEntry(text="带 ref 带写者", source="tool", ref="gate#1",
                        writer="claude-code"),
            LedgerEntry(text="只有写者", source="model", writer="hermes:default"),
            LedgerEntry(text="都没有", source="model"),
        ])
        md = render_ledger_md(led)
        back = parse_ledger_md(md)
        got = [(e.text, e.source, e.ref, e.writer) for e in back.entries("verified")]
        assert got == [
            ("带 ref 带写者", "tool", "gate#1", "claude-code"),
            ("只有写者", "model", "", "hermes:default"),
            ("都没有", "model", "", ""),
        ]
        assert render_ledger_md(back) == md, "render→parse→render 仍逐字往返"

    def test_writer_only_line_does_not_become_a_ref(self):
        """判别力对照：去掉 `_ENTRY_RE` 里 ref 组的 `(?!@)` 前瞻 ⇒ 本条必红
        （`@hermes:default` 会被读成 ref，writer 恒空，而往返等价照样成立）。"""
        e = parse_ledger_md(
            "# t\n\n## Verified\n\n- 只有写者  <sub>[model · @hermes:default]</sub>\n"
        ).entries("verified")[0]
        assert (e.ref, e.writer) == ("", "hermes:default")

    def test_old_two_segment_lines_parse_as_before(self):
        """存量池全是这两种形态。解析改坏不会报错，只会静默错值。"""
        md = ("# t\n\n## Verified\n\n"
              "- 带 ref  <sub>[tool · gate_check#20260728]</sub>\n"
              "- 只有 source  <sub>[model]</sub>\n"
              "- 裸行\n")
        got = [(e.text, e.source, e.ref, e.writer)
               for e in parse_ledger_md(md).entries("verified")]
        assert got == [
            ("带 ref", "tool", "gate_check#20260728", ""),
            ("只有 source", "model", "", ""),
            ("裸行", "user", "", ""),      # 裸行来源默认 user（既有语义）
        ]

    def test_agent_id_with_colon_survives(self):
        """agent 完整 id 带冒号（`hermes:default`）——分隔符不许被它撑破。"""
        e = parse_ledger_md(
            "# t\n\n## Verified\n\n- x  <sub>[tool · r · @hermes:default]</sub>\n"
        ).entries("verified")[0]
        assert (e.ref, e.writer) == ("r", "hermes:default")


class TestSchemaCompat:
    def test_old_dump_without_writer_still_validates(self):
        """Hub 里的存量 `ledger_update` payload 没有这个字段——重放不得失败。"""
        from bladex_core.ledger import Ledger
        led = Ledger.model_validate({
            "ledger_id": "ldg-old",
            "sections": {"next": [{"text": "老条目", "source": "model",
                                   "ref": "r"}]},
            "section_order": ["goal", "next"],
        })
        assert led.entries("next")[0].writer == ""

    def test_writer_is_in_the_dump(self):
        led = _with([LedgerEntry(text="x", source="model", writer="codex")])
        assert led.model_dump()["sections"]["verified"][0]["writer"] == "codex"
