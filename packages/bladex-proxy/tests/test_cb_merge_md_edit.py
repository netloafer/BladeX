"""批 CB · 拍板 (41)-b-1：`edit --from-file` / 编辑器路径不改用户没碰过的条目。

渲染再解析不是恒等的。11-02 快照上会变样的真实形态有两种（内容是构造的，形态是真实的）：
- A：`text` 本身带 `  <sub>[tool · …]</sub>` 尾巴，解析时被拆成别的 text / source / ref（3 条，同一本）；
- B：`ref` 里带 ` · @writer`，解析时 ref 变短、writer 变成 `x · @x`（二十多条）。
"""

from __future__ import annotations

import pytest
from bladex_core.ledger import (
    LedgerEntry,
    LedgerError,
    merge_md_edit,
    new_ledger,
    parse_ledger_md,
    render_ledger_md,
)

SHAPE_A = [
    LedgerEntry(text="峰值 128GB。  <sub>[tool · install.md + models.md + 文档]</sub>", source="model"),
    LedgerEntry(text="可动态分配。  <sub>[tool · arXiv html 2608.1 + models]</sub>", source="model"),
    LedgerEntry(text="社区验证短。  <sub>[tool · CSDN + 新浪 + InfoQ]</sub>", source="model"),
]
SHAPE_B = [
    LedgerEntry(text="写入成功", source="tool", ref="write_file 2026-09-22 · @hermes:default",
                writer="hermes:default"),
    LedgerEntry(text="脚本复核", source="tool", ref="tool · 复核脚本 · @dsh", writer="dsh"),
]
PLAIN = LedgerEntry(text="普通条目", source="model", ref="r1", writer="codex")


def _ledger():
    led = new_ledger(ledger_id="ldg-aaaaaaaaaaaa", title="任务", goal="第一段\n第二段", goal_source="model",
                     created_at="2026-11-02T00:00:00+00:00")
    return led.model_copy(update={"rev": 7, "sections": {
        **led.sections, "verified": [*SHAPE_A, *SHAPE_B, PLAIN],
        "next": [LedgerEntry(text="下一步", source="model")]}})


def test_precondition_roundtrip_really_changes_these_entries() -> None:
    """这条红了说明解析器被修过，上面的形态就不再是病例，用例要重看。"""
    cur = _ledger()
    parsed = parse_ledger_md(render_ledger_md(cur), base=cur)
    got, want = parsed.sections["verified"], cur.sections["verified"]
    assert [a != b for a, b in zip(got, want, strict=True)] == [True] * 5 + [False]


def test_untouched_file_is_no_edit_at_all() -> None:
    cur = _ledger()
    assert merge_md_edit(cur, render_ledger_md(cur)).empty


@pytest.mark.parametrize("victim", range(3))
def test_shape_a_entry_stays_verbatim_when_another_entry_changes(victim: int) -> None:
    cur = _ledger()
    edit = merge_md_edit(cur, render_ledger_md(cur).replace("- 下一步", "- 下一步改过了"))
    assert set(edit.sections) == {"next"}, "没改的段不进请求"
    # 同段里改另一条：SHAPE_A[victim] 仍是当前版的原对象
    text = render_ledger_md(cur).replace("- 普通条目", "- 普通条目改过了")
    merged = merge_md_edit(cur, text).sections["verified"]
    assert merged[victim] == SHAPE_A[victim]
    assert merged[victim].source == "model" and merged[victim].ref == ""


@pytest.mark.parametrize("victim", range(2))
def test_shape_b_entry_stays_verbatim_when_another_entry_changes(victim: int) -> None:
    cur = _ledger()
    merged = merge_md_edit(cur, render_ledger_md(cur).replace("- 普通条目", "- 普通条目改过了")).sections["verified"]
    assert merged[3 + victim] == SHAPE_B[victim]


def test_only_the_changed_entry_changes_and_new_lines_are_users() -> None:
    cur = _ledger()
    text = render_ledger_md(cur).replace("- 普通条目  <sub>[model · r1 · @codex]</sub>",
                                         "- 普通条目改过了\n- 用户加的一条")
    merged = merge_md_edit(cur, text).sections["verified"]
    assert merged[:5] == cur.sections["verified"][:5]
    assert [(e.text, e.source) for e in merged[5:]] == [("普通条目改过了", "user"), ("用户加的一条", "user")]


def test_delete_and_reorder_follow_the_file_and_duplicates_pair_up() -> None:
    led = new_ledger(ledger_id="ldg-aaaaaaaaaaaa", goal="g", created_at="2026-11-02T00:00:00+00:00")
    a = LedgerEntry(text="同文", source="model", ref="r1")
    b = LedgerEntry(text="同文", source="model", ref="r1", writer="")
    c = LedgerEntry(text="另一条", source="tool")
    cur = led.model_copy(update={"sections": {**led.sections, "core": [a, b, c]}})
    lines = render_ledger_md(cur).splitlines()
    i = lines.index("- 另一条  <sub>[tool]</sub>")
    j = lines.index("- 同文  <sub>[model · r1]</sub>")
    lines[i], lines[j] = lines[j], lines[i]
    del lines[lines.index("- 同文  <sub>[model · r1]</sub>")]
    merged = merge_md_edit(cur, "\n".join(lines) + "\n").sections["core"]
    assert merged == [c, a]


def test_goal_and_title_only_when_changed() -> None:
    cur = _ledger()
    edit = merge_md_edit(cur, render_ledger_md(cur).replace("第二段", "第二段改了").replace("# 任务", "# 新标题"))
    assert edit.goal == "第一段\n第二段改了"
    assert edit.title == "新标题"
    assert edit.sections == {}


def test_goal_with_blank_line_not_rewritten_when_untouched() -> None:
    """解析丢空行：Goal 带空行的账本，用户只改条目时 Goal 不能算改过。"""
    led = new_ledger(ledger_id="ldg-aaaaaaaaaaaa", goal="第一段\n\n第二段", created_at="2026-11-02T00:00:00+00:00")
    edit = merge_md_edit(led, render_ledger_md(led).replace("## Core\n\n_(empty)_", "## Core\n\n- 加一条"))
    assert edit.goal is None
    assert [e.text for e in edit.sections["core"]] == ["加一条"]


def test_new_section_is_declared_and_deleted_heading_empties() -> None:
    cur = _ledger()
    edit = merge_md_edit(cur, render_ledger_md(cur) + "\n## Lessons\n\n- 教训\n")
    assert edit.new_sections == ["lessons"]
    assert edit.sections["lessons"][0].text == "教训"
    gone = render_ledger_md(cur).replace("## Next\n\n- 下一步  <sub>[model]</sub>\n", "")
    assert merge_md_edit(cur, gone).sections == {"next": []}


def test_file_for_another_ledger_is_refused() -> None:
    cur = _ledger()
    with pytest.raises(LedgerError, match="ldg-bbbbbbbbbbbb"):
        merge_md_edit(cur, render_ledger_md(cur).replace("ldg-aaaaaaaaaaaa", "ldg-bbbbbbbbbbbb"))


def test_unreadable_line_in_section_is_refused_not_dropped() -> None:
    cur = _ledger()
    with pytest.raises(LedgerError, match="用户随手写的一行"):
        merge_md_edit(cur, render_ledger_md(cur).replace("- 下一步", "用户随手写的一行\n- 下一步"))


def test_multiline_entry_untouched_is_not_refused() -> None:
    led = new_ledger(ledger_id="ldg-aaaaaaaaaaaa", goal="g", created_at="2026-11-02T00:00:00+00:00")
    e = LedgerEntry(text="两行\n第二行", source="model")
    cur = led.model_copy(update={"sections": {**led.sections, "core": [e]}})
    assert merge_md_edit(cur, render_ledger_md(cur)).empty
