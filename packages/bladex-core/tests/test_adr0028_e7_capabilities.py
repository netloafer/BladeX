"""ADR-0028 E7：C 类新能力（core 侧纯函数——文件索引 / Project / 规则文件与画像）。

这批不是"修 bug"，是**补缺口**：ADR-0026 定的三轴坐标里，主题轴
`Project → Matter → TaskUnit` 的最上层一直是空的；文件只有裸路径没有内容；
规则文件正文在信封剥离时被整段丢掉；工具习惯只记了"用过几次"、不记"好不好使"。

（Memory Index 存储与注入接线见 proxy 侧 `test_adr0028_e7_storage.py`。）
"""

from __future__ import annotations

import pytest
from bladex_core.file_index import (
    FileEntry,
    file_id_for,
    make_entry,
    mime_class_of,
    query_wants_files,
    worth_indexing,
)
from bladex_core.project import (
    Project,
    compose_description,
    compose_progress,
    infer_root_from_paths,
    infer_root_from_rulefile,
    languages_of,
    matter_belongs_to_project,
    name_from_root,
    project_id_for,
    render_project_card,
)
from bladex_core.rulefile import (
    extract_rule_files,
    norm_rulefile_name,
    render_agent_md,
    render_profile_card,
    render_user_md,
)


# ── E7.1 文件内容索引 ───────────────────────────────────────────────────


@pytest.mark.parametrize("path,cls", [
    ("/a/b/README.md", "doc"),
    ("/a/b/memory_index.py", "code"),
    ("/a/b/shot.png", "image"),
    ("/a/b/talk.mp3", "av"),
    ("/a/b/blob.bin", "other"),
])
def test_mime_class(path, cls):
    assert mime_class_of(path) == cls


def test_file_id_is_stable_per_path():
    assert file_id_for("/a/b.md") == file_id_for("/a/b.md")
    assert file_id_for("/a/b.md") != file_id_for("/a/c.md")


def test_worth_indexing_requires_read_write_tool():
    """检索类工具的路径参数是"在哪找"，不是"操作了什么"——E1.5 黑名单的反向。"""
    body = "x" * 500
    assert worth_indexing("Read", "/a/b.md", body) is True
    assert worth_indexing("search_files", "/a/b.md", body) is False
    assert worth_indexing("Grep", "/a/b.md", body) is False


def test_worth_indexing_min_chars():
    assert worth_indexing("Read", "/a/b.md", "short") is False
    assert worth_indexing("Read", "/a/b.md", "x" * 200) is True


def test_image_and_av_indexed_as_metadata_only():
    """v1 只录元数据，不做转写（任务卡写死）。"""
    assert worth_indexing("Read", "/a/shot.png", "") is True
    assert worth_indexing("Read", "/a/talk.mp3", "") is True


def test_entry_embed_text_excludes_path():
    """向量建在**有语义**的东西上——路径正是噪声源（裸路径侵占 20.7% 的根因）。"""
    e = make_entry("/Users/alice/dev/BladeX/README.md", "x" * 300,
                   summary="BladeX 的安装与快速上手", keywords=["安装", "quickstart"])
    text = e.embed_text()
    assert "/Users/alice" not in text
    assert "README.md" in text and "BladeX 的安装与快速上手" in text


def test_entry_content_hash_changes_with_content():
    a = make_entry("/a/b.md", "版本一")
    b = make_entry("/a/b.md", "版本二")
    assert a.file_id == b.file_id            # 同 path 同 id（覆盖更新的键）
    assert a.content_hash != b.content_hash  # hash 变 → 需要重新索引


@pytest.mark.parametrize("query,ids,want", [
    ("那个 md 文件里写了什么", [], True),
    ("看看 README 的内容", [], True),
    ("检查 memory_index.py 的实现", ["memory_index.py"], True),
    ("帮我看 BladeX 502 问题", [], False),
])
def test_query_wants_files(query, ids, want):
    assert query_wants_files(query, ids) is want


def test_file_entry_caps():
    e = FileEntry(file_id="x", path="/a/b.md",
                  summary="s" * 500, keywords=[f"k{i}" for i in range(20)])
    assert len(make_entry("/a/b.md", "c", summary=e.summary,
                          keywords=e.keywords).summary) == 300
    assert len(make_entry("/a/b.md", "c", summary="s",
                          keywords=e.keywords).keywords) == 8


# ── E7.2 Project 实体 ──────────────────────────────────────────────────


def test_project_id_deterministic():
    assert project_id_for("/dev/BladeX") == project_id_for("/dev/BladeX")
    assert len(project_id_for("/dev/BladeX")) == 12


def test_infer_root_from_rulefile():
    """识别①：CLAUDE.md 在哪，项目根就在哪（最强信号）。"""
    assert infer_root_from_rulefile("/Users/j/dev/BladeX/CLAUDE.md") == "/Users/j/dev/BladeX"


def test_infer_root_from_paths_needs_five_hits_and_depth_two():
    """识别②：同 session 内 ≥5 次且深度 ≥2 的公共前缀（任务卡写死）。"""
    paths = [f"/Users/j/dev/BladeX/pkg/f{i}.py" for i in range(5)]
    assert infer_root_from_paths(paths).startswith("/Users/j/dev/BladeX")

    # 只出现 4 次 → 不够
    assert infer_root_from_paths(paths[:4]) == ""
    # 深度 1 → 不够（否则所有项目都归到 /Users）
    assert infer_root_from_paths(["/tmp/a.py"] * 9) == ""


def test_infer_root_prefers_longest_prefix():
    """取最长（最具体）前缀——否则所有项目都会归到 /Users 底下。"""
    paths = [f"/Users/j/dev/BladeX/pkg/f{i}.py" for i in range(6)]
    root = infer_root_from_paths(paths)
    assert root == "/Users/j/dev/BladeX/pkg"


def test_languages_from_extensions():
    paths = ["a.py", "b.py", "c.ts", "d.md", "e.py", "f.ts"]
    assert languages_of(paths) == ["Python", "TypeScript", "Markdown"]


def test_name_from_root():
    assert name_from_root("/Users/j/dev/BladeX/") == "BladeX"


def test_matter_belongs_to_project_by_ratio():
    """判定：成员 fact 的路径落在 root_path 下的比例 ≥0.5（任务卡写死）。"""
    root = "/dev/BladeX"
    assert matter_belongs_to_project(
        ["/dev/BladeX/a.py", "/dev/BladeX/b.py", "/other/c.py"], root) is True
    assert matter_belongs_to_project(
        ["/dev/BladeX/a.py", "/other/b.py", "/other/c.py"], root) is False
    assert matter_belongs_to_project([], root) is False


def test_compose_fields_are_deterministic_concatenation():
    """description / progress_note 是**确定性拼接**，不调 LLM（任务卡写死）。"""
    assert compose_description(["卡 A", "卡 B", "卡 A"]) == "卡 A / 卡 B"
    assert compose_progress(["进展一", " ", "进展二"]) == "进展一 | 进展二"


def test_project_card_max_four_lines():
    p = Project(project_id="pipeline", name="BladeX", root_path="/dev/BladeX",
                description="记忆检索重构", progress_note="E6 完成",
                languages=["Python"])
    lines = render_project_card(p, active_matters=3)
    assert len(lines) <= 4
    assert lines[0].startswith("[Project: BladeX]")
    assert any("活跃 Matter: 3" in ln for ln in lines)


# ── E7.3 规则文件 + 画像渲染 ───────────────────────────────────────────


def test_extract_rule_file_body():
    """信封剥离时被整段丢掉的规则文件正文，现在存副本。"""
    text = (
        "Contents of /Users/j/dev/BladeX/CLAUDE.md (project instructions):\n\n"
        "# BladeX 协作指南\n跑测试只用 .venv/bin/python -m pytest\n"
    )
    got = extract_rule_files(text)
    assert len(got) == 1
    name, body, chash = got[0]
    assert name == "claude.md"
    assert "跑测试只用" in body
    assert len(chash) == 16


def test_extract_multiple_rule_files_in_order():
    text = ("Contents of /a/CLAUDE.md:\n规则甲\n"
            "Contents of /a/AGENTS.md:\n规则乙\n")
    got = extract_rule_files(text)
    assert [g[0] for g in got] == ["claude.md", "agents.md"]
    assert "规则甲" in got[0][1] and "规则乙" in got[1][1]


def test_rule_file_name_normalized_to_tail():
    """同一份 CLAUDE.md 在不同机器上路径不同——按尾段规范化才对得上。"""
    assert norm_rulefile_name("/Users/a/dev/X/CLAUDE.md") == "claude.md"
    assert norm_rulefile_name("C:\\proj\\AGENTS.md") == "agents.md"


def test_extract_returns_nothing_without_marker():
    assert extract_rule_files("这段文本里只是提了一嘴 CLAUDE.md，没有正文") == []


def test_render_user_md_sections():
    md = render_user_md(
        user_id="u1", key_count=2,
        agents=[{"agent_id": "claude-code", "turns": 410, "last_seen": "2026-08-05"}],
        recent_activity=[{"agent_id": "claude-code", "turns": 32}],
        preferences=["用户希望回复简洁"],
        coding_preferences=["用户偏好 pytest 跑真实路径"],
    )
    assert "# USER" in md
    assert "claude-code" in md
    assert "## 近 7 日活跃度" in md
    assert "用户希望回复简洁" in md
    assert "## 编程偏好" in md


def test_render_agent_md_with_tool_success_rate():
    """E7.4 的消费面：工具带成功率；失败率高的加一行"常见失败"。"""
    md = render_agent_md(
        agent_base="codex",
        tools=[
            {"name": "Read", "calls": 40, "ok": 40, "err": 0},
            {"name": "Bash", "calls": 10, "ok": 5, "err": 5, "last_err_class": "error"},
        ],
        own_facts=["codex 习惯先跑 pytest 再改代码"],
        rule_files=["agents.md"],
    )
    assert "Read (40 次, ok 100%)" in md
    assert "⚠ 常见失败: error" in md
    assert "codex 习惯先跑 pytest 再改代码" in md


def test_agent_md_no_warning_when_sample_too_small():
    """calls < 5 时不给失败提示——样本太小的比率没有意义。"""
    md = render_agent_md(agent_base="codex", own_facts=[], rule_files=[],
                         tools=[{"name": "Bash", "calls": 2, "ok": 0, "err": 2,
                                 "last_err_class": "error"}])
    assert "常见失败" not in md


def test_profile_card_v2_picks_complete_sentences():
    """render v2：只选完整句子，≤5 行——取代"路径串联乱码"那两行。"""
    user_md = render_user_md(
        user_id="u1", key_count=1,
        agents=[{"agent_id": "hermes", "turns": 10}],
        recent_activity=[{"agent_id": "hermes", "turns": 3}],
        preferences=["用户希望回复简洁", "用户不喜欢 emoji"],
        coding_preferences=[],
    )
    lines = render_profile_card(user_md)
    assert len(lines) <= 5
    assert any("用户希望回复简洁" in ln for ln in lines)
    # 统计行不进卡（"hermes: 3 turns" 这类）
    assert not any(ln.endswith(" turns") for ln in lines)


def test_profile_card_empty_when_no_profile():
    assert render_profile_card("", "") == []
