"""ADR-0028 E7：C 类新能力的 Memory Index 存储与注入接线（proxy 侧）。

真实路径：MemoryIndex（LanceDB + meta + FTS5）真写真查。
覆盖 E7.1 文件索引 / E7.2 Project 实体 / E7.3 规则文件副本与画像文件 / E7.4 工具结果维度。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from bladex_core.fact import Fact, ItemKind
from bladex_core.file_index import make_entry
from bladex_core.matter import (
    EdgeProvenance,
    EdgeRelation,
    EdgeTargetType,
    Matter,
    MatterEdge,
    MatterStatus,
)
from bladex_core.project import Project, project_id_for
from bladex_proxy.models import Identity, ToolEvent, Turn
from bladex_proxy.storage.memory_index import MemoryIndex


class _Embedder:
    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 32
            v[hash(t) % 32] = 1.0
            out.append(v)
        return out

    @property
    def available(self):
        return True


@pytest.fixture()
def index(tmp_path):
    store = MemoryIndex(tmp_path / "index", embedder=_Embedder(), read_only=False)
    store.open()
    yield store
    store.close()


def _turn(tool_events: list[ToolEvent], *, agent: str = "codex") -> Turn:
    return Turn(
        identity=Identity(user_id="u1", agent_id=agent, session_id="s1"),
        model="m", request_messages=[{"role": "user", "content": "干活"}],
        response_text="好了", tool_events=tool_events,
        ts=datetime.now(UTC),
    )


# ── E7.1 文件内容索引 ───────────────────────────────────────────────────


def test_file_index_upsert_and_search(index):
    entry = make_entry("/dev/BladeX/README.md", "x" * 300,
                       summary="BladeX 的安装与快速上手", keywords=["安装"],
                       user_id="u1")
    entry.vector = _Embedder().embed([entry.embed_text()])[0]
    index.upsert_file_index(entry)

    rows = index.search_files(entry.vector, k=3, user_id="u1")
    assert rows and rows[0]["path"] == "/dev/BladeX/README.md"
    assert rows[0]["summary"] == "BladeX 的安装与快速上手"


def test_file_index_same_path_new_hash_overwrites(index):
    """files 表不是记忆总账（历史在 Memory Hub），无 as-of 义务 → 同 path 覆盖更新。"""
    emb = _Embedder()
    for body, summary in (("版本一" * 100, "旧摘要"), ("版本二" * 100, "新摘要")):
        e = make_entry("/dev/BladeX/x.md", body, summary=summary, user_id="u1")
        e.vector = emb.embed([e.embed_text()])[0]
        index.upsert_file_index(e)

    rows = index.search_files(emb.embed(["新摘要"])[0], k=5, user_id="u1")
    paths = [r["path"] for r in rows]
    assert paths.count("/dev/BladeX/x.md") == 1        # 只剩一行
    assert rows[0]["summary"] == "新摘要"


def test_index_files_from_turn_skips_search_tools(index):
    """与 E1.5 同边界：检索类工具不产文件索引。"""
    turn = _turn([
        ToolEvent(tool_name="search_files", direction="call",
                  arguments={"file_path": "/dev/BladeX/x.md"}, tool_call_id="c1"),
        ToolEvent(tool_name="search_files", direction="result",
                  result="y" * 500, tool_call_id="c1"),
    ])
    assert index._index_files_from_turn(turn, "u1") == 0  # noqa: SLF001


def test_index_files_from_turn_indexes_read(index):
    turn = _turn([
        ToolEvent(tool_name="Read", direction="call",
                  arguments={"file_path": "/dev/BladeX/README.md"}, tool_call_id="c1"),
        ToolEvent(tool_name="Read", direction="result",
                  result="安装步骤：" + "x" * 400, tool_call_id="c1"),
    ])
    assert index._index_files_from_turn(turn, "u1") == 1  # noqa: SLF001
    # 同内容再来一次 → hash 命中，不重复索引（也不重复调 LLM）
    assert index._index_files_from_turn(turn, "u1") == 0  # noqa: SLF001


# ── E7.2 Project 实体 ──────────────────────────────────────────────────


def test_project_roundtrip(index):
    pid = project_id_for("/dev/BladeX")
    index.upsert_project(Project(project_id=pid, root_path="/dev/BladeX", name="BladeX"))
    got = index.get_project(pid)
    assert got is not None and got.name == "BladeX" and got.version == 1
    assert [x.project_id for x in index.all_projects()] == [pid]


def test_matter_linked_to_project_by_path_ratio(index):
    pid = project_id_for("/dev/BladeX")
    index.upsert_project(Project(project_id=pid, root_path="/dev/BladeX", name="BladeX"))
    index.add_matter(Matter(matter_id="m1", title="检索重构", summary="E6",
                         status=MatterStatus.ACTIVE))
    for i, path in enumerate(["/dev/BladeX/a.py", "/dev/BladeX/b.py", "/other/c.py"]):
        f = Fact(id=f"f{i}", content=f"文件 {path}", item_kind=ItemKind.FILE_REF,
                 subject=path, attribute="file", source_user_id="u1")
        index.add_fact(f)
        index.add_edge(MatterEdge(matter_id="m1", target_type=EdgeTargetType.FACT,
                               target_key=f.id, provenance=EdgeProvenance.AUTO))

    assert index.link_matters_to_project(pid, "/dev/BladeX") == 1
    edges = index.get_edges("m1")
    assert any(e.relation == EdgeRelation.PART_OF and e.target_key == pid for e in edges)


def test_project_fields_refreshed_deterministically(index):
    pid = project_id_for("/dev/BladeX")
    index.upsert_project(Project(project_id=pid, root_path="/dev/BladeX", name="BladeX"))
    index.add_matter(Matter(matter_id="m1", title="检索重构", summary="E6 融合已落地",
                         status=MatterStatus.ACTIVE))
    index.add_edge(MatterEdge(matter_id="m1", target_type=EdgeTargetType.MATTER,
                           target_key=pid, relation=EdgeRelation.PART_OF,
                           provenance=EdgeProvenance.AUTO))
    index.refresh_project_fields(pid)

    proj = index.get_project(pid)
    assert proj.description == "检索重构"
    assert proj.progress_note == "E6 融合已落地"


def test_project_card_triggered_by_root_path_in_request(index):
    """触发条件是**确定性匹配**：请求正文里出现 root_path 前缀。"""
    pid = project_id_for("/dev/BladeX")
    index.upsert_project(Project(project_id=pid, root_path="/dev/BladeX",
                              name="BladeX", description="记忆检索重构"))
    lines = index.project_card_for("请看 /dev/BladeX/packages 下的实现")
    assert lines and lines[0].startswith("[Project: BladeX]")
    assert index.project_card_for("跟这个项目无关的一句话") == []


def test_maintain_projects_discovers_from_rulefile(index):
    """识别①：CLAUDE.md 的 file_ref → 目录即 root_path。"""
    f = Fact(id="f1", content="文件 /dev/BladeX/CLAUDE.md", item_kind=ItemKind.FILE_REF,
             subject="/dev/BladeX/CLAUDE.md", attribute="file", source_user_id="u1")
    index.add_fact(f)
    assert index.maintain_projects() == 1
    assert [x.root_path for x in index.all_projects()] == ["/dev/BladeX"]


def test_discover_project_from_paths_needs_threshold(index):
    paths = [f"/dev/BladeX/pkg/f{i}.py" for i in range(6)]
    pid = index.discover_project_from_paths(paths)
    assert pid and index.get_project(pid).root_path == "/dev/BladeX/pkg"
    # 不足 5 次 → 不识别（识别不出就不识别，不猜）
    assert index.discover_project_from_paths(["/x/y/a.py"] * 3) == ""


# ── E7.3 规则文件副本 + 画像文件 ───────────────────────────────────────


def test_rule_file_body_captured_and_versioned(index):
    text = ("Contents of /dev/BladeX/CLAUDE.md (project instructions):\n\n"
            "跑测试只用 .venv/bin/python -m pytest\n")
    index._capture_rule_file_bodies("claude-code", [text])  # noqa: SLF001
    assert index.get_rule_files("claude-code") == ["claude.md"]

    # hash 没变 → 不重复写（版本不变）
    before = index._profile_get("rulefile/claude-code/claude.md")  # noqa: SLF001
    index._capture_rule_file_bodies("claude-code", [text])  # noqa: SLF001
    assert index._profile_get("rulefile/claude-code/claude.md") == before  # noqa: SLF001


def test_rule_files_are_per_agent(index):
    text = "Contents of /dev/BladeX/AGENTS.md:\n规则乙\n"
    index._capture_rule_file_bodies("codex", [text])  # noqa: SLF001
    assert index.get_rule_files("codex") == ["agents.md"]
    assert index.get_rule_files("hermes") == []


def test_build_user_md_and_agent_md(index):
    pref = Fact(id="pipeline", content="用户希望回复简洁", item_kind=ItemKind.PREFERENCE,
                source_user_id="u1", importance=0.9, entities=["pytest"])
    pref.agent_id = "claude-code"
    index.add_fact(pref)
    index._capture_rule_file_bodies(  # noqa: SLF001
        "claude-code", ["Contents of /dev/BladeX/CLAUDE.md:\n规则甲\n"])

    user_md = index.build_user_md("u1")
    assert "用户希望回复简洁" in user_md
    assert "claude-code" in user_md
    assert "## 编程偏好" in user_md          # entities 命中技术词表

    agent_md = index.build_agent_md("claude-code", user_id="u1")
    assert "claude.md" in agent_md

    got_user, got_agent = index.get_profile_md("u1", "claude-code")
    assert got_user == user_md and got_agent == agent_md


def test_profile_card_v2_used_when_profile_md_exists(index):
    """render v2：画像卡改由 USER.md 节选，取代"路径串联乱码"的旧渲染。"""
    pref = Fact(id="pipeline", content="用户希望回复简洁不要 emoji",
                item_kind=ItemKind.PREFERENCE, source_user_id="u1", importance=0.9)
    pref.agent_id = "hermes"
    index.add_fact(pref)
    index.build_user_md("u1")

    cards = index.get_profile_cards("u1", "hermes")
    flat = [ln for card in cards for ln in card]
    assert any("用户希望回复简洁不要 emoji" in ln for ln in flat)
    assert all("使用规则文件" not in ln for ln in flat)   # 旧渲染退役
    assert len(flat) <= 5


# ── E7.4 工具习惯：使用结果维度 ────────────────────────────────────────


def test_tool_results_counted_ok_and_err(index):
    turn = _turn([
        ToolEvent(tool_name="Bash", direction="call", arguments={}, tool_call_id="c1"),
        ToolEvent(tool_name="Bash", direction="result",
                  result="Traceback (most recent call last): ...", tool_call_id="c1"),
        ToolEvent(tool_name="Bash", direction="call", arguments={}, tool_call_id="c2"),
        ToolEvent(tool_name="Bash", direction="result", result="ok, done", tool_call_id="c2"),
    ])
    index._update_tool_results("codex", turn)  # noqa: SLF001

    stats = index.get_tool_stats("codex")
    assert stats and stats[0]["name"] == "Bash"
    assert stats[0]["calls"] == 2 and stats[0]["ok"] == 1 and stats[0]["err"] == 1
    assert stats[0]["last_err_class"] == "traceback"


@pytest.mark.parametrize("result,failed", [
    ("Error: file not found", True),
    ("错误：目录不存在", True),
    ("permission denied", True),
    ("一切正常，写入完成", False),
])
def test_tool_result_classification(result, failed):
    assert MemoryIndex._classify_tool_result(result)[0] is failed


def test_tool_result_only_scans_head():
    """只看前 200 字符——正文里偶然出现 "error" 不该把整次调用判失败。"""
    body = "一切正常。" * 60 + "这里才提到 error"
    assert MemoryIndex._classify_tool_result(body)[0] is False


def test_tool_stats_isolated_per_agent(index):
    index._update_tool_results("codex", _turn([  # noqa: SLF001
        ToolEvent(tool_name="Read", direction="call", arguments={}, tool_call_id="c1"),
        ToolEvent(tool_name="Read", direction="result", result="ok", tool_call_id="c1"),
    ]))
    assert index.get_tool_stats("codex")
    assert index.get_tool_stats("hermes") == []      # audience 隔离不变


# ── E7.1 真实 Memory Hub 复核修正（2026-08-05）：白名单与配对都曾使通道恒空转 ──


def test_real_world_tool_names_are_whitelisted():
    """初版白名单是"按通用命名猜"的，实测一次都不会命中。

    真实 Memory Hub：Codex 用 `read_file`（324 次）、Claude Code 用 `Read`（90 次），
    两个都不在初版表里 —— 又一例"机制写完了但从不触发"。
    """
    from bladex_core.file_index import READ_WRITE_TOOLS

    for name in ("read_file", "read", "write", "edit", "apply_patch"):
        assert name in READ_WRITE_TOOLS, name
    # 检索类工具仍然出局（与 E1.5 黑名单互补，不能混）
    for name in ("search_files", "glob", "grep", "exec_command", "web_search"):
        assert name not in READ_WRITE_TOOLS, name


def test_path_key_includes_path_only_for_file_index():
    """`path` 只在文件索引这条通道收，`_extract_file_refs` 仍然不收。

    两处规则**故意不同**：E1.5 删 `path` 是因为 `search_files{path:"/Users/alice"}`；
    而本通道先过了读写工具白名单，检索工具根本进不来。
    """
    from bladex_core.file_index import PATH_ARG_KEYS
    from bladex_proxy.storage.memory_index import _PATH_ARG_KEYS

    assert "path" in PATH_ARG_KEYS
    assert "path" not in _PATH_ARG_KEYS


def test_index_files_pairs_by_tool_name_when_call_has_no_cid(index):
    """🔴 真实捕获形态：call 事件**没有** tool_call_id，cid 只在 result 侧。

    只按 cid 配对 → 一条都配不上 → E7.1 恒空转（实测 1933 轮里 0 命中）。
    修复后按"同 tool_name 顺序"退回配对。
    """
    turn = _turn([
        # call 无 cid（真实形态）
        ToolEvent(tool_name="Read", direction="call",
                  arguments={"file_path": "/dev/BladeX/README.md"}),
        # result 有 cid，但那个 cid 在 call 侧根本不存在
        ToolEvent(tool_name="Read", direction="result",
                  result="安装步骤：" + "x" * 400, tool_call_id="call_abc123"),
    ])
    assert index._index_files_from_turn(turn, "u1") == 1  # noqa: SLF001


def test_index_files_pairing_is_order_deterministic(index):
    """同工具多次调用：按出现顺序一一对应（确定性，重建等价性不破）。"""
    turn = _turn([
        ToolEvent(tool_name="read_file", direction="call", arguments={"path": "/dev/a.md"}),
        ToolEvent(tool_name="read_file", direction="call", arguments={"path": "/dev/b.md"}),
        ToolEvent(tool_name="read_file", direction="result",
                  result="A 的内容" + "x" * 300, tool_call_id="c1"),
        ToolEvent(tool_name="read_file", direction="result",
                  result="B 的内容" + "y" * 300, tool_call_id="c2"),
    ])
    assert index._index_files_from_turn(turn, "u1") == 2  # noqa: SLF001

    from bladex_core.file_index import file_id_for
    emb = _Embedder()
    rows = index.search_files(emb.embed(["a.md"])[0], k=5, user_id="u1")
    ids = {r["file_id"] for r in rows}
    assert file_id_for("/dev/a.md") in ids and file_id_for("/dev/b.md") in ids


def test_search_tool_still_never_indexed_even_with_path_key(index):
    """回归护栏：`search_files{path:"/Users/alice"}` 这个事故不能借 `path` 键复活。"""
    turn = _turn([
        ToolEvent(tool_name="search_files", direction="call",
                  arguments={"path": "/Users/alice"}),
        ToolEvent(tool_name="search_files", direction="result",
                  result="z" * 500, tool_call_id="c1"),
    ])
    assert index._index_files_from_turn(turn, "u1") == 0  # noqa: SLF001


# ── 2026-08-05 真实库复核补：两处"写完了但没接线/被污染" ──


def test_matter_summary_excludes_file_ref_members(index):
    """Matter 摘要要回答"这件事是什么"，不是列一串路径。

    真实库实测：m-26bc3c841e08 的摘要变成了
    `文件 /Users/.../bladex-shortcomings-analysis.md | 文…` ——
    E1.1 防住了 L4 rewrite 污染，但**保底拼接**这条路同样会被 file_ref 占满。
    """
    index.add_matter(Matter(matter_id="m1", title="事", status=MatterStatus.ACTIVE))
    for i, (content, kind, subj) in enumerate([
        ("文件 /a/b.md", ItemKind.FILE_REF, "/a/b.md"),
        ("热路径预算定成了 200ms", ItemKind.ASSERTION, "热路径"),
        ("文件 /c/d.py", ItemKind.FILE_REF, "/c/d.py"),
    ]):
        f = Fact(id=f"f{i}", content=content, item_kind=kind, subject=subj,
                 source_user_id="u1")
        index.add_fact(f)
        index.add_edge(MatterEdge(matter_id="m1", target_type=EdgeTargetType.FACT,
                               target_key=f.id, provenance=EdgeProvenance.AUTO))
    index.update_matter_summary("m1")

    summary = index.get_matter("m1").summary
    assert "热路径预算定成了 200ms" in summary
    assert "文件 /" not in summary


def test_profile_docs_are_actually_generated(index):
    """E7.3 接线护栏：`build_user_md` / `build_agent_md` 必须有调用点。

    真实库实测 `profile/user_md` 恒为空 —— 方法写好了但没人调，
    于是①平面画像卡永远回落 v1（"路径串联乱码"那版）。
    """
    f = Fact(id="pipeline", content="用户希望回复简洁", item_kind=ItemKind.PREFERENCE,
             source_user_id="u1", importance=0.9)
    f.agent_id = "hermes:accept"
    index.add_fact(f)

    assert index.refresh_profile_docs() >= 2          # 1 个 user + 1 个 agent
    user_md, agent_md = index.get_profile_md("u1", "hermes")
    assert "用户希望回复简洁" in user_md
    assert agent_md.startswith("# AGENT hermes")


def test_refresh_profile_docs_is_deterministic(index):
    f = Fact(id="pipeline", content="用户偏好中文回复", item_kind=ItemKind.PREFERENCE,
             source_user_id="u1")
    f.agent_id = "codex"
    index.add_fact(f)
    index.refresh_profile_docs()
    first = index.get_profile_md("u1", "codex")
    index.refresh_profile_docs()
    assert index.get_profile_md("u1", "codex") == first


def test_refresh_profile_docs_noop_on_readonly(tmp_path):
    w = MemoryIndex(tmp_path / "index", embedder=_Embedder())
    w.open()
    w.add_fact(Fact(id="pipeline", content="x", item_kind=ItemKind.PREFERENCE,
                    source_user_id="u1"))
    w.close()
    r = MemoryIndex(tmp_path / "index", embedder=_Embedder(), read_only=True)
    r.open()
    assert r.refresh_profile_docs() == 0
