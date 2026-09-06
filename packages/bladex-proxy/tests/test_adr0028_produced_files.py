"""文件索引的**两类来源**与模型产出文件的本地副本（2026-08-05 拍板）。

## 为什么要分两类

E7.1 落地后，55 条文件索引全部来自工具 result（Read/read_file），
索引里看不出"这份东西是我读来的，还是它写给我的"。这两者的可回溯性完全不同：

- **会话里提到/读到的**（mentioned / read）：正文在磁盘上本来就有，原文件还在那儿，
  再存一份副本是冗余；
- **模型产出的**（produced）：那份内容**只存在于那一次回复里**。索引里的摘要只够
  检索，回溯不了正文——不落盘就永远找不回来了。

所以：① 索引标记不同（`origin`）；② 只有 produced 按 sha256 落盘存副本，
索引用 `blob_ref` 关联。

## 拍板的两条边界

- blob **跟随 forget 级联删**（"删除即删除"，不留影子副本）；
- blob **不进快照**（快照是 JSONL 实体流、从不拷目录——本文件有测试钉住它）。
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from bladex_core.fact import Fact, ItemKind
from bladex_core.file_index import (
    FileOrigin,
    extract_produced_files,
    file_id_for,
    make_entry,
)
from bladex_proxy.models import Identity, ToolEvent, Turn
from bladex_proxy.storage.blob_store import MAX_BLOB_BYTES, BlobStore, blob_id_for
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


def _turn(*, response: str = "", tool_events=None, agent: str = "codex") -> Turn:
    return Turn(
        identity=Identity(user_id="u1", agent_id=agent, session_id="s1"),
        model="m", request_messages=[{"role": "user", "content": "帮我改一下"}],
        response_text=response, tool_events=tool_events or [],
        ts=datetime.now(UTC),
    )


_PRODUCED_REPLY = """改好了，这是新版本：

**packages/bladex_core/fusion.py**

```python
def rrf_fuse(channels):
    scores = {}
    for ch in channels:
        for i, fid in enumerate(ch.ranked_ids):
            scores[fid] = scores.get(fid, 0.0) + 1.0 / (60 + i + 1)
    return scores
```

配置也顺手加了一段：

```toml:config/routing.toml
[[models]]
name = "openai/glm-5.2"
cache_ttl_s = 300
tier = "medium"
provider = "zhipu"
```
"""


# ── 抽取：认得出文件名的才算"一个文件" ──────────────────────────────────


def test_extract_reads_filename_from_bold_line_and_info_string():
    got = dict(extract_produced_files(_PRODUCED_REPLY))
    assert set(got) == {"packages/bladex_core/fusion.py", "config/routing.toml"}
    assert "def rrf_fuse" in got["packages/bladex_core/fusion.py"]
    assert 'name = "openai/glm-5.2"' in got["config/routing.toml"]


def test_extract_skips_blocks_without_a_filename():
    """认不出文件名的代码片段不是"一个文件"——宁漏勿误。"""
    text = "随手贴个片段：\n\n```\n" + "print(1)\n" * 30 + "```\n"
    assert extract_produced_files(text) == []


def test_extract_skips_tiny_blocks():
    text = "**a.py**\n\n```python\nx = 1\n```\n"
    assert extract_produced_files(text) == []


def test_extract_last_version_wins():
    """同一路径在一段回复里出现多次 → 取最后一次（改过的那版才是最终产物）。"""
    body_a = "# 第一版\n" + "a = 1\n" * 20
    body_b = "# 第二版\n" + "b = 2\n" * 20
    text = f"**x.py**\n\n```python\n{body_a}```\n\n改了一下：\n\n```python:x.py\n{body_b}```\n"
    got = dict(extract_produced_files(text))
    assert "第二版" in got["x.py"] and "第一版" not in got["x.py"]


def test_extract_is_deterministic():
    assert extract_produced_files(_PRODUCED_REPLY) == extract_produced_files(_PRODUCED_REPLY)


# ── 两类的索引标记不同 ─────────────────────────────────────────────────


def test_produced_and_read_carry_different_origin(index):
    index._index_produced_files_from_turn(_turn(response=_PRODUCED_REPLY), "u1")  # noqa: SLF001
    index._index_files_from_turn(_turn(tool_events=[  # noqa: SLF001
        ToolEvent(tool_name="Read", direction="call",
                  arguments={"file_path": "/dev/BladeX/README.md"}),
        ToolEvent(tool_name="Read", direction="result",
                  result="安装步骤：" + "x" * 400, tool_call_id="c1"),
    ]), "u1")

    by_path = {r["path"]: r for r in index._files_tbl.to_arrow().to_pylist()}  # noqa: SLF001
    assert by_path["packages/bladex_core/fusion.py"]["origin"] == "produced"
    assert by_path["/dev/BladeX/README.md"]["origin"] == "read"


def test_only_produced_gets_a_local_copy(index):
    index._index_produced_files_from_turn(_turn(response=_PRODUCED_REPLY), "u1")  # noqa: SLF001
    index._index_files_from_turn(_turn(tool_events=[  # noqa: SLF001
        ToolEvent(tool_name="Read", direction="call",
                  arguments={"file_path": "/dev/BladeX/README.md"}),
        ToolEvent(tool_name="Read", direction="result",
                  result="安装步骤：" + "x" * 400, tool_call_id="c1"),
    ]), "u1")

    by_path = {r["path"]: r for r in index._files_tbl.to_arrow().to_pylist()}  # noqa: SLF001
    assert by_path["packages/bladex_core/fusion.py"]["blob_ref"]       # 有副本
    assert not by_path["/dev/BladeX/README.md"]["blob_ref"]            # 读到的不存副本


def test_produced_body_is_retrievable(index):
    """这条通道的全部价值：**当时它给我写的那版，还能原样拿回来**。"""
    index._index_produced_files_from_turn(_turn(response=_PRODUCED_REPLY), "u1")  # noqa: SLF001
    body = index.get_file_body(file_id_for("packages/bladex_core/fusion.py"))
    assert body is not None
    assert "def rrf_fuse" in body and "1.0 / (60 + i + 1)" in body


def test_same_content_twice_is_indexed_once(index):
    turn = _turn(response=_PRODUCED_REPLY)
    assert index._index_produced_files_from_turn(turn, "u1") == 2  # noqa: SLF001
    assert index._index_produced_files_from_turn(turn, "u1") == 0  # noqa: SLF001


def test_produced_files_indexed_from_rebuild_path(index):
    """两类都挂在 rebuild 的同一处接线上（别只接一类）。"""
    turn = _turn(response=_PRODUCED_REPLY, tool_events=[
        ToolEvent(tool_name="read_file", direction="call",
                  arguments={"path": "/dev/BladeX/docs/x.md"}),
        ToolEvent(tool_name="read_file", direction="result",
                  result="文档正文" + "y" * 400, tool_call_id="c9"),
    ])
    index._index_files_from_turn(turn, "u1")            # noqa: SLF001
    index._index_produced_files_from_turn(turn, "u1")   # noqa: SLF001
    origins = {r["path"]: r["origin"]
               for r in index._files_tbl.to_arrow().to_pylist()}  # noqa: SLF001
    assert origins["/dev/BladeX/docs/x.md"] == "read"
    assert origins["config/routing.toml"] == "produced"


# ── blob store 本身 ────────────────────────────────────────────────────


def test_blob_store_is_content_addressed(tmp_path):
    store = BlobStore(tmp_path)
    a = store.put("同一份内容")
    b = store.put("同一份内容")
    assert a == b == blob_id_for("同一份内容")
    assert store.count() == 1                      # 内容寻址：只存一份
    assert store.get(a) == "同一份内容"


def test_blob_store_rejects_oversized(tmp_path):
    store = BlobStore(tmp_path)
    assert store.put("x" * (MAX_BLOB_BYTES + 1)) == ""
    assert store.count() == 0


def test_blob_store_readonly_is_noop(tmp_path):
    ro = BlobStore(tmp_path, read_only=True)
    assert ro.put("内容") == ""
    assert ro.count() == 0


def test_blob_store_leaves_no_tmp_files(tmp_path):
    store = BlobStore(tmp_path)
    store.put("内容")
    assert not list(store.root.rglob("*.tmp"))


# ── 生命周期：跟随 forget 级联删 ───────────────────────────────────────


def test_forget_file_ref_cascades_to_index_and_blob(index):
    """拍板项："删除即删除"，不留影子副本。"""
    index._index_produced_files_from_turn(_turn(response=_PRODUCED_REPLY), "u1")  # noqa: SLF001
    path = "packages/bladex_core/fusion.py"
    fid = file_id_for(path)
    assert index.get_file_body(fid) is not None

    # 一条指向同路径的 file_ref fact（真实管线里 file_ref 与文件索引并存）
    fact = Fact(id="f-ref", content=f"文件 {path}", item_kind=ItemKind.FILE_REF,
                subject=path, attribute="file", source_user_id="u1")
    index.add_fact(fact)
    index.delete_fact("f-ref")

    assert index.get_file_entry(fid) is None
    assert index.get_file_body(fid) is None
    assert index._blob_store().count() == 1        # noqa: SLF001  另一条（routing.toml）还在


def test_shared_blob_not_deleted_while_still_referenced(index):
    """内容寻址意味着多条条目可能共享同一个 blob —— 还有人引用就不能删。"""
    body = "# 同一份内容\n" + "line\n" * 40
    for path in ("a/x.py", "b/x.py"):
        entry = make_entry(path, body, user_id="u1", origin=FileOrigin.PRODUCED)
        entry.blob_ref = index._blob_store().put(body)  # noqa: SLF001
        entry.vector = _Embedder().embed([entry.embed_text()])[0]
        index.upsert_file_index(entry)
    assert index._blob_store().count() == 1        # noqa: SLF001

    index.delete_file_index(file_id_for("a/x.py"))
    assert index.get_file_body(file_id_for("b/x.py")) is not None   # 副本还在
    index.delete_file_index(file_id_for("b/x.py"))
    assert index._blob_store().count() == 0        # noqa: SLF001  最后一个引用没了才删


def test_delete_clears_hash_dedup_record(index):
    """删完要能重新索引——否则 hash 去重记录会让同内容永远进不来。"""
    turn = _turn(response=_PRODUCED_REPLY)
    index._index_produced_files_from_turn(turn, "u1")  # noqa: SLF001
    index.delete_file_index(file_id_for("config/routing.toml"))
    index.delete_file_index(file_id_for("packages/bladex_core/fusion.py"))
    assert index._index_produced_files_from_turn(turn, "u1") == 2  # noqa: SLF001


def test_delete_file_index_noop_on_readonly(tmp_path):
    w = MemoryIndex(tmp_path / "index", embedder=_Embedder())
    w.open()
    w._index_produced_files_from_turn(_turn(response=_PRODUCED_REPLY), "u1")  # noqa: SLF001
    w.close()

    r = MemoryIndex(tmp_path / "index", embedder=_Embedder(), read_only=True)
    r.open()
    assert r.delete_file_index(file_id_for("config/routing.toml")) is False


# ── blob 不进快照 ──────────────────────────────────────────────────────


def test_snapshot_carries_no_blobs(index, tmp_path):
    """拍板项：快照只带索引不带副本（换机后副本重新长出来）。

    快照是 JSONL 实体流、从不拷目录，所以这条天然成立——本测试钉住它，
    防止后来有人"顺手"把 blob 塞进去，让快照体积失控。
    """
    from bladex_proxy.snapshot import export_snapshot

    index._index_produced_files_from_turn(_turn(response=_PRODUCED_REPLY), "u1")  # noqa: SLF001
    out = tmp_path / "snap.jsonl"
    export_snapshot(index, None, ["MUST 用中文"], out)

    text = out.read_text(encoding="utf-8")
    assert "def rrf_fuse" not in text            # 产出正文不在快照里
    assert '"type": "blob"' not in text
