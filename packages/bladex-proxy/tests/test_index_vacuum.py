"""`scripts/index_vacuum.py`（MQ-I15 LanceDB 死版本清理）的读数函数守卫：数版本/文件/索引目录只看目录结构，不 import lancedb。"""

from __future__ import annotations

import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_spec = importlib.util.spec_from_file_location("index_vacuum", os.path.join(_ROOT, "scripts", "index_vacuum.py"))
mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["index_vacuum"] = mod
_spec.loader.exec_module(mod)


def test_table_stats_counts_versions_files_indices(tmp_path):
    t = tmp_path / "files.lance"
    for d, n in (("_versions", 7), ("data", 4), ("_indices", 2)):
        (t / d).mkdir(parents=True)
        for i in range(n):
            (t / d / f"{i}.x").write_bytes(b"0" * 10)
    s = mod.table_stats(t)
    assert (s["versions"], s["data_files"], s["index_dirs"], s["bytes"]) == (7, 4, 2, 130)
    assert mod.table_stats(tmp_path / "nothing.lance") == {"versions": 0, "data_files": 0, "index_dirs": 0, "bytes": 0}


def test_dry_run_never_touches_lancedb(tmp_path, capsys):
    (tmp_path / "lancedb" / "facts.lance" / "_versions").mkdir(parents=True)
    (tmp_path / "lancedb" / "facts.lance" / "_versions" / "1.manifest").write_bytes(b"x")
    assert mod.main(["--index", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "facts.lance" in out and "干跑结束" in out
    assert (tmp_path / "lancedb" / "facts.lance" / "_versions" / "1.manifest").exists(), "干跑不删任何东西"
