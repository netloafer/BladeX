"""V-L6f 回归：账本模板必须按**部署根**解析，不能用相对 cwd 的路径。

2026-08-25 live 回归：首版写 `config/ledger-template.md`（相对 cwd），proxy 的
工作目录不一定是仓库根 ⇒ 每次启动静默回落到结构性兜底，仓库里那份模板从未
被注入过，且只有一条 `ledger_template_unreadable` warning（看日志才发现）。
"""

from __future__ import annotations

from bladex_proxy.agency import load_ledger_template


def _make_root(tmp_path, body: str):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env").write_text("", encoding="utf-8")   # 部署根判据
    (tmp_path / "config" / "ledger-template.md").write_text(body, encoding="utf-8")
    return tmp_path


def test_reads_from_deployment_root_regardless_of_cwd(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "# T\n\n## Goal\n\n## Core\n\n## MyCustom\n")
    monkeypatch.setenv("BLADEX_HOME", str(root))
    monkeypatch.delenv("BLADEX_LEDGER_TEMPLATE", raising=False)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)          # cwd 与部署根不同 —— 正是 live 的形态
    text = load_ledger_template()
    assert "MyCustom" in text
    assert "fallback" not in text.lower()


def test_explicit_env_path_wins(tmp_path, monkeypatch):
    root = _make_root(tmp_path, "# from-root\n\n## Goal\n")
    explicit = tmp_path / "my.md"
    explicit.write_text("# from-env\n\n## Goal\n", encoding="utf-8")
    monkeypatch.setenv("BLADEX_HOME", str(root))
    monkeypatch.setenv("BLADEX_LEDGER_TEMPLATE", str(explicit))
    assert "from-env" in load_ledger_template()


def test_missing_file_falls_back_visibly(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.delenv("BLADEX_LEDGER_TEMPLATE", raising=False)
    text = load_ledger_template()
    # 兜底必须**看得出是兜底**（刻意难看，防"漂亮副本悄悄掩盖文件丢失"）
    assert "fallback" in text.lower()
    assert "## Goal" in text        # 但结构仍可用，首步指令不会消失
    # 兜底里指出该恢复哪个文件（这是有用的指路，不是路径泄漏）
    assert "config/ledger-template.md" in text
