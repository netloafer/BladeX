"""项目识别第一档（2026-08-29，ADR-0032 §4 键分层）。

判据取 wire 原文（live 实证 2026-08-29）：codex `<cwd>` 标签、
claude-code `Working directory:` 行；软链收敛 = "路径不做项目身份"的最硬实证
（同一仓库两路径 720/560 分裂）。
"""

from __future__ import annotations

import os

from bladex_proxy.project_identity import (
    ProjectIdentity,
    extract_cwd,
    resolve_from_request,
    resolve_project,
    _cache,
)


class TestExtractCwd:
    def test_codex_env_context_tag(self):
        msgs = [{"role": "user", "content":
                 "<environment_context>\n  <cwd>/Users/j/dev/BladeX</cwd>\n"
                 "  <shell>zsh</shell>\n</environment_context>"}]
        assert extract_cwd(msgs) == "/Users/j/dev/BladeX"

    def test_claude_code_working_directory_line(self):
        msgs = [{"role": "user", "content":
                 "<env>\nWorking directory: /Users/j/proj\nPlatform: darwin\n</env>"}]
        assert extract_cwd(msgs) == "/Users/j/proj"

    def test_claude_code_primary_working_directory_bullet(self):
        """🔴 live wire 原文（2026-08-29 迁移批复核，turn
        `local/claude-code/cbea1506-…/1788000276218-0` 的 messages[0] @5035）。

        立卡时记的形态是 `Working directory:`，实际是**小写 w + `Primary ` 前缀
        + 行首 `- ` 项目符号**，于是 claude-code 的 project_id 恒空、全部落
        Global，惰性作用域迁移（MQ-L30）第一道门就返回。这条串照抄 wire，
        不照抄我对 wire 的复述——判据必须能被重新复算。
        """
        msgs = [{"role": "system", "content":
                 "You are Claude.\n\n"
                 " - Primary working directory: /Users/alice/dev/BladeX\n"
                 " - Is a git repository: true\n"
                 " - Platform: darwin\n"}]
        assert extract_cwd(msgs) == "/Users/alice/dev/BladeX"

    def test_working_directory_is_case_insensitive_at_line_start_only(self):
        """放宽的是**写法**，不是位置：行首（可带项目符号）才算。

        散文中段的 `… the Working directory: x` 不再命中——原 pattern 无 `^` 锚，
        收紧它是顺带减假阳性（codex 那轮的 tool 结果里，我们自己的文档正文就
        出现过三次 `Working directory:`）。
        """
        assert extract_cwd([{"role": "user", "content":
                             "WORKING DIRECTORY: /Users/j/p"}]) == "/Users/j/p"
        assert extract_cwd([{"role": "user", "content":
                             "see the Working directory: /Users/j/p"}]) == ""

    def test_codex_desktop_agents_md_header(self):
        """desktop 形态无 env_ctx（live fp:6103d775f46f 实证）——路径在指令头。"""
        msgs = [{"role": "user", "content":
                 "# AGENTS.md instructions for /Users/j/dev/BladeX\n\n"
                 "<INSTRUCTIONS>…</INSTRUCTIONS>"}]
        assert extract_cwd(msgs) == "/Users/j/dev/BladeX"

    def test_block_list_content_form(self):
        """Responses/Anthropic 分块形态（live 实证：codex desktop / Pi 的
        content 是 [{'type':'text','text':…}]）——只认 str 会整条跳过。"""
        msgs = [{"role": "user", "content": [
            {"type": "input_text",
             "text": "# AGENTS.md instructions for /Users/j/dev/BladeX\n…"}]}]
        assert extract_cwd(msgs) == "/Users/j/dev/BladeX"

    def test_no_signal_returns_empty(self):
        assert extract_cwd([{"role": "user", "content": "帮我看个报错"}]) == ""
        assert extract_cwd([]) == ""

    def test_only_scans_message_heads(self):
        """头部窗口外的**裸** `<cwd>` 不算（热路径纪律）——MQ-A25 标签定位只放行
        `<environment_context>` 块内的命中，裸标记在窗口外仍是越界。"""
        deep = "x" * 9000 + "<cwd>/late/marker</cwd>"
        assert extract_cwd([{"role": "user", "content": deep}]) == ""

    # ---- MQ-A25 标签定位（2026-09-02 Jason 拍板）----

    def test_codex_env_block_past_head_window(self):
        """🔴 live 形态（12 天 3754 轮，codex user 727/727 `<cwd>` @11890 > 8192，
        `docs/benchmarks/project-scan-window-20260902.txt`）：AGENTS.md 头不在场时
        此前必掉 Global。标签块整段扫，不限 offset。"""
        body = "<INSTRUCTIONS>" + "y" * 11800 + "</INSTRUCTIONS>\n"
        env = ("<environment_context>\n  <cwd>/Users/j/dev/other</cwd>\n"
               "  <shell>zsh</shell>\n</environment_context>")
        assert extract_cwd([{"role": "user", "content": body + env}]) == \
            "/Users/j/dev/other"

    def test_env_block_beats_agents_md_head(self):
        """同一消息 @0 有 AGENTS.md 头、@11890 有标签块：取标签块（agent 声明的
        环境），不再靠指令头"撞上"。"""
        content = ("# AGENTS.md instructions for /Users/j/dev/repo\n"
                   + "z" * 11800 +
                   "\n<environment_context>\n<cwd>/Users/j/dev/repo/sub</cwd>\n"
                   "</environment_context>")
        assert extract_cwd([{"role": "user", "content": content}]) == \
            "/Users/j/dev/repo/sub"

    def test_env_block_outside_scan_messages_is_ignored(self):
        """零误伤：`_SCAN_MESSAGES` 不动——tool 结果转引文档里的标签块
        （codex 那轮 msg[32]/[86]/[130]）在前 3 条之外，本来就不进。"""
        msgs = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {"role": "tool", "content":
                 "<environment_context><cwd>/poison</cwd></environment_context>"}]
        assert extract_cwd(msgs) == ""

    def test_env_block_without_cwd_falls_back_to_head(self):
        """标签块内无 cwd 标记 ⇒ 仍按头部窗口找（不因块存在而短路）。"""
        content = ("# AGENTS.md instructions for /Users/j/dev/repo\n"
                   "<environment_context><shell>zsh</shell></environment_context>")
        assert extract_cwd([{"role": "user", "content": content}]) == \
            "/Users/j/dev/repo"

    def test_env_block_in_block_list_content(self):
        """Responses 分块形态 + 窗口外标签块（codex 走 /v1/responses）。"""
        msgs = [{"role": "user", "content": [
            {"type": "input_text", "text": "q" * 9000},
            {"type": "input_text", "text":
             "<environment_context><cwd>/Users/j/p</cwd></environment_context>"}]}]
        assert extract_cwd(msgs) == "/Users/j/p"


class TestResolveProject:
    def _repo(self, tmp_path, name, remote=""):
        root = tmp_path / name
        (root / ".git").mkdir(parents=True)
        if remote:
            (root / ".git" / "config").write_text(
                f'[core]\n\trepositoryformatversion = 0\n'
                f'[remote "origin"]\n\turl = {remote}\n\tfetch = +refs/*\n',
                encoding="utf-8")
        return root

    def setup_method(self):
        _cache.clear()

    def test_git_remote_is_the_key(self, tmp_path):
        root = self._repo(tmp_path, "proj", remote="git@github.com:j/proj.git")
        sub = root / "src" / "deep"
        sub.mkdir(parents=True)
        a = resolve_project(str(sub))          # 子目录向上找到仓库根
        assert a.source == "git-remote" and a.name == "proj"
        assert a.project_id.startswith("p-") and len(a.project_id) == 14

    def test_symlink_paths_converge_to_one_project(self, tmp_path):
        """🔴 本卡立卡实证：~/dev/BladeX 与旧软链两个 cwd 必须同一 project_id。"""
        root = self._repo(tmp_path, "real")
        link = tmp_path / "linked"
        os.symlink(root, link)
        assert resolve_project(str(root)).project_id == \
            resolve_project(str(link)).project_id != ""

    def test_path_tier_with_project_marker(self, tmp_path):
        """键分层第三档（live 实证：~/dev/BladeX 无 .git 但带 AGENTS.md）：
        无仓库 + 项目声明标志 ⇒ path 档；软链仍收敛（realpath 进键）。"""
        root = tmp_path / "BladeX"
        root.mkdir()
        (root / "AGENTS.md").write_text("# proj", encoding="utf-8")
        r = resolve_project(str(root))
        assert r.source == "path" and r.name == "BladeX" and r.project_id
        link = tmp_path / "linked-bladex"
        os.symlink(root, link)
        assert resolve_project(str(link)).project_id == r.project_id

    def test_git_beats_path_tier(self, tmp_path):
        """有仓库时 git 档优先——路径档只是无 git 的落点。"""
        root = self._repo(tmp_path, "withgit")
        (root / "AGENTS.md").write_text("# proj", encoding="utf-8")
        assert resolve_project(str(root)).source == "git-root"

    def test_no_repo_is_global(self, tmp_path):
        """家目录起的会话（live CC 样本 cwd=/Users/alice）落 Global——
        路径不做项目身份（ADR-0032 拍板）。"""
        plain = tmp_path / "home"
        plain.mkdir()
        assert resolve_project(str(plain)) == ProjectIdentity()
        assert resolve_project("") == ProjectIdentity()

    def test_repo_without_remote_keys_on_root(self, tmp_path):
        root = self._repo(tmp_path, "local-only")
        r = resolve_project(str(root))
        assert r.source == "git-root" and r.project_id

    def test_cache_avoids_repeat_fs_walks(self, tmp_path):
        root = self._repo(tmp_path, "cached")
        first = resolve_project(str(root))
        assert resolve_project(str(root)) is first, "同 cwd 第二次必须走缓存"

    # ── 护栏 C（2026-08-29）：垃圾匹配不许被"识别成 proxy 自己那个项目" ──

    def test_relative_junk_never_resolves_against_proxy_cwd(self, tmp_path,
                                                            monkeypatch):
        """🔴 本护栏存在的唯一理由，用**会失败的形态**钉住。

        没有这道门时：pattern 抓到的相对垃圾串经 `os.path.realpath` 会接到
        **proxy 自己的 cwd** 上，`_git_repo_root` 再向上走 12 层，正好走回
        proxy 所在仓库的 `.git` ⇒ 任何垃圾都变成"proxy 自己那个项目"。
        这里把进程 cwd 挪进一个真仓库，再喂垃圾串——必须仍是 Global。
        """
        repo = self._repo(tmp_path, "proxy-own-repo")
        monkeypatch.chdir(repo)
        assert resolve_project(str(repo)).project_id, "前置：这个 cwd 确是项目"
        for junk in ("` / codex desktop `# AGENTS.md instructions for` 头。",
                     "src/deep", "…", "."):
            assert resolve_project(junk) == ProjectIdentity(), \
                f"相对串 {junk!r} 不得被采纳"

    def test_relative_junk_is_not_cached(self, tmp_path, monkeypatch):
        """垃圾串取值空间无界，而 `_cache` 无逐出策略——让它进缓存就是慢性泄漏。"""
        monkeypatch.chdir(tmp_path)
        before = len(_cache)
        resolve_project("not/an/absolute/path")
        assert len(_cache) == before

    def test_absolute_but_missing_dir_is_global(self, tmp_path):
        """绝对但不存在 ⇒ Global（这一档进缓存：绝对路径取值空间是有界的）。"""
        missing = str(tmp_path / "gone")
        assert resolve_project(missing) == ProjectIdentity()
        assert missing in _cache

    def test_file_path_is_not_a_project(self, tmp_path):
        """指向文件而非目录也不算——`isdir` 不是 `exists`。"""
        f = tmp_path / "AGENTS.md"
        f.write_text("# x", encoding="utf-8")
        assert resolve_project(str(f)) == ProjectIdentity()


class TestKeyLayering:
    def test_explicit_layer_is_placeholder_until_full_sample(self):
        """显式层占位（MQ-A22 修复后重启才有全样本）：占位不猜测，恒空。"""
        ident = resolve_from_request(
            {"x-codex-turn-metadata": '{"session_id": "x"}'},
            [{"role": "user", "content": "hi"}])
        assert ident.source != "explicit"
