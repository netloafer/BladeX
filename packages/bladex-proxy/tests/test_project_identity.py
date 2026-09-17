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

    def test_dsh_runtime_context_workspace(self):
        """🔴 live wire 原文（2026-09-11 tap 录音，MQ-A67）。

        dsh 的载体既不是标签块也不是 `working directory` 行，而是运行时快照里
        的一句带**引号**的声明。三条既有 pattern 都抓 `\\S+`，照抄会在含空格的
        路径上截断 —— 这条取引号内整段。

        判据照抄 wire，不照抄我对 wire 的复述（同 MQ-L30 那条的教训）。
        """
        msgs = [{"role": "user", "content":
                 "Current runtime context. This snapshot supersedes earlier "
                 "runtime-context snapshots.\n\n"
                 "Current DSH file policy: workspace-write. Any available operation "
                 "enforced by the DSH file sandbox may modify files under the session "
                 'workspace: "/Users/you/dev/bladex-bench/w3-dsh".\n\n'
                 "Approval policy: ask."}]
        assert extract_cwd(msgs) == "/Users/you/dev/bladex-bench/w3-dsh"

    def test_dsh_workspace_path_with_spaces_is_not_truncated(self):
        """引号内整段 —— `\\S+` 会在第一个空格断掉，macOS 路径常含空格。"""
        msgs = [{"role": "user", "content":
                 'under the session workspace: "/Users/j/My Projects/BladeX".'}]
        assert extract_cwd(msgs) == "/Users/j/My Projects/BladeX"

    def test_pi_current_working_directory_at_tail_of_developer_message(self):
        """🔴 live wire 原文（2026-09-11 Hub `local/Pi/…`，Hub 全量重放定位）。

        Pi 每轮在 system/developer 消息的**最后一行**声明
        `Current working directory: …` —— 实测两种位置 @8,139 与 @15,397，
        **都在 `_SCAN_CHARS=8192` 之外**，也不在任何标签块里。
        ⇒ 靠 `system`/`developer` 全文扫覆盖（MQ-A25 当年评估过这一档，
        否掉的理由是"对 codex 没用"，不是不安全）。

        ⚠️ 当天曾为 Pi 加过 `<project_instructions path=…>` 专属 pattern，
        当天删除：那指向的是**指令文件**，与 cwd 是两回事，用它当 cwd
        是拿相邻信号冒充判据。
        """
        body = "z" * 8600
        tail = "\n</available_skills>\nCurrent working directory: /Users/you/Documents/Code/pi"
        assert extract_cwd([{"role": "developer", "content": body + tail}]) == \
            "/Users/you/Documents/Code/pi"

    def test_user_role_is_still_head_bounded(self):
        """🔴 全文扫**只给 system/developer**，user 角色仍受头部窗口约束。

        MQ-A25 的中毒源就在 user 侧（tool 结果转引我们自己的文档）。
        这条钉住边界，防止下次"顺手也给 user 放开"。
        """
        deep = "w" * 8600 + "\nCurrent working directory: /Users/j/deep"
        assert extract_cwd([{"role": "user", "content": deep}]) == ""

    def test_hermes_home_dir_carrier_is_caught_but_yields_global(self):
        """🔴 hermes 的 0% 是**正确读数**，不是缺陷（2026-09-11 全量复核）。

        载体 `Current working directory: /Users/you`（200/200 条标准 system
        prompt 都有）。这里其实是**两件事**：
          ① pattern **抓不到** —— 前缀词白名单原来只有 `primary`（claude-code 的写法），
             hermes 写的是 `Current`。**通路是坏的**（已放开前缀词）。
          ② 就算抓到，值是**纯家目录**，`resolve_project` 第三档正确判 Global。
        ⇒ 今天的 0% **结果对而通路坏**：hermes 哪天在某个仓库里跑才会暴露，且同样静默。

        ## 这条测试真正钉的是一次判断错误

        当天曾据**一条**样本里的 `Your working directory is /…/deepseek-harness`
        新加一条句中 pattern，推论覆盖 1,009 轮。全量复核：该串 4/3,102 = 0.13%，
        其中 2 轮还是"用 hermes 装 dsh"那次会话的散文。**n=1 推 n=1009。**
        pattern 已撤回。把结论钉成测试，免得下次看见 0% 又当缺陷去补 pattern。
        """
        msgs = [{"role": "system", "content":
                 "You are Hermes Agent, an intelligent AI assistant.\n\n"
                 "Host: macOS (26.6.2)\n"
                 "User home directory: /Users/you\n"
                 "Current working directory: /Users/you\n"}]
        assert extract_cwd(msgs) == "/Users/you", \
            "载体要抓得到（前缀词 `Current` 曾不在白名单里）"
        assert resolve_project("/Users/you").project_id == "", \
            "纯家目录 ⇒ Global，这是设计上正确的行为"

    def test_carrier_at_fourth_message_is_reachable(self):
        """🔴 MQ-A67 的**真正缺陷**：不是 pattern 缺，是窗口差一条。

        dsh 首轮 `roles=['system','user','user','user']`，载体在 `messages[3]`，
        而 `_SCAN_MESSAGES` 原为 3 ⇒ 循环在够到它之前就 break，
        `project_id` 恒空、全落 Global（两跑 159 轮无一例外）。

        ⚠️ 只加 pattern 不动窗口，这条会红 —— 两处都要改才算修好。
        """
        msgs = [
            {"role": "system", "content": "You are an AI agent powered by DeepSeek Harness."},
            {"role": "user", "content": "任务说明"},
            {"role": "user", "content": "补充约束"},
            {"role": "user", "content":
             'Current runtime context. ... under the session workspace: "/Users/j/dev/w"'},
        ]
        assert extract_cwd(msgs) == "/Users/j/dev/w"

    def test_scan_window_still_bounded(self):
        """窗口放宽到 4 是**有界的**：第 5 条仍然够不着。

        MQ-A25 已否掉"放大窗口"这条路（tool 结果转引我们自己的文档时三个串同现，
        放大 = 往中毒源上扫）。这条钉住边界，防止下次"再放一条就好了"。
        """
        msgs = [{"role": "user", "content": f"filler {i}"} for i in range(4)]
        msgs.append({"role": "user", "content": "<cwd>/Users/j/deep</cwd>"})
        assert extract_cwd(msgs) == ""

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

    def test_opencode_env_block_past_head_window(self):
        """🔴 A67 的 live 判据第一次没兑现，就是被这一格挡的（2026-09-11）。

        重启后 opencode 的 `agent_id` 认出来了，`project_resolved` 仍 `source=global`。
        定位：载体 `<env>\n  Working directory: …` 在 system **@8,696**，
        而 `_SCAN_CHARS=8192` —— **差 512 字节**。
        pattern 对、窗口条数对，**只是块名不认得**：`_ENV_BLOCK` 原来只有
        `<environment_context>` 一种写法，opencode 用的是 `<env>`。

        修法走 MQ-A25 既定的路（认确定性标签块、块内不限 offset），
        `_SCAN_CHARS` 一个字节不动 —— 放大字符窗口才是往中毒源上扫。
        """
        body = "x" * 8600
        env = ("\n<env>\n  Working directory: /Users/you/dev/BladeX\n"
               "  Workspace root folder: /Users/you\n"
               "  Is directory a git repo: yes\n</env>")
        assert extract_cwd([{"role": "system", "content": body + env}]) == \
            "/Users/you/dev/BladeX"

    def test_env_block_names_are_a_closed_set(self):
        """块名是**闭集**，不是"任何 <xxx> 都当环境块"。

        块内不限 offset 是一项特权 —— 给错对象就等于把整条消息全文扫了。
        闭集要从 agent 的 wire 原文读，不从"看起来像"推。
        """
        # 🔴 用 `user` 角色：system/developer 已全文扫，块名特权在那两个角色上
        #    看不出差别 —— 测"特权"必须挑一个**没有**全文扫的角色。
        body = "y" * 8600
        fake = "\n<context>\n  Working directory: /Users/j/nope\n</context>"
        assert extract_cwd([{"role": "user", "content": body + fake}]) == ""
        real = "\n<env>\n  Working directory: /Users/j/yes\n</env>"
        assert extract_cwd([{"role": "user", "content": body + real}]) == "/Users/j/yes"

    def test_env_block_beats_agents_md_head(self):
        """同一消息 @0 有 AGENTS.md 头、@11890 有标签块：取标签块（agent 声明的
        环境），不再靠指令头"撞上"。"""
        content = ("# AGENTS.md instructions for /Users/j/dev/repo\n"
                   + "z" * 11800 +
                   "\n<environment_context>\n<cwd>/Users/j/dev/repo/sub</cwd>\n"
                   "</environment_context>")
        assert extract_cwd([{"role": "user", "content": content}]) == \
            "/Users/j/dev/repo/sub"

    def test_tool_role_carrier_is_never_read(self):
        """零误伤：tool 结果里的标签块**永不采纳**（codex 那轮 msg[32]/[86]/[130]）。

        🔴 **2026-09-11 改写（MQ-A67）**：原判据是"它在前 3 条之外，本来就不进"——
        把零误伤**挂在了窗口大小上**。窗口为了够到 dsh 的 `messages[3]` 放宽到 4，
        这条立刻变红：中毒源就在第 4 条。**这正是 MQ-A25 警告过的形态，
        而它被一条测试当场抓住了。**

        修法不是缩回窗口，是把判据落到**真正的分界**上：tool 结果是**转引的内容**，
        不是 agent 对自己环境的声明，两者不该同权。⇒ `tool` 角色占窗口但不扫。
        这样零误伤与窗口大小**解耦**，下次再调窗口不会把它悄悄破坏。
        """
        poison = "<environment_context><cwd>/poison</cwd></environment_context>"
        # 第 4 条（原判据靠"够不着"）
        assert extract_cwd([{"role": "system", "content": "sys"},
                            {"role": "user", "content": "hi"},
                            {"role": "assistant", "content": "ok"},
                            {"role": "tool", "content": poison}]) == ""
        # 🔴 第 2 条 —— 窗口正中央，"够不着"这条理由完全不成立，
        #    只有"tool 不扫"能挡住它。
        assert extract_cwd([{"role": "system", "content": "sys"},
                            {"role": "tool", "content": poison},
                            {"role": "user", "content": "hi"}]) == ""

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


class TestCarrierPushedOutOfWindow:
    """MQ-A70：载体没消失，是被挤出了窗口。

    live 形态（CC x5-codex-flash 跑，2026-09-11）：压缩摘要作为一条新消息插在
    头部，把 `# Environment`（**system 角色**）从 `messages[2]` 推到 `[4]`，
    而窗口扫 [0..3] ⇒ 50 个主轮 `project_id` 全空。**50/50 轮载体都还在**，
    一轮都没丢 —— 所以修法是窗口策略，不是"会话粘性兜底"。

    🔴 **这不是 CC 特有形态。** 存量重放里 codex 同样有 10 轮被这条边界吃掉；
    codex 看起来免疫只是因为它每轮重发 `<environment_context>`，那是 agent 行为的
    巧合，不是结构上的豁免。任何往会话头部插消息的行为都会触发。
    """

    def _cc_after_compaction(self) -> list[dict]:
        """照抄 live 形态：system 基座 / 压缩摘要 / assistant / tool / system 环境块。"""
        return [
            {"role": "system", "content": "You are Claude Code" + "…" * 2000},
            {"role": "user", "content": "This session is being continued from a "
                                        "previous conversation that ran out of context." + "x" * 17000},
            {"role": "assistant", "content": "根因清楚了"},
            {"role": "tool", "content": "The file /Users/you/dev/other/x.py" + "y" * 29000},
            {"role": "system", "content": "# Environment\nYou have been invoked in the "
                                          "following environment: \n - Primary working directory: "
                                          "/Users/you/dev/bladex-bench/x5-codex-flash\n"},
        ]

    def test_system_carrier_found_beyond_the_message_window(self):
        """第 5 条（索引 4）的 system 载体必须抽到——窗口只有 4 条。"""
        assert extract_cwd(self._cc_after_compaction()) == \
            "/Users/you/dev/bladex-bench/x5-codex-flash"

    def test_carrier_found_however_deep_the_system_message_sits(self):
        """豁免不挂在"第 5 条"这个数字上：再插 40 条也必须抽到。

        钉死的是**依据挂在角色上**。若哪天有人把豁免改回某个固定条数，
        本条会红——那正是 A67→A70 两次踩的同一个坑（放大窗口治标一次）。
        """
        msgs = self._cc_after_compaction()
        filler = [{"role": "user", "content": f"turn {i}"} for i in range(40)]
        assert extract_cwd(msgs[:4] + filler + msgs[4:]) == \
            "/Users/you/dev/bladex-bench/x5-codex-flash"

    def test_user_and_tool_still_bounded_by_the_window(self):
        """🔴 豁免只给 system/developer。user/tool 侧的窗口是**中毒隔离**，不得一起放开。

        MQ-A25 的中毒源就是 tool 结果转引我们自己的文档（`<environment_context>`
        与 `Working directory:` 原样带回）。这里把一条"转引"放在窗口之外，
        它必须仍然抽不到 —— 否则放开窗口就等于往中毒源上扫。
        """
        msgs = [{"role": "user", "content": f"turn {i}"} for i in range(6)]
        msgs.append({"role": "user", "content":
                     "文档里写着 - Primary working directory: /Users/you/dev/quoted"})
        assert extract_cwd(msgs) == ""

    def test_developer_role_gets_the_same_exemption(self):
        """Pi 的载体在 developer 角色、消息最后一行（MQ-A67）——同样不该受条数约束。"""
        msgs = [{"role": "user", "content": f"turn {i}"} for i in range(8)]
        msgs.append({"role": "developer", "content":
                     "x" * 15000 + "\nCurrent working directory: /Users/you/dev/BladeX"})
        assert extract_cwd(msgs) == "/Users/you/dev/BladeX"

    def test_first_carrier_wins_when_several_system_messages_carry_one(self):
        """多条 system 都带载体时取最靠前的一条——CC 的基座 system 在环境块之前。"""
        msgs = [
            {"role": "system", "content": "<cwd>/Users/you/dev/first</cwd>"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": " - Primary working directory: /Users/you/dev/second"},
        ]
        assert extract_cwd(msgs) == "/Users/you/dev/first"
