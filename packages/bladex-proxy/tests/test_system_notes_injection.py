"""V-P6：BladeX 自我介绍（稳定层）+ 注入标记 ⇒ 剥离落点 对账。

## 两件事

**① 自我介绍**（ADR-0032 §3.1 稳定层）：`config/system/{AGENT,TOOLS,SKILLS}.md`
按部署根加载、启动预加载一次、注入**前缀**、门控与工具面同步。

**② 注入标记对账**：全仓库四个 `<bladex-*>` 标记，每个都必须在
`envelope._ENVELOPE_PATTERNS` 里有剥离落点。2026-08-28 普查发现只有一个有——
`<bladex-ledger>`（每轮注、体量最大）和 `<bladex-context-summary>` 全裸，
我们注入的内容会被 agent 回传、再被自己蒸成 fact。

②与 MQ-S46 是**同一个交接缺陷的第三、四例**：生产方与处置方之间没有对账。
`test_distill_only_envelope_contract.py` 守的是「aux 判定 ⇒ envelope 剥离」，
本文件守的是「注入 ⇒ envelope 剥离」。两条交接，同一种烂法。
"""
from __future__ import annotations

import pytest
from bladex_core.envelope import strip_envelopes
from bladex_proxy.agency import (
    ABOUT_BLOCK_CLOSE,
    ABOUT_BLOCK_OPEN,
    INJECTION_MARKERS,
    load_system_notes,
)

# ── ① 注入标记 ⇒ 剥离落点（机制化对账）──────────────────────────────────────

@pytest.mark.parametrize("marker", INJECTION_MARKERS)
def test_injection_markers_have_strip_landings(marker: str) -> None:
    """🔴 每个注入标记都必须被 envelope 剥掉，否则形成自反馈闭环。

    自反馈不是理论风险：ADR-0025 §6 担心过，而真实数据里**已经发生**——
    24 条 user 消息回带了 `<bladex-memory>`（CC 把含注入的历史当 user 回传）。
    `<bladex-ledger>` 每轮都注、体量比 memory 大，却一直没有落点。

    加新标记时本条自动覆盖（parametrize 读 `INJECTION_MARKERS`），
    不给"以后再补 envelope"留缝。
    """
    close = marker.replace("<", "</", 1)
    text = f"用户真正的问题在这里。\n{marker}\n注入的内容\n{close}\n后半句也是用户的话。"
    clean, kinds = strip_envelopes(text)
    assert kinds, f"{marker} 没有任何剥离落点 —— 注入内容会被自己蒸成 fact"
    assert "注入的内容" not in clean, f"{marker} 的内容没被剥掉：{clean[:120]!r}"
    # 成对剥离的意义：标记外的用户原话必须留下
    assert "用户真正的问题在这里" in clean
    assert "后半句也是用户的话" in clean


def test_injection_markers_list_is_complete() -> None:
    """`INJECTION_MARKERS` 必须覆盖仓库里真实存在的 `<bladex-*>` 开标记。

    清单漏一个 ⇒ 上面那条 parametrize 就测不到它，对账变成空跑
    （尺子和被测系统各算各的，MQ-S4 的形态）。故从源码扫。
    """
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[3]
    found: set[str] = set()
    for pkg in ("packages/bladex-core", "packages/bladex-proxy"):
        for py in (root / pkg).rglob("*.py"):
            if "tests" in py.parts:
                continue
            for m in re.findall(r'"(<bladex-[a-z-]+>)"', py.read_text(encoding="utf-8")):
                found.add(m)
    # 🔴 前置断言：路径算错 ⇒ found 为空 ⇒ missing 也为空 ⇒ **假绿**。
    # 分母必须先证明存在，这是今天反复栽的那类（"没落地的注入什么都不证明"）。
    assert len(found) >= len(INJECTION_MARKERS), (
        f"只扫到 {sorted(found)} —— 扫描根 {root} 不对？本条会假绿"
    )
    missing = found - set(INJECTION_MARKERS)
    assert not missing, (
        f"这些注入标记不在 INJECTION_MARKERS 里：{sorted(missing)}。\n"
        "加进清单并给 envelope 补剥离落点——漏登记 = 对账空跑。"
    )


# ── ② 加载：部署根 / env 覆盖 / 缺失处置 ─────────────────────────────────────

def _make_notes(tmp_path, **files):
    d = tmp_path / "config" / "system"
    d.mkdir(parents=True)
    (tmp_path / "config" / ".env").write_text("", encoding="utf-8")  # 部署根判据
    for name, body in files.items():
        (d / f"{name}.md").write_text(body, encoding="utf-8")
    return tmp_path


def test_loads_from_deployment_root_regardless_of_cwd(tmp_path, monkeypatch) -> None:
    """🔴 按**部署根**解析，不用相对 cwd 的路径。

    `load_ledger_template` 的 docstring 记着这个教训：首版写相对路径，
    proxy 的工作目录不一定是仓库根 ⇒ 那份精心写的模板**从未被注入过**，
    只有一条 warning。同型问题不该犯第二次。
    """
    root = _make_notes(tmp_path, AGENT="# A\nagent body",
                       TOOLS="# T\ntools body", SKILLS="# S\nskills body")
    monkeypatch.setenv("BLADEX_HOME", str(root))
    monkeypatch.delenv("BLADEX_SYSTEM_NOTES_DIR", raising=False)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)          # cwd ≠ 部署根，正是 live 的形态
    text = load_system_notes()
    for body in ("agent body", "tools body", "skills body"):
        assert body in text


def test_explicit_env_dir_wins(tmp_path, monkeypatch) -> None:
    root = _make_notes(tmp_path, AGENT="# from-root")
    other = tmp_path / "custom"
    other.mkdir()
    (other / "AGENT.md").write_text("# from-env", encoding="utf-8")
    monkeypatch.setenv("BLADEX_HOME", str(root))
    monkeypatch.setenv("BLADEX_SYSTEM_NOTES_DIR", str(other))
    assert "from-env" in load_system_notes()


def test_concatenation_order_is_agent_tools_skills(tmp_path, monkeypatch) -> None:
    """顺序有意义：先说 BladeX 是什么，再说工具怎么用，最后 skills。"""
    root = _make_notes(tmp_path, AGENT="AAA", TOOLS="TTT", SKILLS="SSS")
    monkeypatch.setenv("BLADEX_HOME", str(root))
    monkeypatch.delenv("BLADEX_SYSTEM_NOTES_DIR", raising=False)
    text = load_system_notes()
    assert text.index("AAA") < text.index("TTT") < text.index("SSS")


def test_all_missing_yields_empty_not_a_shell(tmp_path, monkeypatch) -> None:
    """🔴 三份全缺 ⇒ 返回空串（调用方据此整块不注），**不设结构性兜底**。

    与账本模板刻意不同：模板缺失回落 `FALLBACK_TEMPLATE_MD` 还能保住五段结构，
    而一段**错的**自我介绍会让模型对 BladeX 的能力边界产生错误预期——
    代价方向相反，所以宁可不注。
    """
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("BLADEX_HOME", str(tmp_path))
    monkeypatch.delenv("BLADEX_SYSTEM_NOTES_DIR", raising=False)
    assert load_system_notes() == ""


def test_partial_missing_still_loads_the_rest(tmp_path, monkeypatch) -> None:
    """缺一份不该让整块消失——降级而非全无。"""
    root = _make_notes(tmp_path, AGENT="AAA", TOOLS="TTT")   # 没有 SKILLS
    monkeypatch.setenv("BLADEX_HOME", str(root))
    monkeypatch.delenv("BLADEX_SYSTEM_NOTES_DIR", raising=False)
    text = load_system_notes()
    assert "AAA" in text and "TTT" in text


# ── ③ 注入：位置 / 门控 / 去重 ───────────────────────────────────────────────

class _Runtime:
    """只带 `insert_system_notes` 需要的两个属性，避开真实 AgencyRuntime 的依赖。"""

    def __init__(self, about: str) -> None:
        self.about = about
        self.agents_roster = ""   # V-F3：生产 __init__ 恒有此属性，桩跟上形态
        self.ledger_on = True     # MQ-L77：`ledger_face=None` 时回落到模块门

    insert_system_notes = None  # 由下面 __init_subclass__ 之外的赋值补上


def _rt(about: str = "ABOUT-BODY"):
    from bladex_proxy.agency import AgencyRuntime
    obj = _Runtime(about)
    obj.insert_system_notes = AgencyRuntime.insert_system_notes.__get__(obj)
    return obj


def test_inserted_after_last_system_before_history() -> None:
    """🔴 位置 = 前缀（最后一条 system 之后），与账本块的末尾**刻意相反**。

    稳定层逐字恒定 ⇒ 放前缀，它之后的历史全落在同一个 cache 前缀里。
    动态层每轮重算 ⇒ 追加末尾，对 cache 无损且注意力位置最好（MQ-L10）。
    两层最优位置不同，这不是不一致。

    另：agent 自己的 system prompt 仍在第一位——我们是接入方，不抢开场
    （刚性原则 10）。
    """
    msgs = [{"role": "system", "content": "agent own prompt"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"}]
    out = _rt().insert_system_notes(msgs, toolface_injected=True)
    assert len(out) == 4
    assert out[0]["content"] == "agent own prompt", "抢了 agent 的开场"
    assert ABOUT_BLOCK_OPEN in out[1]["content"]
    assert out[1]["role"] == "system"
    assert [m["content"] for m in out[2:]] == ["q1", "a1"]


def test_does_not_displace_the_ledger_block_from_the_very_end() -> None:
    """🔴 生产形态：BladeX 自己的注入块**也是 `role=system`**。

    ## 这条是 gate 当场红出来的（2026-08-28）

    首版按"最后一条 system 之后"找插入点，而 `ledger_injection_message` 返回的
    role 也是 system、且按 MQ-L10 追加在数组**真正的末尾** ⇒ 自我介绍插到了
    账本块**后面**（live 日志 `agency_about_injected pos=16 total=17`），
    把账本块挤离末尾。那正是 MQ-L10 修掉的病：账本块携带
    "FIRST STEP, every turn" 指令，被埋在后面等于没有。

    ## 为什么单测没抓到

    我的 fixture 是 `[system, user, assistant]` —— **不是生产形态**。
    生产形态长这样：`agent system` + `<bladex-memory> system` + 工具循环 +
    末尾 `<bladex-ledger> system`。用假设的形态造 fixture，测的是我的假设。
    今天第二次栽在这上面（前一次是行为断言用 live 样本而非判据原文）。
    """
    from bladex_proxy.agency import LEDGER_BLOCK_OPEN

    msgs = [
        {"role": "system", "content": "agent own prompt"},
        {"role": "system", "content": "<bladex-memory>\n- fact\n</bladex-memory>"},
        {"role": "user", "content": "深度分析这个网站"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t0"}]},
        {"role": "tool", "tool_call_id": "t0", "content": "img"},
        {"role": "system", "content": f"{LEDGER_BLOCK_OPEN}\nGoal…\n</bladex-ledger>"},
    ]
    out = _rt().insert_system_notes(msgs, toolface_injected=True)

    # 账本块仍在真正的末尾
    assert LEDGER_BLOCK_OPEN in out[-1]["content"], (
        f"账本块被挤离末尾 —— MQ-L10 复发。末条是 {out[-1]['content'][:60]!r}")
    # 自我介绍在 agent 自己的 system 之后、BladeX 任何注入之前
    assert ABOUT_BLOCK_OPEN in out[1]["content"], (
        f"位置错了：{[str(m.get('content'))[:28] for m in out[:3]]}")
    assert out[0]["content"] == "agent own prompt"
    # 🔴 在 `<bladex-memory>` **之前**：它每轮内容都变，插在它之后
    # 稳定层就落进变动区，prompt cache 收益归零——而那是选前缀的全部理由。
    assert "<bladex-memory>" in out[2]["content"]


def test_inserted_at_head_when_no_system_message() -> None:
    msgs = [{"role": "user", "content": "q1"}]
    out = _rt().insert_system_notes(msgs, toolface_injected=True)
    assert ABOUT_BLOCK_OPEN in out[0]["content"]


def test_multiple_system_messages_insert_after_the_last() -> None:
    msgs = [{"role": "system", "content": "s1"},
            {"role": "system", "content": "s2"},
            {"role": "user", "content": "q"}]
    out = _rt().insert_system_notes(msgs, toolface_injected=True)
    assert [m["content"] for m in out[:2]] == ["s1", "s2"]
    assert ABOUT_BLOCK_OPEN in out[2]["content"]


def test_not_injected_without_toolface() -> None:
    """🔴 门控与工具面同步：没工具面就是**有令无器**，纯注意力税。

    AGENT.md/TOOLS.md 通篇在讲 "call bladex_ledger_switch / update"。
    与 MQ-L7 拆开账本正文与首步指令是同一条理由，也是三红线之一
    「模型不调用时零行为差异」的直接推论。
    """
    msgs = [{"role": "user", "content": "q"}]
    assert _rt().insert_system_notes(msgs, toolface_injected=False) is msgs


def test_not_injected_when_notes_empty() -> None:
    msgs = [{"role": "user", "content": "q"}]
    assert _rt("").insert_system_notes(msgs, toolface_injected=True) is msgs


def test_not_duplicated_when_agent_replays_it() -> None:
    """agent 把上一轮注入当历史回传（CC/Codex 实测会）⇒ 不叠第二份。"""
    msgs = [{"role": "system", "content": "s"},
            {"role": "user",
             "content": f"{ABOUT_BLOCK_OPEN}\nstale\n{ABOUT_BLOCK_CLOSE}"}]
    assert _rt().insert_system_notes(msgs, toolface_injected=True) is msgs


def test_does_not_mutate_caller_list() -> None:
    """返回新数组——就地改会让调用方的 `original_messages` 被污染。"""
    msgs = [{"role": "user", "content": "q"}]
    out = _rt().insert_system_notes(msgs, toolface_injected=True)
    assert len(msgs) == 1 and out is not msgs


def test_injected_block_is_itself_strippable() -> None:
    """自我介绍块被回传时能被剥掉——闭环自检，不靠 ① 的构造样本。"""
    msgs = [{"role": "user", "content": "q"}]
    out = _rt("REAL NOTES BODY").insert_system_notes(msgs, toolface_injected=True)
    clean, kinds = strip_envelopes(out[0]["content"])
    assert "bladex_about_echo" in kinds
    assert "REAL NOTES BODY" not in clean


# ── ④ 定稿内容的实体检查（改文件会红，提醒同步 ADR/文档）────────────────────

def test_shipped_notes_state_the_three_red_lines() -> None:
    """仓库里那三份文件必须说清三红线——它们是**对模型的承诺**。

    不检措辞（那会让每次润色都红），只检三条承诺各自的关键实体都在：
    工具只读写记忆/账本 · Goal 仅用户可改 · 不调用则零行为差异。
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[3]
    agent = (root / "config" / "system" / "AGENT.md").read_text(encoding="utf-8")
    low = agent.lower()
    assert "goal" in low and "user" in low, "没说清 Goal 属于用户"
    assert "cannot run code" in low or "only touch memory" in low, "没说清工具边界"
    assert "changes nothing" in low or "stays out of your way" in low, "没说清零差异"


# ── ⑤ MQ-L77：账本段随账本面装卸（2026-09-18）─────────────────────────────────

def _shipped(name: str) -> str:
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    return (root / "config" / "system" / name).read_text(encoding="utf-8")


def test_ledger_face_off_strips_every_ledger_mention_from_shipped_notes() -> None:
    """🔴 核心守卫：账本面不注 ⇒ 注入正文里**一个** `bladex_ledger_` 都不许有。

    阴性对照：去掉 `insert_system_notes` 里的 `render_system_notes(..., ledger=_ledger)`
    （直接用 `self.about`）⇒ 本用例红。
    """
    from bladex_proxy.agency.notes import LEDGER_SECTION_OPEN
    about = "\n\n---\n\n".join(_shipped(n) for n in ("AGENT.md", "TOOLS.md", "SKILLS.md"))
    assert LEDGER_SECTION_OPEN in about, "仓库文件必须用标记包住账本段，否则本守卫测的是空气"
    out = _rt(about).insert_system_notes(
        [{"role": "user", "content": "hi"}], toolface_injected=True, ledger_face=False)
    body = out[0]["content"]
    assert "bladex_ledger_" not in body, "模块/档位关了仍在教账本工具（MQ-L77）"
    assert "<!-- bladex:ledger" not in body and "<!-- /bladex:ledger" not in body
    assert "bladex_memory_search" in body, "记忆族说明不许被顺手剥掉"


def test_ledger_face_on_keeps_ledger_sections_but_never_the_markers() -> None:
    about = "\n\n---\n\n".join(_shipped(n) for n in ("AGENT.md", "TOOLS.md"))
    out = _rt(about).insert_system_notes(
        [{"role": "user", "content": "hi"}], toolface_injected=True, ledger_face=True)
    body = out[0]["content"]
    assert "bladex_ledger_switch" in body and "bladex_ledger_update" in body
    assert "<!-- bladex:ledger" not in body and "<!-- /bladex:ledger" not in body


def test_ledger_face_defaults_to_module_gate() -> None:
    from bladex_proxy.agency.notes import LEDGER_SECTION_CLOSE, LEDGER_SECTION_OPEN
    about = f"keep\n{LEDGER_SECTION_OPEN}\ncall bladex_ledger_switch\n{LEDGER_SECTION_CLOSE}\ntail"
    rt = _rt(about)
    rt.ledger_on = False
    body = rt.insert_system_notes([{"role": "user", "content": "x"}], toolface_injected=True)[0]["content"]
    assert "bladex_ledger_switch" not in body and "keep" in body and "tail" in body


def test_render_leaves_unclosed_marker_visible() -> None:
    """写错标记宁可可见，不静默吞半篇。"""
    from bladex_proxy.agency.notes import LEDGER_SECTION_OPEN, render_system_notes
    text = f"a\n{LEDGER_SECTION_OPEN}\nb"
    assert "b" in render_system_notes(text, ledger=False)


def test_orchestration_passes_ledger_face_to_notes() -> None:
    """接线守卫：server 侧必须把账本面传给自我介绍，否则默认回落模块门就把
    「档位不在集合」这一档漏了（那正是 live n4 的形态：模块开、档位关）。"""
    from _source_probe import package_source
    src = package_source("server")
    i = src.index("agency.insert_system_notes(")
    assert "ledger_face=" in src[i:i + 300]
