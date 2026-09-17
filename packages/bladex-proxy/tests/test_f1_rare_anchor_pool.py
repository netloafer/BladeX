"""稀有锚：路径类单独剔除，判据在**两个池**上都要成立（F1.2 / MQ-A51，2026-09-08）。

## A47 错在哪

A47 的立论是"路径 token 是通用词、df 高，用 df 就能剔"。复核实测（09-07 全天
`usable=True` 17 行）：`documents,hermes` **仍在 5 行的 `rare_sample` 里**，
一条没剔掉；被剔掉的反而是「报告/更新/结果/结论」这类真通用词。
成因：31 份账本 MD 里 `df(documents)=1`、`df(hermes)=3`，阈值 `max(2, N//10)`≈3
⇒ 它们**按 df 就是稀有的**。前提只在"池内多本谈不同项目"时成立，本池主题集中。

⇒ 底噪的本质是**类别**（文件路径的寻址片段），不是词频。两条判据现在**并联**：
df 稀有 ∧ 非路径类。

## 🔴 为什么必须有两个池

A47 的判别力对照用的是"不做稀有锚则 live 病例回放必红"——**回放的是同一个小池**，
错误前提在测试与生产里是同一个，于是测试只能证明"代码按设计跑了"，证不了设计对
（feedback_instrument_reference_frame：尺子与被测对象共享同一个错误前提时，
自检必然通过）。所以本文件的每条判据都要在两个前提相反的池上同时成立：

- **池①** live 病例池：主题高度集中（同一批任务），路径 token df **低**；
- **池②** 人工多主题池：5 本谈 5 个不同项目、路径各异，真通用词 df **高**。

池① 单独通过 = A47 的形态，不算过。
"""

from __future__ import annotations

import pathlib
import re

from bladex_core.ledger import ACTOR_MODEL, LedgerEntry, add_entry, new_ledger
from bladex_core.ledger_runtime import (
    candidate_units,
    looks_like_path,
    path_like_units,
)
from bladex_proxy.agency import rare_pending_units


def _led(lid: str, *pending: str):
    led = new_ledger(ledger_id=lid, title=lid, goal="g")
    for text in pending:
        led = add_entry(led, "next", LedgerEntry(text=text), actor=ACTOR_MODEL)
    return led


def _pending_text(led) -> str:
    return " ".join(e.text for sec in ("next", "open") for e in led.entries(sec))


def _rare(pool: dict, lid: str) -> frozenset[str]:
    """生产侧 `_build_ledger_block` 的那三句（口径不许在测试里另写一份）。"""
    text = _pending_text(pool[lid])
    return rare_pending_units(pool, frozenset(candidate_units(text)),
                              pending_text=text)


# ── 池① live 病例池（09-07 那几本的 Next/Open 文本作夹具）─────────────────


_LIVE_TARGET = "ldg-c5103"


def _live_pool() -> dict:
    """主题集中：同一批台球视觉任务，路径 token 只出现在少数几本里（df 低）。"""
    pool = {
        _LIVE_TARGET: _led(
            _LIVE_TARGET,
            "用户决策：缺口补全移交 Hermes default 执行，交接提示词已生成存档 "
            "~/Documents/Pi/台球视觉调研-Hermes交接提示词-20260907.md",
            "重建 detect_track.py/classic_hough.py/viz_trajectory.py 并实测复现，"
            "map50 与 coco 预训练基准对齐",
        ),
        "ldg-b": _led("ldg-b", "把收尾报告 ~/Documents/Hermes/台球视觉调研收尾"
                               "任务报告-20260907.md 的 50 epochs 表述更正"),
        "ldg-c": _led("ldg-c", "核验单应性标定的结果并写进结论"),
    }
    for i in range(4):
        pool[f"ldg-x{i}"] = _led(f"ldg-x{i}", f"更新报告与结果，确认结论{i}")
    return pool


def test_live_pool_drops_path_segments():
    """🔴 A51 的正面判据：`documents` / `hermes` 出 `rare`。

    这两个词 df 低（本池主题集中）⇒ df 那一条永远抓不到它们；
    抓到它们的是路径类判据。
    """
    rare = _rare(_live_pool(), _LIVE_TARGET)
    assert "documents" not in rare and "hermes" not in rare, \
        f"路径段仍在 rare：{sorted(rare)}"


def test_live_pool_keeps_the_real_anchors():
    """剔底噪不能把信号一起剔掉：`map50` / `coco` / `classic_hough.py` 必须留下。

    🔴 `classic_hough.py` 是**最容易误伤**的那个：模型用 `/` 连写文件名清单
    （`detect_track.py/classic_hough.py/viz_trajectory.py`，live 病例里真实存在），
    含 `/` 且末段带扩展名——只看这两条就会把它当路径整片剔掉。
    `path_like_units` 因此还要求"至少有一个不带扩展名的路径段"。
    """
    rare = _rare(_live_pool(), _LIVE_TARGET)
    for u in ("map50", "coco", "classic_hough.py"):
        assert u in rare, f"{u} 是只有这件事才有的词，不该被剔：{sorted(rare)}"


def test_live_pool_keeps_cjk_inside_file_names():
    """文件名里的中文是**内容**不是寻址：`台球` / `调研` 必须留下。

    整片剔掉路径 = 把 `台球视觉` 一起剔走，那是把信号当底噪的第二种错法。

    （`调研` 不在断言里：它在 `candidate_units` 的对话停表里，从来就不是一个单元。）
    """
    rare = _rare(_live_pool(), _LIVE_TARGET)
    assert "台球" in rare and "交接" in rare, sorted(rare)


# ── 池② 人工多主题池（前提与池① 相反）───────────────────────────────────


_MULTI_TARGET = "ldg-m0"


def _multi_pool() -> dict:
    """5 本谈 5 个不同项目、路径各异；「报告 / 更新」在每本里都出现（df 高）。"""
    specs = [
        ("ldg-m0", "更新 /srv/freetoken/api/handler.py 的限流阈值，报告结果"),
        ("ldg-m1", "更新 ~/work/taishan/brew/recipe.md 的配比，报告结果"),
        ("ldg-m2", "更新 /opt/lantern/etl/loader.py 的重试，报告结果"),
        ("ldg-m3", "更新 ~/dev/BladeX/packages/bladex-core/flags.py 的默认值，报告结果"),
        ("ldg-m4", "更新 /data/orchid/vision/train.py 的学习率，报告结果"),
    ]
    return {lid: _led(lid, text) for lid, text in specs}


def test_multi_topic_pool_drops_the_truly_common_words():
    """前提相反的池上，df 那一条要真的在干活：「更新 / 报告 / 结果」出 `rare`。"""
    rare = _rare(_multi_pool(), _MULTI_TARGET)
    for u in ("更新", "报告", "结果"):
        assert u not in rare, f"{u} 在 5/5 本里都有，不该算稀有：{sorted(rare)}"


def test_multi_topic_pool_keeps_the_project_name():
    """项目名（`freetoken`）留下 —— 它在池里只出现一次，正是"只有这件事才有"的词。

    🔴 它出现在**路径里**（`/srv/freetoken/api/handler.py`），所以本条同时是
    路径类剔除的**代价说明**：路径段一律不当锚，`freetoken` 也不例外。
    留下它的是同一条 Next 里的正文吗？不是——这里它只在路径里出现过。
    ⇒ 断言的是 `hits_rare` 的**方向**：它是下界，不是等号。
    """
    rare = _rare(_multi_pool(), _MULTI_TARGET)
    assert "freetoken" not in rare, \
        "路径里的项目名同样按路径类剔除——少算比多算诚实（见 path_like_units 文档）"
    # 但这本账本必须**还剩**锚，否则整行进不了分母、读数直接归零
    assert rare, f"整本被剔空 ⇒ 这把尺子在多主题池上失效：{sorted(rare)}"
    assert "限流" in rare or "阈值" in rare


# ── 小池保护：分母是"Next/Open 非空的本"，不是 `len(pool)` ──────────────


def test_small_corpus_still_strips_paths_but_does_not_fake_df():
    """Next/Open 非空的本 < 2 ⇒ df 分辨不出稀有 ⇒ 只做路径类剔除后原样返回。

    🔴 判据从 `len(pool) < 2` 改成"**非空**的本 < 2"（MQ-A51 连带）：
    空 Next/Open 的账本一个词都不贡献 df，却照样把 `len(pool)` 撑到过关
    —— 同批第六次"分母不是被测对象"。
    """
    pool = {"a": _led("a", "更新 ~/Documents/Hermes/x.md 的 map50"),
            "b": _led("b"), "c": _led("c"), "d": _led("d")}   # 三本待办为空
    rare = _rare(pool, "a")
    assert "map50" in rare, "df 不可用时不猜——留着"
    assert "documents" not in rare and "hermes" not in rare, \
        "路径类剔除与 df 无关，小池上照做"


def test_degenerate_pools_do_not_raise():
    """脏池 / 空池只影响 df，不炸（`rare_pending_units` 在热路径上）。"""
    units = frozenset({"documents", "map50"})
    assert rare_pending_units({}, units, pending_text="") == units
    assert rare_pending_units({"a": None}, units, pending_text="") == units
    assert rare_pending_units({}, frozenset(), pending_text="x") == frozenset()


# ── `path_like_units` / `looks_like_path` 本体 ─────────────────────────────


def test_path_like_units_only_takes_latin_from_real_paths():
    got = path_like_units("存档 ~/Documents/Pi/台球视觉调研-20260907.md 完毕")
    assert "documents" in got
    assert "台球" not in got and "调研" not in got, "文件名里的中文是内容"


def test_path_like_units_ignores_a_slash_joined_file_list():
    assert path_like_units(
        "重建 detect_track.py/classic_hough.py/viz_trajectory.py") == frozenset()


def test_path_like_units_ignores_bare_words():
    assert path_like_units("把 map50 提到 0.82，跑 classic_hough.py") == frozenset()


def test_looks_like_path_has_one_definition():
    """🔴 守卫：全仓只剩一个定义点（探针 import，不留副本）。"""
    root = pathlib.Path(__file__).resolve().parents[3]
    pat = re.compile(r"^def looks_like_path\(", re.M)
    # 生成物不算定义点：public-tree/ 是 build_public_tree.py 的输出（私仓源码的复印件，
    # 09-18 批 O gate 实跑被它红过一次），.venv/ 是安装产物。
    # 🔴 按**相对仓库根**的路径排除——公开树自己就住在 <私仓>/public-tree/ 下，
    # 按绝对路径的 parts 排会把公开树里所有文件排光（公开树自测 0 命中，同日第二次红）。
    skip = {"__pycache__", "archive", "public-tree", ".venv"}
    hits = [p for p in root.rglob("*.py")
            if not (skip & set(p.relative_to(root).parts))
            and pat.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert len(hits) == 1, f"两份判据迟早分叉：{[str(p) for p in hits]}"
    assert hits[0].name == "ledger_runtime.py"


def test_looks_like_path_behaviour_is_unchanged_by_the_move():
    """下沉是**搬家不是改判据**：原判别力对照逐条照抄。"""
    assert looks_like_path("/Users/j/Documents/Hermes/x.md")
    assert looks_like_path("scripts/probe_ledger_files.py")
    assert not looks_like_path("/Users/you")          # 末段无扩展名
    assert not looks_like_path("~/Documents/Hermes/")     # 目录
    assert not looks_like_path(None) and not looks_like_path("")
    assert not looks_like_path("a/b.md\nc")               # 含换行


# ── 判别力 ─────────────────────────────────────────────────────────────────


def test_discriminative_without_path_class_the_live_pool_goes_red(monkeypatch):
    """去掉路径类剔除 ⇒ 池① 必红（`documents` / `hermes` 回到 rare）。

    🔴 打**消费方子模块**（`bladex_proxy.agency.runtime` 函数体内 import 的那个），
    F0 拆包后打门面不生效。
    """
    import bladex_core.ledger_runtime as lr
    monkeypatch.setattr(lr, "path_like_units", lambda text: frozenset())
    rare = _rare(_live_pool(), _LIVE_TARGET)
    assert "documents" in rare and "hermes" in rare, \
        "去掉路径类剔除后底噪必须回来——否则剔掉它们的不是这条判据"


def test_discriminative_without_df_the_multi_topic_pool_goes_red(monkeypatch):
    """反向对照：只剩路径类、去掉 df ⇒ 池② 必红（真通用词回到 rare）。

    两个方向都钉住，才说明"并联"里的两条**各自都在干活**——
    少了这一条，一个恒真的 df 判据也能让上面的测试全绿。
    """
    import bladex_proxy.agency.runtime as rt
    monkeypatch.setattr(rt, "_pending_corpus", lambda pool: [])   # df 语料清空
    rare = _rare(_multi_pool(), _MULTI_TARGET)
    assert "更新" in rare and "报告" in rare, \
        "去掉 df 后真通用词必须回来——否则剔掉它们的不是 df"
