"""批 CB：交接包 v1 的纯函数（`bladex_core.handoff`）。

钉：sanitize 的路径 / 凭证口径（含白名单留存）· 哈希覆盖全部字段值 · ①永不裁、③→② 裁剪与计数 ·
确定性 · 字段白名单（`goal_verbatim` 等不进包）· 导入校验顺序与落地身份。

样本是按真实形态构造的，不是 Hub 原文（MQ-V40）。
"""

from __future__ import annotations

import copy
import json

import pytest
from bladex_core.handoff import (
    HandoffImportError,
    HandoffLimits,
    HandoffOmitted,
    HandoffRewrites,
    RawRound,
    build_package,
    canonical_json,
    content_hash,
    ledger_from_package,
    parse_package,
    render_markdown,
    rounds_from_events,
    sanitize,
)
from bladex_core.ledger import LedgerEntry, add_entry, new_ledger, replay_ledger_events
from bladex_core.ledger_runtime import ledger_anchor_matter_id

ROOT = "/Users/alice/dev/proj"


def _s(text: str, roots: tuple[str, ...] = (ROOT,)) -> tuple[str, HandoffRewrites]:
    c = HandoffRewrites()
    return sanitize(text, roots, c), c


@pytest.mark.parametrize(("src", "want"), [
    ("/Users/alice/dev/proj/a.py", "<project>/a.py"),
    ("跑 /Users/alice/dev/proj 下的测试", "跑 <project> 下的测试"),
    ("/Users/alice/dev/other/b.py", "~/dev/other/b.py"),
    ("/home/bob/x", "~/x"),
    ("/root/.ssh/id", "~/.ssh/id"),
    ("PATH=a:/home/bob/bin", "PATH=a:~/bin"),
    ("file:///Users/alice/a.md", "~/a.md"),
    ("C:\\Users\\bob\\x.txt", "~\\x.txt"),
    ("C:\\tools\\x.exe", "<path>"),
    ("/tmp/a.log 以及 /etc/hosts", "<path> 以及 <path>"),
    ("日志在 /private/tmp/x/y。", "日志在 <path>。"),
    ("/srv/app/run.sh", "<path>"),
    ("/custom/place/run.sh", "<path>"),
])
def test_sanitize_rewrites_local_paths(src: str, want: str) -> None:
    assert _s(src)[0] == want


def test_sanitize_cjk_touching_path_is_still_a_path() -> None:
    """第一版原型用 `\\w` 当边界：中文算 `\\w`，紧贴中文的路径漏掉了。74 本真样本抓到的。"""
    assert _s("裁定书存到/Users/alice/Documents/a.pdf 了")[0] == "裁定书存到~/Documents/a.pdf 了"


def test_sanitize_project_root_wins_over_home_and_longest_root_first() -> None:
    assert _s("/Users/alice/dev/proj/sub/a", (ROOT, f"{ROOT}/sub"))[0] == "<project>/a"
    # 根是前缀但不在路径段边界 ⇒ 不是这个项目
    assert _s("/Users/alice/dev/proj2/a")[0] == "~/dev/proj2/a"


@pytest.mark.parametrize("kept", [
    "/admin/ledgers/user-edit",
    "/v1/responses",
    "/api/v1/search",
    "/扣非11.35亿/经营现金流11.00亿",
    "/404=167/301=101/500=55",
    "/12核GPU/双16核Neural",
    "orders 6/users 4/users/999999",
    "https://example.com/home/index",
    "src/bladex/a.py",
    "~/dev/x",
    "task-codex-mini-very-long-name",
])
def test_sanitize_whitelist_stays_verbatim(kept: str) -> None:
    """(41)-a-⑤：改写之外的留存面写成白名单。这些是 live 账本里的正文形状，不是本地路径。"""
    out, c = _s(kept)
    assert out == kept
    assert c == HandoffRewrites()


@pytest.mark.parametrize(("src", "secret"), [
    ("api_key=abcdef-ghijk-lmnopqrstuv", "abcdef-ghijk-lmnopqrstuv"),
    ("api_key: `abcdef-ghijk-lmnopqrstuv`", "abcdef-ghijk-lmnopqrstuv"),
    ('"password": "hunter2hunter2"', "hunter2hunter2"),
    ("用 sk-abcdefghijklmnopqrstuvwx 调", "sk-abcdefghijklmnopqrstuvwx"),
    ("ghp_abcdefghijklmnopqrstuvwxyz123456", "ghp_abcdefghijklmnopqrstuvwxyz123456"),
    ("Authorization: Bearer abcdefghij0123456789", "abcdefghij0123456789"),
    ("token=abc123def456ghi789", "abc123def456ghi789"),
])
def test_sanitize_credentials_replaced_and_only_counted(src: str, secret: str) -> None:
    out, c = _s(src)
    assert secret not in out
    assert "<redacted>" in out
    assert c.credential == 1
    assert secret not in c.model_dump_json()


@pytest.mark.parametrize(("src", "secret"), [
    # 拍板 (43)-a-②：键不分大小写（此前去掉 `(?i)` 全绿，没有用例）
    ("OPENAI_API_KEY=abcdef-ghijk-lmnop", "abcdef-ghijk-lmnop"),
    ("API_KEY: abcdefghijklmnop", "abcdefghijklmnop"),
    ("Secret = abcdefgh", "abcdefgh"),
    ("PASSWORD=abcdefgh", "abcdefgh"),
    # 带特殊符号的、短的密码（此前值字符类不含 `!` `#`，且要 ≥ 8）
    ("password = hunter2!x", "hunter2!x"),
    ("passwd: p#ss!w", "p#ss!w"),
    ("secret=a%b^c&d", "a%b^c&d"),
    ('"password": "h!nter"', "h!nter"),
    # MQ-L108：`secret` 后接 `_key`、前接 `client_`
    ("SECRET_KEY=abcdef123456", "abcdef123456"),
    ("secret_key: abcdef123456", "abcdef123456"),
    ("SECRET-KEY = abcdef123456", "abcdef123456"),
    ("client_secret: abcdef123456", "abcdef123456"),
    ('"client_secret": "abcdef123456"', "abcdef123456"),
    ("CLIENT_SECRET=abcdef123456", "abcdef123456"),
    ('"SECRET_KEY": "abcdef123456"', "abcdef123456"),
])
def test_sanitize_uppercase_keys_and_special_char_passwords(src: str, secret: str) -> None:
    out, c = _s(src)
    assert secret not in out
    assert "<redacted>" in out
    assert c.credential == 1


@pytest.mark.parametrize("kept", [
    "密码：见文档",
    "password：见文档",
    "password: <your-password>",
    "secret=${MY_SECRET}",
    "API_KEY=<your-key>",
    "api_key=${API_KEY}",
    "password: 见文档",
    "password=abc",                           # 短于 6
    "secret: 这是中文正文",
    # MQ-L108 阴性：`secret` 只是更长单词/目录键的一部分，不是凭证键
    "secretary: abcdefghijkl",
    "secrets_dir=/abcdefghijkl",
    "secret_keeper: abcdefghijkl",
    "SECRET_KEY=${MY_SECRET}",
    "SECRET_KEY=<your-key>",
])
def test_sanitize_placeholders_and_prose_stay(kept: str) -> None:
    out, c = _s(kept)
    assert out == kept
    assert c.credential == 0


@pytest.mark.parametrize("kept", [
    "token: 12345678901234567890",            # 计数
    "prompt token: 每轮 300",                  # 正文
    "api_key: <redacted>",                    # 已是占位
    "api_key=${MY_KEY}",
    "password 必须轮换",
])
def test_sanitize_credential_lookalikes_stay(kept: str) -> None:
    out, c = _s(kept)
    assert out == kept
    assert c.credential == 0


def _ledger(**kw):
    led = new_ledger(ledger_id="ldg-aaaaaaaaaaaa", title=kw.get("title", "任务"),
                     goal=kw.get("goal", "把 X 做完"), goal_source="model",
                     goal_verbatim="用户原话不许进包", goal_origin="u/a/s/1-0",
                     created_at="2026-11-02T00:00:00+00:00")
    for sec, text in kw.get("entries", [("core", "一条 Core"), ("verified", "一条 Verified")]):
        led = add_entry(led, sec, LedgerEntry(text=text, source="model", ref="r1", writer="codex"),
                        actor="model")
    return led


def _pkg(**kw):
    led = kw.pop("led", None) or _ledger()
    return build_package(led, exported_at="2026-11-02T01:00:00+00:00", bladex_version="0.4.0",
                         source_project="p-1", project_roots=(ROOT,), **kw)


def test_field_table_matches_f1_and_nothing_else() -> None:
    d = _pkg(matters=[("标题", "摘要")],
             rounds=[RawRound(at="2026-11-02T00:30:00+00:00", agent="codex",
                              added=[("next", "下一步")], user_text="原话")]).model_dump(mode="json")
    assert set(d) == {"format_version", "bladex_version", "exported_at", "source_ledger_id",
                      "source_project", "target_exposure", "content_hash", "ledger", "matters",
                      "recent_rounds", "omitted", "trimmed", "rewrites"}
    assert set(d["ledger"]) == {"title", "status", "rev", "goal", "goal_source", "section_order",
                                "sections"}
    text = canonical_json(d)
    for leak in ("用户原话不许进包", "u/a/s/1-0", "goal_verbatim", "goal_origin", "matter_id",
                 "parent_ledger_id", "superseded_by"):
        assert leak not in text


def test_deterministic_same_input_same_bytes() -> None:
    a = canonical_json(_pkg().model_dump(mode="json"))
    assert a == canonical_json(_pkg().model_dump(mode="json"))


def test_goal_and_core_verbatim_under_sanitize_contract() -> None:
    goal = f"按 {ROOT}/plan.md 做\n\n第二段  保留空行和空格 "
    core = f"约束：不改 {ROOT}/x.py；路由 /admin/ledgers 不动"
    led = _ledger(goal=goal, entries=[("core", core), ("core", "没有路径的一条")])
    pkg = _pkg(led=led)
    assert pkg.ledger.goal == sanitize(goal, (ROOT,))
    assert [e.text for e in pkg.ledger.sections["core"]] == [sanitize(core, (ROOT,)), "没有路径的一条"]
    assert pkg.ledger.sections["core"][1].text == "没有路径的一条"
    assert pkg.rewrites.project == 2


def test_section_order_and_unknown_sections_kept() -> None:
    led = _ledger()
    led = led.model_copy(update={"section_order": [*led.section_order, "lessons"],
                                 "sections": {**led.sections, "lessons": [LedgerEntry(text="教训")]}})
    pkg = _pkg(led=led)
    assert pkg.ledger.section_order[-1] == "lessons"
    assert pkg.ledger.sections["lessons"][0].text == "教训"


def _big_rounds(n: int) -> list[RawRound]:
    return [RawRound(at=f"2026-11-02T00:{59 - i:02d}:00+00:00", agent="codex",
                     added=[("next", f"第{i}轮" + "x" * 300)], user_text="y" * 200)
            for i in range(n)]


def test_trim_rounds_oldest_first_then_matters_never_sections() -> None:
    led = _ledger(entries=[("core", "c" * 3000)])
    full = _pkg(led=led, matters=[("m", "s" * 1000)], rounds=_big_rounds(5))
    assert full.trimmed.rounds == 0
    size = len(canonical_json(full.model_dump(mode="json")))
    tight = _pkg(led=led, matters=[("m", "s" * 1000)], rounds=_big_rounds(5),
                 limits=HandoffLimits(max_chars=size - 600))
    assert tight.trimmed.rounds >= 1 and tight.trimmed.matters == 0
    assert len(tight.recent_rounds) == 5 - tight.trimmed.rounds
    assert tight.recent_rounds[0].added[0].text.startswith("第0轮"), "最新的一轮留下，去的是最旧的"
    assert tight.ledger == full.ledger
    only_sections = _pkg(led=led, matters=[("m", "s" * 1000)], rounds=_big_rounds(5),
                         limits=HandoffLimits(max_chars=4200))
    assert only_sections.recent_rounds == [] and only_sections.matters == []
    assert only_sections.trimmed.rounds == 5 and only_sections.trimmed.matters == 1
    assert only_sections.ledger == full.ledger
    assert only_sections.trimmed.over_limit is False


def test_sections_alone_over_limit_still_exported_and_flagged() -> None:
    led = _ledger(entries=[("core", "c" * 5000)])
    pkg = _pkg(led=led, limits=HandoffLimits(max_chars=1000))
    assert pkg.ledger.sections["core"][0].text == "c" * 5000
    assert pkg.trimmed.over_limit is True


def test_round_caps_are_counted_and_truncation_is_marked() -> None:
    r = RawRound(at="t", added=[("next", str(i)) for i in range(11)],
                 removed=[("open", str(i)) for i in range(9)], user_text="z" * 40 + "\n\n第二段")
    pkg = _pkg(rounds=[r, *[RawRound(at="t", added=[("next", "x")]) for _ in range(6)]],
               limits=HandoffLimits(max_rounds=5, user_text_chars=10, round_entries=8))
    assert len(pkg.recent_rounds) == 5
    assert pkg.trimmed.rounds == 2
    assert pkg.trimmed.round_entries == 3 + 1
    head = pkg.recent_rounds[0]
    assert head.user_text_truncated is True
    assert head.user_text_head == "z" * 10 + "…<bladex:truncated>"
    assert "第二段" not in head.user_text_head
    assert pkg.omitted.rounds_without_user_text == 4


def test_omitted_counts_pass_through_and_input_not_mutated() -> None:
    om = HandoffOmitted(matters_exposure=1, rounds_exposure=2)
    pkg = _pkg(omitted=om, rounds=[RawRound(at="t", added=[("next", "x")])])
    assert pkg.omitted.matters_exposure == 1 and pkg.omitted.rounds_exposure == 2
    assert om.rounds_without_user_text == 0


def test_rounds_group_by_writer_turn_and_skip_no_change() -> None:
    def ev(etype, sections, **payload):
        return {"type": etype, "ts": payload.pop("ts", "2026-11-02T00:00:00+00:00"), "payload": {
            "ledger": {"ledger_id": "ldg-1", "sections": {
                k: [{"text": t} for t in v] for k, v in sections.items()}}, **payload}}
    events = [
        ev("ledger_create", {"next": ["a"]}),
        ev("ledger_update", {"next": ["a", "b"]}, session_id="s", turn_index=3, agent_id="cx",
           ts="2026-11-02T00:01:00+00:00"),
        ev("ledger_update", {"next": ["b"], "verified": ["a 完成"]}, session_id="s", turn_index=3,
           agent_id="cx", ts="2026-11-02T00:02:00+00:00"),
        ev("ledger_update", {"next": ["b"], "verified": ["a 完成"]}, session_id="s", turn_index=4,
           agent_id="cx", ts="2026-11-02T00:03:00+00:00"),         # 正文没变
        ev("ledger_user_edit", {"next": ["b", "用户加的"], "verified": ["a 完成"]},
           ts="2026-11-02T00:04:00+00:00"),                        # 只推进基线
        ev("ledger_update", {"next": ["用户加的"], "verified": ["a 完成"]},
           ts="2026-11-02T00:05:00+00:00"),                        # 历史形态：没带写者
        {"type": "ledger_update", "ts": "t", "payload": {"ledger": {"ledger_id": "ldg-2",
                                                                    "sections": {"next": [{"text": "别本"}]}}}},
    ]
    rounds = rounds_from_events(events, "ldg-1")
    assert len(rounds) == 2
    old_form, merged = rounds
    assert old_form.session_id == "" and old_form.removed == [("next", "b")]
    assert merged.turn_index == 3
    assert merged.added == [("next", "b"), ("verified", "a 完成")]
    assert merged.removed == [("next", "a")]


def test_markdown_body_never_starts_a_line_with_slash() -> None:
    led = _ledger(goal="第一行\n/v1/messages 第二行", entries=[("core", "/admin/ledgers 路由\n/v1/x 续行")])
    md = render_markdown(_pkg(led=led, matters=[("m", "摘要\n/v1/y")],
                              rounds=[RawRound(at="t", added=[("next", "/v1/z\n/v1/w")], user_text="/v1/u")]))
    assert not [ln for ln in md.splitlines() if ln.startswith("/")]
    assert "--format json" in md


# ── 导入校验 ────────────────────────────────────────────────────────────────


def _raw(**kw) -> dict:
    return _pkg(**kw).model_dump(mode="json")


def _leaf_paths(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaf_paths(v, (*path, k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaf_paths(v, (*path, i))
    elif isinstance(obj, str) and path != ("content_hash",):
        yield path


def test_tampering_any_string_field_is_rejected() -> None:
    raw = _raw(matters=[("标题", "摘要")],
               rounds=[RawRound(at="t", agent="cx", added=[("next", "下一步")],
                                removed=[("open", "旧")], user_text="原话")])
    paths = list(_leaf_paths(raw))
    assert len(paths) > 15
    for path in paths:
        bad = copy.deepcopy(raw)
        node = bad
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = node[path[-1]] + "x"
        with pytest.raises(HandoffImportError) as e:
            parse_package(bad)
        assert e.value.code in ("hash", "shape"), path


def test_tampering_numbers_and_rounds_is_rejected() -> None:
    raw = _raw(rounds=[RawRound(at="t", added=[("next", "x")])])
    for mutate in (lambda d: d["ledger"].__setitem__("rev", 99),
                   lambda d: d["rewrites"].__setitem__("credential", 7),
                   lambda d: d["trimmed"].__setitem__("rounds", 7),
                   lambda d: d["recent_rounds"].clear(),
                   lambda d: d["omitted"].__setitem__("matters_exposure", 7)):
        bad = copy.deepcopy(raw)
        mutate(bad)
        with pytest.raises(HandoffImportError) as e:
            parse_package(bad)
        assert e.value.code == "hash"


def test_reformatting_whitespace_outside_strings_is_not_tampering() -> None:
    raw = _raw()
    assert parse_package(json.loads(json.dumps(raw, indent=4))).content_hash == raw["content_hash"]


@pytest.mark.parametrize("version", [2, 0, "1", None, True])
def test_unknown_version_rejected_and_named_even_with_valid_hash(version) -> None:
    raw = _raw()
    raw["format_version"] = version
    raw["content_hash"] = content_hash(raw)
    with pytest.raises(HandoffImportError) as e:
        parse_package(raw)
    assert e.value.code == "version"
    assert repr(version) in str(e.value)


@pytest.mark.parametrize("mutate", [
    lambda d: d.__setitem__("extra", 1),
    lambda d: d["ledger"].__setitem__("goal_verbatim", "x"),
    lambda d: d["ledger"]["sections"]["core"][0].__setitem__("source", "bogus"),
    lambda d: d["ledger"]["sections"].__setitem__("Bad Name", []),
    lambda d: d["ledger"]["sections"].__setitem__("goal", []),
    lambda d: d["ledger"].__setitem__("goal", "   "),
])
def test_bad_shape_rejected_even_with_recomputed_hash(mutate) -> None:
    raw = _raw()
    mutate(raw)
    raw["content_hash"] = content_hash(raw)
    with pytest.raises(HandoffImportError) as e:
        parse_package(raw)
    assert e.value.code == "shape"


@pytest.mark.parametrize("raw", [[], "x", 1, None])
def test_non_object_rejected(raw) -> None:
    with pytest.raises(HandoffImportError):
        parse_package(raw)


def test_ledger_from_package_identity_is_user_and_rounds_stay_out() -> None:
    raw = _raw(matters=[("标题", "摘要只存不注入")],
               rounds=[RawRound(at="t", added=[("next", "轮次只存不注入")], user_text="原话只存不注入")])
    pkg = parse_package(raw)
    led = ledger_from_package(pkg, ledger_id="ldg-bbbbbbbbbbbb", now_iso="2026-11-02T02:00:00+00:00")
    assert led.goal == pkg.ledger.goal
    assert led.goal_source == "user"
    assert led.goal_origin == f"handoff:ldg-aaaaaaaaaaaa@{pkg.content_hash[:12]}"
    assert led.goal_verbatim == "" and led.goal_revisions == 0
    assert led.rev == 1 and led.last_writer == "user" and led.status == "active"
    assert led.matter_id == ledger_anchor_matter_id("ldg-bbbbbbbbbbbb")
    assert led.parent_ledger_id == "" and led.superseded_by == ""
    e = led.sections["core"][0]
    assert (e.text, e.source, e.ref, e.writer) == ("一条 Core", "import", "r1", "codex")
    assert "不注入" not in led.model_dump_json()


def test_imported_event_replays_equal() -> None:
    pkg = parse_package(_raw())
    led = ledger_from_package(pkg, ledger_id="ldg-bbbbbbbbbbbb", now_iso="2026-11-02T02:00:00+00:00")
    pool, bindings = replay_ledger_events([{"type": "ledger_user_edit", "payload": {
        "ledger": led.model_dump(), "origin": "handoff_import", "handoff": {"package": {}}}}])
    assert pool["ldg-bbbbbbbbbbbb"] == led
    assert bindings == {}


def test_flags_supply_the_limits(monkeypatch) -> None:
    from bladex_core.handoff import limits_from_flags
    assert limits_from_flags() == HandoffLimits()
    monkeypatch.setenv("BLADEX_HANDOFF_MAX_ROUNDS", "2")
    assert limits_from_flags().max_rounds == 2
