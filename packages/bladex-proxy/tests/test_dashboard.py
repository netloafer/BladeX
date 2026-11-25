"""Beta T18–T20 dashboard 验收（服务面 + 数据端点；浏览器交互留用户机真机）。

钉住：/dashboard 从包资源正常出页（wheel 安装态可用）；页面不含内嵌密钥；
数据面走 /admin/*（auth 开启时无 key 401 = "无 key 访问被拒"）；
/admin/turns 列表端点（回放页数据源）prefix/分页/排序。
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from bladex_proxy.config import ProxyConfig
from bladex_proxy.models import Identity, Turn
from bladex_proxy.server import create_app
from fastapi.testclient import TestClient


def _make_config(auth: bool = False) -> ProxyConfig:
    tmpdir = tempfile.mkdtemp()
    return ProxyConfig(
        auth_enabled=auth,
        client_keys_raw="bladex-valid||test" if auth else "",
        hard_rules=["MUST be polite"],
        upstream_model="openai/test",
        upstream_api_key="sk-fake",
        rocksdb_path=f"{tmpdir}/rocksdb",
        index_path=f"{tmpdir}/index",
    )


def test_dashboard_serves_html_with_all_views():
    app = create_app(_make_config())
    with TestClient(app) as client:
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        html = resp.text
        # T18 状态总览 + T19 记忆/Matter 管理 + T20 图/回放 的关键部件都在
        for marker in ("/admin/status", "/admin/facts", "/admin/matters",
                       "/admin/sessions", "/admin/turns", "/admin/hard_rules",
                       "graph-canvas", "sessionStorage", "decision_meta",
                       "forgetFact", "matterOp"):
            assert marker in html, f"dashboard missing {marker}"


def test_dashboard_page_contains_no_secrets():
    """页面是无鉴权静态资源：不得内嵌任何 key/凭证。"""
    app = create_app(_make_config(auth=True))
    with TestClient(app) as client:
        html = client.get("/dashboard").text
        assert "bladex-valid" not in html
        # 数据面在 auth 开启时无 key 被拒（dashboard 的"登录"语义）
        assert client.get("/admin/status").status_code == 401


def test_admin_turns_index_prefix_and_order():
    app = create_app(_make_config())
    with TestClient(app) as client:
        ledger = app.state.hub
        for i, ts in enumerate(["2026-08-01T10:00:00+00:00",
                                "2026-08-03T10:00:00+00:00",
                                "2026-08-02T10:00:00+00:00"]):
            from datetime import datetime
            ledger.put(f"u1/agentA/s1/{i:04d}",
                   Turn(identity=Identity(user_id="u1", agent_id="agentA",
                                          session_id="s1"),
                        ts=datetime.fromisoformat(ts)))
        ledger.put("u1/agentB/s2/0000",
               Turn(identity=Identity(user_id="u1", agent_id="agentB",
                                      session_id="s2")))

        d = client.get("/admin/turns",
                       params={"prefix": "u1/agentA/s1/"}).json()
        assert d["total"] == 3
        # ts 降序
        keys = [t["key"] for t in d["turns"]]
        assert keys[0].endswith("0001")  # 08-03 最新
        # 分页
        d2 = client.get("/admin/turns",
                        params={"prefix": "u1/agentA/s1/", "limit": 2}).json()
        assert len(d2["turns"]) == 2 and d2["total"] == 3
        # 无 prefix 全量
        assert client.get("/admin/turns").json()["total"] == 4


class TestTabRegistration:
    """导航结构自洽：TABS / VIEWS / SUBS / IC 四张表必须对齐。

    加这条是因为差点漏了一步：`TABS` 里注册了 `ledgers`，`VIEWS` 里忘了写，
    页面**能显示导航项，一点就抛**——四张表各自看起来都对，只有对照才发现。
    纯字符串检查（页面是单文件 SPA，没有 JS 运行时可用）。
    """

    @staticmethod
    def _html() -> str:
        from importlib import resources
        return (resources.files("bladex_proxy") / "dashboard.html").read_text(
            encoding="utf-8")

    @staticmethod
    def _tab_ids(html: str) -> list[str]:
        import re
        block = re.search(r"const TABS=\[(.*?)\];", html, re.S)
        assert block, "TABS 声明找不到了——改了形态就得同步改这条守卫"
        ids = re.findall(r'\["([a-z]+)",', block.group(1))
        assert len(ids) >= 5, ids
        return ids

    def test_every_tab_has_a_view(self):
        html = self._html()
        import re
        views = re.search(r"const VIEWS=\{(.*?)\};", html, re.S)
        assert views
        registered = set(re.findall(r"(\w+):", views.group(1)))
        missing = [t for t in self._tab_ids(html) if t not in registered]
        assert not missing, f"TABS 里有、VIEWS 里没有（点了会抛）: {missing}"

    def test_every_tab_has_subtitle_and_icon(self):
        html = self._html()
        import re
        subs = re.search(r"const SUBS=\{(.*?)\};", html, re.S)
        ic = re.search(r"const IC=\{(.*?)\n\};", html, re.S)
        assert subs and ic
        has_sub = set(re.findall(r"(\w+):", subs.group(1)))
        has_ic = set(re.findall(r"(\w+):", ic.group(1)))
        for t in self._tab_ids(html):
            assert t in has_sub, f"{t} 缺副标题"
            assert t in has_ic, f"{t} 缺图标"

    def test_ledgers_tab_is_present(self):
        assert "ledgers" in self._tab_ids(self._html())


# ══════════════════════════════════════════════════════════════════════════
# 批 DS1：dashboard 接线 —— 页面 JS 在 node 里真跑（假 DOM + 假 fetch），
# 截获它发出的请求，再原样打给 TestClient 上的服务端，与 CLI 同操作的 Hub 事件对账。
#
# 为什么要真跑：DS1 的判据是行为（发了什么 body、按钮在请求结束前是不是不可点、
# 202 显示成什么），字符串检查对拆掉加载状态、发错 origin 这类变异是绿的。
# 页面是单文件、没有构建链，venv 里也没有 JS 引擎；node 在本机可用（Jason 机 v22），
# 没有 node 的机器上这组用例 skip 并写明原因。skip 只留给公开树（贡献者裸跑 pytest）：
# 私仓收口用的 gate_check.sh 发现没有 node 直接失败（批 DS2，NO-DATA 不等于通过）。
# ══════════════════════════════════════════════════════════════════════════

_PAGE = Path(__file__).resolve().parents[1] / "bladex_proxy" / "dashboard.html"

#: 最小假 DOM：元素按选择器建档（`$("#x")` 恒拿到同一个对象），innerHTML 是字符串，
#: 不解析——断言就对字符串做。fetch 由场景用 `respond()` 定响应，`hold` 的响应要
#: `release()` 才回，用来观察"请求在途"时的页面状态。
_HARNESS = r"""
"use strict";
const vm=require("vm"),fs=require("fs");
const [pagePath,scenPath]=process.argv.slice(2);
const html=fs.readFileSync(pagePath,"utf8");
const src=/<script>([\s\S]*?)<\/script>/.exec(html)[1];
const els=new Map();
class El{
  constructor(sel){this.sel=sel;this.innerHTML="";this.textContent="";this.value="";this.checked=false;
    this.disabled=false;this.hidden=false;this.style={};this.className="";this.title="";this.clientWidth=800;
    const c=new Set();this.classList={add:x=>c.add(x),remove:x=>c.delete(x),contains:x=>c.has(x),
      toggle:(x,on)=>{((on===undefined)?!c.has(x):on)?c.add(x):c.delete(x);}};}
  querySelector(s){return q(s);} querySelectorAll(){return [];} closest(){return this;}
  addEventListener(){} setAttribute(){} removeAttribute(){} insertAdjacentHTML(_p,h){this.innerHTML+=h;}
  appendChild(c){return c;} remove(){} scrollIntoView(){} focus(){} showModal(){} close(){} click(){}
  getContext(){return new Proxy({},{get:()=>()=>{}});} getBoundingClientRect(){return {left:0,top:0};}
}
function q(sel){if(!els.has(sel))els.set(sel,new El(sel));return els.get(sel);}
const body=new El("body");body.contains=()=>false;
const toasts=[];q("#toasts").appendChild=c=>{toasts.push({msg:c.textContent,type:c.className});return c;};
const fetchLog=[],routes=[],held=[];
function respond(method,re,status,body,opts){routes.unshift({method,re:new RegExp(re),status,body,hold:!!(opts&&opts.hold)});}
async function fetch(url,init){
  const method=(init&&init.method)||"GET";
  const entry={method,url,body:init&&init.body!==undefined?JSON.parse(init.body):null};
  fetchLog.push(entry);
  const r=routes.find(x=>x.method===method&&x.re.test(url));
  const status=r?r.status:200,b=r?(typeof r.body==="function"?r.body(entry):r.body):{};
  const resp={status,ok:status>=200&&status<300,json:async()=>JSON.parse(JSON.stringify(b??{}))};
  if(r&&r.hold)return new Promise(res=>held.push(()=>res(resp)));
  return resp;
}
function release(){while(held.length)held.shift()();}
async function tick(n){for(let i=0;i<(n||10);i++)await new Promise(r=>setImmediate(r));}
const store={};
const sb={document:{querySelector:q,querySelectorAll:()=>[],createElement:t=>new El(t),body,hidden:false,addEventListener(){}},
  fetch,console,setTimeout:()=>0,clearTimeout(){},setInterval:()=>0,clearInterval(){},requestAnimationFrame:()=>0,
  sessionStorage:{getItem:k=>(k in store?store[k]:null),setItem:(k,v)=>{store[k]=String(v);}},
  location:{hash:""},URLSearchParams,Intl,devicePixelRatio:1,addEventListener(){},
  respond,release,tick,fetchLog,toasts,el:q,out:{}};
sb.window=sb;
vm.createContext(sb);
vm.runInContext(src,sb,{filename:"dashboard.js"});
const scen=fs.readFileSync(scenPath,"utf8");
const prelude=`var asked=out.__asked=[],answers=[];
ask=async o=>{asked.push({title:o.title,body:o.body||"",fields:(o.fields||[]).map(f=>({name:f.name,type:f.type||"",value:f.value,options:f.options||null}))});
  return answers.length?answers.shift():null;};
await tick();fetchLog.length=0;toasts.length=0;`;
vm.runInContext("(async()=>{"+prelude+"\n"+scen+"\n})()",sb,{filename:"scenario.js"})
  .then(()=>{const asked=sb.out.__asked;delete sb.out.__asked;
    process.stdout.write(JSON.stringify({out:sb.out,fetchLog,toasts,asked})+"\n");})
  .catch(e=>{process.stderr.write(String(e&&e.stack||e)+"\n");process.exit(1);});
"""


def _run_page(scenario: str, tmp_path: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not on PATH: DS1 dashboard behaviour cases need a JS runtime")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    scen = tmp_path / "scenario.js"
    scen.write_text(scenario, encoding="utf-8")
    proc = subprocess.run([node, str(harness), str(_PAGE), str(scen)],
                           capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _js(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _posts(res: dict) -> list[dict]:
    return [f for f in res["fetchLog"] if f["method"] != "GET"]


# ── 账本组：夹具 ───────────────────────────────────────────────────────────

_LA, _LB = "ldg-ds1aaaa00001", "ldg-ds1bbbb00002"


def _seed_ledgers(*, closed_a: bool = False, rev_a: int = 11):
    from bladex_core.ledger import LedgerEntry, close_ledger, new_ledger

    a = new_ledger(ledger_id=_LA, title="DS1 任务 A", goal="把 A 做完", goal_source="model",
                   created_at="2026-10-29T01:00:00+00:00")
    a = a.model_copy(update={"rev": rev_a, "last_writer": "claude-code", "sections": {
        "core": [LedgerEntry(text="核心一", source="model", ref="t/1", writer="claude-code")],
        "open": [LedgerEntry(text="待办一", source="model", writer="codex")]}})
    if closed_a:
        a = close_ledger(a, closed_at="2026-10-29T02:00:00+00:00")
    b = new_ledger(ledger_id=_LB, title="DS1 任务 B", goal="接着做", goal_source="model",
                   created_at="2026-10-29T01:30:00+00:00")
    return {_LA: a, _LB: b}


class _LedgerApp:
    """一个真 Hub 的 app + 一个接在这个 Hub 上的 AgencyRuntime。"""

    def __init__(self, **seed_kw) -> None:
        from bladex_proxy.agency import AgencyRuntime

        self._cm = TestClient(create_app(_make_config()))
        self.client = self._cm.__enter__()
        self.hub = self.client.app.state.hub
        self.agency = AgencyRuntime(hub=self.hub)
        self.agency.pool.update(_seed_ledgers(**seed_kw))
        self.client.app.state.agency = self.agency

    def events(self) -> list[dict]:
        return [e.model_dump(mode="json", exclude={"ts"}) for _k, e in self.hub.scan_admin_events()]

    def send(self, method: str, url: str, body: dict | None):
        from urllib.parse import urlsplit
        parts = urlsplit(url)
        return self.client.request(method, parts.path + (f"?{parts.query}" if parts.query else ""),
                                   json=body)

    def close(self) -> None:
        self._cm.__exit__(None, None, None)


@pytest.fixture()
def ledger_apps():
    made: list[_LedgerApp] = []

    def make(**kw) -> _LedgerApp:
        made.append(_LedgerApp(**kw))
        return made[-1]

    yield make
    for a in made:
        a.close()


class _Stop(Exception):
    pass


def _capture_cli(monkeypatch, invoke) -> tuple[str, str, dict | None]:
    """CLI 实际发出的 (method, url, body)——同 test_ds1b_body_strict 的截获法。"""
    import bladex_proxy.cli as cli
    from bladex_proxy.admin_contract import ADMIN_CONTRACT_VERSION

    seen: dict = {}

    def fake_http_request(method, url, *, key=None, body=None, timeout=15.0, **_kw):  # noqa: ANN001, ANN003
        seen.update(method=method, url=url, body=body)
        raise _Stop

    monkeypatch.setattr(cli, "_http_request", fake_http_request)
    monkeypatch.setattr("bladex_proxy.cli.ledger_cmds._require_server_contract",
                        lambda what: ADMIN_CONTRACT_VERSION, raising=False)
    with pytest.raises(_Stop):
        invoke()
    return seen["method"], seen["url"], seen["body"]


def _ledger_rows(pool) -> list[dict]:
    return [{"ledger_id": lid, "title": led.title, "status": led.status, "active_in": [],
             "entry_counts": {}, "matter_id": "", "parent_ledger_id": "", "updated_at": ""}
            for lid, led in pool.items()]


def _ledger_detail_body(led) -> dict:
    return {"ledger": led.model_dump(mode="json"), "children": [], "active_in": [],
            "superseded_chain": [], "supersedes": []}


# ── DS-02：状态列 + 筛选 ─────────────────────────────────────────────────────

def test_ds02_ledger_list_has_status_column_and_filter(tmp_path) -> None:
    pool = _seed_ledgers(closed_a=True)
    res = _run_page(f"""
respond("GET","^/admin/ledgers$",200,{{ledgers:{_js(_ledger_rows(pool))},total:2}});
await viewLedgers();
out.all=el("#ltab tbody").innerHTML; out.count=el("#lcount").textContent;
el("#lstatus").value="closed"; renderLedgerRows(); out.closed=el("#ltab tbody").innerHTML;
el("#lstatus").value="active"; renderLedgerRows(); out.active=el("#ltab tbody").innerHTML;
""", tmp_path)
    o = res["out"]
    # 筛选不再发请求（/admin/status 是首屏取一次被拒直编，MQ-F16）
    assert sorted(f["url"] for f in res["fetchLog"]) == ["/admin/ledgers", "/admin/status"]
    assert '<span class="tag">closed</span>' in o["all"] and '<span class="tag ok">active</span>' in o["all"]
    assert o["count"] == "2 shown · 1 active / 1 closed"
    assert _LA in o["closed"] and _LB not in o["closed"]
    assert _LB in o["active"] and _LA not in o["active"]


# ── DS-03：取代链 + 关闭者 ───────────────────────────────────────────────────

def test_ds03_drawer_renders_supersede_chain_and_closer(tmp_path) -> None:
    from bladex_core.ledger import close_ledger

    pool = _seed_ledgers()
    a = close_ledger(pool[_LA], superseded_by=_LB, closed_at="2026-10-29T02:00:00+00:00",
                     closed_by_agent="hermes:default", closed_by_session="sess-9")
    body = _ledger_detail_body(a)
    body["superseded_chain"] = [{"ledger_id": _LB, "title": "DS1 任务 B", "status": "active"}]
    body["supersedes"] = ["ldg-older000003"]
    res = _run_page(f"""
respond("GET","^/admin/ledgers/{_LA}$",200,{_js(body)});
await ledgerDetail("{_LA}"); out.d=el("#drawer").innerHTML;
""", tmp_path)
    d = res["out"]["d"]
    assert "Superseded by" in d and f"ledgerDetail('{_LB}')" in d
    assert "Supersedes" in d and "ldg-older000003" in d
    assert "hermes:default" in d and "sess-9" in d and "2026-10-29T02:00:00+00:00" in d
    # 关了的：给 Reopen，不给 Close；取代过的不再给 Supersede
    assert 'data-act="reopen"' in d and 'data-act="close"' not in d and 'data-act="supersede"' not in d


def test_ds03_admin_close_renders_without_agent(tmp_path) -> None:
    pool = _seed_ledgers(closed_a=True)
    res = _run_page(f"""
respond("GET","^/admin/ledgers/{_LA}$",200,{_js(_ledger_detail_body(pool[_LA]))});
await ledgerDetail("{_LA}"); out.d=el("#drawer").innerHTML;
""", tmp_path)
    assert "by you / CLI (no agent session)" in res["out"]["d"]


# ── DS-01：关 / 重开 / 取代 —— 与 CLI 同端点、同 Hub 事件 ─────────────────────

def _lifecycle_scenario(op: str, pool, answer: dict) -> str:
    return f"""
respond("GET","^/admin/ledgers$",200,{{ledgers:{_js(_ledger_rows(pool))},total:2}});
respond("GET","^/admin/ledgers/",200,{_js(_ledger_detail_body(pool[_LA]))});
respond("POST","^/admin/ledgers/{_LA}/{op}$",200,{{status:"ok"}});
answers.push({_js(answer)});
const b=el("#btn"); await ledgerLifecycle("{op}","{_LA}",b);
out.disabledAfter=b.disabled;
"""


_CLI_LIFECYCLE = {
    "close": lambda lc: lc.ledger_close(_LA, reason="做完了"),
    "reopen": lambda lc: lc.ledger_reopen(_LA, reason="做完了"),
    "supersede": lambda lc: lc.ledger_supersede(_LA, by=_LB, reason="做完了"),
}


@pytest.mark.parametrize("op", ["close", "reopen", "supersede"])
def test_ds01_lifecycle_matches_cli_hub_event(op, tmp_path, ledger_apps, monkeypatch) -> None:
    from bladex_proxy.cli import ledger_cmds

    closed = op == "reopen"
    pool = _seed_ledgers(closed_a=closed)
    res = _run_page(_lifecycle_scenario(op, pool, {"reason": "做完了", "target": _LB}), tmp_path)
    posts = _posts(res)
    assert len(posts) == 1
    sent = posts[0]
    assert sent["url"] == f"/admin/ledgers/{_LA}/{op}"
    want = {"origin": "user", "reason": "做完了"}
    if op == "supersede":
        want["superseded_by"] = _LB
        sel = res["asked"][0]["fields"][0]
        assert sel["type"] == "select" and [o["value"] for o in sel["options"]] == [_LB]
    assert sent["body"] == want

    dash = ledger_apps(closed_a=closed)
    r = dash.send(sent["method"], sent["url"], sent["body"])
    assert r.status_code == 200, r.text
    cli_app = ledger_apps(closed_a=closed)
    method, url, body = _capture_cli(monkeypatch, lambda: _CLI_LIFECYCLE[op](ledger_cmds))
    assert url.endswith(sent["url"]) and method == sent["method"]
    assert cli_app.send(method, url, body).status_code == 200

    ev_dash, ev_cli = dash.events(), cli_app.events()
    assert len(ev_dash) == 1 and len(ev_cli) == 1
    # 逐字段相同；唯一的差别是 origin——它本来就该指认来路（原则 13 层 4）
    assert ev_dash[0]["payload"].pop("origin") == "user"
    assert ev_cli[0]["payload"].pop("origin") == "cli"
    assert ev_dash == ev_cli
    assert dash.agency.pool[_LA].status == cli_app.agency.pool[_LA].status
    assert dash.agency.pool[_LA].superseded_by == cli_app.agency.pool[_LA].superseded_by


# ── DS-04：编辑（批 DS2 起走 /edit：expected_rev 必带、只发改过的段、origin=dashboard）──

def _edit_answer() -> dict:
    # section_order 默认 goal/core/verified/open/next ⇒ sec_0=core, sec_1=verified, sec_2=open
    return {"title": "DS1 任务 A", "goal": "把 A 做完，并写测试",
            "sec_0": "核心一\n核心二", "sec_1": "", "sec_2": "待办一", "sec_3": ""}


def _edit_scenario(led, post_status: int = 200, post_body: dict | None = None) -> str:
    return f"""
respond("GET","^/admin/ledgers/",200,{_js(_ledger_detail_body(led))});
respond("POST","^/admin/ledgers/{led.ledger_id}/edit$",{post_status},{_js(post_body or {"status": "ok", "rev": led.rev + 1})});
answers.push({_js(_edit_answer())});
const b=el("#btn"); await ledgerEdit("{led.ledger_id}",b);
out.drawer=el("#drawer").innerHTML;
"""


def test_ds04_edit_sends_only_changed_sections_with_origin(tmp_path) -> None:
    pool = _seed_ledgers()
    res = _run_page(_edit_scenario(pool[_LA]), tmp_path)
    posts = _posts(res)
    assert len(posts) == 1 and posts[0]["url"] == f"/admin/ledgers/{_LA}/edit"
    body = posts[0]["body"]
    # 只有 goal 和 core 改了：title / open 不发，页面不算 goal_source
    assert set(body) == {"goal", "sections", "expected_rev", "origin"}
    assert body["expected_rev"] == 11 and body["origin"] == "dashboard"
    assert body["goal"] == "把 A 做完，并写测试"
    assert set(body["sections"]) == {"core"}
    core = body["sections"]["core"]
    assert core[0] == {"text": "核心一", "source": "model", "ref": "t/1", "writer": "claude-code"}
    assert core[1] == {"text": "核心二", "source": "user", "ref": "", "writer": ""}


def test_ds04_edit_event_is_distinguishable_from_flash_adoption(tmp_path, ledger_apps) -> None:
    """MQ-F18：dashboard 改的事件带 origin=dashboard_edit；flash 采纳走 /user-edit，没有 origin。"""
    from bladex_core.ledger import update_goal

    pool = _seed_ledgers()
    res = _run_page(_edit_scenario(pool[_LA]), tmp_path)
    sent = _posts(res)[0]
    dash = ledger_apps()
    r = dash.send(sent["method"], sent["url"], sent["body"])
    assert r.status_code == 200, r.text
    ev = dash.events()
    assert len(ev) == 1 and ev[0]["event_type"] == "ledger_user_edit"
    assert ev[0]["payload"]["origin"] == "dashboard_edit"
    led = dash.agency.pool[_LA]
    assert led.rev == 12 and led.last_writer == "user"
    # Goal 只经 update_goal（dashboard 是合法写者），不是页面自己算的
    ref = update_goal(pool[_LA], "把 A 做完，并写测试", actor="dashboard", updated_at=led.updated_at)
    for k in ("goal", "goal_source", "goal_revisions"):
        assert getattr(led, k) == getattr(ref, k), k
    # 没点名的段一字不动
    assert led.sections["open"] == pool[_LA].sections["open"]

    daemon = ledger_apps()
    assert daemon.send("POST", "/admin/ledgers/user-edit",
                       {"ledger": led.model_dump(mode="json")}).status_code == 200
    assert "origin" not in daemon.events()[0]["payload"]


def test_ds04_conflict_409_does_not_overwrite(tmp_path, ledger_apps) -> None:
    pool = _seed_ledgers()
    res = _run_page(_edit_scenario(pool[_LA]), tmp_path)
    sent = _posts(res)[0]
    app = ledger_apps(rev_a=12)                  # 打开编辑之后 agent 又写了一次
    before = app.agency.pool[_LA].model_dump(mode="json")
    r = app.send(sent["method"], sent["url"], sent["body"])
    assert r.status_code == 409
    assert app.agency.pool[_LA].model_dump(mode="json") == before and app.events() == []
    # 页面拿到这个 409：提示刷新、只发了一次（不自动重发）、草稿摆出来
    res = _run_page(_edit_scenario(pool[_LA], 409, r.json()), tmp_path)
    assert len(_posts(res)) == 1
    assert {"msg": "The ledger was updated meanwhile; refresh before editing", "type": "toast err"} in res["toasts"]
    assert "NOT saved" in res["out"]["drawer"] and "核心二" in res["out"]["drawer"]


def test_ds04_no_change_sends_nothing(tmp_path) -> None:
    pool = _seed_ledgers()
    led = pool[_LA]
    same = {"title": led.title, "goal": led.goal, "sec_0": "核心一", "sec_1": "", "sec_2": "待办一",
            "sec_3": ""}
    res = _run_page(f"""
respond("GET","^/admin/ledgers/",200,{_js(_ledger_detail_body(led))});
answers.push({_js(same)}); await ledgerEdit("{_LA}",el("#btn"));
""", tmp_path)
    assert _posts(res) == [] and res["toasts"][-1]["msg"] == "No changes"


# ── 通用要求：202 / 两种 400 / 加载状态 ─────────────────────────────────────

def test_write_202_shows_recorded_not_failure(tmp_path) -> None:
    pool = _seed_ledgers()
    res = _run_page(_lifecycle_scenario("close", pool, {"reason": ""}).replace(
        '"POST","^/admin/ledgers/' + _LA + '/close$",200', '"POST","^/admin/ledgers/' + _LA + '/close$",202'),
        tmp_path)
    assert {"msg": "Recorded; takes effect shortly", "type": "toast warn"} in res["toasts"]
    assert not any(t["type"] == "toast err" for t in res["toasts"])


def test_both_400_shapes_read_as_a_sentence(tmp_path, ledger_apps) -> None:
    """两种 400 都取自真服务端：严格 body 的 `detail`，与端点自报的 `error`。"""
    app = ledger_apps()
    detail_400 = app.send("POST", f"/admin/ledgers/{_LA}/close", {"origin": "user", "reson": "x"})
    error_400 = app.send("POST", f"/admin/ledgers/{_LA}/close", {"origin": "dashboard"})
    assert detail_400.status_code == 400 and isinstance(detail_400.json().get("detail"), str)
    assert error_400.status_code == 400 and "error" in error_400.json()
    pool = _seed_ledgers()
    for resp, needle in ((detail_400, "reson"), (error_400, "origin 'dashboard'")):
        scen = _lifecycle_scenario("close", pool, {"reason": ""}).replace(
            f'"POST","^/admin/ledgers/{_LA}/close$",200,{{status:"ok"}}',
            f'"POST","^/admin/ledgers/{_LA}/close$",400,{_js(resp.json())}')
        res = _run_page(scen, tmp_path)
        errs = [t["msg"] for t in res["toasts"] if t["type"] == "toast err"]
        assert len(errs) == 1 and needle in errs[0] and "HTTP 400" not in errs[0], errs
        assert res["out"]["disabledAfter"] is False        # 失败后按钮恢复


def test_ledger_write_shows_loading_and_blocks_second_click(tmp_path) -> None:
    pool = _seed_ledgers()
    res = _run_page(f"""
respond("GET","^/admin/ledgers$",200,{{ledgers:{_js(_ledger_rows(pool))},total:2}});
respond("GET","^/admin/ledgers/",200,{_js(_ledger_detail_body(pool[_LA]))});
respond("POST","^/admin/ledgers/{_LA}/close$",200,{{status:"closed"}},{{hold:true}});
const b=el("#btn"); b.textContent="Close";
answers.push({{reason:""}});
const p=ledgerLifecycle("close","{_LA}",b); await tick();
out.during={{disabled:b.disabled,text:b.textContent}};
answers.push({{reason:""}}); await ledgerLifecycle("close","{_LA}",b);   // 第二下：被挡
await ledgerEdit("{_LA}",el("#btn2"));                                   // 别的账本写：也被挡
out.postsDuring=fetchLog.filter(f=>f.method==="POST").length;
release(); await p; await tick();
out.after={{disabled:b.disabled,text:b.textContent}};
""", tmp_path)
    o = res["out"]
    assert o["during"] == {"disabled": True, "text": "Closing…"}
    assert o["postsDuring"] == 1
    assert o["after"] == {"disabled": False, "text": "Close"}
    assert any("still being saved" in t["msg"] for t in res["toasts"])


# ── DS-08：回放抽屉的账本 / 拼接 / 工具 / 耗时 ───────────────────────────────

def test_ds08_turn_drawer_renders_ledger_tab(tmp_path) -> None:
    turn = {"identity": {"user_id": "u", "agent_id": "a", "session_id": "s"},
            "ts": "2026-10-29T01:00:00Z", "decision_meta": {}, "response_meta": {},
            "ledger_face": True,
            "ledger_anchor": {"ledger_id": _LA, "rev": 11, "sections": {"core": 120, "open": 40},
                              "about_chars": 30, "roster_chars": 0, "added_hash": "abcd1234"},
            "splice_records": [{"session_prefix": "u/a/s/", "anchor": "anc-1", "calls": [{}],
                                "results": [], "created_ms": 1, "age_turns": 2}],
            "tool_events": [{"tool_name": "bladex_ledger_update", "direction": "call",
                             "tool_call_id": "call_9"}],
            "ms_identity": 1.25, "ms_inject": 12.0, "ms_total": 980.5}
    bare = {**turn, "ledger_face": None, "ledger_anchor": None, "splice_records": [], "tool_events": []}
    res = _run_page(f"""
respond("GET","^/admin/turns/u/a/s/000001$",200,{_js(turn)});
respond("GET","^/admin/turns/u/a/s/000002$",200,{_js(bare)});
await turnDetail("u/a/s/000001"); out.full=el("#drawer").innerHTML;
await turnDetail("u/a/s/000002"); out.bare=el("#drawer").innerHTML;
""", tmp_path)
    full, bare_html = res["out"]["full"], res["out"]["bare"]
    assert 'id="tt-led"' in full and f"ledgerDetail('{_LA}')" in full
    assert "core 120" in full and "abcd1234" in full and "anc-1" in full
    assert "bladex_ledger_update" in full and "call_9" in full
    assert "1.3 ms" in full and "12.0 ms" in full and "980.5 ms" in full
    assert '<span class="tag ok">yes</span>' in full
    assert bare_html.count("n/a (not recorded for this turn)") == 2   # face + anchor 缺数报缺数


# ── DS-09：Matters 列表的账本锚列 ───────────────────────────────────────────

def test_ds09_matters_list_has_ledger_anchor_column(tmp_path) -> None:
    rows = [{"matter_id": "m-1", "title": "锚出来的", "status": "active", "origin": "auto",
             "topic_keys": {}, "updated_at": "", "ledger_id": _LA},
            {"matter_id": "m-2", "title": "自动的", "status": "active", "origin": "auto",
             "topic_keys": {}, "updated_at": "", "ledger_id": ""}]
    res = _run_page(f"""
respond("GET","^/admin/matters\\\\?",200,{{matters:{_js(rows)},total:2}});
await viewMatters(); out.t=el("#mtab tbody").innerHTML;
""", tmp_path)
    t = res["out"]["t"]
    assert f"event.stopPropagation();ledgerDetail('{_LA}')" in t
    assert t.count("ledgerDetail(") == 1


# ── 清理组 ─────────────────────────────────────────────────────────────────

def test_ds23_item_kind_options_come_from_the_core_enum(tmp_path) -> None:
    """闭集出处：`bladex_core.fact.ItemKind`。页面那份列表与它逐项相同、同序。"""
    from bladex_core.fact import ItemKind

    res = _run_page("""
respond("GET","^/admin/facts\\\\?",200,{facts:[],total:0,mode:"list"});
out.kinds=ITEM_KINDS; await viewFacts(); out.main=el("#main").innerHTML;
el("#fitem").value="lesson"; fOffset=0; await loadFacts();
""", tmp_path)
    assert res["out"]["kinds"] == [k.value for k in ItemKind]
    main = res["out"]["main"]
    assert 'id="fitem"' in main and 'id="fkind"' not in main
    for k in ItemKind:
        assert f'<option value="{k.value}">' in main
    urls = [f["url"] for f in res["fetchLog"]]
    assert urls[-1].startswith("/admin/facts?") and "item_kind=lesson" in urls[-1]
    assert not any("&kind=" in u or "?kind=" in u for u in urls)


def test_ds23_item_kind_filter_is_honoured_by_the_server() -> None:
    """页面发的 `item_kind=` 是服务端真认的参数名（拼错只会被静默忽略——GET 不严格）。"""
    from bladex_core.fact import Fact, ItemKind

    app = create_app(_make_config())
    with TestClient(app) as client:
        idx = app.state.index
        fake = [Fact(id="f-a", content="a", item_kind=ItemKind.LESSON),
                Fact(id="f-b", content="b", item_kind=ItemKind.ASSERTION)]
        idx.all_facts = lambda: list(fake)
        d = client.get("/admin/facts", params={"item_kind": "lesson"}).json()
        assert [f["id"] for f in d["facts"]] == ["f-a"]


def test_ds12_config_says_hot_reload_not_restart(tmp_path) -> None:
    res = _run_page("""
respond("GET","^/admin/hard_rules$",200,{hard_rules:["MUST x"],count:1});
respond("GET","^/admin/status$",200,{memory_config:{effective:{BLADEX_HARD_RULES_RELOAD_S:5},flags:{},drift:[]}});
await viewConfig(); out.on=el("#main").innerHTML;
respond("GET","^/admin/status$",200,{memory_config:{effective:{BLADEX_HARD_RULES_RELOAD_S:0},flags:{},drift:[]}});
await viewConfig(); out.off=el("#main").innerHTML;
""", tmp_path)
    on, off = res["out"]["on"], res["out"]["off"]
    assert "within 5 s" in on and "Obsidian" in on and "later write wins" in on
    assert "restart the proxy" not in on.split("Effective memory configuration")[0]
    assert "Hot reload is off" in off and "restart the proxy" in off


def test_ds11_profiles_drop_deleted_concepts(tmp_path) -> None:
    md = "# USER\n\n- agents:\n  - codex（12 turns，最近 ?）\n  - hermes（3 turns，最近 2026-10-28）\n"
    res = _run_page(f"""
respond("GET","^/admin/profiles$",200,{{user_md:{_js(md)},agent_count:1,
  agents:[{{agent_base:"codex",agent_md:"note",rule_files:[],tools:[]}}]}});
await viewProfiles(); out.main=el("#main").innerHTML; out.sub=SUBS.profiles;
profileDetail(0); out.drawer=el("#drawer").innerHTML;
""", tmp_path)
    page = res["out"]["main"] + res["out"]["drawer"] + res["out"]["sub"]
    for gone in ("recall reference frame", "Habit doc", "habit doc", "USER.md, per-agent"):
        assert gone not in page, gone
    assert "最近 ?" not in res["out"]["main"]
    assert "codex（12 turns）" in res["out"]["main"]
    assert "hermes（3 turns，最近 2026-10-28）" in res["out"]["main"]   # 真值照显
    assert "not injected" in res["out"]["main"] and "/admin/behavior" in res["out"]["main"]
    assert "has been left out" in res["out"]["main"]                   # 删了什么写明


# ── 小接线组 ───────────────────────────────────────────────────────────────

def test_ds15_long_session_pages_with_offset_and_shows_total(tmp_path) -> None:
    res = _run_page("""
respond("GET","^/admin/turns\\\\?",200,e=>{const o=+new URLSearchParams(e.url.split("?")[1]).get("offset");
  const n=Math.min(100,170-o);return {turns:Array.from({length:n},(_,i)=>({key:"u/a/s/"+String(o+i).padStart(6,"0"),ts:""})),total:170,limit:100,offset:o};});
await loadTurns("u/a/s/"); out.p1=el("#sdetail").innerHTML;
tPage(1); await tick(); out.p2=el("#sdetail").innerHTML;
tPage(-1); await tick();
""", tmp_path)
    urls = [f["url"] for f in res["fetchLog"]]
    offsets = [u.split("offset=")[1] for u in urls]
    assert offsets == ["0", "100", "0"] and all("prefix=u%2Fa%2Fs%2F" in u and "limit=100" in u for u in urls)
    assert "(170 total)" in res["out"]["p1"] and "1-100 of 170" in res["out"]["p1"]
    assert "101-170 of 170" in res["out"]["p2"] and "000169" in res["out"]["p2"]


def test_ds15_server_honours_the_paging_params() -> None:
    from datetime import datetime, timedelta

    app = create_app(_make_config())
    with TestClient(app) as client:
        t0 = datetime.fromisoformat("2026-10-01T00:00:00+00:00")
        for i in range(130):
            app.state.hub.put(f"u1/agentA/s1/{i:06d}",
                              Turn(identity=Identity(user_id="u1", agent_id="agentA", session_id="s1"),
                                   ts=t0 + timedelta(seconds=i)))
        d = client.get("/admin/turns?" + "prefix=u1%2FagentA%2Fs1%2F&limit=100&offset=100").json()
        assert d["total"] == 130 and len(d["turns"]) == 30


def test_ds18_item_kind_tag_filters(tmp_path) -> None:
    res = _run_page("""
respond("GET","^/admin/facts\\\\?",200,{facts:[{id:"f1",content:"c",item_kind:"lesson",tags:"",trust:1}],total:1,mode:"list"});
await viewFacts(); out.row=el("#ftab tbody").innerHTML;
filterItemKind("procedure"); await tick(); filterItemKind("bogus"); await tick();
""", tmp_path)
    assert "filterItemKind('lesson')" in res["out"]["row"]
    urls = [f["url"] for f in res["fetchLog"]]
    assert "item_kind=procedure" in urls[-1] and len(urls) == 2      # 闭集外的值不发请求


def test_ds19_fact_drawer_renders_lineage_fields(tmp_path) -> None:
    fact = {"id": "f-1", "content": "x", "item_kind": "assertion", "trust": 1, "tags": "",
            "audience": "agent:codex", "matter_id": "m-7", "superseded_by": "f-2",
            "source_ledger_key": "u/a/s/000003", "corrects": "earlier I said y", "importance": 1.75}
    res = _run_page(f"""
respond("GET","^/admin/facts/f-1$",200,{_js(fact)});
await factDetail("f-1"); out.d=el("#drawer").innerHTML;
""", tmp_path)
    d = res["out"]["d"]
    assert "agent:codex" in d and "1.75" in d and "earlier I said y" in d
    assert "matterDetail('m-7')" in d and "factDetail('f-2')" in d and "turnDetail('u/a/s/000003')" in d


def test_ds20_overview_has_flash_card_and_ledger_counts(tmp_path) -> None:
    pool = _seed_ledgers(closed_a=True)
    flash = {"available": True, "flash_lag_s": 4, "lag_bound_s": 300, "lag_exceeded": False,
             "computed_at": "2026-10-29T01:00:00Z", "turns_folded": 3, "turns_total": 19366,
             "ledgers_projected": 52}
    res = _run_page(f"""
respond("GET","^/admin/ledgers$",200,{{enabled:true,ledgers:{_js(_ledger_rows(pool))},total:2}});
respond("GET","^/admin/status$",200,{{flash:{_js(flash)}}});
await viewOverview(); out.on=el("#main").innerHTML;
respond("GET","^/admin/status$",200,{{flash:{{available:false,ledgers_projected:0}}}});
await viewOverview(); out.off=el("#main").innerHTML;
""", tmp_path)
    on, off = res["out"]["on"], res["out"]["off"]
    assert "Memory Flash" in on and "4 s" in on and "within bound" in on and "19,366" in on and "52" in on
    assert "Active / closed" in on and "1 / 1" in on
    assert "no reading" in off and "within bound" not in off


def test_ds20_status_really_carries_flash() -> None:
    app = create_app(_make_config())
    with TestClient(app) as client:
        assert "available" in client.get("/admin/status").json()["flash"]


def _graph_rows(n: int) -> list[dict]:
    return [{"matter_id": f"m-{i}", "title": f"M{i}", "status": "active", "origin": "auto"}
            for i in range(n)]


_GRAPH_RESPOND = """
respond("GET","^/admin/matters\\\\?",200,e=>{const p=new URLSearchParams(e.url.split("?")[1]);
  const off=+p.get("offset")||0,lim=+p.get("limit")||100,rows=ALL.slice(off,off+lim),members={};
  if(p.get("include")==="members")for(const m of rows)members[m.matter_id]={edges:[{target_type:"fact",target_key:"f-"+m.matter_id,provenance:"auto"}],facts:[]};
  return {matters:rows,total:ALL.length,members};});
respond("GET","^/admin/matters/[^?]+",200,e=>({edges:[{target_type:"fact",target_key:"f-x",provenance:"auto"}],facts:[]}));
"""


@pytest.mark.parametrize("n,want_requests", [(329, 1), (1200, 3), (1600, 3)])
def test_ds26_graph_first_screen_is_at_most_three_requests(n, want_requests, tmp_path) -> None:
    res = _run_page(f"var ALL={_js(_graph_rows(n))};" + _GRAPH_RESPOND + """
tab="graph"; await viewGraph(); out.prog=el("#gprog").textContent;
""", tmp_path)
    urls = [f["url"] for f in res["fetchLog"]]
    assert len(urls) == want_requests <= 3, urls
    assert all("include=members" in u and "limit=500" in u for u in urls)
    drawn = min(n, 1500)
    assert res["out"]["prog"].startswith(f"{drawn}" + (f" of {n}" if n > 1500 else "") + " Matters")
    assert f"{drawn * 2} nodes" in res["out"]["prog"]                 # 每个 Matter + 它那条边的 fact


def test_ds26_graph_url_is_accepted_by_the_server(tmp_path) -> None:
    res = _run_page("var ALL=[];" + _GRAPH_RESPOND + 'tab="graph"; await viewGraph();', tmp_path)
    url = res["fetchLog"][0]["url"]
    app = create_app(_make_config())
    with TestClient(app) as client:
        r = client.get(url)
        assert r.status_code == 200 and "members" in r.json(), r.text


def test_slow_views_show_loading_and_do_not_stack_requests(tmp_path) -> None:
    """MQ-I42：Graph 首屏 / Merge candidates / Agents —— 在途时有加载标记、按钮不可点、再进不叠请求。"""
    res = _run_page("var ALL=[{matter_id:'m-1',title:'a',status:'active',origin:'auto'}];" + _GRAPH_RESPOND + """
respond("GET","^/admin/matters\\\\?include",200,{matters:ALL,total:1,members:{}},{hold:true});
tab="graph"; const g=viewGraph(); await tick();
out.g={loading:el("#main").innerHTML.includes('data-loading="1"'),
       reloadDisabled:/id="greload"[^>]*disabled/.test(el("#main").innerHTML)};
viewGraph(); await tick(); out.gReq=fetchLog.length;
release(); await g; await tick();
out.gAfter={loading:el("#main").innerHTML.includes('data-loading="1"'),
            reloadDisabled:/id="greload"[^>]*disabled/.test(el("#main").innerHTML)};

fetchLog.length=0;
respond("GET","^/admin/matters/merge_candidates$",200,{candidates:[]},{hold:true});
const b=el("#mcbtn"); b.textContent="Merge candidates";
const mc=mergeCandidates(b); await tick();
out.mc={disabled:b.disabled,text:b.textContent,loading:el("#drawer").innerHTML.includes('data-loading="1"')};
mergeCandidates(b); mergeCandidates(); await tick(); out.mcReq=fetchLog.length;
release(); await mc; await tick();
out.mcAfter={disabled:b.disabled,text:b.textContent,loading:el("#drawer").innerHTML.includes('data-loading="1"')};

fetchLog.length=0;
respond("GET","^/admin/agents$",200,{unknown_buckets:[],known_agents:[],claims:[]},{hold:true});
tab="agents"; const a=viewAgents(); await tick();
out.ag={loading:el("#main").innerHTML.includes('data-loading="1"'),
        refreshDisabled:/id="arefresh"[^>]*disabled/.test(el("#main").innerHTML)};
viewAgents(); await tick(); out.agReq=fetchLog.length;
release(); await a; await tick();
out.agAfter={loading:el("#main").innerHTML.includes('data-loading="1"'),
             refreshDisabled:/id="arefresh"[^>]*disabled/.test(el("#main").innerHTML)};
""", tmp_path)
    o = res["out"]
    assert o["g"] == {"loading": True, "reloadDisabled": True} and o["gReq"] == 1
    assert o["gAfter"] == {"loading": False, "reloadDisabled": False}
    assert o["mc"] == {"disabled": True, "text": "Finding candidates…", "loading": True}
    assert o["mcReq"] == 1
    assert o["mcAfter"] == {"disabled": False, "text": "Merge candidates", "loading": False}
    assert o["ag"] == {"loading": True, "refreshDisabled": True} and o["agReq"] == 1
    assert o["agAfter"] == {"loading": False, "refreshDisabled": False}


def test_slow_view_result_does_not_overwrite_another_tab(tmp_path) -> None:
    res = _run_page("""
respond("GET","^/admin/agents$",200,{unknown_buckets:[],known_agents:[],claims:[]},{hold:true});
tab="agents"; const a=viewAgents(); await tick();
tab="ledgers"; el("#main").innerHTML="LEDGERS PAGE";
release(); await a; out.main=el("#main").innerHTML;
""", tmp_path)
    assert res["out"]["main"] == "LEDGERS PAGE"


# ══════════════════════════════════════════════════════════════════════════
# 批 DS2 · DS1 尾巴（MQ-I42）：认领 / 改名的页面级加载提示。
# 预取两次全扫约 10 s，按钮传不进函数（签名由 test_admin_agents_drift_ui 钉死），
# 所以在函数内部点亮 #page-loading。对话框打开时必须已经熄灭：提示不能盖着用户的输入。
# ══════════════════════════════════════════════════════════════════════════

_LOADING_SPY = """
const loading=()=>{const e=el("#page-loading");return {on:!e.hidden,text:e.textContent};};
const realAsk=ask; out.atAsk=[];
ask=async o=>{out.atAsk.push(loading().on);return realAsk(o);};
"""


def _agents_respond(*, hold: str = "", detail_status: int = 200, agents_status: int = 200) -> str:
    def opt(on: bool) -> str:
        return ",{hold:true}" if on else ""
    return f"""
respond("GET","^/admin/agents/[^/]+/detail$",{detail_status},
  {{suggested_rule:{{}},detail:"detail boom"}}{opt(hold == "get")});
respond("GET","^/admin/agents$",{agents_status},
  {{known_agents:[{{agent_id:"codex"}}],unknown_buckets:[],claims:[],detail:"agents boom"}}{opt(hold == "get")});
"""


def test_page_loading_element_starts_hidden() -> None:
    assert '<div id="page-loading" role="status" aria-live="polite" hidden></div>' in _PAGE.read_text("utf-8")
    assert "#page-loading[hidden]{display:none}" in _PAGE.read_text("utf-8")


def test_claim_prefetch_shows_loading_until_the_dialog_opens(tmp_path) -> None:
    res = _run_page(_LOADING_SPY + _agents_respond(hold="get") + """
const p=agentClaim("bucket-1"); await tick();
out.first=loading(); out.firstReq=fetchLog.length;
agentClaim("bucket-1"); agentRename("bucket-1"); await tick();
out.afterReclick=fetchLog.length;
release(); await tick();
out.second=loading(); out.secondReq=fetchLog.length;
release(); await p; await tick();
out.after=loading();
""", tmp_path)
    o = res["out"]
    assert o["first"]["on"] is True and "bucket-1" in o["first"]["text"]
    assert o["firstReq"] == 1 and o["afterReclick"] == 1      # 在途时再点不叠请求
    assert o["second"]["on"] is True and o["secondReq"] == 2  # 两次预取之间不闪灭
    assert o["atAsk"] == [False]                              # 对话框打开前已熄灭
    assert o["after"] == {"on": False, "text": ""}
    assert res["asked"][0]["title"] == "Claim bucket-1"
    assert res["toasts"] == []


def test_rename_prefetch_shows_loading_then_clears(tmp_path) -> None:
    res = _run_page(_LOADING_SPY + _agents_respond(hold="get") + """
const p=agentRename("codex"); await tick();
out.during=loading();
release(); await p; await tick();
out.after=loading();
""", tmp_path)
    o = res["out"]
    assert o["during"]["on"] is True and "codex" in o["during"]["text"]
    assert o["atAsk"] == [False] and o["after"]["on"] is False
    assert res["asked"][0]["title"] == "Rename codex"


def test_claim_submit_shows_loading_while_posting(tmp_path) -> None:
    res = _run_page(_LOADING_SPY + _agents_respond() + """
respond("POST","^/admin/agents/claim$",200,{status:"ok"},{hold:true});
answers.push({to:"renamed"});
const p=agentRename("codex"); await tick(30);
out.during=loading(); out.posts=fetchLog.filter(f=>f.method==="POST").length;
release(); await p; await tick();
out.after=loading();
""", tmp_path)
    o = res["out"]
    assert o["posts"] == 1
    assert o["during"]["on"] is True and "renamed" in o["during"]["text"]
    assert o["after"]["on"] is False
    assert any(t["msg"] == "Claimed" for t in res["toasts"])


@pytest.mark.parametrize(("kw", "needle"), [
    ({"detail_status": 500}, "detail boom"),
    ({"agents_status": 500}, "agents boom"),
])
def test_claim_prefetch_failure_shows_error_and_still_opens_dialog(tmp_path, kw, needle) -> None:
    res = _run_page(_LOADING_SPY + _agents_respond(**kw) + """
await agentClaim("bucket-1"); await tick();
out.after=loading();
""", tmp_path)
    assert res["out"]["after"]["on"] is False
    errs = [t for t in res["toasts"] if "err" in t["type"]]
    assert len(errs) == 1 and needle in errs[0]["msg"]
    assert res["out"]["atAsk"] == [False]        # 取不到不挡操作
    assert res["asked"][0]["title"] == "Claim bucket-1"


def test_claim_submit_failure_clears_loading_and_shows_error(tmp_path) -> None:
    res = _run_page(_LOADING_SPY + _agents_respond() + """
respond("POST","^/admin/agents/claim$",400,{detail:"claim boom"});
answers.push({to:"renamed"});
await agentRename("codex"); await tick();
out.after=loading();
""", tmp_path)
    assert res["out"]["after"]["on"] is False
    assert [t["msg"] for t in res["toasts"] if "err" in t["type"]] == ["claim boom"]


def test_prefetch_guard_resets_after_failure(tmp_path) -> None:
    """预取全失败后，下一次点击不能被在途标记挡住。"""
    res = _run_page(_LOADING_SPY + _agents_respond(detail_status=500, agents_status=500) + """
await agentClaim("bucket-1"); await tick();
out.first=fetchLog.length;
await agentClaim("bucket-1"); await tick();
out.second=fetchLog.length;
""", tmp_path)
    assert res["out"]["first"] == 2 and res["out"]["second"] == 4


# ══════════════════════════════════════════════════════════════════════════
# 批 DS2 第二步：用量 / 项目 / 交接 + 与 CLI 对账。
# 对账做法：node 跑页面、截获请求打真服务端；CLI 真跑（CliRunner），`_admin_call` 接到同一个真服务端。
# ══════════════════════════════════════════════════════════════════════════

def _route_cli(monkeypatch, client, module) -> list[tuple[str, str, dict | None, str]]:
    """CLI 的 `_admin_call` → 同一个 TestClient（URL 拼法与真 `_admin_call` 相同）。"""
    from urllib.parse import urlencode

    calls: list[tuple[str, str, dict | None, str]] = []

    def fake(method, path, body=None, params=None, timeout=15.0):  # noqa: ANN001
        url = path
        if params:
            url += "?" + urlencode({k: v for k, v in params.items() if v not in ("", None)})
        r = client.request(method, url, json=body)
        calls.append((method, url, body, r.text))
        return r.status_code, r.json()

    monkeypatch.setattr(module, "_admin_call", fake)
    return calls


def _cli(*argv: str):
    from bladex_proxy.cli import app as cli_app
    from typer.testing import CliRunner

    return CliRunner().invoke(cli_app, list(argv))


def _query(url: str) -> dict:
    from urllib.parse import parse_qs, urlsplit

    return parse_qs(urlsplit(url).query)


def _cells(row_html: str) -> list[str]:
    import html
    import re

    return [html.unescape(re.sub(r"<[^>]+>", "", c)).strip()
            for c in re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.S)]


# ── 用量 ─────────────────────────────────────────────────────────────────────

_USAGE_BASE = "2026-10-20T01:00:00+00:00"


def _seed_usage(hub) -> None:  # noqa: ANN001
    from datetime import datetime, timedelta

    from bladex_proxy.models import ResponseMeta

    base = datetime.fromisoformat(_USAGE_BASE)
    for i in range(6):
        agent = ("claude-code", "codex")[i % 2]
        # 第 0 轮没有 usage：input / output / cache_read 都是「没有读数」
        usage = {} if i == 0 else {"prompt": 1000 + i, "completion": 5, "total": 1005 + i}
        hub.put(f"local/{agent}/s{i % 2}/{i:06d}", Turn(
            identity=Identity(user_id="local", agent_id=agent, session_id=f"s{i % 2}"),
            model="openai/glm-5.3", ts=base + timedelta(hours=7 * i),
            response_meta=ResponseMeta(usage=usage)))


_USAGE_WINDOW = {"since": "2026-10-20T00:00:00+00:00", "until": "2026-10-21T00:00:00+00:00",
                 "tz": "+08:00"}


def test_usage_page_and_cli_match_for_the_same_window(tmp_path, monkeypatch) -> None:
    from bladex_proxy.cli import ops_cmds

    app = create_app(_make_config())
    with TestClient(app) as client:
        _seed_usage(app.state.hub)
        res = _run_page(f"""
tab="usage";
el("#uwin").value="custom"; el("#usince").value={_js(_USAGE_WINDOW["since"])};
el("#uuntil").value={_js(_USAGE_WINDOW["until"])}; el("#utz").value={_js(_USAGE_WINDOW["tz"])};
await loadUsage(el("#uapply"));
""", tmp_path)
        page_url = res["fetchLog"][-1]["url"]
        assert page_url.startswith("/admin/usage?")
        page_resp = client.get(page_url)
        assert page_resp.status_code == 200

        calls = _route_cli(monkeypatch, client, ops_cmds)
        out = _cli("usage", "--json", "--since", _USAGE_WINDOW["since"],
                   "--until", _USAGE_WINDOW["until"], "--tz", _USAGE_WINDOW["tz"])
        assert out.exit_code == 0, out.output
        assert _query(calls[0][1]) == _query(page_url)
        assert calls[0][3] == page_resp.text      # 同窗口：服务端响应逐字相等
        printed = [ln for ln in out.stdout.splitlines() if ln.startswith('{"window"')]
        data = page_resp.json()
        assert printed == [json.dumps(data, ensure_ascii=False)]
        assert data["totals"]["all"]["turns"] == 4

        # 页面把这份响应画出来：窗口行、每一行合计与 CLI 表格逐字相同
        res = _run_page(f"""
tab="usage";
respond("GET","^/admin/usage",200,{_js(data)});
await loadUsage(el("#uapply"));
out.window=el("#uresult").innerHTML; out.rows=el("#ugroups-body").innerHTML;
""", tmp_path)
        html = res["out"]["window"]
        lines = ops_cmds._usage_lines(data)
        import html as _html
        assert lines[0] in _html.unescape(html)
        for lane in data["totals"]:
            cli_line = next(ln for ln in lines if ln.startswith(f"  {lane} "))
            import re
            row = re.search(rf"<tr><td>{lane}</td>.*?</tr>", html, re.S).group(0)
            assert _cells(row) == re.split(r"\s{2,}", cli_line.strip())
        assert lines[-1] in _html.unescape(html)   # 底部说明逐字相同


_USAGE_NODATA = {
    "window": {"since": None, "until": None, "tz": "+08:00"},
    "filters": {"agent": "", "model": "", "session": ""},
    "groups": [],
    "totals": {"all": {"turns": 3, "input": 0, "output": 0, "cache_read": 1234,
                       "input_nodata": 3, "output_nodata": 0, "cache_read_nodata": 1}},
}


def test_usage_no_reading_is_not_shown_as_zero(tmp_path) -> None:
    from bladex_proxy.cli.ops_cmds import _usage_lines

    res = _run_page(f"""
tab="usage";
respond("GET","^/admin/usage",200,{_js(_USAGE_NODATA)});
await loadUsage(el("#uapply"));
out.html=el("#uresult").innerHTML;
""", tmp_path)
    cells = _cells("<td>all</td>" + res["out"]["html"].split("<tr><td>all</td>")[1].split("</tr>")[0])
    assert cells[2] == "n/a (+3 n/d)"       # 全部没有读数：不是 0
    assert cells[3] == "0"                  # 真的是 0
    assert cells[4] == "1,234 (+1 n/d)"     # 部分没有读数：和 + 缺几轮
    cli = next(ln for ln in _usage_lines(_USAGE_NODATA) if ln.startswith("  all "))
    assert "n/a (+3 n/d)" in cli and "1,234 (+1 n/d)" in cli


def test_usage_is_on_demand_and_handles_empty_error_loading(tmp_path) -> None:
    res = _run_page("""
await viewOverview(); await tick();
out.overviewUsage=fetchLog.filter(f=>f.url.includes("/admin/usage")).length;
fetchLog.length=0;

tab="usage";
respond("GET","^/admin/usage",200,{window:{tz:"+08:00"},filters:{},groups:[],
  totals:{all:{turns:0,input:0,output:0,cache_read:0}}},{hold:true});
const p=viewUsage(); await tick();
out.loading=el("#uresult").innerHTML.includes('data-loading="1"');
out.req=fetchLog.map(f=>f.url);
release(); await p; await tick();
out.empty=el("#ugroups-body").innerHTML; out.loadingAfter=el("#uresult").innerHTML.includes('data-loading="1"');

respond("GET","^/admin/usage",400,{detail:"days must be > 0"});
await loadUsage(el("#uapply")); out.err=el("#uresult").innerHTML;

fetchLog.length=0;
el("#uwin").value="custom"; el("#usince").value=""; el("#uuntil").value="";
await loadUsage(el("#uapply")); out.customReq=fetchLog.length;
""", tmp_path)
    o = res["out"]
    assert o["overviewUsage"] == 0                      # 不进 Overview 轮询
    assert o["loading"] is True and o["loadingAfter"] is False
    assert o["req"] == ["/admin/usage?days=7"]          # 默认 7 天，与 CLI 相同
    assert "No turns in this window" in o["empty"]
    assert "days must be &gt; 0" in o["err"]
    assert o["customReq"] == 0                          # 自定义窗口没填：不发请求
    assert any("custom window" in t["msg"] for t in res["toasts"])


def test_usage_result_does_not_overwrite_another_tab_and_pages_on_client(tmp_path) -> None:
    rows = [{"day": "2026-10-20", "agent": "a", "model": "m", "lane": "main", "session": f"s{i}",
             "turns": 1, "input": 1, "output": 1, "cache_read": 1} for i in range(230)]
    body = {"window": {"tz": "+08:00"}, "filters": {}, "groups": rows,
            "totals": {"all": {"turns": 230, "input": 230, "output": 230, "cache_read": 230}}}
    res = _run_page(f"""
tab="usage";
respond("GET","^/admin/usage",200,{_js(body)},{{hold:true}});
const p=loadUsage(el("#uapply")); await tick();
tab="facts"; const before=el("#uresult").innerHTML;
release(); await p; await tick();
out.kept=el("#uresult").innerHTML===before;

tab="usage";
respond("GET","^/admin/usage",200,{_js(body)});
fetchLog.length=0;
await loadUsage(el("#uapply"));
out.first=el("#ugroups-body").innerHTML.split("<tr>").length-1;
uPage(1); uPage(1);
out.last=el("#ugroups-body").innerHTML.split("<tr>").length-1;
out.reqs=fetchLog.length;
""", tmp_path)
    o = res["out"]
    assert o["kept"] is True
    assert o["first"] == 100 and o["last"] == 30 and o["reqs"] == 1   # 翻页不再发请求


# ── 项目（DS-14）─────────────────────────────────────────────────────────────

def test_projects_page_and_cli_match(tmp_path, monkeypatch) -> None:
    from bladex_core.project import Project
    from bladex_proxy.cli import ops_cmds
    from bladex_proxy.storage.memory_index import MemoryIndex

    index = MemoryIndex(f"{tempfile.mkdtemp()}/p2w", embedder=None, read_only=False)
    index.upsert_project(Project(project_id="p-aaaa", name="BladeX", root_path="/r/BladeX",
                                 agent_ids=["codex"]))
    index.upsert_project(Project(project_id="p-bbbb", name="", root_path=""))
    app = create_app(_make_config())
    with TestClient(app) as client:
        app.state.index = index
        res = _run_page("""
await viewProjects();
""", tmp_path)
        page_url = res["fetchLog"][-1]["url"]
        page_resp = client.get(page_url)
        calls = _route_cli(monkeypatch, client, ops_cmds)
        q = _query(page_url)
        out = _cli("project", "list", "--json", "--limit", q["limit"][0], "--offset", q["offset"][0])
        assert out.exit_code == 0, out.output
        assert _query(calls[0][1]) == q and calls[0][3] == page_resp.text
        printed = [ln for ln in out.stdout.splitlines() if ln.startswith('{"projects"')]
        assert printed == [json.dumps(page_resp.json(), ensure_ascii=False)]

        res = _run_page(f"""
respond("GET","^/admin/projects",200,{_js(page_resp.json())});
await viewProjects(); out.rows=el("#ptab tbody").innerHTML;
""", tmp_path)
        rows = res["out"]["rows"]
        assert "p-aaaa" in rows and "/r/BladeX" in rows and "p-bbbb" in rows
        assert 'n/a' in rows       # 没有 root 的不留空白
    index.close()


def test_projects_page_pages_with_total_and_empty_text(tmp_path) -> None:
    res = _run_page("""
respond("GET","^/admin/projects",200,{projects:[{project_id:"p-1",name:"x"}],total:120,limit:50,offset:0});
await viewProjects(); out.first=fetchLog.map(f=>f.url); out.pager=el("#ppager").innerHTML;
fetchLog.length=0; pPage(1); await tick(); out.next=fetchLog.map(f=>f.url);
respond("GET","^/admin/projects",200,{projects:[],total:0});
await viewProjects(); out.empty=el("#ptab tbody").innerHTML;
""", tmp_path)
    o = res["out"]
    assert o["first"] == ["/admin/projects?limit=50&offset=0"]
    assert "120" in o["pager"]
    assert o["next"] == ["/admin/projects?limit=50&offset=50"]
    assert "request" in o["empty"]            # 来源说明补上了请求身份


def test_project_drawer_links_active_ledgers_and_no_reading_is_not_none(tmp_path) -> None:
    res = _run_page(f"""
respond("GET","^/admin/projects/",200,{{project_id:"p-1",name:"x",root_path:""}});
respond("GET","^/admin/ledgers$",200,{{ledgers:[
  {{ledger_id:"{_LA}",title:"hit",status:"active",active_in:[{{agent:"codex",project:"p-1"}}]}},
  {{ledger_id:"{_LB}",title:"other",status:"active",active_in:[{{agent:"codex",project:"p-2"}}]}}]}});
await projectDetail("p-1"); out.hit=el("#drawer").innerHTML;
respond("GET","^/admin/ledgers$",200,{{ledgers:[]}});
await projectDetail("p-1"); out.none=el("#drawer").innerHTML;
respond("GET","^/admin/ledgers$",500,{{detail:"pool down"}});
await projectDetail("p-1"); out.fail=el("#drawer").innerHTML;
""", tmp_path)
    o = res["out"]
    assert _LA in o["hit"] and _LB not in o["hit"]
    assert "No ledger is active in this project" in o["none"]
    assert "pool down" in o["fail"] and "No ledger is active" not in o["fail"]


# ── 交接页（DS-05）───────────────────────────────────────────────────────────

_EXPORT_LINES = ("- exported:", "- content hash:")


def _strip_export_lines(md: str) -> str:
    return "\n".join(ln for ln in md.split("\n") if not ln.startswith(_EXPORT_LINES))


def _handoff_body(**over) -> dict:
    pkg = {"omitted": {"matters_exposure": 2, "rounds_exposure": 3, "matters_unavailable": 0,
                       "rounds_without_user_text": 1},
           "trimmed": {"rounds": 4, "matters": 5, "round_entries": 6, "over_limit": False},
           "rewrites": {"project": 7, "home": 8, "path": 9, "credential": 10},
           "matters": [], "recent_rounds": [], "source_ledger_id": _LA}
    pkg.update(over)
    return {"package": pkg, "markdown": "# <Handoff>\n- body"}


def test_handoff_defaults_to_public_and_shows_warning_and_counts(tmp_path) -> None:
    res = _run_page(f"""
respond("GET","/handoff",200,{_js(_handoff_body())},{{hold:true}});
const p=ledgerHandoff("{_LA}"); await tick();
out.loading=el("#drawer").innerHTML.includes('data-loading="1"');
out.warnWhileLoading=el("#drawer").innerHTML;
release(); await p; await tick();
out.html=el("#drawer").innerHTML;
out.urls=fetchLog.map(f=>f.url).filter(u=>u.includes("/handoff"));
""", tmp_path)
    o = res["out"]
    assert o["urls"] == [f"/admin/ledgers/{_LA}/handoff?exposure=public"]
    assert o["loading"] is True
    warn = "Review it before sending: the five sections are not filtered for personal information."
    assert warn in o["warnWhileLoading"] and warn in o["html"]
    assert '<option value="public" selected>' in o["html"]
    import html as _html
    import re
    fields = dict(re.findall(r'<div class="fk">(.*?)</div><div class="fv[^"]*">(.*?)</div>', o["html"]))
    assert fields["Matters above this exposure"] == "2" and fields["rounds above this exposure"] == "3"
    assert fields["round entries"] == "6" and fields["credentials"] == "10"
    assert "No anchored Matter" in o["html"] and "No recent rounds" in o["html"]
    assert "over its size limit" not in o["html"]
    assert "&lt;Handoff&gt;" in o["html"]     # markdown 转义后显示
    assert _html.unescape("&lt;Handoff&gt;") not in o["html"]


def test_handoff_over_limit_error_and_stale_exposure(tmp_path) -> None:
    over = _handoff_body(trimmed={"rounds": 1, "matters": 0, "round_entries": 0, "over_limit": True})
    res = _run_page(f"""
respond("GET","/handoff",200,{_js(over)});
await ledgerHandoff("{_LA}","local"); out.over=el("#drawer").innerHTML; out.url=fetchLog.filter(f=>f.url.includes("/handoff")).at(-1).url;
respond("GET","/handoff",404,{{status:"not_found",detail:"unknown ledger {_LA}"}});
await ledgerHandoff("{_LA}"); out.err=el("#drawer").innerHTML;
out.dl=(handoffDownload("md"),"no crash");

respond("GET","exposure=public",200,{_js(_handoff_body())},{{hold:true}});
const slow=ledgerHandoff("{_LA}","public"); await tick();
respond("GET","exposure=private",200,{{package:{{}},markdown:"PRIVATE-VERSION"}});
await ledgerHandoff("{_LA}","private");
release(); await slow; await tick();
out.stale=el("#drawer").innerHTML;
""", tmp_path)
    o = res["out"]
    assert "over its size limit" in o["over"] and o["url"].endswith("exposure=local")
    assert f"unknown ledger {_LA}" in o["err"] and "Retry" in o["err"]
    assert "PRIVATE-VERSION" in o["stale"]    # 慢的旧请求回来不覆盖新选择


def test_handoff_preview_matches_cli(tmp_path, ledger_apps, monkeypatch) -> None:
    from bladex_proxy.cli import ledger_cmds

    app = ledger_apps()
    res = _run_page(f"""await ledgerHandoff("{_LA}");""", tmp_path)
    page_url = next(f["url"] for f in res["fetchLog"] if "/handoff" in f["url"])
    page = app.client.get(page_url)
    assert page.status_code == 200, page.text
    data = page.json()
    assert data["markdown"]
    _route_cli(monkeypatch, app.client, ledger_cmds)

    md_file, json_file = tmp_path / "h.md", tmp_path / "h.json"
    assert _cli("ledger", "handoff", _LA, "--out", str(md_file)).exit_code == 0
    warned = _cli("ledger", "handoff", _LA, "--out", str(json_file), "--format", "json")
    assert warned.exit_code == 0
    assert _strip_export_lines(md_file.read_text("utf-8")) == _strip_export_lines(data["markdown"])
    cli_pkg = json.loads(json_file.read_text("utf-8"))
    for pkg in (cli_pkg, data["package"]):
        pkg.pop("exported_at"), pkg.pop("content_hash")
    assert cli_pkg == data["package"]

    # 页面显示的就是这份 markdown；Download .json 与 CLI --format json 同形
    res = _run_page(f"""
var saved=[]; window.Blob=function(parts){{saved.push(parts[0]);}};
window.URL={{createObjectURL:()=>"blob:x",revokeObjectURL(){{}}}};
respond("GET","/handoff",200,{_js(data)});
await ledgerHandoff("{_LA}"); out.html=el("#drawer").innerHTML;
handoffDownload("json"); handoffDownload("md"); out.saved=saved;
""", tmp_path)
    import html as _html
    assert _html.escape(data["markdown"], quote=True).replace("&#x27;", "&#39;") in res["out"]["html"]
    assert res["out"]["saved"][0] == json.dumps(data["package"], ensure_ascii=False, indent=2) + "\n"
    assert res["out"]["saved"][1] == data["markdown"]


def test_handoff_warning_is_the_cli_sentence(tmp_path, ledger_apps, monkeypatch, capsys) -> None:
    from bladex_proxy.cli import ledger_cmds

    app = ledger_apps()
    _route_cli(monkeypatch, app.client, ledger_cmds)
    assert ledger_cmds._handoff_export(_LA, str(tmp_path / "h.md"), "public", "md") == 0
    cli_warning = [ln for ln in capsys.readouterr().err.splitlines() if ln.startswith("Review it")]
    assert len(cli_warning) == 1 and f'"{cli_warning[0]}"' in _PAGE.read_text("utf-8")


#: 两边本来就该不同的字段：来路、随机 id 及其派生、时间戳
_EVENT_VOLATILE = {"ledger_id", "matter_id", "created_at", "updated_at", "origin", "key", "ts"}


def _normal_event(ev: dict) -> dict:
    def clean(x):  # noqa: ANN001, ANN202
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items() if k not in _EVENT_VOLATILE}
        if isinstance(x, list):
            return [clean(v) for v in x]
        return x
    return clean(ev)


def test_ledger_create_matches_cli_event_and_is_not_activated(tmp_path, ledger_apps, monkeypatch) -> None:
    from bladex_proxy.cli import ledger_cmds

    res = _run_page("""
tab="ledgers";
respond("POST","^/admin/ledgers/create$",200,{status:"ok",ledger_id:"ldg-new",rev:1});
respond("GET","^/admin/ledgers",200,{ledgers:[],ledger:{ledger_id:"ldg-new",section_order:[]}});
answers.push({goal:"  把事做完\\n第二行 ",title:" 标题 ",core:"核一\\n\\n核二",verified:"",open:"待办",next:""});
await ledgerCreate(el("#btn"));
""", tmp_path)
    sent = _posts(res)[0]
    assert sent["url"] == "/admin/ledgers/create"
    assert sent["body"] == {"goal": "把事做完\n第二行", "title": "标题", "core": ["核一", "核二"],
                            "verified": [], "open": ["待办"], "next": [], "origin": "dashboard"}
    assert any("not active yet" in t["msg"] for t in res["toasts"])
    assert not any("activate" in f["url"] or "switch" in f["url"] for f in res["fetchLog"])

    dash, cli_side = ledger_apps(), ledger_apps()
    activation_before = dict(dash.agency.activation.snapshot())
    assert dash.send(sent["method"], sent["url"], sent["body"]).status_code == 200
    assert dash.agency.activation.snapshot() == activation_before
    _route_cli(monkeypatch, cli_side.client, ledger_cmds)
    out = _cli("ledger", "create", "--goal", "把事做完\n第二行", "--title", "标题",
               "--core", "核一", "--core", "核二", "--open", "待办")
    assert out.exit_code == 0, out.output
    ev_d, ev_c = dash.events(), cli_side.events()
    assert ev_d[0]["payload"]["origin"] == "dashboard_create"
    assert ev_c[0]["payload"]["origin"] == "cli_create"
    assert _normal_event(ev_d[0]) == _normal_event(ev_c[0])


def test_ledger_edit_matches_cli_event(tmp_path, ledger_apps, monkeypatch) -> None:
    from bladex_proxy.cli import ledger_cmds

    pool = _seed_ledgers()
    res = _run_page(_edit_scenario(pool[_LA]), tmp_path)
    sent = _posts(res)[0]
    dash, cli_side = ledger_apps(), ledger_apps()
    assert dash.send(sent["method"], sent["url"], sent["body"]).status_code == 200
    _route_cli(monkeypatch, cli_side.client, ledger_cmds)
    out = _cli("ledger", "edit", _LA, "--goal", "把 A 做完，并写测试", "--add", "core=核心二")
    assert out.exit_code == 0, out.output
    ev_d, ev_c = dash.events(), cli_side.events()
    assert ev_d[0]["payload"]["origin"] == "dashboard_edit"
    assert ev_c[0]["payload"]["origin"] == "cli_edit"
    assert _normal_event(ev_d[0]) == _normal_event(ev_c[0])
    # 被剔除的字段里，账本 id 两边应当相同（编辑不换 id）
    assert ev_d[0]["payload"]["ledger"]["ledger_id"] == ev_c[0]["payload"]["ledger"]["ledger_id"] == _LA


def test_ledger_create_failure_keeps_the_draft_and_empty_goal_sends_nothing(tmp_path) -> None:
    res = _run_page("""
respond("POST","^/admin/ledgers/create$",400,{error:{message:"goal: boom",type:"validation_error"}});
answers.push({goal:"   ",title:"t"});
answers.push({goal:"写好的目标",title:"t"});
await ledgerCreate(el("#btn"));
out.posts=fetchLog.filter(f=>f.method==="POST").length;
""", tmp_path)
    assert res["out"]["posts"] == 1
    msgs = [t["msg"] for t in res["toasts"] if "err" in t["type"]]
    assert msgs == ["Goal cannot be empty", "goal: boom"]
    # 第三次打开对话框：服务端拒了，稿子还在
    goal_field = next(f for f in res["asked"][2]["fields"] if f["name"] == "goal")
    assert goal_field["value"] == "写好的目标"


_IMPORT_DRY = {"status": "dry_run", "dry_run": True, "ledger_id": "ldg-x", "id_conflict": False,
               "title": "<t>", "goal": "<img src=x onerror=alert(1)>", "entry_counts": {"core": 2}}


def test_import_dry_runs_first_and_cancel_writes_nothing(tmp_path) -> None:
    res = _run_page(f"""
respond("POST","^/admin/ledgers/import$",200,{_js(_IMPORT_DRY)});
answers.push({{package:'{{"a":1}}'}});
await ledgerImport(el("#btn"));
""", tmp_path)
    posts = _posts(res)
    assert [p["body"] for p in posts] == [{"package": {"a": 1}, "dry_run": True}]
    assert res["asked"][1]["title"] == "Import this ledger?"
    body = res["asked"][1]["body"]
    assert "&lt;img src=x" in body and "<img" not in body    # Goal 是外来内容，要转义
    assert "shown to your agents on every turn" in body


def test_import_confirm_writes_once_after_dry_run(tmp_path) -> None:
    res = _run_page(f"""
tab="ledgers";
respond("POST","^/admin/ledgers/import$",200,b=>b.body.dry_run?{_js(_IMPORT_DRY)}:{{status:"ok",ledger_id:"ldg-x",id_conflict:false}});
respond("GET","^/admin/ledgers",200,{{ledgers:[],ledger:{{ledger_id:"ldg-x",section_order:[]}}}});
answers.push({{package:'{{"a":1}}'}}); answers.push({{}});
await ledgerImport(el("#btn"));
""", tmp_path)
    assert [p["body"]["dry_run"] for p in _posts(res)] == [True, False]
    assert any("not active yet" in t["msg"] for t in res["toasts"])


def test_import_rejects_bad_json_and_stops_without_dry_run_ack(tmp_path) -> None:
    res = _run_page("""
answers.push({package:"not json"});
await ledgerImport(el("#btn"));
out.bad=fetchLog.filter(f=>f.method==="POST").length;

respond("POST","^/admin/ledgers/import$",200,{status:"ok",ledger_id:"ldg-x"});
answers.push({package:'{"a":1}'}); answers.push({});
await ledgerImport(el("#btn"));
out.noAck=fetchLog.filter(f=>f.method==="POST").length; out.askedAfter=out.__asked.length;
""", tmp_path)
    assert res["out"]["bad"] == 0
    assert res["out"]["noAck"] == 1            # 没确认 dry-run：不再发第二次
    msgs = [t["msg"] for t in res["toasts"] if "err" in t["type"]]
    assert any("not JSON" in m for m in msgs) and any("did not confirm a dry run" in m for m in msgs)


def test_import_round_trip_on_real_server_keeps_it_inactive(tmp_path, ledger_apps) -> None:
    """真服务端：导出 → 页面导入（dry-run → 确认）→ 同 id 已占用 ⇒ 换新 id，不激活。"""
    app = ledger_apps()
    pkg = app.client.get(f"/admin/ledgers/{_LA}/handoff").json()["package"]
    before = dict(app.agency.activation.snapshot())
    dry = app.client.post("/admin/ledgers/import", json={"package": pkg, "dry_run": True}).json()
    res = _run_page(f"""
tab="ledgers";
respond("POST","^/admin/ledgers/import$",200,{_js(dry)});
respond("GET","^/admin/ledgers",200,{{ledgers:[],ledger:{{ledger_id:"x",section_order:[]}}}});
answers.push({{package:{_js(json.dumps(pkg, ensure_ascii=False))}}}); answers.push({{}});
await ledgerImport(el("#btn"));
""", tmp_path)
    assert "already taken" in res["asked"][1]["body"]
    sent = _posts(res)[1]
    assert sent["body"] == {"package": pkg, "dry_run": False}
    r = app.send(sent["method"], sent["url"], sent["body"])
    assert r.status_code == 200, r.text
    assert r.json()["ledger_id"] != _LA and r.json()["id_conflict"] is True
    assert app.agency.activation.snapshot() == before
    assert app.events()[-1]["payload"]["origin"] == "handoff_import"


# ── 批 DS2 第三步 · DS-13：fact 更正与修改史 ─────────────────────────────────

_FID = "f-ds13"
_FACT = {"id": _FID, "content": "原文字", "item_kind": "assertion"}


def _fact_app(tmp_path):
    from datetime import UTC, datetime

    from bladex_core.fact import Fact
    from bladex_proxy.storage.memory_index import MemoryIndex

    cfg = _make_config()
    idx = MemoryIndex(cfg.index_path, embedder=None, read_only=False)
    idx.open()
    idx.add_fact(Fact(id=_FID, content="原文字", source_user_id="u1", created_at=datetime(2026, 11, 1, tzinfo=UTC)))
    idx.close()
    return create_app(cfg)


def _correct_events(app) -> list[dict]:
    return [e.model_dump(mode="json", exclude={"ts"}) for _k, e in app.state.hub.scan_admin_events()
            if e.event_type.value == "fact_correct"]


def test_ds13_page_correct_matches_the_backflow_event_except_origin(tmp_path) -> None:
    """仓库没有 `bladex memory correct`，这条端点的另一个调用方是回流通道 `AdminBackflowClient`。"""
    from bladex_proxy.export_backflow import AdminBackflowClient, BackflowEdit

    res = _run_page(f"""
respond("GET","^/admin/facts/{_FID}$",200,{_js(_FACT)});
respond("POST","/correct$",200,{{status:"corrected"}});
answers.push({{content:"新文字\\n第二行"}});
await factCorrect("{_FID}",el("#btn"));
""", tmp_path)
    sent = _posts(res)[0]
    assert sent["url"] == f"/admin/facts/{_FID}/correct"
    assert sent["body"] == {"content": "新文字 第二行", "origin": "dashboard", "submitter": ""}

    page_app, flow_app = _fact_app(tmp_path), _fact_app(tmp_path)
    with TestClient(page_app) as a:
        assert a.post(sent["url"], json=sent["body"]).status_code in (200, 202)
        ev_page = _correct_events(page_app)
    with TestClient(flow_app) as b:
        def req(method, path, body):  # noqa: ANN001, ANN202
            r = b.request(method, path, json=body)
            return r.status_code, r.json()
        assert AdminBackflowClient(req).apply(BackflowEdit(
            "correct", "obsidian", "", fact_id=_FID, content="新文字 第二行")).status == "accepted"
        ev_flow = _correct_events(flow_app)
    assert len(ev_page) == len(ev_flow) == 1
    assert ev_page[0]["payload"].pop("origin") == "dashboard"
    assert ev_flow[0]["payload"].pop("origin") == "obsidian"
    for ev in (ev_page[0], ev_flow[0]):
        ev["payload"]["entity"].pop("updated_at")     # 时间戳
    assert ev_page == ev_flow


def test_ds13_correct_shows_new_text_without_rereading_and_history(tmp_path) -> None:
    hist = {"fact_state": "present", "events": [
        {"type": "fact_correct", "at": "2026-11-05T01:00:00+00:00", "key": "k", "origin": "dashboard",
         "submitter": "", "previous_content": "原<文字>", "content": "新文字"},
        {"type": "tombstone", "at": "2026-11-05T02:00:00+00:00", "key": "t", "origin": "", "source": "user"}]}
    res = _run_page(f"""
respond("GET","^/admin/facts/{_FID}$",200,{_js(_FACT)});
respond("GET","/history$",200,{_js(hist)});
respond("POST","/correct$",200,{{status:"corrected"}});
answers.push({{content:"新文字"}});
await factCorrect("{_FID}",el("#btn"));
out.html=el("#drawer").innerHTML;
""", tmp_path)
    html = res["out"]["html"]
    assert "新文字" in html.split("History")[0]                # Index 读句柄落后时也显示新文字
    assert "Corrected" in html and "dashboard" in html and "Deleted" in html
    assert "原&lt;文字&gt;" in html                            # 改前文字转义后显示
    assert "Never corrected" not in html
    assert any(t["msg"] == "Fact corrected" for t in res["toasts"])


def test_ds13_no_change_empty_and_rejected_keep_the_draft(tmp_path) -> None:
    res = _run_page(f"""
respond("GET","^/admin/facts/{_FID}$",200,{_js(_FACT)});
respond("POST","/correct$",400,{{error:{{message:"correct boom",type:"validation_error"}}}});
answers.push({{content:"原文字"}});
await factCorrect("{_FID}",el("#btn"));
out.afterSame=fetchLog.filter(f=>f.method==="POST").length;
answers.push({{content:"   "}}); answers.push({{content:"写好的稿子"}});
await factCorrect("{_FID}",el("#btn"));
out.after=fetchLog.filter(f=>f.method==="POST").length;
""", tmp_path)
    assert res["out"]["afterSame"] == 0 and res["out"]["after"] == 1
    assert res["toasts"][0]["msg"] == "No change"
    assert [t["msg"] for t in res["toasts"] if "err" in t["type"]] == ["Text cannot be empty", "correct boom"]
    assert res["asked"][-1]["fields"][0]["value"] == "写好的稿子"


def test_ds13_deferred_is_recorded_not_shown_as_applied(tmp_path) -> None:
    res = _run_page(f"""
respond("GET","^/admin/facts/{_FID}$",200,{_js(_FACT)});
respond("POST","/correct$",202,{{status:"deferred"}});
answers.push({{content:"新文字"}});
await factCorrect("{_FID}",el("#btn"));
out.html=el("#drawer").innerHTML;
""", tmp_path)
    assert {"msg": "Recorded; takes effect shortly", "type": "toast warn"} in res["toasts"]
    assert "新文字" not in res["out"]["html"].split("History")[0]


@pytest.mark.parametrize(("hist", "want", "not_want"), [
    ({"events": []}, "Never corrected or deleted", "n/a"),
    ({}, "n/a", "Never corrected"),
    ({"events": [{"type": "audience_set", "at": "2026-11-05T01:00:00+00:00", "kind": "contradiction_mark"}]},
     "contradiction_mark", "Never corrected"),
])
def test_ds13_history_reading_states(tmp_path, hist, want, not_want) -> None:
    res = _run_page(f"""
respond("GET","^/admin/facts/{_FID}$",200,{_js(_FACT)});
respond("GET","/history$",200,{_js(hist)});
await factDetail("{_FID}"); out.html=el("#drawer").innerHTML;
""", tmp_path)
    tail = res["out"]["html"].split('<div class="sect">History</div>')[1].split("<details")[0]
    assert want in tail and not_want not in tail


def test_ds13_history_failure_is_not_never_corrected(tmp_path) -> None:
    res = _run_page(f"""
respond("GET","^/admin/facts/{_FID}$",200,{_js(_FACT)});
respond("GET","/history$",500,{{detail:"scan boom"}});
await factDetail("{_FID}"); out.html=el("#drawer").innerHTML;
""", tmp_path)
    assert "scan boom" in res["out"]["html"] and "Never corrected" not in res["out"]["html"]


def test_ds13_history_url_is_accepted_by_the_server(tmp_path) -> None:
    res = _run_page(f"""respond("GET","^/admin/facts/{_FID}$",200,{_js(_FACT)}); await factDetail("{_FID}");""", tmp_path)
    url = next(f["url"] for f in res["fetchLog"] if f["url"].endswith("/history"))
    app = _fact_app(tmp_path)
    with TestClient(app) as c:
        r = c.get(url)
        assert r.status_code == 200 and r.json()["fact_state"] == "present"


# ── 批 DS2 第三步 · MQ-F16：被拒的直编 ───────────────────────────────────────

_REFUSED = [{"ledger_id": _LA, "reason": "cannot read <this> line", "at": "2026-11-05T01:00:00+00:00",
             "kind": "refused"}]


@pytest.mark.parametrize(("value", "want"), [
    (None, '<span class="dim2">n/a</span>'),
    ([], ">0<"),
    (_REFUSED, ">1<"),
])
def test_f16_overview_card_separates_no_reading_from_zero(tmp_path, value, want) -> None:
    res = _run_page(f"""out.html=flashCard({_js({"available": False, "user_edit_refused": value})});""", tmp_path)
    cell = res["out"]["html"].split('id="frefused">')[1].split("</div>")[0]
    assert (want in cell) if want.startswith("<") else (want in f">{cell.rsplit('<', 1)[0]}<")
    if value is None:
        assert "0" not in cell.replace("dim2", "")


def _ledger_scenario(flash: object) -> str:
    pool = _seed_ledgers()
    return f"""
respond("GET","^/admin/status$",200,{_js({"flash": {"user_edit_refused": flash}})});
respond("GET","^/admin/ledgers$",200,{{ledgers:{_js(_ledger_rows(pool))},total:2}});
respond("GET","^/admin/ledgers/{_LA}$",200,{_js(_ledger_detail_body(pool[_LA]))});
respond("GET","/handoff",200,{{package:{{}},markdown:"md"}});
await viewLedgers();
out.toolbar=el("#lrefused").innerHTML; out.rows=el("#ltab tbody").innerHTML;
await ledgerDetail("{_LA}"); out.drawer=el("#drawer").innerHTML;
await ledgerHandoff("{_LA}"); out.handoff=el("#drawer").innerHTML;
"""


def test_f16_ledger_pages_show_the_refusal(tmp_path) -> None:
    o = _run_page(_ledger_scenario(_REFUSED) + """
ledgerRefusedToggle(); out.only=el("#ltab tbody").innerHTML;
""", tmp_path)["out"]
    assert "1 direct edit not adopted" in o["toolbar"]
    assert "edit not adopted" in o["rows"]
    assert "cannot read &lt;this&gt; line" in o["drawer"] and "Your file was not overwritten" in o["drawer"]
    assert "cannot read &lt;this&gt; line" in o["handoff"] and "last adopted version" in o["handoff"]
    assert _LA in o["only"] and _LB not in o["only"]


def test_f16_ledger_pages_no_reading_and_none(tmp_path) -> None:
    o = _run_page(_ledger_scenario(None), tmp_path)["out"]
    assert "n/a" in o["toolbar"] and "0" not in o["toolbar"]
    assert 'data-refused="1"' not in o["drawer"] and 'data-refused="1"' not in o["handoff"]
    o = _run_page(_ledger_scenario([]), tmp_path)["out"]
    assert o["toolbar"] == "" and 'data-refused="1"' not in o["drawer"]


def test_f16_status_failure_does_not_break_the_ledger_page(tmp_path) -> None:
    pool = _seed_ledgers()
    o = _run_page(f"""
respond("GET","^/admin/status$",500,{{detail:"boom"}});
respond("GET","^/admin/ledgers$",200,{{ledgers:{_js(_ledger_rows(pool))},total:2}});
await viewLedgers(); out.rows=el("#ltab tbody").innerHTML; out.toolbar=el("#lrefused").innerHTML;
""", tmp_path)["out"]
    assert _LA in o["rows"] and "n/a" in o["toolbar"]
