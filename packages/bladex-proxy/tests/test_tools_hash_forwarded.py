"""`tools_hash` 必须记**实际转发上游**的工具面——三个入站协议一致（V-R1 取证）。

🔴 立卡背景：`/v1/chat/completions` 的注释白纸黑字写着
"Hub 的 tools_hash 记录的是实际转发形态"，而这条不变量**只在三个端点里的一个上成立**：

  端点                      augment_tools 与 _build_request_params 的先后
  /v1/chat/completions      augment 在前  ✅  且增补直接写回 `req.tools`
  /v1/messages              build 在前    ❌  且增补写进 `extra_kwargs["tools"]`
  /v1/responses             build 在前    ❌  同上

后两个协议**光调换顺序都救不回来**——`_build_request_params` 读的是 `req.tools`，
与增补落地的 `extra_kwargs["tools"]` 是两个对象。

代价不是"少记一个字段"：它让"模型实际看到哪些工具"在 **Claude Code（messages）
与 Codex（responses）上查不出来**。取证时据此差点得出"CC/Codex 从没拿到过
bladex 工具面"的结论——实际**无法判定**。仪器盲区伪装成读数。

同型：MQ-L14（锚层报告自己已就绪却一条都没命中）、
「仪器骗人的第一形态：参照系脱钩」。

**为什么不走端到端**：记录这一步在 P0 管线之后，端到端要真 Redis；
沙盒没有 ⇒ turn 进不了 Hub ⇒ 断言会在**存储没跑**的情况下假红/假绿。
故拆成"取值单元 + 接线结构"两层，两层都不依赖外部进程。
"""

from __future__ import annotations

import inspect

import pytest

_AGENT_TOOL = {"type": "function",
               "function": {"name": "their_tool", "description": "x",
                            "parameters": {"type": "object", "properties": {}}}}
_BLADEX_TOOL = {"type": "function",
                "function": {"name": "bladex_ledger_switch", "description": "y",
                             "parameters": {"type": "object", "properties": {}}}}


class _StubHub:
    """只记下被要求存的那份工具面。"""

    def __init__(self) -> None:
        self.stored: list | None = None

    def store_tools_schema(self, tools):
        self.stored = tools
        return "hash-" + str(len(tools or []))


class _StubReq:
    def __init__(self, tools):
        self.model = "m"
        self.temperature = None
        self.max_tokens = None
        self.tool_choice = None
        self.tools = tools


class _StubRequest:
    def __init__(self, hub):
        self.app = type("A", (), {"state": type("S", (), {"hub": hub})()})()


class TestForwardedToolsWins:
    def _run(self, req_tools, forwarded):
        from bladex_proxy.server import _build_request_params

        hub = _StubHub()
        rp = _build_request_params(_StubRequest(hub), _StubReq(req_tools),
                                   forwarded_tools=forwarded)
        return hub.stored, rp

    def test_records_forwarded_not_request(self):
        """🔴 阳性对照：增补后的 `bladex_*` 必须是被记下的那份。"""
        stored, rp = self._run([_AGENT_TOOL], [_AGENT_TOOL, _BLADEX_TOOL])
        names = [t["function"]["name"] for t in stored]
        assert names == ["their_tool", "bladex_ledger_switch"], names
        assert rp.tools_hash

    def test_omitting_the_override_keeps_old_behaviour(self):
        """阴性对照：不传 `forwarded_tools` = 逐字回到旧路径（chat/completions 用）。"""
        stored, _ = self._run([_AGENT_TOOL], None)
        assert stored == [_AGENT_TOOL]

    def test_empty_forwarded_list_is_not_a_fallback(self):
        """`[]` 是"转发时没有工具"，不该悄悄回落去记 `req.tools`。"""
        stored, rp = self._run([_AGENT_TOOL], [])
        assert stored is None and rp.tools_hash == ""

    def test_hub_failure_does_not_break_the_hot_path(self):
        from bladex_proxy.server import _build_request_params

        class _Boom:
            def store_tools_schema(self, tools):
                raise RuntimeError("hub down")

        rp = _build_request_params(_StubRequest(_Boom()), _StubReq([_AGENT_TOOL]),
                                   forwarded_tools=[_AGENT_TOOL, _BLADEX_TOOL])
        assert rp.tools_hash == ""      # 留空，不抛


class TestWiringOrderAcrossProtocols:
    """接线结构：三个端点必须一致，否则上面那条断言在某个协议上静默失效。

    V-A1 T2 编排函数化（2026-08-28）后 augment 收进 `_apply_agency_surfaces`
    单实现；顺序不变量升级为"每个端点的 `_apply_agency_surfaces` 调用先于
    它的 `_build_request_params`"。取源按路径读文本（MQ-V9）。
    """

    @staticmethod
    def _src() -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parents[1]
                / "bladex_proxy" / "server.py").read_text(encoding="utf-8")

    def test_build_always_after_augment(self):
        src = self._src().split("\n")
        aug = [i for i, l in enumerate(src) if "_apply_agency_surfaces(" in l
               and "def _apply_agency_surfaces" not in l]
        bld = [i for i, l in enumerate(src) if "request_params = _build_request_params(" in l]
        assert len(aug) == 3, f"入站端点数变了？_apply_agency_surfaces 调用 {len(aug)} 次"
        assert len(bld) == 3, f"_build_request_params 出现 {len(bld)} 次"
        for a, b in zip(sorted(aug), sorted(bld)):
            assert a < b, ("_build_request_params 排在 _apply_agency_surfaces 之前 —— "
                           "记下的会是增补前的工具面（V-R1 病例）")

    def test_protocol_endpoints_pass_the_forwarded_list(self):
        """增补落在 `extra_kwargs` 的那两个端点必须显式传 `forwarded_tools=`。

        只靠顺序不够：读的对象不同，排在后面也照样记错。
        """
        src = self._src()
        assert src.count("forwarded_tools=extra_kwargs.get(\"tools\")") == 2, (
            "/v1/messages 与 /v1/responses 必须各传一次实际转发形态")

    @pytest.mark.parametrize("ep", ["/v1/chat/completions", "/v1/messages",
                                    "/v1/responses"])
    def test_endpoint_still_registered(self, ep):
        """防止上面的计数守卫因为端点被删/改名而变成空转。"""
        assert f'"{ep}"' in self._src()
