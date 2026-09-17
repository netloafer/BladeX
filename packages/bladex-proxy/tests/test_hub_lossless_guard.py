"""刚性原则 13 守卫（2026-08-29，Jason 拍板：元数据与正文同等，入 Hub 必须无损）。

三层方案的静态半边（层 1 + 层 2）：
- 层 1：存储路径源文件禁**未登记的有损处理**——数值切片只放行两类：
  ① hash/id 派生（sha256/hexdigest/uuid）；② 带显式标记或 `lossless-ok` 注解的
  截断。案发驱动：MQ-A22（header 200 静默截断把 x-codex-turn-metadata 切半截）。
- 层 2：元数据生产的单实现点——resolve_identity / 项目识别在 server 只许有
  一个调用点（三协议共享 _prepare_round），防 MQ-A18/P7 的
  "同一件事只在一个端点做对了"。
层 3（live 非空率对账）在 scripts/probe_turn_field_coverage.py——静态守卫
杀不了"字段存在但生产者断线"（Turn.subagent 全 0 的形态），只有读数能看见。

MQ-V9 纪律：源码断言按路径读文本，不用 inspect.getsource。
"""

from __future__ import annotations

import os
import re

import bladex_proxy as _pkg_mod

_PKG = os.path.dirname(_pkg_mod.__file__)   # 按包定位——测试从任何目录跑都成立

#: 层 1 扫描面：捕获数据流进 Turn/Hub 的模块（读侧/展示侧不在此列——
#: 展示截断不丢底账）。新模块进入存储路径时必须加进来。
_STORAGE_FILES = (
    "agent_bucket.py",
    "capture.py",
    "models.py",
    "project_identity.py",
    os.path.join("storage", "memory_hub.py"),
    os.path.join("storage", "pipeline_worker.py"),
    os.path.join("storage", "pipeline_redis.py"),
    os.path.join("storage", "blob_store.py"),
)

#: 数值切片，≥32 才疑似内容截断（[:12]/[:16] 是 hash/id 派生的惯用宽度）。
_SLICE = re.compile(r"\[\s*:\s*(\d{2,})\s*\]")
#: 同行出现即放行的记号：hash/id 派生、显式截断标记、注解。
_ALLOW = ("hexdigest", "sha256", "uuid", "_TRUNCATED_MARKER", "lossless-ok")


def scan_lossless(text: str) -> list[tuple[int, str]]:
    """返回未登记的疑似有损切片行（纯函数，供判别力自测）。"""
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(text.splitlines(), 1):
        m = _SLICE.search(line)
        if not m or int(m.group(1)) < 32:
            continue
        if any(tok in line for tok in _ALLOW):
            continue
        s = line.strip()
        if s.startswith("#") or "logger." in line:
            continue          # 注释与日志行不是存储
        hits.append((i, s))
    return hits


class TestLayer1NoUnregisteredLoss:
    def test_scanner_flags_a_planted_truncation(self):
        """判别力实测：先证明尺子能抓到坏样本，再拿它量真文件。"""
        bad = 'out[name] = value[:200]\n'
        good = 'ref = hashlib.sha256(x).hexdigest()[:16]\n'
        marked = 'value = value[:4096] + _TRUNCATED_MARKER\n'
        annotated = 'head = content[:8192]  # lossless-ok: 只读匹配不入库\n'
        assert scan_lossless(bad), "阳性对照没被抓到 ⇒ 尺子是坏的"
        assert not scan_lossless(good + marked + annotated)

    def test_storage_path_has_no_silent_truncation(self):
        offenders = {}
        for rel in _STORAGE_FILES:
            path = os.path.join(_PKG, rel)
            with open(path, encoding="utf-8") as f:
                hits = scan_lossless(f.read())
            if hits:
                offenders[rel] = hits
        assert not offenders, (
            f"存储路径出现未登记的有损处理（刚性原则 13）：{offenders}\n"
            "允许的只有 hash/id 派生与带显式标记/lossless-ok 注解的截断。")

    def test_header_truncation_is_marked_and_wide(self):
        """MQ-A22 案发点钉死：上限 ≥4096 且超限必带显式标记。"""
        with open(os.path.join(_PKG, "agent_bucket.py"), encoding="utf-8") as f:
            src = f.read()
        m = re.search(r"_MAX_HEADER_VALUE_CHARS\s*=\s*(\d+)", src)
        assert m and int(m.group(1)) >= 4096
        assert "_TRUNCATED_MARKER" in src


class TestLayer2SingleProducerSite:
    """元数据生产必须走三协议共享的 _prepare_round（MQ-A18/P7 族防复发）。"""

    def _server_src(self) -> str:
        from _source_probe import package_source
        return package_source("server")   # F0.1 拆包：server.py → server/ 包，按包拼接读源码

    def test_identity_resolution_single_call_site(self):
        src = self._server_src()
        assert src.count("resolve_identity(") == 1, \
            "resolve_identity 出现多个调用点 ⇒ 某协议在旁路生产身份元数据"

    def test_project_resolution_single_call_site(self):
        src = self._server_src()
        assert src.count("_resolve_project(") == 1
