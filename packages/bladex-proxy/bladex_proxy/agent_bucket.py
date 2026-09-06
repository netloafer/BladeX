"""未识别 agent 的分桶与 header 脱敏（执行卡 G11.1 + G11.9，台账接入域 MQ-A1–A5）。

**本模块只做分桶，不做命名。** 这个区分是整个方案的核心，改动前务必读懂：

原方案曾想用 User-Agent 的 product token 直接给 agent **命名**，被样本证伪并撤回
（2026-08-18 Jason 拍板）。UA 不可作身份权威来源的四条实证：

1. UA 随版本换 product token，变的不只是版本号——Claude Code 存在
   ``claude-cli/2.0.37 (external, cli)`` 与 ``claude-code/*`` 两种形态；
2. UA 用户可配——Codex ``http_headers = { "User-Agent" = "Codex" }``、
   Gemini CLI ``GEMINI_CLI_SURFACE`` 环境变量、Hermes per-provider ``extra_headers``；
3. UA 可缺失——生产 proxy 需给 Codex 请求注入默认 UA；
4. 一个 agent 多个 UA，且主路径上不是自己的名字——Hermes 的
   ``_HERMES_USER_AGENT`` 只用于 ``/v1/models`` 与 pricing catalog，chat 主路径走
   OpenAI SDK，UA = SDK 默认值。

把 UA 当身份 = 与 2026-08-16 修掉的 ``user_id = key hash8`` 同形（把外部可变的东西
当身份）。**降为分桶信号后失败模式变了**：分桶只要求"同一 agent 稳定落同一桶"，
不要求名字正确；UA 漂移的后果从"错误署名"（不可逆的记忆污染）降级为"多开一个桶"
（用户在 dashboard 合并一下，走 AGENT_CLAIM 可回滚）。

同理，**排除名单的角色也变轻了**：它原本要防"把 SDK 名当 agent 名"（漏一项 = 错误
署名），现在只判断"UA 有没有区分度"（漏一项 = 多一个低质量桶，可修复）。

命名只有两个来源：规则库匹配（G11.10）与用户认领（G11.11）。本模块不参与。
"""

from __future__ import annotations

import hashlib
import re

__all__ = [
    "GENERIC_CLIENT_TOKENS",
    "INFRA_VENDORS",
    "REDACTED_HEADERS",
    "RESERVED_VENDOR_SEGMENTS",
    "VENDOR_SESSION_SUFFIXES",
    "VENDOR_SUFFIXES",
    "BucketResult",
    "bucket_unknown_agent",
    "extract_vendor_ids",
    "extract_vendor_segments",
    "normalize_ua_flavor",
    "parse_ua_product",
    "redact_headers",
]


# ── UA 解析 ──────────────────────────────────────────────────────────────────

#: RFC 9110 §10.1.5 的 ``product["/" product-version]``，后接可选 comment。
#: 只取**第一个** product token；版本段丢弃（抗版本漂移的机制在这里，不在名单里）。
_UA_PRODUCT_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)(?:/[^\s]*)?")

#: 通用 HTTP 客户端 / SDK 的 product token —— **不是 agent 名**。
#:
#: 命中即视为"UA 无区分度"，触发 tool 签名补位（见 :func:`bucket_unknown_agent`）。
#: 漏一项的后果只是多一个低区分度的桶，不会造成错误署名——这是降级为分桶信号后
#: 才成立的宽容度，别退回用它做命名。
GENERIC_CLIENT_TOKENS: frozenset[str] = frozenset({
    # 官方 SDK
    "openai", "anthropic", "google-genai", "google-api-python-client",
    "azure-sdk-for-python", "aws-sdk-js", "boto3", "botocore",
    "litellm", "langchain", "llamaindex",
    # 🔴 同一 SDK 的**异步客户端类名**（2026-08-26 实测漏项）。
    # openai-python 的 `AsyncOpenAI` 会把 UA 写成 `AsyncOpenAI/Python 2.24.0`
    # ⇒ product token 归一成 `asyncopenai`，与 `openai` 是两个字符串，
    # 于是它**逃过了这份名单**：`suggest_rule` 会给它一条 UA 规则。
    # 实际后果（Hermes 的视觉子调用桶 `unknown-9eb9e3a9`）：认领时会写出
    # `(?i)^asyncopenai\b` —— 从此**任何用 async-openai 的客户端都算 hermes**。
    # 这与 `openai` 在名单里的理由完全一样：它标识的是 SDK 不是 agent。
    # 加一项就得想到它的同类：其它 SDK 的异步/流式客户端也走这套命名。
    "asyncopenai", "asyncanthropic", "asyncazureopenai", "azureopenai",
    # Python HTTP 栈
    "python-httpx", "httpx", "python-requests", "requests", "urllib3",
    "aiohttp", "python-urllib", "python",
    # Node / JS HTTP 栈
    "axios", "node-fetch", "undici", "got", "superagent", "node",
    # 其它语言 / 工具
    "go-http-client", "okhttp", "java", "reactor-netty", "apache-httpclient",
    "curl", "wget", "postmanruntime", "insomnia", "restsharp",
})


#: 版本号形态（`6.40.0` / `1.99` / `v2`），归一时整体剔除。
_VERSION_RE = re.compile(r"\bv?\d+(?:\.\d+)*\b")


def normalize_ua_flavor(ua: str) -> str:
    """把整条 UA 归一成**去版本**的"SDK 风味"串，供传输身份兜底用。

    存在理由（2026-08-19 Pi 真实流量逼出来的）：:func:`parse_ua_product` 只取**第一个**
    product token，于是 ``OpenAI/JS 6.40.0`` 与 ``OpenAI/Python 1.99.1`` 双双塌缩成
    ``openai`` —— 两个完全不同的客户端拿到同一个传输身份。实测后果：Hermes 先说话，
    Pi 的请求就被同源继承署名成 ``hermes:default``，正是 G11 要消灭的那个形态。

    **区分信息本来就在 UA 里，是我在取 product token 时把它丢了。**

    这不违反"UA 不能当身份权威来源"：这里产出的仍然只是**分桶/同源信号**，不参与
    命名。SDK 风味（`openai/js` vs `openai/python`）是客户端级稳定属性，不随会话变，
    去掉版本号后也不随升级漂移。

    :param ua: 原始 ``User-Agent``。
    :returns: 小写、去版本、空白折叠后的串；空 UA 返回空串。
    """
    if not ua:
        return ""
    flavor = _VERSION_RE.sub(" ", ua.lower())
    flavor = re.sub(r"[\s,;]+", " ", flavor).strip()
    # 结尾常残留孤立的 "/"（`python-httpx/0.27` → `python-httpx/`）
    return flavor.rstrip("/ ").strip()


def parse_ua_product(ua: str) -> str | None:
    """取 User-Agent 的首个 product token，**去掉版本段**，小写归一。

    去版本是抗漂移的机制所在：``deepseek-harness/0.3.1`` 与 ``deepseek-harness/0.4.0``
    必须给出同一个 token，否则 agent 每升一次级就换一个桶（G11.9 第一验收项）。

    :param ua: 原始 ``User-Agent`` header 值；空串/无法解析返回 ``None``。
    :returns: 小写 product token，或 ``None``。
    """
    if not ua:
        return None
    m = _UA_PRODUCT_RE.match(ua)
    if not m:
        return None
    token = m.group(1).strip().lower()
    return token or None


# ── 厂商前缀 header ──────────────────────────────────────────────────────────

#: ``x-{vendor}-{suffix}`` 里我们认的后缀。**封闭枚举**，不做开放匹配——
#: 否则 ``x-forwarded-for`` 会解析出 vendor=``forwarded``、``x-real-ip`` 解析出
#: vendor=``real`` 这类噪声。
#:
#: 这是本模块**唯一**一份后缀定义：:data:`VENDOR_SESSION_SUFFIXES` 是它的子集，
#: 别在别处另抄一份字面量（守卫见 ``test_g113_vendor_session.py``）。
VENDOR_SUFFIXES: tuple[str, ...] = (
    "user-id", "session-id", "conversation-id", "thread-id",
    "client-id", "app-id", "agent-id", "run-id", "trace-id", "request-id",
)

_VENDOR_RE = re.compile(
    r"^x-(?P<vendor>[a-z0-9][a-z0-9-]*)-(?P<suffix>" + "|".join(VENDOR_SUFFIXES) + r")$"
)

#: 我方自己的两段式 header 被正则拆出来的伪 vendor 段。
#:
#: ``x-agent-id`` 会解析成 vendor=``agent``、``x-session-id`` 解析成 vendor=``session``
#: ——它们不是厂商前缀，是我们自己的约定 header，必须排除。
RESERVED_VENDOR_SEGMENTS: frozenset[str] = frozenset({
    "agent", "session", "api", "client", "user", "request", "trace",
})

#: 可以当作**显式 session 信号**回收的厂商后缀（G11.3）。
#:
#: 🔴 只有 ``session-id`` 一项，且这不是"先放一个以后再加"。``conversation-id`` /
#: ``thread-id`` 与 session **粒度不同**——一个 conversation 可能横跨多个 session，
#: 也可能反过来。粒度对不上的 id 拿去当 session_id，产生的是"会话被错误地并成一段"
#: 或"一段被切碎"，两种都直接污染 Memory Hub key 与 CAP/装配的会话边界。
#: 要加一项，先拿真实流量标定它与 session 的粒度关系，别按名字像不像来收。
VENDOR_SESSION_SUFFIXES: frozenset[str] = frozenset({"session-id"})

#: 基础设施 / 我方自有的 vendor 段 —— 提取到也丢弃。
#:
#: ``stainless`` 是 openai-python / anthropic SDK 的代码生成器，会发一组
#: ``x-stainless-*``；它标识的是 SDK 不是 agent，和 GENERIC_CLIENT_TOKENS 同性质。
INFRA_VENDORS: frozenset[str] = frozenset({
    "bladex",       # 我方自有
    "stainless",    # openai/anthropic SDK 代码生成器
    "forwarded", "real", "amzn", "amz", "aws", "azure", "goog", "google",
    "cf", "cloudflare", "vercel", "envoy", "b3", "datadog", "newrelic",
    "correlation", "http",
})


def _iter_vendor_headers(
    headers: dict[str, str],
) -> list[tuple[str, str, str, str]]:
    """把"哪些 header 算厂商前缀"这件事收成**一处判定**。

    两个消费者共用：:func:`extract_vendor_segments`（只要 vendor 段，用于分桶）与
    :func:`extract_vendor_ids`（要值，用于 G11.3 回收 session 信号）。分成两份写
    就会有一份先漂——排除名单少一项的后果是把 Cloudflare 的 ``x-cf-session-id``
    当成 agent 的会话 id。

    :param headers: header 字典（键大小写不敏感）。
    :returns: ``(vendor, suffix, 小写 header 名, 值)`` 列表，顺序同入参；已剔除
        :data:`RESERVED_VENDOR_SEGMENTS` 与 :data:`INFRA_VENDORS`。
    """
    out: list[tuple[str, str, str, str]] = []
    for raw_name, raw_value in headers.items():
        name = raw_name.strip().lower()
        m = _VENDOR_RE.match(name)
        if not m:
            continue
        vendor = m.group("vendor")
        if vendor in RESERVED_VENDOR_SEGMENTS or vendor in INFRA_VENDORS:
            continue
        out.append((vendor, m.group("suffix"), name, "" if raw_value is None else str(raw_value)))
    return out


def extract_vendor_segments(headers: dict[str, str]) -> list[str]:
    """从 ``x-{vendor}-{已知后缀}`` 形态的 header 名里提取 vendor 段。

    只看 header **名字**，不看值——值里装的是 session id / user id 这类每轮都变的
    东西，进 hash 会让桶不稳定（G11.9 的头号反面用例）。

    :param headers: header 字典（键大小写不敏感）。
    :returns: 去重排序后的 vendor 段列表，已剔除 :data:`INFRA_VENDORS`。
    """
    return sorted({vendor for vendor, _suffix, _name, _value in _iter_vendor_headers(headers)})


def extract_vendor_ids(
    headers: dict[str, str],
    suffixes: frozenset[str],
) -> list[tuple[str, str, str]]:
    """取厂商前缀 header 的**值**（G11.3）。

    与 :func:`extract_vendor_segments` 的分工：那个只读名字（值不稳定，进 hash 会
    毁掉分桶），这个专门读值——因为 ``x-{vendor}-session-id`` 的值**正是**我们要的
    显式会话信号。两者共用 :func:`_iter_vendor_headers` 的准入判定。

    **通配，不是名单**：不认识的 vendor（``x-foo-bar-session-id``）照样返回。刚性
    原则 10 —— 适配 agent 的既有行为，而不是每接一个新 agent 就回来改我们的代码。

    本函数**不判断值好不好用**（空串、带 ``/``、超长都原样返回）。校验是调用方的
    政策，放这里会变成第二处判据。

    :param headers: header 字典（键大小写不敏感）。
    :param suffixes: 要收的后缀，取自 :data:`VENDOR_SESSION_SUFFIXES` 这类闭集。
    :returns: ``(vendor, 小写 header 名, 值)``，**按 vendor 段排序**——多家同时发时
        选谁必须与 dict 插入顺序无关，否则同一个客户端会因为 header 顺序变化而换
        session。
    """
    return sorted(
        (vendor, name, value)
        for vendor, suffix, name, value in _iter_vendor_headers(headers)
        if suffix in suffixes
    )


# ── 分桶 ─────────────────────────────────────────────────────────────────────


class BucketResult:
    """分桶结果。

    **两个键，用途不同——别合并成一个**（2026-08-18 被测试逼出来的修正）：

    ``bucket_id``（**表面身份**，含 tool 签名）
        给未识别 agent 命名用。tool 集合区分度高，能把两个都走 openai-python
        的陌生 agent 分开。

    ``origin_key``（**传输身份**，只含 UA product token + vendor 段）
        同源继承（G11.5）判据用。**不含 tool 签名**，因为工具集在同一个客户端内部
        会变——主请求带全套工具、标题生成/摘要这类子请求往往一个工具都不带。拿
        含工具的键做同源判据，子请求会匹配不上父请求，退化成"每个子请求各自成桶"，
        正是 ADR-0021 修正当年解决的那个问题（子任务被当新请求误路由）。

    取舍写明：``origin_key`` 更粗，代价是**两个 UA 均为 SDK 默认值、又都不发厂商
    header 的 agent 会共享同一个 origin_key**，其中一个被指纹识别出来后，另一个的
    请求会继承它。这是"传输层确实给不出区分信号"的诚实结果——真要分开只能靠指纹
    规则库（G11.10）把其中一个识别出来。相比改动前"任何陌生请求继承最后说话的那个
    agent"，暴露面已从"全部"收窄到"传输层同形的少数"。
    """

    __slots__ = ("bucket_id", "origin_key", "basis", "signals")

    def __init__(
        self, bucket_id: str, basis: str, signals: list[str], origin_key: str = ""
    ) -> None:
        self.bucket_id = bucket_id
        self.origin_key = origin_key
        self.basis = basis
        self.signals = signals

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"BucketResult(bucket_id={self.bucket_id!r}, "
            f"origin_key={self.origin_key!r}, basis={self.basis!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BucketResult):
            return NotImplemented
        return self.bucket_id == other.bucket_id and self.signals == other.signals


def bucket_unknown_agent(
    ua: str,
    headers: dict[str, str],
    tool_names: set[str] | None = None,
) -> BucketResult:
    """给未识别的 agent 算一个稳定的桶 id：``unknown-<hash8>``。

    取代此前"所有未识别 agent 共用一个 ``unknown``"的形态——那等于把互不相干的
    agent 合进同一个记忆命名空间，**是换了名字的误合并**（与 dsh 被署名成 hermes
    同形）。分桶后是"先分开、可合并"，与 Matter 的 provisional 机制同构。

    信号取舍（2026-08-18 定案）：

    ==================  ==========================================================
    信号                取它的理由 / 不取别的的理由
    ==================  ==========================================================
    UA product token    去版本后跨版本稳定；漂移只多开一个桶，可合并
    厂商 vendor 段      协议约定而非展示字符串，比 UA 稳
    tool 名集合（补位） 仅当上面两者都无区分度时启用。内容派生 = 我方自己从请求体
                        算，不随对方 UA 配置与版本漂移，区分度极高
    ==================  ==========================================================

    🔴 **绝不 hash header 全集**：header 里有 session id（dsh 每会话变）、版本号、
    request/trace id，全量 hash 会让同一 agent **每轮一个新身份**，记忆碎成粉末，
    比合并成一个 ``unknown`` 更糟。这是本函数存在的第一约束。

    **已知弱点（tool 补位路径）**：同一 agent 的主请求与子 agent 请求可能带不同的
    tool 子集 → 落不同桶。后果是多开桶（可合并），不是误合并，故接受。真实数据若
    显示噪声偏高，缓解方向是改用 tool 名的**命名空间前缀**（``cordis_*`` → ``cordis``）
    而非全集——但那要真实样本支持后再做，不预先优化。

    :param ua: ``User-Agent`` header 值。
    :param headers: 完整 header 字典（只读名字，不读值）。
    :param tool_names: 本轮请求里出现的 tool 名集合；``None``/空表示无工具信号。
    :returns: :class:`BucketResult`，``bucket_id`` 形如 ``unknown-a1b2c3d4``。
    """
    # ── 传输身份：UA product token + vendor 段。客户端级稳定，同源继承用。──
    distinctive: list[str] = []

    product = parse_ua_product(ua)
    if product and product not in GENERIC_CLIENT_TOKENS:
        distinctive.append(f"ua:{product}")

    for vendor in extract_vendor_segments(headers):
        distinctive.append(f"vendor:{vendor}")

    # SDK 风味兜底：product token 落在通用名单里（说明 UA 报的是 SDK 不是 agent）时，
    # **不要把整条 UA 一起丢掉** —— `openai/js` 与 `openai/python` 是不同的客户端。
    # 🔴 丢掉的后果实测过：Pi（OpenAI/JS）与 Hermes（OpenAI/Python）传输身份相同，
    # Hermes 先说话，Pi 就被同源继承署名成 hermes:default。
    transport = list(distinctive)
    if not transport:
        flavor = normalize_ua_flavor(ua)
        if flavor:
            transport.append(f"sdk:{flavor}")

    origin_basis = "|".join(transport) if transport else "none"
    origin_key = hashlib.sha256(origin_basis.encode()).hexdigest()[:8]

    # ── 表面身份：传输身份 + tool 签名补位。命名未识别 agent 用。──
    #
    # 补位的触发条件看的是 **distinctive 是否为空**，不是 transport ——
    # SDK 风味只说明"这是个 node/python 客户端"，区分度远不足以给 agent 命名，
    # 同一 SDK 下的两个陌生 agent 仍要靠工具集分开。
    signals = list(transport)
    if not distinctive and tool_names:
        # UA 与 vendor 都没有区分度 —— 补内容派生信号，把走同一个 SDK 的不同
        # agent 分开。**只在这里补，不进 origin_key**（理由见 BucketResult）。
        normalized = sorted({name.strip().lower() for name in tool_names if name.strip()})
        if normalized:
            digest = hashlib.sha256("\n".join(normalized).encode()).hexdigest()[:8]
            signals.append(f"tools:{digest}")

    if not signals:
        # 三种信号全无（无 UA、无厂商 header、无工具）。仍给一个确定性的桶，
        # 而不是退回共用 `unknown`——同样是"宁分勿合"，只是这个桶注定较杂。
        signals.append("none")

    basis = "|".join(signals)
    bucket = hashlib.sha256(basis.encode()).hexdigest()[:8]
    return BucketResult(
        bucket_id=f"unknown-{bucket}",
        basis=basis,
        signals=signals,
        origin_key=origin_key,
    )


# ── header 脱敏（G11.1）─────────────────────────────────────────────────────

#: 值必须脱敏的 header（小写）。凭证类一律不落盘、不进日志。
REDACTED_HEADERS: frozenset[str] = frozenset({
    "authorization", "proxy-authorization",
    "x-api-key", "api-key", "x-goog-api-key",
    "cookie", "set-cookie",
})

#: 值虽非凭证但每轮都变的 header —— 留名不留值，避免把噪声写进 Memory Hub。
_VOLATILE_HEADERS: frozenset[str] = frozenset({
    "date", "content-length", "traceparent", "tracestate",
})

_REDACTED_PLACEHOLDER = "<redacted>"
_VOLATILE_PLACEHOLDER = "<volatile>"

#: 单个 header 值留存上限（防超长 header——如巨型 cookie——撑爆 Memory Hub）。
#: 🔴 2026-08-29（Jason）：200 静默截断违背 Hub 原始留存原则——
#: `x-codex-turn-metadata` 的 JSON 被切在半截（MQ-A22 实害）。改 4096 +
#: 截断必须留显式标记（静默丢数据比丢数据本身更糟）。
_MAX_HEADER_VALUE_CHARS = 4096
_TRUNCATED_MARKER = "…<bladex:truncated>"


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """脱敏 header 快照，供日志与 Memory Hub 留存（G11.1 / MQ-A5）。

    留存的目的是**新 agent 第一次来就有据可查**（自愈），不是事前标定——本机没有那
    么多 agent 可测，且各家 header 会随版本漂移，靠预先枚举注定追不上
    （2026-08-18 Jason 拍板否掉"跑多 agent 标定"路径）。

    :param headers: 原始 header 字典。
    :returns: 新字典，键小写；凭证类值替换为 ``<redacted>``、易变类替换为
        ``<volatile>``、其余截断到 200 字符。**保留键名**——键名本身是识别信号
        （见 :func:`extract_vendor_segments`），丢了就等于丢了这次留存的意义。
    """
    out: dict[str, str] = {}
    for raw_name, raw_value in headers.items():
        name = raw_name.strip().lower()
        if name in REDACTED_HEADERS:
            out[name] = _REDACTED_PLACEHOLDER
        elif name in _VOLATILE_HEADERS:
            out[name] = _VOLATILE_PLACEHOLDER
        else:
            value = "" if raw_value is None else str(raw_value)
            if len(value) > _MAX_HEADER_VALUE_CHARS:
                value = value[:_MAX_HEADER_VALUE_CHARS] + _TRUNCATED_MARKER
            out[name] = value
    return out
