"""批 BD · 步骤 1：账本块「等长无关文本」填充（bench 专用，默认关）。

用途：BD 受控对照需要一个「末尾仍有一段、但内容与账本无关且逐轮不变」的臂，
用来把 glm 零缓存的成因分成「块内容」与「末尾多一段 / 账本工具定义在场」。
方案全文见 `docs/planning/report-batch-BD-precheck-20261117.md` §4。

本模块**只在** 账本填充开关 > 0 时被 `AgencyRuntime.insert_ledger_block` 用到。
默认 0 ⇒ 注入路径逐字节与改前相同。开关名只在 `bladex_core.flags` 出现一次（守卫 ③）；
本模块经 `bench_ledger_filler_chars()` 读取，不自己写开关名。
"""

from __future__ import annotations

import functools
import hashlib

import structlog
from bladex_core.flags import bench_ledger_filler_chars

logger = structlog.get_logger(__name__)

#: 与 `agency.runtime.LEDGER_BLOCK_OPEN/CLOSE` 同值（splice 与「剥掉上一轮标记块」认这对标记）。
#: 单测断言两处相等；本模块不 import runtime，避免循环依赖。
_OPEN = "<bladex-ledger>"
_CLOSE = "</bladex-ledger>"

#: 固定种子。改它 = 改填充文本 = 作废已跑的对照，所以带版本号。
_SEED = "bladex-bench-filler-v1"

#: 下限 = 两个标记 + 两个换行 + 至少 1 个正文字符。n 小于它按它算（日志里 `filler_chars` 报的是实际值）。
MIN_CHARS = len(_OPEN) + len(_CLOSE) + 3

#: 256 个中性词（河流/石头/季节/颜色/动作一类）。**禁用词表**见 `FORBIDDEN_WORDS`，用例逐词断言。
#: 任何一个词都不许以禁用词开头（否则等长截断会在词中间切出禁用词）——同样由用例断言。
WORDS: tuple[str, ...] = (
    "river stone autumn amber meadow whisper harbor violet pebble lantern willow copper "
    "breeze orchard velvet thunder marble canyon feather sunrise garden crystal maple "
    "island ribbon valley cobalt thistle puddle cinder walnut glacier pasture saffron "
    "hollow ember juniper trellis kettle drizzle quartz bramble lagoon cedar parchment "
    "dune orchid tundra mosaic fennel pine silver rain sparrow clover granite basil "
    "plum horizon pond fern coral ivory heron mist birch cloud wheat reef moss dawn "
    "lilac spruce brook hazel cliff nectar prairie rust sand teal vine yarn zephyr "
    "acorn badger candle dahlia eagle falcon gravel heather iris jasmine kelp lemon "
    "mango nutmeg olive pepper quail radish sorrel tulip umber walrus yellow "
    "zinnia apricot beacon cabin daisy elm fjord gale haven inlet jade knoll larch "
    "mesa nest oak plateau quill ridge shore tide urchin vale wharf yew zenith aspen "
    "bluff cove delta estuary fog grove hill isle jetty knot lake marsh narrows oasis "
    "peak quarry rapids summit trail upland vista weir yarrow zeal anchor bridge canal "
    "dock easel forge gable hammock ironwood jar kiln loom mill needle oar paddle quilt "
    "rope sail string umbrella vessel wagon yoke zither alder birchbark cypress dogwood "
    "elder fir gorse holly ivy juniperus kale linden mulberry nettle oat poplar quince "
    "rowan sage thyme upas verbena willowherb yucca azure bronze crimson denim emerald "
    "fawn golden hazy indigo khaki lavender mauve navy ochre peach rose scarlet tan "
    "ultramarine vermilion wheaten xanthic harvest blossom cascade driftwood evergreen firefly "
    "glimmer hillside icicle jewel kestrel lullaby moonlight nightfall outcrop petal quietude "
    "ripple snowfall twilight undergrowth waterfall sunbeam rainbow seashell starlight tidepool"
).split()

#: 账本 / 工具 / 任务相关的词。填充正文（小写后）不得包含其中任何一个子串。
FORBIDDEN_WORDS: tuple[str, ...] = (
    "ledger", "goal", "task", "verified", "open", "next", "core", "bladex", "tool",
    "user", "agent", "memory", "skill", "session", "switch", "update", "read", "write",
    "assistant", "system", "instruction", "rule", "must", "never", "todo", "done",
)


def _body(length: int) -> str:
    """固定种子的 sha256 流 → 逐字节取词 → 拼到恰好 `length` 个字符。纯函数，无任何外部状态。"""
    out: list[str] = []
    total = 0
    counter = 0
    while total - 1 < length:
        digest = hashlib.sha256(f"{_SEED}:{counter}".encode()).digest()
        counter += 1
        for b in digest:
            w = WORDS[b % len(WORDS)]
            out.append(w)
            total += len(w) + 1
            if total - 1 >= length:
                break
    text = " ".join(out)[:length]
    # 截断可能落在空格上：换成句点，保持恰好 length 个字符且不以空白收尾。
    return text[:-1] + "." if text.endswith(" ") else text


@functools.lru_cache(maxsize=16)
def bench_filler(n: int) -> str:
    """恰为 `max(n, MIN_CHARS)` 个字符的填充块，首尾是账本标记。

    同进程按 n 缓存；种子固定 ⇒ 跨重启也逐字节相同。**不含**轮次计数、会话、rev、时间。
    """
    total = max(int(n), MIN_CHARS)
    body_len = total - len(_OPEN) - len(_CLOSE) - 2
    return f"{_OPEN}\n{_body(body_len)}\n{_CLOSE}"


def warn_if_bench_filler_active() -> int:
    """启动时调用：开关 > 0 打一条 `bench_filler_active` warning。返回 n（0 = 关）。"""
    n = bench_ledger_filler_chars()
    if n > 0:
        logger.warning("bench_filler_active", chars=n,
                       effective_chars=max(n, MIN_CHARS),
                       note="BENCH ONLY: ledger block is replaced by constant filler text; "
                            "unset this switch for real use")
    return n
