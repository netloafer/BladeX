"""对话延续信号 —— 任务身份的载体是延续关系，不是标题相似度（GM-1，ADR-0024 §4.5 邻接）。

立卡 `docs/planning/memory-quality-tasks-20260815.md` §GM-1 形态改判；
台账 `memory-quality-problem-ledger-20260815.md` MQ-S13。

## 为什么不走标题

到 2026-08-18 为止 Matter 收敛一直走"标签相似度"：每轮各自产一个提案标题，事后拿
键 / 向量 / LLM 去调和。三轮读数一路在否它——开卡那一刻种子键达标率只有 **3.8%**，
用累积键算出的 30.8% 是乐观界且逐条看含明显误命中
（`litellm max_completion_tokens` ⇒ `泰山啤酒共益债`）。

**这不是参数问题：标题是每轮独立生成、措辞随机的产物**，拿它当身份载体，后面加多少
层匹配都是补救。任务身份的真实载体是**对话的延续关系**——「你再复核一下」「把 P1
详细说明」在结构上就是同一件事的连续追问。

## 判据（两条，缺一不可）

    ① 延续     turn.identity.session_id_source == TAIL_CONTINUATION 且与上一轮同 session
    ② 无新主体  本轮 user 文本里，"上一轮（user + assistant 回复）完全没出现过的
               有效主题信号"少于阈值

🔴 **①几乎每轮都命中**（agent 每轮重发整段对话），所以承重的是②。
把①单独当判据 = 把整个 session 合成一张卡，而 session 里话题交叉是常态
（更何况 `session_id` 实际是跨天的指纹桶，MQ-S9）。

🔴 ②的参照集**必须包含上一轮的 assistant 回复**：「把 P1 详细说明」里的 `P1` 是上一轮
**回复里**提出的，不是新主体。只比 user↔user 会把追问误判成新任务。

## 🔴 这个信号只进候选，不做判据

满足两条 ⇒ 把上一轮 fact 所属的 Matter **加进候选池**，最终仍由 L4 裁决、判 same 才继承。
2026-08-18 一天内这条纪律被撞到五次（共轮精度 50% ／ 键强匹配须送 L4 ／ 同 session
最大牵 47 张卡 ／ 按键反查泛词 `bladex` 一键 57 卡 ／ 本项）。这样把"参照集吞噬"那类
假收益（`qwen3.8 部署` ⇒ `张开利集资`）的代价从**误合并**降级为**多送一次裁决**。

## 与 topic_keys 的口径区别（不许混用）

生产 `derive_topic_keys` 走 `_SPLIT`，只切空白与 ASCII 标点、**中文不分词**——
「消息断掉了吗?」整句会变成**一个**键，与上一轮永远无交集，于是**任何中文追问都会被
判成"引入了新主体"**（探针首跑实测：新主体键数最大 409、中位 0，正是这个形态）。

做主体新颖度判定必须避开它，故本模块对长 CJK 串按 **2-gram 展开**。这是**新颖度判定
专用口径**，不改变 `topic_keys` 的任何行为，也不产生任何进库的键。

## 🔴 口径的已知局限（2026-08-18 实测，登记 MQ-S13 附录，本轮**不修**）

把探针那份实现搬进生产时逐条量过，三条局限是真实存在的，写在这里免得下一棒
把它们当 bug 重新发现一遍：

| # | 形态 | 实测 |
|---|---|---|
| A | 短 ASCII 标识符被 `_MIN_ASCII=3` 丢掉 | `P1` / `T3` 不成键——卡面举的「把 P1 详细说明」这个例子**不是靠 P1 本身**判对的，靠的是周围 CJK bigram 与上一轮的重叠 |
| B | 2–4 字 CJK 词整串成键，≥5 字才拆 bigram | `设计使然、` 在上一轮是整串键，在本轮长句里是 bigram → **同一个词跨不过去**。「设计使然这一支再展开」实测 7 个新主体（判为换话题），是漏合的一大来源 |
| C | 跨词边界的 bigram 是噪声 | 「你再复核一下」切出 `再复` / `核一`，实测 3 个新主体、恰好卡在默认门槛上 |

**为什么不顺手修**：这三条正是探针取那份读数时的口径。改了它，
「阈值 1→12 收益 14.9%→18.0% vs 风险 6.5%→2.3%」这份**立项依据就不再适用于新口径**，
得重测一遍才能说方向还成立。改口径与改机制是两件事，不在同一轮里做
（教训 51 的反面：尺子和被测对象一起动，读数就没有意义了）。
它们的代价方向也是安全的——一律让机制**更保守**（该继承的没进候选），
不会造成误合并。

实测参照（上一轮 = 真实长度的 user + assistant 回复）：
`继续`=1、`碎片化的成因`=0、`注入面存在空转`=0 → 进候选；
换话题 `qwen3.8 部署`=10、`泰山啤酒优先债权人`=21 → 不进。两档分离得很开。

设计约束：**叶子模块**，只依赖 `topic_keys` / `envelope` 两个同层叶子。
"""

from __future__ import annotations

import re

from .envelope import strip_envelopes
from .topic_keys import _SPLIT, is_valid_topic_key, normalize_key

#: 「有新主体」的默认门槛：本轮相对上一轮新增的主体信号数 ≥ 该值 → 判定引入了新任务。
#:
#: 🔴 **这个值不承重**：延续信号只往候选池里加卡，最终由 L4 裁决。探针扫过 1→12
#: 全程收益列稳定高于风险列、两列不交叉（14.9%→18.0% vs 6.5%→2.3%），所以结论不是
#: 阈值的产物。取 3 = 探针呈现那份 2×2 所用的值（换个数就对不上那份读数）；
#: 调它只改"多送几次裁决"的量。
#:
#: 配合上面局限 B/C 看：3 是**偏紧**的一档（`你再复核一下` 实测恰好 3、被判为换话题）。
#: 偏紧的代价是漏合，方向与误合并=0 一致，故不急着放宽——要放也得先有读数。
DEFAULT_NEW_SUBJECT_MAX = 3

#: CJK 虚词字：2-gram 里两个字都落在这里 = 没有主体性（「一下」「你再」「是谁」）。
#: 不滤的话，真正的延续追问（「发现的问题和风险你再复核一下」）会因为切出一堆虚词
#: bigram 而被判成"引入新主体"。
_CJK_FUNCTION_CHARS = frozenset(
    "的了在是和与并把被这那个我你他她它们吗呢吧啊呀就都也还很更再又"
    "下上出来去有没不会要能可以着过给对从向到为所之其如果因为但是"
    "一二三两多少几什么怎样如何时候现在已经还是或者以及然后而且"
)

#: 长 CJK 串（≥5 字）才做 2-gram 展开；短串本来就够细。
_CJK_RUN = re.compile(r"[一-鿿]{5,}")


def subject_signals(text: str) -> set[str]:
    """从自由文本确定性抽"主体"信号（新颖度判定专用口径，零 LLM）。

    ASCII/短串走 `topic_keys` 的准入（毒词/纯数字/长度门），长 CJK 串按 2-gram 展开
    并滤掉纯虚词 bigram。

    🔴 长 CJK 串**只留 2-gram、不留整串**：整串对任何措辞不同的句子都必然"新"，
    留着等于没修（探针首版实测「消息断掉了吗？」整串仍被判新主体）。
    """
    clean = strip_envelopes(text or "")[0]
    out: set[str] = set()
    for tok in _SPLIT.split(clean):
        k = normalize_key(tok)
        if not k:
            continue
        runs = _CJK_RUN.findall(k)
        if runs:
            for run in runs:
                for i in range(len(run) - 1):
                    bigram = run[i:i + 2]
                    if all(c in _CJK_FUNCTION_CHARS for c in bigram):
                        continue
                    out.add(bigram)
            continue
        if is_valid_topic_key(k):
            out.add(k)
    return out


def count_new_subjects(current_text: str, reference_text: str) -> int:
    """本轮相对参照集新增的主体信号数。

    `reference_text` 应为上一轮的 **user 文本 + assistant 回复**拼接
    （见模块 docstring：只比 user↔user 会把追问误判成新任务）。
    """
    return len(subject_signals(current_text) - subject_signals(reference_text))


def is_task_continuation(
    *,
    session_id_source: str,
    current_text: str,
    reference_text: str,
    new_subject_max: int = DEFAULT_NEW_SUBJECT_MAX,
) -> tuple[bool, int]:
    """本轮是否为上一轮任务的延续（两条判据同时成立）。

    返回 `(是否延续, 新主体信号数)`——第二个值供日志/探针读数，**不要拿它当判据之外
    的用途**。

    `session_id_source` 取 `Turn.identity.session_id_source`，按子串包含 `tail` 判定
    （枚举值 `tail_continuation`），兼容枚举与裸字符串两种传入形态。
    调用方负责保证 `reference_text` 来自**同一 session 的上一轮**、且两端都不是 aux 轮。
    """
    if "tail" not in str(session_id_source or "").lower():
        return False, -1
    n_new = count_new_subjects(current_text, reference_text)
    return n_new < new_subject_max, n_new
