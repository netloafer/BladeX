"""词法第二路：SQLite FTS5 真倒排（ADR-0028 E6.2）。

## 取代什么

`memory_index._keyword_facts` 是**全表 substring 扫描**：每次查询把所有 fact 拉出来，
逐条 `t in content.lower()`。复杂度 O(N × 全文长度)，随库线性劣化，
所以它一直挂着 `BLADEX_KEYWORD_CHANNEL` 默认关——等于词法通道从未真正上线。
E6.2 落地后，那个 flag 与 substring 实现一并删除。

## 为什么是 FTS5 + trigram

    tokenize = 'trigram'

CJK 不需要分词器即可子串检索（中文库的必要条件），标识符与报错串同样受益
——`bxe3663a38` 与 `bx3a984bcd` 的 cosine 是 **0.9890**（dense 通道对标识符是瞎的），
但在 trigram 倒排里它们是两个完全不同的 key。

排序用 FTS5 内置 **BM25**（Robertson & Zaragoza 2009），不自己发明打分。

## 单写者与 Memory Index 一致

写入点只有 consolidator：`add_fact` 时 INSERT、墓碑/取代（t_invalid）时 DELETE、
rebuild 时随库重建。file_ref 不入（与 E1.3 同边界——裸路径不该参与检索）。
读端只读，读不到就退化为"没有词法通道"，绝不影响 dense 主路径。
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_FILE_NAME = "fts.sqlite"
_TABLE = "facts_fts"

# ── 三段式 T2（2026-08-10）：schema v2 = 加 topic 列 ──────────────────────
# FTS5 虚表加列 = 建新表迁移（SQLite 不支持 ALTER 虚表）。打开时检测：
# 表存在但无 topic 列（sqlite_master 的建表 SQL 里没有 'topic'）且可写 →
# drop 重建 + PRAGMA user_version=2；行数归零后 `backfill_fts`（幂等、已有
# 自动入口 memory_index.py consolidation pass）把存量 fact 连 topic 一起灌回。
# 只读端遇旧 schema 不迁移（写权只在 consolidator）——检索照常工作（MATCH
# 是全表口径，不点名列），degraded 但不崩。
_SCHEMA_VERSION = 2
_TABLE_COLS = "fact_id UNINDEXED, content, entities, subject, attribute, topic"
# M0-5（复核 I1 🔴）：二字中文词的影子倒排表。
#
# 问题：trigram 分词器索引的是**三**元组，FTS5 对短于 3 字符的 MATCH 词项直接
# 返回空——于是「美伊」「伊朗」「油价」这类二字词在词法通道上恒零命中，
# 而中文里最具判别性的实体名恰恰大量是二字（国名简称、商品名、机构简称）。
# live 库有 15 条含"伊朗"的 fact，查"伊朗"一条都捞不到。
#
# 方案选型（任务卡要求"选实现代价小者并写明理由"）：
#   方案 B（查询侧对二字 CJK 降级 `LIKE '%词%'`）看似最省，但它 ①退回全表扫描
#     ——正是 E6.2 用真倒排替换掉的那个形态；②**产不出秩**，而下游 RRF 融合
#     只吃秩（`for fid, _rank in fts.search(...)` 里 rank 实际被丢弃、用的是位置），
#     没有秩就只能拍一个常数位置，等于往融合里灌噪声。
#   方案 A（bigram 影子表）选定：一张 `unicode61` 分词的 FTS5 表，写入时把 CJK
#     片段预切成空格分隔的二元组。二字词变成一个**普通词元**，BM25 照常工作、
#     秩天然可用；写侧多一次 INSERT，读侧只在 query 含 CJK 时多查一次。
#     trigram 表原样保留——标识符 phrase 精确匹配仍归它（`bxe3663a38` 不能被切）。
_BIGRAM_TABLE = "facts_bigram"

# 查询词元上限（超长 query 的尾部词元对 BM25 贡献极小，却让查询成本线性上涨）
_MAX_TERMS = 12
# 单次词法召回上限（与 dense 通道对称，供 RRF 融合）
DEFAULT_LIMIT = 50

_TOKEN_SPLIT = re.compile(r"[\s,./;:!?()（）\[\]{}\"'`，。、？！：；]+")


def _terms(query: str) -> list[str]:
    """切词元（中英文通吃：长度 ≥2 的片段；trigram 分词器负责真正的匹配）。"""
    out: list[str] = []
    for t in _TOKEN_SPLIT.split(query or ""):
        t = t.strip()
        if len(t) >= 2 and t not in out:
            out.append(t)
        if len(out) >= _MAX_TERMS:
            break
    return out


def _quote(term: str) -> str:
    """FTS5 字符串字面量（双引号转义），当作 phrase 查询。"""
    return '"' + term.replace('"', '""') + '"'


def _is_cjk(text: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in text)


_CJK_RUN = re.compile(r"[一-鿿]{2,}")
# 单条 fact 产出的二元组上限（防超长 source_text 把影子表撑爆；
# 事实内容本身很短，这个上限实际几乎不触发）。
_MAX_BIGRAMS = 400


def _bigrams(text: str) -> str:
    """把文本里的 CJK 连续片段切成空格分隔的二元组（M0-5 影子索引的写侧）。

    只切 CJK：拉丁词与标识符由 trigram 表负责，重复索引一遍只会让表变大、
    并让同一条 fact 在两路里都命中而虚高它的融合秩。

    `美伊局势紧张` → `美伊 伊局 局势 势紧 紧张`
    """
    out: list[str] = []
    for run in _CJK_RUN.findall(text or ""):
        for i in range(len(run) - 1):
            out.append(run[i:i + 2])
            if len(out) >= _MAX_BIGRAMS:
                return " ".join(out)
    return " ".join(out)


def _bigram_limit() -> int:
    """影子表返回条数上限。**默认 0 = 关闭这条路**（见 flags.BLADEX_FTS_BIGRAM_LIMIT）。

    完整来历与四次实测数据在 `bladex_core/flags.py` 那条注释里——一句话：
    "二字词在 trigram 上零命中"为真，但"补一条 bigram 路能改善端到端检索"
    **被评估集否掉了**（三种接法全是净负）。机制保留、默认关闭，
    等 MS-9 给词法通道独立权重后再评估。
    """
    from bladex_core.flags import flag_number

    return max(0, int(flag_number("BLADEX_FTS_BIGRAM_LIMIT")))


def _cjk_query_grams(term: str) -> list[str]:
    """把 CJK 查询词切成二元组查询项，与写侧口径一致。

    长词照常展开——中文没有空格，`_terms` 切出来的是整个短语，
    不展开的话长 query 里的 2 字实体永远找不到（见 `_BIGRAM_LIMIT_DEFAULT` 上的记录）。
    噪声由**返回条数封顶**控制，不由这里控制。
    """
    if len(term) < 2:
        return []
    if len(term) == 2:
        return [term]
    grams = [term[i:i + 2] for i in range(len(term) - 1)]
    if len(grams) <= _MAX_CJK_GRAMS:
        return grams
    # 超长：均匀取样保首尾（首尾往往最具判别性）
    step = len(grams) / _MAX_CJK_GRAMS
    picked = [grams[int(i * step)] for i in range(_MAX_CJK_GRAMS)]
    if grams[-1] not in picked:
        picked[-1] = grams[-1]
    return picked


# CJK 长词切成三元组查询时的上限（防单个长句炸出几十个 OR 项）
_MAX_CJK_GRAMS = 6


def _expand_cjk(term: str) -> list[str]:
    """把 CJK 长词展开成三元组查询项。

    为什么需要（真实失败例）：中文没有空格，"泰山啤酒破产案" 会被切成**一个**词元；
    当作 phrase 查全串，只能匹配到逐字包含这七个字的文档——而库里那条是
    "关于泰山啤酒的记录"，明明高度相关却一个都不命中。

    trigram 分词器索引的是三元组，所以查询侧也按三元组来：
    "泰山啤酒破产案" → 泰山啤 / 山啤酒 / 啤酒破 / 酒破产 / 破产案，
    "关于泰山啤酒的记录" 命中前两个 → BM25 排上来。

    ≤4 字的短词保持整体 phrase（够短，本来就能整体匹配，切开只会引噪声）。
    """
    if len(term) <= 4:
        return [term]
    grams = [term[i:i + 3] for i in range(len(term) - 2)]
    if len(grams) <= _MAX_CJK_GRAMS:
        return grams
    # 超长：均匀取样，保首尾（首尾往往是最具判别性的部分）
    step = len(grams) / _MAX_CJK_GRAMS
    picked = [grams[int(i * step)] for i in range(_MAX_CJK_GRAMS)]
    if grams[-1] not in picked:
        picked[-1] = grams[-1]
    return picked


def _merge_by_rank(
    a: list[tuple[str, float]], b: list[tuple[str, float]], limit: int,
) -> list[tuple[str, float]]:
    """合并两路词法结果：**trigram 在前，bigram 只补它没捞到的**（M0-5）。

    🔴 三次实测才收敛到这个语义，前两次都错在不同地方，值得完整记下来：

    **第一版：按最好名次交错合并（`min(pos_tri, pos_bi)`）。**
    净退步（总分 recall@10 0.430 → 0.413）。当时归因为"bigram 候选太多把通道灌满"，
    但 A/B 实测推翻了这个解释：**bigram 关掉时词法路已经有 19/39 条 query 顶到
    50 条上限**（那是 trigram 自己的），开到 50 也只多 4 条。量不是问题。

    真正的机制是**秩交错**：两路按名次对等合并，等于给了它们**相同的权重**。
    bigram 路是 2-gram 的 OR 匹配（`泰山啤酒` → `泰山/山啤/啤酒`），精度天然低得多，
    却能凭 `pos=1,2,3…` 挤进合并结果的头部，把 trigram 的精确命中顶下去。
    下游 RRF 只吃位置，于是这些低精度候选直接变成高权重信号。

    **第二版：收窄到只服务 len<3 的词元。** 分数逐字节退回未修状态——中文没有空格，
    `_terms` 切出来的是整个短语，几乎从不是 2 字词元，机制成了死代码。

    **现在的语义**：trigram 结果**原序在前**，bigram 只把 trigram 没返回过的
    补在**后面**（数量另有上限 `_bigram_limit()`）。这样：
      - 二字词（trigram 物理上匹配不到）仍能进候选集——本卡要修的洞是修好的；
      - 但它们永远排在精确匹配之后，不可能挤掉 trigram 的头部。
    两路的优先级差异是**结构性**的，不靠调参维持。
    """
    seen = {fid for fid, _ in a}
    out = list(a)
    for fid, score in b:
        if fid in seen:
            continue
        seen.add(fid)
        out.append((fid, score))
    return out[:limit]


class FtsIndex:
    """Memory Index 的词法倒排索引（SQLite FTS5 / trigram）。

    与 MemoryIndex 同目录、同生命周期；打不开就整体禁用（degraded 但不崩）。
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self._path = Path(path) / _FILE_NAME
        self._read_only = read_only
        self._db: sqlite3.Connection | None = None
        self._available = False

    # ── 生命周期 ──

    def open(self) -> bool:
        if self._db is not None:
            return self._available
        try:
            if self._read_only and not self._path.exists():
                # 读端遇到空库：不创建文件（写权只有 consolidator），静默禁用
                self._available = False
                return False
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self._path), check_same_thread=False)
            self._migrate_schema_if_needed()
            self._db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {_TABLE} USING fts5("
                f"{_TABLE_COLS}, tokenize='trigram')"
            )
            # M0-5：二字中文词的影子倒排（unicode61 分词 + 预切二元组）
            self._db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {_BIGRAM_TABLE} USING fts5("
                "fact_id UNINDEXED, grams, tokenize='unicode61')"
            )
            if not self._read_only:
                self._db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._db.commit()
            self._available = True
            logger.info("fts_opened", path=str(self._path), read_only=self._read_only)
        except Exception as e:  # noqa: BLE001 —— 词法通道不可用不该打死检索
            logger.warning("fts_unavailable", path=str(self._path), error=str(e))
            self._db = None
            self._available = False
        return self._available

    def _migrate_schema_if_needed(self) -> None:
        """T2：旧 schema（无 topic 列）自动迁移（可写端；只读端不动）。

        判据 = sqlite_master 里 facts_fts 的建表 SQL 不含 'topic'。
        drop 后表由 open() 里的 CREATE 以 v2 列集重建；行数归零 →
        下一次 `backfill_fts`（自动入口已存在）把存量 fact 灌回。
        """
        try:
            row = self._db.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (_TABLE,)
            ).fetchone()
        except Exception as e:  # noqa: BLE001
            logger.warning("fts_schema_probe_failed", error=str(e))
            return
        if row is None or not row[0] or "topic" in row[0]:
            return          # 表不存在（将按 v2 建）或已是 v2
        if self._read_only:
            logger.warning(
                "fts_schema_stale_readonly",
                hint="old schema (no topic column); consolidator will migrate")
            return
        try:
            self._db.execute(f"DROP TABLE IF EXISTS {_TABLE}")
            self._db.execute(f"DROP TABLE IF EXISTS {_BIGRAM_TABLE}")
            self._db.commit()
            logger.info("fts_schema_migrated", to_version=_SCHEMA_VERSION,
                        hint="tables recreated; backfill_fts will repopulate")
        except Exception as e:  # noqa: BLE001
            logger.warning("fts_schema_migrate_failed", error=str(e))

    @property
    def available(self) -> bool:
        return self._available and self._db is not None

    def close(self) -> None:
        if self._db is not None:
            try:
                self._db.close()
            finally:
                self._db = None
                self._available = False

    # ── 写（单写者 = consolidator）──

    def upsert(self, fact: Any) -> None:  # noqa: ANN401  Fact
        """写一条（先删后插 = upsert）。file_ref 不入（E1.3 同边界）。"""
        if not self.available or self._read_only:
            return
        kind = getattr(fact.item_kind, "value", str(fact.item_kind))
        if kind == "file_ref":
            return
        # 被取代/失效的条目退出词法召回（与 dense 的 current-only 同口径）
        if getattr(fact, "t_invalid", None) is not None:
            self.delete(fact.id)
            return
        try:
            entities = " ".join(str(e) for e in (fact.entities or []))
            subject = getattr(fact, "subject", "") or ""
            attribute = getattr(fact, "attribute", "") or ""
            topic = getattr(fact, "topic", "") or ""     # T2：轮级主题进词法面
            self._db.execute(f"DELETE FROM {_TABLE} WHERE fact_id = ?", (fact.id,))
            self._db.execute(
                f"INSERT INTO {_TABLE} (fact_id, content, entities, subject, attribute, topic) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (fact.id, fact.content or "", entities, subject, attribute, topic),
            )
            # M0-5：影子二元组表同步写（同一个事务，两表永远同进同退）
            # T2：topic 文本并入 grams（二字中文主题词可经影子表命中）
            grams = _bigrams(
                " ".join([fact.content or "", entities, subject, attribute, topic])
            )
            self._db.execute(
                f"DELETE FROM {_BIGRAM_TABLE} WHERE fact_id = ?", (fact.id,))
            if grams:
                self._db.execute(
                    f"INSERT INTO {_BIGRAM_TABLE} (fact_id, grams) VALUES (?, ?)",
                    (fact.id, grams),
                )
            self._db.commit()
        except Exception as e:  # noqa: BLE001
            logger.warning("fts_upsert_failed", fact_id=fact.id, error=str(e))

    def delete(self, fact_id: str) -> None:
        if not self.available or self._read_only:
            return
        try:
            self._db.execute(f"DELETE FROM {_TABLE} WHERE fact_id = ?", (fact_id,))
            self._db.execute(
                f"DELETE FROM {_BIGRAM_TABLE} WHERE fact_id = ?", (fact_id,))
            self._db.commit()
        except Exception as e:  # noqa: BLE001
            logger.warning("fts_delete_failed", fact_id=fact_id, error=str(e))

    def clear(self) -> None:
        if not self.available or self._read_only:
            return
        try:
            self._db.execute(f"DELETE FROM {_TABLE}")
            self._db.execute(f"DELETE FROM {_BIGRAM_TABLE}")
            self._db.commit()
        except Exception as e:  # noqa: BLE001
            logger.warning("fts_clear_failed", error=str(e))

    def count(self) -> int:
        return self._count_of(_TABLE)

    def bigram_count(self) -> int:
        """影子二元组表行数（M0-5 回填判据：两表任一缺行都要补）。"""
        return self._count_of(_BIGRAM_TABLE)

    def _count_of(self, table: str) -> int:
        if not self.available:
            return 0
        try:
            return int(self._db.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        except Exception:  # noqa: BLE001
            return 0

    def drop_and_recreate(self) -> bool:
        """删表重建（M0-5 要求的 drop+rebuild 入口）。

        什么时候需要：分词方案变了（比如本卡新增影子二元组表），存量索引的
        内容仍是旧口径 —— 靠 `upsert` 增量补不齐，必须整表重来。
        调用方随后应跑 `backfill_fts(force=True)` 把 fact 重新灌进去。
        """
        if not self.available or self._read_only:
            return False
        try:
            self._db.execute(f"DROP TABLE IF EXISTS {_TABLE}")
            self._db.execute(f"DROP TABLE IF EXISTS {_BIGRAM_TABLE}")
            self._db.execute(
                f"CREATE VIRTUAL TABLE {_TABLE} USING fts5("
                f"{_TABLE_COLS}, tokenize='trigram')"
            )
            self._db.execute(
                f"CREATE VIRTUAL TABLE {_BIGRAM_TABLE} USING fts5("
                "fact_id UNINDEXED, grams, tokenize='unicode61')"
            )
            self._db.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._db.commit()
            logger.info("fts_dropped_and_recreated", path=str(self._path))
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("fts_drop_failed", error=str(e))
            return False

    # ── 读 ──

    def search(
        self, query: str, *, identifiers: list[str] | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> list[tuple[str, float]]:
        """BM25 检索，返回 [(fact_id, bm25_rank)]，rank 越小越相关（FTS5 语义）。

        查询构造（任务卡 E6.2）：
            Q_id 各词组成 phrase 查询  OR  剩余词元（≤12 个）的 OR 查询

        标识符走 phrase 是关键：`bxe3663a38` 必须整体匹配，
        不能被拆成三元组之后与 `bx3a984bcd` 混在一起。
        """
        if not self.available:
            return []
        terms = _terms(query)
        # 标识符恒走整体 phrase：`bxe3663a38` 必须精确匹配，
        # 切开就等于把它和 `bx3a984bcd` 混回一起（那正是 dense 通道的毛病）。
        parts = [_quote(t) for t in (identifiers or [])]
        for t in terms:
            for piece in (_expand_cjk(t) if _is_cjk(t) else [t]):
                # trigram 表对 <3 字符的词项恒零命中（FTS5 语义），
                # 塞进去只会白跑一次查询——那部分交给影子二元组表。
                if len(piece) < 3:
                    continue
                q = _quote(piece)
                if q not in parts:
                    parts.append(q)

        tri_rows = self._match(_TABLE, parts, limit) if parts else []

        # M0-5：CJK 词项另走影子二元组表（二字词在这里是一个普通词元）。
        # **返回条数单独封顶**——词法路只该补充 dense 通道、不该主导它。
        bi_parts: list[str] = []
        for t in terms:
            if not _is_cjk(t):
                continue
            for gram in _cjk_query_grams(t):
                q = _quote(gram)
                if q not in bi_parts:
                    bi_parts.append(q)
        bi_limit = min(_bigram_limit(), limit)
        bi_rows = (self._match(_BIGRAM_TABLE, bi_parts, bi_limit)
                   if bi_parts and bi_limit > 0 else [])

        if not bi_rows:
            return tri_rows
        if not tri_rows:
            return bi_rows
        return _merge_by_rank(tri_rows, bi_rows, limit)

    def _match(
        self, table: str, parts: list[str], limit: int,
    ) -> list[tuple[str, float]]:
        """在指定表上跑一次 BM25 OR 查询；出错返回空（词法路失败不打死检索）。"""
        if not parts:
            return []
        expr = " OR ".join(parts)
        try:
            rows = self._db.execute(
                f"SELECT fact_id, bm25({table}) AS rank FROM {table} "
                f"WHERE {table} MATCH ? ORDER BY rank LIMIT ?",
                (expr, limit),
            ).fetchall()
        except Exception as e:  # noqa: BLE001 —— 查询语法/损坏不该打死检索
            logger.warning("fts_search_failed", table=table, error=str(e),
                           terms=len(parts))
            return []
        return [(r[0], float(r[1])) for r in rows]
