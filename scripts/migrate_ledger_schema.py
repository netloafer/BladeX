"""Memory Hub schema 统一迁移 —— ADR-0026 §2.3 + ADR-0025 §5 / U2（离线重写一次到位）。

合并三来源 schema 变更为**一次**离线迁移：
  1. ADR-0026：key `user/agent/session/{entry_id}` → `principal/agent/session/{entry_id}`
     （个人模式 principal == user_id，key **字节一致**，语义零变化）；
     （Turn 曾追加 `api_key_id` / `org_id` 审计字段，2026-09-06 F0.3 H1 销账已删）。
  2. ADR-0025 T1：Turn 追加 `reconstruction`（历史默认 None）。
  3. `request_messages` 语义不变（恒为原始）—— 🔴 U2 红线（I1/I5 schema 层保险）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
为什么是「逐 key 原样拷贝」而不是「反序列化再写」：
  - Memory Hub 里 Turn 存的是**去重后**形态（content 被 content_ref 占位，正文在 __msg__/ 池）。
    走 MemoryHub.put 会重跑 dedup + prefix 检测，改变去重结构、破坏重建等价性。
  - 新增字段全部**追加式带默认**，旧数据经 Pydantic `model_validate` 读出时自动
    materialize（reconstruction=None）—— 无需物理改写正文。
  - 故迁移 = **raw bytes 逐 key 拷贝**（保 dedup 结构 + 全部内部 key），
    个人模式 key 不变；企业模式经 --principal-map 重写 Turn key 首段（legacy→principal）。
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

本机跑：

  # 个人模式（principal == user_id，key 字节一致；本质是「验证 + 新库落地」）：
  .venv/bin/python scripts/migrate_ledger_schema.py \
      --src data/bladex_hub --dst data/bladex_hub_v0026

  # 企业模式（多 legacy key → 一 principal，重写 Turn key 首段）：
  .venv/bin/python scripts/migrate_ledger_schema.py \
      --src data/bladex_hub --dst data/bladex_hub_v0026 \
      --principal-map legacy_a=alice --principal-map legacy_b=alice

  # 校验（默认自动跑）：计数逐条一致 + 抽样反序列化新字段 + 内部 key 全数保留。

迁移后：原库归档不删（--src 只读打开）；切换前用 eval_reconstruction / rebuild 复核等价性。
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("packages/bladex-core", "packages/bladex-proxy"):
    _full = os.path.join(_ROOT, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

import msgpack  # noqa: E402
import rocksdict  # noqa: E402
from bladex_proxy.models import Turn  # noqa: E402
from bladex_proxy.storage.memory_hub import _is_internal_key  # noqa: E402


def _decode(k: bytes | str) -> str:
    return k.decode() if isinstance(k, bytes) else str(k)


def _rewrite_turn_key(key: str, principal_map: dict[str, str]) -> str:
    """把 Turn key 的首段（旧 user/principal）按映射改写。

    key 格式：`{principal}/{agent}/{session}/{entry_id}`。只改首段，其余原样。
    映射缺省（个人模式）→ 原样返回，key 字节一致。
    """
    if not principal_map:
        return key
    head, sep, rest = key.partition("/")
    if not sep:
        return key
    return f"{principal_map.get(head, head)}{sep}{rest}"


def migrate(src: str, dst: str, principal_map: dict[str, str]) -> dict[str, int]:
    """逐 key 拷贝 src → dst，Turn key 按 principal_map 重写首段。返回计数字典。"""
    if os.path.exists(dst) and os.listdir(dst):
        raise SystemExit(f"目标目录非空，拒绝覆盖：{dst}（请指向全新目录）")
    os.makedirs(dst, exist_ok=True)

    # src 只读打开（归档不删；read_only 不取 LOCK 亦可与运行中 proxy 共存）
    src_db = rocksdict.Rdict(
        src, rocksdict.Options(), None, rocksdict.AccessType.read_only()
    )
    dst_db = rocksdict.Rdict(dst)

    counts = {"turn": 0, "internal": 0, "turn_key_rewritten": 0}
    try:
        for k_raw, v_raw in src_db.items():
            k = _decode(k_raw)
            if _is_internal_key(k):
                # 内部 key（__msg__ / tombstone / admin_event / distill / judgment / annotation）
                # 原样拷贝 —— 包括 __msg__ 内容池，保 content_ref 一致性。
                dst_db[k_raw] = v_raw
                counts["internal"] += 1
                continue
            new_k = _rewrite_turn_key(k, principal_map)
            if new_k != k:
                counts["turn_key_rewritten"] += 1
            dst_db[new_k.encode()] = v_raw  # 值原样（去重结构不动）
            counts["turn"] += 1
    finally:
        src_db.close()
        dst_db.close()
    return counts


def verify(src: str, dst: str, principal_map: dict[str, str], sample: int) -> None:
    """校验：计数逐条一致 + 抽样反序列化确认新字段 materialize + request_messages 不变。"""
    src_db = rocksdict.Rdict(src, rocksdict.Options(), None, rocksdict.AccessType.read_only())
    dst_db = rocksdict.Rdict(dst, rocksdict.Options(), None, rocksdict.AccessType.read_only())
    try:
        src_turn = src_internal = 0
        src_turn_vals: dict[str, bytes] = {}
        for k_raw, v_raw in src_db.items():
            k = _decode(k_raw)
            if _is_internal_key(k):
                src_internal += 1
            else:
                src_turn += 1
                if len(src_turn_vals) < sample:
                    src_turn_vals[_rewrite_turn_key(k, principal_map)] = bytes(v_raw)

        dst_turn = dst_internal = 0
        for k_raw in dst_db.keys():
            if _is_internal_key(_decode(k_raw)):
                dst_internal += 1
            else:
                dst_turn += 1

        assert src_turn == dst_turn, f"Turn 计数不一致 src={src_turn} dst={dst_turn}"
        assert src_internal == dst_internal, f"内部 key 计数不一致 src={src_internal} dst={dst_internal}"

        # 抽样：新库能反序列化 + 新字段存在 + request_messages 与源逐字一致
        checked = 0
        for new_k, src_val in src_turn_vals.items():
            dst_val = dst_db.get(new_k.encode())
            assert dst_val is not None, f"迁移后缺 key：{new_k}"
            src_data = msgpack.unpackb(src_val, raw=False)
            dst_data = msgpack.unpackb(bytes(dst_val), raw=False)
            # 值 raw 拷贝 → 字节一致（含 request_messages 原始，I1/I5 红线）
            assert src_data.get("request_messages") == dst_data.get("request_messages"), (
                f"request_messages 被改写：{new_k}"
            )
            turn = Turn.model_validate(dst_data)  # 新 schema 校验通过
            # 新字段 materialize（个人模式历史数据为默认空/None）
            assert turn.reconstruction is None or bool(turn.reconstruction.layer)
            checked += 1
        print(
            f"✓ 校验通过：Turn {src_turn} 条 / 内部 key {src_internal} 条计数一致；"
            f"抽样 {checked} 条新 schema 反序列化 + request_messages 逐字一致"
        )
    finally:
        src_db.close()
        dst_db.close()


def _parse_map(pairs: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"--principal-map 格式应为 legacy=principal，收到：{p}")
        legacy, principal = p.split("=", 1)
        out[legacy.strip()] = principal.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Memory Hub schema 统一迁移（ADR-0026/0025 / U2）")
    ap.add_argument("--src", required=True, help="旧 Memory Hub RocksDB 目录（只读，归档不删）")
    ap.add_argument("--dst", required=True, help="新 Memory Hub RocksDB 目录（必须全新/空）")
    ap.add_argument(
        "--principal-map",
        action="append",
        default=[],
        metavar="legacy=principal",
        help="企业模式 legacy_id→principal_id 映射（可重复；个人模式留空 = key 字节一致）",
    )
    ap.add_argument("--sample", type=int, default=50, help="校验抽样条数")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    pmap = _parse_map(args.principal_map)
    counts = migrate(args.src, args.dst, pmap)
    print(
        f"迁移完成：Turn {counts['turn']} 条（重写首段 {counts['turn_key_rewritten']} 条）"
        f" + 内部 key {counts['internal']} 条 → {args.dst}"
    )
    if not args.no_verify:
        verify(args.src, args.dst, pmap, args.sample)
    print("原库保留不删（归档）；切换 proxy/consolidator 前请再跑 eval_reconstruction 复核等价性。")


if __name__ == "__main__":
    main()
