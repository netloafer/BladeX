"""内容寻址副本存储（ADR-0028 E7.1 增补，2026-08-05 拍板）。

## 为什么只给"模型产出的文件"存副本

文件索引有两类来源（`file_index.FileOrigin`）：

- **会话里提到/读到的文件**（MENTIONED / READ）：正文在磁盘上本来就有，
  原文件还躺在那儿——再存一份是冗余；
- **模型产出的文件**（PRODUCED）：那份内容**只存在于那一次回复里**。
  不落盘，"当时它给我写的那版长什么样"就永远找不回来了。索引里的摘要
  只够检索，回溯不了正文。

所以本模块只服务 PRODUCED：把正文按 sha256 落盘，索引条目用 `blob_ref` 关联。

## 边界（都是拍板项）

- **内容寻址**：同一份内容只存一份；不同条目可以共享同一个 blob；
- **跟随 forget 级联删**：删索引条目时，若没有别的条目再引用该 blob 就一并删
  （与 ADR-0012 墓碑语义一致——"删除即删除"，不留影子副本）；
- **不进快照**：快照是 JSONL 实体流、从不拷目录，所以天然不带 blob；
  换机后副本重新长出来。有测试钉住这条，别顺手加进去。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

logger = structlog.get_logger()

#: 单个 blob 的体积上限（超出不存——副本是给人回溯的，不是备份系统）
MAX_BLOB_BYTES = 1_000_000


def blob_id_for(content: str) -> str:
    """blob id = sha256 全长十六进制（内容寻址的键）。"""
    return hashlib.sha256((content or "").encode("utf-8", "ignore")).hexdigest()


class BlobStore:
    """`<index_dir>/blobs/<id[:2]>/<id>` 的内容寻址文件存储。

    分两级目录是为了别让单目录塞进上万个文件（某些文件系统上会很慢）。
    """

    def __init__(self, root: str | Path, *, read_only: bool = False) -> None:
        self._root = Path(root) / "blobs"
        self._read_only = read_only

    @property
    def root(self) -> Path:
        return self._root

    def _path_of(self, blob_id: str) -> Path:
        return self._root / blob_id[:2] / blob_id

    def put(self, content: str) -> str:
        """写入并返回 blob_id。已存在则直接返回（内容寻址天然幂等）。

        read_only / 超限 / 写失败 → 返回空串，调用方按"没有副本"处理（不抛）。
        """
        if self._read_only or not content:
            return ""
        data = content.encode("utf-8", "ignore")
        if len(data) > MAX_BLOB_BYTES:
            logger.info("blob_skipped_too_large", bytes=len(data), cap=MAX_BLOB_BYTES)
            return ""
        blob_id = blob_id_for(content)
        path = self._path_of(blob_id)
        if path.exists():
            return blob_id
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # 先写临时文件再 rename：崩在中间不会留下半截 blob
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        except Exception as e:  # noqa: BLE001 —— 存副本失败不该影响索引主流程
            logger.warning("blob_put_failed", blob_id=blob_id[:12], error=str(e))
            return ""
        return blob_id

    def get(self, blob_id: str) -> str | None:
        """读回正文。不存在返回 None。"""
        if not blob_id:
            return None
        path = self._path_of(blob_id)
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:  # noqa: BLE001
            logger.warning("blob_get_failed", blob_id=blob_id[:12], error=str(e))
            return None

    def exists(self, blob_id: str) -> bool:
        return bool(blob_id) and self._path_of(blob_id).is_file()

    def delete(self, blob_id: str) -> bool:
        """删除一个 blob（调用方须先确认没有别的索引条目还引用它）。"""
        if self._read_only or not blob_id:
            return False
        path = self._path_of(blob_id)
        if not path.is_file():
            return False
        try:
            path.unlink()
            # 顺手清空目录（两级分片下很容易留下一堆空目录）
            try:
                path.parent.rmdir()
            except OSError:
                pass
            logger.info("blob_deleted", blob_id=blob_id[:12])
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("blob_delete_failed", blob_id=blob_id[:12], error=str(e))
            return False

    def count(self) -> int:
        if not self._root.is_dir():
            return 0
        return sum(1 for p in self._root.rglob("*") if p.is_file()
                   and not p.name.endswith(".tmp"))

    def total_bytes(self) -> int:
        if not self._root.is_dir():
            return 0
        return sum(p.stat().st_size for p in self._root.rglob("*") if p.is_file())
