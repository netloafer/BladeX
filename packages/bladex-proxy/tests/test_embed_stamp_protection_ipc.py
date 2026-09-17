"""盖章保护必须覆盖「任何会加载本地模型的档」，不只是 `local`（2026-08-31）。

## 背景

「默认值变更保护」（2026-07-26 默认从 e5-large 改成 bge-small-en 时加的）：
用户没显式写 model、但 Memory Index 已盖章某个 local 模型 ⇒ 沿用库内模型，
升级 BladeX 不因默认值变更而废掉现有向量空间。

判据原本写死 `s.backend == BACKEND_LOCAL`，而后来加的 **`ipc` 档回落路径就是
加载本地模型**（`build_ipc_adapter` 的 `_local_factory`）。配了 ipc 又没显式写
model 的部署，回落时会拿到新默认 `bge-small-en`，与库内 e5-large 的章不符
⇒ 索引转只读、检索整体停摆。**枚举加了新成员、旧分支没跟上。**

本文件的每条都带判别力：断言"沿用库内"的同时，断言"若没有章则回到新默认"，
两侧都测，避免测出一个恒真的东西。
"""

from __future__ import annotations

import types

import pytest
from bladex_proxy.embedding import (
    BACKEND_API,
    BACKEND_IPC,
    BACKEND_LOCAL,
    DEFAULT_LOCAL_MODEL,
    resolve_embed_settings,
)

STAMPED = "intfloat/multilingual-e5-large"


@pytest.fixture
def stamped_index(tmp_path):
    """一个"已盖章 e5-large"的索引目录（只写 stamp 侧车，不碰 RocksDB）。"""
    (tmp_path / "embed_model_id").write_text(f"local:{STAMPED}", encoding="utf-8")
    return tmp_path


def _cfg(index_path, backend: str):
    """最小 cfg 桩：`resolve_embed_settings` 只用到这几个属性。"""
    return types.SimpleNamespace(
        index_path=str(index_path),
        routing_config=types.SimpleNamespace(
            embedding=types.SimpleNamespace(backend=backend, model="")),
    )


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    """env 是覆盖通道，本文件测的是 toml/默认那一层——先清干净。"""
    for k in ("BLADEX_EMBED_BACKEND", "BLADEX_EMBED_MODEL"):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("backend", [BACKEND_LOCAL, BACKEND_IPC])
def test_stamp_wins_for_every_backend_that_loads_a_local_model(
        backend, stamped_index, monkeypatch) -> None:
    """local 与 ipc 都要沿用库内模型 —— ipc 是 2026-08-31 补进来的那一档。"""
    s = resolve_embed_settings(_cfg(stamped_index, backend))
    assert s.model == STAMPED, f"backend={backend} 没有沿用库内盖章的模型"
    assert s.sources.get("model") == "index stamp"


@pytest.mark.parametrize("backend", [BACKEND_LOCAL, BACKEND_IPC])
def test_without_a_stamp_the_new_default_applies(backend, tmp_path) -> None:
    """判别力对照：没有章时**必须**落到新默认，否则上一条测的是恒真。"""
    s = resolve_embed_settings(_cfg(tmp_path, backend))
    assert not s.model, "没有章、也没显式配置 ⇒ model 留空，由下游取内置默认"


def test_ipc_fallback_model_is_the_stamped_one_not_the_new_default(
        stamped_index) -> None:
    """把保护的**后果**也钉住：ipc 连不上时加载的本地模型 = 库内那一个。

    这一条才是事故形态：`_local_model_for` 拿到空 model 会返回
    `DEFAULT_LOCAL_MODEL`（bge-small-en），向量空间当场分裂。
    """
    from bladex_proxy.embedding import _local_model_for

    s = resolve_embed_settings(_cfg(stamped_index, BACKEND_IPC))
    assert _local_model_for(s, True) == STAMPED
    assert _local_model_for(s, True) != DEFAULT_LOCAL_MODEL


def test_api_backend_is_not_covered_on_purpose(stamped_index) -> None:
    """api 档**不**在保护范围里：它不加载本地模型，沿用库内模型名毫无意义
    （还会把一个本地模型名当成远端模型名发出去）。边界要测，不能只测覆盖面。"""
    s = resolve_embed_settings(_cfg(stamped_index, BACKEND_API))
    assert s.model != STAMPED


def test_explicit_model_always_beats_the_stamp(stamped_index) -> None:
    """保护只在"没显式配"时生效——它压内置默认，不压任何显式设置。"""
    cfg = _cfg(stamped_index, BACKEND_IPC)
    cfg.routing_config.embedding.model = "BAAI/bge-small-en-v1.5"
    s = resolve_embed_settings(cfg)
    assert s.model == "BAAI/bge-small-en-v1.5"
    assert s.sources.get("model") == "routing.toml"
