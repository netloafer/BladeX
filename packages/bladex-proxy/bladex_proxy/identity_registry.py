"""ADR-0021 §2 身份两体系分离 - API Key 体系（凭证层）+ 组织架构体系（可选组织层）。

两套独立配置，唯一交点是 ``principal``：
  - 体系 A（[[keys]]）：多 key -> 一 principal（凭证生命周期，发放/吊销/轮换）。
  - 体系 B（[[orgs]]/[[teams]]）：org -> 多级 team 树 -> principal 挂载（组织调整）。
  - 个人用户只配（或不配）体系 A，体系 B 不存在 -> 零配置复杂度不变。

三级回落（§2.3，向后兼容是硬约束）：
  1. identity.toml 不存在 -> **单一本地身份**（`user_id = "local"`，与凭证解耦）。
     2026-08-16 修订：原为 `user_id = key hash8`，那等于把凭证当身份，
     换 key / 多设备就静默劈开记忆（ADR-0021 §2.3 修订记录）。
  2. 文件存在但 key 未声明 -> 隐式 principal（hash8）+ warning，不拒绝。
  3. principal 无 teams -> 可见集合只有 personal。

legacy_ids 归并（§2.5）：principal 声明存量 user_id(hash8)，启动时对照 Memory Hub journal
追加 IDENTITY_MERGE 管理事件；Memory Index 重建读侧映射 legacy user_id -> principal。Memory Hub 不重写。

凭证与身份分离（本 ADR 主旨）：auth.KeyStore 继续做准入校验，registry 做身份解析，
两者独立--key 变更不动身份，组织调整不动凭证。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger()


# ── 体系 A / B / 交点 配置模型（TOML 直射）──


class KeyEntry(BaseModel):
    """体系 A：一条 API Key -> principal 映射。"""

    key: str
    label: str = ""
    principal: str  # 指向 Principal.id

    model_config = {"extra": "ignore"}


class Principal(BaseModel):
    """两体系交点：一个身份主体（人/服务账号）。"""

    id: str
    name: str = ""
    # §2.5: 存量 user_id(hash8) 归并声明。Memory Index 重建读侧映射 legacy -> 本 principal。
    legacy_ids: list[str] = Field(default_factory=list)
    # 体系 B 挂载点；个人用户省略 -> 可见集合只有 personal。
    teams: list[str] = Field(default_factory=list)
    # §3.1: principal 维度敏感等级（体系 B 的自然属性，如法务团队全员敏感）。空 = 未标注。
    sensitivity: str = ""

    model_config = {"extra": "ignore"}


class Org(BaseModel):
    """体系 B：组织（v1 单 org 够用，多 org = 多套部署，§2.6 不做）。"""

    id: str
    name: str = ""

    model_config = {"extra": "ignore"}


class Team(BaseModel):
    """体系 B：团队（可多级嵌套，parent 指向上级 team）。"""

    id: str
    org: str = ""       # 所属 org
    parent: str = ""    # 上级 team（空 = 顶级）
    # §3.1: team 维度敏感等级。空 = 未标注。
    sensitivity: str = ""

    model_config = {"extra": "ignore"}


class ResolvedIdentity(BaseModel):
    """从 api_key 解析出的身份（挂到 Identity，供 Memory Index 检索 + 敏感度解析消费）。

    实现 bladex_core.sensitivity.IdentitySensitivity Protocol（principal_level/team_levels）。
    """

    principal_id: str
    # §2.4 可见集合：["personal:<pid>", "team:<tid>"..., "org:<oid>"]
    visibility: list[str] = Field(default_factory=list)
    principal_sensitivity: str = ""
    team_sensitivities: list[str] = Field(default_factory=list)
    legacy_ids: list[str] = Field(default_factory=list)
    # 回落 2：key 未声明 -> 隐式 principal（hash8），记 warning 不拒绝。
    implicit: bool = False

    @property
    def principal_level(self) -> str | None:
        return self.principal_sensitivity or None

    @property
    def team_levels(self) -> list[str]:
        return self.team_sensitivities


class IdentityRegistry:
    """身份两体系注册表（启动时从 identity.toml 加载 + 校验）。

    empty() = 无文件 = 完全现状回落（resolve 返回 None，调用方走 hash8）。
    """

    def __init__(
        self,
        keys: list[KeyEntry],
        principals: list[Principal],
        orgs: list[Org],
        teams: list[Team],
    ) -> None:
        self.keys = keys
        self.principals = principals
        self.orgs = orgs
        self.teams = teams
        self._key_index: dict[str, KeyEntry] = {k.key: k for k in keys}
        self._principal_index: dict[str, Principal] = {p.id: p for p in principals}
        self._team_index: dict[str, Team] = {t.id: t for t in teams}
        self._org_index: dict[str, Org] = {o.id: o for o in orgs}
        self.empty = not keys and not principals and not orgs and not teams

    @classmethod
    def empty_registry(cls) -> IdentityRegistry:
        """无 identity.toml = 完全现状（resolve 恒返 None）。"""
        return cls([], [], [], [])

    @classmethod
    def from_toml(cls, path: str | Path | None) -> IdentityRegistry:
        """从 identity.toml 加载 + 启动校验。文件不存在 -> empty（现状回落）。

        校验违例（key 重复 / 引用缺失 / team 环）-> ValueError（调用方拒启动）。
        """
        if path is None:
            return cls.empty_registry()
        p = Path(path)
        if not p.is_file():
            logger.info("identity_config_not_found", path=str(path),
                        hint="personal mode: single local identity (user_id = 'local'), "
                             "decoupled from credentials -- ADR-0021 section 2.3")
            return cls.empty_registry()
        with p.open("rb") as f:
            data = tomllib.load(f)

        keys = [KeyEntry(**k) for k in data.get("keys", [])]
        principals = [Principal(**pr) for pr in data.get("principals", [])]
        orgs = [Org(**o) for o in data.get("orgs", [])]
        teams = [Team(**t) for t in data.get("teams", [])]
        reg = cls(keys, principals, orgs, teams)
        reg._validate()
        logger.info(
            "identity_registry_loaded",
            keys=len(keys), principals=len(principals),
            orgs=len(orgs), teams=len(teams),
        )
        return reg

    # ── 启动校验（违例拒启动，配置错误必须显性）──

    def _validate(self) -> None:
        # key 全局唯一
        seen: set[str] = set()
        for k in self.keys:
            if k.key in seen:
                raise ValueError(f"duplicate key in identity.toml: {k.key}")
            seen.add(k.key)

        # key.principal 引用存在
        for k in self.keys:
            if k.principal not in self._principal_index:
                raise ValueError(
                    f"key '{k.label or k.key[:12]}' references unknown principal '{k.principal}'"
                )

        # principal.teams 引用存在
        for pr in self.principals:
            for tid in pr.teams:
                if tid not in self._team_index:
                    raise ValueError(
                        f"principal '{pr.id}' references unknown team '{tid}'"
                    )

        # team.org 引用存在 + parent 引用存在 + 树无环
        for t in self.teams:
            if t.org and t.org not in self._org_index:
                raise ValueError(f"team '{t.id}' references unknown org '{t.org}'")
            if t.parent and t.parent not in self._team_index:
                raise ValueError(
                    f"team '{t.id}' references unknown parent team '{t.parent}'"
                )
        self._check_team_no_cycle()

    def _check_team_no_cycle(self) -> None:
        """team parent 链无环（每个 team 沿 parent 走必须到 root，不回到自身）。"""
        for t in self.teams:
            visited: set[str] = set()
            cur = t.id
            while cur:
                if cur in visited:
                    raise ValueError(f"team parent cycle detected at '{cur}'")
                visited.add(cur)
                team = self._team_index.get(cur)
                if team is None:
                    break
                cur = team.parent

    # ── 解析 ──

    def resolve(
        self,
        api_key: str | None,
        fallback_user_id: str | None = None,
    ) -> ResolvedIdentity | None:
        """key -> principal -> teams(含祖先链) -> org -> 可见集合 + 敏感度。

        返回 None = 无注册表或无 key（调用方走 hash8 现状）。
        回落 2：文件存在但 key 未声明 -> 隐式 principal（fallback_user_id）+ warning。
        """
        if self.empty or api_key is None:
            return None

        entry = self._key_index.get(api_key)
        if entry is None:
            # 回落 2：未声明 key -> 隐式 principal（hash8）
            pid = fallback_user_id or api_key[:8]
            logger.warning(
                "identity_key_not_declared",
                key_prefix=api_key[:8],
                hint="key not in identity.toml -> implicit principal (hash8); "
                     "add a [[keys]] entry to merge this key's memory into a principal",
            )
            return ResolvedIdentity(
                principal_id=pid,
                visibility=[f"personal:{pid}"],
                implicit=True,
            )

        principal = self._principal_index.get(entry.principal)
        if principal is None:
            # _validate 已防，兜底
            return None

        # 展开团队祖先链 + 收集可见集合 + 敏感度
        visibility: list[str] = [f"personal:{principal.id}"]
        team_sensitivities: list[str] = []
        seen_teams: set[str] = set()
        for tid in principal.teams:
            for atid in self._team_ancestor_chain(tid):
                if atid in seen_teams:
                    continue
                seen_teams.add(atid)
                visibility.append(f"team:{atid}")
                team = self._team_index.get(atid)
                if team and team.sensitivity:
                    team_sensitivities.append(team.sensitivity)
                if team and team.org:
                    org_scope = f"org:{team.org}"
                    if org_scope not in visibility:
                        visibility.append(org_scope)

        return ResolvedIdentity(
            principal_id=principal.id,
            visibility=visibility,
            principal_sensitivity=principal.sensitivity,
            team_sensitivities=team_sensitivities,
            legacy_ids=list(principal.legacy_ids),
        )

    def _team_ancestor_chain(self, team_id: str) -> list[str]:
        """team 及其祖先链（从自身到 root，含自身）。"""
        chain: list[str] = []
        cur = team_id
        visited: set[str] = set()
        while cur and cur not in visited:
            visited.add(cur)
            chain.append(cur)
            team = self._team_index.get(cur)
            if team is None:
                break
            cur = team.parent
        return chain

    # ── legacy_ids 归并 ──

    def legacy_map(self) -> dict[str, str]:
        """所有 principal 的 legacy_ids -> principal_id 映射（Memory Index 重建读侧映射用）。

        合并 identity.toml 声明（live）与 Memory Hub journal 历史记录由调用方决定；
        本方法只给 identity.toml 的当前声明。
        """
        m: dict[str, str] = {}
        for pr in self.principals:
            for lid in pr.legacy_ids:
                m[lid] = pr.id
        return m

    def has_legacy_declarations(self) -> bool:
        return any(pr.legacy_ids for pr in self.principals)

    def sync_legacy_to_journal(self, ledger: Any) -> int:
        """启动时对照 Memory Hub journal，为新声明的 legacy_ids 追加 IDENTITY_MERGE 事件。

        幂等：已记录的 principal 不重复写。返回新增事件数。
        Memory Hub 不可用 -> 跳过（不阻塞启动）。
        """
        if not self.has_legacy_declarations():
            return 0
        try:
            from bladex_proxy.models import AdminEventType
        except Exception:  # noqa: BLE001
            return 0
        # 已记录的 principal（扫 journal）
        journaled: set[str] = set()
        try:
            for _key, ev in ledger.scan_admin_events():
                if ev.event_type == AdminEventType.IDENTITY_MERGE:
                    journaled.add(ev.matter_id)  # matter_id 复用为 principal_id
        except Exception as e:  # noqa: BLE001
            logger.warning("identity_legacy_journal_scan_failed", error=str(e))
            return 0

        added = 0
        for pr in self.principals:
            if not pr.legacy_ids:
                continue
            if pr.id in journaled:
                continue
            try:
                ledger.append_admin_event(
                    AdminEventType.IDENTITY_MERGE, pr.id,
                    principal_id=pr.id, legacy_ids=list(pr.legacy_ids),
                )
                added += 1
                logger.info("identity_legacy_merged_to_journal",
                            principal=pr.id, legacy_count=len(pr.legacy_ids))
            except Exception as e:  # noqa: BLE001
                logger.warning("identity_legacy_journal_append_failed",
                               principal=pr.id, error=str(e))
        return added
