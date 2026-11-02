"""Domain vocabulary isolated from the retrieval and reasoning core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class DomainProfile:
    name: str
    actor_aliases: tuple[str, ...] = ()
    family_equivalents: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    metric_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    action_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def indicators_in(self, text: str) -> tuple[str, ...]:
        normalized = str(text or "").lower().replace("ё", "е")
        return tuple(
            canonical
            for canonical, aliases in self.metric_aliases.items()
            if any(alias.lower().replace("ё", "е") in normalized for alias in aliases)
        )

    def actions_in(self, text: str) -> tuple[str, ...]:
        normalized = str(text or "").lower().replace("ё", "е")
        return tuple(
            canonical
            for canonical, aliases in self.action_aliases.items()
            if any(alias.lower().replace("ё", "е") in normalized for alias in aliases)
        )


BUILTIN_DOMAIN_PROFILES: dict[str, DomainProfile] = {
    "general": DomainProfile(
        name="general",
        actor_aliases=("group", "team", "crew", "unit", "department", "service", "device", "system"),
        action_aliases={
            "worked": ("работал", "работала", "работали", "worked"),
            "used": ("использован", "использована", "использовали", "used"),
            "recorded": ("зафиксирован", "зафиксирована", "recorded"),
            "confirmed": ("подтвержден", "подтверждена", "confirmed"),
            "deployed": ("развернут", "развернута", "deployed"),
            "operated": ("эксплуатировался", "эксплуатировалась", "operated"),
        },
    ),
    "finance": DomainProfile(
        name="finance",
        metric_aliases={
            "revenue": ("выручка", "оборот", "revenue", "turnover"),
            "profit": ("прибыль", "profit"),
            "cost": ("затраты", "расходы", "cost", "expenses"),
        },
    ),
    "military_reports": DomainProfile(
        name="military_reports",
        actor_aliases=(
            "рбпла", "бпла", "шгр", "дшб", "одшбр", "омсбр", "мсбр", "гр",
            "группа", "расчет", "расчёт", "подразделение", "uav", "drone",
            "group", "team", "crew", "unit",
        ),
        family_equivalents={"рбпла": ("рбпла", "бпла", "uav")},
        action_aliases={
            "struck": ("поражен", "поражена", "поражены", "struck"),
            "destroyed": ("уничтожен", "уничтожена", "уничтожены", "destroyed"),
            "damaged": ("поврежден", "повреждена", "повреждены", "damaged"),
            "suppressed": ("подавлен", "подавлена", "подавлены", "suppressed"),
            "detected": ("обнаружен", "обнаружена", "обнаружены", "detected"),
            "attacked": ("контратак", "нанесен удар", "attacked"),
            "transferred": ("переброск", "transferred"),
            "concentrated": ("сосредотач", "concentrated"),
            "used": ("применени", "использован", "использовал", "used"),
        },
    ),
    "technical": DomainProfile(
        name="technical",
        action_aliases={
            "installed": ("установлен", "установлена", "установлены", "installed"),
            "activated": ("активирован", "активирована", "активированы", "activated"),
            "failed": ("отказ", "неисправность", "failed"),
        },
    ),
}


def _custom_profile(data: Mapping[str, Any]) -> DomainProfile:
    def aliases(key: str) -> dict[str, tuple[str, ...]]:
        raw = data.get(key) or {}
        return {
            str(name): tuple(str(item) for item in values if str(item).strip())
            for name, values in dict(raw).items()
        }

    return DomainProfile(
        name="custom",
        actor_aliases=tuple(str(item) for item in (data.get("actor_aliases") or ()) if str(item).strip()),
        family_equivalents=aliases("family_equivalents"),
        metric_aliases=aliases("metric_aliases"),
        action_aliases=aliases("action_aliases"),
    )


def get_domain_profile(text: str = "", config: Mapping[str, Any] | None = None) -> DomainProfile:
    cfg = dict(config or {})
    if not cfg:
        try:
            from core.config import get_config

            cfg = dict((get_config().data or {}).get("domain_profile") or {})
        except Exception:
            cfg = {}
    selected = str(cfg.get("name") or "general").strip().lower()
    if selected == "custom":
        return _custom_profile(dict(cfg.get("custom") or {}))
    profile = BUILTIN_DOMAIN_PROFILES.get(selected, BUILTIN_DOMAIN_PROFILES["general"])
    if text and not profile.indicators_in(text) and not profile.actions_in(text):
        # Auto-detection is data-driven by profile aliases; core logic remains
        # unaware of finance, technical, or report-specific vocabulary.
        for candidate_name in ("finance", "technical", "military_reports", "general"):
            candidate = BUILTIN_DOMAIN_PROFILES[candidate_name]
            if candidate.indicators_in(text) or candidate.actions_in(text):
                return candidate
    return profile


__all__ = ["DomainProfile", "BUILTIN_DOMAIN_PROFILES", "get_domain_profile"]
