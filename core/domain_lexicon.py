"""Configurable vocabulary for document-specific actor families.

The retrieval core stays domain-neutral; deployments may extend this small
lexicon without editing extraction, storage, and statistics modules separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from core.domain_profile import BUILTIN_DOMAIN_PROFILES, get_domain_profile


DEFAULT_ACTOR_ALIASES = BUILTIN_DOMAIN_PROFILES["military_reports"].actor_aliases


def _normal(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "").lower().replace("ё", "е"))


def _literal_alias_pattern(alias: str) -> str:
    compact = _normal(alias)
    if compact == "рбпла":
        return r"(?:р\s*)?бпла"
    return re.escape(alias).replace(r"\ ", r"\s+")


@dataclass(frozen=True)
class DomainLexicon:
    actor_aliases: tuple[str, ...]
    family_equivalents: dict[str, tuple[str, ...]]

    @classmethod
    def from_config(cls, data: dict[str, Any] | None = None) -> "DomainLexicon":
        cfg = dict(data or {})
        profile = get_domain_profile(config=cfg.get("profile") or {})
        aliases = tuple(dict.fromkeys(str(x).strip() for x in cfg.get("actor_aliases", profile.actor_aliases or DEFAULT_ACTOR_ALIASES) if str(x).strip()))
        raw_groups = cfg.get("family_equivalents") or profile.family_equivalents or BUILTIN_DOMAIN_PROFILES["military_reports"].family_equivalents
        groups = {
            _normal(canonical): tuple(dict.fromkeys(_normal(x) for x in values if _normal(x)))
            for canonical, values in dict(raw_groups).items()
        }
        return cls(actor_aliases=aliases, family_equivalents=groups)

    @property
    def actor_pattern(self) -> str:
        patterns = [_literal_alias_pattern(alias) for alias in sorted(self.actor_aliases, key=len, reverse=True)]
        return "(?:" + "|".join(patterns) + ")"

    def canonical_family(self, family: str) -> str:
        value = _normal(family)
        for canonical, equivalents in self.family_equivalents.items():
            if value == canonical or value in equivalents:
                return canonical
        return value

    def aliases_for(self, family: str) -> tuple[str, ...]:
        canonical = self.canonical_family(family)
        aliases = self.family_equivalents.get(canonical)
        return aliases or (canonical,)

    def mentions_family(self, text: str, family: str) -> bool:
        normalized = _normal(text)
        return any(alias and alias in normalized for alias in self.aliases_for(family))

    def families_in_text(self, text: str) -> set[str]:
        normalized = _normal(text)
        found = set()
        for alias in self.actor_aliases:
            compact = _normal(alias)
            if compact and compact in normalized:
                found.add(self.canonical_family(compact))
        return found

    def family_matches(self, left: str, right: str) -> bool:
        return bool(left and right and self.canonical_family(left) == self.canonical_family(right))

    def reference_terms(self, family: str, number: str) -> list[str]:
        terms = []
        for alias in self.aliases_for(family):
            terms.extend([f"{alias} №{number}", f"{alias} {number}"])
        terms.append(f"№{number}")
        return list(dict.fromkeys(terms))


@lru_cache(maxsize=1)
def get_domain_lexicon() -> DomainLexicon:
    try:
        from core.config import get_config

        data = (get_config().data or {}).get("domain_lexicon", {})
    except Exception:
        data = {}
    return DomainLexicon.from_config(data)
