from __future__ import annotations

from typing import Any

from .catalog import (
    ensure_user,
    get_user_profile,
    update_preference,
)
from .database import db_session, utc_now
from .query import matched_concept_aliases
from .schemas import ParsedQuery
from .taxonomy import CONCEPT_ALIASES


LIKE_WORDS = ("喜欢", "只要", "只买", "优先", "偏好", "常买")
DISLIKE_WORDS = ("不要", "不买", "不喜欢", "排除", "别推荐")


def put_memory(
    user_id: str,
    memory_type: str,
    memory_key: str,
    memory_value: str,
    confidence: float,
    evidence: str,
) -> None:
    ensure_user(user_id)
    with db_session() as connection:
        connection.execute(
            """
            INSERT INTO memories(
                user_id, memory_type, memory_key, memory_value,
                confidence, evidence, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, memory_type, memory_key) DO UPDATE SET
                memory_value=excluded.memory_value,
                confidence=MAX(memories.confidence, excluded.confidence),
                evidence=excluded.evidence,
                updated_at=excluded.updated_at
            """,
            (
                user_id,
                memory_type,
                memory_key,
                memory_value,
                confidence,
                evidence[:500],
                utc_now(),
            ),
        )


def remember_turn(
    user_id: str,
    message: str,
    parsed: ParsedQuery | dict[str, Any] | None,
) -> None:
    if parsed is None:
        return
    if isinstance(parsed, dict):
        parsed_dict = parsed
    else:
        parsed_dict = parsed.to_dict()

    category = parsed_dict.get("category")
    if category:
        update_preference(user_id, "category", category, 0.08, f"search:{message}")
        put_memory(
            user_id,
            "shopping_context",
            "recent_category",
            category,
            0.7,
            message,
        )

    for brand in [
        *(parsed_dict.get("brands") or []),
        *(parsed_dict.get("preferred_brands") or []),
    ]:
        weight = 0.35 if any(word in message for word in LIKE_WORDS) else 0.12
        update_preference(user_id, "brand", brand, weight, message)
    for brand in parsed_dict.get("negative_brands") or []:
        update_preference(user_id, "brand", brand, -0.9, message)
        put_memory(user_id, "negative_preference", f"brand:{brand}", brand, 0.95, message)

    positive_terms = {
        str(term).lower() for term in parsed_dict.get("concepts") or []
    }
    negative_terms = {
        str(term).lower() for term in parsed_dict.get("negative_terms") or []
    }
    for alias in matched_concept_aliases(message):
        expanded = {term.lower() for term in CONCEPT_ALIASES.get(alias, [])}
        if not (expanded & positive_terms or expanded & negative_terms):
            continue
        explicit_like = any(word in message for word in LIKE_WORDS)
        explicit_dislike = bool(expanded & negative_terms)
        delta = -0.6 if explicit_dislike else 0.3 if explicit_like else 0.07
        update_preference(user_id, "concept", alias, delta, message)

    price_max = parsed_dict.get("price_max")
    if price_max is not None:
        put_memory(
            user_id,
            "price",
            category or "global",
            str(price_max),
            0.75,
            message,
        )


def get_profile(user_id: str) -> dict[str, Any]:
    """Compatibility API retained for earlier code and interview demos."""
    profile = get_user_profile(user_id)
    result = dict(profile.get("profile") or {})
    for pref in profile.get("preferences", []):
        pref_type = pref["preference_type"]
        key = pref["preference_key"]
        if pref_type == "category" and pref["weight"] > 0:
            result.setdefault("category_affinity", {})[key] = pref["weight"]
        elif pref_type == "brand" and pref["weight"] > 0:
            result.setdefault("brand_like", []).append(key)
        elif pref_type == "brand" and pref["weight"] < 0:
            result.setdefault("brand_dislike", []).append(key)
    result["user_id"] = user_id
    return result


def update_profile(user_id: str, delta: dict[str, Any]) -> dict[str, Any]:
    for category, weight in (delta.get("category_affinity") or {}).items():
        update_preference(user_id, "category", category, float(weight), "compat:update_profile")
    for brand in delta.get("brand_like") or []:
        update_preference(user_id, "brand", brand, 0.5, "compat:update_profile")
    for brand in delta.get("brand_dislike") or []:
        update_preference(user_id, "brand", brand, -0.8, "compat:update_profile")
    return get_profile(user_id)


def memory_inject(state: dict[str, Any]) -> dict[str, Any]:
    user_id = state.get("user_id", "demo-user")
    return {"profile": get_profile(user_id)}


def memory_update(state: dict[str, Any]) -> dict[str, Any]:
    parsed_queries = state.get("parsed_queries") or []
    for parsed in parsed_queries:
        remember_turn(
            state.get("user_id", "demo-user"),
            state.get("effective_message") or state.get("message", ""),
            parsed,
        )
    return {}
