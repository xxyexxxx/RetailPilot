from __future__ import annotations

import math
from typing import Any, Iterable

from .database import db_session, init_db, json_dumps, json_loads, row_to_dict, utc_now

EVENT_WEIGHTS = {
    "impression": 0.05,
    "click": 0.4,
    "favorite": 1.0,
    "cart": 1.5,
    "purchase": 3.0,
    "dislike": -2.0,
}


PRODUCT_SELECT = """
SELECT p.*,
       COALESCE(c.platform, '') AS platform,
       COALESCE(c.platform_product_id, '') AS platform_product_id,
       COALESCE(c.spu_id, p.sku) AS spu_id,
       COALESCE(c.seller_name, '') AS seller_name,
       COALESCE(c.currency, 'USD') AS currency,
       COALESCE(c.category_path, p.category_label) AS category_path,
       COALESCE(c.variants_json, '{}') AS variants_json,
       COALESCE(c.services_json, '[]') AS services_json,
       COALESCE(c.provenance_json, '{}') AS provenance_json,
       COALESCE(c.dataset_split, '') AS dataset_split,
       COALESCE(c.license_note, '') AS license_note
FROM products p
LEFT JOIN product_commerce c ON c.sku = p.sku
"""


def hydrate_product(row: Any) -> dict[str, Any]:
    product = row_to_dict(row) or {}
    product.pop("vector", None)
    product.pop("dim", None)
    for source, target, default in (
        ("features_json", "features", []),
        ("attributes_json", "attributes", {}),
        ("tags_json", "tags", []),
        ("variants_json", "variants", {}),
        ("services_json", "services", []),
        ("provenance_json", "provenance", {}),
    ):
        product[target] = json_loads(product.pop(source, None), default)
    return product


def get_product(sku: str) -> dict[str, Any] | None:
    init_db()
    with db_session() as connection:
        row = connection.execute(f"{PRODUCT_SELECT} WHERE p.sku = ?", (sku,)).fetchone()
    return hydrate_product(row) if row else None


def get_products(skus: Iterable[str]) -> list[dict[str, Any]]:
    values = list(dict.fromkeys(skus))
    if not values:
        return []
    placeholders = ",".join("?" for _ in values)
    with db_session() as connection:
        rows = connection.execute(
            f"{PRODUCT_SELECT} WHERE p.sku IN ({placeholders})", values
        ).fetchall()
    by_sku = {row["sku"]: hydrate_product(row) for row in rows}
    return [by_sku[sku] for sku in values if sku in by_sku]


def list_categories() -> list[dict[str, Any]]:
    with db_session() as connection:
        rows = connection.execute(
            """
            SELECT category, category_label, COUNT(*) AS count,
                   MIN(price) AS min_price, MAX(price) AS max_price
            FROM products
            GROUP BY category, category_label
            ORDER BY count DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_brands(limit: int = 500) -> list[str]:
    with db_session() as connection:
        rows = connection.execute(
            """
            SELECT brand, COUNT(*) AS count
            FROM products
            WHERE brand <> ''
            GROUP BY brand
            ORDER BY count DESC, brand
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [row["brand"] for row in rows]


def catalog_stats() -> dict[str, Any]:
    init_db()
    with db_session() as connection:
        product = connection.execute(
            """
            SELECT COUNT(*) AS products, COUNT(DISTINCT category) AS categories,
                   COUNT(DISTINCT NULLIF(brand, '')) AS brands,
                   ROUND(AVG(rating), 2) AS average_rating
            FROM products
            """
        ).fetchone()
        users = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        interactions = connection.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        searches = connection.execute("SELECT COUNT(*) FROM search_logs").fetchone()[0]
        domestic = connection.execute(
            "SELECT COUNT(*) FROM product_commerce WHERE currency = 'CNY'"
        ).fetchone()[0]
        platforms = connection.execute(
            "SELECT COUNT(DISTINCT NULLIF(platform, '')) FROM product_commerce"
        ).fetchone()[0]
    return {
        **dict(product),
        "users": users,
        "interactions": interactions,
        "searches": searches,
        "domestic_products": domestic,
        "platforms": platforms,
    }


def format_price(product: dict[str, Any] | float, currency: str | None = None) -> str:
    """Render prices without assuming every source uses US dollars."""
    if isinstance(product, dict):
        amount = float(product.get("price") or 0)
        currency = str(product.get("currency") or "USD")
    else:
        amount = float(product)
        currency = currency or "USD"
    symbols = {"CNY": "¥", "RMB": "¥", "USD": "$", "EUR": "€", "JPY": "¥"}
    symbol = symbols.get(currency.upper())
    return f"{symbol}{amount:,.2f}" if symbol else f"{amount:,.2f} {currency.upper()}"


def ensure_user(user_id: str, display_name: str = "") -> None:
    with db_session() as connection:
        connection.execute(
            """
            INSERT INTO users(user_id, display_name, profile_json)
            VALUES(?, ?, '{}')
            ON CONFLICT(user_id) DO NOTHING
            """,
            (user_id, display_name),
        )


def get_user_profile(user_id: str) -> dict[str, Any]:
    ensure_user(user_id)
    with db_session() as connection:
        user = connection.execute(
            "SELECT user_id, display_name, profile_json FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        preferences = connection.execute(
            """
            SELECT preference_type, preference_key, weight, evidence, updated_at
            FROM user_preferences
            WHERE user_id = ?
            ORDER BY ABS(weight) DESC, updated_at DESC
            """,
            (user_id,),
        ).fetchall()
        memories = connection.execute(
            """
            SELECT memory_type, memory_key, memory_value, confidence, evidence, updated_at
            FROM memories
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT 20
            """,
            (user_id,),
        ).fetchall()
    return {
        "user_id": user_id,
        "display_name": user["display_name"] if user else "",
        "profile": json_loads(user["profile_json"] if user else "{}", {}),
        "preferences": [dict(row) for row in preferences],
        "memories": [dict(row) for row in memories],
    }


def preference_map(user_id: str) -> dict[str, dict[str, float]]:
    ensure_user(user_id)
    with db_session() as connection:
        rows = connection.execute(
            """
            SELECT preference_type, preference_key, weight
            FROM user_preferences
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        result.setdefault(row["preference_type"], {})[row["preference_key"]] = float(
            row["weight"]
        )
    return result


def update_preference(
    user_id: str,
    preference_type: str,
    preference_key: str,
    delta: float,
    evidence: str,
) -> None:
    if not preference_key:
        return
    ensure_user(user_id)
    with db_session() as connection:
        connection.execute(
            """
            INSERT INTO user_preferences(
                user_id, preference_type, preference_key, weight, evidence, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, preference_type, preference_key) DO UPDATE SET
                weight = MAX(-4, MIN(4, user_preferences.weight + excluded.weight)),
                evidence = excluded.evidence,
                updated_at = excluded.updated_at
            """,
            (user_id, preference_type, preference_key, delta, evidence[:500], utc_now()),
        )


def add_interaction(
    user_id: str,
    sku: str,
    event_type: str,
    query: str = "",
) -> dict[str, Any]:
    ensure_user(user_id)
    product = get_product(sku)
    if product is None:
        raise KeyError(f"unknown product: {sku}")
    if event_type not in EVENT_WEIGHTS:
        raise ValueError(f"unsupported event type: {event_type}")
    weight = EVENT_WEIGHTS[event_type]
    with db_session() as connection:
        connection.execute(
            """
            INSERT INTO interactions(user_id, sku, event_type, weight, query)
            VALUES(?, ?, ?, ?, ?)
            """,
            (user_id, sku, event_type, weight, query[:500]),
        )
    if event_type in {"click", "favorite", "cart", "purchase", "dislike"}:
        factor = 1 if weight > 0 else -1
        update_preference(
            user_id,
            "category",
            product["category"],
            factor * min(abs(weight) * 0.2, 0.7),
            f"{event_type}:{sku}",
        )
        if product["brand"]:
            update_preference(
                user_id,
                "brand",
                product["brand"],
                factor * min(abs(weight) * 0.15, 0.6),
                f"{event_type}:{sku}",
            )
    return {"user_id": user_id, "sku": sku, "event_type": event_type, "weight": weight}


def interaction_scores(user_id: str, candidate_skus: list[str]) -> dict[str, float]:
    if not candidate_skus:
        return {}
    placeholders = ",".join("?" for _ in candidate_skus)
    with db_session() as connection:
        rows = connection.execute(
            f"""
            SELECT sku, SUM(weight) AS score
            FROM interactions
            WHERE user_id = ? AND sku IN ({placeholders})
            GROUP BY sku
            """,
            [user_id, *candidate_skus],
        ).fetchall()
    return {row["sku"]: math.tanh(float(row["score"]) / 3) for row in rows}


class SessionOwnershipError(PermissionError):
    pass


class SessionConflictError(RuntimeError):
    pass


def get_session(session_id: str, user_id: str) -> dict[str, Any]:
    ensure_user(user_id)
    with db_session() as connection:
        connection.execute(
            "INSERT INTO sessions(session_id,user_id,context_json,pending_json) VALUES(?,?,'{}','{}') ON CONFLICT DO NOTHING",
            (session_id, user_id),
        )
        row = connection.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row["user_id"] != user_id:
            raise SessionOwnershipError("session belongs to another user")
        context = json_loads(row["context_json"], {})
        through = int(context.get("history_summary_through_id") or 0)
        messages = connection.execute(
            "SELECT id,role,content,payload_json,created_at FROM messages WHERE session_id=? AND id>? ORDER BY id",
            (session_id, through),
        ).fetchall()
    return {
        "session_id": session_id,
        "user_id": user_id,
        "version": row["version"],
        "updated_at": row["updated_at"],
        "context": context,
        "pending": json_loads(row["pending_json"], {}),
        "messages": [
            {
                "id": item["id"],
                "role": item["role"],
                "content": item["content"],
                "payload": json_loads(item["payload_json"], {}),
                "created_at": item["created_at"],
            }
            for item in messages
        ],
    }


def _write_session(
    connection, session_id, user_id, context, pending, expected_version=None
):
    row = connection.execute(
        "SELECT user_id,version FROM sessions WHERE session_id=?", (session_id,)
    ).fetchone()
    if row and row["user_id"] != user_id:
        raise SessionOwnershipError("session belongs to another user")
    if row and expected_version is not None and row["version"] != expected_version:
        raise SessionConflictError("session changed; reload history before retrying")
    if row:
        cursor = connection.execute(
            "UPDATE sessions SET context_json=?,pending_json=?,version=version+1,updated_at=? WHERE session_id=? AND version=?",
            (
                json_dumps(context),
                json_dumps(pending or {}),
                utc_now(),
                session_id,
                row["version"],
            ),
        )
        if cursor.rowcount != 1:
            raise SessionConflictError("session changed")
    else:
        if expected_version not in (None, 0):
            raise SessionConflictError("session missing")
        connection.execute(
            "INSERT INTO sessions(session_id,user_id,context_json,pending_json,updated_at,version) VALUES(?,?,?,?,?,1)",
            (
                session_id,
                user_id,
                json_dumps(context),
                json_dumps(pending or {}),
                utc_now(),
            ),
        )


def save_session(
    session_id: str,
    user_id: str,
    context: dict[str, Any],
    pending: dict[str, Any] | None = None,
) -> None:
    ensure_user(user_id)
    with db_session() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _write_session(connection, session_id, user_id, context, pending)


def persist_turn(
    session_id: str,
    user_id: str,
    message: str,
    answer: str,
    context: dict[str, Any],
    payload: dict[str, Any],
    *,
    expected_version: int,
    pending: dict[str, Any] | None = None,
) -> None:
    """CAS context + both messages commit together or roll back together."""
    with db_session() as connection:
        connection.execute("BEGIN IMMEDIATE")
        _write_session(
            connection, session_id, user_id, context, pending, expected_version
        )
        connection.executemany(
            "INSERT INTO messages(session_id,role,content,payload_json) VALUES(?,?,?,?)",
            [
                (session_id, "user", message, "{}"),
                (session_id, "assistant", answer, json_dumps(payload)),
            ],
        )


def session_history(
    session_id: str, user_id: str, *, limit: int = 100, before_id: int | None = None
) -> dict[str, Any]:
    """Owner-checked paginated transcript, independent of the model summary."""
    with db_session() as connection:
        row = connection.execute(
            "SELECT user_id,version FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if not row:
            return {
                "session_id": session_id,
                "version": 0,
                "messages": [],
                "next_before_id": None,
            }
        if row["user_id"] != user_id:
            raise SessionOwnershipError("session belongs to another user")
        rows = connection.execute(
            "SELECT id,role,content FROM messages WHERE session_id=? AND id<? ORDER BY id DESC LIMIT ?",
            (session_id, before_id or 9223372036854775807, limit + 1),
        ).fetchall()
    has_more = len(rows) > limit
    selected = list(reversed(rows[:limit]))
    return {
        "session_id": session_id,
        "version": row["version"],
        "messages": [dict(item) for item in selected],
        "next_before_id": selected[0]["id"] if has_more else None,
    }


def add_message(
    session_id: str,
    role: str,
    content: str,
    payload: dict[str, Any] | None = None,
) -> None:
    with db_session() as connection:
        connection.execute(
            """
            INSERT INTO messages(session_id, role, content, payload_json)
            VALUES(?, ?, ?, ?)
            """,
            (session_id, role, content, json_dumps(payload or {})),
        )


def compare_products(skus: list[str]) -> dict[str, Any]:
    products = get_products(skus)
    fields: list[str] = []
    for product in products:
        fields.extend(product["attributes"].keys())
    fields = list(dict.fromkeys(fields))[:12]
    return {
        "products": products,
        "fields": [
            {
                "field": field,
                "values": {
                    product["sku"]: product["attributes"].get(field, "-")
                    for product in products
                },
            }
            for field in fields
        ],
    }
