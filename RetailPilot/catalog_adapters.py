from __future__ import annotations

import gzip
import hashlib
import json
import re
import uuid
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .config import settings
from .database import db_session, init_db, json_dumps, utc_now
from .text import compact_text, fast_catalog_tokens, hash_vector, vector_to_blob


SHOPSIMULATOR_LICENSE_NOTE = (
    "Local research import. ShopSimulator states that Taobao data was collected "
    "under authorization; verify upstream redistribution terms before publishing data."
)


def _text_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [compact_text(item, 300) for item in value if compact_text(item, 300)]
    if value:
        return [compact_text(value, 300)]
    return []


def _first_price(item: dict[str, Any]) -> float:
    values: list[float] = []
    for value in item.get("pricing") or []:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    for options in (item.get("customization_options") or {}).values():
        for option in options or []:
            try:
                values.append(float(option.get("price")))
            except (TypeError, ValueError):
                continue
    return min((value for value in values if value >= 0), default=0.0)


def _shop_brand(shop_name: str) -> str:
    value = compact_text(shop_name, 120)
    for suffix in ("官方旗舰店", "旗舰店", "专卖店", "企业店", "工厂店", "店"):
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)]
    return ""


def _variant_payload(raw: Any) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for name, options in (raw or {}).items():
        normalized = []
        for option in options or []:
            value = compact_text(option.get("value"), 300)
            if not value:
                continue
            normalized.append(
                {
                    "sku_id": str(option.get("asin") or ""),
                    "value": value,
                    "available": bool(option.get("is_available", True)),
                    "price": float(option.get("price") or 0),
                    "image_url": str(option.get("image") or ""),
                }
            )
        if normalized:
            result[compact_text(name, 80)] = normalized
    return result


def normalize_shopsimulator_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize one upstream ShopSimulator item into the local catalog contract."""
    platform_id = str(item.get("asin") or "").strip()
    if not platform_id:
        raise ValueError("ShopSimulator item is missing asin")
    title = compact_text(item.get("title"), 1000)
    if not title:
        raise ValueError(f"ShopSimulator item {platform_id} is missing title")
    attributes = _text_list(item.get("attribute"))
    category_path = compact_text(item.get("category"), 500)
    domain = compact_text(item.get("domain_zh"), 100) or "其他"
    variants = _variant_payload(item.get("customization_options"))
    features = list(dict.fromkeys([*attributes, *[part for part in re.split(r"[›>/]", category_path) if part]]))
    tags = list(dict.fromkeys([domain, *features]))
    images = _text_list(item.get("images"))
    price = _first_price(item)
    available_options = sum(
        1 for options in variants.values() for option in options if option["available"]
    )
    sku = f"tb-{platform_id}"
    # Cap index text explicitly: upstream descriptions can contain extremely
    # long page dumps, while title/category/curated attributes carry the useful
    # shopping signal. This keeps 24K and million-scale imports predictable.
    searchable = compact_text(
        " ".join([title, category_path, " ".join(attributes), " ".join(tags)]),
        2400,
    )
    fts_bigrams = " ".join(fast_catalog_tokens(compact_text(f"{title} {category_path}", 1200)))
    return {
        "sku": sku,
        "title": title,
        "brand": _shop_brand(str(item.get("shop_name") or "")),
        "category": domain,
        "category_label": domain,
        "description": compact_text(item.get("full_description") or item.get("sub_title"), 5000),
        "price": price,
        "rating": 0.0,
        "rating_count": 0,
        "stock": max(available_options, 1),
        "features": features,
        "attributes": {
            "店铺": compact_text(item.get("shop_name"), 200),
            "类目路径": category_path,
            "平台": "淘宝",
            "可选规格数": available_options,
        },
        "tags": tags,
        "tags_text": " ".join([*tags, fts_bigrams]),
        "attributes_text": searchable,
        "image_url": images[0] if images else "",
        "source": "ShopSimulator/Taobao",
        "source_url": "https://github.com/ShopAgent-Team/ShopSimulator",
        "quality_score": min(1.0, 0.55 + min(len(attributes), 8) * 0.04 + (0.08 if variants else 0)),
        "popularity_score": 0.5,
        "vector": vector_to_blob(hash_vector(searchable)),
        "commerce": {
            "platform": "taobao",
            "platform_product_id": platform_id,
            "spu_id": platform_id,
            "seller_name": compact_text(item.get("shop_name"), 200),
            "currency": "CNY",
            "category_path": category_path,
            "variants": variants,
            "services": [],
            "provenance": {
                "adapter": "shopsimulator-v1",
                "upstream_tag": str(item.get("tag") or ""),
            },
            "dataset_split": str(item.get("tag") or ""),
            "license_note": SHOPSIMULATOR_LICENSE_NOTE,
        },
        "instructions": item.get("instructions") or [],
        "user_persona": item.get("user_persona") or {},
    }


def iter_shopsimulator(path: Path | str, limit: int | None = None) -> Iterator[dict[str, Any]]:
    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("ShopSimulator catalog must be a JSON list")
    for index, item in enumerate(payload):
        if limit is not None and index >= limit:
            break
        yield normalize_shopsimulator_item(item)


def _insert_product(connection: Any, product: dict[str, Any], replace: bool) -> bool:
    verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
    cursor = connection.execute(
        f"""
        {verb} INTO products(
            sku, title, brand, category, category_label, description, price,
            rating, rating_count, stock, features_json, attributes_json,
            tags_json, tags_text, attributes_text, image_url, source,
            source_url, quality_score, popularity_score, updated_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            product["sku"], product["title"], product["brand"], product["category"],
            product["category_label"], product["description"], product["price"],
            product["rating"], product["rating_count"], product["stock"],
            json_dumps(product["features"]), json_dumps(product["attributes"]),
            json_dumps(product["tags"]), product["tags_text"], product["attributes_text"],
            product["image_url"], product["source"], product["source_url"],
            product["quality_score"], product["popularity_score"], utc_now(),
        ),
    )
    if cursor.rowcount != 1:
        return False
    sku = product["sku"]
    connection.execute("DELETE FROM product_fts WHERE sku = ?", (sku,))
    connection.execute(
        """
        INSERT INTO product_fts(sku, title, brand, category, tags, description, attributes)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sku, product["title"], product["brand"],
            f"{product['category']} {product['category_label']}", product["tags_text"],
            compact_text(product["description"], 800), compact_text(product["attributes_text"], 2400),
        ),
    )
    connection.execute(
        "INSERT OR REPLACE INTO product_vectors(sku, dim, vector) VALUES(?, ?, ?)",
        (sku, settings.vector_dim, product["vector"]),
    )
    commerce = product["commerce"]
    connection.execute(
        """
        INSERT OR REPLACE INTO product_commerce(
            sku, platform, platform_product_id, spu_id, seller_name, currency,
            category_path, variants_json, services_json, provenance_json,
            dataset_split, license_note, updated_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            sku, commerce["platform"], commerce["platform_product_id"], commerce["spu_id"],
            commerce["seller_name"], commerce["currency"], commerce["category_path"],
            json_dumps(commerce["variants"]), json_dumps(commerce["services"]),
            json_dumps(commerce["provenance"]), commerce["dataset_split"],
            commerce["license_note"], utc_now(),
        ),
    )
    return True


def _bulk_insert_products(
    connection: Any, products: list[dict[str, Any]], replace: bool
) -> int:
    """Batch the catalog write path; SQLite round trips dominate large imports."""
    if not products:
        return 0
    connection.execute("DROP TABLE IF EXISTS temp.import_skus")
    connection.execute("CREATE TEMP TABLE import_skus(sku TEXT PRIMARY KEY)")
    connection.executemany(
        "INSERT OR IGNORE INTO import_skus(sku) VALUES(?)",
        [(product["sku"],) for product in products],
    )
    if replace:
        selected = products
        connection.execute(
            "DELETE FROM product_fts WHERE sku IN (SELECT sku FROM import_skus)"
        )
    else:
        existing = {
            row["sku"]
            for row in connection.execute(
                "SELECT p.sku FROM products p JOIN import_skus i ON i.sku = p.sku"
            ).fetchall()
        }
        selected = [product for product in products if product["sku"] not in existing]
    if not selected:
        connection.execute("DROP TABLE temp.import_skus")
        return 0

    verb = "INSERT OR REPLACE" if replace else "INSERT"
    connection.executemany(
        f"""
        {verb} INTO products(
            sku, title, brand, category, category_label, description, price,
            rating, rating_count, stock, features_json, attributes_json,
            tags_json, tags_text, attributes_text, image_url, source,
            source_url, quality_score, popularity_score, updated_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                product["sku"], product["title"], product["brand"], product["category"],
                product["category_label"], product["description"], product["price"],
                product["rating"], product["rating_count"], product["stock"],
                json_dumps(product["features"]), json_dumps(product["attributes"]),
                json_dumps(product["tags"]), product["tags_text"], product["attributes_text"],
                product["image_url"], product["source"], product["source_url"],
                product["quality_score"], product["popularity_score"], utc_now(),
            )
            for product in selected
        ],
    )
    connection.executemany(
        """
        INSERT INTO product_fts(sku, title, brand, category, tags, description, attributes)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                product["sku"], product["title"], product["brand"],
                f"{product['category']} {product['category_label']}", product["tags_text"],
                compact_text(product["description"], 800), compact_text(product["attributes_text"], 2400),
            )
            for product in selected
        ],
    )
    connection.executemany(
        "INSERT OR REPLACE INTO product_vectors(sku, dim, vector) VALUES(?, ?, ?)",
        [(product["sku"], settings.vector_dim, product["vector"]) for product in selected],
    )
    connection.executemany(
        """
        INSERT OR REPLACE INTO product_commerce(
            sku, platform, platform_product_id, spu_id, seller_name, currency,
            category_path, variants_json, services_json, provenance_json,
            dataset_split, license_note, updated_at
        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                product["sku"], product["commerce"]["platform"],
                product["commerce"]["platform_product_id"], product["commerce"]["spu_id"],
                product["commerce"]["seller_name"], product["commerce"]["currency"],
                product["commerce"]["category_path"], json_dumps(product["commerce"]["variants"]),
                json_dumps(product["commerce"]["services"]), json_dumps(product["commerce"]["provenance"]),
                product["commerce"]["dataset_split"], product["commerce"]["license_note"], utc_now(),
            )
            for product in selected
        ],
    )
    connection.execute("DROP TABLE temp.import_skus")
    return len(selected)


def upsert_normalized_products(
    products: Iterable[dict[str, Any]], *, replace: bool = False
) -> int:
    init_db()
    count = 0
    with db_session() as connection:
        for product in products:
            count += int(_insert_product(connection, product, replace))
    return count


def import_shopsimulator(
    path: Path | str,
    *,
    limit: int | None = None,
    replace: bool = False,
    seed_tasks: bool = False,
) -> dict[str, Any]:
    """Import ShopSimulator products. `seed_tasks` is accepted for backward
    compatibility but no longer populates the removed `shopping_tasks` table."""
    init_db()
    source = Path(path).resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    products = list(iter_shopsimulator(source, limit=limit))
    with db_session() as connection:
        inserted = _bulk_insert_products(connection, products, replace)
        task_count = 0
        manifest = {
            "sha256": digest,
            "source_file": source.name,
            "adapter": "shopsimulator-v1",
            "license_note": SHOPSIMULATOR_LICENSE_NOTE,
            "requested_limit": limit,
        }
        import_id = f"shopsim-{digest[:12]}-{limit or 'all'}"
        connection.execute(
            """
            INSERT OR REPLACE INTO catalog_imports(
                import_id, adapter, source_path, dataset_version,
                product_count, task_count, manifest_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                import_id, "shopsimulator-v1", str(source), digest[:12], inserted,
                task_count, json_dumps(manifest),
            ),
        )
    return {
        "import_id": import_id,
        "source": str(source),
        "products_seen": len(products),
        "products_inserted": inserted,
        "tasks_upserted": task_count,
        "sha256": digest,
    }
