from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.config import settings
from RetailPilot.database import db_session, init_db, json_dumps, set_metadata
from RetailPilot.eval_dataset import seed_retrieval_eval_dataset
from RetailPilot.taxonomy import category_label, category_tags
from RetailPilot.text import compact_text, hash_vector, plain_text, vector_to_blob


HF_REPOSITORY = "McAuley-Lab/Amazon-Reviews-2023"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPOSITORY}/resolve/main/raw/meta_categories"

SOURCES = [
    ("All_Beauty", "All Beauty"),
    ("Appliances", "Appliances"),
    ("Cell_Phones_and_Accessories", "Cell Phones & Accessories"),
    ("Electronics", "Electronics"),
    ("Sports_and_Outdoors", "Sports & Outdoors"),
    ("Video_Games", "Video Games"),
]
LAPTOP_SOURCE_SLUG = "Electronics"
LAPTOP_CHUNK_BYTES = 8 * 1024 * 1024
LAPTOP_SOURCE_SIZE = 5_246_144_134

EVENT_WEIGHTS = {
    "impression": 0.05,
    "click": 0.4,
    "favorite": 1.0,
    "cart": 1.5,
    "purchase": 3.0,
    "dislike": -2.0,
}

APP_NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def source_url(file_slug: str) -> str:
    return f"{HF_BASE}/meta_{file_slug}.jsonl"


def is_primary_laptop_record(record: dict[str, Any]) -> bool:
    categories = [plain_text(value, 160) for value in (record.get("categories") or [])]
    return "Laptops" in categories


def _main_image(images: Any) -> str:
    if not isinstance(images, list):
        return ""
    for image in images:
        if not isinstance(image, dict):
            continue
        if image.get("variant") == "MAIN":
            return image.get("hi_res") or image.get("large") or image.get("thumb") or ""
    for image in images:
        if isinstance(image, dict):
            value = image.get("hi_res") or image.get("large") or image.get("thumb")
            if value:
                return value
    return ""


def _brand(record: dict[str, Any]) -> str:
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    for value in (
        record.get("store"),
        details.get("Brand"),
        details.get("Manufacturer"),
        details.get("Publisher"),
    ):
        if value and str(value).strip().lower() not in {"unknown", "none", "n/a"}:
            return plain_text(value, 100)
    return ""


def _attributes(record: dict[str, Any]) -> dict[str, str]:
    details = record.get("details") if isinstance(record.get("details"), dict) else {}
    attributes: dict[str, str] = {}
    for key, value in details.items():
        if value is None or isinstance(value, (dict, list)):
            continue
        clean_key = plain_text(key, 80)
        clean_value = plain_text(value, 240)
        if clean_key and clean_value:
            attributes[clean_key] = clean_value
        if len(attributes) >= 24:
            break
    return attributes


def _stock(sku: str) -> int:
    digest = hashlib.blake2b(sku.encode("utf-8"), digest_size=4).digest()
    return 12 + int.from_bytes(digest, "little") % 289


def normalize_product(record: dict[str, Any], category: str) -> dict[str, Any] | None:
    title = plain_text(record.get("title"), 500)
    sku = compact_text(record.get("parent_asin"), 40)
    price = record.get("price")
    image_url = _main_image(record.get("images"))
    if not title or not sku or not image_url or not isinstance(price, (int, float)):
        return None
    price = float(price)
    if price <= 1 or price > 10000:
        return None

    rating = float(record.get("average_rating") or 0)
    rating_count = int(record.get("rating_number") or 0)
    features = [
        plain_text(value, 360)
        for value in (record.get("features") or [])
        if plain_text(value, 360)
    ][:8]
    description = plain_text(record.get("description") or features, 4000)
    attributes = _attributes(record)
    brand = _brand(record)
    tags = category_tags(category)
    tags.extend(plain_text(value, 120) for value in (record.get("categories") or [])[:8])
    tags = list(dict.fromkeys(tag for tag in tags if tag))

    prior_mean = 4.1
    prior_weight = 40
    bayesian_rating = (rating_count * rating + prior_weight * prior_mean) / (
        rating_count + prior_weight
    )
    quality_score = max(0.0, min(1.0, bayesian_rating / 5))
    popularity_score = max(0.0, min(1.0, math.log1p(rating_count) / 12))
    source_page = f"https://www.amazon.com/dp/{sku}"
    search_text = " ".join(
        [
            title,
            brand,
            category,
            category_label(category),
            " ".join(tags),
            " ".join(features),
            description,
            compact_text(attributes, 3000),
        ]
    )
    return {
        "sku": sku,
        "title": title,
        "brand": brand,
        "category": category,
        "category_label": category_label(category),
        "description": description,
        "price": price,
        "rating": rating,
        "rating_count": rating_count,
        "stock": _stock(sku),
        "features": features,
        "attributes": attributes,
        "tags": tags,
        "tags_text": " ".join(tags + features),
        "attributes_text": compact_text(attributes, 4000),
        "image_url": image_url,
        "source": "Amazon Reviews 2023",
        "source_url": source_page,
        "quality_score": quality_score,
        "popularity_score": popularity_score,
        "vector": vector_to_blob(hash_vector(search_text)),
    }


def stream_products(file_slug: str, category: str, limit: int) -> Iterable[dict[str, Any]]:
    url = source_url(file_slug)
    headers = {"User-Agent": "RetailPilot/1.0"}
    with requests.get(url, stream=True, timeout=(20, 120), headers=headers) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        accepted = 0
        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if category == "Electronics" and is_primary_laptop_record(record):
                continue
            product = normalize_product(record, category)
            if product is None:
                continue
            yield product
            accepted += 1
            if accepted >= limit:
                break


def _range_laptop_products(start: int, end: int) -> list[dict[str, Any]]:
    headers = {
        "User-Agent": "RetailPilot/1.0",
        "Range": f"bytes={start}-{end}",
    }
    with requests.get(
        source_url(LAPTOP_SOURCE_SLUG),
        stream=True,
        timeout=(20, 120),
        headers=headers,
    ) as response:
        response.raise_for_status()
        payload = response.content
    if start:
        payload = payload.split(b"\n", 1)[1] if b"\n" in payload else b""
    if not payload.endswith(b"\n"):
        payload = payload.rsplit(b"\n", 1)[0] if b"\n" in payload else b""

    products: list[dict[str, Any]] = []
    for raw_line in payload.splitlines():
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not is_primary_laptop_record(record):
            continue
        product = normalize_product(record, "Computers")
        if product:
            products.append(product)
    return products


def stream_laptop_products(limit: int) -> Iterable[dict[str, Any]]:
    """Sample the large Electronics file by byte ranges and keep laptop SPUs only."""
    if limit <= 0:
        return
    sample_count = max(24, min(48, math.ceil(limit / 13) + 5))
    max_start = max(0, LAPTOP_SOURCE_SIZE - LAPTOP_CHUNK_BYTES)
    offsets = [
        int(max_start * index / max(1, sample_count - 1))
        for index in range(sample_count)
    ]
    yielded = 0
    seen: set[str] = set()
    batch_size = 8
    for batch_start in range(0, len(offsets), batch_size):
        batch = offsets[batch_start : batch_start + batch_size]
        futures = {}
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            for offset in batch:
                futures[executor.submit(
                    _range_laptop_products,
                    offset,
                    min(LAPTOP_SOURCE_SIZE - 1, offset + LAPTOP_CHUNK_BYTES - 1),
                )] = offset
            completed: dict[int, list[dict[str, Any]]] = {}
            for future in as_completed(futures):
                completed[futures[future]] = future.result()
        for offset in sorted(completed):
            for product in completed[offset]:
                if product["sku"] in seen:
                    continue
                seen.add(product["sku"])
                yield product
                yielded += 1
                if yielded >= limit:
                    return


def reset_database() -> None:
    init_db()
    with db_session() as connection:
        connection.execute("DELETE FROM product_fts")
        for table in (
            "catalog_imports",
            "llm_judge_cache",
            "evaluation_samples",
            "search_result_impressions",
            "search_route_logs",
            "eval_runs",
            "eval_qrels",
            "eval_cases",
            "search_logs",
            "logistics_events",
            "orders",
            "messages",
            "sessions",
            "memories",
            "interactions",
            "user_preferences",
            "users",
            "product_commerce",
            "product_vectors",
            "products",
            "metadata",
        ):
            connection.execute(f"DELETE FROM {table}")


def insert_products(products: Iterable[dict[str, Any]]) -> int:
    count = 0
    with db_session() as connection:
        for product in products:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO products(
                    sku, title, brand, category, category_label, description, price,
                    rating, rating_count, stock, features_json, attributes_json,
                    tags_json, tags_text, attributes_text, image_url, source,
                    source_url, quality_score, popularity_score
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product["sku"],
                    product["title"],
                    product["brand"],
                    product["category"],
                    product["category_label"],
                    product["description"],
                    product["price"],
                    product["rating"],
                    product["rating_count"],
                    product["stock"],
                    json_dumps(product["features"]),
                    json_dumps(product["attributes"]),
                    json_dumps(product["tags"]),
                    product["tags_text"],
                    product["attributes_text"],
                    product["image_url"],
                    product["source"],
                    product["source_url"],
                    product["quality_score"],
                    product["popularity_score"],
                ),
            )
            if cursor.rowcount != 1:
                continue
            connection.execute(
                """
                INSERT INTO product_fts(sku, title, brand, category, tags, description, attributes)
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    product["sku"],
                    product["title"],
                    product["brand"],
                    f"{product['category']} {product['category_label']}",
                    product["tags_text"],
                    product["description"],
                    product["attributes_text"],
                ),
            )
            connection.execute(
                "INSERT INTO product_vectors(sku, dim, vector) VALUES(?, ?, ?)",
                (product["sku"], settings.vector_dim, product["vector"]),
            )
            count += 1
    return count


def seed_business_data() -> None:
    now = APP_NOW
    with db_session() as connection:
        connection.executemany(
            """
            INSERT INTO users(user_id, display_name, profile_json)
            VALUES(?, ?, ?)
            ON CONFLICT(user_id) DO NOTHING
            """,
            [
                (
                    "demo-user",
                    "Alex",
                    json_dumps(
                        {
                            "price_sensitivity": "medium",
                            "preferred_categories": ["Electronics", "Sports & Outdoors"],
                            "concerns": ["性价比", "轻便", "高评分"],
                        }
                    ),
                ),
                ("new-user", "New user", "{}"),
            ],
        )
        connection.executemany(
            """
            INSERT INTO user_preferences(
                user_id, preference_type, preference_key, weight, evidence
            ) VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(user_id, preference_type, preference_key)
            DO UPDATE SET weight=excluded.weight, evidence=excluded.evidence
            """,
            [
                ("demo-user", "category", "Electronics", 1.2, "demo profile"),
                ("demo-user", "category", "Sports & Outdoors", 0.7, "demo profile"),
                ("demo-user", "concept", "lightweight", 0.6, "demo profile"),
                ("demo-user", "concept", "value", 0.5, "demo profile"),
            ],
        )

        products = connection.execute(
            """
            SELECT sku, price FROM products
            WHERE category IN ('Electronics', 'Cell Phones & Accessories')
            ORDER BY quality_score DESC, popularity_score DESC
            LIMIT 3
            """
        ).fetchall()
        if products:
            order_rows = []
            event_rows = []
            for index, product in enumerate(products[:2], start=1):
                order_id = f"JD2026080{index}"
                tracking = f"SF20260810{index:03d}"
                created = now - timedelta(days=7 + index * 3)
                status = "运输中" if index == 1 else "已签收"
                order_rows.append(
                    (
                        order_id,
                        "demo-user",
                        product["sku"],
                        product["price"],
                        status,
                        tracking,
                        created.isoformat(timespec="seconds"),
                    )
                )
                events = [
                    (
                        order_id,
                        (created + timedelta(hours=2)).isoformat(timespec="minutes"),
                        "已揽收",
                        "深圳",
                        "商家已发货，包裹由承运商揽收",
                    ),
                    (
                        order_id,
                        (created + timedelta(days=1)).isoformat(timespec="minutes"),
                        "运输中",
                        "广州转运中心",
                        "包裹已到达转运中心",
                    ),
                ]
                if index == 1:
                    events.append(
                        (
                            order_id,
                            (now - timedelta(hours=8)).isoformat(timespec="minutes"),
                            "运输中",
                            "杭州转运中心",
                            "包裹正在发往目的地网点",
                        )
                    )
                else:
                    events.append(
                        (
                            order_id,
                            (created + timedelta(days=3)).isoformat(timespec="minutes"),
                            "已签收",
                            "上海",
                            "包裹已由本人签收",
                        )
                    )
                event_rows.extend(events)
            connection.executemany(
                """
                INSERT OR REPLACE INTO orders(
                    order_id, user_id, sku, amount, status, tracking_no, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                order_rows,
            )
            connection.executemany(
                """
                INSERT INTO logistics_events(
                    order_id, event_time, status, location, description
                ) VALUES(?, ?, ?, ?, ?)
                """,
                event_rows,
            )

        connection.executemany(
            """
            INSERT OR REPLACE INTO faq_documents(
                doc_id, topic, title, content, source, platform, category,
                effective_from, effective_to, version, priority, status, metadata_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "faq-shipping",
                    "运费",
                    "配送与运费",
                    "订单满 99 元免基础运费；未满 99 元收取 10 元。偏远地区和超大件商品以结算页为准。",
                    "本地演示商城规则",
                    "demo-mall", "*", "2026-01-01", "", "2.0", 5, "active", "{}",
                ),
                (
                    "faq-return",
                    "退换货",
                    "七天无理由退货",
                    "符合完好标准的商品支持签收后 7 天内申请无理由退货；定制、数字内容及影响二次销售的商品除外。",
                    "本地演示商城规则",
                    "demo-mall", "*", "2026-01-01", "", "2.0", 5, "active", "{}",
                ),
                (
                    "faq-invoice",
                    "发票",
                    "电子发票",
                    "订单完成后可申请电子发票，系统将在 3 个工作日内发送至账户绑定邮箱。",
                    "本地演示商城规则",
                    "demo-mall", "*", "2026-01-01", "", "2.0", 5, "active", "{}",
                ),
                (
                    "faq-warranty",
                    "保修",
                    "质量问题与保修",
                    "商品保修期以商品页和品牌政策为准。质量问题可提交照片和订单号，由人工客服审核处理。",
                    "本地演示商城规则",
                    "demo-mall", "*", "2026-01-01", "", "2.0", 5, "active", "{}",
                ),
                (
                    "faq-return-digital-exception",
                    "退换货",
                    "数字内容和定制商品例外",
                    "数字内容、激活后的软件服务、定制商品，以及已经影响二次销售的商品，不适用七天无理由退货；质量问题仍需按售后流程提交凭证审核。",
                    "本地演示商城规则",
                    "demo-mall", "Video Games", "2026-01-01", "", "2.0", 9, "active",
                    json_dumps({"rule_type": "category_exception"}),
                ),
                (
                    "faq-refund-timing",
                    "退款",
                    "退款到账时效",
                    "退款审核通过后原路退回；平台处理通常需要 1 至 3 个工作日，银行或支付机构的实际入账时间可能更长。",
                    "本地演示商城规则",
                    "demo-mall", "*", "2026-01-01", "", "1.0", 5, "active", "{}",
                ),
            ],
        )


def seed_eval_cases() -> None:
    cases = [
        {
            "case_id": "intent-electronics-budget",
            "name": "预算硬约束",
            "query": "推荐 50 美元以内评分高的无线耳机",
            "expected": {
                "intent": "search",
                "category": "Electronics",
                "price_max": 50,
                "min_results": 1,
                "constraint_pass": True,
            },
            "tags": ["query", "budget", "retrieval"],
        },
        {
            "case_id": "category-beauty-sensitive",
            "name": "中文跨语言概念",
            "query": "敏感肌用的保湿护肤品，预算 30 美元",
            "expected": {
                "intent": "search",
                "category": "All Beauty",
                "price_max": 30,
                "concepts": ["sensitive skin", "moisturizing"],
                "min_results": 1,
            },
            "tags": ["query", "semantic", "beauty"],
        },
        {
            "case_id": "category-phone-accessories",
            "name": "SPU 类目归一",
            "query": "找一个便携快充的手机充电器",
            "expected": {
                "intent": "search",
                "category": "Cell Phones & Accessories",
                "concepts": ["portable", "fast charging"],
                "min_results": 1,
            },
            "tags": ["spu", "concept"],
        },
        {
            "case_id": "category-sports",
            "name": "场景属性",
            "query": "跑步用的轻便防水装备，100 美元以内",
            "expected": {
                "intent": "search",
                "category": "Sports & Outdoors",
                "price_max": 100,
                "min_results": 1,
            },
            "tags": ["scenario", "attribute"],
        },
        {
            "case_id": "negative-brand",
            "name": "负向约束",
            "query": "推荐数码产品，不要 Apple，价格 200 美元以内",
            "expected": {
                "intent": "search",
                "category": "Electronics",
                "negative_brand": "Apple",
                "price_max": 200,
                "constraint_pass": True,
            },
            "tags": ["negative", "brand"],
        },
        {
            "case_id": "complex-headphone-constraints",
            "name": "SPU 与负向形态约束",
            "query": "预算100美元以内的无线降噪耳机，排除入耳式，最好索尼",
            "expected": {
                "intent": "search",
                "category": "Electronics",
                "price_max": 100,
                "preferred_brand": "Sony",
                "product_term": "headphone",
                "negative_title_terms": ["earbud", "earphone"],
                "min_results": 1,
                "constraint_pass": True,
            },
            "tags": ["spu", "negative", "brand", "relaxation"],
        },
        {
            "case_id": "after-sales-order",
            "name": "售后订单意图",
            "query": "帮我查一下订单 JD20260801 到哪了",
            "expected": {"intent": "order"},
            "tags": ["intent", "after_sales"],
        },
        {
            "case_id": "clarification",
            "name": "澄清策略",
            "query": "给我推荐一个",
            "expected": {"intent": "search", "clarification_slot": "category"},
            "tags": ["clarify", "dialog"],
        },
        {
            "case_id": "laptop-office-best",
            "name": "笔记本电脑办公最佳商品",
            "query": "笔记本 办公 最好的",
            "expected": {
                "intent": "search",
                "category": "Computers",
                "sort": "rating_desc",
                "product_term": "laptop",
                "min_results": 3,
                "forbidden_title_terms": [
                    "headphone",
                    "monitor",
                    "sleeve",
                    "case",
                    "bag",
                    "charger",
                    "stand",
                ],
                "min_candidate_count": 10,
            },
            "tags": ["badcase", "spu", "scenario", "ranking"],
        },
        {
            "case_id": "correction-single-search",
            "name": "用户纠错只保留一个搜索意图",
            "query": "啥啊，我说的 笔记本电脑。你就算给我写的笔记本都行，为什么给我耳机、电脑包、显示器？",
            "context": {
                "last_parsed_query": {
                    "category": "Computers",
                    "concepts": ["office", "business", "productivity", "work"],
                    "sort": "rating_desc",
                }
            },
            "expected": {
                "intent": "search",
                "category": "Computers",
                "sort": "rating_desc",
                "product_term": "laptop",
                "min_results": 3,
                "forbidden_title_terms": [
                    "headphone",
                    "monitor",
                    "sleeve",
                    "case",
                    "bag",
                    "charger",
                    "stand",
                ],
            },
            "tags": ["badcase", "correction", "intent"],
        },
    ]
    with db_session() as connection:
        connection.executemany(
            """
            INSERT OR REPLACE INTO eval_cases(
                case_id, name, query, context_json, expected_json, tags_json, enabled
            ) VALUES(?, ?, ?, ?, ?, ?, 1)
            """,
            [
                (
                    case["case_id"],
                    case["name"],
                    case["query"],
                    json_dumps(case.get("context") or {}),
                    json_dumps(case["expected"]),
                    json_dumps(case["tags"]),
                )
                for case in cases
            ],
        )


def bootstrap(per_category: int, reset: bool = True) -> dict[str, Any]:
    init_db()
    if reset:
        reset_database()
    inserted_by_source: dict[str, int] = {}
    for file_slug, category in SOURCES:
        print(f"[dataset] {category}: streaming up to {per_category} products")
        products = stream_products(file_slug, category, per_category)
        inserted = insert_products(products)
        inserted_by_source[category] = inserted
        print(f"[dataset] {category}: inserted {inserted}")

    laptop_limit = max(20, min(per_category, 600))
    print(f"[dataset] Computers: sampling up to {laptop_limit} primary laptop products")
    laptop_inserted = insert_products(stream_laptop_products(laptop_limit))
    inserted_by_source["Computers"] = laptop_inserted
    print(f"[dataset] Computers: inserted {laptop_inserted}")

    with db_session() as connection:
        category_rows = connection.execute(
            "SELECT category, COUNT(*) AS count FROM products GROUP BY category ORDER BY category"
        ).fetchall()
        total = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    category_counts = {row["category"]: row["count"] for row in category_rows}
    if total < max(50, per_category):
        raise RuntimeError(f"too few products imported: {total}")

    seed_business_data()
    seed_eval_cases()
    seed_retrieval_eval_dataset()
    set_metadata(
        "dataset",
        {
            "repository": HF_REPOSITORY,
            "source_urls": [source_url(slug) for slug, _ in SOURCES],
            "imported_at": APP_NOW.isoformat(timespec="seconds"),
            "products": total,
            "categories": category_counts,
            "inventory_note": "stock is deterministic demo inventory; catalog facts come from source metadata",
            "targeted_sampling": {
                "Computers": {
                    "source_category": "Electronics",
                    "source_path": "Electronics > Computers & Accessories > Computers & Tablets > Laptops",
                    "sampling": "evenly spaced byte ranges with exact source category filter",
                }
            },
        },
    )
    return {"products": total, "categories": category_counts, "database": str(settings.database_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the local RetailPilot SQLite database")
    parser.add_argument("--per-category", type=int, default=600)
    parser.add_argument("--no-reset", action="store_true")
    args = parser.parse_args()
    report = bootstrap(per_category=max(20, args.per_category), reset=not args.no_reset)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
