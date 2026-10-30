from __future__ import annotations

import json

from RetailPilot.agent import chat
from RetailPilot.catalog import add_interaction, get_user_profile, list_brands
from RetailPilot.query import parse_query
from RetailPilot.retrieval import search_engine


def verify_query_and_constraints() -> None:
    parsed = parse_query(
        "推荐 50 美元以内评分高的无线耳机，不要 Apple",
        known_brands=list_brands(),
    )
    assert parsed.category == "Electronics"
    assert parsed.price_max == 50
    assert "Apple" in parsed.negative_brands
    result = search_engine.search(parsed, user_id="new-user", limit=10)
    assert result["products"]
    assert all(product["price"] <= 50 for product in result["products"])
    assert all("apple" not in product["title"].lower() for product in result["products"])
    assert result["judge"]["pass"]
    print("[pass] query understanding + hard constraints + factual citations")


def verify_multiturn() -> None:
    first = chat("verify-multiturn", "推荐 50 美元以内的无线耳机", "demo-user")
    second = chat("verify-multiturn", "对比前两款", "demo-user")
    third = chat("verify-multiturn", "第一款怎么样", "demo-user")
    cheaper = chat("verify-multiturn", "再便宜一点", "demo-user")
    assert len(first["products"]) >= 2
    assert len(second["products"]) == 2
    assert len(third["products"]) == 1
    assert cheaper["parsed_queries"][0]["price_max"] < third["products"][0]["price"]
    print("[pass] session context + reference resolution + cheaper follow-up")


def verify_memory() -> None:
    product = search_engine.search(
        "便携快充手机充电器",
        user_id="demo-user",
        limit=1,
        known_brands=list_brands(),
    )["products"][0]
    add_interaction("demo-user", product["sku"], "favorite", "便携快充手机充电器")
    profile = get_user_profile("demo-user")
    assert any(
        item["preference_type"] == "category"
        and item["preference_key"] == product["category"]
        for item in profile["preferences"]
    )
    print("[pass] interaction feedback -> persistent preference")


if __name__ == "__main__":
    verify_query_and_constraints()
    verify_multiturn()
    verify_memory()
    print(json.dumps({"status": "all checks passed"}, ensure_ascii=False))
