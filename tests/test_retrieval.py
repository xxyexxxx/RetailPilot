from dataclasses import replace

from RetailPilot.catalog import add_interaction, get_user_profile, list_brands
from RetailPilot.query import parse_query
from RetailPilot.retrieval import search_engine


def test_hybrid_search_respects_budget_and_negative_brand():
    parsed = parse_query(
        "推荐 50 美元以内的无线耳机，不要 Apple",
        known_brands=list_brands(),
    )
    result = search_engine.search(parsed, user_id="new-user", limit=10)
    assert result["products"]
    assert all(item["price"] <= 50 for item in result["products"])
    assert all("apple" not in item["title"].lower() for item in result["products"])
    assert result["judge"]["pass"]
    timing = next(
        item["timings_ms"]
        for item in result["trace"]
        if item.get("node") == "stage_timings"
    )
    assert {
        "structured_filter",
        "fts_recall",
        "dense_recall",
        "candidate_payload_fetch",
        "candidate_features",
        "rank_and_ground",
    } <= set(timing)
    assert result["retrieval"]["candidate_funnel"]["recalled_candidates"] <= 300


def test_interaction_persists_preference():
    product = search_engine.search(
        "便携快充的手机充电器",
        user_id="new-user",
        limit=1,
        known_brands=list_brands(),
    )["products"][0]
    add_interaction("new-user", product["sku"], "favorite", "快充手机充电器")
    profile = get_user_profile("new-user")
    assert any(
        pref["preference_type"] == "category"
        and pref["preference_key"] == product["category"]
        for pref in profile["preferences"]
    )


def test_search_uses_controlled_attribute_relaxation_when_strict_pool_is_empty():
    base = parse_query(
        "100 美元以内的无线耳机，排除入耳式",
        known_brands=list_brands(),
    )
    # FTS 召回唯一来源是 LLM 检索词（中文数据 → 中文检索词）；
    # must_have_groups 模拟"一组属性不存在、另一组部分命中"触发受控放松。
    parsed = replace(
        base,
        grounded_target_terms=["无线耳机", "耳机"],
        must_have_groups=[
            ["attribute-that-does-not-exist"],
            ["蓝牙", "无线"],
        ],
    )
    result = search_engine.search(parsed, user_id="new-user", limit=10)

    assert result["products"]
    assert result["relaxations"]
    assert all(item["price"] <= 100 for item in result["products"])


def test_underspecified_search_is_blocked_before_global_pool():
    parsed = parse_query("办公 最好的")
    result = search_engine.search(parsed, user_id="new-user", limit=10)
    assert result["products"] == []
    assert result["total_candidates"] == 0
    assert any(item.get("node") == "query_guard" for item in result["trace"])


def test_fictional_domestic_product_still_fails_catalog_evidence_gate():
    result = search_engine.search("想买量子隐形传送鞋，预算500元", user_id="new-user", limit=10)
    assert result["products"] == []
    assert any(
        item.get("node") == "query_guard" and item.get("blocked")
        for item in result["trace"]
    )


def test_approximate_budget_keeps_nearby_exact_catalog_match():
    result = search_engine.search(
        "想给亚克力手机壳染色，要高浓度水性染料，价格30元左右",
        user_id="new-user",
        limit=10,
    )
    assert result["products"]
    assert any(product["price"] <= 40 for product in result["products"])
