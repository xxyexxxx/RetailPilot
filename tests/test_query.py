from RetailPilot.catalog import list_brands
from RetailPilot.query import extract_correction_target, parse_query


def test_scenario_infers_sports_category():
    parsed = parse_query("跑步用的轻便防水装备，100 美元以内")
    assert parsed.category == "Sports & Outdoors"
    assert parsed.price_max == 100
    assert "waterproof" in parsed.concepts


def test_generic_recommendation_requests_clarification():
    parsed = parse_query("给我推荐一个")
    assert parsed.clarification_slot == "category"


def test_constraints_stay_with_one_search_intent():
    text = "预算100美元以内的无线降噪耳机，排除入耳式，最好索尼"
    parsed = parse_query(text, known_brands=list_brands())

    assert parsed.brands == []
    # 品牌提取已移除本地词表，品牌完全交给意图 LLM；本地解析不再输出。
    assert parsed.preferred_brands == []
    assert "replacement" in parsed.excluded_title_terms
    assert any("wireless" in group for group in parsed.must_have_groups)
    assert any("noise cancelling" in group for group in parsed.must_have_groups)
    assert "in ear" in parsed.negative_terms
    assert "earbud" in parsed.negative_terms
    assert "wireless" in parsed.concepts
    assert "noise cancelling" in parsed.concepts
    assert all("earbud" not in concept for concept in parsed.concepts)
    assert "排除入耳式" not in parsed.rewritten_query


def test_accessory_query_disables_primary_product_guard():
    parsed = parse_query("找一个索尼耳机替换耳罩", known_brands=list_brands())
    assert parsed.excluded_title_terms == []


def test_bare_notebook_asks_for_product_type():
    parsed = parse_query("笔记本")
    assert parsed.category is None
    assert parsed.clarification_slot == "product_type"


def test_product_feature_switch_is_not_misread_as_self_correction():
    text = "全自动按摩仪，不用时可以切换成节能模式，价格别超过五百"
    assert extract_correction_target(text) is None
    parsed = parse_query(text)
    assert "按摩仪" in parsed.rewritten_query
    assert parsed.price_max == 500
    assert parsed.sources["catalog_scope"] == "domestic_cny"


def test_unknown_product_query_does_not_fall_back_to_global_catalog():
    parsed = parse_query("办公 最好的")
    assert parsed.clarification_slot == "category"


def test_age_range_in_domestic_product_title_is_not_parsed_as_budget():
    parsed = parse_query("枕头 泰国天然乳胶儿童护颈枕 4-6岁满天星")
    assert parsed.category == "家居家装"
    assert parsed.price_min is None
    assert parsed.price_max is None


def test_grounded_price_range_still_parses():
    parsed = parse_query("预算500-1000元的枕头")
    assert parsed.price_min == 500
    assert parsed.price_max == 1000


def test_money_span_does_not_confuse_specs_with_budget():
    cases = {
        "室外插座额定电流16安，3根线芯，价格35元左右": 35,
        "40克装羊脂膏，接受价格在160元左右": 160,
        "孩子满7岁，买8支装替换刷头，80元能拿下": 80,
        "身高1米3左右，价格预算在300以内": 300,
    }
    for query, expected in cases.items():
        assert parse_query(query).price_max == expected


def test_chinese_number_budget_is_supported():
    assert parse_query("预算两百元左右的夹克").price_max == 200
    assert parse_query("价格不超过九十元的试纸").price_max == 90


def test_large_chinese_and_wan_budgets_are_not_split_into_digits():
    assert parse_query("价格为一千六上下的基因检测").price_max == 1600
    assert parse_query("4万块左右的沙发").price_max == 40000
    parsed = parse_query("价格在1万到1.1万之间的直流电源")
    assert parsed.price_min == 10000
    assert parsed.price_max == 11000


def test_approximate_and_hard_budgets_are_distinguished():
    assert parse_query("30元左右的染料").sources["price_constraint"] == "soft:approximate"
    assert parse_query("30元以内的染料").sources["price_constraint"] == "hard:upper_bound"


def test_apple_body_feature_is_not_parsed_as_apple_brand():
    parsed = parse_query("适合苹果肌的全框眼镜，预算1100元")
    assert "Apple" not in parsed.brands
    assert "apple" not in parsed.concepts


def test_compatibility_brand_is_not_a_hard_product_brand():
    parsed = parse_query("适合 Nespresso 咖啡机的胶囊咖啡，40元左右", known_brands=["NESPRESSO"])
    assert "NESPRESSO" not in parsed.brands
