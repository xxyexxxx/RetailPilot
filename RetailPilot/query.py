from __future__ import annotations

import re
from typing import Any

from .schemas import ParsedQuery
from .taxonomy import (
    ALIAS_SEARCH_TERMS,
    CATEGORY_ALIASES,
    CATEGORY_SEARCH_TERMS,
    CONCEPT_ALIASES,
    MUST_HAVE_CONCEPT_ALIASES,
    PRODUCT_ACCESSORY_EXCLUSIONS,
    SORT_ALIASES,
    alias_in_text,
)
from .text import compact_text


NEGATIVE_MARKERS = ("不要", "排除", "不想要", "不喜欢", "别推荐", "除了", "除外")
PREFERENCE_MARKERS = ("最好", "优先", "偏好", "倾向", "尽量", "首选", "更喜欢")
CHEAPER_WORDS = ("便宜点", "更便宜", "再便宜", "价格低一点", "预算少一点")
SIMILAR_WORDS = ("类似", "相似", "差不多", "同款", "像它")
ACCESSORY_QUERY_MARKERS = (
    "配件",
    "替换",
    "耳罩",
    "耳垫",
    "耳机线",
    "连接线",
    "音频线",
    "保护壳",
    "收纳盒",
    "支架",
    "转接器",
)
COMPUTER_CONTEXT_MARKERS = (
    "办公",
    "商务",
    "电脑",
    "编程",
    "开发",
    "游戏",
    "轻薄",
    "续航",
    "内存",
    "硬盘",
    "处理器",
    "显卡",
    "屏幕",
    "windows",
    "chrome os",
)

_ORDER_ID = re.compile(r"\b(?:JD|A)\d{5,16}\b", re.IGNORECASE)
_ARABIC_OR_CHINESE_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千万]+)"
_MONEY_UNIT = r"(?:人民币|美元|块钱|元|块(?!块)|刀|usd)"
_LEGACY_CATALOG_CATEGORIES = {
    "All Beauty",
    "Appliances",
    "Cell Phones & Accessories",
    "Computers",
    "Electronics",
    "Sports & Outdoors",
    "Video Games",
}


def extract_correction_target(text: str) -> str | None:
    compact = compact_text(text, 1000)
    patterns = (
        r"(?:不是|不要)\s*[^，,。；;！!？?]{1,40}?"
        r"(?:，|,|\s)*(?:而是|是|要的是)\s*([^，,。；;！!？?]{1,40})",
        r"(?:我说的是|我说的|我要的是|我想要的是|我指的是|指的是|应该是|改成|"
        r"(?<!切)(?<!更)(?<!替)(?<!变)(?<!兑)换成)"
        r"\s*([^，,。；;！!？?]{1,40})",
    )
    for pattern in patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if not match:
            continue
        target = match.group(1).strip(" ：:，,。；;！!？?")
        target = re.sub(r"^(?:是|要|买|找|看)\s*", "", target).strip()
        if target and not any(
            phrase in target for phrase in ("为什么", "怎么会", "怎么给", "给我推荐")
        ):
            return target
    return None


def _number_value(raw: str) -> float:
    try:
        return float(raw)
    except ValueError:
        pass
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    # Colloquial shopping budgets commonly omit the final unit: “一千六”
    # means 1600, while “一千零六” still means 1006.
    colloquial = re.fullmatch(r"([一二两三四五六七八九])千([一二三四五六七八九])", raw)
    if colloquial:
        return float(digits[colloquial.group(1)] * 1000 + digits[colloquial.group(2)] * 100)
    units = {"十": 10, "百": 100, "千": 1000, "万": 10000}
    total = 0
    section = 0
    number = 0
    for char in raw:
        if char in digits:
            number = digits[char]
        elif char == "万":
            total += (section + number) * units[char]
            section = 0
            number = 0
        elif char in units:
            section += (number or 1) * units[char]
            number = 0
    return float(total + section + number)


def _scaled_number(raw: str, unit: str | None = None) -> float:
    value = _number_value(raw)
    if unit == "万":
        value *= 10000
    return value


def _parse_price(text: str) -> tuple[float | None, float | None]:
    """Extract a budget only from a locally grounded money span.

    Shopping queries contain many numbers for age, size, current, weight and
    quantity. A money word elsewhere in the sentence must not turn the first
    number into a price.
    """
    number = f"({_ARABIC_OR_CHINESE_NUMBER})"
    range_pattern = re.compile(
        rf"(?P<left>{_ARABIC_OR_CHINESE_NUMBER})\s*(?P<left_scale>万)?\s*"
        rf"(?:-|~|到|至)\s*(?P<right>{_ARABIC_OR_CHINESE_NUMBER})\s*"
        rf"(?P<right_scale>万)?\s*(?P<unit>{_MONEY_UNIT})?",
        re.IGNORECASE,
    )
    for match in range_pattern.finditer(text):
        prefix = text[max(0, match.start() - 12) : match.start()]
        if match.group("unit") or re.search(
            r"预算|价格|价位|售价|不超过|低于|最多", prefix, re.IGNORECASE
        ):
            shared_scale = match.group("right_scale") or match.group("left_scale")
            left = _scaled_number(match.group("left"), match.group("left_scale") or shared_scale)
            right = _scaled_number(match.group("right"), match.group("right_scale") or shared_scale)
            return min(left, right), max(left, right)

    # Explicit currency suffixes are the strongest evidence. Prefer the last
    # one because specifications normally precede the final budget clause.
    suffixed = list(
        re.finditer(
            rf"{number}\s*(?P<scale>万)?\s*(?:{_MONEY_UNIT})\s*"
            rf"(?P<qualifier>以内|以下|内|封顶|左右|上下|出头)?",
            text,
            re.IGNORECASE,
        )
    )
    if suffixed:
        match = suffixed[-1]
        return None, _scaled_number(match.group(1), match.group("scale"))

    # Currency can be omitted after an explicit budget/limit phrase.
    prefixed = list(
        re.finditer(
            rf"(?:价格预算|预算|价格|价位|售价|接受价格|期望价格|"
            rf"不能超过|不超过|不得超过|别超过|不能高于|低于|小于|最多|"
            rf"控制在|限制在|给到)\s*"
            rf"(?:是|为|在|约|大概|差不多|不超过|不能超过|不能高于)?\s*"
            rf"{number}\s*(?P<scale>万)?\s*(?:{_MONEY_UNIT})?\s*"
            rf"(?P<qualifier>以内|以下|内|封顶|左右|上下)?",
            text,
            re.IGNORECASE,
        )
    )
    if prefixed:
        match = prefixed[-1]
        return None, _scaled_number(match.group(1), match.group("scale"))
    return None, None


def _negative_near(text: str, value: str) -> bool:
    position = text.lower().find(value.lower())
    if position < 0:
        return False
    clause_prefix = re.split(r"[，,。；;！!？?]", text[:position])[-1][-12:]
    clause_suffix = re.split(
        r"[，,。；;！!？?]", text[position + len(value) :]
    )[0][:6]
    return any(marker in clause_prefix for marker in NEGATIVE_MARKERS) or "除外" in clause_suffix


def _preference_near(text: str, value: str) -> bool:
    position = text.lower().find(value.lower())
    if position < 0:
        return False
    clause_prefix = re.split(r"[，,。；;！!？?]", text[:position])[-1][-10:]
    clause_suffix = re.split(
        r"[，,。；;！!？?]", text[position + len(value) :]
    )[0][:6]
    return any(
        marker in clause_prefix or marker in clause_suffix
        for marker in PREFERENCE_MARKERS
    )


def _compatibility_near(text: str, value: str) -> bool:
    position = text.lower().find(value.lower())
    if position < 0:
        return False
    clause_prefix = re.split(r"[，,。；;！!？?]", text[:position])[-1][-12:]
    return any(marker in clause_prefix for marker in ("适合", "适用于", "兼容", "适配", "用于", "给"))


def _extract_category(text: str) -> tuple[str | None, str | None]:
    lowered = text.lower()
    # 长别名优先：避免"手机壳"因含"手机"而先命中短别名（虽常同类目，
    # 但保持正确性）；单字/极短别名要求是独立词元，避免"说明书"被单字
    # "书"命中、或"手机壳"被"手机"命中这类跨词误判。
    for alias, category in sorted(
        CATEGORY_ALIASES.items(), key=lambda item: -len(item[0])
    ):
        alias_low = alias.lower()
        if alias_low not in lowered:
            continue
        if len(alias) == 1:
            # 单字别名必须是独立词元（"书"在"ai相关的书"中是独立词，
            # 在"说明书"中不是），否则会因子串匹配误伤。
            if alias not in _segment_words(text):
                continue
        return category, alias
    return None, None


def _segment_words(text: str) -> set[str]:
    """jieba 分词结果集合，用于单字类别别名的独立词元判定。"""
    try:
        import jieba
    except ImportError:  # pragma: no cover - jieba 是运行时依赖
        return {text}
    return set(jieba.lcut(text))


def _positive_query_text(text: str) -> str:
    clauses = [
        clause.strip()
        for clause in re.split(r"[，,。；;！!？?]", text)
        if clause and clause.strip()
    ]
    positive = [
        clause
        for clause in clauses
        if not any(marker in clause for marker in NEGATIVE_MARKERS)
    ]
    return " ".join(positive) or text


def parse_query(
    text: str,
    context: dict[str, Any] | None = None,
    known_brands: list[str] | None = None,
) -> ParsedQuery:
    context = context or {}
    known_brands = known_brands or []
    original_raw = compact_text(text, 1000)
    correction_target = extract_correction_target(original_raw)
    raw = correction_target or original_raw
    intent = "search"
    category, category_alias = _extract_category(raw)
    # "美元" contains the CJK character “元”, so a bare substring check would
    # incorrectly route USD queries into the domestic catalogue.
    cny_catalog = bool(
        re.search(r"人民币|块钱|块|(?<!美)元", raw, re.IGNORECASE)
        or (
            re.search(r"预算|价格|价钱|售价|别超过|不超过", raw)
            and re.search(r"[零〇一二两三四五六七八九十百千万]", raw)
            and not re.search(r"美元|美刀|usd", raw, re.IGNORECASE)
        )
    )
    legacy_category_hint: str | None = None
    legacy_alias_hint: str | None = None
    if cny_catalog and category in _LEGACY_CATALOG_CATEGORIES:
        # The same Chinese word can denote an Amazon-style product type or a
        # use/material mention in a domestic long-tail query. Keep it as a hint
        # but do not use it as a hard SQL category filter.
        legacy_category_hint = category
        legacy_alias_hint = category_alias
        category = None
        category_alias = None
    ambiguous_notebook = (
        "笔记本" in raw
        and "笔记本电脑" not in raw
        and not any(marker.lower() in raw.lower() for marker in COMPUTER_CONTEXT_MARKERS)
    )
    if ambiguous_notebook:
        category = None
        category_alias = None
    normalized_target = re.sub(r"[\s，,。；;！!？?]", "", raw)
    ambiguous_pen = bool(
        re.fullmatch(r"(?:我想|我要|想要)?(?:买|找|推荐|搜)?(?:一支|一只|一个)?笔", normalized_target)
    )
    price_min, price_max = _parse_price(raw)
    # 品牌提取完全交给意图 LLM（constraints.brands / negative_brands），
    # 不再用本地 COMMON_BRANDS 词表死规则判断"要/不要"——词表覆盖不全，
    # 且"不要华为"这类否定依赖 LLM 理解，程序判断必然不准。
    positive_brands: list[str] = []
    preferred_brands: list[str] = []
    negative_brands: list[str] = []
    sources: dict[str, str] = {}
    if ambiguous_notebook:
        sources["ambiguity"] = "query:ambiguous_notebook"
    if ambiguous_pen:
        sources["ambiguity"] = "query:ambiguous_pen"
    if cny_catalog:
        sources["catalog_scope"] = "domestic_cny"
        sources["currency"] = "query:cny"
    if legacy_category_hint:
        sources["legacy_category_hint"] = legacy_category_hint
        sources["legacy_alias_hint"] = str(legacy_alias_hint or "")
    if correction_target:
        sources["correction"] = "query:explicit_self_correction"
        sources["correction_target"] = correction_target
    concepts: list[str] = []
    negative_terms: list[str] = []
    must_have_groups: list[list[str]] = []
    for alias, terms in CONCEPT_ALIASES.items():
        if not alias_in_text(raw, alias):
            continue
        if alias == "苹果" and any(
            phrase in raw for phrase in ("苹果肌", "苹果酸", "苹果味", "苹果绿", "苹果树", "苹果汁")
        ):
            continue
        if _negative_near(raw, alias):
            negative_terms.extend(terms)
        else:
            # The alias itself is the canonical concept form (e.g. the Chinese
            # "人工智能"); its term table holds related search terms. Keeping
            # the key guarantees Chinese bigram recall for Chinese titles.
            concepts.append(alias)
            concepts.extend(terms)
            if (
                alias in MUST_HAVE_CONCEPT_ALIASES
                and not _preference_near(raw, alias)
            ):
                must_have_groups.append(terms)
    if cny_catalog:
        # English Amazon concept expansions are not guaranteed to exist in
        # Chinese ShopSimulator attributes and must not become hard filters.
        must_have_groups = []
    concepts = list(dict.fromkeys(concepts))
    negative_terms = list(dict.fromkeys(negative_terms))
    negative_terms = list(dict.fromkeys([*negative_brands, *negative_terms]))
    unique_groups: list[list[str]] = []
    seen_groups: set[tuple[str, ...]] = set()
    for group in must_have_groups:
        key = tuple(sorted(term.lower() for term in group))
        if key not in seen_groups:
            seen_groups.add(key)
            unique_groups.append(group)
    must_have_groups = unique_groups
    excluded_title_terms = (
        []
        if any(marker in raw for marker in ACCESSORY_QUERY_MARKERS)
        else list(PRODUCT_ACCESSORY_EXCLUSIONS.get(category_alias or "", []))
    )
    if category_alias:
        concepts = list(
            dict.fromkeys([*ALIAS_SEARCH_TERMS.get(category_alias, []), *concepts])
        )
    concepts = [
        concept
        for concept in concepts
        if not any(term.lower() in concept.lower() for term in negative_terms)
    ]
    if (
        not category
        and not cny_catalog
        and any(word in raw for word in ("跑步", "健身", "运动装备", "户外用品"))
    ):
        category = "Sports & Outdoors"
        sources["category"] = "query:scenario_inference"

    if category and "category" not in sources:
        sources["category"] = f"query:{category_alias}"
    if price_max is not None:
        sources["price_max"] = "query:explicit_budget"
        has_explicit_range = bool(
            re.search(
                rf"{_ARABIC_OR_CHINESE_NUMBER}\s*(?:万)?\s*(?:-|~|到|至)\s*"
                rf"{_ARABIC_OR_CHINESE_NUMBER}",
                raw,
            )
        )
        if re.search(r"左右|上下|差不多|大概|约|出头", raw) or (
            has_explicit_range
            and not re.search(r"不超过|不得超过|以内|以下|封顶|最多", raw)
        ):
            sources["price_constraint"] = "soft:approximate"
        else:
            sources["price_constraint"] = "hard:upper_bound"
    if concepts:
        sources["concepts"] = "query:concept_expansion"
    if excluded_title_terms:
        sources["excluded_title_terms"] = "query:primary_product_guard"
    if must_have_groups:
        sources["must_have_groups"] = "query:explicit_attributes"
    if positive_brands:
        sources["brands"] = "query:brand"
    if preferred_brands:
        sources["preferred_brands"] = "query:soft_brand_preference"
    if negative_brands:
        sources["negative_brands"] = "query:negative_brand"

    last_parsed = context.get("last_parsed_query") or {}
    if correction_target and last_parsed:
        inherited: list[str] = []
        if price_min is None and last_parsed.get("price_min") is not None:
            price_min = float(last_parsed["price_min"])
            inherited.append("price_min")
        if price_max is None and last_parsed.get("price_max") is not None:
            price_max = float(last_parsed["price_max"])
            inherited.append("price_max")
        previous_concepts = list(last_parsed.get("concepts") or [])
        if previous_concepts:
            concepts = list(dict.fromkeys([*concepts, *previous_concepts]))
            inherited.append("concepts")
        previous_groups = list(last_parsed.get("must_have_groups") or [])
        if previous_groups:
            must_have_groups = [*must_have_groups, *previous_groups]
            inherited.append("must_have_groups")
        if not preferred_brands and last_parsed.get("preferred_brands"):
            preferred_brands = list(last_parsed["preferred_brands"])
            inherited.append("preferred_brands")
        if not negative_brands and last_parsed.get("negative_brands"):
            negative_brands = list(last_parsed["negative_brands"])
            inherited.append("negative_brands")
        if last_parsed.get("negative_terms"):
            negative_terms = list(
                dict.fromkeys([*negative_terms, *last_parsed["negative_terms"]])
            )
            inherited.append("negative_terms")
        negative_terms = list(dict.fromkeys([*negative_brands, *negative_terms]))
        deduped_groups: list[list[str]] = []
        seen_group_keys: set[tuple[str, ...]] = set()
        for group in must_have_groups:
            key = tuple(sorted(term.lower() for term in group))
            if key not in seen_group_keys:
                seen_group_keys.add(key)
                deduped_groups.append(group)
        must_have_groups = deduped_groups
        if inherited:
            sources["inherited_constraints"] = (
                "session:last_query:" + ",".join(dict.fromkeys(inherited))
            )
    if not category and any(word in raw for word in (*CHEAPER_WORDS, *SIMILAR_WORDS, "再来几个")):
        category = last_parsed.get("category")
        if category:
            sources["category"] = "session:last_category"
    if price_max is None and any(word in raw for word in CHEAPER_WORDS):
        last_prices = [
            float(value)
            for value in (context.get("last_result_prices") or [])
            if isinstance(value, (int, float))
        ]
        previous_budget = last_parsed.get("price_max")
        if previous_budget:
            price_max = round(float(previous_budget) * 0.8, 2)
        elif last_prices:
            price_max = round(sorted(last_prices)[len(last_prices) // 2] * 0.8, 2)
        if price_max:
            sources["price_max"] = "session:cheaper_followup"
    if not concepts and any(word in raw for word in (*SIMILAR_WORDS, *CHEAPER_WORDS)):
        concepts = list(last_parsed.get("concepts") or [])
        if concepts:
            sources["concepts"] = "session:last_concepts"

    sort = "relevance"
    for alias, value in SORT_ALIASES.items():
        if alias in raw:
            sort = value
            break
    if correction_target and sort == "relevance" and last_parsed.get("sort"):
        sort = str(last_parsed["sort"])
        sources["sort"] = "session:last_query"

    terms = [_positive_query_text(raw)]
    if category:
        terms.extend(CATEGORY_SEARCH_TERMS.get(category, []))
    terms.extend(concepts)
    if positive_brands:
        terms.extend(positive_brands)
    if preferred_brands:
        terms.extend(preferred_brands)
    rewritten = " ".join(dict.fromkeys(compact_text(term, 200) for term in terms if term))

    clarification_slot = None
    clarification_question = None
    if intent == "search" and ambiguous_notebook:
        clarification_slot = "product_type"
        clarification_question = "您说的“笔记本”是笔记本电脑，还是纸质笔记本？"
    if intent == "search" and ambiguous_pen:
        clarification_slot = "product_type"
        clarification_question = "您想买书写笔、触控笔，还是水质测试笔/美妆笔？请再说明一下用途。"
    if (
        intent == "search"
        and not clarification_slot
        and not category
        and not positive_brands
        and not preferred_brands
    ):
        clarification_slot = "category"
        clarification_question = (
            "我还不能确定具体商品类型。您想买什么商品？"
            "比如笔记本电脑、无线耳机、护肤品或运动装备。"
        )
    # 指代解析由意图 LLM 负责（本地序数词规则已删除）。detail/compare 需要
    # 至少一个可引用的历史商品；状态表没有时兜底澄清，避免空转。
    available_skus = context.get("last_result_skus") or []
    if intent == "compare" and len(available_skus) < 2:
        clarification_slot = "products"
        clarification_question = "您想比较哪几款？可以先搜索，再从结果里勾选 2 到 4 款。"
    if intent == "detail" and not available_skus:
        clarification_slot = "product"
        clarification_question = "您问的是哪一款商品？可以先搜索，再从结果里选择。"

    return ParsedQuery(
        raw_query=original_raw,
        rewritten_query=rewritten,
        intent=intent,
        category=category,
        price_min=price_min,
        price_max=price_max,
        brands=positive_brands,
        preferred_brands=preferred_brands,
        negative_brands=negative_brands,
        excluded_title_terms=excluded_title_terms,
        must_have_groups=must_have_groups,
        concepts=concepts,
        negative_terms=negative_terms,
        sort=sort,
        sources=sources,
        confidence=0.6 if clarification_slot else 0.95,
        clarification_slot=clarification_slot,
        clarification_question=clarification_question,
    )


def extract_order_id(text: str) -> str | None:
    match = _ORDER_ID.search(text)
    return match.group(0).upper() if match else None


def matched_concept_aliases(text: str) -> list[str]:
    return [alias for alias in CONCEPT_ALIASES if alias_in_text(text, alias)]
