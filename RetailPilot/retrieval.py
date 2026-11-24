from __future__ import annotations

import hashlib
import math
import re
import time
import uuid
from collections import defaultdict
from dataclasses import replace
from functools import lru_cache
from typing import Any

from .asset_config import asset_settings
from .asset_evidence import meaningful_overlap, verify_image
from .catalog import (
    format_price,
    hydrate_product,
    interaction_scores,
    preference_map,
)
from .config import settings
from .database import db_session, json_dumps
from .knowledge_evidence import hydrate, public_hit
from .knowledge_index import model_spec
from .knowledge_index import version as knowledge_version
from .knowledge_retrieval import (
    RetrievalFilters,
    RetrievalQuery,
    aggregate_skus,
    retrieve,
)
from .query import parse_query
from .schemas import ParsedQuery
from .taxonomy import correction_candidates
from .text import fast_catalog_tokens, tokenize

TITLE_SCOPED_NEGATIVE_TERMS = {
    "in ear",
    "in-ear",
    "earbud",
    "earbuds",
    "earphone",
}


def _group_filter_terms(group: list[str]) -> list[str]:
    precise = [term for term in group if len(term.strip()) > 3]
    return precise or group


def _sql_filters(parsed: ParsedQuery, *, level: str = "hard") -> tuple[str, list[Any]]:
    """Build either reliable hard filters or the legacy strict text filter.

    Neural/BM25 recall must see the hard pool. Product-type aliases, raw text
    attributes and accessory heuristics are incomplete metadata and are applied
    after candidate generation where they can be traced and relaxed.
    """
    clauses = ["1 = 1"]
    params: list[Any] = []
    searchable = (
        "LOWER(p.title || ' ' || p.tags_text || ' ' || "
        "p.description || ' ' || p.attributes_text)"
    )
    if parsed.sources.get("catalog_scope") == "domestic_cny":
        clauses.append("p.sku LIKE 'tb-%'")
    if parsed.category:
        clauses.append("p.category = ?")
        params.append(parsed.category)
    if (
        parsed.price_min is not None
        and parsed.sources.get("catalog_scope") != "domestic_cny"
        and parsed.sources.get("price_constraint") != "soft:approximate"
    ):
        clauses.append("p.price >= ?")
        params.append(parsed.price_min)
    if parsed.price_max is not None:
        if parsed.sources.get("price_constraint") == "soft:approximate":
            # Natural-language "around" budgets are preferences. A small
            # catalog tolerance prevents 30-vs-31 rounding from deleting the
            # exact product while still bounding the candidate pool.
            clauses.append("p.price <= ?")
            params.append(max(parsed.price_max * 1.35, parsed.price_max + 10.0))
        else:
            clauses.append("p.price <= ?")
            params.append(parsed.price_max)
    if parsed.brands:
        placeholders = ",".join("?" for _ in parsed.brands)
        clauses.append(f"p.brand IN ({placeholders})")
        params.extend(parsed.brands)
    if parsed.negative_brands:
        placeholders = ",".join("?" for _ in parsed.negative_brands)
        clauses.append(f"p.brand NOT IN ({placeholders})")
        params.extend(parsed.negative_brands)
    if level == "strict":
        for term in parsed.excluded_title_terms:
            clauses.append("LOWER(p.title) NOT LIKE ?")
            params.append(f"%{term.lower()}%")
        for group in parsed.must_have_groups:
            filter_terms = _group_filter_terms(group)
            if not filter_terms:
                continue
            clauses.append(
                "(" + " OR ".join(f"{searchable} LIKE ?" for _ in filter_terms) + ")"
            )
            params.extend(f"%{term.lower()}%" for term in filter_terms)
        for term in parsed.negative_terms[:8]:
            target = (
                "LOWER(p.title)"
                if term.lower() in TITLE_SCOPED_NEGATIVE_TERMS
                else searchable
            )
            clauses.append(f"{target} NOT LIKE ?")
            params.append(f"%{term.lower()}%")
    return " AND ".join(clauses), params


def _fts_expression(text: str) -> str:
    tokens = list(dict.fromkeys(tokenize(text)))[:28]
    safe = [token.replace('"', '""') for token in tokens if token]
    return " OR ".join(f'"{token}"' for token in safe)


def _structured_skus(
    parsed: ParsedQuery,
    limit: int | None = None,
    *,
    level: str = "hard",
) -> list[str]:
    where, params = _sql_filters(parsed, level=level)
    effective_limit = settings.candidate_pool_limit if limit is None else limit
    query = f"""
            SELECT p.sku
            FROM products p
            LEFT JOIN product_commerce c ON c.sku = p.sku
            WHERE ({where})
            AND COALESCE(c.dataset_split, '') != 'final'
            AND p.source != 'Amazon Reviews 2023'
    """
    query_params: list[Any] = [*params]
    if effective_limit is not None and effective_limit > 0:
        query += "\n            ORDER BY p.quality_score DESC, p.popularity_score DESC LIMIT ?"
        query_params.append(effective_limit)
    with db_session() as connection:
        rows = connection.execute(query, query_params).fetchall()
    return [str(row["sku"]) for row in rows]


def _products_for_skus(skus: list[str]) -> list[dict[str, Any]]:
    if not skus:
        return []
    placeholders = ",".join("?" for _ in skus)
    with db_session() as connection:
        rows = connection.execute(
            f"""
            SELECT p.*,
                   COALESCE(c.platform, '') AS platform,
                   COALESCE(c.platform_product_id, '') AS platform_product_id,
                   COALESCE(c.spu_id, p.sku) AS spu_id,
                   COALESCE(c.seller_name, '') AS seller_name,
                   COALESCE(c.currency, 'USD') AS currency,
                   COALESCE(c.category_path, p.category_label) AS category_path,
                   COALESCE(c.variants_json, '{{}}') AS variants_json,
                   COALESCE(c.services_json, '[]') AS services_json,
                   COALESCE(c.provenance_json, '{{}}') AS provenance_json,
                   COALESCE(c.dataset_split, '') AS dataset_split,
                   COALESCE(c.license_note, '') AS license_note
            FROM products p
            LEFT JOIN product_commerce c ON c.sku = p.sku
            WHERE p.sku IN ({placeholders})
            """,
            skus,
        ).fetchall()
    by_sku = {row["sku"]: dict(row) for row in rows}
    return [by_sku[sku] for sku in skus if sku in by_sku]


_OPEN_QUERY_NOISE = re.compile(
    r"给我|帮我|请问|麻烦|推荐|选一款|找一款|找一个|"
    r"我要买|我想给|我想|想给|想买|想要|需要|还要|"
    r"有没有|有推荐吗|哪有|价格|价位|预算|售价|不超过|不能超过|左右|以内|"
    r"元|块钱|块|人民币|商品|产品|东西|一个|一种|不存在|"
    r"可以|能够|使用|用的|功能|具有|具备|要求|最好|正品"
)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


@lru_cache(maxsize=2048)
def _open_query_features(raw_query: str) -> tuple[tuple[str, ...], frozenset[str]]:
    query = _OPEN_QUERY_NOISE.sub(" ", raw_query)
    runs = tuple(_CJK_RUN.findall(query))
    terms = frozenset(
        term.casefold()
        for term in tokenize(query, include_expansion=False)
        if not term.isdigit() and len(term) >= 2
    )
    return runs, terms


def _catalog_lexical_evidence(
    parsed: ParsedQuery, product: dict[str, Any]
) -> dict[str, Any]:
    """Require coherent catalog evidence before an open long-tail query is admitted."""
    target = " ".join(
        [
            str(product.get("title") or ""),
            str(product.get("tags_text") or ""),
            str(product.get("attributes_text") or "")[:2400],
        ]
    )
    query_runs, query_terms = _open_query_features(parsed.raw_query)
    target_text = "".join(_CJK_RUN.findall(target))
    matched_four: set[str] = set()
    matched_three: set[str] = set()
    # Product imports already materialize CJK bigrams. This avoids invoking
    # jieba hundreds of times for one open-catalog request.
    target_terms = set(fast_catalog_tokens(target))
    matched_terms = query_terms & target_terms
    for run in query_runs:
        matched_four.update(
            run[index : index + 4]
            for index in range(max(0, len(run) - 3))
            if run[index : index + 4] in target_text
        )
        matched_three.update(
            run[index : index + 3]
            for index in range(max(0, len(run) - 2))
            if run[index : index + 3] in target_text
        )
    # Several independent catalog tokens can ground a target even when query
    # and title use different word order (e.g. "意式 沙发 真皮").
    passed = bool(matched_four) or len(matched_three) >= 2 or len(matched_terms) >= 3
    return {
        "pass": passed,
        "matched_fourgrams": sorted(matched_four)[:8],
        "matched_trigrams": sorted(matched_three)[:12],
        "matched_terms": sorted(matched_terms, key=lambda value: (-len(value), value))[
            :12
        ],
    }


def _needs_product_anchor(parsed: ParsedQuery) -> bool:
    return not any(
        (
            parsed.category,
            parsed.grounded_target_terms,
            parsed.brands,
            parsed.reference_skus,
            parsed.sources.get("catalog_anchor"),
        )
    )


def _preference_score(
    row: dict[str, Any],
    preferences: dict[str, dict[str, float]],
    parsed: ParsedQuery,
) -> float:
    category_score = preferences.get("category", {}).get(row["category"], 0.0)
    brand_score = preferences.get("brand", {}).get(row["brand"], 0.0)
    requested_brand_score = 0.45 if row["brand"] in parsed.preferred_brands else 0.0
    haystack = " ".join(
        [
            row["title"],
            row["tags_text"],
            row["description"][:1200],
            row["attributes_text"][:1200],
        ]
    ).lower()
    concept_score = sum(
        weight
        for concept, weight in preferences.get("concept", {}).items()
        if concept.lower() in haystack
    )
    requested = sum(1 for concept in parsed.concepts if concept.lower() in haystack)
    return math.tanh(
        category_score
        + brand_score
        + requested_brand_score
        + concept_score
        + requested * 0.12
    )


def _personal_recall(
    user_id: str,
    parsed: ParsedQuery,
    pool: list[dict[str, Any]],
    limit: int = 240,
) -> tuple[list[str], dict[str, float]]:
    preferences = preference_map(user_id)
    explicit_scores = {
        row["sku"]: _preference_score(row, preferences, parsed) for row in pool
    }
    history_scores = interaction_scores(user_id, list(explicit_scores))
    combined = {
        sku: explicit_scores.get(sku, 0.0) * 0.75 + history_scores.get(sku, 0.0) * 0.25
        for sku in explicit_scores
    }
    ranked = sorted(combined, key=combined.get, reverse=True)[:limit]
    return ranked, combined


def _term_match_score(
    parsed: ParsedQuery, product: dict[str, Any]
) -> tuple[float, list[str]]:
    # category 是脏标签（Amazon 数据常把无关商品错标进一个大类，如调料被标成
    # Cell Phones & Accessories），不能作为"内容匹配"证据：category 含 "Phone"
    # 会让错标调料命中 "phone"。匹配只应基于真实内容（title/brand/description/
    # attributes），category 仅作 SQL 过滤标签，不作为词匹配依据。
    haystack = " ".join(
        [
            product["title"],
            product["brand"],
            product["tags_text"],
            product["description"][:1600],
            product["attributes_text"][:1600],
        ]
    ).lower()
    negative_tokens = set(
        tokenize(" ".join(parsed.negative_terms), include_expansion=False)
    )
    llm_concept_values = {term.casefold() for term in parsed.llm_concepts}
    deterministic_concepts = [
        term for term in parsed.concepts if term.casefold() not in llm_concept_values
    ]
    primary_terms = [
        term
        for term in list(
            dict.fromkeys(
                tokenize(
                    " ".join(
                        [
                            *parsed.grounded_target_terms,
                            *deterministic_concepts,
                        ]
                    )
                )
            )
        )[:28]
        if term not in negative_tokens
    ]
    llm_terms = [
        term
        for term in list(dict.fromkeys(tokenize(" ".join(parsed.llm_concepts))))[:16]
        if term not in negative_tokens and term not in primary_terms
    ]
    lexical_terms = [
        term
        for term in list(dict.fromkeys(tokenize(parsed.raw_query)))[:20]
        if not term.isdigit()
    ]
    primary_matched = [term for term in primary_terms if term.lower() in haystack]
    llm_matched = [term for term in llm_terms if term.lower() in haystack]
    lexical_matched = [term for term in lexical_terms if term.lower() in haystack]
    lexical_score = len(lexical_matched) / max(1, len(lexical_terms))
    llm_score = len(llm_matched) / max(1, len(llm_terms)) if llm_terms else 0.0
    if primary_terms:
        score = 0.72 * len(primary_matched) / len(primary_terms)
        score += 0.20 * lexical_score + 0.08 * llm_score
    elif llm_terms:
        score = 0.55 * llm_score + 0.45 * lexical_score
    else:
        score = lexical_score
    matched = list(dict.fromkeys([*primary_matched, *lexical_matched, *llm_matched]))
    return score, matched[:8]


def _passes_non_relaxable_constraints(
    parsed: ParsedQuery, product: dict[str, Any]
) -> bool:
    title = product["title"].casefold()
    searchable = " ".join(
        [
            product["title"],
            product["tags_text"],
            product["description"][:2000],
            product["attributes_text"][:2000],
        ]
    ).casefold()
    if any(term.casefold() in title for term in parsed.excluded_title_terms):
        return False
    asset_text = " ".join(
        item.get("text_content", "") for item in product.get("asset_evidence") or []
    ).casefold()
    for term in parsed.negative_terms[:8]:
        target = (
            title
            if term.casefold() in TITLE_SCOPED_NEGATIVE_TERMS
            else f"{searchable} {asset_text}"
        )
        if term.casefold() in target:
            return False
    return True


def _must_have_score(parsed: ParsedQuery, product: dict[str, Any]) -> float:
    if not parsed.must_have_groups:
        return 1.0
    haystack = " ".join(
        [
            product["title"],
            product["tags_text"],
            product["description"][:1600],
            product["attributes_text"][:1600],
        ]
    ).lower()
    haystack += " " + product.get("visual_observation", "").lower()
    if product.get("asset_evidence"):
        haystack += (
            " "
            + " ".join(
                item.get("text_content", "") for item in product["asset_evidence"]
            ).lower()
        )
    matched = sum(
        1
        for group in parsed.must_have_groups
        if any(term.lower() in haystack for term in _group_filter_terms(group))
    )
    return matched / len(parsed.must_have_groups)


def _apply_candidate_policy(
    parsed: ParsedQuery,
    products: list[dict[str, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    non_relaxable: list[dict[str, Any]] = []
    for product in products:
        semantic_match = bool(product.get("knowledge_evidence"))
        product["constraint_profile"] = {
            "non_relaxable_pass": _passes_non_relaxable_constraints(parsed, product),
            "semantic_match": semantic_match,
            "must_have_match": product.get("score_breakdown", {}).get("must_have", 1.0),
        }
        if not product["constraint_profile"]["non_relaxable_pass"]:
            continue
        non_relaxable.append(product)
    # 相关性放行由召回阶段（FTS 全文 / Dense 向量）本身保证，不再叠加第二层
    # 词面命中/语义/品牌/类目来源校验——那与 SQL 硬过滤、召回匹配重复。
    relevant = non_relaxable

    strict = [
        product
        for product in relevant
        if float(product["constraint_profile"]["must_have_match"]) >= 1.0
    ]
    selected_pool = strict
    relaxations: list[dict[str, Any]] = []
    minimum_groups_matched = len(parsed.must_have_groups)
    if parsed.must_have_groups and len(strict) < limit:
        minimum_groups_matched = max(0, len(parsed.must_have_groups) - 1)
        minimum_score = minimum_groups_matched / len(parsed.must_have_groups)
        strict_skus = {product["sku"] for product in strict}
        near = [
            product
            for product in relevant
            if product["sku"] not in strict_skus
            and float(product["constraint_profile"]["must_have_match"]) >= minimum_score
        ]
        selected_pool = [*strict, *near]
        if near:
            relaxations.append(
                {
                    "type": "low_candidate_attribute_relaxation",
                    "message": "严格属性候选不足，保留全部硬约束，并补充最多缺少一组文本属性的近似候选。",
                    "required_groups": parsed.must_have_groups,
                    "strict_candidates": len(strict),
                    "minimum_groups_matched": minimum_groups_matched,
                }
            )

    funnel = {
        "recalled_candidates": len(products),
        "non_relaxable_pass": len(non_relaxable),
        "relevance_pass": len(relevant),
        "attribute_strict": len(strict),
        "attribute_policy_pass": len(selected_pool),
        "minimum_groups_matched": minimum_groups_matched,
    }
    return selected_pool, relaxations, funnel


def _value_score(product: dict[str, Any]) -> float:
    price = max(1.0, float(product["price"]))
    return min(
        1.0,
        float(product["quality_score"]) * 0.6
        + float(product["popularity_score"]) * 0.2
        + 0.2 / math.log10(price + 10),
    )


def _score_candidates(
    parsed: ParsedQuery,
    user_id: str,
    pool_by_sku: dict[str, dict[str, Any]],
    routes: dict[str, list[str]],
    dense_scores: dict[str, float],
    personal_scores: dict[str, float],
    mode: str,
) -> list[dict[str, Any]]:
    fused = routes
    max_fused = max(fused.values(), default=1.0)
    interaction = (
        interaction_scores(user_id, list(pool_by_sku)) if mode == "hybrid" else {}
    )
    ranked: list[dict[str, Any]] = []
    for sku, route_score in fused.items():
        row = pool_by_sku.get(sku)
        if row is None:
            continue
        product = hydrate_product(row)
        product["evidence_assessment"] = row.get("evidence_assessment")
        product["knowledge_evidence"] = row.get("knowledge_evidence", [])
        product["visual_observation"] = row.get("visual_observation", "")
        product["asset_evidence"] = row.get("asset_evidence", [])
        match_score, matched_terms = _term_match_score(parsed, product)
        quality = float(product["quality_score"])
        popularity = float(product["popularity_score"])
        personal = (
            max(
                -1.0,
                min(
                    1.0,
                    personal_scores.get(sku, 0.0) + interaction.get(sku, 0.0) * 0.2,
                ),
            )
            if mode == "hybrid"
            else 0.0
        )
        dense = dense_scores.get(sku, 0.0)
        rrf_score = route_score / max_fused
        value = _value_score(product)
        must_have = _must_have_score(parsed, product)
        preferred_brand = 1.0 if product["brand"] in parsed.preferred_brands else 0.0

        if mode == "fts":
            final = rrf_score * 0.72 + quality * 0.18 + popularity * 0.10
        else:
            final = (
                rrf_score * 0.34
                + dense * 0.18
                + match_score * 0.18
                + quality * 0.11
                + popularity * 0.07
                + ((personal + 1) / 2) * 0.08
                + value * 0.04
                + must_have * 0.10
                + preferred_brand * 0.12
            )
        if parsed.sort == "price_asc":
            final += 0.18 / math.log10(float(product["price"]) + 10)
        elif parsed.sort == "rating_desc":
            final += quality * 0.28
        elif parsed.sort == "popular_desc":
            final += popularity * 0.12
        elif parsed.sort == "value_desc":
            final += value * 0.12

        product["score"] = round(final, 6)
        product["score_breakdown"] = {
            "fusion": round(rrf_score, 4),
            "dense": round(dense, 4),
            "term_match": round(match_score, 4),
            "quality": round(quality, 4),
            "popularity": round(popularity, 4),
            "personalization": round(personal, 4),
            "value": round(value, 4),
            "must_have": round(must_have, 4),
            "preferred_brand": preferred_brand,
        }
        product["matched_terms"] = matched_terms
        ranked.append(product)
    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _diversify(
    products: list[dict[str, Any]], limit: int, single_category: bool
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    candidates = products[: max(limit * 5, 40)]
    while candidates and len(selected) < limit:
        best = None
        best_adjusted = -10.0
        for product in candidates:
            brand_penalty = sum(
                0.035
                for existing in selected
                if existing["brand"] and existing["brand"] == product["brand"]
            )
            category_penalty = (
                0
                if single_category
                else sum(
                    0.02
                    for existing in selected
                    if existing["category"] == product["category"]
                )
            )
            adjusted = float(product["score"]) - brand_penalty - category_penalty
            if adjusted > best_adjusted:
                best = product
                best_adjusted = adjusted
        if best is None:
            break
        selected.append(best)
        candidates.remove(best)
    return selected


def _reason(
    parsed: ParsedQuery, product: dict[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    pieces: list[str] = []
    citations: list[dict[str, Any]] = []
    if product["matched_terms"]:
        pieces.append(f"命中 {', '.join(product['matched_terms'][:3])}")
    must_have = float(product.get("score_breakdown", {}).get("must_have", 1.0))
    if parsed.must_have_groups and must_have < 1.0:
        matched = round(must_have * len(parsed.must_have_groups))
        pieces.append(
            f"近似候选，命中 {matched}/{len(parsed.must_have_groups)} 组关键属性"
        )
    if product["brand"] in parsed.preferred_brands:
        pieces.append(f"品牌符合偏好 {product['brand']}")
    if product["rating"]:
        pieces.append(
            f"评分 {product['rating']:.1f}，共 {product['rating_count']} 条评价"
        )
        citations.append(
            {
                "sku": product["sku"],
                "field": "rating",
                "value": product["rating"],
                "source": product["source"],
            }
        )
    pieces.append(f"价格 {format_price(product)}")
    citations.append(
        {
            "sku": product["sku"],
            "field": "price",
            "value": product["price"],
            "source": product["source"],
        }
    )
    feature = next(
        (
            item
            for item in product["features"]
            if any(term.lower() in item.lower() for term in product["matched_terms"])
        ),
        product["features"][0] if product["features"] else "",
    )
    if feature:
        pieces.append(feature[:120])
        citations.append(
            {
                "sku": product["sku"],
                "field": "features",
                "value": feature,
                "source": product["source"],
            }
        )
    if parsed.price_max is not None:
        budget_currency = str(product.get("currency") or "USD")
        pieces.append(
            f"符合不超过 {format_price(parsed.price_max, budget_currency)} 的预算"
        )
    return "；".join(pieces[:4]), citations


def validate_citations(
    product: dict[str, Any], citations: list[dict[str, Any]]
) -> dict[str, Any]:
    issues: list[str] = []
    for citation in citations:
        field = citation["field"]
        value = citation["value"]
        if field == "features":
            if value not in product.get("features", []):
                issues.append(f"{product['sku']} feature citation not found")
        elif field in product:
            actual = product[field]
            if isinstance(actual, float):
                if abs(float(actual) - float(value)) > 1e-6:
                    issues.append(f"{product['sku']} {field} mismatch")
            elif actual != value:
                issues.append(f"{product['sku']} {field} mismatch")
        else:
            issues.append(f"{product['sku']} missing field {field}")
    return {"pass": not issues, "issues": issues}


def _facets(products: list[dict[str, Any]]) -> dict[str, Any]:
    categories: dict[str, int] = defaultdict(int)
    brands: dict[str, int] = defaultdict(int)
    prices = []
    for product in products:
        categories[product["category_label"]] += 1
        if product["brand"]:
            brands[product["brand"]] += 1
        prices.append(float(product["price"]))
    return {
        "categories": dict(sorted(categories.items(), key=lambda item: -item[1])),
        "brands": dict(sorted(brands.items(), key=lambda item: -item[1])[:12]),
        "price": {
            "min": round(min(prices), 2) if prices else None,
            "max": round(max(prices), 2) if prices else None,
        },
    }


def _suggestions(parsed: ParsedQuery, products: list[dict[str, Any]]) -> list[str]:
    suggestions: list[str] = []
    if products:
        suggestions.extend(
            [
                "对比前两款",
                "只看评分 4.5 以上",
                "再便宜一点",
            ]
        )
    if parsed.category == "All Beauty":
        suggestions.append("有没有敏感肌可用的")
    elif parsed.category in {"Electronics", "Cell Phones & Accessories"}:
        suggestions.append("优先看便携和续航")
    elif parsed.category == "Sports & Outdoors":
        suggestions.append("优先看轻便和防水")
    return list(dict.fromkeys(suggestions))[:4]


def _sampling_reasons(result: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    products = result.get("products") or []
    retrieval = result.get("retrieval") or {}
    if not products:
        reasons.append("zero_result")
    if result.get("relaxations"):
        reasons.append("constraint_relaxation")
    funnel = retrieval.get("candidate_funnel") or {}
    policy_count = int(funnel.get("attribute_policy_pass") or 0)
    if products and policy_count < max(20, len(products) * 3):
        reasons.append("low_candidate_count")
    routes = retrieval.get("routes") or {}
    fts = set((routes.get("fts") or [])[:20])
    dense = set((routes.get("dense") or [])[:20])
    if fts and dense:
        overlap = len(fts & dense) / max(1, len(fts | dense))
        if overlap < 0.2:
            reasons.append("route_disagreement")
    if float(result.get("latency_ms") or 0.0) >= 1000:
        reasons.append("high_latency")
    digest = int(
        hashlib.blake2b(result["request_id"].encode(), digest_size=4).hexdigest(), 16
    )
    if digest / 0xFFFFFFFF < settings.eval_sample_rate:
        reasons.append("random_sample")
    return list(dict.fromkeys(reasons))


class SearchEngine:
    def search(
        self,
        query: str | ParsedQuery,
        user_id: str = "demo-user",
        limit: int = 12,
        mode: str = "hybrid",
        session_id: str = "",
        context: dict[str, Any] | None = None,
        known_brands: list[str] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        stage_started = started
        stage_timings: dict[str, float] = {}
        request_id = uuid.uuid4().hex[:12]
        parsed = (
            query
            if isinstance(query, ParsedQuery)
            else parse_query(query, context=context, known_brands=known_brands)
        )
        stage_timings["query_parse"] = (time.perf_counter() - stage_started) * 1000
        trace: list[dict[str, Any]] = []

        # 入口纠错（业界 QP 流水线第 1 层）：同音/形似错别字生成纠错候选，
        # 并入 grounded_target_terms（LLM 归一化检索词的通道，已参与 FTS/dense/
        # 相关性匹配）。raw_query 原样保留。
        corrected = correction_candidates(parsed.raw_query)
        if corrected:
            parsed = replace(
                parsed,
                grounded_target_terms=[*parsed.grounded_target_terms, *corrected],
            )
            trace.append(
                {
                    "node": "query_correction",
                    "raw": parsed.raw_query,
                    "corrected": corrected,
                }
            )

        where, params = _sql_filters(parsed)
        where += " AND p.stock > 0"
        for term in parsed.excluded_title_terms:
            where += " AND LOWER(p.title) NOT LIKE ?"
            params.append("%" + term.casefold() + "%")
        for term in parsed.negative_terms[:8]:
            where += " AND LOWER(p.title || ' ' || p.description || ' ' || p.attributes_text) NOT LIKE ?"
            params.append("%" + term.casefold() + "%")
        filters = RetrievalFilters(product_where=where, product_params=params)
        retrieval_text = " ".join(
            dict.fromkeys(
                [parsed.raw_query, *parsed.grounded_target_terms, *parsed.llm_concepts]
            )
        )
        business_browse = parsed.raw_query.strip() in {
            "推荐热门商品",
            "随便看看",
            "热门商品",
        }
        if business_browse:
            hits, unit_routes, knowledge_trace = (
                [],
                {},
                {
                    "latency_ms": 0.0,
                    "unit_counts": {},
                    "errors": {},
                    "business_only": True,
                },
            )
        else:
            hits, unit_routes, knowledge_trace = retrieve(
                RetrievalQuery(retrieval_text, "product"),
                filters,
                limit=240,
                methods=("knowledge_bm25",) if mode == "fts" else None,
            )
        grouped = aggregate_skus(hits, evidence_limit=asset_settings.evidence_per_sku)
        pool_by_sku = {row["sku"]: row for row in _products_for_skus(list(grouped))}
        visual_by_id = {
            hit.unit_id: hit for hit in unit_routes.get("knowledge_visual", [])
        }
        grouped = {sku: group for sku, group in grouped.items() if sku in pool_by_sku}
        from .evidence_runtime import mode as evidence_mode, evaluate_products
        anchors = {}
        verify_calls = 0
        with db_session() as evidence_db:
            evidence_by_sku = {
                sku: [
                    item
                    for hit in group["hits"]
                    if (item := public_hit(hit, db=evidence_db))
                ]
                for sku, group in grouped.items()
            }
        for sku, group in grouped.items():
            evidence = evidence_by_sku[sku]
            pool_by_sku[sku]["knowledge_evidence"] = evidence
            pool_by_sku[sku]["asset_evidence"] = [
                item for item in evidence if item.get("asset_id")
            ]
            product = pool_by_sku[sku]
            structured_anchor = bool(
                parsed.category or parsed.brands or parsed.reference_skus
            )
            text_anchor = any(
                meaningful_overlap(
                    parsed.raw_query,
                    item["text_content"] + " " + item["matched_description"],
                )
                for item in evidence
            )
            visual_anchor = False
            if not text_anchor:
                for hit in group["hits"]:
                    visual_hit = visual_by_id.get(hit.unit_id)
                    if visual_hit is None:
                        continue
                    package = hydrate(hit.unit_id)
                    # A caption-free visual candidate requires original-image verification.
                    if evidence_mode()=="unified":
                        visual_anchor = True  # provisional relevance only; workflow verifies original image
                        break
                    if package and verify_calls < asset_settings.verification_images:
                        verify_calls += 1
                        observation = verify_image(package, parsed.raw_query)
                        if observation:
                            visual_anchor = True
                            product["visual_observation"] = observation
                            break
            visual_only = all(
                set(hit.metadata.get("source_ranks", {})) <= {"knowledge_visual"}
                for hit in group["hits"]
            )
            if (
                visual_anchor
                if visual_only
                else (structured_anchor or text_anchor or visual_anchor)
            ):
                anchors[sku] = (
                    "visual"
                    if visual_anchor
                    else "text"
                    if text_anchor
                    else "structured"
                )
        grouped = {
            sku: group
            for sku, group in grouped.items()
            if sku in anchors
            or not all(
                set(hit.metadata.get("source_ranks", {})) <= {"knowledge_visual"}
                for hit in group["hits"]
            )
        }
        pool_by_sku = {sku: row for sku, row in pool_by_sku.items() if sku in grouped}
        guarded = _needs_product_anchor(parsed)
        if guarded:
            grouped = {sku: group for sku, group in grouped.items() if sku in anchors}
            pool_by_sku = {
                sku: row for sku, row in pool_by_sku.items() if sku in grouped
            }
        if grouped and guarded:
            parsed = replace(
                parsed,
                clarification_slot=None,
                clarification_question=None,
                sources={**parsed.sources, "catalog_anchor": "knowledge:evidence"},
            )
        trace.append(
            {
                "node": "query_guard",
                "blocked": guarded and not grouped,
                "reason": "knowledge_anchor" if grouped else "missing_product_anchor",
                "anchors": anchors,
            }
        )
        # Explicit browse requests use business candidates; no semantic route is invented.
        browse = (
            parsed.intent
            in {
                "recommend",
                "browse",
            }
            or business_browse
        )
        if not grouped and browse:
            candidates = _products_for_skus(_structured_skus(parsed, limit=100))
            pool_by_sku = {row["sku"]: row for row in candidates if row["stock"] > 0}
            grouped = {sku: {"score": 1.0, "hits": []} for sku in pool_by_sku}
        checked, evidence_trace = evaluate_products(parsed.raw_query,list(pool_by_sku.values()),parsed=parsed,task_id=request_id)
        trace.extend(evidence_trace)
        pool_by_sku = {p["sku"]:p for p in checked}
        grouped = {sku:g for sku,g in grouped.items() if sku in pool_by_sku}
        candidate_skus = list(grouped)
        stage_timings["knowledge_retrieval"] = knowledge_trace["latency_ms"]
        trace.append(
            {
                "node": "structured_filter",
                "count": len(candidate_skus),
                "policy": "hard_only",
            }
        )
        trace.append(
            {
                "node": "multi_recall",
                "routes": {name: len(route) for name, route in unit_routes.items()},
                "unit_ids": {
                    name: [hit.unit_id for hit in route[:100]]
                    for name, route in unit_routes.items()
                },
                "degradations": knowledge_trace["errors"],
            }
        )
        _personal, personal_scores = (
            _personal_recall(user_id, parsed, list(pool_by_sku.values()))
            if mode == "hybrid"
            else ([], {})
        )
        routes = {
            name: [hit.unit_id for hit in route] for name, route in unit_routes.items()
        }
        dense_scores = {}
        scored_before_gate = _score_candidates(
            parsed,
            user_id,
            pool_by_sku,
            {sku: group["score"] for sku, group in grouped.items()},
            dense_scores,
            personal_scores,
            mode,
        )
        scored, relaxations, funnel = _apply_candidate_policy(
            parsed, scored_before_gate, limit
        )
        trace.append(
            {
                "node": "candidate_policy",
                "before": len(scored_before_gate),
                "after": len(scored),
                "funnel": funnel,
                "relaxations": relaxations,
            }
        )
        trace.append(
            {
                "node": "relevance_gate",
                "before": len(scored_before_gate),
                "after": len(scored),
                "dropped": len(scored_before_gate) - len(scored),
            }
        )
        rank_started = time.perf_counter()
        asset_errors = knowledge_trace["errors"]
        selected = _diversify(scored, limit, single_category=bool(parsed.category))
        all_citations: list[dict[str, Any]] = []
        judge_issues: list[str] = []
        for product in selected:
            reason, citations = _reason(parsed, product)
            product["reason"] = reason
            product["citations"] = citations
            verdict = validate_citations(product, citations)
            product["fact_check"] = verdict
            all_citations.extend(citations)
            judge_issues.extend(verdict["issues"])
        trace.append(
            {
                "node": "rank_and_diversify",
                "scored": len(scored),
                "returned": len(selected),
                "top_scores": [item["score"] for item in selected[:5]],
            }
        )
        stage_timings["rank_and_ground"] = (time.perf_counter() - rank_started) * 1000
        trace.append(
            {
                "node": "stage_timings",
                "timings_ms": {
                    key: round(value, 2) for key, value in stage_timings.items()
                },
            }
        )
        latency_ms = (time.perf_counter() - started) * 1000
        result = {
            "request_id": request_id,
            "query": parsed.raw_query,
            "parsed_query": parsed.to_dict(),
            "total_candidates": len(candidate_skus),
            "products": selected,
            "facets": _facets(scored[:100]),
            "suggestions": _suggestions(parsed, selected),
            "trace": trace,
            "latency_ms": round(latency_ms, 2),
            "judge": {"pass": not judge_issues, "issues": judge_issues},
            "citations": all_citations,
            "relaxations": relaxations,
            "retrieval": {
                "routes": {name: ranking[:100] for name, ranking in routes.items()},
                "unit_to_sku": {hit.unit_id: hit.metadata["sku"] for hit in hits},
                "knowledge": knowledge_trace,
                "route_scores": {
                    name: {
                        hit.unit_id: round(float(hit.score), 6) for hit in route[:100]
                    }
                    for name, route in unit_routes.items()
                },
                "raw_candidate_ranking": [
                    item["sku"] for item in scored_before_gate[:100]
                ],
                "candidate_ranking": [item["sku"] for item in scored[:100]],
                "dense_backend": "knowledge",
                "embedding_model": model_spec("product")[0],
                "embedding_index_version": knowledge_version("product"),
                "semantic_dense": bool(unit_routes.get("knowledge_dense"))
                and model_spec("product")[0] not in {"disabled", "hash"},
                "dense_fallback_reason": str(knowledge_trace["errors"]),
                "candidate_funnel": funnel,
                **(
                    {"asset_retrieval_status": asset_errors}
                    if asset_settings.enabled
                    else {}
                ),
            },
            "mode": mode,
        }
        self._log(result, user_id, session_id, mode)
        return result

    @staticmethod
    def _log(result: dict[str, Any], user_id: str, session_id: str, mode: str) -> None:
        trace = [*result.get("trace", []), {"node": "mode", "value": mode}]
        retrieval = result.get("retrieval") or {}
        with db_session() as connection:
            connection.execute(
                """
                INSERT INTO search_logs(
                    request_id, session_id, user_id, query, parsed_json,
                    result_skus_json, trace_json, latency_ms
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result["request_id"],
                    session_id,
                    user_id,
                    result["query"],
                    json_dumps(result["parsed_query"]),
                    json_dumps([item["sku"] for item in result.get("products", [])]),
                    json_dumps(trace),
                    result["latency_ms"],
                ),
            )
            route_scores = retrieval.get("route_scores") or {}
            route_rows = []
            for route, skus in (retrieval.get("routes") or {}).items():
                scores = route_scores.get(route) or {}
                for rank, sku in enumerate(skus, start=1):
                    route_rows.append(
                        (
                            result["request_id"],
                            route,
                            rank,
                            (retrieval.get("unit_to_sku") or {}).get(sku, sku),
                            scores.get(sku),
                            str(retrieval.get("embedding_index_version") or ""),
                        )
                    )
            if route_rows:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO search_route_logs(
                        request_id, route, rank, sku, score, model_version
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    route_rows,
                )
            products = result.get("products") or []
            if products:
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO search_result_impressions(
                        request_id, user_id, query, sku, rank, score
                    ) VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            result["request_id"],
                            user_id,
                            result["query"],
                            product["sku"],
                            rank,
                            float(product.get("score", 0.0)),
                        )
                        for rank, product in enumerate(products, start=1)
                    ],
                )
            reasons = _sampling_reasons(result)
            if reasons:
                sample_payload = {
                    "request_id": result["request_id"],
                    "query": result["query"],
                    "parsed_query": result["parsed_query"],
                    "products": [
                        {
                            "sku": product["sku"],
                            "title": product["title"],
                            "brand": product["brand"],
                            "category": product["category"],
                            "price": product["price"],
                            "score": product.get("score"),
                            "matched_terms": product.get("matched_terms", []),
                            "reason": product.get("reason", ""),
                        }
                        for product in products[:10]
                    ],
                    "retrieval": retrieval,
                    "trace": result.get("trace") or [],
                    "relaxations": result.get("relaxations") or [],
                    "latency_ms": result.get("latency_ms", 0.0),
                }
                connection.execute(
                    """
                    INSERT OR IGNORE INTO evaluation_samples(
                        sample_id, request_id, sampling_reasons_json, payload_json
                    ) VALUES(?, ?, ?, ?)
                    """,
                    (
                        f"sample-{result['request_id']}",
                        result["request_id"],
                        json_dumps(reasons),
                        json_dumps(sample_payload),
                    ),
                )


search_engine = SearchEngine()


def search(
    query: str,
    category: str = "",
    price_max: float | None = None,
    top_k: int = 3,
    profile: dict | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for the original project API."""
    parsed = parse_query(query)
    if category:
        parsed.category = category
    if price_max is not None:
        parsed.price_max = price_max
    result = search_engine.search(
        parsed, user_id=(profile or {}).get("user_id", "demo-user"), limit=top_k
    )
    return result["products"]
