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

import numpy as np

from .catalog import (
    format_price,
    get_product,
    hydrate_product,
    interaction_scores,
    preference_map,
)
from .config import settings
from .database import db_session, json_dumps
from .embeddings import DenseRecall, get_neural_index
from .query import parse_query
from .schemas import ParsedQuery
from .taxonomy import correction_candidates
from .text import fast_catalog_tokens, hash_vector, tokenize, vector_from_blob


RRF_K = 60
ROUTE_WEIGHTS = {
    "fts": 0.42,
    "dense": 0.28,
    "popular": 0.12,
    "personal": 0.18,
}
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


def _sql_filters(
    parsed: ParsedQuery, *, level: str = "hard"
) -> tuple[str, list[Any]]:
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


def _fts_recall(parsed: ParsedQuery, limit: int = 240) -> list[str]:
    # 业界 agentic search（Algolia 等）：LLM 生成的检索词是唯一召回来源。
    # 原文 query 不进 FTS（带"推荐/给老子/个"等动词废话，污染匹配）；
    # 也没有原文兜底路径——LLM 失效时意图层直接返回空任务，不会走到检索。
    # 无检索词时返回空结果，由上层默认回复兜底。
    expression = _fts_expression(
        " ".join(
            [
                *parsed.grounded_target_terms,
                *parsed.llm_concepts,
            ]
        )
    )
    if not expression:
        return []
    where, params = _sql_filters(parsed, level="hard")
    with db_session() as connection:
        rows = connection.execute(
            f"""
            SELECT p.sku, bm25(product_fts, 0.0, 5.0, 3.0, 2.0, 2.0, 1.0, 1.5) AS rank
            FROM product_fts
            JOIN products p ON p.sku = product_fts.sku
            LEFT JOIN product_commerce c ON c.sku = p.sku
            WHERE product_fts MATCH ? AND {where}
            AND COALESCE(c.dataset_split, '') != 'final'
            AND p.source != 'Amazon Reviews 2023'
            ORDER BY rank
            LIMIT ?
            """,
            [expression, *params, limit],
        ).fetchall()
    return [row["sku"] for row in rows]


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


def _catalog_lexical_evidence(parsed: ParsedQuery, product: dict[str, Any]) -> dict[str, Any]:
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
        "matched_terms": sorted(matched_terms, key=lambda value: (-len(value), value))[:12],
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


def _pool_with_relaxation(
    parsed: ParsedQuery,
) -> tuple[list[str], ParsedQuery, list[dict[str, Any]]]:
    # Compatibility boundary: relaxation now happens after sparse/dense recall,
    # so semantic candidates are not deleted before they can be scored.
    return _structured_skus(parsed, level="hard"), parsed, []


_HASH_INDEX: tuple[str, list[str], dict[str, int], np.ndarray] | None = None


def _hash_index() -> tuple[list[str], dict[str, int], np.ndarray]:
    global _HASH_INDEX
    cache_key = str(settings.database_path.resolve())
    if _HASH_INDEX is not None and _HASH_INDEX[0] == cache_key:
        return _HASH_INDEX[1], _HASH_INDEX[2], _HASH_INDEX[3]
    with db_session() as connection:
        rows = connection.execute(
            "SELECT sku, dim, vector FROM product_vectors ORDER BY sku"
        ).fetchall()
    skus = [row["sku"] for row in rows]
    matrix = np.vstack(
        [vector_from_blob(row["vector"], int(row["dim"])) for row in rows]
    )
    sku_to_index = {sku: index for index, sku in enumerate(skus)}
    _HASH_INDEX = (cache_key, skus, sku_to_index, matrix)
    return skus, sku_to_index, matrix


def _dense_recall_result(
    parsed: ParsedQuery, candidate_skus: list[str], limit: int = 240
) -> DenseRecall:
    if not candidate_skus:
        return DenseRecall([], {}, "hash-flat", "stable-hash-384", "hash-v1", False)
    # 向量召回保留原文 query（embedding 天然容忍动词/废话，且提供中文侧语义）；
    # LLM 生成的检索词补充标准词。rewritten_query 与 raw_query 同值，不重复拼。
    query_text = " ".join(
        [
            parsed.raw_query,
            *parsed.grounded_target_terms,
            *parsed.llm_concepts,
        ]
    )
    fallback_reason = ""
    try:
        neural = get_neural_index()
        if neural is not None:
            return neural.recall(query_text, candidate_skus, limit)
    except Exception as exc:
        # This is an optional third-party boundary: missing weights, a copied
        # but incompatible index, or unavailable CUDA must not take search
        # down. The reason is exposed in trace instead of being silently lost.
        fallback_reason = f"{type(exc).__name__}: {str(exc)[:240]}"
    query_vector = hash_vector(
        query_text
    )
    _, sku_to_index, matrix = _hash_index()
    pairs = [
        (sku, sku_to_index[sku])
        for sku in candidate_skus
        if sku in sku_to_index
    ]
    if not pairs:
        return DenseRecall(
            [], {}, "hash-flat", "stable-hash-384", "hash-v1", False, fallback_reason
        )
    indices = np.fromiter((index for _, index in pairs), dtype=np.int64)
    scores = matrix[indices] @ query_vector
    top_n = min(limit, len(scores))
    if top_n < len(scores):
        top = np.argpartition(-scores, top_n - 1)[:top_n]
        order = top[np.argsort(-scores[top])]
    else:
        order = np.argsort(-scores)
    ranks = [pairs[int(index)][0] for index in order]
    score_map = {
        pairs[int(index)][0]: float(max(0.0, scores[int(index)])) for index in order
    }
    return DenseRecall(
        ranks,
        score_map,
        "hash-flat",
        "stable-hash-384",
        "hash-v1",
        False,
        fallback_reason,
    )


def _dense_recall(
    parsed: ParsedQuery, candidate_skus: list[str], limit: int = 240
) -> tuple[list[str], dict[str, float]]:
    result = _dense_recall_result(parsed, candidate_skus, limit)
    return result.ranking, result.scores


def _popular_recall(pool: list[dict[str, Any]], limit: int = 240) -> list[str]:
    ranked = sorted(
        pool,
        key=lambda row: (
            float(row["quality_score"]) * 0.65 + float(row["popularity_score"]) * 0.35
        ),
        reverse=True,
    )
    return [row["sku"] for row in ranked[:limit]]


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


def _rrf(routes: dict[str, list[str]]) -> dict[str, float]:
    scores: dict[str, float] = defaultdict(float)
    for route, ranking in routes.items():
        weight = ROUTE_WEIGHTS.get(route, 0.0)
        for rank, sku in enumerate(ranking, start=1):
            scores[sku] += weight / (RRF_K + rank)
    return dict(scores)


def _term_match_score(parsed: ParsedQuery, product: dict[str, Any]) -> tuple[float, list[str]]:
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
    negative_tokens = set(tokenize(" ".join(parsed.negative_terms), include_expansion=False))
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
        for term in list(
            dict.fromkeys(tokenize(" ".join(parsed.llm_concepts)))
        )[:16]
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
    for term in parsed.negative_terms[:8]:
        target = title if term.casefold() in TITLE_SCOPED_NEGATIVE_TERMS else searchable
        if term.casefold() in target:
            return False
    return True


def _semantic_threshold(model_name: str) -> float:
    threshold = settings.embedding_min_score
    if "e5" in model_name.casefold() and threshold <= 0.5:
        return 0.75
    return threshold


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
    matched = sum(
        1
        for group in parsed.must_have_groups
        if any(term.lower() in haystack for term in _group_filter_terms(group))
    )
    return matched / len(parsed.must_have_groups)


def _apply_candidate_policy(
    parsed: ParsedQuery,
    products: list[dict[str, Any]],
    dense: DenseRecall,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    non_relaxable: list[dict[str, Any]] = []
    semantic_threshold = _semantic_threshold(dense.model)
    for product in products:
        dense_score = dense.scores.get(product["sku"], -1.0)
        semantic_match = dense.semantic and dense_score >= semantic_threshold
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
    fused = _rrf(routes)
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
        match_score, matched_terms = _term_match_score(parsed, product)
        quality = float(product["quality_score"])
        popularity = float(product["popularity_score"])
        personal = (
            max(
                -1.0,
                min(
                    1.0,
                    personal_scores.get(sku, 0.0)
                    + interaction.get(sku, 0.0) * 0.2,
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
                0.035 for existing in selected if existing["brand"] and existing["brand"] == product["brand"]
            )
            category_penalty = (
                0
                if single_category
                else sum(0.02 for existing in selected if existing["category"] == product["category"])
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


def _reason(parsed: ParsedQuery, product: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    pieces: list[str] = []
    citations: list[dict[str, Any]] = []
    if product["matched_terms"]:
        pieces.append(f"命中 {', '.join(product['matched_terms'][:3])}")
    must_have = float(product.get("score_breakdown", {}).get("must_have", 1.0))
    if parsed.must_have_groups and must_have < 1.0:
        matched = round(must_have * len(parsed.must_have_groups))
        pieces.append(f"近似候选，命中 {matched}/{len(parsed.must_have_groups)} 组关键属性")
    if product["brand"] in parsed.preferred_brands:
        pieces.append(f"品牌符合偏好 {product['brand']}")
    if product["rating"]:
        pieces.append(f"评分 {product['rating']:.1f}，共 {product['rating_count']} 条评价")
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
        pieces.append(f"符合不超过 {format_price(parsed.price_max, budget_currency)} 的预算")
    return "；".join(pieces[:4]), citations


def validate_citations(product: dict[str, Any], citations: list[dict[str, Any]]) -> dict[str, Any]:
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
    digest = int(hashlib.blake2b(result["request_id"].encode(), digest_size=4).hexdigest(), 16)
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

        seeded_fts: list[str] | None = None
        seeded_skus: list[str] | None = None
        guard_started = time.perf_counter()
        if _needs_product_anchor(parsed):
            open_fts = _fts_recall(parsed, limit=240)
            open_rows = _products_for_skus(open_fts)
            evidence_by_sku = {
                row["sku"]: _catalog_lexical_evidence(parsed, row) for row in open_rows
            }
            seeded_fts = [
                sku for sku in open_fts if evidence_by_sku.get(sku, {}).get("pass")
            ]
            seeded_skus = list(seeded_fts)
            if seeded_fts:
                parsed = replace(
                    parsed,
                    clarification_slot=None,
                    clarification_question=None,
                    confidence=max(parsed.confidence, 0.8),
                    sources={**parsed.sources, "catalog_anchor": "fts:lexical_evidence"},
                )
                trace.append(
                    {
                        "node": "query_guard",
                        "blocked": False,
                        "reason": "catalog_lexical_evidence",
                        "fts_candidates": len(open_fts),
                        "grounded_candidates": len(seeded_fts),
                    }
                )
        stage_timings["query_guard"] = (time.perf_counter() - guard_started) * 1000
        if _needs_product_anchor(parsed):
            latency_ms = (time.perf_counter() - started) * 1000
            trace.append(
                {
                    "node": "stage_timings",
                    "timings_ms": {
                        key: round(value, 2) for key, value in stage_timings.items()
                    },
                }
            )
            trace.append(
                {
                    "node": "query_guard",
                    "blocked": True,
                    "reason": "missing_product_anchor",
                }
            )
            result = {
                "request_id": request_id,
                "query": parsed.raw_query,
                "parsed_query": parsed.to_dict(),
                "total_candidates": 0,
                "products": [],
                "facets": {},
                "suggestions": [
                    "笔记本电脑",
                    "无线耳机",
                    "护肤品",
                    "运动装备",
                ],
                "trace": trace,
                "latency_ms": round(latency_ms, 2),
                "judge": {"pass": True, "issues": []},
                "relaxations": [],
                "mode": mode,
            }
            self._log(result, user_id, session_id, mode)
            return result

        filter_started = time.perf_counter()
        if seeded_skus is not None:
            candidate_skus, recall_query, relaxations = seeded_skus, parsed, []
        else:
            candidate_skus, recall_query, relaxations = _pool_with_relaxation(parsed)
        stage_timings["structured_filter"] = (
            time.perf_counter() - filter_started
        ) * 1000
        trace.append(
            {
                "node": "structured_filter",
                "count": len(candidate_skus),
                "policy": "hard_only",
                "filters": {
                    "category": parsed.category,
                    "price_min": parsed.price_min,
                    "price_max": parsed.price_max,
                    "brands": parsed.brands,
                    "preferred_brands": parsed.preferred_brands,
                    "negative_brands": parsed.negative_brands,
                    "grounded_target_terms": parsed.grounded_target_terms,
                    "llm_concepts": parsed.llm_concepts,
                    "excluded_title_terms": parsed.excluded_title_terms,
                    "must_have_groups": parsed.must_have_groups,
                },
                "deferred_to_candidate_policy": {
                    "excluded_title_terms": parsed.excluded_title_terms,
                    "must_have_groups": parsed.must_have_groups,
                    "negative_terms": parsed.negative_terms[:8],
                },
                "relaxations": relaxations,
            }
        )
        if not candidate_skus:
            latency_ms = (time.perf_counter() - started) * 1000
            trace.append(
                {
                    "node": "stage_timings",
                    "timings_ms": {
                        key: round(value, 2) for key, value in stage_timings.items()
                    },
                }
            )
            result = {
                "request_id": request_id,
                "query": parsed.raw_query,
                "parsed_query": parsed.to_dict(),
                "total_candidates": 0,
                "products": [],
                "facets": {},
                "suggestions": ["放宽预算", "换一个类目或关键词"],
                "trace": trace,
                "latency_ms": round(latency_ms, 2),
                "judge": {"pass": True, "issues": []},
                "relaxations": relaxations,
            }
            self._log(result, user_id, session_id, mode)
            return result

        fts_query = replace(
            recall_query,
            rewritten_query=" ".join(
                [recall_query.rewritten_query, *parsed.grounded_target_terms]
            ).strip(),
        )
        fts_started = time.perf_counter()
        fts = seeded_fts if seeded_fts is not None else _fts_recall(fts_query)
        stage_timings["fts_recall"] = (time.perf_counter() - fts_started) * 1000
        dense_started = time.perf_counter()
        if mode == "hybrid":
            dense_result = _dense_recall_result(parsed, candidate_skus)
            dense, dense_scores = dense_result.ranking, dense_result.scores
        else:
            dense_result = DenseRecall(
                [], {}, "disabled", "disabled", "disabled", False
            )
            dense, dense_scores = [], {}
        stage_timings["dense_recall"] = (time.perf_counter() - dense_started) * 1000

        # Sparse/dense recall returns only identifiers. Hydrate the bounded
        # union once, then compute popularity and personalization inside that
        # relevant set so neither route scans the full hard-filtered catalogue.
        recall_scores = _rrf({"fts": fts, "dense": dense})
        recalled_skus = sorted(
            recall_scores,
            key=lambda sku: (-recall_scores[sku], sku),
        )[:300]
        fetch_started = time.perf_counter()
        pool = _products_for_skus(recalled_skus)
        pool_by_sku = {row["sku"]: row for row in pool}
        stage_timings["candidate_payload_fetch"] = (
            time.perf_counter() - fetch_started
        ) * 1000

        feature_started = time.perf_counter()
        popular = _popular_recall(pool)
        if mode == "hybrid":
            personal, personal_scores = _personal_recall(user_id, parsed, pool)
        else:
            personal, personal_scores = [], {}
        stage_timings["candidate_features"] = (
            time.perf_counter() - feature_started
        ) * 1000
        routes = {
            "fts": fts,
            "dense": dense if mode == "hybrid" else [],
            "popular": popular,
            "personal": personal if mode == "hybrid" else [],
        }
        trace.append(
            {
                "node": "multi_recall",
                "routes": {name: len(ranking) for name, ranking in routes.items()},
                "top_skus": {name: ranking[:100] for name, ranking in routes.items()},
                "dense_backend": dense_result.backend,
                "embedding_model": dense_result.model,
                "embedding_index_version": dense_result.index_version,
                "semantic_dense": dense_result.semantic,
                "dense_fallback_reason": dense_result.fallback_reason,
            }
        )

        rank_started = time.perf_counter()
        scored_before_gate = _score_candidates(
            parsed,
            user_id,
            pool_by_sku,
            routes,
            dense_scores,
            personal_scores,
            mode,
        )
        scored, policy_relaxations, funnel = _apply_candidate_policy(
            parsed, scored_before_gate, dense_result, limit
        )
        relaxations.extend(policy_relaxations)
        trace.append(
            {
                "node": "candidate_policy",
                "before": len(scored_before_gate),
                "after": len(scored),
                "dropped": len(scored_before_gate) - len(scored),
                "grounded_target_terms": parsed.grounded_target_terms,
                "funnel": funnel,
                "relaxations": policy_relaxations,
            }
        )
        trace.append(
            {
                "node": "relevance_gate",
                "before": funnel["non_relaxable_pass"],
                "after": funnel["relevance_pass"],
                "dropped": funnel["non_relaxable_pass"] - funnel["relevance_pass"],
                "grounded_target_terms": parsed.grounded_target_terms,
                "semantic_dense": dense_result.semantic,
                "dense_fallback_reason": dense_result.fallback_reason,
            }
        )
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
        stage_timings["rank_and_ground"] = (
            time.perf_counter() - rank_started
        ) * 1000
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
                "route_scores": {
                    "dense": {
                        sku: round(float(dense_scores.get(sku, 0.0)), 6)
                        for sku in dense[:100]
                    }
                },
                "raw_candidate_ranking": [
                    item["sku"] for item in scored_before_gate[:100]
                ],
                "candidate_ranking": [item["sku"] for item in scored[:100]],
                "dense_backend": dense_result.backend,
                "embedding_model": dense_result.model,
                "embedding_index_version": dense_result.index_version,
                "semantic_dense": dense_result.semantic,
                "dense_fallback_reason": dense_result.fallback_reason,
                "candidate_funnel": funnel,
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
                            sku,
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
    result = search_engine.search(parsed, user_id=(profile or {}).get("user_id", "demo-user"), limit=top_k)
    return result["products"]
