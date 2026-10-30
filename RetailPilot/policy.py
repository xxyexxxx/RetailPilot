from __future__ import annotations

from datetime import date
from typing import Any, Literal

from .catalog import get_products
from .config import settings
from .database import db_session, json_loads
from .policy_index import (
    PolicyDenseRecall,
    get_policy_dense_index,
    get_policy_reranker,
)
from .text import tokenize


POLICY_RETRIEVAL_VERSION = "policy-chunk-hybrid-v2"
PolicyTopic = Literal["退款", "发票", "保修", "配送", "退换货"]

# Topic meaning is selected by the LLM planner. The retrieval layer only
# validates the enum and applies this explicit metadata compatibility fence.
TOPIC_DOCUMENT_COMPATIBILITY: dict[str, frozenset[str]] = {
    "退款": frozenset({"退款"}),
    "发票": frozenset({"发票"}),
    "保修": frozenset({"保修"}),
    "配送": frozenset({"配送", "运费"}),
    "退换货": frozenset({"退换货"}),
}


def _product_scope(
    product_context: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    context = {
        key: value
        for key, value in dict(product_context or {}).items()
        if value not in (None, "")
    }
    sku = str(context.get("sku") or "").strip()
    products = get_products([sku]) if sku else []
    categories: set[str] = set()
    if products:
        product = products[0]
        context.update(
            {
                key: value
                for key, value in {
                    "sku": product.get("sku"),
                    "category": product.get("category"),
                    "category_path": product.get("category_path"),
                }.items()
                if value
            }
        )
    for product in products:
        for field in ("category", "category_label", "category_path"):
            value = str(product.get(field) or "").strip()
            if value:
                categories.add(value.casefold())
    for field in ("category", "category_path"):
        value = str(context.get(field) or "").strip()
        if value:
            categories.add(value.casefold())
    return context, products, categories


def _hard_filter_sql(
    *,
    topics: list[str],
    platform: str,
    categories: set[str],
    effective_on: str,
) -> tuple[str, list[Any]]:
    topic_marks = ",".join("?" for _ in topics)
    clauses = [
        "d.status = 'active'",
        "d.audience IN ('customer', 'public')",
        "d.platform IN (?, '*')",
        f"d.topic IN ({topic_marks})",
        "d.effective_from <= ?",
        "(d.effective_to = '' OR d.effective_to >= ?)",
        "c.chunk_type != 'parent'",
    ]
    params: list[Any] = [platform, *topics, effective_on, effective_on]
    if categories:
        category_marks = ",".join("?" for _ in categories)
        clauses.append(f"LOWER(d.category_scope) IN ('*', {category_marks})")
        params.extend(sorted(categories))
    else:
        clauses.append("d.category_scope = '*'")
    return " AND ".join(clauses), params


def _fts_expression(query: str) -> tuple[str, list[str]]:
    terms = list(dict.fromkeys(tokenize(query, include_expansion=False)))[:40]
    escaped = [f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms]
    return " OR ".join(escaped), terms


def _rrf(
    rankings: list[list[str]], *, k: int, limit: int
) -> tuple[list[str], dict[str, float], dict[str, list[int]]]:
    scores: dict[str, float] = {}
    source_ranks: dict[str, list[int]] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            source_ranks.setdefault(chunk_id, []).append(rank)
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:limit]
    return ordered, scores, source_ranks


def _row_payload(row: Any) -> dict[str, Any]:
    item = dict(row)
    item["content"] = item.pop("chunk_content")
    item["source"] = item.pop("source_name")
    item["metadata"] = json_loads(item.pop("document_metadata_json"), {})
    item["chunk_metadata"] = json_loads(item.pop("chunk_metadata_json"), {})
    return item


def retrieve_policies(
    query: str,
    *,
    topic: PolicyTopic,
    platform: str | None = None,
    product_context: dict[str, Any] | None = None,
    as_of: date | None = None,
    limit: int = 6,
) -> dict[str, Any]:
    """Chunk-level BM25 + Qdrant recall, RRF fusion and optional reranking."""

    platform = platform or settings.commerce_platform
    product_context, products, categories = _product_scope(product_context)
    effective_date = as_of or date.today()
    effective_on = effective_date.isoformat()
    compatible_topics = sorted(
        TOPIC_DOCUMENT_COMPATIBILITY.get(topic, frozenset({topic}))
    )
    where_sql, filter_params = _hard_filter_sql(
        topics=compatible_topics,
        platform=platform,
        categories=categories,
        effective_on=effective_on,
    )
    select_sql = f"""
        SELECT
            c.chunk_id, c.doc_id, c.parent_chunk_id, c.chunk_type, c.ordinal,
            c.section_path, c.page_no, c.char_start, c.char_end,
            c.content AS chunk_content, c.contextualized_content,
            c.token_count, c.content_hash,
            c.metadata_json AS chunk_metadata_json,
            d.policy_id, d.topic, d.title, d.source_name, d.source_uri,
            d.platform, d.jurisdiction, d.language, d.category_scope AS category,
            d.audience, d.effective_from, d.effective_to, d.version,
            d.authority, d.priority,
            d.metadata_json AS document_metadata_json, d.updated_at
        FROM policy_chunks c
        JOIN policy_documents d ON d.doc_id = c.doc_id
        WHERE {where_sql}
    """

    expression, query_terms = _fts_expression(query)
    with db_session() as connection:
        candidate_rows = connection.execute(select_sql, filter_params).fetchall()
        candidate_by_id = {
            str(row["chunk_id"]): _row_payload(row) for row in candidate_rows
        }
        bm25_rows = []
        if expression:
            bm25_rows = connection.execute(
                f"""
                SELECT f.chunk_id, bm25(policy_chunk_fts) AS bm25_score
                FROM policy_chunk_fts f
                JOIN policy_chunks c ON c.chunk_id = f.chunk_id
                JOIN policy_documents d ON d.doc_id = c.doc_id
                WHERE policy_chunk_fts MATCH ? AND {where_sql}
                ORDER BY bm25_score ASC, f.chunk_id ASC
                LIMIT ?
                """,
                [expression, *filter_params, settings.policy_bm25_limit],
            ).fetchall()

    bm25_ranking = [str(row["chunk_id"]) for row in bm25_rows]
    bm25_scores = {
        str(row["chunk_id"]): -float(row["bm25_score"]) for row in bm25_rows
    }
    bm25_rank_by_id = {
        chunk_id: rank for rank, chunk_id in enumerate(bm25_ranking, start=1)
    }

    dense_index = get_policy_dense_index()
    if dense_index is None:
        dense = PolicyDenseRecall(
            ranking=[],
            scores={},
            backend=settings.policy_vector_backend,
            model=settings.policy_embedding_model,
            index_version="",
            available=False,
            error="disabled",
        )
    else:
        dense = dense_index.search(
            query,
            topics=compatible_topics,
            platform=platform,
            categories=categories,
            as_of=effective_date,
            limit=settings.policy_dense_limit,
        )
    # SQL source-of-truth verification prevents stale vector payloads from
    # bypassing current platform/topic/effective-date/category policy fences.
    dense_ranking = [
        chunk_id for chunk_id in dense.ranking if chunk_id in candidate_by_id
    ]
    dense_rank_by_id = {
        chunk_id: rank for rank, chunk_id in enumerate(dense_ranking, start=1)
    }

    fused, rrf_scores, source_ranks = _rrf(
        [bm25_ranking, dense_ranking],
        k=settings.policy_rrf_k,
        limit=settings.policy_fusion_limit,
    )
    fused_candidates = [candidate_by_id[chunk_id] for chunk_id in fused]
    rerank = get_policy_reranker().rerank(query, fused_candidates)
    ordered_ids = rerank.ranking if rerank.available else fused

    selected: list[dict[str, Any]] = []
    seen_sections: set[tuple[str, str]] = set()
    for chunk_id in ordered_ids:
        item = dict(candidate_by_id[chunk_id])
        section_key = (str(item["doc_id"]), str(item["section_path"]))
        if section_key in seen_sections:
            continue
        seen_sections.add(section_key)
        item["bm25_score"] = round(bm25_scores.get(chunk_id, 0.0), 6)
        item["dense_score"] = round(dense.scores.get(chunk_id, 0.0), 6)
        item["rrf_score"] = round(rrf_scores.get(chunk_id, 0.0), 8)
        item["rerank_score"] = (
            round(rerank.scores[chunk_id], 6)
            if chunk_id in rerank.scores
            else None
        )
        item["source_ranks"] = source_ranks.get(chunk_id, [])
        item["score_breakdown"] = {
            "bm25_rank": bm25_rank_by_id.get(chunk_id),
            "bm25_score": item["bm25_score"],
            "dense_rank": dense_rank_by_id.get(chunk_id),
            "dense_score": item["dense_score"],
            "rrf_score": item["rrf_score"],
            "rerank_score": item["rerank_score"],
        }
        item["evidence"] = item["content"]
        item["context"] = item["content"]
        item["citation"] = {
            "chunk_id": item["chunk_id"],
            "doc_id": item["doc_id"],
            "title": item["title"],
            "source_uri": item["source_uri"],
            "section_path": item["section_path"],
            "page_no": item["page_no"],
            "char_start": item["char_start"],
            "char_end": item["char_end"],
            "version": item["version"],
            "effective_from": item["effective_from"],
            "effective_to": item["effective_to"],
            "quoted_text": item["content"],
        }
        selected.append(item)
        if len(selected) >= limit:
            break

    parent_ids = sorted(
        {
            str(item["parent_chunk_id"])
            for item in selected
            if item.get("parent_chunk_id")
        }
    )
    if parent_ids:
        placeholders = ",".join("?" for _ in parent_ids)
        with db_session() as connection:
            parent_rows = connection.execute(
                f"""
                SELECT chunk_id, content FROM policy_chunks
                WHERE chunk_id IN ({placeholders}) AND chunk_type = 'parent'
                """,
                parent_ids,
            ).fetchall()
        parent_content = {
            str(row["chunk_id"]): str(row["content"]) for row in parent_rows
        }
        for item in selected:
            if item.get("parent_chunk_id") in parent_content:
                item["context"] = parent_content[str(item["parent_chunk_id"])]

    return {
        "query": query,
        "topic": topic,
        "platform": platform,
        "product_context": product_context,
        "product_skus": (
            [str(product_context["sku"])] if product_context.get("sku") else []
        ),
        "product_categories": sorted(categories),
        "documents": selected,
        "needs_product_context": bool(
            topic in {"退换货", "保修"}
            and not (product_context.get("sku") or categories)
        ),
        "grounded": bool(selected),
        "retrieval_version": POLICY_RETRIEVAL_VERSION,
        "trace": {
            "hard_filter": {
                "topics": compatible_topics,
                "platforms": [platform, "*"],
                "categories": sorted(categories) or ["*"],
                "audience": ["customer", "public"],
                "effective_on": effective_on,
                "candidate_chunks": len(candidate_by_id),
            },
            "bm25": {
                "query_terms": query_terms,
                "returned": len(bm25_ranking),
            },
            "dense": {
                "backend": dense.backend,
                "model": dense.model,
                "index_version": dense.index_version,
                "available": dense.available,
                "error": dense.error,
                "returned": len(dense_ranking),
            },
            "fusion": {
                "method": "rrf",
                "rrf_k": settings.policy_rrf_k,
                "returned": len(fused),
            },
            "rerank": {
                "model": rerank.model,
                "available": rerank.available,
                "error": rerank.error,
                "returned": len(ordered_ids),
            },
            "parent_expanded": len(parent_ids),
            "returned": len(selected),
        },
    }
