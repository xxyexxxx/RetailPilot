"""Policy applicability and evidence ranking over the shared knowledge corpus."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from .catalog import get_products
from .config import settings
from .database import db_session, json_loads
from .knowledge_evidence import hydrate
from .knowledge_index import model_spec
from .knowledge_retrieval import (
    JOINS,
    RetrievalFilters,
    RetrievalQuery,
    retrieve,
    scope,
)
from .policy_index import get_policy_reranker

POLICY_RETRIEVAL_VERSION = "knowledge-policy-v1"
PolicyTopic = Literal["退款", "发票", "保修", "配送", "退换货"]
TOPIC_DOCUMENT_COMPATIBILITY = {
    "退款": frozenset({"退款"}),
    "发票": frozenset({"发票"}),
    "保修": frozenset({"保修"}),
    "配送": frozenset({"配送", "运费"}),
    "退换货": frozenset({"退换货"}),
}


def _product_scope(product_context):
    context = {
        k: v for k, v in dict(product_context or {}).items() if v not in (None, "")
    }
    products = get_products([context["sku"]]) if context.get("sku") else []
    categories = set()
    if products:
        context.update(
            {
                key: products[0][key]
                for key in ("sku", "category", "category_path")
                if products[0].get(key)
            }
        )
    for item in [*products, context]:
        categories.update(
            str(item[key]).casefold()
            for key in ("category", "category_label", "category_path")
            if item.get(key)
        )
    return context, products, categories


def retrieve_policies(
    query: str,
    *,
    topic: PolicyTopic,
    platform: str | None = None,
    product_context: dict[str, Any] | None = None,
    as_of: date | None = None,
    limit: int = 6,
    jurisdiction: str = "*",
    authority: str | None = None,
    version: str | None = None,
) -> dict:
    platform = platform or settings.commerce_platform
    product_context, _products, categories = _product_scope(product_context)
    when = (as_of or date.today()).isoformat()
    topics = sorted(TOPIC_DOCUMENT_COMPATIBILITY.get(topic, frozenset({topic})))
    rq = RetrievalQuery(query, "policy")
    filters = RetrievalFilters(
        topics=topics,
        platform=platform,
        categories=categories,
        effective_on=when,
        jurisdiction=jurisdiction,
        authority=authority,
        version=version,
    )
    hits, routes, trace = retrieve(
        rq,
        filters,
        limit=settings.policy_fusion_limit,
        methods=("knowledge_bm25", "knowledge_dense"),
    )
    candidates = []
    where, params = scope(rq, filters)
    with db_session() as db:
        count = db.execute(f"SELECT COUNT(*) {JOINS} WHERE {where}", params).fetchone()[
            0
        ]
        docs = {
            row["doc_id"]: dict(row)
            for row in db.execute("SELECT * FROM policy_documents")
        }
        for hit in hits:
            package = hydrate(hit.unit_id, db=db)
            if not package:
                continue
            document = docs[package.doc_id]
            loc = package.source_location
            candidates.append(
                {
                    **{
                        key: document.get(key)
                        for key in (
                            "doc_id",
                            "policy_id",
                            "topic",
                            "title",
                            "platform",
                            "jurisdiction",
                            "language",
                            "audience",
                            "effective_from",
                            "effective_to",
                            "version",
                            "authority",
                            "priority",
                        )
                    },
                    "unit_id": hit.unit_id,
                    "chunk_id": hit.unit_id,
                    "source_id": package.provenance["source_id"],
                    "source": document["source_name"],
                    "source_uri": loc["source_uri"],
                    "category": document["category_scope"],
                    "section_path": loc["section_path"],
                    "page_no": loc["page_no"],
                    "char_start": loc.get("char_start", 0),
                    "char_end": loc.get("char_end", 0),
                    "parent_chunk_id": None,
                    "chunk_type": "leaf",
                    "content": package.text_content,
                    "contextualized_content": hit.metadata["search_text"],
                    "metadata": json_loads(document["metadata_json"], {}),
                    "chunk_metadata": {
                        "unit_id": hit.unit_id,
                        **({"asset_id": package.asset_id} if package.asset_id else {}),
                    },
                    "rrf_score": hit.score,
                    "source_ranks": list(hit.metadata["source_ranks"].values()),
                    "asset_location": loc if package.asset_id else None,
                    "bm25_score": next(
                        (
                            h.score
                            for h in routes.get("knowledge_bm25", [])
                            if h.unit_id == hit.unit_id
                        ),
                        0,
                    ),
                    "dense_score": next(
                        (
                            h.score
                            for h in routes.get("knowledge_dense", [])
                            if h.unit_id == hit.unit_id
                        ),
                        0,
                    ),
                }
            )
    rerank = get_policy_reranker().rerank(query, candidates)
    mapping = {item["unit_id"]: item for item in candidates}
    ordered = rerank.ranking if rerank.available else list(mapping)
    selected = []
    for uid in ordered:
        if uid not in mapping:
            continue
        item = mapping[uid]
        item.update(
            evidence=item["content"],
            context=item["content"],
            rerank_score=rerank.scores.get(uid),
            score_breakdown={
                "rrf_score": item["rrf_score"],
                "bm25_score": item["bm25_score"],
                "dense_score": item["dense_score"],
            },
        )
        item["citation"] = {
            key: item.get(key)
            for key in (
                "unit_id",
                "chunk_id",
                "source_id",
                "doc_id",
                "source_uri",
                "section_path",
                "page_no",
                "char_start",
                "char_end",
                "version",
                "effective_from",
                "effective_to",
            )
        }
        item["citation"]["quoted_text"] = item["content"]
        selected.append(item)
        if len(selected) >= limit:
            break
    dense_error = trace["errors"].get("knowledge_dense", "")
    return {
        "query": query,
        "topic": topic,
        "platform": platform,
        "product_context": product_context,
        "product_skus": [product_context["sku"]] if product_context.get("sku") else [],
        "product_categories": sorted(categories),
        "documents": selected,
        "grounded": bool(selected),
        "needs_product_context": topic in {"退换货", "保修"}
        and not (product_context.get("sku") or categories),
        "retrieval_version": POLICY_RETRIEVAL_VERSION,
        "trace": {
            "hard_filter": {
                "topics": topics,
                "platforms": [platform, "*"],
                "categories": sorted(categories) or ["*"],
                "audience": ["customer", "public"],
                "effective_on": when,
                "candidate_chunks": count,
            },
            "bm25": {"returned": len(routes.get("knowledge_bm25", []))},
            "dense": {
                "backend": "knowledge",
                "model": model_spec("policy")[0],
                "available": bool(routes.get("knowledge_dense")),
                "error": dense_error
                or ("disabled" if not routes.get("knowledge_dense") else ""),
                "returned": len(routes.get("knowledge_dense", [])),
            },
            "fusion": {"method": "unit_rrf", "returned": len(hits)},
            "rerank": {
                "model": rerank.model,
                "available": rerank.available,
                "error": rerank.error,
                "returned": len(ordered),
            },
            "parent_expanded": 0,
            "returned": len(selected),
            "knowledge": trace,
        },
    }
