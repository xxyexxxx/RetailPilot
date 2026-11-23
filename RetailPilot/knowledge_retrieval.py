"""Method-level retrieval and Unit-ID fusion over one canonical corpus."""

from __future__ import annotations

import heapq
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date
from itertools import islice
from typing import Protocol

import numpy as np

from .asset_config import asset_settings
from .asset_embeddings import visual_encoder, visual_version
from .config import settings
from .database import db_session
from .knowledge import sync_products, terms
from .knowledge_index import encode_text, model_spec, version
from .text import vector_from_blob


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    domain: str
    image: bytes | None = None


@dataclass
class RetrievalFilters:
    product_where: str = "1=1"
    product_params: list = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    platform: str = ""
    categories: set[str] = field(default_factory=set)
    effective_on: str = ""
    jurisdiction: str = "*"
    audiences: tuple[str, ...] = ("customer", "public")
    authority: str | None = None
    version: str | None = None
    include_assets: bool = True
    source_formats: tuple[str, ...] = ()


@dataclass
class RetrievalHit:
    unit_id: str
    route: str
    score: float
    rank: int
    metadata: dict = field(default_factory=dict)


retrieval_options = ContextVar("knowledge_retrieval_options", default=None)


class Retriever(Protocol):
    def retrieve(
        self, query: RetrievalQuery, filters: RetrievalFilters, limit: int
    ) -> list[RetrievalHit]: ...


JOINS = """FROM knowledge_units u JOIN knowledge_sources s USING(source_id)
LEFT JOIN products p ON p.sku=s.sku LEFT JOIN product_commerce c ON c.sku=s.sku
LEFT JOIN policy_documents d ON d.doc_id=s.doc_id LEFT JOIN assets a ON a.asset_id=s.asset_id
LEFT JOIN asset_elements e ON e.element_id=u.element_id"""


def scope(query, filters):
    where = ["s.domain=?"]
    params = [query.domain]
    if query.domain == "product":
        where.extend(
            [
                filters.product_where,
                "COALESCE(c.dataset_split,'')!='final'",
                "p.source!='Amazon Reviews 2023'",
            ]
        )
        params.extend(filters.product_params)
    else:
        where.extend(["d.status='active'", "s.version=d.version"])
        for column, values in (
            ("d.topic", filters.topics),
            ("d.platform", [filters.platform, "*"]),
            ("d.audience", filters.audiences),
            ("LOWER(d.category_scope)", ["*", *sorted(filters.categories)]),
            ("d.jurisdiction", ["*", filters.jurisdiction]),
        ):
            if values:
                where.append(f"{column} IN ({','.join('?' for _ in values)})")
                params.extend(values)
        when = filters.effective_on or date.today().isoformat()
        where.extend(
            ["d.effective_from<=?", "(d.effective_to='' OR d.effective_to>=?)"]
        )
        params.extend([when, when])
        for column, value in (
            ("d.authority", filters.authority),
            ("d.version", filters.version),
        ):
            if value:
                where.append(f"{column}=?")
                params.append(value)
    where.append(
        "(a.asset_id IS NULL OR (a.status IN ('ready','partial') AND (s.doc_id IS NULL OR a.source_version=d.version)))"
    )
    if not filters.include_assets or not asset_settings.enabled:
        where.append("s.asset_id IS NULL")
    if filters.source_formats:
        where.append(
            f"s.source_format IN ({','.join('?' for _ in filters.source_formats)})"
        )
        params.extend(filters.source_formats)
    return " AND ".join(f"({part})" for part in where), params


def _metadata(row):
    return {
        key: row[key]
        for key in (
            "sku",
            "doc_id",
            "asset_id",
            "source_id",
            "source_format",
            "modality",
            "content_hash",
            "evidence_text",
            "search_text",
            "visual_hash",
        )
    }


class BM25Retriever:
    def retrieve(self, query, filters, limit):
        tokens = terms(query.text).split()[:40]
        if not tokens:
            return []
        expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in tokens)
        where, params = scope(query, filters)
        with db_session() as db:
            rows = db.execute(
                f"""SELECT u.*,s.sku,s.doc_id,s.asset_id,s.source_format,e.preview_key AS visual_hash,
                bm25(knowledge_fts,0,10,5,3,2,1) AS score
                {JOINS} JOIN knowledge_fts f ON f.unit_id=u.unit_id
                WHERE knowledge_fts MATCH ? AND {where}
                ORDER BY CASE WHEN lower(s.sku)=lower(?) THEN 0 ELSE 1 END,score,u.unit_id LIMIT ?""",
                [expression, *params, query.text.strip(), limit],
            ).fetchall()
        exact = query.text.strip().casefold()
        rows = sorted(
            rows,
            key=lambda r: (
                0 if r["sku"] and r["sku"].casefold() == exact else 1,
                r["score"],
                r["unit_id"],
            ),
        )
        return [
            RetrievalHit(
                r["unit_id"], "knowledge_bm25", -float(r["score"]), i, _metadata(r)
            )
            for i, r in enumerate(rows, 1)
        ]


class TextDenseRetriever:
    space = "text"

    def retrieve(self, query, filters, limit):
        model_version = (
            version(query.domain) if self.space == "text" else visual_version()
        )
        where, params = scope(query, filters)
        with db_session() as db:
            # Probe the Unit/space primary key: the model index can cause a quadratic join.
            rows = db.execute(
                f"""SELECT u.*,s.sku,s.doc_id,s.asset_id,s.source_format,e.preview_key AS visual_hash,v.dim,v.vector
                {JOINS} JOIN knowledge_vectors v INDEXED BY sqlite_autoindex_knowledge_vectors_1 ON v.unit_id=u.unit_id
                WHERE {where} AND v.space=? AND v.model_version=? AND v.content_hash=u.content_hash""",
                [*params, self.space, model_version],
            )
            first = rows.fetchone()
            if first is None:
                return []
            if self.space == "text":
                vector = encode_text([query.text], query.domain, query=True)[0]
                floor = (
                    0.15
                    if model_spec(query.domain)[0] == "hash"
                    else settings.embedding_min_score
                )
            else:
                encoder = visual_encoder()
                vector = (
                    encoder.images([query.image])
                    if query.image
                    else encoder.texts([query.text])
                )[0]
                floor = asset_settings.visual_min_score
            from .knowledge_qdrant import query_mirror

            remote = query_mirror(
                db, query, filters, vector, self.space, model_version, limit, floor
            )
            if remote is not None:
                return remote

            def scored():
                batch = [first]
                while batch:
                    valid = [r for r in batch if r["dim"] == len(vector)]
                    if valid:
                        matrix = np.stack(
                            [vector_from_blob(r["vector"], r["dim"]) for r in valid]
                        )
                        for row, score in zip(valid, matrix @ vector, strict=True):
                            if score >= floor:
                                yield row, float(score)
                    batch = list(islice(rows, 512))

            best = heapq.nsmallest(
                limit, scored(), key=lambda hit: (-hit[1], hit[0]["unit_id"])
            )
        return [
            RetrievalHit(
                r["unit_id"],
                "knowledge_" + ("dense" if self.space == "text" else "visual"),
                score,
                i,
                _metadata(r),
            )
            for i, (r, score) in enumerate(best, 1)
        ]


class VisualRetriever(TextDenseRetriever):
    space = "visual"


def fuse(routes, *, limit=240):
    scores = {}
    ranks = {}
    # Equal method weights, with a bounded SKU aggregation downstream.
    for name, hits in routes.items():
        seen = set()
        for hit in hits:
            if hit.unit_id in seen:
                continue
            seen.add(hit.unit_id)
            scores[hit.unit_id] = scores.get(hit.unit_id, 0) + 1 / (60 + hit.rank)
            ranks.setdefault(hit.unit_id, {})[name] = hit.rank
    meta = {hit.unit_id: hit.metadata for hits in routes.values() for hit in hits}
    return [
        RetrievalHit(
            uid, "rrf", scores[uid], i, {**meta[uid], "source_ranks": ranks[uid]}
        )
        for i, uid in enumerate(
            sorted(scores, key=lambda uid: (-scores[uid], uid))[:limit], 1
        )
    ]


def retrieve(query, filters, *, limit=80, methods=None):
    started = time.perf_counter()
    if query.domain == "product":
        with db_session() as db:
            sync_products(db)
    options = retrieval_options.get() or {}
    methods = (
        options.get("methods")
        or methods
        or ("knowledge_bm25", "knowledge_dense", "knowledge_visual")
    )
    if options.get("formats") and query.domain == "product":
        from dataclasses import replace

        filters = replace(filters, source_formats=options["formats"])
    implementations = {
        "knowledge_bm25": BM25Retriever(),
        "knowledge_dense": TextDenseRetriever(),
        "knowledge_visual": VisualRetriever(),
    }
    routes = {}
    errors = {}
    for name in methods:
        if name == "knowledge_visual" and (
            query.domain != "product"
            or not asset_settings.enabled
            or not asset_settings.visual_enabled
        ):
            continue
        try:
            routes[name] = implementations[name].retrieve(query, filters, limit)
        except Exception as exc:  # noqa: BLE001 - independent optional retrieval method
            errors[name] = type(exc).__name__
    where, params = scope(query, filters)
    with db_session() as db:
        allowed = {
            row[0]
            for row in db.execute(f"SELECT u.unit_id {JOINS} WHERE {where}", params)
        }
    routes = {
        name: [hit for hit in hits if hit.unit_id in allowed]
        for name, hits in routes.items()
    }
    fused = fuse(routes, limit=limit)
    if query.domain == "product":
        from .policy_index import get_policy_reranker

        rerank = get_policy_reranker().rerank(
            query.text,
            [
                {
                    "chunk_id": hit.unit_id,
                    "contextualized_content": hit.metadata["search_text"],
                }
                for hit in fused
            ],
        )
        if rerank.available:
            by_id = {hit.unit_id: hit for hit in fused}
            fused = [by_id[uid] for uid in rerank.ranking if uid in by_id]
    return (
        fused,
        routes,
        {
            "errors": errors,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "unit_counts": {k: len(v) for k, v in routes.items()},
        },
    )


def aggregate_skus(hits, *, evidence_limit=3):
    grouped = {}
    for hit in hits:
        sku = hit.metadata["sku"]
        if sku:
            grouped.setdefault(sku, []).append(hit)
    out = {}
    for sku, units in grouped.items():
        selected = []
        seen = set()
        for hit in units:
            # Same evidence content or duplicate binary source receives one evidence slot.
            key = (
                ("image", hit.metadata["visual_hash"])
                if hit.metadata.get("visual_hash")
                else ("content", hit.metadata["content_hash"])
            )
            if key in seen:
                continue
            seen.add(key)
            selected.append(hit)
            if len(selected) >= evidence_limit:
                break
        # Best unit plus a capped distinct-evidence contribution.
        best = max(h.score for h in selected)
        bonus = min(best * 0.1, sum(h.score for h in selected[1:]) * 0.05)
        out[sku] = {"score": best + bonus, "hits": selected}
    return out
