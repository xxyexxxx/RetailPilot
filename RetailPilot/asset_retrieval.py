"""Independent BM25, text dense and shared-space visual recall for product assets."""

from __future__ import annotations

import heapq
from dataclasses import asdict, dataclass
from itertools import islice

import numpy as np

from .asset_config import asset_settings as config
from .asset_embeddings import text_encode, text_version, visual_encoder, visual_version
from .asset_ingestion import terms
from .database import db_session
from .text import vector_from_blob


@dataclass(frozen=True)
class AssetHit:
    chunk_id: str
    asset_id: str
    sku: str
    rank: int
    score: float
    content_hash: str


_PRODUCT_SCOPE = """
    a.status IN ('ready','partial') AND a.sku IS NOT NULL
    AND COALESCE(c.dataset_split, '') != 'final'
    AND p.source != 'Amazon Reviews 2023'
"""

_JOIN = """
    JOIN assets a ON a.asset_id = ch.asset_id
    JOIN products p ON p.sku = a.sku
    LEFT JOIN product_commerce c ON c.sku = p.sku
"""


def _top_dense(db, query, image, space, where, params, limit, errors):
    model_version = text_version() if space == "text" else visual_version()
    rows = db.execute(
        f"""
        SELECT ch.chunk_id,a.asset_id,a.sku,a.content_hash,v.dim,v.vector
        FROM asset_vectors v JOIN asset_chunks ch ON ch.chunk_id=v.chunk_id
        {_JOIN}
        WHERE v.space=? AND v.model_version=? AND {where}
        """,
        [space, model_version, *params],
    )
    first = rows.fetchone()
    if first is None:
        return []
    if space == "text":
        vector = text_encode([query], query=True)[0]
        min_score = 0.15 if config.text_model == "hash" else 0.42
    else:
        encoder = visual_encoder()
        vector = (encoder.images([image]) if image else encoder.texts([query]))[0]
        min_score = config.visual_min_score
    # Bound memory while scanning the filtered flat index.
    if space == "visual" and config.qdrant_url:
        from .asset_qdrant import query_mirror

        try:
            remote = query_mirror(db, vector, where, params, limit, min_score)
            if remote is not None:
                return remote
        except Exception as exc:  # noqa: BLE001 - remote mirror falls back to SQLite
            errors["qdrant"] = type(exc).__name__

    def scored():
        batch = [first]
        while batch:
            valid = [row for row in batch if row["dim"] == len(vector)]
            if valid:
                matrix = np.stack(
                    [vector_from_blob(row["vector"], row["dim"]) for row in valid]
                )
                for row, score in zip(valid, matrix @ vector, strict=True):
                    if score >= min_score:
                        yield float(score), row
            batch = list(islice(rows, 512))

    best = heapq.nsmallest(
        limit, scored(), key=lambda hit: (-hit[0], hit[1]["chunk_id"])
    )
    return [(row, score) for score, row in best]


def retrieve_product_assets(
    query: str,
    *,
    where: str = "1=1",
    params: list | None = None,
    dense: bool = True,
    visual: bool = True,
    image: bytes | None = None,
    in_stock: bool = True,
) -> tuple[dict[str, list[AssetHit]], dict[str, str]]:
    if not config.enabled or (not query.strip() and image is None):
        return {}, {}
    params = params or []
    where = f"({_PRODUCT_SCOPE}) AND ({where})"
    if in_stock:
        where += " AND p.stock > 0"
    # Retrieve enough records to preserve SKU diversity before the final cap.
    limit = config.top_k * 6
    rankings, errors = {}, {}
    with db_session() as db:
        tokens = terms(query).split()[:40]
        if tokens:
            expression = " OR ".join(
                '"' + word.replace('"', '""') + '"' for word in tokens
            )
            rows = db.execute(
                f"""
              SELECT ch.chunk_id,a.asset_id,a.sku,a.content_hash,bm25(asset_fts) AS score
              FROM asset_fts JOIN asset_chunks ch ON ch.chunk_id=asset_fts.chunk_id
              {_JOIN}
              WHERE asset_fts MATCH ? AND {where}
              ORDER BY score, ch.chunk_id LIMIT ?
            """,
                [expression, *params, limit],
            ).fetchall()
            rankings["asset_bm25"] = [(row, -row["score"]) for row in rows]
        for space, enabled in (
            ("text", dense and bool(tokens)),
            ("visual", visual and config.visual_enabled),
        ):
            if not enabled:
                continue
            try:
                rankings["asset_" + space] = _top_dense(
                    db, query, image, space, where, params, limit, errors
                )
            except Exception as exc:  # noqa: BLE001 - independent optional encoder
                errors[space] = type(exc).__name__
    out = {}
    for route, ranking in rankings.items():
        counts: dict[str, int] = {}
        hits = []
        for rank, (row, score) in enumerate(ranking, 1):
            sku = row["sku"]
            if counts.get(sku, 0) >= config.evidence_per_sku:
                continue
            counts[sku] = counts.get(sku, 0) + 1
            hits.append(
                AssetHit(
                    row["chunk_id"],
                    row["asset_id"],
                    sku,
                    rank,
                    score,
                    row["content_hash"],
                )
            )
            if len(hits) >= config.top_k:
                break
        out[route] = hits
    return out, errors


def aggregate_routes(
    routes: dict[str, list[AssetHit]],
) -> tuple[dict[str, list[str]], dict[str, list[dict]]]:
    """A SKU contributes at most one rank to each existing RRF route."""
    sku_routes: dict[str, list[str]] = {}
    by_sku: dict[str, dict[str, dict]] = {}
    for route, hits in routes.items():
        sku_routes[route] = list(dict.fromkeys(hit.sku for hit in hits))
        for hit in hits:
            offers = by_sku.setdefault(hit.sku, {})
            item = offers.setdefault(
                hit.chunk_id, {"hit": hit, "routes": {}, "fused": 0.0}
            )
            item["routes"][route] = {"rank": hit.rank, "score": round(hit.score, 6)}
            item["fused"] += 1 / (60 + hit.rank)
    evidence = {}
    for sku, offers in by_sku.items():
        used = set()
        evidence[sku] = []
        for offer in sorted(offers.values(), key=lambda item: -item["fused"]):
            hit = offer["hit"]
            if hit.content_hash in used:
                continue
            used.add(hit.content_hash)
            public = asdict(hit)
            public.pop("content_hash")
            evidence[sku].append({**public, "routes": offer["routes"]})
            if len(used) >= config.evidence_per_sku:
                break
    return sku_routes, evidence
