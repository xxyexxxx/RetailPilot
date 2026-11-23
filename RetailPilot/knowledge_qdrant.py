"""Recoverable, optional Qdrant mirror of canonical Unit vectors."""

from __future__ import annotations

import os
import uuid
from functools import lru_cache
from hashlib import sha256

from .database import db_session

NAMESPACE = uuid.UUID("268fbda1-d544-4b81-a43f-92a760769551")


def _url():
    return os.getenv("RETAILPILOT_KNOWLEDGE_QDRANT_URL", "")


def _collection(domain, space, model):
    suffix = sha256(model.encode()).hexdigest()[:12]
    return f"retailpilot_knowledge_{domain}_{space}_{suffix}"


@lru_cache(maxsize=1)
def _client(url):
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=url,
        api_key=os.getenv("RETAILPILOT_KNOWLEDGE_QDRANT_KEY") or None,
        timeout=3,
        check_compatibility=False,
    )


def _point(uid):
    return str(uuid.uuid5(NAMESPACE, uid))


def _target():
    from .asset_embeddings import visual_version
    from .knowledge_index import version

    return sha256(
        f"{_url()}|{version('product')}|{version('policy')}|{visual_version()}".encode()
    ).hexdigest()


def status(db):
    row = db.execute("SELECT * FROM knowledge_index_state").fetchone()
    return {
        "configured": bool(_url()),
        "generation": row["generation"],
        "synchronized": row["target"] == _target()
        and row["generation"] == row["synced_generation"],
    }


def query_mirror(db, query, filters, vector, space, model, limit, floor):
    if not _url() or not status(db)["synchronized"]:
        return None
    from qdrant_client import models

    from .knowledge_retrieval import JOINS, RetrievalHit, _metadata, scope

    where, params = scope(query, filters)
    rows = db.execute(
        f"""SELECT u.*,s.sku,s.doc_id,s.asset_id,s.source_format,e.preview_key AS visual_hash
        {JOINS} JOIN knowledge_vectors v INDEXED BY sqlite_autoindex_knowledge_vectors_1 ON v.unit_id=u.unit_id
        WHERE {where} AND v.space=? AND v.model_version=? AND v.dim=?
        AND v.content_hash=u.content_hash LIMIT 5001""",
        [*params, space, model, len(vector)],
    ).fetchall()
    if not rows or len(rows) > 5000:
        return [] if not rows else None
    mapping = {_point(r["unit_id"]): r for r in rows}
    try:
        result = _client(_url()).query_points(
            collection_name=_collection(query.domain, space, model),
            query=vector.tolist(),
            query_filter=models.Filter(
                must=[models.HasIdCondition(has_id=list(mapping))]
            ),
            with_payload=False,
            limit=limit,
            score_threshold=floor,
        )
        return [
            RetrievalHit(
                mapping[str(p.id)]["unit_id"],
                "knowledge_dense" if space == "text" else "knowledge_visual",
                float(p.score),
                i,
                _metadata(mapping[str(p.id)]),
            )
            for i, p in enumerate(result.points, 1)
            if str(p.id) in mapping
        ]
    except Exception:  # noqa: BLE001 - stale/unavailable mirror uses the local vector index
        return None


def sync():
    if not _url():
        raise ValueError("knowledge_qdrant_url_required")
    from qdrant_client import models

    from .asset_embeddings import visual_version
    from .knowledge_index import version
    from .text import vector_from_blob

    with db_session() as db:
        db.execute("BEGIN IMMEDIATE")
        generation = db.execute(
            "SELECT generation FROM knowledge_index_state"
        ).fetchone()[0]
        token = uuid.uuid4().hex
        claimed = db.execute(
            """UPDATE knowledge_index_state SET sync_token=?,sync_expires=unixepoch()+1800,
                              synced_generation=-1 WHERE sync_expires<=unixepoch()""",
            (token,),
        ).rowcount
    if not claimed:
        raise RuntimeError("knowledge_sync_busy")
    total = 0
    try:
        client = _client(_url())
        for domain in ("product", "policy"):
            for space in ("text", "visual"):
                model = version(domain) if space == "text" else visual_version()
                name = _collection(domain, space, model)
                valid = set()
                with db_session() as db:
                    cursor = db.execute(
                        """SELECT v.* FROM knowledge_vectors v JOIN knowledge_units u USING(unit_id)
                        JOIN knowledge_sources s USING(source_id)
                        WHERE s.domain=? AND v.space=? AND v.model_version=? AND v.content_hash=u.content_hash""",
                        (domain, space, model),
                    )
                    while batch := cursor.fetchmany(64):
                        if not client.collection_exists(name):
                            client.create_collection(
                                name,
                                vectors_config=models.VectorParams(
                                    size=batch[0]["dim"],
                                    distance=models.Distance.COSINE,
                                ),
                            )
                        points = []
                        for row in batch:
                            pid = _point(row["unit_id"])
                            valid.add(pid)
                            points.append(
                                models.PointStruct(
                                    id=pid,
                                    vector=vector_from_blob(
                                        row["vector"], row["dim"]
                                    ).tolist(),
                                    payload={
                                        "unit_id": row["unit_id"],
                                        "model_version": model,
                                    },
                                )
                            )
                        client.upsert(name, points=points, wait=True)
                if client.collection_exists(name):
                    offset = None
                    while True:
                        points, offset = client.scroll(
                            name, offset=offset, limit=256, with_payload=False
                        )
                        stale = [p.id for p in points if str(p.id) not in valid]
                        if stale:
                            client.delete(
                                name, models.PointIdsList(points=stale), wait=True
                            )
                        if offset is None:
                            break
                total += len(valid)
        with db_session() as db:
            applied = db.execute(
                """UPDATE knowledge_index_state SET synced_generation=?,target=?
                WHERE generation=? AND sync_token=?""",
                (generation, _target(), generation, token),
            ).rowcount
        return {"points": total, "synchronized": bool(applied)}
    finally:
        with db_session() as db:
            db.execute(
                "UPDATE knowledge_index_state SET sync_token='',sync_expires=0 WHERE sync_token=?",
                (token,),
            )
