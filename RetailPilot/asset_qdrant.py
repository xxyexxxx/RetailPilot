"""Optional remote visual index mirrored from the transactional SQLite index."""

from __future__ import annotations

import uuid
from functools import lru_cache
from hashlib import sha256

from .asset_config import asset_settings as config
from .asset_embeddings import visual_version
from .database import db_session
from .text import vector_from_blob

_NAMESPACE = uuid.UUID("268fbda1-d544-4b81-a43f-92a760769551")


@lru_cache(maxsize=2)
def _client(url: str, key: str):
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=url, api_key=key or None, timeout=3, check_compatibility=False
    )


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _state(db) -> tuple[int, str, int]:
    row = db.execute(
        "SELECT generation, target, synced_generation FROM asset_vector_state"
    ).fetchone()
    return row["generation"], row["target"], row["synced_generation"]


def generation_status(db) -> dict:
    generation, target, synced = _state(db)
    return {
        "configured": bool(config.qdrant_url),
        "generation": generation,
        "synchronized": target == _target() and generation == synced,
    }


def _target() -> str:
    return sha256(
        f"{config.qdrant_url}|{config.qdrant_collection}|{visual_version()}".encode()
    ).hexdigest()


def query_mirror(db, vector, where, params, limit: int, min_score: float):
    if not config.qdrant_url or not generation_status(db)["synchronized"]:
        return None
    from qdrant_client import models

    rows = db.execute(
        f"""
        SELECT ch.chunk_id,a.asset_id,a.sku,a.content_hash FROM asset_vectors v
        JOIN asset_chunks ch ON ch.chunk_id=v.chunk_id JOIN assets a ON a.asset_id=ch.asset_id
        JOIN products p ON p.sku=a.sku LEFT JOIN product_commerce c ON c.sku=p.sku
        WHERE v.space='visual' AND v.model_version=? AND v.dim=? AND {where}
        LIMIT 5001
    """,
        [visual_version(), len(vector), *params],
    ).fetchall()
    if not rows or len(rows) > 5000:
        return [] if not rows else None
    ids = {_point_id(row["chunk_id"]): row for row in rows}
    query = _client(config.qdrant_url, config.qdrant_key).query_points(
        collection_name=config.qdrant_collection,
        query=vector.tolist(),
        query_filter=models.Filter(must=[models.HasIdCondition(has_id=list(ids))]),
        with_payload=False,
        limit=limit,
        score_threshold=min_score,
    )
    return [
        (ids[str(point.id)], float(point.score))
        for point in query.points
        if str(point.id) in ids
    ]


def sync_visual() -> dict:
    if not config.qdrant_url:
        raise ValueError("qdrant_url_required")
    from qdrant_client import models

    with db_session() as db:
        db.execute("BEGIN IMMEDIATE")
        generation, _, _ = _state(db)
        token = uuid.uuid4().hex
        claimed = db.execute(
            """UPDATE asset_vector_state SET synced_generation=-1,sync_token=?,
               sync_expires=unixepoch()+1800 WHERE sync_expires<=unixepoch()""",
            (token,),
        ).rowcount
    if not claimed:
        raise RuntimeError("visual_mirror_sync_in_progress")
    try:
        points = _write_mirror(models)
        with db_session() as db:
            updated = db.execute(
                """UPDATE asset_vector_state SET synced_generation=?,target=?
                   WHERE generation=? AND sync_token=?""",
                (generation, _target(), generation, token),
            ).rowcount
        return {"points": points, "synchronized": bool(updated)}
    finally:
        with db_session() as db:
            db.execute(
                "UPDATE asset_vector_state SET sync_token='',sync_expires=0 WHERE sync_token=?",
                (token,),
            )


def _write_mirror(models) -> int:
    client = _client(config.qdrant_url, config.qdrant_key)
    name = config.qdrant_collection
    valid = set()
    with db_session() as db:
        rows = db.execute(
            """
            SELECT v.*, a.sku, a.asset_id FROM asset_vectors v
            JOIN asset_chunks ch ON ch.chunk_id=v.chunk_id JOIN assets a ON a.asset_id=ch.asset_id
            WHERE space='visual' AND model_version=? AND a.status IN ('ready','partial')
            ORDER BY v.chunk_id
            """,
            (visual_version(),),
        )
        while batch := rows.fetchmany(64):
            if not client.collection_exists(name):
                client.create_collection(
                    name,
                    vectors_config=models.VectorParams(
                        size=batch[0]["dim"], distance=models.Distance.COSINE
                    ),
                )
            points = []
            for row in batch:
                point = _point_id(row["chunk_id"])
                valid.add(point)
                points.append(
                    models.PointStruct(
                        id=point,
                        vector=vector_from_blob(row["vector"], row["dim"]).tolist(),
                        payload={
                            "asset_id": row["asset_id"],
                            "sku": row["sku"],
                            "chunk_id": row["chunk_id"],
                            "model_version": visual_version(),
                        },
                    )
                )
            client.upsert(name, points=points, wait=True)
    if client.collection_exists(name):
        offset = None
        while True:
            page, offset = client.scroll(
                name, offset=offset, limit=256, with_payload=False
            )
            stale = [point.id for point in page if str(point.id) not in valid]
            if stale:
                client.delete(name, models.PointIdsList(points=stale), wait=True)
            if offset is None:
                break
    return len(valid)
