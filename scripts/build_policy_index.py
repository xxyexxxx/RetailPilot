from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, date, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.config import settings
from RetailPilot.database import db_session, init_db
from RetailPilot.policy_index import resolve_local_model_reference


POINT_NAMESPACE = uuid.UUID("1a14cf35-7cf0-4c5a-96f7-a6047682a09a")


def _ordinal(value: str, *, open_end: bool = False) -> int:
    if not value:
        return date.max.toordinal() if open_end else date.min.toordinal()
    return date.fromisoformat(value).toordinal()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the policy chunk collection in Qdrant."
    )
    parser.add_argument("--database", type=Path, default=settings.database_path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--cache-reranker",
        action="store_true",
        help="Also download/load the configured cross-encoder and run a probe.",
    )
    args = parser.parse_args()

    from qdrant_client import QdrantClient, models
    from sentence_transformers import SentenceTransformer

    init_db(args.database)
    with db_session(args.database) as connection:
        rows = connection.execute(
            """
            SELECT
                c.chunk_id, c.contextualized_content,
                d.doc_id, d.topic, d.platform, d.category_scope, d.audience,
                d.status, d.effective_from, d.effective_to, d.version,
                d.source_uri, d.jurisdiction
            FROM policy_chunks c
            JOIN policy_documents d ON d.doc_id = c.doc_id
            WHERE c.chunk_type != 'parent'
            ORDER BY d.doc_id, c.ordinal
            """
        ).fetchall()
    if not rows:
        raise SystemExit("No policy chunks found; import/seed the policy corpus first.")

    model_kwargs = {"local_files_only": args.local_files_only}
    if settings.embedding_device != "auto":
        model_kwargs["device"] = settings.embedding_device
    model = SentenceTransformer(
        resolve_local_model_reference(settings.policy_embedding_model),
        **model_kwargs,
    )
    probe = model.encode(
        [str(rows[0]["contextualized_content"])],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    dimension = int(probe.shape[1])

    if settings.policy_qdrant_url:
        client = QdrantClient(
            url=settings.policy_qdrant_url,
            api_key=settings.policy_qdrant_api_key or None,
            timeout=30,
        )
    else:
        settings.policy_qdrant_path.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(settings.policy_qdrant_path))
    collection = settings.policy_qdrant_collection
    if args.recreate and client.collection_exists(collection):
        client.delete_collection(collection)
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )

    index_version = (
        f"{settings.policy_embedding_model}@"
        f"{datetime.now(UTC).isoformat(timespec='seconds')}"
    )
    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset : offset + args.batch_size]
        texts = [str(row["contextualized_content"]) for row in batch]
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        points = []
        for row, vector in zip(batch, vectors):
            chunk_id = str(row["chunk_id"])
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid5(POINT_NAMESPACE, chunk_id)),
                    vector=vector.tolist(),
                    payload={
                        "chunk_id": chunk_id,
                        "doc_id": row["doc_id"],
                        "topic": row["topic"],
                        "platform": row["platform"],
                        "category_scope": row["category_scope"],
                        "audience": row["audience"],
                        "status": row["status"],
                        "effective_from_ordinal": _ordinal(row["effective_from"]),
                        "effective_to_ordinal": _ordinal(
                            row["effective_to"], open_end=True
                        ),
                        "version": row["version"],
                        "source_uri": row["source_uri"],
                        "jurisdiction": row["jurisdiction"],
                        "index_version": index_version,
                    },
                )
            )
        client.upsert(collection_name=collection, points=points, wait=True)

    with db_session(args.database) as connection:
        connection.execute(
            """
            UPDATE policy_chunks
            SET embedding_model = ?, embedding_version = ?, index_version = ?
            WHERE chunk_type != 'parent'
            """,
            (
                settings.policy_embedding_model,
                settings.policy_embedding_model,
                index_version,
            ),
        )
    reranker_cached = False
    if args.cache_reranker:
        from sentence_transformers import CrossEncoder

        reranker_kwargs = {"local_files_only": args.local_files_only}
        if settings.embedding_device != "auto":
            reranker_kwargs["device"] = settings.embedding_device
        reranker = CrossEncoder(
            resolve_local_model_reference(settings.policy_reranker_model),
            **reranker_kwargs,
        )
        reranker.predict(
            [["退款多久到账", "信用卡退款通常需要三到十个工作日"]],
            show_progress_bar=False,
        )
        reranker_cached = True
    print(
        json.dumps(
            {
                "collection": collection,
                "chunks": len(rows),
                "dimension": dimension,
                "embedding_model": settings.policy_embedding_model,
                "reranker_model": settings.policy_reranker_model,
                "reranker_cached": reranker_cached,
                "index_version": index_version,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
