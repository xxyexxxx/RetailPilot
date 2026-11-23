"""One encoder and index lifecycle for each domain, with independent visual space."""

from __future__ import annotations

import os
from functools import lru_cache

from .asset_config import asset_settings
from .asset_embeddings import normalized, visual_encoder, visual_version
from .config import settings
from .database import db_session
from .text import hash_vector, vector_to_blob


def model_spec(domain):
    fallback = (
        settings.embedding_model
        if domain == "product"
        else (
            settings.policy_embedding_model
            if settings.policy_vector_backend == "qdrant"
            else "disabled"
        )
    )
    if fallback == "auto":
        fallback = "disabled"
    return (
        os.getenv(f"RETAILPILOT_KNOWLEDGE_{domain.upper()}_MODEL", fallback),
        os.getenv(f"RETAILPILOT_KNOWLEDGE_{domain.upper()}_REVISION", "main"),
    )


def version(domain):
    model, revision = model_spec(domain)
    return f"knowledge-text-v1/{domain}/{model}@{revision}"


@lru_cache(maxsize=2)
def _model(name, revision, device, local):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        name,
        revision=revision,
        device=None if device == "auto" else device,
        local_files_only=local,
    )


def encode_text(texts, domain, *, query=False):
    name, revision = model_spec(domain)
    if name == "disabled":
        raise RuntimeError("text_dense_disabled")
    if name == "hash":
        return normalized([hash_vector(text) for text in texts])
    prefix = ("query: " if query else "passage: ") if "e5" in name.casefold() else ""
    return normalized(
        _model(
            name, revision, settings.embedding_device, asset_settings.model_local_only
        ).encode(
            [prefix + text for text in texts],
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )


def build(domain, *, space="text", force=False, source_id=None):
    if space == "text" and model_spec(domain)[0] == "disabled":
        return {
            "domain": domain,
            "space": space,
            "built": 0,
            "failed": 0,
            "unchanged": 0,
            "disabled": True,
        }
    from .asset_store import store
    from .knowledge import sync_products

    model_version = version(domain) if space == "text" else visual_version()
    built, failed = 0, 0
    with db_session() as db:
        sync_products(db)
        rows = db.execute(
            """SELECT u.unit_id,u.search_text,u.content_hash,e.preview_key,v.model_version,
                          v.content_hash AS indexed_hash FROM knowledge_units u
                          JOIN knowledge_sources s USING(source_id)
                          LEFT JOIN asset_elements e USING(element_id)
                          LEFT JOIN knowledge_vectors v ON v.unit_id=u.unit_id AND v.space=?
                          WHERE s.domain=? AND (? IS NULL OR s.source_id=?) AND (?='text' AND trim(u.search_text)!='' OR
                          ?='visual' AND u.modality='image' AND e.preview_key IS NOT NULL)
                          ORDER BY u.unit_id""",
            (space, domain, source_id, source_id, space, space),
        ).fetchall()
    pending = [
        r
        for r in rows
        if force
        or r["model_version"] != model_version
        or r["indexed_hash"] != r["content_hash"]
    ]
    for offset in range(0, len(pending), 16):
        batch = pending[offset : offset + 16]
        try:
            matrix = (
                encode_text([r["search_text"] for r in batch], domain)
                if space == "text"
                else (
                    visual_encoder().images(
                        [store().read(r["preview_key"]) for r in batch]
                    )
                )
            )
            with db_session() as db:
                for row, vector in zip(batch, matrix, strict=True):
                    current = db.execute(
                        "SELECT content_hash FROM knowledge_units WHERE unit_id=?",
                        (row["unit_id"],),
                    ).fetchone()
                    if current and current[0] == row["content_hash"]:
                        db.execute(
                            "INSERT OR REPLACE INTO knowledge_vectors VALUES(?,?,?,?,?,?)",
                            (
                                row["unit_id"],
                                space,
                                model_version,
                                row["content_hash"],
                                len(vector),
                                vector_to_blob(vector),
                            ),
                        )
                        built += 1
        except Exception:  # noqa: BLE001 - retryable optional model batch
            failed += len(batch)
    return {
        "domain": domain,
        "space": space,
        "model": model_version,
        "built": built,
        "failed": failed,
        "unchanged": len(rows) - len(pending),
    }
