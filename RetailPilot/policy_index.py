from __future__ import annotations

import threading
import os
import atexit
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import settings


def resolve_local_model_reference(model_name: str) -> str:
    """Use a complete HF snapshot even when the `main` ref points elsewhere."""

    direct = Path(model_name)
    if direct.exists():
        return str(direct)
    configured_home = Path(
        os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    cache_roots = [
        Path(os.getenv("HF_HUB_CACHE", configured_home / "hub")),
        Path.home() / ".cache" / "huggingface" / "hub",
    ]
    repositories = [
        root / f"models--{model_name.replace('/', '--')}" / "snapshots"
        for root in dict.fromkeys(cache_roots)
    ]
    complete = [
        path
        for repository in repositories
        if repository.exists()
        for path in repository.iterdir()
        if path.is_dir()
        and (path / "config.json").exists()
        and any(
            (path / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
    ]
    if not complete:
        return model_name
    return str(max(complete, key=lambda path: path.stat().st_mtime))


@dataclass(slots=True)
class PolicyDenseRecall:
    ranking: list[str]
    scores: dict[str, float]
    backend: str
    model: str
    index_version: str
    available: bool
    error: str = ""


@dataclass(slots=True)
class PolicyRerank:
    ranking: list[str]
    scores: dict[str, float]
    model: str
    available: bool
    error: str = ""


class QdrantPolicyIndex:
    def __init__(self) -> None:
        self._client: Any | None = None
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client
        from qdrant_client import QdrantClient

        if settings.policy_qdrant_url:
            self._client = QdrantClient(
                url=settings.policy_qdrant_url,
                api_key=settings.policy_qdrant_api_key or None,
                timeout=10,
            )
        else:
            settings.policy_qdrant_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(settings.policy_qdrant_path))
        atexit.register(self._client.close)
        return self._client

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            kwargs: dict[str, Any] = {"local_files_only": True}
            if settings.embedding_device != "auto":
                kwargs["device"] = settings.embedding_device
            self._model = SentenceTransformer(
                resolve_local_model_reference(settings.policy_embedding_model),
                **kwargs,
            )
        return self._model

    @staticmethod
    def _filter(
        *,
        topics: Iterable[str],
        platform: str,
        categories: Iterable[str],
        as_of: date,
    ) -> Any:
        from qdrant_client import models

        category_values = ["*", *sorted(set(categories))]
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="topic", match=models.MatchAny(any=list(topics))
                ),
                models.FieldCondition(
                    key="platform", match=models.MatchAny(any=[platform, "*"])
                ),
                models.FieldCondition(
                    key="status", match=models.MatchValue(value="active")
                ),
                models.FieldCondition(
                    key="audience",
                    match=models.MatchAny(any=["customer", "public"]),
                ),
                models.FieldCondition(
                    key="category_scope",
                    match=models.MatchAny(any=category_values),
                ),
                models.FieldCondition(
                    key="effective_from_ordinal",
                    range=models.Range(lte=as_of.toordinal()),
                ),
                models.FieldCondition(
                    key="effective_to_ordinal",
                    range=models.Range(gte=as_of.toordinal()),
                ),
            ]
        )

    def search(
        self,
        query: str,
        *,
        topics: Iterable[str],
        platform: str,
        categories: Iterable[str],
        as_of: date,
        limit: int,
    ) -> PolicyDenseRecall:
        try:
            client = self._load_client()
            if not client.collection_exists(settings.policy_qdrant_collection):
                raise RuntimeError("policy Qdrant collection has not been built")
            model = self._load_model()
            vector = model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0].astype(np.float32, copy=False)
            result = client.query_points(
                collection_name=settings.policy_qdrant_collection,
                query=vector.tolist(),
                query_filter=self._filter(
                    topics=topics,
                    platform=platform,
                    categories=categories,
                    as_of=as_of,
                ),
                limit=limit,
                with_payload=True,
            )
            points = list(result.points)
            ranking = [
                str((point.payload or {}).get("chunk_id") or point.id)
                for point in points
            ]
            scores = {
                str((point.payload or {}).get("chunk_id") or point.id): float(
                    point.score
                )
                for point in points
            }
            return PolicyDenseRecall(
                ranking=ranking,
                scores=scores,
                backend="qdrant",
                model=settings.policy_embedding_model,
                index_version=str(
                    (points[0].payload or {}).get("index_version", "")
                    if points
                    else ""
                ),
                available=True,
            )
        except Exception as exc:
            return PolicyDenseRecall(
                ranking=[],
                scores={},
                backend="qdrant",
                model=settings.policy_embedding_model,
                index_version="",
                available=False,
                error=str(exc),
            )


class PolicyCrossEncoder:
    def __init__(self) -> None:
        self._model: Any | None = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import CrossEncoder

            kwargs: dict[str, Any] = {"local_files_only": True}
            if settings.embedding_device != "auto":
                kwargs["device"] = settings.embedding_device
            self._model = CrossEncoder(
                resolve_local_model_reference(settings.policy_reranker_model),
                **kwargs,
            )
        return self._model

    def rerank(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> PolicyRerank:
        if not settings.policy_reranker_enabled:
            return PolicyRerank(
                ranking=[str(item["chunk_id"]) for item in candidates],
                scores={},
                model=settings.policy_reranker_model,
                available=False,
                error="disabled",
            )
        try:
            model = self._load()
            pairs = [
                [query, str(item["contextualized_content"])]
                for item in candidates
            ]
            raw_scores = np.asarray(model.predict(pairs, show_progress_bar=False)).reshape(-1)
            scored = sorted(
                zip(candidates, raw_scores.tolist()),
                key=lambda item: (-float(item[1]), str(item[0]["chunk_id"])),
            )
            return PolicyRerank(
                ranking=[str(item[0]["chunk_id"]) for item in scored],
                scores={
                    str(item[0]["chunk_id"]): float(item[1]) for item in scored
                },
                model=settings.policy_reranker_model,
                available=True,
            )
        except Exception as exc:
            return PolicyRerank(
                ranking=[str(item["chunk_id"]) for item in candidates],
                scores={},
                model=settings.policy_reranker_model,
                available=False,
                error=str(exc),
            )


_DENSE_INDEX: QdrantPolicyIndex | None = None
_RERANKER: PolicyCrossEncoder | None = None


def get_policy_dense_index() -> QdrantPolicyIndex | None:
    global _DENSE_INDEX
    if settings.policy_vector_backend != "qdrant":
        return None
    if _DENSE_INDEX is None:
        _DENSE_INDEX = QdrantPolicyIndex()
    return _DENSE_INDEX


def get_policy_reranker() -> PolicyCrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = PolicyCrossEncoder()
    return _RERANKER


def reset_policy_index_cache() -> None:
    global _DENSE_INDEX, _RERANKER
    _DENSE_INDEX = None
    _RERANKER = None
