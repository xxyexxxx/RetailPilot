from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import settings
from .text import compact_text


INDEX_FORMAT_VERSION = 1
TEXT_TEMPLATE_VERSION = "product-v1"


def model_slug(model_name: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", model_name.casefold()).strip("-")


def retrieval_prefixes(model_name: str) -> tuple[str, str]:
    if "e5" in model_name.casefold():
        return "query: ", "passage: "
    return "", ""


def product_document(product: dict[str, Any]) -> str:
    """Stable, compact product text used by every neural index version."""
    fields = [
        product.get("title", ""),
        product.get("brand", ""),
        product.get("category_label", ""),
        product.get("category", ""),
        product.get("tags_text", ""),
        product.get("attributes_text", ""),
        product.get("description", ""),
    ]
    return compact_text(" | ".join(str(value) for value in fields if value), 6000)


@dataclass(slots=True)
class DenseRecall:
    ranking: list[str]
    scores: dict[str, float]
    backend: str
    model: str
    index_version: str
    semantic: bool
    fallback_reason: str = ""


class NeuralEmbeddingIndex:
    def __init__(self, directory: Path, metadata: dict[str, Any]) -> None:
        self.directory = directory
        self.metadata = metadata
        self.model_name = str(metadata["model"])
        self.index_version = str(metadata.get("index_version") or "unknown")
        self.query_prefix = str(metadata.get("query_prefix") or "")
        self.skus = json.loads((directory / metadata["skus_file"]).read_text(encoding="utf-8"))
        self.matrix = np.load(directory / metadata["matrix_file"], mmap_mode="r")
        if self.matrix.ndim != 2 or self.matrix.shape[0] != len(self.skus):
            raise ValueError("embedding index matrix and SKU list are inconsistent")
        self.sku_to_index = {sku: index for index, sku in enumerate(self.skus)}
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self._query_cache: dict[str, np.ndarray] = {}

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._model_lock:
            if self._model is not None:
                return self._model
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "neural index exists but sentence-transformers is not installed; "
                    "install requirements-embedding.txt"
                ) from exc
            kwargs: dict[str, Any] = {}
            if settings.embedding_device != "auto":
                kwargs["device"] = settings.embedding_device
            # The index builder is responsible for downloading the model. Runtime
            # retrieval stays offline and falls back to hash recall if the local
            # model cache is incomplete.
            self._model = SentenceTransformer(
                self.model_name, local_files_only=True, **kwargs
            )
        return self._model

    def encode_query(self, text: str) -> np.ndarray:
        cache_key = compact_text(text, 2000)
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached
        vector = self._load_model().encode(
            [f"{self.query_prefix}{cache_key}"],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32, copy=False)
        if len(self._query_cache) >= 256:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[cache_key] = vector
        return vector

    def recall(
        self, query: str, candidate_skus: Iterable[str], limit: int
    ) -> DenseRecall:
        pairs = [
            (sku, self.sku_to_index[sku])
            for sku in candidate_skus
            if sku in self.sku_to_index
        ]
        if not pairs:
            return DenseRecall([], {}, "neural-flat", self.model_name, self.index_version, True)
        query_vector = self.encode_query(query)
        indices = np.fromiter((index for _, index in pairs), dtype=np.int64)
        scores = np.asarray(self.matrix[indices], dtype=np.float32) @ query_vector
        top_n = min(limit, len(scores))
        if top_n < len(scores):
            top = np.argpartition(-scores, top_n - 1)[:top_n]
            order = top[np.argsort(-scores[top])]
        else:
            order = np.argsort(-scores)
        ranking = [pairs[int(position)][0] for position in order]
        score_map = {
            pairs[int(position)][0]: float(scores[int(position)]) for position in order
        }
        return DenseRecall(
            ranking,
            score_map,
            "neural-flat",
            self.model_name,
            self.index_version,
            True,
        )


_INDEX_CACHE: tuple[str, NeuralEmbeddingIndex | None] | None = None
_INDEX_LOCK = threading.Lock()


def _read_metadata(directory: Path) -> dict[str, Any] | None:
    path = directory / "metadata.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("format_version") != INDEX_FORMAT_VERSION:
        return None
    if not (directory / str(payload.get("matrix_file", ""))).exists():
        return None
    if not (directory / str(payload.get("skus_file", ""))).exists():
        return None
    return payload


def _resolve_index() -> tuple[Path, dict[str, Any]] | None:
    root = settings.embedding_index_dir
    if not root.exists():
        return None
    requested = settings.embedding_model
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for directory in root.iterdir():
        if not directory.is_dir():
            continue
        metadata = _read_metadata(directory)
        if metadata is None:
            continue
        if requested not in {"", "auto", "hash"} and metadata.get("model") != requested:
            continue
        candidates.append((directory, metadata))
    if requested == "hash" or not candidates:
        return None
    return max(
        candidates,
        key=lambda item: str(item[1].get("built_at") or item[0].stat().st_mtime),
    )


def get_neural_index() -> NeuralEmbeddingIndex | None:
    global _INDEX_CACHE
    cache_key = f"{settings.embedding_index_dir}|{settings.embedding_model}"
    if _INDEX_CACHE is not None and _INDEX_CACHE[0] == cache_key:
        return _INDEX_CACHE[1]
    with _INDEX_LOCK:
        resolved = _resolve_index()
        index = NeuralEmbeddingIndex(*resolved) if resolved else None
        _INDEX_CACHE = (cache_key, index)
        return index


def reset_embedding_index_cache() -> None:
    global _INDEX_CACHE
    _INDEX_CACHE = None


def index_metadata(
    model_name: str,
    dimension: int,
    document_count: int,
    matrix_file: str = "vectors.npy",
    skus_file: str = "skus.json",
) -> dict[str, Any]:
    query_prefix, passage_prefix = retrieval_prefixes(model_name)
    built_at = datetime.now(UTC).isoformat(timespec="seconds")
    return {
        "format_version": INDEX_FORMAT_VERSION,
        "index_version": f"{model_slug(model_name)}-{built_at}",
        "model": model_name,
        "dimension": dimension,
        "document_count": document_count,
        "normalized": True,
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "text_template_version": TEXT_TEMPLATE_VERSION,
        "matrix_file": matrix_file,
        "skus_file": skus_file,
        "built_at": built_at,
    }
