from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings


def resolve_local_model_reference(model_name: str) -> str:
    """Use a complete HF snapshot even when the `main` ref points elsewhere."""

    direct = Path(model_name)
    if direct.exists():
        return str(direct)
    configured_home = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface"))
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
class PolicyRerank:
    ranking: list[str]
    scores: dict[str, float]
    model: str
    available: bool
    error: str = ""


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

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> PolicyRerank:
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
                [query, str(item["contextualized_content"])] for item in candidates
            ]
            raw_scores = np.asarray(
                model.predict(pairs, show_progress_bar=False)
            ).reshape(-1)
            scored = sorted(
                zip(candidates, raw_scores.tolist()),
                key=lambda item: (-float(item[1]), str(item[0]["chunk_id"])),
            )
            return PolicyRerank(
                ranking=[str(item[0]["chunk_id"]) for item in scored],
                scores={str(item[0]["chunk_id"]): float(item[1]) for item in scored},
                model=settings.policy_reranker_model,
                available=True,
            )
        except Exception as exc:  # noqa: BLE001 - optional reranker boundary
            return PolicyRerank(
                ranking=[str(item["chunk_id"]) for item in candidates],
                scores={},
                model=settings.policy_reranker_model,
                available=False,
                error=str(exc),
            )


_RERANKER: PolicyCrossEncoder | None = None


def get_policy_reranker() -> PolicyCrossEncoder:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = PolicyCrossEncoder()
    return _RERANKER


def reset_policy_index_cache() -> None:
    global _RERANKER
    _RERANKER = None
