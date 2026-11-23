"""Text and image vector spaces have independent versioned encoders."""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .asset_config import asset_settings as config
from .config import settings
from .embeddings import retrieval_prefixes
from .text import hash_vector


def text_version() -> str:
    return (
        f"asset-text-v1/hash/{settings.vector_dim}"
        if config.text_model == "hash"
        else f"asset-text-v1/{config.text_model}@{config.text_revision}"
    )


def visual_version() -> str:
    return f"asset-visual-v1/{config.visual_model}@{config.visual_revision}"


def normalized(matrix) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float32)
    if result.ndim != 2 or not result.shape[1] or not np.isfinite(result).all():
        raise ValueError("invalid_embedding_matrix")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    return result / np.maximum(norms, 1e-12)


@lru_cache(maxsize=1)
def _sentence_model(name: str, revision: str, device: str, local_only: bool):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        name,
        device=None if device == "auto" else device,
        revision=revision,
        local_files_only=local_only,
    )


def text_encode(values: list[str], *, query: bool = False) -> np.ndarray:
    if config.text_model == "hash":
        return normalized([hash_vector(value) for value in values])
    model = _sentence_model(
        config.text_model,
        config.text_revision,
        settings.embedding_device,
        config.model_local_only,
    )
    prefix = retrieval_prefixes(config.text_model)[0 if query else 1]
    return normalized(
        model.encode(
            [prefix + value for value in values],
            batch_size=16,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )


class SiglipEncoder:
    def __init__(self, name: str, revision: str, local_only: bool, device: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else device
        )
        options = {"revision": revision, "local_files_only": local_only}
        self.processor = AutoProcessor.from_pretrained(name, **options)
        self.model = AutoModel.from_pretrained(name, **options).eval().to(self.device)
        if self.model.config.model_type not in {"siglip", "siglip2"}:
            raise ValueError("unsupported_visual_encoder")

    def _features(self, inputs, encoder) -> np.ndarray:
        with self.torch.inference_mode():
            result = encoder(**inputs.to(self.device))
            if hasattr(result, "pooler_output"):
                result = result.pooler_output
            return normalized(result.float().cpu().numpy())

    def texts(self, values: list[str]) -> np.ndarray:
        inputs = self.processor(
            text=values,
            padding="max_length",
            truncation=True,
            max_length=self.model.config.text_config.max_position_embeddings,
            return_tensors="pt",
        )
        return self._features(inputs, self.model.get_text_features)

    def images(self, values: list[bytes]) -> np.ndarray:
        from io import BytesIO

        from PIL import Image

        images = []
        for raw in values:
            with Image.open(BytesIO(raw)) as image:
                images.append(image.convert("RGB"))
        return self._features(
            self.processor(images=images, return_tensors="pt"),
            self.model.get_image_features,
        )


@lru_cache(maxsize=1)
def _visual(name: str, revision: str, local_only: bool, device: str) -> SiglipEncoder:
    return SiglipEncoder(name, revision, local_only, device)


def visual_encoder() -> SiglipEncoder:
    return _visual(
        config.visual_model,
        config.visual_revision,
        config.model_local_only,
        settings.embedding_device,
    )
