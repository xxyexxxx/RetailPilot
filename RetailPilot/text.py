from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from html import unescape
from html.parser import HTMLParser

import jieba
import numpy as np

from .config import settings
from .taxonomy import CONCEPT_ALIASES, alias_in_text


_ASCII_TOKEN = re.compile(r"[a-z0-9][a-z0-9.+#_-]*", re.IGNORECASE)
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def plain_text(value: object, limit: int = 4000) -> str:
    parser = _PlainTextParser()
    parser.feed(unescape(str(value or "")))
    parser.close()
    return compact_text(" ".join(parser.parts), limit)


def compact_text(value: object, limit: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        text = " ".join(compact_text(item, limit=limit) for item in value)
    elif isinstance(value, dict):
        text = " ".join(f"{key} {compact_text(val, limit=limit)}" for key, val in value.items())
    else:
        text = str(value)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def expand_concepts(text: str) -> list[str]:
    expanded: list[str] = []
    for alias, terms in CONCEPT_ALIASES.items():
        if alias_in_text(text, alias):
            # The concept key itself is the canonical form (e.g. the Chinese
            # "人工智能"); its term table contains only related search terms.
            # Keeping the key guarantees Chinese bigram recall for Chinese titles.
            expanded.append(alias)
            expanded.extend(terms)
    return list(dict.fromkeys(expanded))


def tokenize(text: str, include_expansion: bool = True) -> list[str]:
    lowered = compact_text(text, limit=12000).lower()
    tokens = _ASCII_TOKEN.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        tokens.extend(token.strip() for token in jieba.cut(run) if token.strip())
        # Character bigrams make Chinese catalog recall robust for model names,
        # brand compounds and rare words that are absent from jieba's lexicon.
        if len(run) > 1:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    if include_expansion:
        for phrase in expand_concepts(lowered):
            tokens.extend(_ASCII_TOKEN.findall(phrase.lower()))
    return [token for token in tokens if len(token) > 1 or token.isdigit()]


def fast_catalog_tokens(text: str) -> list[str]:
    """Deterministic tokenizer for large one-off imports without jieba overhead."""
    lowered = compact_text(text, limit=12000).lower()
    tokens = _ASCII_TOKEN.findall(lowered)
    for run in _CJK_RUN.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


def fast_hash_vector(text: str, dim: int | None = None) -> np.ndarray:
    size = dim or settings.vector_dim
    counts = Counter(fast_catalog_tokens(text))
    vector = np.zeros(size, dtype=np.float32)
    for token, count in counts.items():
        index, sign = _hash_slot(token, size)
        vector[index] += sign * (1.0 + math.log(count))
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


def _hash_slot(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, "little", signed=False)
    index = number % dim
    sign = 1.0 if number & 1 else -1.0
    return index, sign


def hash_vector(text: str, dim: int | None = None) -> np.ndarray:
    size = dim or settings.vector_dim
    counts = Counter(tokenize(text))
    vector = np.zeros(size, dtype=np.float32)
    for token, count in counts.items():
        index, sign = _hash_slot(token, size)
        vector[index] += sign * (1.0 + math.log(count))
    tokens = list(counts)
    for left, right in zip(tokens, tokens[1:]):
        index, sign = _hash_slot(f"{left}::{right}", size)
        vector[index] += sign * 0.5
    norm = float(np.linalg.norm(vector))
    if norm:
        vector /= norm
    return vector


def vector_to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype=np.float32).tobytes()


def vector_from_blob(blob: bytes, dim: int | None = None) -> np.ndarray:
    vector = np.frombuffer(blob, dtype=np.float32)
    expected = dim or settings.vector_dim
    if vector.size != expected:
        raise ValueError(f"vector dimension mismatch: expected {expected}, got {vector.size}")
    return vector


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    if not left.size or not right.size:
        return 0.0
    return float(np.dot(left, right))
