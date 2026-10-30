from __future__ import annotations

from typing import Any

from .catalog import list_brands
from .llm import analyze_intents
from .query import parse_query


def understand_query(
    text: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return parse_query(text, context=context, known_brands=list_brands()).to_dict()


__all__ = [
    "analyze_intents",
    "parse_query",
    "understand_query",
]
