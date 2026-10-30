"""Prefix-stable message assembly for prompt caching.

Provider prompt caches (OpenAI / DeepSeek / Qwen / Anthropic) key on a
byte-exact prefix starting at token 0.  Any change inside the prefix
invalidates everything after it.  Two rules follow:

1.  Static content goes first, dynamic content goes last.  The stable part is
    passed as a system message; everything that changes per turn is serialized
    into a single human message.
2.  Serialization must be deterministic.  JSON payloads are dumped with sorted
    keys so an incidental dict-ordering change cannot break the prefix.

The ``cache_control`` marker is only meaningful for providers that support
explicit caching (Anthropic, Qwen).  Providers that cache automatically
(OpenAI, DeepSeek) ignore unknown fields, so emitting it unconditionally is
safe and keeps a single code path.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}


def stable_dumps(value: Any) -> str:
    """Serialize a payload deterministically for prefix-stable prompts."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def prefix_fingerprint(stable_prefix: str) -> str:
    """Return a short hash identifying a cacheable prefix version."""

    return hashlib.sha256(stable_prefix.encode("utf-8")).hexdigest()[:16]


def build_cached_messages(
    stable_prefix: str,
    dynamic: dict[str, Any] | None = None,
    *,
    dynamic_text: str | None = None,
    with_cache_control: bool = True,
) -> list[Any]:
    """Build a two-message prompt with a byte-stable system prefix.

    Args:
        stable_prefix: Content that must not change between turns.
        dynamic: Per-turn payload serialized with sorted keys.
        dynamic_text: Pre-rendered per-turn text (used when the caller already
            maintains a stable rendering, e.g. append-only history blocks).
        with_cache_control: Attach the explicit cache breakpoint to the stable
            block.  Provider-agnostic; ignored where unsupported.
    """

    if dynamic is not None and dynamic_text is not None:
        raise ValueError("pass either dynamic or dynamic_text, not both")

    if dynamic_text is not None:
        suffix = dynamic_text
    elif dynamic is not None:
        suffix = stable_dumps(dynamic)
    else:
        suffix = "{}"

    if with_cache_control:
        system_content: Any = [
            {
                "type": "text",
                "text": stable_prefix,
                "cache_control": CACHE_CONTROL_EPHEMERAL,
            }
        ]
    else:
        system_content = stable_prefix

    return [SystemMessage(content=system_content), HumanMessage(content=suffix)]
