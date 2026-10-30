"""RetailPilot.

``chat`` and ``graph`` remain public package attributes, but loading a data,
index or evaluation submodule must not eagerly initialize the whole Agent and
its optional LLM stack.
"""

from typing import Any


__version__ = "1.0.0"
__all__ = ["chat", "graph"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .agent import chat, graph

        return {"chat": chat, "graph": graph}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
