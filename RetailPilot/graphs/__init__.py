"""Lazy compatibility exports avoid a graph/facade import cycle."""

__all__ = ["graph", "retrieval_search"]


def __getattr__(name):
    if name == "graph":
        from ..agent import graph

        return graph
    if name == "retrieval_search":
        from ..retrieval import search

        return search
    raise AttributeError(name)
