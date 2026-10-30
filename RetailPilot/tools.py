from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .catalog import get_product
from .config import settings
from .execution import tool_pool
from .database import db_session
from .policy import PolicyTopic, retrieve_policies
from .retrieval import search_engine
from .schemas import ParsedQuery

ToolScope = Literal["pre_sales", "after_sales"]
RetryPolicy = Literal["never", "safe_once"]
SideEffect = Literal["read_only", "write"]


class CatalogSearchInput(BaseModel):
    query: str = Field(default="", max_length=500)
    parsed_query: dict[str, Any] | None = None
    user_id: str = Field(default="demo-user", min_length=1, max_length=80)
    limit: int = Field(default=10, ge=1, le=30)
    mode: Literal["hybrid", "fts"] = "hybrid"
    session_id: str = Field(default="", max_length=100)

    @model_validator(mode="after")
    def require_query(self) -> "CatalogSearchInput":
        if not self.query.strip() and not self.parsed_query:
            raise ValueError("query or parsed_query is required")
        return self


class ProductDetailInput(BaseModel):
    sku: str = Field(min_length=2, max_length=80)


class PolicyProductContext(BaseModel):
    sku: str | None = Field(default=None, max_length=80)
    category: str | None = Field(default=None, max_length=160)
    category_path: str | None = Field(default=None, max_length=300)
    product_type: str | None = Field(default=None, max_length=80)
    seller_type: str | None = Field(default=None, max_length=80)
    seller_name: str | None = Field(default=None, max_length=160)
    fulfillment_mode: str | None = Field(default=None, max_length=80)
    purchased_at: str | None = Field(default=None, max_length=40)


class PolicyRetrieveInput(BaseModel):
    as_of: date | None = None
    query: str = Field(min_length=1, max_length=500)
    topic: PolicyTopic
    platform: str = Field(
        default_factory=lambda: settings.commerce_platform,
        min_length=1,
        max_length=80,
    )
    product_context: PolicyProductContext | None = None
    limit: int = Field(default=4, ge=1, le=8)


class OrderQueryInput(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    order_id: str | None = Field(default=None, max_length=80)
    product_query: str | None = Field(default=None, max_length=120)
    created_after: str | None = Field(default=None, max_length=32)
    created_before: str | None = Field(default=None, max_length=32)
    limit: int = Field(default=5, ge=1, le=5)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    scopes: frozenset[str]
    input_model: type[BaseModel]
    function: Callable[..., dict[str, Any] | None]
    side_effect: SideEffect = "read_only"
    preconditions: tuple[str, ...] = ()
    retry_policy: RetryPolicy = "never"
    timeout_ms: int = 3000
    output_compressor: Callable[[Any], Any] | None = None

    def public_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "scopes": sorted(self.scopes),
            "input_schema": self.input_model.model_json_schema(),
            "side_effect": self.side_effect,
            "preconditions": list(self.preconditions),
            "retry_policy": self.retry_policy,
            "timeout_ms": self.timeout_ms,
        }


class ToolRegistry:
    """Schema-first, scope-bounded tool registry for customer-service skills."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"duplicate tool: {spec.name}")
        self._specs[spec.name] = spec

    def get_spec(self, name: str, scope: str) -> ToolSpec:
        if name not in self._specs:
            raise KeyError(name)
        spec = self._specs[name]
        if scope not in spec.scopes:
            raise PermissionError(f"{name} is not allowed in scope {scope}")
        return spec

    def get(self, name: str, scope: str) -> Callable[..., Any]:
        """Compatibility API retained for earlier examples."""

        return self.get_spec(name, scope).function

    def list(self, scope: str) -> list[str]:
        return sorted(name for name, spec in self._specs.items() if scope in spec.scopes)

    def schemas(self, scope: str) -> list[dict[str, Any]]:
        return [self.get_spec(name, scope).public_schema() for name in self.list(scope)]

    def execute(
        self,
        name: str,
        scope: str,
        arguments: dict[str, Any],
        *,
        state: dict[str, Any] | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        spec: ToolSpec | None = None
        attempts = 0
        timed_out = False
        try:
            spec = self.get_spec(name, scope)
            validated = spec.input_model.model_validate(arguments)
            missing = [key for key in spec.preconditions if not (state or {}).get(key)]
            if missing:
                raise ValueError(f"missing tool preconditions: {', '.join(missing)}")

            max_attempts = (
                2
                if spec.retry_policy == "safe_once" and spec.side_effect == "read_only"
                else 1
            )
            while True:
                attempts += 1
                try:
                    remaining_ms = spec.timeout_ms if deadline is None else min(
                        spec.timeout_ms, int((deadline - time.perf_counter()) * 1000)
                    )
                    if remaining_ms <= 0:
                        raise _ToolTimeoutError("tool deadline exhausted")
                    data, observation = self._invoke(spec, validated.model_dump(), remaining_ms)
                    break
                except _ToolTimeoutError:
                    timed_out = True
                    if attempts >= max_attempts or (deadline is not None and time.perf_counter() >= deadline):
                        raise
                except Exception as exc:
                    if attempts >= max_attempts or (deadline is not None and time.perf_counter() >= deadline):
                        raise _ToolExecutionError(str(exc)) from exc

            return {
                "ok": True,
                "tool": name,
                "data": data,
                "observation": observation,
                "error": None,
                "trace": {
                    "node": "tool_execution",
                    "tool": name,
                    "scope": scope,
                    "side_effect": spec.side_effect,
                    "retry_policy": spec.retry_policy,
                    "attempts": attempts,
                    "timeout_ms": spec.timeout_ms,
                    "timed_out": timed_out,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "observation_size": len(str(observation)),
                    "status": "success",
                    "error": None,
                },
            }
        except (KeyError, PermissionError, ValueError) as exc:
            error_message = str(exc)[:400]
            error_type = (
                "unknown_tool"
                if isinstance(exc, KeyError)
                else "scope_denied"
                if isinstance(exc, PermissionError)
                else "invalid_arguments"
            )
        except _ToolTimeoutError as exc:
            error_type = "timeout"
            error_message = str(exc)[:400]
        except Exception as exc:  # tool boundary converts failures to data
            error_type = "tool_error"
            error_message = str(exc)[:400]
        error = {"type": error_type, "message": error_message}
        return {
            "ok": False,
            "tool": name,
            "data": None,
            "observation": None,
            "error": error,
            "trace": {
                "node": "tool_execution",
                "tool": name,
                "scope": scope,
                "side_effect": spec.side_effect if spec else None,
                "retry_policy": spec.retry_policy if spec else None,
                "attempts": attempts,
                "timeout_ms": spec.timeout_ms if spec else None,
                "timed_out": timed_out,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "status": "error",
                "error_type": error_type,
                "error": error,
            },
        }

    @staticmethod
    def _invoke(
        spec: ToolSpec, arguments: dict[str, Any], timeout_ms: int | None = None
    ) -> tuple[Any, Any]:
        """Run one tool attempt without blocking the caller past its deadline."""

        def call():
            data = spec.function(**arguments)
            observation = spec.output_compressor(data) if spec.output_compressor else data
            return data, observation

        try:
            return tool_pool.call(
                call,
                time.perf_counter()
                + (spec.timeout_ms if timeout_ms is None else timeout_ms) / 1000,
            )
        except TimeoutError as exc:
            raise _ToolTimeoutError(f"{spec.name} deadline exhausted") from exc


class _ToolTimeoutError(TimeoutError):
    """Internal deadline marker converted to structured tool output."""


class _ToolExecutionError(RuntimeError):
    """Marks an exception raised after a validated tool invocation began."""


def _parsed_query(payload: dict[str, Any]) -> ParsedQuery:
    allowed = {item.name for item in fields(ParsedQuery)}
    return ParsedQuery(**{key: value for key, value in payload.items() if key in allowed})


def catalog_search(
    query: str = "",
    parsed_query: dict[str, Any] | None = None,
    user_id: str = "demo-user",
    limit: int = 10,
    mode: str = "hybrid",
    session_id: str = "",
) -> dict[str, Any]:
    parsed: str | ParsedQuery = _parsed_query(parsed_query) if parsed_query else query
    return search_engine.search(
        parsed, user_id=user_id, limit=limit, mode=mode, session_id=session_id
    )


def product_detail(sku: str) -> dict[str, Any] | None:
    return get_product(sku)


def policy_retrieve(
    query: str,
    topic: PolicyTopic,
    platform: str | None = None,
    product_context: dict[str, Any] | None = None,
    limit: int = 4,
    as_of: date | None = None,
) -> dict[str, Any]:
    return retrieve_policies(
        query,
        platform=platform or settings.commerce_platform,
        product_context=product_context,
        as_of=as_of,
        topic=topic,
        limit=limit,
    )


def order_query(
    user_id: str,
    order_id: str | None = None,
    product_query: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    with db_session() as connection:
        if order_id:
            rows = connection.execute(
                """
                SELECT o.*, p.title, p.image_url, p.category, p.category_label,
                       COALESCE(pc.category_path, '') AS category_path,
                       COALESCE(pc.seller_name, '') AS seller_name
                FROM orders o
                JOIN products p ON p.sku = o.sku
                LEFT JOIN product_commerce pc ON pc.sku = p.sku
                WHERE o.order_id = ? AND o.user_id = ?
                """,
                (order_id, user_id),
            ).fetchall()
        else:
            clauses = ["o.user_id = ?"]
            params: list[Any] = [user_id]
            if product_query:
                clauses.append("LOWER(p.title) LIKE LOWER(?)")
                params.append(f"%{product_query.strip()}%")
            if created_after:
                clauses.append("o.created_at >= ?")
                params.append(created_after)
            if created_before:
                clauses.append("o.created_at <= ?")
                params.append(created_before)
            # No lookup criteria means the backwards-compatible latest-order
            # behavior. Product/date lookups return several candidates so the
            # planner can ask the user to disambiguate instead of guessing.
            effective_limit = 1 if not any((product_query, created_after, created_before)) else limit
            rows = connection.execute(
                f"""
                SELECT o.*, p.title, p.image_url, p.category, p.category_label,
                       COALESCE(pc.category_path, '') AS category_path,
                       COALESCE(pc.seller_name, '') AS seller_name
                FROM orders o
                JOIN products p ON p.sku = o.sku
                LEFT JOIN product_commerce pc ON pc.sku = p.sku
                WHERE {' AND '.join(clauses)}
                ORDER BY o.created_at DESC LIMIT ?
                """,
                [*params, effective_limit],
            ).fetchall()

        orders = []
        for row in rows:
            order = {**dict(row), "events": []}
            order["product_context"] = {
                key: value
                for key, value in {
                    "sku": order.get("sku"),
                    "category": order.get("category"),
                    "category_path": order.get("category_path"),
                    "seller_name": order.get("seller_name"),
                    "purchased_at": order.get("created_at"),
                }.items()
                if value not in (None, "")
            }
            orders.append(order)
        # Logistics are only needed after a unique match. Candidate lists stay
        # small and do not carry unnecessary event payloads into the planner.
        if len(orders) == 1:
            events = connection.execute(
                """
                SELECT event_time, status, location, description
                FROM logistics_events WHERE order_id = ? ORDER BY event_time
                """,
                (orders[0]["order_id"],),
            ).fetchall()
            orders[0]["events"] = [dict(item) for item in events]

    order = orders[0] if len(orders) == 1 else None
    citations = []
    if order:
        events = order.get("events") or []
        citations = [
            {
                "order_id": order["order_id"],
                "field": "title",
                "value": order["title"],
                "source": "local_orders",
            },
            {
                "order_id": order["order_id"],
                "field": "status",
                "value": order["status"],
                "source": "local_orders",
            },
        ]
        if events:
            citations.extend(
                [
                    {
                        "order_id": order["order_id"],
                        "field": "latest_logistics",
                        "value": events[-1].get("description") or events[-1].get("status"),
                        "source": "local_logistics_events",
                    },
                    {
                        "order_id": order["order_id"],
                        "field": "logistics_events",
                        "value": events,
                        "source": "local_logistics_events",
                    },
                ]
            )
    return {
        "order": order,
        "orders": orders,
        "match_count": len(orders),
        "lookup": {
            "product_query": product_query,
            "created_after": created_after,
            "created_before": created_before,
            "latest_fallback": not any((product_query, created_after, created_before)),
        },
        "citations": citations,
    }


def _compress_search(data: Any) -> dict[str, Any]:
    payload = data or {}
    return {
        "request_id": payload.get("request_id"),
        "products": [
            {
                "sku": item.get("sku"),
                "title": item.get("title"),
                "price": item.get("price"),
                "reason": item.get("reason"),
            }
            for item in (payload.get("products") or [])[:8]
        ],
        "relaxations": payload.get("relaxations") or [],
    }


def _compress_policy(data: Any) -> dict[str, Any]:
    payload = data or {}
    return {
        "topic": payload.get("topic"),
        "product_context": payload.get("product_context") or {},
        "documents": [
            {
                "doc_id": item.get("doc_id"),
                "chunk_id": item.get("chunk_id"),
                "section_path": item.get("section_path"),
                "title": item.get("title"),
                "version": item.get("version"),
                "category": item.get("category"),
                "evidence": item.get("evidence"),
            }
            for item in (payload.get("documents") or [])[:4]
        ],
        "retrieval_version": payload.get("retrieval_version"),
        "trace": payload.get("trace") or {},
    }


registry = ToolRegistry()
registry.register(
    ToolSpec(
        "catalog_search",
        "Search and rank catalog products under parsed customer constraints.",
        frozenset({"pre_sales"}),
        CatalogSearchInput,
        catalog_search,
        retry_policy="safe_once",
        timeout_ms=5000,
        output_compressor=_compress_search,
    )
)
registry.register(
    ToolSpec(
        "product_detail",
        "Read one catalog product by SKU.",
        frozenset({"pre_sales"}),
        ProductDetailInput,
        product_detail,
        retry_policy="safe_once",
    )
)
registry.register(
    ToolSpec(
        "policy_retrieve",
        "Retrieve versioned after-sales policy evidence with metadata filters.",
        frozenset({"after_sales"}),
        PolicyRetrieveInput,
        policy_retrieve,
        retry_policy="safe_once",
        timeout_ms=120000,
        output_compressor=_compress_policy,
    )
)
registry.register(
    ToolSpec(
        "order_query",
        "Read the authenticated user's order history by order ID, product, or date; return one match or bounded candidates.",
        frozenset({"after_sales"}),
        OrderQueryInput,
        order_query,
        retry_policy="safe_once",
    )
)
