"""Explicit public transport boundary; internal Agent fields are never extras."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class TraceSummary(BaseModel):
    node: str
    tool: str | None = None
    status: str | None = None
    ok: bool | None = None
    latency_ms: float | None = None
    task_index: int | None = None
    hop: int | None = None
    next_action: Literal["answer", "retrieve_more", "ask_user", "refuse"] | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ProductResponse(BaseModel):
    sku: str
    title: str
    price: float
    currency: str = "USD"
    brand: str = ""
    category: str = ""
    category_label: str = ""
    category_path: str = ""
    description: str = ""
    image_url: str = ""
    rating: float = 0
    rating_count: int = 0
    stock: int = 0
    features: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    variants: dict[str, Any] = Field(default_factory=dict)
    services: list[Any] = Field(default_factory=list)
    platform: str = ""
    seller_name: str = ""
    source: str = ""
    source_url: str = ""
    reason: str = ""
    score: float = 0
    score_breakdown: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)


class TaskResult(BaseModel):
    task_id: str | None = None
    task_text: str | None = None
    intent: str | None = None
    branch: str | None = None
    domain: str | None = None
    action: str | None = None
    tool: str | None = None
    response: str | None = None
    product_skus: list[str] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    grounding: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] | None = None
    order: dict[str, Any] | None = None
    escalated: bool = False
    tool_failure: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    reply: str
    products: list[ProductResponse] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    claims: list[dict[str, Any]] = Field(default_factory=list)
    grounding: dict[str, Any] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification: dict[str, Any] | None = None
    intents: list[str] = Field(default_factory=list)
    intent_details: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    task_results: list[TaskResult] = Field(default_factory=list)
    parsed_queries: list[dict[str, Any]] = Field(default_factory=list)
    trace_summary: list[TraceSummary] = Field(default_factory=list)


class SearchResponse(BaseModel):
    request_id: str
    parsed_query: dict[str, Any]
    products: list[ProductResponse]
    total_candidates: int = 0
    latency_ms: float = 0
    mode: str = "hybrid"
    relaxations: list[dict[str, Any]] = Field(default_factory=list)
    retrieval: dict[str, Any] = Field(default_factory=dict)
    judge: dict[str, Any] = Field(default_factory=dict)
    trace_summary: list[TraceSummary] = Field(default_factory=list)


class CompareResponse(BaseModel):
    products: list[ProductResponse]
    fields: list[dict[str, Any]] = Field(default_factory=list)


class LoginRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=256)
