from __future__ import annotations

import hmac
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .agent import chat
from .catalog import (
    add_interaction,
    catalog_stats,
    compare_products,
    get_product,
    get_user_profile,
    list_brands,
    list_categories,
)
from .config import settings
from .database import catalog_ready, get_metadata, init_db
from .llm import maybe_refine_query
from .query import parse_query
from .retrieval import search_engine
from .schemas import (
    ChatRequest,
    CompareRequest,
    InteractionRequest,
    SearchRequest,
)
from .text import tokenize


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    tokenize("搜索推荐服务预热")
    yield


def _api_access(request: Request) -> None:
    """Require a configured API key for every /api endpoint by default.

    Local single-user demos may set RETAILPILOT_ALLOW_OPEN_API=true, but the
    safe clone default refuses anonymous access instead of silently serving
    sessions, traces and user profiles.
    """

    if settings.allow_open_api:
        return
    expected = settings.api_key
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "API access is disabled because RETAILPILOT_API_KEY is not set. "
                "Set it and send `Authorization: Bearer <key>` (or "
                "X-API-Key: <key>), or explicitly set "
                "RETAILPILOT_ALLOW_OPEN_API=true for a local demo."
            ),
        )
    supplied = request.headers.get("x-api-key") or ""
    authorization = request.headers.get("authorization") or ""
    if authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def _summarize_trace(trace: Any) -> Any:
    if not isinstance(trace, list):
        return trace
    summary: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        entry: dict[str, Any] = {}
        for key in ("node", "tool", "ok", "error_type", "task_index", "tasks"):
            if key in item:
                entry[key] = item[key]
        summary.append(entry)
    return summary


def _redact_public_payload(payload: Any) -> Any:
    """Remove server-side context and full traces from public API responses."""

    value = deepcopy(payload)
    if isinstance(value, dict):
        for key in (
            "context",
            "context_budget",
            "history_window",
            "session",
            "memory",
            "retrieval_trace",  # full internal query/metadata traces are server-side
            "evidence_assessment",
            "policy_answer_evidence",
            "policy_answer_targets",
            "answer_sections",
            "final_branch_outputs",
            "faithfulness",
        ):
            value.pop(key, None)
        if "trace" in value:
            value["trace_summary"] = _summarize_trace(value.pop("trace"))
        for key, nested in list(value.items()):
            if key not in {"trace_summary"}:
                value[key] = _redact_public_payload(nested)
    elif isinstance(value, list):
        return [_redact_public_payload(item) for item in value]
    return value


app = FastAPI(
    title="RetailPilot",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/api/health", dependencies=[Depends(_api_access)])
def health() -> dict[str, Any]:
    return {
        "status": "ok" if catalog_ready() else "uninitialized",
        "database": str(settings.database_path),
        "stats": catalog_stats(),
    }


@app.get("/api/catalog", dependencies=[Depends(_api_access)])
def catalog() -> dict[str, Any]:
    return {
        "stats": catalog_stats(),
        "categories": list_categories(),
        "brands": list_brands(limit=80),
        "dataset": get_metadata("dataset", {}),
    }


@app.post("/api/search", dependencies=[Depends(_api_access)])
def search(request: SearchRequest) -> dict[str, Any]:
    # 手动调用等价于"人当 LLM 传参"：target_terms/concepts 由调用方给出，
    # 是 FTS 召回唯一来源；query 仅作原文记录与向量召回。无 LLM 词时不做
    # 本地词表兜底（与 agent 链路一致：LLM 失效不往后走）。
    parsed = parse_query(request.query, known_brands=list_brands())
    parsed = maybe_refine_query(parsed)
    parsed.grounded_target_terms = [t for t in request.target_terms if t][:12]
    parsed.llm_concepts = [c for c in request.concepts if c][:12]
    if parsed.grounded_target_terms:
        parsed.sources["grounded_target_terms"] = "api:params"
    if parsed.llm_concepts:
        parsed.sources["concepts"] = "api:params"
    if request.category:
        parsed.category = request.category
        parsed.sources["category"] = "api:filter"
    if request.price_min is not None:
        parsed.price_min = request.price_min
        parsed.sources["price_min"] = "api:filter"
    if request.price_max is not None:
        parsed.price_max = request.price_max
        parsed.sources["price_max"] = "api:filter"
    if request.brands:
        parsed.brands = request.brands
        parsed.sources["brands"] = "api:filter"
    return _redact_public_payload(
        search_engine.search(
            parsed,
            user_id=request.user_id,
            limit=request.limit,
            mode=request.mode,
            session_id="search-lab",
        )
    )


@app.post("/api/chat", dependencies=[Depends(_api_access)])
def chat_endpoint(request: ChatRequest) -> dict[str, Any]:
    return _redact_public_payload(
        chat(request.session_id, request.message, request.user_id)
    )


@app.get("/api/products/{sku}", dependencies=[Depends(_api_access)])
def product_detail(sku: str) -> dict[str, Any]:
    product = get_product(sku)
    if product is None:
        raise HTTPException(status_code=404, detail="product not found")
    return product


@app.post("/api/products/compare", dependencies=[Depends(_api_access)])
def product_compare(request: CompareRequest) -> dict[str, Any]:
    result = compare_products(request.skus)
    if len(result["products"]) < 2:
        raise HTTPException(status_code=400, detail="at least two valid products are required")
    return result


@app.post("/api/interactions", dependencies=[Depends(_api_access)])
def interactions(request: InteractionRequest) -> dict[str, Any]:
    try:
        return add_interaction(
            request.user_id,
            request.sku,
            request.event_type,
            request.query,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/profile/{user_id}", dependencies=[Depends(_api_access)])
def profile(user_id: str) -> dict[str, Any]:
    return get_user_profile(user_id)


@app.get("/")
def web_index() -> FileResponse:
    return FileResponse(settings.web_dir / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


app.mount(
    "/assets",
    StaticFiles(directory=settings.web_dir / "assets"),
    name="assets",
)
