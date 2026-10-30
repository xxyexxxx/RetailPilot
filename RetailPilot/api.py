from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from queue import Empty, Queue
from threading import Event
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from . import auth
from .agent import chat, graph
from .api_models import (
    ChatResponse,
    CompareResponse,
    LoginRequest,
    ProductResponse,
    SearchResponse,
    TraceSummary,
)
from .catalog import (
    SessionConflictError,
    SessionOwnershipError,
    add_interaction,
    catalog_stats,
    compare_products,
    get_product,
    get_user_profile,
    list_brands,
    list_categories,
    session_history,
)
from .config import settings
from .database import catalog_ready, db_session, get_metadata, init_db
from .execution import chat_pool
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


def _api_access(request: Request):
    return auth.authenticate(request)


def _summarize_trace(trace: Any) -> Any:
    if not isinstance(trace, list):
        return trace
    summary: list[dict[str, Any]] = []
    for item in trace:
        if not isinstance(item, dict):
            continue
        entry = {"node": str(item.get("node") or "unknown")[:120]}
        for key in ("tool", "status"):
            if isinstance(item.get(key), str) and (
                key == "tool"
                or item[key]
                in {
                    "success",
                    "error",
                    "passed",
                    "refused",
                    "repaired",
                    "skipped",
                    "started",
                    "completed",
                    "rebuilt_and_grounded",
                }
            ):
                entry[key] = item[key][:80]
        if isinstance(item.get("ok"), bool):
            entry["ok"] = item["ok"]
        for key in ("latency_ms", "task_index", "hop"):
            value = item.get(key)
            if isinstance(value, (int, float)) and 0 <= value < 1e12:
                entry[key] = value
        if item.get("next_action") in {"answer", "retrieve_more", "ask_user", "refuse"}:
            entry["next_action"] = item["next_action"]
        if "status" not in entry:
            entry["status"] = (
                "passed"
                if item.get("passed") is True
                else "error"
                if item.get("passed") is False
                else "completed"
            )
        entry["counts"] = {
            key: item[key]
            for key in (
                "tasks",
                "returned",
                "new_chunk_count",
                "new_information_count",
                "cumulative_evidence_count",
                "evidence_count",
                "assessment_calls",
                "extra_retrieval_calls",
            )
            if isinstance(item.get(key), int)
        }
        entry["timings_ms"] = {
            key: value
            for key, value in (item.get("timings_ms") or {}).items()
            if key
            in {
                "structured_filter",
                "fts_recall",
                "dense_recall",
                "candidate_payload_fetch",
                "candidate_features",
                "rank_and_ground",
                "rerank",
            }
            and isinstance(value, (int, float))
            and 0 <= value < 1e12
        }
        entry = TraceSummary.model_validate(entry).model_dump(exclude_none=True)
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


@app.middleware("http")
async def private_api_cache(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.get("/api/auth/session")
def auth_session(request: Request):
    try:
        principal = auth.authenticate(request)
        return {
            "mode": principal.mode,
            "user_id": principal.user_id,
            "csrf_token": principal.csrf_token,
        }
    except HTTPException as exc:
        if exc.status_code in {401, 503}:
            return {"mode": "login_required", "user_id": None, "csrf_token": ""}
        raise


@app.post("/api/auth/login")
def auth_login(body: LoginRequest, request: Request, response: Response):
    token, principal = auth.login(request, body.user_id, body.password)
    response.set_cookie(
        auth.COOKIE,
        token,
        max_age=settings.auth_session_seconds,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="strict",
    )
    response.headers["Cache-Control"] = "no-store"
    return {
        "mode": "cookie",
        "user_id": principal.user_id,
        "csrf_token": principal.csrf_token,
    }


@app.post("/api/auth/logout", dependencies=[Depends(_api_access)])
def auth_logout(request: Request, response: Response):
    token = request.cookies.get(auth.COOKIE, "")
    with db_session() as connection:
        connection.execute(
            "DELETE FROM web_sessions WHERE token_hash=?",
            (hashlib.sha256(token.encode()).hexdigest(),),
        )
    response.delete_cookie(
        auth.COOKIE,
        secure=settings.auth_cookie_secure,
        httponly=True,
        samesite="strict",
    )
    return {"ok": True}


@app.get("/api/sessions/{session_id}/messages", dependencies=[Depends(_api_access)])
def history(
    session_id: str,
    http: Request,
    user_id: str | None = None,
    before_id: int | None = None,
):
    try:
        return session_history(
            session_id, auth.request_user(http, user_id), before_id=before_id
        )
    except SessionOwnershipError as exc:
        raise HTTPException(403, "session is not available to this user") from exc


@app.post("/api/chat/stream", dependencies=[Depends(_api_access)])
async def chat_stream(body: ChatRequest, request: Request):
    user_id = auth.request_user(
        request, body.user_id if "user_id" in body.model_fields_set else None
    )
    # Check ownership before SSE headers. CAS still protects concurrent turns.
    try:
        await run_in_threadpool(session_history, body.session_id, user_id)
    except SessionOwnershipError as exc:
        raise HTTPException(403, "session is not available to this user") from exc
    events = Queue(maxsize=64)
    cancelled = Event()

    def produce():
        started = time.perf_counter()
        try:
            for update in graph.stream(
                {
                    "session_id": body.session_id,
                    "user_id": user_id,
                    "message": body.message,
                    "branch_outputs": [],
                    "trace": [],
                },
                stream_mode="updates",
            ):
                if cancelled.is_set():
                    break
                for node, value in update.items():
                    if node == "persist":
                        payload = ChatResponse.model_validate(
                            _redact_public_payload(value["final"])
                        ).model_dump(exclude_none=True)
                        events.put_nowait(("final", payload))
                    else:
                        events.put_nowait(
                            (
                                "progress",
                                {
                                    "node": node,
                                    "status": "completed",
                                    "latency_ms": round(
                                        (time.perf_counter() - started) * 1000, 2
                                    ),
                                },
                            )
                        )
        except SessionConflictError:
            events.put_nowait(
                ("error", {"detail": "session changed; reload history", "status": 409})
            )
        except Exception:
            if not cancelled.is_set():
                events.put_nowait(
                    ("error", {"detail": "chat execution failed", "status": 500})
                )
        finally:
            if not cancelled.is_set():
                events.put_nowait(("done", {}))

    try:
        stream_deadline = time.perf_counter() + 300
        future = chat_pool.submit(produce, deadline=stream_deadline)
    except RuntimeError as exc:
        raise HTTPException(503, "chat capacity exhausted; try later") from exc

    async def stream():
        last_heartbeat = time.perf_counter()
        try:
            while True:
                if time.perf_counter() >= stream_deadline:
                    yield 'event: error\ndata: {"detail":"chat deadline exceeded; restore history before retrying","status":408}\n\n'
                    break
                if await request.is_disconnected():
                    break
                try:
                    kind, payload = events.get_nowait()
                except Empty:
                    if future.done():
                        yield 'event: error\ndata: {"detail":"chat execution stopped","status":503}\n\n'
                        break
                    if time.perf_counter() - last_heartbeat >= 10:
                        yield ": keep-alive\n\n"
                        last_heartbeat = time.perf_counter()
                    await asyncio.sleep(0.05)
                    continue
                if kind == "done":
                    break
                yield (
                    "event: "
                    + kind
                    + "\ndata: "
                    + json.dumps(payload, ensure_ascii=False)
                    + "\n\n"
                )
        finally:
            cancelled.set()
            future.cancel()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health", dependencies=[Depends(_api_access)])
def health() -> dict[str, Any]:
    return {
        "status": "ok" if catalog_ready() else "uninitialized",
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


@app.post(
    "/api/search",
    response_model=SearchResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(_api_access)],
)
def search(request: SearchRequest, http: Request) -> dict[str, Any]:
    # 手动调用等价于"人当 LLM 传参"：target_terms/concepts 由调用方给出，
    # 是 FTS 召回唯一来源；query 仅作原文记录与向量召回。无 LLM 词时不做
    # 本地词表兜底（与 agent 链路一致：LLM 失效不往后走）。
    user_id = auth.request_user(
        http, request.user_id if "user_id" in request.model_fields_set else None
    )
    manual = request.query_mode == "manual" or (
        "query_mode" not in request.model_fields_set
        and (request.target_terms is not None or request.concepts is not None)
    )
    parsed = parse_query(request.query, known_brands=list_brands())
    if manual:
        parsed.grounded_target_terms = [
            term.strip() for term in request.target_terms or [] if term.strip()
        ][:12]
        parsed.llm_concepts = [
            term.strip() for term in request.concepts or [] if term.strip()
        ][:12]
        parsed.sources["query_mode"] = "api:manual"
    else:
        parsed = maybe_refine_query(parsed)
        parsed.sources["query_mode"] = "api:auto"
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
            user_id=user_id,
            limit=request.limit,
            mode=request.mode,
            session_id="search-lab",
        )
    )


@app.post(
    "/api/chat",
    response_model=ChatResponse,
    response_model_exclude_none=True,
    dependencies=[Depends(_api_access)],
)
def chat_endpoint(request: ChatRequest, http: Request) -> dict[str, Any]:
    user_id = auth.request_user(
        http, request.user_id if "user_id" in request.model_fields_set else None
    )
    try:
        return _redact_public_payload(
            chat(request.session_id, request.message, user_id)
        )
    except SessionOwnershipError as exc:
        raise HTTPException(403, "session is not available to this user") from exc
    except SessionConflictError as exc:
        raise HTTPException(
            409, "session changed; reload history before retrying"
        ) from exc


@app.get(
    "/api/products/{sku}",
    response_model=ProductResponse,
    dependencies=[Depends(_api_access)],
)
def product_detail(sku: str) -> dict[str, Any]:
    product = get_product(sku)
    if product is None:
        raise HTTPException(status_code=404, detail="product not found")
    return product


@app.post(
    "/api/products/compare",
    response_model=CompareResponse,
    dependencies=[Depends(_api_access)],
)
def product_compare(request: CompareRequest) -> dict[str, Any]:
    result = compare_products(request.skus)
    if len(result["products"]) < 2:
        raise HTTPException(
            status_code=400, detail="at least two valid products are required"
        )
    return result


@app.post("/api/interactions", dependencies=[Depends(_api_access)])
def interactions(request: InteractionRequest, http: Request) -> dict[str, Any]:
    try:
        return add_interaction(
            auth.request_user(
                http, request.user_id if "user_id" in request.model_fields_set else None
            ),
            request.sku,
            request.event_type,
            request.query,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/profile/{user_id}", dependencies=[Depends(_api_access)])
def profile(user_id: str, http: Request) -> dict[str, Any]:
    return get_user_profile(auth.request_user(http, user_id))


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
