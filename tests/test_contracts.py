from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

from RetailPilot import api as api_module
from RetailPilot.auth import _attempts, password_hash
from RetailPilot.catalog import (
    SessionConflictError,
    SessionOwnershipError,
    get_session,
    persist_turn,
    save_session,
    session_history,
)
from RetailPilot.config import settings
from RetailPilot.database import db_session
from RetailPilot.execution import BoundedPool, call_deadline
from RetailPilot.schemas import ParsedQuery


@pytest.fixture
def protected(monkeypatch):
    monkeypatch.setattr(settings, "allow_open_api", False)
    monkeypatch.setattr(settings, "api_key", "service-secret")
    monkeypatch.setattr(settings, "api_user_id", "owner")
    monkeypatch.setattr(settings, "auth_cookie_secure", False)
    monkeypatch.setattr(
        settings,
        "auth_users_json",
        json.dumps({"owner": password_hash("test-password")}),
    )
    _attempts.clear()
    with TestClient(api_module.app) as client:
        yield client


def test_cookie_login_csrf_and_bound_identity(protected):
    client = protected
    assert client.get("/api/catalog").status_code == 401
    response = client.post(
        "/api/auth/login", json={"user_id": "owner", "password": "test-password"}
    )
    assert response.status_code == 200 and "HttpOnly" in response.headers["set-cookie"]
    assert (
        "test-password" not in response.text and "service-secret" not in response.text
    )
    csrf = response.json()["csrf_token"]
    assert client.get("/api/catalog").status_code == 200
    assert client.get("/api/profile/other").status_code == 403
    assert (
        client.post("/api/chat", json={"session_id": "x", "message": "hi"}).status_code
        == 403
    )
    assert (
        client.post(
            "/api/chat",
            headers={"X-CSRF-Token": csrf},
            json={"user_id": "other", "session_id": "x", "message": "hi"},
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/api/chat",
            headers={"X-CSRF-Token": csrf, "Origin": "https://evil.example"},
            json={"message": "hi"},
        ).status_code
        == 403
    )
    assert (
        client.post("/api/auth/logout", headers={"X-CSRF-Token": csrf}).status_code
        == 200
    )
    assert client.get("/api/catalog").status_code == 401


def test_service_identity_cannot_be_impersonated(protected):
    headers = {"Authorization": "Bearer service-secret"}
    assert protected.get("/api/profile/owner", headers=headers).status_code == 200
    assert protected.get("/api/profile/other", headers=headers).status_code == 403
    assert (
        protected.post(
            "/api/chat", headers=headers, json={"message": "hi", "user_id": "other"}
        ).status_code
        == 403
    )


def test_expired_cookie_and_login_rate_limit(protected):
    protected.post(
        "/api/auth/login", json={"user_id": "owner", "password": "test-password"}
    )
    with db_session() as connection:
        connection.execute("UPDATE web_sessions SET expires_at=0")
    assert protected.get("/api/catalog").status_code == 401
    _attempts.clear()
    for _ in range(5):
        assert (
            protected.post(
                "/api/auth/login", json={"user_id": "owner", "password": "wrong"}
            ).status_code
            == 401
        )
    assert (
        protected.post(
            "/api/auth/login", json={"user_id": "owner", "password": "wrong"}
        ).status_code
        == 429
    )


def test_session_ownership_cannot_be_reassigned():
    get_session("ownership", "owner")
    with pytest.raises(SessionOwnershipError):
        get_session("ownership", "other")
    with pytest.raises(SessionOwnershipError):
        save_session("ownership", "other", {})
    with pytest.raises(SessionOwnershipError):
        session_history("ownership", "other")
    assert get_session("ownership", "owner")["user_id"] == "owner"


def test_turn_is_atomic_and_compacted_rows_are_not_loaded():
    session = get_session("atomic", "owner")
    persist_turn(
        "atomic",
        "owner",
        "question",
        "answer",
        {},
        {},
        expected_version=session["version"],
    )
    previous = get_session("atomic", "owner")
    with db_session() as connection:
        connection.execute(
            "CREATE TRIGGER fail_assistant BEFORE INSERT ON messages WHEN NEW.session_id='atomic' AND NEW.role='assistant' BEGIN SELECT RAISE(ABORT,'fixture failure'); END"
        )
    try:
        with pytest.raises(Exception, match="fixture failure"):
            persist_turn(
                "atomic",
                "owner",
                "should roll back",
                "bad",
                {"changed": True},
                {},
                expected_version=previous["version"],
            )
    finally:
        with db_session() as connection:
            connection.execute("DROP TRIGGER fail_assistant")
    loaded = get_session("atomic", "owner")
    assert (
        len(loaded["messages"]) == 2
        and loaded["version"] == previous["version"]
        and loaded["context"] == {}
    )
    save_session(
        "atomic",
        "owner",
        {
            "history_summary_through_id": loaded["messages"][-1]["id"],
            "history_summary": {"facts": ["retained summary"]},
        },
    )
    assert get_session("atomic", "owner")["messages"] == []
    assert len(session_history("atomic", "owner")["messages"]) == 2


def test_two_turns_with_same_version_only_one_commits():
    version = get_session("concurrent", "owner")["version"]

    def commit(number):
        try:
            persist_turn(
                "concurrent",
                "owner",
                str(number),
                str(number),
                {"number": number},
                {},
                expected_version=version,
            )
            return "ok"
        except SessionConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(commit, [1, 2])) == ["conflict", "ok"]
    assert len(get_session("concurrent", "owner")["messages"]) == 2


def test_search_auto_keeps_refinement_and_manual_is_explicit(monkeypatch):
    seen = []
    monkeypatch.setattr(
        api_module,
        "parse_query",
        lambda *a, **kw: ParsedQuery(
            "query", "query", grounded_target_terms=["自动耳机"], llm_concepts=["降噪"]
        ),
    )

    def refine(parsed):
        parsed.grounded_target_terms = ["模型耳机"]
        return parsed

    monkeypatch.setattr(api_module, "maybe_refine_query", refine)

    def search(parsed, **kwargs):
        seen.append(parsed)
        return {
            "request_id": "fixture",
            "parsed_query": parsed.to_dict(),
            "products": [],
            "trace": [{"node": "fts_recall", "latency_ms": 1.5}],
        }

    monkeypatch.setattr(api_module.search_engine, "search", search)
    with TestClient(api_module.app) as client:
        assert (
            client.post(
                "/api/search", json={"query": "耳机", "mode": "fts"}
            ).status_code
            == 200
        )
        assert seen[-1].grounded_target_terms == ["模型耳机"] and seen[
            -1
        ].llm_concepts == ["降噪"]
        client.post(
            "/api/search",
            json={
                "query": "耳机",
                "query_mode": "manual",
                "target_terms": ["人工耳机"],
                "concepts": [],
            },
        )
        assert (
            seen[-1].grounded_target_terms == ["人工耳机"]
            and seen[-1].llm_concepts == []
        )
        client.post(
            "/api/search",
            json={"query": "耳机", "query_mode": "manual", "target_terms": []},
        )
        assert seen[-1].grounded_target_terms == []
        client.post(
            "/api/search",
            json={"query": "耳机", "query_mode": "auto", "target_terms": []},
        )
        assert seen[-1].grounded_target_terms == ["模型耳机"]


def test_public_trace_has_metrics_without_internal_text():
    value = api_module._redact_public_payload(
        {
            "trace": [
                {
                    "node": "policy_followup",
                    "status": "success",
                    "latency_ms": 12.5,
                    "new_chunk_count": 2,
                    "query": "secret order",
                    "reason": "secret prompt",
                    "documents": ["secret"],
                }
            ]
        }
    )
    assert "secret" not in str(value)
    summary = value["trace_summary"][0]
    assert summary["latency_ms"] == 12.5 and summary["counts"]["new_chunk_count"] == 2


def test_response_models_reject_internal_extras_and_publish_openapi():
    from RetailPilot.api_models import ChatResponse

    value = ChatResponse.model_validate(
        {"session_id": "x", "user_id": "u", "reply": "safe", "context": {"secret": 1}}
    ).model_dump()
    assert "context" not in value
    schema = api_module.app.openapi()
    assert "ChatResponse" in schema["components"]["schemas"]
    assert "SearchResponse" in schema["components"]["schemas"]


def test_real_product_uses_public_response_model():
    with db_session() as connection:
        sku = connection.execute("SELECT sku FROM products LIMIT 1").fetchone()[0]
    with TestClient(api_module.app) as client:
        response = client.get(f"/api/products/{sku}")
    assert response.status_code == 200 and response.json()["sku"] == sku
    assert "vector" not in response.json() and "provenance_json" not in response.json()


def test_history_pagination_has_no_overlap():
    get_session("paged", "owner")
    for index in range(3):
        session = get_session("paged", "owner")
        persist_turn(
            "paged",
            "owner",
            str(index),
            str(index),
            {},
            {},
            expected_version=session["version"],
        )
    latest = session_history("paged", "owner", limit=2)
    previous = session_history(
        "paged", "owner", limit=2, before_id=latest["next_before_id"]
    )
    assert len(latest["messages"]) == len(previous["messages"]) == 2
    assert {item["id"] for item in latest["messages"]}.isdisjoint(
        item["id"] for item in previous["messages"]
    )


def test_stream_conflict_has_no_unvalidated_final(monkeypatch):
    class ConflictGraph:
        def stream(self, *args, **kwargs):
            yield {"load_context": {}}
            raise SessionConflictError("private context")

    monkeypatch.setattr(api_module, "graph", ConflictGraph())
    with TestClient(api_module.app) as client:
        response = client.post(
            "/api/chat/stream", json={"session_id": "stream-conflict", "message": "hi"}
        )
    assert "event: error" in response.text and "409" in response.text
    assert (
        "private context" not in response.text and "event: final" not in response.text
    )


def test_stream_emits_progress_and_only_validated_final(monkeypatch):
    class Graph:
        def stream(self, *a, **kw):
            yield {
                "execute_tool": {
                    "response": "UNVERIFIED",
                    "trace": [{"query": "SECRET"}],
                }
            }
            yield {"verify_answer_faithfulness": {}}
            yield {
                "persist": {
                    "final": {
                        "session_id": "stream",
                        "user_id": "demo-user",
                        "reply": "verified",
                        "products": [],
                        "context": {"secret": 1},
                        "trace": [{"node": "persist"}],
                    }
                }
            }

    monkeypatch.setattr(api_module, "graph", Graph())
    with TestClient(api_module.app) as client:
        response = client.post(
            "/api/chat/stream", json={"session_id": "stream", "message": "hi"}
        )
        assert response.status_code == 200
        assert "UNVERIFIED" not in response.text and "SECRET" not in response.text
        assert response.text.index("event: progress") < response.text.index(
            "event: final"
        )
        assert "verified" in response.text and '"context"' not in response.text


def test_history_api_and_stream_enforce_owner(protected):
    get_session("someone-else", "other")
    headers = {"Authorization": "Bearer service-secret"}
    assert (
        protected.get(
            "/api/sessions/someone-else/messages", headers=headers
        ).status_code
        == 403
    )
    assert (
        protected.post(
            "/api/chat/stream",
            headers=headers,
            json={"session_id": "someone-else", "message": "hi"},
        ).status_code
        == 403
    )


def test_bounded_execution_retains_capacity_until_running_call_stops():
    pool = BoundedPool("fixture-pool", 1, 1)
    started, release = Event(), Event()

    def stuck():
        started.set()
        release.wait(1)
        return call_deadline.get()

    with pytest.raises(TimeoutError):
        pool.call(stuck, time.perf_counter() + 0.03)
    assert started.is_set()
    with pytest.raises(RuntimeError, match="capacity"):
        pool.submit(lambda: None, deadline=time.perf_counter() + 1)
    release.set()


def test_queued_cancellation_does_not_start_work():
    pool = BoundedPool("fixture-cancel", 1, 2)
    started, release = Event(), Event()
    first = pool.submit(
        lambda: (started.set(), release.wait(1)), deadline=time.perf_counter() + 2
    )
    assert started.wait(0.2)
    invoked = []
    pending = pool.submit(
        lambda: invoked.append(True), deadline=time.perf_counter() + 1
    )
    assert pending.cancel()
    release.set()
    first.result(timeout=1)
    assert not invoked
