from __future__ import annotations

import time
from threading import Event
from typing import Any

import pytest
from pydantic import BaseModel

from RetailPilot.tools import (
    OrderQueryInput,
    PolicyRetrieveInput,
    ToolRegistry,
    ToolSpec,
    registry,
)


class ExampleInput(BaseModel):
    value: int


def make_spec(
    function: Any,
    *,
    retry_policy: str = "never",
    side_effect: str = "read_only",
    timeout_ms: int = 100,
    output_compressor: Any = None,
) -> ToolSpec:
    return ToolSpec(
        name="example",
        description="Test tool.",
        scopes=frozenset({"pre_sales"}),
        input_model=ExampleInput,
        function=function,
        retry_policy=retry_policy,  # type: ignore[arg-type]
        side_effect=side_effect,  # type: ignore[arg-type]
        timeout_ms=timeout_ms,
        output_compressor=output_compressor,
    )


def execute(spec: ToolSpec, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    local_registry = ToolRegistry()
    local_registry.register(spec)
    return local_registry.execute(
        "example", "pre_sales", arguments or {"value": 1}
    )


def test_default_registry_exposes_only_four_read_only_tools() -> None:
    assert registry.list("pre_sales") == ["catalog_search", "product_detail"]
    assert registry.list("after_sales") == ["order_query", "policy_retrieve"]
    assert all(
        schema["side_effect"] == "read_only"
        for scope in ("pre_sales", "after_sales")
        for schema in registry.schemas(scope)
    )


def test_policy_retrieve_requires_llm_selected_canonical_topic() -> None:
    missing = registry.execute(
        "policy_retrieve",
        "after_sales",
        {"query": "耳机坏了怎么处理"},
    )
    alias = registry.execute(
        "policy_retrieve",
        "after_sales",
        {"query": "耳机能退吗", "topic": "退货"},
    )

    assert missing["error"]["type"] == "invalid_arguments"
    assert alias["error"]["type"] == "invalid_arguments"
    assert missing["trace"]["attempts"] == 0
    assert alias["trace"]["attempts"] == 0


def test_policy_retrieve_accepts_authoritative_product_context_not_sku_list() -> None:
    parsed = PolicyRetrieveInput(
        query="这双鞋能退吗",
        topic="退换货",
        product_context={
            "sku": "sku-shoe",
            "category": "Apparel",
            "purchased_at": "2026-08-01",
        },
    )

    assert "product_skus" not in PolicyRetrieveInput.model_fields
    assert parsed.product_context is not None
    assert parsed.product_context.category == "Apparel"


def test_order_lookup_accepts_product_and_date_filters():
    parsed = OrderQueryInput(
        user_id="user-1",
        product_query="鞋",
        created_after="2026-01-01",
        created_before="2026-12-31",
        limit=5,
    )

    assert parsed.product_query == "鞋"
    assert parsed.created_after == "2026-01-01"
    assert parsed.limit == 5


def test_execute_validates_compresses_and_records_success_trace() -> None:
    result = execute(
        make_spec(
            lambda value: {"value": value, "hidden": "large"},
            output_compressor=lambda data: {"value": data["value"]},
        )
    )

    assert result["ok"] is True
    assert result["data"] == {"value": 1, "hidden": "large"}
    assert result["observation"] == {"value": 1}
    assert result["trace"] | {
        "attempts": 1,
        "timeout_ms": 100,
        "timed_out": False,
        "error": None,
    } == result["trace"]


def test_validation_and_scope_failures_never_invoke_tool() -> None:
    calls = 0

    def tool(value: int) -> dict[str, int]:
        nonlocal calls
        calls += 1
        return {"value": value}

    spec = make_spec(tool, retry_policy="safe_once")
    local_registry = ToolRegistry()
    local_registry.register(spec)

    invalid = local_registry.execute("example", "pre_sales", {"value": "bad"})
    denied = local_registry.execute("example", "after_sales", {"value": 1})

    assert calls == 0
    assert invalid["error"]["type"] == "invalid_arguments"
    assert invalid["trace"]["attempts"] == 0
    assert invalid["trace"]["timeout_ms"] == 100
    assert denied["error"]["type"] == "scope_denied"
    assert denied["trace"]["attempts"] == 0


def test_safe_once_read_only_tool_retries_one_exception() -> None:
    calls = 0

    def flaky(value: int) -> dict[str, int]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary failure")
        return {"value": value}

    result = execute(make_spec(flaky, retry_policy="safe_once"))

    assert result["ok"] is True
    assert calls == 2
    assert result["trace"]["attempts"] == 2
    assert result["trace"]["error"] is None


@pytest.mark.parametrize(
    ("retry_policy", "side_effect"),
    [("never", "read_only"), ("safe_once", "write")],
)
def test_retry_requires_both_safe_once_and_read_only(
    retry_policy: str, side_effect: str
) -> None:
    calls = 0

    def broken(value: int) -> dict[str, int]:
        nonlocal calls
        calls += 1
        raise ValueError(f"failure for {value}")

    result = execute(
        make_spec(
            broken,
            retry_policy=retry_policy,
            side_effect=side_effect,
        )
    )

    assert result["ok"] is False
    assert calls == 1
    assert result["error"] == {
        "type": "tool_error",
        "message": "failure for 1",
    }
    assert result["trace"]["attempts"] == 1
    assert result["trace"]["error"] == result["error"]


def test_safe_retry_stops_after_second_failure() -> None:
    calls = 0

    def broken(value: int) -> dict[str, int]:
        nonlocal calls
        calls += 1
        raise RuntimeError(f"failure {calls} for {value}")

    result = execute(make_spec(broken, retry_policy="safe_once"))

    assert result["ok"] is False
    assert calls == 2
    assert result["trace"]["attempts"] == 2
    assert result["error"]["type"] == "tool_error"
    assert result["error"]["message"] == "failure 2 for 1"


def test_timeout_is_enforced_and_safe_read_is_retried_once() -> None:
    release = Event()
    calls = 0

    def blocked(value: int) -> dict[str, int]:
        nonlocal calls
        calls += 1
        release.wait(1)
        return {"value": value}

    started = time.perf_counter()
    result = execute(
        make_spec(blocked, retry_policy="safe_once", timeout_ms=20)
    )
    elapsed = time.perf_counter() - started
    release.set()

    assert elapsed < 0.25
    assert calls == 2
    assert result["ok"] is False
    assert result["error"]["type"] == "timeout"
    assert result["trace"]["attempts"] == 2
    assert result["trace"]["timeout_ms"] == 20
    assert result["trace"]["timed_out"] is True
    assert result["trace"]["error"] == result["error"]
