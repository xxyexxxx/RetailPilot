from __future__ import annotations

import pytest
from pydantic import ValidationError

from RetailPilot.decision import Decision, build_decision


def test_builds_catalog_search_route_without_tool_arguments() -> None:
    decision = build_decision(
        {"intent": "search", "query_text": "推荐静音机械键盘", "source": "llm"}
    )

    assert decision.model_dump() == {
        "domain": "catalog",
        "action": "search",
        "tool": "catalog_search",
        "confidence": 1.0,
        "source": "llm",
    }
    assert "arguments" not in decision.model_dump()
    assert "missing_slots" not in decision.model_dump()


@pytest.mark.parametrize(
    ("intent", "domain", "action", "tool"),
    [
        ("detail", "catalog", "detail", "product_detail"),
        ("compare", "catalog", "compare", "product_detail"),
        ("service", "after_sales", "policy", "policy_retrieve"),
        ("order", "order", "status", "order_query"),
        ("chitchat", "chat", "direct", "none"),
    ],
)
def test_intent_only_selects_supervisor_route(
    intent: str, domain: str, action: str, tool: str
) -> None:
    decision = build_decision(
        {
            "intent": intent,
            "query_text": "这个 query 由域内规划器消费",
            "arguments": {"order_id": "SHOULD_NOT_CROSS_BOUNDARY"},
        },
        dialogue_frame={"identifier": "ALSO_IGNORED", "required_slots": ["x"]},
        context={
            "references": {"focused_skus": ["sku-7"], "order_id": "JD20260801"},
            "pending_slots": ["stale_slot"],
        },
    )

    assert (decision.domain, decision.action, decision.tool) == (
        domain,
        action,
        tool,
    )
    assert decision.model_dump().keys() == {
        "domain",
        "action",
        "tool",
        "confidence",
        "source",
    }


def test_unknown_intent_routes_to_unsupported_without_request_arguments() -> None:
    decision = build_decision({"intent": "payment", "text": "帮我支付"})

    assert (decision.domain, decision.action, decision.tool) == (
        "unsupported",
        "reject",
        "none",
    )
    assert decision.model_dump().keys() == {
        "domain",
        "action",
        "tool",
        "confidence",
        "source",
    }


def test_route_confidence_is_preserved_and_clamped() -> None:
    assert build_decision({"intent": "service", "confidence": 0.86}).confidence == 0.86
    assert build_decision({"intent": "service", "confidence": 4}).confidence == 1.0
    assert build_decision({"intent": "service", "confidence": -1}).confidence == 0.0


def test_schema_rejects_tool_domain_mismatch() -> None:
    with pytest.raises(ValidationError, match="requires tool 'catalog_search'"):
        Decision(
            domain="catalog",
            action="search",
            tool="order_query",
        )


def test_schema_rejects_removed_tool_argument_fields() -> None:
    with pytest.raises(ValidationError):
        Decision(
            domain="after_sales",
            action="policy",
            tool="policy_retrieve",
            arguments={"topic": "refund"},
        )
