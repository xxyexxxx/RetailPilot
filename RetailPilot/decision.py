from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


DecisionDomain = Literal["catalog", "after_sales", "order", "chat", "unsupported"]
DecisionAction = Literal[
    "search",
    "detail",
    "compare",
    "policy",
    "status",
    "logistics",
    "direct",
    "reject",
]
DecisionTool = Literal[
    "catalog_search",
    "product_detail",
    "policy_retrieve",
    "order_query",
    "none",
]


# These tables are deliberately only route metadata.  They do not describe
# tool arguments, required fields, or business semantics.  Those decisions
# belong to the domain branch (for example, ``plan_after_sales``).
_ACTIONS_BY_DOMAIN: dict[str, frozenset[str]] = {
    "catalog": frozenset({"search", "detail", "compare"}),
    "after_sales": frozenset({"policy"}),
    "order": frozenset({"status", "logistics"}),
    "chat": frozenset({"direct"}),
    "unsupported": frozenset({"reject"}),
}

_TOOL_BY_ROUTE: dict[tuple[str, str], str] = {
    ("catalog", "search"): "catalog_search",
    ("catalog", "detail"): "product_detail",
    ("catalog", "compare"): "product_detail",
    ("after_sales", "policy"): "policy_retrieve",
    ("order", "status"): "order_query",
    ("order", "logistics"): "order_query",
    ("chat", "direct"): "none",
    ("unsupported", "reject"): "none",
}


class Decision(BaseModel):
    """Validated supervisor routing metadata for one intent task.

    A Decision is intentionally *not* a tool-call envelope.  It contains no
    arguments and no ``missing_slots`` because extracting domain parameters
    in this layer duplicates the domain planner and makes the handoff neither
    self-contained nor faithful to the rewritten query.  The domain branch
    receives the task's ``query_text`` together with session context/history
    and owns planning, clarification, and tool arguments.
    """

    model_config = ConfigDict(extra="forbid")

    domain: DecisionDomain
    action: DecisionAction
    tool: DecisionTool
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = Field(default="unknown", min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_route_contract(self) -> "Decision":
        if self.action not in _ACTIONS_BY_DOMAIN[self.domain]:
            raise ValueError(
                f"action {self.action!r} is not valid for domain {self.domain!r}"
            )
        expected_tool = _TOOL_BY_ROUTE[(self.domain, self.action)]
        if self.tool != expected_tool:
            raise ValueError(
                f"route {self.domain}/{self.action} requires tool {expected_tool!r}"
            )
        return self


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _confidence(value: Any) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


def build_decision(
    task: Mapping[str, Any],
    *,
    dialogue_frame: Mapping[str, Any] | None = None,
    message: str = "",
    context: Mapping[str, Any] | None = None,
) -> Decision:
    """Convert an intent task into supervisor-only route metadata.

    ``dialogue_frame``, ``message`` and ``context`` remain accepted as
    compatibility arguments for callers from the previous contract, but are
    intentionally ignored.  The query and all structured retrieval signals
    stay on the upstream intent task; domain agents receive that task plus
    their authorized context and decide arguments/clarifications themselves.
    """

    del dialogue_frame, message, context
    intent = _first_text(task.get("intent")).casefold()
    source = _first_text(task.get("source"), task.get("intent_source"), "unknown")
    confidence = _confidence(task.get("confidence"))

    if intent in {"search", "detail", "compare"}:
        action: DecisionAction = intent  # type: ignore[assignment]
        return Decision(
            domain="catalog",
            action=action,
            tool="catalog_search" if action == "search" else "product_detail",
            confidence=confidence,
            source=source,
        )

    if intent == "service":
        return Decision(
            domain="after_sales",
            action="policy",
            tool="policy_retrieve",
            confidence=confidence,
            source=source,
        )

    if intent == "order":
        # The order planner, not the supervisor, distinguishes status from
        # logistics/item/detail requests and extracts the order identifier.
        return Decision(
            domain="order",
            action="status",
            tool="order_query",
            confidence=confidence,
            source=source,
        )

    if intent == "chitchat":
        return Decision(
            domain="chat",
            action="direct",
            tool="none",
            confidence=confidence,
            source=source,
        )

    return Decision(
        domain="unsupported",
        action="reject",
        tool="none",
        confidence=confidence,
        source=source,
    )
