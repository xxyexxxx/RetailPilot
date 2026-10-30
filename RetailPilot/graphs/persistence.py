"""Equivalent graph nodes; dependencies resolve through the compatibility facade."""

from __future__ import annotations

from .. import agent as runtime


def persist(state: runtime.AgentState) -> dict[str, runtime.Any]:
    outputs = state.get("final_branch_outputs") or state.get("branch_outputs") or []
    context = runtime.update_customer_context(
        state.get("context") or {},
        branch_outputs=outputs,
        parsed_queries=state.get("parsed_queries") or [],
        products=state.get("products") or [],
        pending=state.get("pending") or {},
    )
    assistant_payload = {
        "products": [product["sku"] for product in state.get("products") or []],
        "citations": state.get("citations") or [],
        "suggestions": state.get("suggestions") or [],
        "claims": state.get("claims") or [],
        "grounding": state.get("grounding") or {},
    }
    runtime.persist_turn(
        state["session_id"],
        state["user_id"],
        state["message"],
        state["response"],
        context,
        assistant_payload,
        expected_version=int((state.get("session") or {}).get("version", 0)),
        pending=state.get("pending") or {},
    )
    for parsed in state.get("parsed_queries") or []:
        runtime.remember_turn(state["user_id"], state["effective_message"], parsed)

    final = {
        "session_id": state["session_id"],
        "user_id": state["user_id"],
        "reply": state["response"],
        "products": state.get("products") or [],
        "citations": state.get("citations") or [],
        "claims": state.get("claims") or [],
        "grounding": state.get("grounding") or {},
        "suggestions": state.get("suggestions") or [],
        "needs_clarification": bool(state.get("pending")),
        "clarification": state.get("pending") or None,
        "intents": [item["intent"] for item in state.get("intents") or []],
        "intent_details": state.get("intents") or [],
        "decisions": state.get("decisions") or [],
        "task_results": [
            {
                "task_id": output.get("task_id"),
                "task_text": output.get("task_text"),
                "intent": output.get("intent"),
                "branch": output.get("branch"),
                "domain": (output.get("decision") or {}).get("domain"),
                "action": (output.get("decision") or {}).get("action"),
                "tool": (output.get("decision") or {}).get("tool"),
                "response": output.get("response"),
                "product_skus": [
                    product["sku"] for product in output.get("products") or []
                ],
                "claims": output.get("claims") or [],
                "grounding": output.get("grounding") or {},
                "policy": output.get("policy"),
                "order": output.get("order"),
                "escalated": bool(output.get("escalated")),
                "tool_failure": output.get("tool_failure"),
            }
            for output in sorted(
                outputs,
                key=lambda item: item.get("task_index") or 999,
            )
        ],
        "parsed_queries": state.get("parsed_queries") or [],
        "trace": state.get("trace") or [],
        "context": context,
        "context_budget": state.get("context_budget") or {},
    }
    return {
        "context": context,
        "final": final,
        "trace": runtime._trace(
            "memory_update", parsed_queries=len(state.get("parsed_queries") or [])
        ),
    }
