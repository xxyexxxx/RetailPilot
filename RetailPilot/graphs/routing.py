"""Equivalent graph nodes; dependencies resolve through the compatibility facade."""

from __future__ import annotations

from .. import agent as runtime


def load_context(state: runtime.AgentState) -> dict[str, runtime.Any]:
    session = runtime.get_session(state["session_id"], state["user_id"])
    context = runtime.normalize_customer_context(session.get("context") or {})
    history_window, budget = runtime.bounded_history(
        session.get("messages") or [],
        context=context,
        current_message=state.get("message") or "",
    )
    return {
        "session": session,
        "context": context,
        "history_window": history_window,
        "context_budget": budget,
        "effective_message": state["message"],
        "trace": runtime._trace(
            "memory_inject",
            **budget,
        ),
    }


def understand(state: runtime.AgentState) -> dict[str, runtime.Any]:
    current_message = str(state["message"])
    # Intent detection is LLM-only.  The heuristic markers (_current_turn_intent,
    # _contextual_business_intent, ...) are deliberately *not* consulted here:
    # routing by keyword tables is what previously swallowed business turns like
    # "配送员电话" / "但是查不到" into chitchat and starved the LLM of work.
    # If analyze_intents returns empty (LLM unavailable / no valid tasks), the
    # empty intent list flows through to build_response which replies with a
    # default message instead of falling back to rule-based intent detection.
    intents = runtime.analyze_intents(
        state["effective_message"],
        context=state.get("context") or {},
        history=state.get("history_window") or [],
    )
    followup_kind = runtime._business_followup_kind(current_message)
    if intents and all(item["intent"] == "chitchat" for item in intents):
        intents = [
            {**intents[0], "text": current_message, "query_text": current_message}
        ]
    intents = [
        {
            **item,
            "dialogue_frame": runtime._dialogue_frame(
                current_message,
                str(item["intent"]),
                followup_kind=followup_kind,
            ),
        }
        for item in intents
    ]
    return {
        "intents": intents,
        "trace": runtime._trace(
            "intent_analyzer",
            tasks=[
                {
                    "intent": item["intent"],
                    "text": item["text"],
                    "source": item.get("source", "unknown"),
                    "confidence": item.get("confidence"),
                    "dialogue_frame": item.get("dialogue_frame") or {},
                }
                for item in intents
            ],
        ),
    }


def route_and_extract(state: runtime.AgentState) -> dict[str, runtime.Any]:
    """Produce one small, inspectable route label for every user objective."""

    routed = runtime.understand(state)
    intents = routed.get("intents") or []
    decisions = [runtime.build_decision(item).model_dump() for item in intents]
    return {
        "intents": intents,
        "decisions": decisions,
        "trace": [
            *(routed.get("trace") or []),
            {
                "node": "route_and_extract",
                "decisions": [
                    {
                        "domain": item["domain"],
                        "action": item["action"],
                        "tool": item["tool"],
                    }
                    for item in decisions
                ],
            },
        ],
    }


def execute_tool(state: runtime.AgentState) -> dict[str, runtime.Any]:
    """Dispatch rewritten intent tasks to their domain branches.

    The supervisor only selects a branch.  Domain planners own arguments,
    slot checks, and tool sequencing; this node does not gate execution based
    on a Decision-side tool/slot envelope.
    """

    tasks: list[dict[str, runtime.Any]] = []
    for task_index, (intent, decision) in enumerate(
        zip(state.get("intents") or [], state.get("decisions") or []), start=1
    ):
        domain = str(decision["domain"])
        branch = (
            "pre_sales"
            if domain == "catalog"
            else ("after_sales" if domain in {"after_sales", "order"} else "chitchat")
        )
        tasks.append(
            {
                "task_id": f"task-{task_index}",
                "task_index": task_index,
                "intent": intent["intent"],
                "intent_source": intent.get("source", "unknown"),
                "confidence": intent.get("confidence"),
                "dialogue_frame": dict(intent.get("dialogue_frame") or {}),
                "branch": branch,
                "text": intent.get("query_text")
                or intent.get("text")
                or state["message"],
                "decision": dict(decision),
                # Retrieval signals emitted by the intent call (category,
                # target_terms, structured constraints) flow through to the
                # branch so pre_sales can use them as authoritative signals.
                "category": intent.get("category"),
                "target_terms": intent.get("target_terms") or [],
                "constraints": intent.get("constraints") or {},
            }
        )

    payload = {
        "session_id": state["session_id"],
        "user_id": state["user_id"],
        "message": state["message"],
        "effective_message": state["effective_message"],
        "session": state["session"],
        "context": state["context"],
        "history_window": state.get("history_window") or [],
        "context_budget": state.get("context_budget") or {},
    }
    outputs: list[dict[str, runtime.Any]] = []
    traces: list[dict[str, runtime.Any]] = []
    policy_answer_evidence: dict[str, dict[str, runtime.Any]] = {}
    executors = {
        "pre_sales": runtime.pre_sales_branch,
        "after_sales": runtime.after_sales_branch,
        "chitchat": runtime.chitchat_branch,
    }
    for task in tasks:
        outcome = executors[task["branch"]]({**payload, "active_task": task})
        outputs.extend(outcome.get("branch_outputs") or [])
        traces.extend(outcome.get("trace") or [])
        policy_answer_evidence.update(outcome.get("policy_answer_evidence") or {})
    executed_tools = [
        str(item.get("tool"))
        for item in traces
        if item.get("node") == "tool_execution" and item.get("tool")
    ]
    return {
        "tasks": tasks,
        "branch_outputs": outputs,
        "policy_answer_evidence": policy_answer_evidence,
        "trace": [
            {
                "node": "execute_tool",
                "tasks": len(tasks),
                "routes": [
                    {
                        "domain": decision["domain"],
                        "action": decision["action"],
                        "tool": decision["tool"],
                        "query_text": intent.get("query_text")
                        or intent.get("text")
                        or state["message"],
                    }
                    for intent, decision in zip(
                        state.get("intents") or [], state.get("decisions") or []
                    )
                ],
                "executed_tools": executed_tools,
            },
            *traces,
        ],
    }


def compact_context(state: runtime.AgentState) -> dict[str, runtime.Any]:
    """Compact the oldest session messages when the harness reaches 80%.

    This is a maintenance node.  It runs before intent routing, never replans
    a business task, and never deletes the raw transcript from SQLite.
    """

    budget = dict(state.get("context_budget") or {})
    if not budget.get("needs_compaction"):
        return {"trace": runtime._trace("context_compaction", skipped=True)}

    session = state.get("session") or {}
    messages = list(session.get("messages") or [])
    context = runtime.normalize_customer_context(state.get("context") or {})
    through_id = int(context.get("history_summary_through_id") or 0)
    candidates = []
    for index, item in enumerate(messages):
        message_id = int(item.get("id") or 0)
        if through_id and (
            (message_id and message_id <= through_id)
            or (not message_id and index < through_id)
        ):
            continue
        if (
            str(item.get("role") or "") in {"user", "assistant"}
            and str(item.get("content") or "").strip()
        ):
            candidates.append(item)

    if not candidates:
        return {
            "context_budget": {**budget, "needs_compaction": False},
            "trace": runtime._trace(
                "context_compaction",
                skipped=True,
                reason="no_old_messages",
            ),
        }
    # Keep the most recent exchange context raw.  The number is a recency
    # guard, not a post-compaction occupancy target.  If the first compaction
    # still leaves the harness above 80%, summarize additional old exchanges
    # in the same maintenance node until the ratio clears or no old messages
    # remain.
    remaining = list(candidates)
    keep_raw = min(8, max(0, len(remaining) - 1))
    summary = context.get("history_summary") or {}
    compacted_count = 0
    compacted_budget = dict(budget)
    while remaining:
        split = max(1, len(remaining) - keep_raw)
        old_messages = remaining[:split]
        summary = runtime.summarize_history(
            old_messages,
            existing_summary=summary,
            context=context,
        )
        compacted_count += len(old_messages)
        last_old_id = int(old_messages[-1].get("id") or 0)
        if last_old_id:
            context["history_summary_through_id"] = last_old_id
        context["history_summary"] = summary

        history_window, compacted_budget = runtime.bounded_history(
            messages,
            context=context,
            current_message=state.get("message") or "",
        )
        if not compacted_budget.get("needs_compaction"):
            break
        remaining = remaining[split:]
        if not remaining:
            break
        keep_raw = max(0, min(keep_raw - 2, len(remaining) - 1))

    compacted_budget.update(
        {
            "compaction_applied": True,
            "compacted_messages": compacted_count,
            "compaction_source": "structured_summary_llm",
        }
    )
    # Persist immediately so a process crash after compaction cannot lose the
    # summary anchor.  The normal persist node writes the final turn state too.
    if "version" not in (state.get("session") or {}):
        # Compatibility for direct maintenance adapters; live graph writes the
        # summary atomically with this turn, under its loaded version.
        runtime.save_session(state["session_id"], state["user_id"], context)
    return {
        "context": context,
        "history_window": history_window,
        "context_budget": compacted_budget,
        "trace": runtime._trace(
            "context_compaction",
            skipped=False,
            compacted_messages=compacted_count,
            context_usage_ratio=compacted_budget.get("context_usage_ratio"),
            summary_fields=sorted(summary),
            compaction_exhausted=bool(compacted_budget.get("needs_compaction")),
        ),
    }
