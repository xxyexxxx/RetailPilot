from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str
    message: str
    effective_message: str
    session: dict[str, Any]
    history_window: list[dict[str, str]]
    context_budget: dict[str, Any]
    intents: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    active_task: dict[str, Any]
    branch_outputs: Annotated[list[dict[str, Any]], add]
    trace: Annotated[list[dict[str, Any]], add]
    response: str
    products: list[dict[str, Any]]
    citations: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    grounding: dict[str, Any]
    # Request-local evidence, never serialized into public response/session memory.
    policy_answer_evidence: dict[str, dict[str, Any]]
    policy_answer_targets: list[str]
    answer_sections: list[dict[str, Any]]
    final_branch_outputs: list[dict[str, Any]]
    faithfulness: dict[str, Any]
    suggestions: list[str]
    parsed_queries: list[dict[str, Any]]
    pending: dict[str, Any]
    context: dict[str, Any]
    final: dict[str, Any]
