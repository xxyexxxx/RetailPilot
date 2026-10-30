"""Evidence-driven orchestration around the existing read-only policy tool."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from contextvars import copy_context
from functools import partial
from queue import Empty, Queue
from threading import Thread
from typing import Any

from .config import settings
from .llm_config import PolicyCallBudget, policy_call_budget
from .policy_evidence import refusal, validate_assessment

# Expansion is relative to the planner's ORIGINAL topic, never transitive.
FOLLOWUP_TOPICS: dict[str, frozenset[str]] = {
    "退换货": frozenset({"退换货", "保修", "退款"}),
    "保修": frozenset({"保修", "退换货"}),
    "退款": frozenset({"退款", "退换货"}),
    "配送": frozenset({"配送"}),
    "发票": frozenset({"发票"}),
}


def _bounded_call(call: Callable[[], Any], deadline: float) -> Any:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("policy deadline")
    result: Queue[tuple[bool, Any]] = Queue(maxsize=1)
    context = copy_context()

    def run() -> None:
        try:
            result.put((True, context.run(call)))
        except Exception as exc:  # noqa: BLE001 - fail-closed external call boundary
            result.put((False, exc))

    Thread(target=run, daemon=True, name="policy-followup").start()
    try:
        ok, value = result.get(timeout=remaining)
    except Empty as exc:
        raise TimeoutError("policy deadline") from exc
    if not ok:
        raise value
    return value


def _identity(document: dict[str, Any]) -> tuple[str, ...]:
    # Never collapse identical words across different applicability/version.
    return tuple(
        str(document.get(key) or "")
        for key in (
            "platform",
            "category",
            "version",
            "effective_from",
            "effective_to",
            "audience",
            "topic",
        )
    ) + (re.sub(r"\s+", "", str(document.get("content") or "")),)


def evidence_size(document: dict[str, Any]) -> int:
    # Conservative byte envelope rather than a language-specific token guess.
    return (
        len(json.dumps(document, ensure_ascii=False, default=str).encode("utf-8")) + 128
    )


def select_evidence(
    pool: list[dict[str, Any]],
    previous: dict[str, Any] | None = None,
    *,
    max_chunks: int | None = None,
    token_budget: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Coverage-first, then new rounds/section diversity and retriever ranking.

    Full leaf content is indivisible. Parent context stays in the pool; selected
    snapshots use leaf text consistently for grading, generation and citations.
    No clause is truncated to pretend that it fits the budget.
    """
    max_chunks = (
        settings.policy_max_evidence_chunks if max_chunks is None else max_chunks
    )
    token_budget = (
        settings.policy_evidence_token_budget if token_budget is None else token_budget
    )
    mapping = (previous or {}).get("aspect_evidence_map") or {}
    by_id = {item["chunk_id"]: item for item in pool}
    ordered: list[str] = []
    # Round-robin over aspects prevents one long section consuming the budget.
    for ids in mapping.values():
        if ids:
            ordered.append(ids[0])
    fresh = sorted(
        pool,
        key=lambda item: (
            -int(item.get("first_seen_hop", 0)),
            -float(item.get("rerank_score") or item.get("rrf_score") or 0),
            -int(item.get("priority") or 0),
            item["chunk_id"],
        ),
    )
    sections: set[tuple[str, str]] = set()
    for item in fresh:
        section = (item.get("doc_id", ""), item.get("section_path", ""))
        if section not in sections:
            ordered.append(item["chunk_id"])
            sections.add(section)
    ordered.extend(cid for ids in mapping.values() for cid in ids)
    ordered.extend(item["chunk_id"] for item in fresh)
    selected: list[dict[str, Any]] = []
    used = 0
    for cid in dict.fromkeys(ordered):
        if cid not in by_id:
            continue
        item = dict(by_id[cid])
        item["context"] = item["content"]
        size = evidence_size(item)
        if len(selected) < max_chunks and used + size <= token_budget:
            selected.append(item)
            used += size
    selected_ids = {item["chunk_id"] for item in selected}
    return selected, [
        item["chunk_id"] for item in pool if item["chunk_id"] not in selected_ids
    ]


def run_policy_followup(
    *,
    query: str,
    topic: str,
    initial_payload: dict[str, Any],
    retrieve: Callable[[dict[str, Any]], dict[str, Any]],
    assess: Callable[..., dict[str, Any]],
    retrieval_arguments: dict[str, Any],
    started_at: float | None = None,
    enabled: bool | None = None,
) -> dict[str, Any]:
    """Bounded loop with injected grader/tool; only the tool owns applicability."""
    started = time.perf_counter() if started_at is None else started_at
    deadline = started + settings.policy_followup_timeout_seconds
    budget = PolicyCallBudget(settings.policy_followup_llm_token_budget, deadline)
    budget_token = policy_call_budget.set(budget)
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    information: set[tuple[str, ...]] = set()
    trace: list[dict[str, Any]] = []
    calls = 0
    assessments = 0
    previous: dict[str, Any] | None = None
    signatures: list[tuple[Any, ...]] = []
    selected: list[dict[str, Any]] = []
    used_queries = {(topic, re.sub(r"\s+", "", query).casefold())}
    stop = "unknown"
    verdict = refusal("not_assessed")
    active = settings.policy_followup_enabled if enabled is None else enabled
    max_hops = settings.policy_max_extra_hops if active else 0

    def merge(documents: list[dict[str, Any]], hop: int) -> tuple[int, int]:
        new_ids = 0
        added = 0
        for raw in documents:
            cid = raw.get("chunk_id")
            if (
                not cid
                or not raw.get("doc_id")
                or not raw.get("content")
                or cid in seen
            ):
                continue
            seen.add(cid)
            new_ids += 1
            identity = _identity(raw)
            if identity in information:
                continue
            information.add(identity)
            pool.append({**raw, "first_seen_hop": hop})
            added += 1
        return new_ids, added

    try:
        merge(initial_payload.get("documents") or [], 0)
        trace.append(
            {
                "node": "after_sales.policy_initial",
                "hop": 0,
                "query": query,
                "topic": topic,
                "evidence_count": len(pool),
                "retrieval_trace": initial_payload.get("trace") or {},
            }
        )
        for hop in range(max_hops + 1):
            selected, omitted = select_evidence(pool, previous)
            stage = time.perf_counter()
            try:
                assessments += 1
                raw = _bounded_call(
                    partial(
                        assess,
                        query,
                        topic=topic,
                        documents=selected,
                        product_context=retrieval_arguments.get("product_context")
                        or {},
                    ),
                    deadline,
                )
                verdict = validate_assessment(raw, selected)
            except TimeoutError:
                stop = "assessment_timeout"
                verdict = refusal(stop)
                break
            except Exception as exc:  # noqa: BLE001 - fail-closed external call boundary
                stop = "assessment_error:" + type(exc).__name__
                verdict = refusal(stop)
                break
            trace.append(
                {
                    "node": "after_sales.policy_evidence",
                    "hop": hop,
                    **verdict,
                    "selected_chunk_ids": [item["chunk_id"] for item in selected],
                    "omitted_chunk_ids": omitted,
                    "cumulative_evidence_count": len(pool),
                    "latency_ms": round((time.perf_counter() - stage) * 1000, 2),
                }
            )
            action = verdict["next_action"]
            if time.perf_counter() >= deadline:
                stop = "timeout_budget"
                verdict = refusal(stop)
                break
            if action == "answer":
                # Unseen distinct evidence could carry an exception or conflict.
                if omitted:
                    stop = "evidence_context_budget"
                    verdict = refusal(stop)
                else:
                    support = set(verdict["supporting_chunk_ids"])
                    final_snapshot = [
                        item for item in selected if item["chunk_id"] in support
                    ]
                    if len(final_snapshot) != len(selected):
                        # At most one confirmation on the exact generation set.
                        # An answer may not depend on evidence it then discards.
                        assessments += 1
                        try:
                            confirm = _bounded_call(
                                partial(
                                    assess,
                                    query,
                                    topic=topic,
                                    documents=final_snapshot,
                                    product_context=retrieval_arguments.get(
                                        "product_context"
                                    )
                                    or {},
                                ),
                                deadline,
                            )
                            confirm = validate_assessment(confirm, final_snapshot)
                        except Exception as exc:  # noqa: BLE001 - fail-closed external call boundary
                            stop = "final_selection_error:" + type(exc).__name__
                            verdict = refusal(stop)
                            break
                        trace.append(
                            {
                                "node": "after_sales.policy_evidence",
                                "hop": hop,
                                "phase": "final_selection",
                                **confirm,
                                "selected_chunk_ids": [
                                    item["chunk_id"] for item in final_snapshot
                                ],
                            }
                        )
                        if (
                            confirm["next_action"] != "answer"
                            or set(confirm["covered_aspects"])
                            != set(verdict["covered_aspects"])
                            or set(confirm["supporting_chunk_ids"]) != support
                        ):
                            stop = "final_selection_not_sufficient"
                            verdict = refusal(stop)
                            break
                        verdict = confirm
                        selected = final_snapshot
                    if time.perf_counter() >= deadline:
                        stop = "timeout_budget"
                        verdict = refusal(stop)
                    else:
                        stop = "answer"
                break
            if action in {"ask_user", "refuse"}:
                stop = (
                    "policy_conflict"
                    if verdict["conflict"]
                    else verdict["reason"]
                    if verdict["reason"].startswith("assessment_error:")
                    else action
                )
                break
            signature = (
                tuple(sorted(verdict["covered_aspects"])),
                tuple(sorted(verdict["missing_aspects"])),
            )
            if signature in signatures:
                stop = (
                    "no_semantic_progress"
                    if signatures[-1] == signature
                    else "assessment_oscillation"
                )
                verdict = refusal(stop)
                break
            signatures.append(signature)
            if hop >= max_hops:
                stop = "max_extra_hops" if active else "followup_disabled"
                verdict = refusal(stop)
                break
            suggestions = verdict["suggested_queries"] or [
                {
                    "aspect": verdict["missing_aspects"][0],
                    "topic": topic,
                    "query": f"{query}；补充核对：{verdict['missing_aspects'][0]}",
                }
            ]
            candidate = suggestions[0]  # one tool invocation per hop
            if len(candidate["query"]) > 500:
                candidate = {**candidate, "query": candidate["aspect"]}
            target_topic = candidate["topic"]
            if target_topic not in FOLLOWUP_TOPICS.get(topic, frozenset()):
                stop = "cross_topic_denied"
                verdict = refusal(stop)
                trace.append(
                    {
                        "node": "after_sales.policy_expansion_denied",
                        "topic": target_topic,
                        "reason": candidate["aspect"],
                    }
                )
                break
            actual_query = candidate["query"].strip()
            expected_query = f"{query}；补充核对：{candidate['aspect']}"
            if actual_query not in {expected_query, candidate["aspect"]}:
                stop = "invalid_followup_query"
                verdict = refusal(stop)
                break
            key = (target_topic, re.sub(r"\s+", "", actual_query).casefold())
            if key in used_queries:
                stop = "repeated_query"
                verdict = refusal(stop)
                break
            used_queries.add(key)
            # Explicit construction: grader can never inject metadata or writes.
            args: dict[str, Any] = {
                key: retrieval_arguments[key]
                for key in ("platform", "product_context", "limit", "as_of")
                if key in retrieval_arguments
            }
            args.update(query=actual_query, topic=target_topic)
            stage = time.perf_counter()
            calls += 1
            trace.append(
                {
                    "node": "after_sales.policy_followup",
                    "hop": hop + 1,
                    "query": actual_query,
                    "topic": target_topic,
                    "reason": candidate["aspect"],
                }
            )
            try:
                result = _bounded_call(partial(retrieve, args), deadline)
                if not isinstance(result, dict) or not isinstance(
                    result.get("data") or {}, dict
                ):
                    raise TypeError("invalid policy tool result")
                returned_documents = (result.get("data") or {}).get("documents") or []
                if not isinstance(returned_documents, list) or any(
                    not isinstance(item, dict) for item in returned_documents
                ):
                    raise ValueError("invalid policy tool documents")
            except TimeoutError:
                stop = "retrieval_timeout"
                verdict = refusal(stop)
                break
            except Exception as exc:  # noqa: BLE001 - fail-closed external call boundary
                stop = "retrieval_error:" + type(exc).__name__
                verdict = refusal(stop)
                break
            trace[-1].update(
                {
                    "tool_trace": result.get("trace") or {},
                    "retrieval_trace": (result.get("data") or {}).get("trace") or {},
                    "latency_ms": round((time.perf_counter() - stage) * 1000, 2),
                }
            )
            if not result.get("ok"):
                if result.get("trace"):
                    trace.append(result["trace"])
                stop = "retrieval_" + str(
                    (result.get("error") or {}).get("type") or "error"
                )
                verdict = refusal(stop)
                break
            new_ids, added = merge(
                (result.get("data") or {}).get("documents") or [], hop + 1
            )
            trace[-1].update(
                new_chunk_count=new_ids,
                new_information_count=added,
                cumulative_evidence_count=len(pool),
            )
            if result.get("trace"):
                trace.append(result["trace"])
            if not added:
                stop = "no_new_chunks" if not new_ids else "no_new_information"
                verdict = refusal(stop)
                break
            previous = verdict
        final_ids = (
            verdict["supporting_chunk_ids"]
            if verdict["next_action"] == "answer"
            else []
        )
        # Keep EXACT selected snapshot for both consumers, no hidden parent expansion.
        final = [item for item in selected if item["chunk_id"] in final_ids]
        actual_tokens = sum(event.get("total_tokens", 0) for event in budget.events)
        costs = [event.get("estimated_cost") for event in budget.events]
        summary = {
            "node": "after_sales.policy_followup_complete",
            "stop_reason": stop,
            "extra_retrieval_calls": calls,
            "assessment_calls": assessments,
            "extra_assessment_calls": max(0, assessments - 1),
            "llm_calls": len(budget.events),
            "llm_tokens": actual_tokens,
            "llm_reserved_tokens": budget.reserved_tokens,
            "estimated_cost": sum(float(value) for value in costs if value is not None)
            if costs and all(v is not None for v in costs)
            else None,
            "selected_chunk_ids": [item["chunk_id"] for item in final],
            "cumulative_evidence_count": len(pool),
            "retrieved_chunk_ids": [item["chunk_id"] for item in pool],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        trace.append(summary)
        return {
            "assessment": verdict,
            "documents": final,
            "evidence_pool": pool,
            "assessment_documents": selected,
            "trace": trace,
            "summary": summary,
        }
    finally:
        policy_call_budget.reset(budget_token)
