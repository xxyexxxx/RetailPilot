from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
from RetailPilot.config import settings
from RetailPilot.llm_config import (
    PolicyCallBudget,
    invoke_with_telemetry,
    policy_call_budget,
)
from RetailPilot.policy_evidence import refusal, validate_assessment
from RetailPilot.policy_followup import run_policy_followup, select_evidence

from RetailPilot import agent, llm

QUERY = "定制商品超过七天出现质量问题还能退货吗？"


def doc(cid, content=None, **kw):
    return {
        "chunk_id": cid,
        "doc_id": "d-" + cid,
        "content": content or f"{cid}政策条款。",
        "context": "完整父章节",
        "source": "fixture",
        "source_uri": "memory://policy",
        "title": "退换货政策",
        "section_path": cid,
        "version": "2026",
        "effective_from": "2026-01-01",
        "effective_to": "",
        "topic": "退换货",
        "platform": "demo-mall",
        "category": "*",
        "audience": "public",
        **kw,
    }


def grade(action, *, covered=None, missing=None, topic="退换货", query=None):
    mapping = covered or {}
    return {
        "decision": {
            "answer": "answer",
            "retrieve_more": "clarify",
            "ask_user": "clarify",
            "refuse": "refuse",
        }[action],
        "next_action": action,
        "sufficient": action == "answer",
        "conflict": False,
        "reason": "fixture semantic verdict",
        "covered_aspects": list(mapping),
        "missing_aspects": missing or [],
        "aspect_evidence_map": mapping,
        "supporting_chunk_ids": list(
            dict.fromkeys(cid for ids in mapping.values() for cid in ids)
        ),
        "required_user_fields": ["order_id"] if action == "ask_user" else [],
        "clarification_question": "请提供要核验的订单号。"
        if action == "ask_user"
        else None,
        "suggested_queries": [
            {
                "aspect": missing[0],
                "query": query or f"{QUERY}；补充核对：{missing[0]}",
                "topic": topic,
            }
        ]
        if action == "retrieve_more" and missing
        else [],
    }


def run(grades, batches=(), initial=None, **kw):
    assessments, queries = [], []
    grades = iter(grades)
    batches = iter(batches)

    def assess(*args, **kwargs):
        assessments.append(kwargs["documents"])
        value = next(grades)
        if isinstance(value, Exception):
            raise value
        return value

    def retrieve(args):
        queries.append(args)
        value = next(batches)
        if isinstance(value, Exception):
            raise value
        if isinstance(value, dict):
            return value
        return {
            "ok": True,
            "data": {"documents": value},
            "trace": {"node": "tool_execution"},
        }

    result = run_policy_followup(
        query=QUERY,
        topic="退换货",
        initial_payload={"documents": initial or [doc("c1")]},
        retrieve=retrieve,
        assess=assess,
        retrieval_arguments={
            "platform": "demo-mall",
            "product_context": {"sku": "SKU1"},
            "limit": 6,
        },
        **kw,
    )
    return result, assessments, queries


def test_first_evidence_sufficient_no_followup():
    result, assessments, queries = run([grade("answer", covered={"完整规则": ["c1"]})])
    assert result["summary"]["stop_reason"] == "answer"
    assert len(assessments) == 1 and queries == []
    assert result["documents"][0]["context"] == result["documents"][0]["content"]


@pytest.mark.parametrize("hops", [1, 2])
def test_accumulates_missing_conditions_for_one_or_two_hops(hops):
    grades = [
        grade(
            "retrieve_more", covered={"一般限制": ["c1"]}, missing=["质量例外", "期限"]
        )
    ]
    if hops == 2:
        grades.append(
            grade(
                "retrieve_more",
                covered={"一般限制": ["c1"], "质量例外": ["c2"]},
                missing=["期限"],
            )
        )
    final_map = {"一般限制": ["c1"], "质量例外": ["c2"]}
    if hops == 2:
        final_map["期限"] = ["c3"]
    grades.append(grade("answer", covered=final_map))
    result, assessments, queries = run(
        grades, [[doc(f"c{i + 2}")] for i in range(hops)]
    )
    assert len(queries) == hops and len(assessments) == hops + 1
    assert {item["chunk_id"] for item in result["documents"]} == {
        cid for ids in final_map.values() for cid in ids
    }
    assert result["evidence_pool"][-1]["first_seen_hop"] == hops
    assert all(item["product_context"] == {"sku": "SKU1"} for item in queries)


def test_budget_exhausted_fails_closed(monkeypatch):
    monkeypatch.setattr(settings, "policy_max_extra_hops", 1)
    result, _, queries = run(
        [
            grade("retrieve_more", missing=["质量例外", "期限"]),
            grade("retrieve_more", covered={"质量例外": ["c2"]}, missing=["期限"]),
        ],
        [[doc("c2")]],
    )
    assert result["assessment"]["next_action"] == "refuse"
    assert result["summary"]["stop_reason"] == "max_extra_hops" and len(queries) == 1


@pytest.mark.parametrize(
    "verdict,stop",
    [
        (grade("ask_user"), "ask_user"),
        (refusal("conflicting rules", conflict=True), "policy_conflict"),
    ],
)
def test_personal_facts_and_conflict_do_not_retrieve(verdict, stop):
    result, _, queries = run([verdict])
    assert result["summary"]["stop_reason"] == stop and not queries


def test_query_repetition_stops():
    # A targeted query identical to the original is never executed.
    result, _, queries = run([grade("retrieve_more", missing=[QUERY], query=QUERY)])
    assert result["summary"]["stop_reason"] == "repeated_query" and not queries


@pytest.mark.parametrize(
    "batch,stop",
    [
        ([], "no_new_chunks"),
        ([doc("c1")], "no_new_chunks"),
        ([doc("c2", "c1政策条款。")], "no_new_information"),
    ],
)
def test_no_new_valid_information_stops(batch, stop):
    result, _, _ = run([grade("retrieve_more", missing=["质量例外"])], [batch])
    assert result["summary"]["stop_reason"] == stop


def test_duplicate_chunk_keeps_original_metadata():
    result, _, _ = run(
        [
            grade("retrieve_more", missing=["质量例外"]),
            grade("answer", covered={"限制": ["c1"], "例外": ["c2"]}),
        ],
        [[doc("c1"), doc("c2")]],
    )
    assert len(result["evidence_pool"]) == 2
    assert result["evidence_pool"][0]["first_seen_hop"] == 0
    assert result["evidence_pool"][0]["context"] == "完整父章节"


@pytest.mark.parametrize(
    "patch",
    [
        {"sufficient": False},
        {"conflict": True},
        {"next_action": "guess"},
        {"unknown": 1},
        {
            "supporting_chunk_ids": ["invented"],
            "aspect_evidence_map": {"限制": ["invented"]},
        },
        {"aspect_evidence_map": {"限制": ["invented"]}},
        {"missing_aspects": ["例外"]},
        {"sufficient": "true"},
    ],
)
def test_invalid_or_invented_verdicts_fail_closed(patch):
    value = grade("answer", covered={"限制": ["c1"]}) | patch
    with pytest.raises(ValueError):
        validate_assessment(value, [doc("c1")])
    result, _, queries = run([value])
    assert result["assessment"]["next_action"] == "refuse" and not queries


@pytest.mark.parametrize(
    "response", ["not json", '{"decision":"answer"}', "```json\n{}\n```"]
)
def test_llm_invalid_json_is_never_answer(monkeypatch, response):
    monkeypatch.setattr(llm, "_invoke_json", lambda *a, **kw: response)
    result = llm.assess_policy_evidence(QUERY, topic="退换货", documents=[doc("c1")])
    assert result["next_action"] == "refuse"


def test_grader_unavailable_never_uses_retrieval_score():
    result, _, queries = run(
        [RuntimeError("offline")], initial=[doc("c1", rerank_score=100)]
    )
    assert result["assessment"]["next_action"] == "refuse" and not queries


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("tool"),
        RuntimeError("bm25 offline"),
        {"ok": False, "error": {"type": "timeout"}},
        {"ok": False, "error": {"type": "tool_error"}},
    ],
)
def test_retrieval_errors_terminal(failure):
    result, _, queries = run([grade("retrieve_more", missing=["质量例外"])], [failure])
    assert result["assessment"]["next_action"] == "refuse" and len(queries) == 1
    assert result["summary"]["stop_reason"].startswith("retrieval_")


@pytest.mark.parametrize("topic,allowed", [("保修", True), ("发票", False)])
def test_cross_topic_allowlist(topic, allowed):
    result, _, queries = run(
        [
            grade("retrieve_more", missing=["质量例外"], topic=topic),
            grade("answer", covered={"限制": ["c1"], "质量": ["c2"]}),
        ],
        [[doc("c2", topic=topic)]],
    )
    assert bool(queries) == allowed
    assert result["summary"]["stop_reason"] == (
        "answer" if allowed else "cross_topic_denied"
    )


def test_unconstrained_query_cannot_invent_business_facts():
    result, _, queries = run(
        [grade("retrieve_more", missing=["质量例外"], query="我的手机订单123已获退款")]
    )
    assert result["summary"]["stop_reason"] == "invalid_followup_query" and not queries


def test_context_budget_cannot_authorize_hidden_evidence(monkeypatch):
    monkeypatch.setattr(settings, "policy_max_evidence_chunks", 1)
    result, _, _ = run(
        [grade("answer", covered={"限制": ["c1"]})], initial=[doc("c1"), doc("c2")]
    )
    assert result["summary"]["stop_reason"] == "evidence_context_budget"


def test_selector_retains_late_exception_and_aspect_diversity():
    pool = [doc(f"c{i}", first_seen_hop=0) for i in range(8)] + [
        doc("exception", first_seen_hop=1)
    ]
    selected, _ = select_evidence(
        pool,
        grade("retrieve_more", covered={"限制": ["c0"]}, missing=["例外"]),
        max_chunks=2,
    )
    assert {item["chunk_id"] for item in selected} == {"c0", "exception"}


def test_semantic_no_progress_and_oscillation():
    a = grade("retrieve_more", missing=["A条款"])
    b = grade("retrieve_more", missing=["B条款"])
    result, _, queries = run([a, a], [[doc("c2")]])
    assert (
        result["summary"]["stop_reason"] == "no_semantic_progress" and len(queries) == 1
    )
    result, _, queries = run([a, b, a], [[doc("c2")], [doc("c3")]])
    assert (
        result["summary"]["stop_reason"] == "assessment_oscillation"
        and len(queries) == 2
    )


def test_disabled_is_single_pass():
    result, _, queries = run(
        [grade("retrieve_more", missing=["质量例外"])], enabled=False
    )
    assert result["summary"]["stop_reason"] == "followup_disabled" and not queries


def test_grader_wall_deadline(monkeypatch):
    monkeypatch.setattr(settings, "policy_followup_timeout_seconds", 0.02)

    def slow(*a, **kw):
        time.sleep(0.2)
        return grade("answer", covered={"限制": ["c1"]})

    start = time.perf_counter()
    result = run_policy_followup(
        query=QUERY,
        topic="退换货",
        initial_payload={"documents": [doc("c1")]},
        retrieval_arguments={},
        assess=slow,
        retrieve=lambda a: pytest.fail("unexpected retrieve"),
    )
    assert time.perf_counter() - start < 0.18
    assert result["summary"]["stop_reason"] == "assessment_timeout"


def test_llm_token_reservation_blocks_before_network():
    budget = PolicyCallBudget(10, time.perf_counter() + 1)
    token = policy_call_budget.set(budget)
    try:
        with pytest.raises(RuntimeError, match="token_budget"):
            invoke_with_telemetry(
                SimpleNamespace(invoke=lambda a: pytest.fail("network")),
                "long input",
                prompt_kind="policy_evidence",
            )
    finally:
        policy_call_budget.reset(token)


def setup_branch(monkeypatch, grades, batches):
    monkeypatch.setattr(
        agent,
        "plan_after_sales",
        lambda *a, **kw: {
            "route": "policy_question",
            "topic": "退换货",
            "standalone_query": QUERY,
            "steps": [{"step_id": "policy", "tool": "policy_retrieve", "args": {}}],
        },
    )
    batches = iter(batches)
    grades = iter(grades)
    calls = []

    def execute(name, scope, arguments, **kw):
        calls.append((name, arguments))
        return {
            "ok": True,
            "data": {"topic": arguments["topic"], "documents": next(batches)},
            "trace": {"node": "tool_execution", "tool": name},
        }

    monkeypatch.setattr(agent.registry, "execute", execute)
    monkeypatch.setattr(agent, "assess_policy_evidence", lambda *a, **kw: next(grades))
    return calls


def branch_state():
    return {
        "active_task": {
            "task_id": "task-1",
            "task_index": 1,
            "intent": "service",
            "text": QUERY,
            "decision": {"domain": "after_sales"},
        },
        "user_id": "demo-user",
        "context": {},
        "history_window": [],
    }


@pytest.mark.parametrize("cross", [False, True])
def test_branch_multicondition_evidence_reaches_reply_citations_and_claims(
    monkeypatch, cross
):
    first = doc("c1", "定制商品不适用七天无理由退货。")
    second = doc(
        "c2",
        "超过七天出现质量问题可申请核验，是否退货须按检测结果审核。",
        topic="保修" if cross else "退换货",
    )
    calls = setup_branch(
        monkeypatch,
        [
            grade(
                "retrieve_more",
                covered={"定制限制": ["c1"]},
                missing=["质量例外"],
                topic=second["topic"],
            ),
            grade("answer", covered={"定制限制": ["c1"], "质量例外": ["c2"]}),
        ],
        [[first], [second]],
    )
    captured = {}

    def compose(kind, **kwargs):
        captured.update(kwargs["data"])
        # deterministic fallback must include both conditions

    monkeypatch.setattr(agent, "compose_reply", compose)
    result = agent.after_sales_branch(branch_state())
    payload = result["branch_outputs"][0]
    assert len(calls) == 2 and all(name == "policy_retrieve" for name, _ in calls)
    assert {item["chunk_id"] for item in captured["documents"]} == {"c1", "c2"}
    assert (
        first["content"] in payload["response"]
        and second["content"] in payload["response"]
    )
    assert "保证退货成功" not in payload["response"]
    assert {item["chunk_id"] for item in payload["citations"]} == {"c1", "c2"}
    assert {item["citation_id"] for item in payload["claims"]} == {
        item["citation_id"] for item in payload["citations"]
    }
    assert payload["grounding"]["pass"] is True
    assert payload["policy"]["adaptive_summary"]["stop_reason"] == "answer"


@pytest.mark.parametrize("action", ["answer", "ask_user", "refuse"])
def test_branch_simple_clarification_and_refusal(monkeypatch, action):
    verdict = grade(action, covered={"规则": ["c1"]} if action == "answer" else {})
    calls = setup_branch(monkeypatch, [verdict], [[doc("c1")]])
    payload = agent.after_sales_branch(branch_state())["branch_outputs"][0]
    assert len(calls) == 1
    assert bool(payload["citations"]) == (action == "answer")
    if action == "ask_user":
        assert payload["required_slots"] == ["order_id"]


def test_non_policy_path_never_grades(monkeypatch):
    monkeypatch.setattr(
        agent,
        "plan_after_sales",
        lambda *a, **kw: {"route": "human_handoff", "steps": []},
    )
    monkeypatch.setattr(
        agent, "run_policy_followup", lambda **kw: pytest.fail("non-policy retrieval")
    )
    assert (
        agent.after_sales_branch(branch_state())["branch_outputs"][0]["policy"] is None
    )


@pytest.mark.parametrize(
    "text", ["保证退货成功", '{"statements":[{"chunk_id":"missing","text":"虚构"}]}']
)
def test_composer_cannot_strengthen_or_reference_hidden_rules(monkeypatch, text):
    monkeypatch.setattr(settings, "llm_mode", "llm")
    monkeypatch.setattr(llm, "get_model", lambda **kw: object())
    monkeypatch.setattr(
        llm, "invoke_with_telemetry", lambda *a, **kw: SimpleNamespace(content=text)
    )
    assert (
        llm.compose_reply("policy", user_text=QUERY, data={"documents": [doc("c1")]})
        is None
    )


def test_policy_prompt_uses_real_ids_and_whole_selected_collection(monkeypatch):
    documents = [doc(f"c{i}") for i in range(9)]

    def fake(prompt, **kw):
        assert "c8" in prompt[1].content  # no historical [:6] truncation
        return json.dumps(grade("answer", covered={"条件": ["c8"]}), ensure_ascii=False)

    monkeypatch.setattr(llm, "_invoke_json", fake)
    assert (
        llm.assess_policy_evidence(QUERY, topic="退换货", documents=documents)[
            "next_action"
        ]
        == "answer"
    )


def test_final_generation_is_not_limited_to_two_chunks(monkeypatch):
    setup_branch(
        monkeypatch,
        [
            grade(
                "retrieve_more",
                covered={"定制": ["c1"], "期限": ["c2"]},
                missing=["质量例外"],
            ),
            grade(
                "answer", covered={"定制": ["c1"], "期限": ["c2"], "质量例外": ["c3"]}
            ),
        ],
        [[doc("c1"), doc("c2")], [doc("c3")]],
    )
    output = agent.after_sales_branch(branch_state())["branch_outputs"][0]
    assert len(output["citations"]) == 3
    assert len(output["claims"]) == 3
    assert all(f"c{i}政策条款。" in output["response"] for i in range(1, 4))


def test_final_support_subset_is_reassessed_before_generation():
    verdict = grade("answer", covered={"完整规则": ["c1"]})
    result, assessments, _ = run([verdict, verdict], initial=[doc("c1"), doc("noise")])
    assert len(assessments) == 2
    assert assessments[-1] == result["documents"]
    assert [item["chunk_id"] for item in assessments[-1]] == ["c1"]


def test_final_support_subset_loses_condition_fails_closed():
    result, _, _ = run(
        [
            grade("answer", covered={"完整规则": ["c1"]}),
            grade("retrieve_more", missing=["质量例外"]),
        ],
        initial=[doc("c1"), doc("noise")],
    )
    assert result["summary"]["stop_reason"] == "final_selection_not_sufficient"
    assert result["documents"] == []


def test_public_api_removes_internal_policy_details():
    from RetailPilot.api import _redact_public_payload

    result = _redact_public_payload(
        {
            "task_results": [
                {
                    "policy": {
                        "retrieval_trace": {"query": "secret"},
                        "evidence_assessment": {"reason": "secret"},
                        "adaptive_summary": {"stop_reason": "answer"},
                    }
                }
            ]
        }
    )
    assert "secret" not in str(result)
    assert (
        result["task_results"][0]["policy"]["adaptive_summary"]["stop_reason"]
        == "answer"
    )


def test_evaluation_sidecar_does_not_fabricate_semantic_success():
    from RetailPilot.evaluation import evaluate_policy_runs

    result, _, _ = run([grade("answer", covered={"完整规则": ["c1"]})])
    report = evaluate_policy_runs([{"trace": result["trace"]}])
    assert report["semantic_verified"] is False
    assert report["answer_correctness_faithfulness_judge"] is None
    assert report["hard_filter_violation_rate"] is None
    assert report["average_estimated_cost"] is None


@pytest.mark.parametrize(
    "wrong", [{"chunk_id": "invented"}, {"doc_id": "wrong-document"}]
)
def test_grounding_requires_chunk_and_document_to_match(wrong):
    from RetailPilot.grounding import claim, validate_claim_citations

    document = doc("c1")
    citation = {
        "chunk_id": "c1",
        "doc_id": document["doc_id"],
        "source": "fixture",
        "field": "content",
        "value": document["content"],
        "quoted_text": document["content"],
    } | wrong
    verdict = validate_claim_citations(
        [claim(document["content"], citation)], [citation], policy_documents=[document]
    )
    assert verdict["pass"] is False


def test_release_gate_requires_semantic_and_filter_measurements():
    from RetailPilot.evaluation import policy_quality_gate

    assert (
        policy_quality_gate(
            {},
            {
                "hard_filter_violation_rate_max": 0,
                "independent_semantic_judge_required": True,
            },
        )["pass"]
        is False
    )


def test_frozen_date_and_product_scope_are_not_mutated_by_suggestion():
    from datetime import date

    seen = []
    verdicts = iter(
        [
            grade("retrieve_more", missing=["质量例外"], topic="保修"),
            grade("answer", covered={"限制": ["c1"], "质量": ["c2"]}),
        ]
    )

    def retrieve(args):
        seen.append(args)
        return {"ok": True, "data": {"documents": [doc("c2", topic="保修")]}}

    result = run_policy_followup(
        query=QUERY,
        topic="退换货",
        initial_payload={"documents": [doc("c1")]},
        assess=lambda *a, **kw: next(verdicts),
        retrieve=retrieve,
        retrieval_arguments={
            "platform": "demo-mall",
            "as_of": date(2026, 8, 23),
            "product_context": {"category": "家居家装"},
        },
    )
    assert result["summary"]["stop_reason"] == "answer"
    assert seen[0]["as_of"] == date(2026, 8, 23)
    assert seen[0]["product_context"] == {"category": "家居家装"}


def test_valid_composer_uses_all_reviewed_chunks(monkeypatch):
    documents = [doc(f"c{i}") for i in range(3)]
    monkeypatch.setattr(settings, "llm_mode", "llm")
    monkeypatch.setattr(llm, "get_model", lambda **kw: object())
    output = json.dumps(
        {
            "statements": [
                {"chunk_id": item["chunk_id"], "text": item["content"]}
                for item in documents
            ]
        }
    )
    monkeypatch.setattr(
        llm, "invoke_with_telemetry", lambda *a, **kw: SimpleNamespace(content=output)
    )
    reply = llm.compose_reply("policy", user_text=QUERY, data={"documents": documents})
    assert reply is not None and all(item["content"] in reply for item in documents)


def test_controller_uses_existing_telemetry_and_preflight_token_budget(monkeypatch):
    calls = []

    def invoke(prompt):
        calls.append(prompt)
        return SimpleNamespace(
            content=json.dumps(grade("answer", covered={"完整条件": ["c1"]})),
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 23,
                "total_tokens": 123,
            },
        )

    monkeypatch.setattr(llm, "get_model", lambda **kw: SimpleNamespace(invoke=invoke))
    arguments = {
        "query": QUERY,
        "topic": "退换货",
        "initial_payload": {"documents": [doc("c1")]},
        "assess": llm.assess_policy_evidence,
        "retrieve": lambda a: pytest.fail("unexpected retrieval"),
        "retrieval_arguments": {},
    }
    result = run_policy_followup(**arguments)
    assert result["summary"]["stop_reason"] == "answer"
    assert result["summary"]["llm_calls"] == 1
    assert result["summary"]["llm_tokens"] == 123
    assert result["summary"]["llm_reserved_tokens"] > 123
    monkeypatch.setattr(settings, "policy_followup_llm_token_budget", 1)
    result = run_policy_followup(**arguments)
    assert result["summary"]["stop_reason"] == "assessment_error: PolicyBudgetExceeded"
    assert len(calls) == 1  # the rejected reservation never invokes the model


def test_tool_deadline_bounds_retries():
    from RetailPilot.tools import ToolRegistry, ToolSpec
    from pydantic import BaseModel

    class Input(BaseModel):
        pass

    calls = []

    def slow():
        calls.append(True)
        time.sleep(0.15)
        return {}

    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            "slow",
            "fixture",
            frozenset({"after_sales"}),
            Input,
            slow,
            timeout_ms=1000,
            retry_policy="safe_once",
        )
    )
    start = time.perf_counter()
    result = registry.execute("slow", "after_sales", {}, deadline=start + 0.02)
    assert not result["ok"] and result["error"]["type"] == "timeout"
    assert time.perf_counter() - start < 0.12
    assert len(calls) == 1
