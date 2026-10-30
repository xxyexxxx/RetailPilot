from __future__ import annotations

import hashlib
import json
import time
import uuid
from copy import deepcopy
from types import SimpleNamespace

import pytest
from RetailPilot.config import settings
from RetailPilot.database import db_session, json_loads
from RetailPilot.grounding import (
    attach_citation_ids,
    claim,
    policy_citation,
    validate_claim_citations,
)

from RetailPilot import agent, llm
from RetailPilot import answer_faithfulness as faith

QUERY = "定制商品超过七天出现质量问题还能退货吗？"
POLICY = "超过七天出现质量问题可申请核验，是否退货须按检测结果审核。"
BAD = "超过七天出现质量问题保证退货成功。"


def document(cid="c1", content=POLICY):
    return {
        "chunk_id": cid,
        "doc_id": "doc-" + cid,
        "content": content,
        "context": content,
        "title": "质量售后政策",
        "source": "test-policy",
        "source_uri": "memory://policy",
        "section_path": "质量问题",
        "version": "2026",
        "effective_from": "2026-01-01",
        "effective_to": "",
    }


def answer_state(text=POLICY, *, mixed=False, documents=None):
    documents = documents or [document()]
    citations = attach_citation_ids([policy_citation(item) for item in documents])
    claims = [claim(text, citation, claim_type="policy_rule") for citation in citations]
    grounding = validate_claim_citations(claims, citations, policy_documents=documents)
    output = {
        "task_id": "task-1",
        "task_index": 1,
        "task_text": QUERY,
        "branch": "after_sales",
        "intent": "service",
        "response": text,
        "claims": claims,
        "citations": citations,
        "grounding": grounding,
        "policy": {"topic": "退换货", "evidence_assessment": {"next_action": "answer"}},
        "decision": {
            "domain": "after_sales",
            "action": "policy",
            "tool": "policy_retrieve",
        },
    }
    outputs = [output]
    if mixed:
        citation = {
            "sku": "product-1",
            "field": "price",
            "value": 99,
            "source": "catalog",
        }
        outputs.append(
            {
                "task_id": "task-2",
                "task_index": 2,
                "task_text": "推荐耳机",
                "branch": "pre_sales",
                "intent": "search",
                "response": "推荐商品价格99元。",
                "claims": [claim("价格99元", citation)],
                "citations": attach_citation_ids([citation]),
                "grounding": {"pass": True, "claims": 1, "issues": []},
                "products": [{"sku": "product-1", "price": 99}],
                "decision": {"domain": "catalog"},
            }
        )
    state = {
        "message": QUERY,
        "branch_outputs": outputs,
        "history_window": [],
        "policy_answer_evidence": {
            "task-1": {
                "question": QUERY,
                "documents": documents,
                "tool_results": {"product_context": {"sku": "SKU1"}},
            }
        },
    }
    state.update(agent.build_response(state))
    return state


def verdict(supported, spans=None):
    return {
        "supported": supported,
        "hallucinated_spans": spans or [],
        "rationale": "全部条件由原文支持"
        if supported
        else "答案将核验资格错误扩大成保证退货，或遗漏必要例外",
    }


def repair_payload(text=POLICY, ids=None, task_id="task-1"):
    return {
        "replacements": [
            {
                "task_id": task_id,
                "statements": [{"text": text, "chunk_ids": ids or ["c1"]}],
            }
        ]
    }


def fake_llm(monkeypatch, results):
    calls = []
    iterator = iter(results)

    def invoke(prompt, **kwargs):
        calls.append(
            {
                "payload": json.loads(prompt[1].content),
                "kind": kwargs["prompt_kind"],
                "version": kwargs["prompt_version"],
            }
        )
        value = next(iterator)
        if isinstance(value, Exception):
            raise value
        return (
            value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        )

    monkeypatch.setattr(faith, "_invoke_json", invoke)
    return calls


def test_supported_answer_is_preserved_and_final_visible_text_is_checked(monkeypatch):
    state = answer_state(mixed=True)
    # A composition wrapper is part of the visible answer and must be checked.
    state["response"] = "本次政策核对如下：\n" + state["response"]
    calls = fake_llm(monkeypatch, [verdict(True)])
    result = faith.verify_answer_faithfulness(state)
    assert result["faithfulness"]["status"] == "passed"
    assert result["faithfulness"]["repair_attempted"] is False
    assert calls[0]["payload"]["final_visible_answer"] == state["response"]
    assert (
        calls[0]["payload"]["final_visible_answer"]
        != state["branch_outputs"][0]["response"]
    )
    assert (
        "response" not in result
        and "claims" not in result
        and "citations" not in result
    )
    assert len(calls) == 1


def test_hallucination_is_repaired_once_then_rechecked_and_regrounded(monkeypatch):
    state = answer_state(BAD)
    assert state["grounding"][
        "pass"
    ]  # lexical Grounding alone can miss this semantic error
    old_claims = deepcopy(state["claims"])
    calls = fake_llm(
        monkeypatch, [verdict(False, ["保证退货成功"]), repair_payload(), verdict(True)]
    )
    result = faith.verify_answer_faithfulness(state)
    assert result["response"] == POLICY
    assert result["faithfulness"]["status"] == "repaired"
    assert result["grounding"]["pass"] and result["claims"] != old_claims
    assert all(item["claim"] in result["response"] for item in result["claims"])
    assert {item["citation_id"] for item in result["claims"]} == {
        item["citation_id"] for item in result["citations"]
    }
    assert result["final_branch_outputs"][0]["response"] == result["response"]
    assert [call["kind"] for call in calls] == [
        "faithfulness_check",
        "faithfulness_repair",
        "faithfulness_check",
    ]
    assert calls[1]["payload"]["critique"]["hallucinated_spans"] == ["保证退货成功"]
    assert calls[2]["payload"]["final_visible_answer"] == POLICY
    assert (
        result["faithfulness"]["answer_sha256"]
        == hashlib.sha256(POLICY.encode()).hexdigest()
    )
    assert "保证退货成功" not in str(result["final_branch_outputs"])


def test_mixed_response_repair_preserves_catalog_section_and_citations(monkeypatch):
    state = answer_state(BAD, mixed=True)
    catalog = deepcopy(state["branch_outputs"][1])
    fake_llm(monkeypatch, [verdict(False), repair_payload(), verdict(True)])
    result = faith.verify_answer_faithfulness(state)
    assert result["final_branch_outputs"][1] == catalog
    assert "【推荐耳机】\n推荐商品价格99元。" in result["response"]
    assert catalog["citations"][0] in result["citations"]


@pytest.mark.parametrize(
    "last", [verdict(False), TimeoutError("checker"), "invalid JSON"]
)
def test_failed_recheck_refuses_and_removes_unverified_claims(monkeypatch, last):
    state = answer_state(BAD)
    calls = fake_llm(monkeypatch, [verdict(False), repair_payload(), last])
    result = faith.verify_answer_faithfulness(state)
    assert result["response"] == faith.SAFE_POLICY_REPLY
    assert result["claims"] == [] and result["citations"] == []
    assert result["faithfulness"]["status"] == "refused"
    assert result["faithfulness"]["check_count"] == 2
    assert result["faithfulness"]["repair_attempted"] is True
    assert len(calls) == 3
    assert result["final_branch_outputs"][0]["response"] == faith.SAFE_POLICY_REPLY


@pytest.mark.parametrize(
    "bad",
    [
        "not JSON",
        "```json\n{}\n```",
        {"supported": True},
        {"supported": "true", "hallucinated_spans": [], "rationale": "invalid boolean"},
        verdict(True, ["出现质量问题"]),
        verdict(False, ["最终答案中不存在的片段"]),
        verdict(True) | {"extra": 1},
        RuntimeError("offline"),
        TimeoutError("deadline"),
    ],
)
def test_invalid_or_unavailable_checker_fails_closed_without_repair(monkeypatch, bad):
    calls = fake_llm(monkeypatch, [bad])
    result = faith.verify_answer_faithfulness(answer_state())
    assert result["response"] == faith.SAFE_POLICY_REPLY and result["citations"] == []
    assert result["faithfulness"]["repair_attempted"] is False
    assert result["faithfulness"]["reason"].startswith("checker_unavailable:")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "bad",
    [
        "invalid JSON",
        RuntimeError("repair offline"),
        repair_payload(ids=["invented"]),
        repair_payload(task_id="task-2"),
        repair_payload(text="火星天气晴朗。"),
        {"replacements": [repair_payload()["replacements"][0]] * 2},
    ],
)
def test_invalid_repair_never_retains_old_citations(monkeypatch, bad):
    calls = fake_llm(monkeypatch, [verdict(False), bad])
    result = faith.verify_answer_faithfulness(answer_state(BAD))
    assert result["response"] == faith.SAFE_POLICY_REPLY
    assert result["claims"] == result["citations"] == []
    assert len(calls) == 2


def test_repair_must_not_drop_policy_exception(monkeypatch):
    documents = [document(), document("c2", "定制商品不适用七天无理由退货。")]
    state = answer_state(POLICY + documents[1]["content"], documents=documents)
    fake_llm(monkeypatch, [verdict(False), repair_payload()])
    result = faith.verify_answer_faithfulness(state)
    assert result["response"] == faith.SAFE_POLICY_REPLY
    assert not result["citations"]


def test_lexical_overlap_cannot_override_negative_semantic_verdict(monkeypatch):
    calls = fake_llm(monkeypatch, [verdict(False), repair_payload(BAD), verdict(False)])
    result = faith.verify_answer_faithfulness(answer_state(BAD))
    assert len(calls) == 3  # structurally valid but semantically false rewrite
    assert result["response"] == faith.SAFE_POLICY_REPLY


@pytest.mark.parametrize(
    "kind", ["chitchat", "order", "search", "clarification", "refusal"]
)
def test_non_policy_clarification_and_existing_refusal_are_skipped(monkeypatch, kind):
    state = answer_state()
    output = state["branch_outputs"][0]
    if kind in {"chitchat", "order", "search"}:
        output.update(branch=kind, policy=None)
    elif kind == "clarification":
        output["required_slots"] = ["order_id"]
    else:
        output["policy"]["evidence_assessment"]["next_action"] = "refuse"
    state.update(agent.build_response(state))
    fake_llm(monkeypatch, [])
    result = faith.verify_answer_faithfulness(state)
    assert result["trace"][0]["status"] == "skipped" and "response" not in result


def test_missing_internal_evidence_does_not_authorize_answer(monkeypatch):
    state = answer_state()
    state["policy_answer_evidence"] = {}
    fake_llm(monkeypatch, [])
    result = faith.verify_answer_faithfulness(state)
    assert result["response"] == faith.SAFE_POLICY_REPLY
    assert result["faithfulness"]["reason"] == "missing_answer_evidence"


def test_failed_original_grounding_cannot_be_bypassed(monkeypatch):
    state = answer_state()
    state["branch_outputs"][0]["citations"][0]["chunk_id"] = "invented"
    fake_llm(monkeypatch, [verdict(True)])
    result = faith.verify_answer_faithfulness(state)
    assert result["response"] == faith.SAFE_POLICY_REPLY
    assert result["faithfulness"]["reason"] == "original_policy_grounding_failed"


def test_prior_grounding_refusal_is_preserved_and_nested_policy_answer_removed(
    monkeypatch,
):
    state = answer_state()
    state["grounding"] = {
        "pass": False,
        "claims": 1,
        "issues": [{"type": "citation_source_mismatch"}],
    }
    state.update(agent.verify_grounding(state))
    fake_llm(monkeypatch, [])
    result = faith.verify_answer_faithfulness(state)
    assert "response" not in result
    assert result["final_branch_outputs"][0]["response"] == faith.SAFE_POLICY_REPLY


def test_repetition_middleware_clarification_does_not_expose_old_policy_answer(
    monkeypatch,
):
    state = answer_state()
    state["history_window"] = [
        {"role": "user", "content": QUERY},
        {"role": "assistant", "content": state["response"]},
    ]
    state.update(agent.build_response(state))
    fake_llm(monkeypatch, [])
    assert state["policy_answer_targets"] == []
    assert state["final_branch_outputs"][0]["claims"] == []
    assert faith.verify_answer_faithfulness(state)["trace"][0]["status"] == "skipped"


def test_checker_has_a_wall_clock_deadline(monkeypatch):
    monkeypatch.setattr(settings, "policy_faithfulness_timeout_seconds", 0.02)

    def slow(payload):
        time.sleep(0.2)
        return faith.FaithfulnessResult(**verdict(True))

    monkeypatch.setattr(faith, "check_answer_faithfulness", slow)
    started = time.perf_counter()
    result = faith.verify_answer_faithfulness(answer_state())
    assert time.perf_counter() - started < 0.18
    assert result["faithfulness"]["reason"] == "checker_unavailable:TimeoutError"


def test_existing_llm_telemetry_and_token_budget_are_used(monkeypatch):
    calls = []

    def invoke(prompt):
        calls.append(prompt)
        return SimpleNamespace(
            content=json.dumps(verdict(True)),
            usage_metadata={
                "input_tokens": 101,
                "output_tokens": 25,
                "total_tokens": 126,
            },
        )

    monkeypatch.setattr(llm, "get_model", lambda **kw: SimpleNamespace(invoke=invoke))
    result = faith.verify_answer_faithfulness(answer_state())
    assert result["faithfulness"]["llm_calls"] == 1
    assert result["faithfulness"]["llm_tokens"] == 126
    assert result["faithfulness"]["llm_reserved_tokens"] > 126
    monkeypatch.setattr(settings, "policy_faithfulness_llm_token_budget", 1)
    rejected = faith.verify_answer_faithfulness(answer_state())
    assert (
        rejected["faithfulness"]["reason"] == "checker_unavailable:PolicyBudgetExceeded"
    )
    assert len(calls) == 1


def install_policy_graph_fakes(monkeypatch):
    monkeypatch.setattr(
        agent,
        "analyze_intents",
        lambda *a, **kw: [
            {
                "intent": "service",
                "text": QUERY,
                "query_text": QUERY,
                "confidence": 0.99,
                "source": "test",
            }
        ],
    )
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
    monkeypatch.setattr(
        agent.registry,
        "execute",
        lambda *a, **kw: {
            "ok": True,
            "data": {"documents": [document()], "needs_product_context": False},
            "trace": {"node": "tool_execution", "tool": "policy_retrieve"},
        },
    )
    monkeypatch.setattr(
        agent,
        "assess_policy_evidence",
        lambda *a, **kw: {
            "decision": "answer",
            "next_action": "answer",
            "sufficient": True,
            "conflict": False,
            "reason": "fixed evidence fixture",
            "covered_aspects": ["条件与核验"],
            "supporting_chunk_ids": ["c1"],
            "aspect_evidence_map": {"条件与核验": ["c1"]},
        },
    )
    # Inject a semantically false generator output despite structurally valid source claims.
    monkeypatch.setattr(agent, "compose_reply", lambda *a, **kw: BAD)


@pytest.mark.parametrize("success", [True, False])
def test_real_graph_repairs_or_refuses_before_persist_and_public_return(
    monkeypatch, success
):
    install_policy_graph_fakes(monkeypatch)
    calls = fake_llm(
        monkeypatch,
        [verdict(False, ["保证退货成功"]), repair_payload(), verdict(success)],
    )
    session_id = "faithfulness-" + uuid.uuid4().hex
    result = agent.chat(session_id, QUERY)
    expected = POLICY if success else faith.SAFE_POLICY_REPLY
    assert result["reply"] == expected
    assert result["task_results"][0]["response"] == expected
    assert "保证退货成功" not in str(result["claims"])
    assert result["grounding"]["pass"] is True
    assert len(calls) == 3
    assert "policy_answer_evidence" not in result
    with db_session() as connection:
        message = connection.execute(
            "SELECT content, payload_json FROM messages WHERE session_id = ? AND role = 'assistant'",
            (session_id,),
        ).fetchone()
        context = connection.execute(
            "SELECT context_json FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
    assert message["content"] == expected
    assert "保证退货成功" not in message["payload_json"]
    assert "documents" not in context and "policy_answer_evidence" not in context
    assert bool(json_loads(message["payload_json"], {})["citations"]) == success


def test_public_api_does_not_expose_internal_faithfulness_payload():
    from RetailPilot.api import _redact_public_payload

    result = _redact_public_payload(
        {
            "policy_answer_evidence": {"secret": "policy"},
            "answer_sections": [{"response": BAD}],
            "faithfulness": {"rationale": "secret"},
            "trace": [{"node": "verify_answer_faithfulness", "rationale": "secret"}],
        }
    )
    assert "secret" not in str(result) and BAD not in str(result)
