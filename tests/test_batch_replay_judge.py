from __future__ import annotations

import json

import pytest

from RetailPilot.customer_eval import validate_customer_batch_judge_output
from scripts.batch_replay_judge import apply_verdicts, build_judge_cases


def _record():
    return {
        "conversation_id": "c1",
        "turns": [
            {
                "turn_index": 1,
                "message": "查订单 [ORDERID_1]",
                "reply": "没有查到对应订单，请核对订单号。",
                "intents": ["order"],
                "task_results": [
                    {
                        "intent": "order",
                        "branch": "after_sales",
                        "order": None,
                        "policy": None,
                        "escalated": False,
                    }
                ],
                "trace": [
                    {
                        "node": "tool_execution",
                        "tool": "order_query",
                        "status": "success",
                    }
                ],
                "grounding": {"pass": True},
                "source_context_tail": [],
                "scope_gate": {"in_scope": True, "domain": "order_read"},
                "semantic_judge": {"judge_mode": "not_selected", "pass": None},
                "evaluation_evidence": {},
            }
        ],
    }


def test_batch_judge_cases_include_tool_and_outcome_evidence():
    cases = build_judge_cases([_record()])
    assert len(cases) == 1
    evidence = cases[0]["evidence"]
    assert evidence["tool_evidence"][0]["tool"] == "order_query"
    assert evidence["task_outcomes"][0]["order_found"] is False


def test_batch_judge_requires_exact_case_ids_and_applies_verdict():
    records = [_record()]
    cases = build_judge_cases(records)
    case_id = cases[0]["case_id"]
    payload = {
        "items": [
            {
                "case_id": case_id,
                "evaluable": True,
                "pass": True,
                "intent_fit": 2,
                "requirement_coverage": 2,
                "context_use": 2,
                "relevance": 2,
                "concise_and_non_repetitive": 2,
                "failure_type": "none",
                "summary": "正确处理查无订单边界",
            }
        ]
    }
    items = validate_customer_batch_judge_output(
        json.dumps(payload, ensure_ascii=False), {case_id}
    )
    apply_verdicts(records, {case_id: items[0]})
    semantic = records[0]["turns"][0]["semantic_judge"]
    assert semantic["judge_mode"] == "llm"
    assert semantic["batch_judge"] is True
    assert semantic["pass"] is True

    payload["items"][0]["case_id"] = "unexpected"
    with pytest.raises(ValueError, match="case_id mismatch"):
        validate_customer_batch_judge_output(
            json.dumps(payload, ensure_ascii=False), {case_id}
        )
