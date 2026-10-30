from __future__ import annotations

import RetailPilot.customer_eval as customer_eval
from scripts.run_customer_service_eval import _markdown


def _cases(count: int) -> list[dict]:
    return [
        {
            "case_id": "semantic-metrics",
            "turns": [
                {"message": f"message-{index}", "expected": {}}
                for index in range(count)
            ],
        }
    ]


def _chat(_session_id: str, message: str, _user_id: str) -> dict:
    return {
        "reply": f"reply-{message}",
        "intents": ["chitchat"],
        "task_results": [],
        "trace": [],
        "grounding": {"pass": True},
    }


def test_semantic_metrics_only_count_evaluable_real_llm_verdicts(monkeypatch):
    verdicts = iter(
        [
            {"judge_mode": "llm", "pass": True, "failure_type": "none"},
            {
                "judge_mode": "llm",
                "evaluable": True,
                "pass": False,
                "failure_type": "context_miss",
            },
            {
                "judge_mode": "llm",
                "evaluable": False,
                "pass": False,
                "failure_type": "not_evaluable",
            },
            {
                "judge_mode": "deterministic",
                "pass": False,
                "failure_type": "duplicate_task",
            },
            {
                "judge_mode": "deterministic_fallback",
                "pass": True,
                "failure_type": "none",
            },
            {
                "judge_mode": "not_run",
                "pass": False,
                "failure_type": "system_error",
            },
        ]
    )
    monkeypatch.setattr(
        customer_eval,
        "judge_customer_turn",
        lambda *_args, **_kwargs: next(verdicts),
    )

    report = customer_eval.run_customer_cases(
        _cases(6), chat_function=_chat, use_llm_judge=True
    )
    semantic = report["summary"]["semantic_sidecar"]

    assert semantic["llm_judged_turns"] == 3
    assert semantic["llm_evaluable_turns"] == 2
    assert semantic["llm_not_evaluable_turns"] == 1
    assert semantic["turns_passed"] == 1
    assert semantic["turns_failed"] == 1
    assert semantic["turn_pass_rate"] == 0.5
    assert semantic["failure_counts"] == {"context_miss": 1}
    assert semantic["deterministic_turns"] == 1
    assert semantic["deterministic_fallback_turns"] == 1
    assert semantic["not_run_turns"] == 1
    assert semantic["judge_mode_counts"] == {
        "deterministic": 1,
        "deterministic_fallback": 1,
        "llm": 3,
        "not_run": 1,
    }


def test_semantic_pass_rate_is_null_without_real_llm_and_markdown_renders_na():
    report = customer_eval.run_customer_cases(
        _cases(1), chat_function=_chat, use_llm_judge=False
    )
    semantic = report["summary"]["semantic_sidecar"]

    assert semantic["llm_judged_turns"] == 0
    assert semantic["llm_evaluable_turns"] == 0
    assert semantic["turns_passed"] == 0
    assert semantic["turns_failed"] == 0
    assert semantic["turn_pass_rate"] is None
    assert semantic["failure_counts"] == {}
    assert semantic["deterministic_turns"] == 1
    assert semantic["deterministic_fallback_turns"] == 0
    assert semantic["not_run_turns"] == 0
    assert "pass 0 (N/A)" in _markdown(report)
