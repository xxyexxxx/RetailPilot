from __future__ import annotations

import json

from RetailPilot import scope_eval


def _conversations():
    return [
        {
            "conversation_id": "c1",
            "source": "test",
            "turns": [
                {"role": "user", "content": "推荐办公键盘"},
                {"role": "assistant_reference", "content": "你喜欢无线吗"},
                {"role": "user", "content": "对，最好静音"},
                {"role": "user", "content": "你是什么模型"},
                {"role": "user", "content": "查订单 JD20260801"},
            ],
        }
    ]


def test_build_scope_cases_only_uses_prior_dialogue():
    cases = scope_eval.build_scope_cases(_conversations())
    assert len(cases) == 4
    assert cases[0]["recent_history"] == []
    assert cases[1]["recent_history"][-1] == {
        "role": "assistant_reference",
        "content": "你喜欢无线吗",
    }
    assert "你是什么模型" not in json.dumps(cases[1], ensure_ascii=False)


def test_batch_scope_classifier_validates_every_case(monkeypatch):
    cases = scope_eval.build_scope_cases(_conversations())[:2]
    payload = {
        "items": [
            {
                "case_id": cases[0]["case_id"],
                "label": "pre_sales",
                "confidence": "high",
            },
            {
                "case_id": cases[1]["case_id"],
                "label": "pre_sales",
                "confidence": "high",
            },
        ]
    }
    monkeypatch.setattr(
        scope_eval, "invoke_judge", lambda *args, **kwargs: json.dumps(payload)
    )
    result = scope_eval.classify_scope_batch(cases)
    assert result["classifier_mode"] == "llm"
    assert result["attempts"] == 1
    assert len(result["items"]) == 2


def test_batch_scope_classifier_does_not_accept_partial_output(monkeypatch):
    cases = scope_eval.build_scope_cases(_conversations())[:2]
    partial = {
        "items": [
            {
                "case_id": cases[0]["case_id"],
                "label": "pre_sales",
                "confidence": "high",
            }
        ]
    }
    monkeypatch.setattr(
        scope_eval, "invoke_judge", lambda *args, **kwargs: json.dumps(partial)
    )
    result = scope_eval.classify_scope_batch(cases)
    assert result["classifier_mode"] == "fallback"
    assert result["attempts"] == 2
    assert result["items"] == []


def test_materialize_llm_scope_corpus_splits_filtered_turns():
    cases = scope_eval.build_scope_cases(_conversations())
    labels = ["pre_sales", "pre_sales", "out_of_scope", "order_read"]
    verdicts = {
        case["case_id"]: {
            "case_id": case["case_id"],
            "label": label,
            "confidence": "high",
        }
        for case, label in zip(cases, labels, strict=True)
    }
    eligible, filtered, uncertain, summary = scope_eval.materialize_llm_scope_corpus(
        _conversations(), verdicts
    )
    assert len(eligible) == 2
    assert [
        turn["source_turn_index"]
        for conversation in eligible
        for turn in conversation["turns"]
        if turn["role"] == "user"
    ] == [1, 2, 4]
    assert filtered[0]["turn_index"] == 3
    assert uncertain == []
    assert summary["eligible_user_turns"] == 3
    assert summary["scope_filtered_turns"] == 1
