from __future__ import annotations

from RetailPilot.customer_eval import grade_customer_turn, judge_customer_turn, load_customer_cases
from RetailPilot.conversation_replay import (
    apply_project_scope,
    badcase_turns,
    cluster_badcases,
    filter_conversations_by_project_scope,
    grade_replay_turn,
    grade_replay_turn_detailed,
    iter_jddc_uss,
    iter_jsonl,
    judge_selection,
    llm_judge_evidence,
    project_scope_gate,
    replay_turn_is_badcase,
    scope_filtered_turns,
    summarize_replay,
)
from RetailPilot.grounding import attach_citation_ids, claim, validate_claim_citations
from RetailPilot.tools import registry
import scripts.retry_replay_judge as retry_module
from scripts.run_multiturn_replay import load_badcase_conversation_ids


def test_badcase_replay_loader_deduplicates_conversations(tmp_path):
    path = tmp_path / "badcases.jsonl"
    path.write_text(
        '{"conversation_id":"c-1","turn_index":1}\n'
        '{"conversation_id":"c-1","turn_index":2}\n'
        '{"conversation_id":"c-2","turn_index":1}\n',
        encoding="utf-8",
    )

    assert load_badcase_conversation_ids(path) == {"c-1", "c-2"}


def test_badcase_replay_loader_collapses_duplicate_scope_suffixes(tmp_path):
    path = tmp_path / "badcases.jsonl"
    path.write_text(
        '{"conversation_id":"c-1::llm-scope-001::scope-001::scope-001"}\n',
        encoding="utf-8",
    )

    assert load_badcase_conversation_ids(path) == {
        "c-1::llm-scope-001::scope-001"
    }


def test_hard_duplicate_check_exempts_identical_idempotent_customer_retry():
    result = {
        "reply": "已记录该地址。",
        "task_results": [],
        "grounding": {"pass": True},
    }

    detail = grade_replay_turn_detailed(
        result,
        previous_reply="已记录该地址。",
        previous_message="广西[地址x]",
        current_message="广西[地址x]",
    )

    assert detail["pass"] is True
    duplicate = next(
        item for item in detail["checks"] if item["name"] == "not_duplicate_previous_reply"
    )
    assert duplicate["evidence"]["same_customer_input"] is True


def test_tool_registry_is_schema_first_and_scope_bounded():
    pre_sales = registry.schemas("pre_sales")
    after_sales = registry.schemas("after_sales")
    assert {item["name"] for item in pre_sales} == {
        "catalog_search",
        "product_detail",
    }
    assert {item["name"] for item in after_sales} == {
        "policy_retrieve",
        "order_query",
    }
    denied = registry.execute(
        "order_query",
        "pre_sales",
        {"user_id": "demo-user"},
    )
    invalid = registry.execute(
        "product_detail",
        "pre_sales",
        {"sku": ""},
    )
    assert denied["error"]["type"] == "scope_denied"
    assert invalid["error"]["type"] == "invalid_arguments"


def test_claim_citation_hard_check_detects_unsupported_and_mismatch():
    citation = {
        "sku": "sku-1",
        "field": "price",
        "value": 99.0,
        "source": "catalog",
    }
    citations = attach_citation_ids([citation])
    good = claim("价格 99", citation, claim_type="product_price")
    unsupported = {**good, "citation_id": "missing"}
    mismatch = {**good, "value": 100.0}
    product = {"sku": "sku-1", "price": 99.0}

    good_result = validate_claim_citations([good], citations, products=[product])
    bad_result = validate_claim_citations(
        [unsupported, mismatch], citations, products=[product]
    )
    assert good_result["pass"]
    assert not bad_result["pass"]
    assert set(bad_result["failure_types"]) == {
        "unsupported_claim",
        "citation_mismatch",
    }


def test_customer_harness_maps_trace_failures_to_agent_engineering_layers():
    result = {
        "intents": ["search"],
        "trace": [{"node": "tool_execution", "tool": "catalog_search"}],
        "grounding": {"pass": True, "issues": []},
        "task_results": [],
        "context_budget": {"history_chars": 100, "history_char_budget": 2400},
    }
    grade = grade_customer_turn(
        result,
        {"intent": "service", "tools": ["policy_retrieve"], "escalated": True},
    )
    assert not grade.passed
    assert set(grade.failure_types) == {
        "intent_route_error",
        "wrong_tool_selected",
        "missing_handoff",
    }


def test_customer_case_catalog_has_presales_and_after_sales():
    cases = load_customer_cases("config/customer-service-cases.json")
    intents = {
        turn["expected"].get("intent")
        for case in cases
        for turn in case["turns"]
    }
    assert {"search", "compare", "service", "order", "chitchat"} <= intents


def test_semantic_judge_offline_fallback_detects_duplicate_tasks():
    verdict = judge_customer_turn(
        "闲聊",
        {
            "reply": "重复回复",
            "intents": ["chitchat", "chitchat"],
            "task_results": [
                {"intent": "chitchat", "task_text": "闲聊"},
                {"intent": "chitchat", "task_text": "闲聊"},
            ],
        },
        use_llm=False,
    )
    assert not verdict["pass"]
    assert verdict["failure_type"] == "duplicate_task"


def test_jddc_uss_parser_keeps_real_user_turns_and_skips_overall(tmp_path):
    source = tmp_path / "JDDC.txt"
    source.write_text(
        "SYSTEM\t您好\n"
        "USER\t订单怎么还没到\t物流\t1,2,2\n"
        "SYSTEM\t我帮您查\n"
        "USER\t能催一下吗\t物流\t2,2,3\n"
        "USER\tOVERALL\t\t1,1,1\t不满意\n\n",
        encoding="utf-8",
    )
    records = list(iter_jddc_uss(source))
    assert len(records) == 1
    assert [turn["content"] for turn in records[0]["turns"]] == [
        "您好",
        "订单怎么还没到",
        "我帮您查",
        "能催一下吗",
    ]
    assert records[0]["turns"][1]["source_satisfaction"] == [1, 2, 2]


def test_generic_jsonl_parser_and_replay_anomaly_grade(tmp_path):
    source = tmp_path / "dialogues.jsonl"
    source.write_text(
        '{"conversation_id":"c1","turns":['
        '{"role":"customer","text":"推荐键盘"},'
        '{"role":"agent","text":"好的"},'
        '{"role":"customer","text":"静音一点"}]}\n',
        encoding="utf-8",
    )
    record = next(iter_jsonl(source))
    assert [turn["role"] for turn in record["turns"]] == [
        "user", "assistant_reference", "user"
    ]
    passed, failures = grade_replay_turn(
        {
            "reply": "同一句",
            "task_results": [
                {"intent": "search", "task_text": "推荐键盘"},
                {"intent": "search", "task_text": "推荐键盘"},
            ],
            "grounding": {"pass": True},
        },
        previous_reply="同一句",
    )
    assert not passed
    assert set(failures) == {"duplicate_reply", "duplicate_task"}


def test_replay_judge_selection_is_reproducible():
    conversations = [
        {
            "conversation_id": "c1",
            "turns": [
                {"role": "user", "content": "推荐键盘"},
                {"role": "user", "content": "再便宜一点"},
            ],
        },
        {
            "conversation_id": "c2",
            "turns": [{"role": "user", "content": "查订单 JD20260801"}],
        },
    ]
    first = judge_selection(conversations, judge_limit=2, seed=7)
    assert first == judge_selection(conversations, judge_limit=2, seed=7)
    assert len(first) == 2
    clipped = judge_selection(
        conversations, judge_limit=0, seed=7, max_user_turns=1
    )
    assert clipped == {("c1", 1), ("c2", 1)}


def test_project_scope_gate_follows_declared_agent_direction():
    assert project_scope_gate("推荐 AI 相关书籍")["domain"] == "pre_sales"
    assert project_scope_gate("这个能七天无理由退吗")["domain"] == "after_sales_rag"
    assert project_scope_gate("订单 JD20260801 到哪了")["domain"] == "order_read"
    inherited = project_scope_gate("但是却没开", active_domain="after_sales_rag")
    assert inherited["in_scope"] and inherited["inherited"]
    presales_followup = project_scope_gate("再静音一点，最好无线", active_domain="pre_sales")
    assert presales_followup["in_scope"] and presales_followup["inherited"]
    assert project_scope_gate("最近电脑降价了，推荐一台")["domain"] == "pre_sales"

    filtered = {
        message: project_scope_gate(message)["reason"]
        for message in (
            "你是什么模型",
            "我的火车票怎么退",
            "现在想买东西，没额度了",
            "购物车白屏",
            "帮我直接下单付款",
            "我前几天买的，就降价了",
        )
    }
    assert filtered == {
        "你是什么模型": "smalltalk_or_meta",
        "我的火车票怎么退": "third_party_service",
        "现在想买东西，没额度了": "account_or_credit",
        "购物车白屏": "commerce_ui_support",
        "帮我直接下单付款": "transaction_execution",
        "我前几天买的，就降价了": "price_protection",
    }


def test_scope_filter_splits_segments_before_agent_replay():
    conversations = [
        {
            "conversation_id": "source-c1",
            "source": "test",
            "turns": [
                {"role": "user", "content": "推荐办公键盘"},
                {"role": "assistant_reference", "content": "参考回复一"},
                {"role": "user", "content": "再静音一点"},
                {"role": "user", "content": "你是什么模型"},
                {"role": "assistant_reference", "content": "参考回复二"},
                {"role": "user", "content": "查订单 JD20260801"},
                {"role": "user", "content": "完整物流"},
            ],
        }
    ]
    eligible, filtered, summary = filter_conversations_by_project_scope(conversations)

    assert [item["conversation_id"] for item in eligible] == [
        "source-c1::scope-001",
        "source-c1::scope-002",
    ]
    assert [
        turn["source_turn_index"]
        for conversation in eligible
        for turn in conversation["turns"]
        if turn["role"] == "user"
    ] == [1, 2, 4, 5]
    assert filtered[0]["turn_index"] == 3
    assert filtered[0]["reason"] == "smalltalk_or_meta"
    assert summary["source_user_turns"] == 5
    assert summary["eligible_user_turns"] == 4
    assert summary["scope_filtered_turns"] == 1
    assert summary["eligible_segments"] == 2


def test_scope_filter_preserves_trusted_llm_preclassification():
    llm_scope = {
        "scope_version": "project-scope-llm-v1",
        "classifier_mode": "llm",
        "in_scope": True,
        "domain": "after_sales_rag",
        "confidence": "high",
    }
    conversations = [
        {
            "conversation_id": "llm-scope-c1",
            "source": "test",
            "metadata": {"scope_gate_version": "project-scope-llm-v1"},
            "turns": [
                {
                    "role": "user",
                    "content": "这个要怎么办",
                    "scope_gate": llm_scope,
                }
            ],
        }
    ]

    eligible, filtered, summary = filter_conversations_by_project_scope(conversations)

    assert filtered == []
    assert summary["eligible_user_turns"] == 1
    assert summary["scope_gate_version"] == "project-scope-llm-v1"
    preserved = eligible[0]["turns"][0]["scope_gate"]
    assert preserved == llm_scope
    assert judge_selection(eligible, judge_limit=0, seed=7) == {
        (eligible[0]["conversation_id"], 1)
    }


def test_scope_filtered_turn_is_neither_pass_nor_badcase():
    records = [
        {
            "conversation_id": "scope-c1",
            "turns": [
                {
                    "turn_index": 1,
                    "message": "购物车白屏",
                    "reply": "任意旧回复",
                    "hard_pass": False,
                    "hard_failure_types": ["duplicate_reply"],
                    "semantic_judge": {
                        "judge_mode": "llm",
                        "pass": False,
                        "failure_type": "off_topic",
                    },
                    "evaluation_evidence": {},
                    "latency_ms": 1,
                },
                {
                    "turn_index": 2,
                    "message": "推荐键盘",
                    "reply": "推荐结果",
                    "hard_pass": True,
                    "hard_failure_types": [],
                    "semantic_judge": {
                        "judge_mode": "llm",
                        "pass": True,
                        "failure_type": "none",
                    },
                    "evaluation_evidence": {},
                    "latency_ms": 1,
                },
            ],
        }
    ]
    apply_project_scope(records)
    assert not replay_turn_is_badcase(records[0]["turns"][0])
    assert badcase_turns(records) == []
    assert scope_filtered_turns(records)[0]["reason"] == "commerce_ui_support"
    summary = summarize_replay(records)
    assert summary["turns"] == 2
    assert summary["in_scope_turns"] == 1
    assert summary["scope_filtered_turns"] == 1
    assert summary["llm_judged_turns"] == 1
    assert summary["llm_turns_passed"] == 1
    assert summary["hard_pass_rate"] == 1.0


def test_replay_hard_check_emits_auditable_evidence():
    detail = grade_replay_turn_detailed(
        {
            "reply": "同一句",
            "task_results": [
                {"intent": "search", "task_text": "推荐键盘"},
                {"intent": "search", "task_text": "推荐键盘"},
            ],
            "grounding": {
                "pass": False,
                "failure_types": ["unsupported_claim"],
            },
        },
        previous_reply="同一句",
    )
    assert detail["checker"] == "program_hard_check"
    assert not detail["pass"]
    assert detail["failure_types"] == [
        "duplicate_reply",
        "duplicate_task",
        "unsupported_claim",
    ]
    checks = {item["name"]: item for item in detail["checks"]}
    assert checks["not_duplicate_previous_reply"]["evidence"]["normalized_equal"]
    assert checks["unique_agent_tasks"]["evidence"] == {
        "task_count": 2,
        "unique_task_count": 1,
    }


def test_fallback_is_never_counted_as_real_llm_judge_or_badcase():
    fallback = {
        "judge_mode": "deterministic_fallback",
        "pass": False,
        "failure_type": "duplicate_task",
        "summary": "程序降级诊断",
        "fallback_reason": "missing key",
    }
    evidence = llm_judge_evidence(fallback, selected=True, requested=True)
    assert not evidence["executed"]
    assert not evidence["counted_as_llm_judge"]
    assert evidence["pass"] is None
    assert evidence["fallback_diagnostic"]["pass"] is False

    turn = {
        "hard_pass": True,
        "hard_failure_types": [],
        "semantic_judge": fallback,
        "latency_ms": 1,
    }
    assert not replay_turn_is_badcase(turn)
    summary = summarize_replay([{"conversation_id": "c1", "turns": [turn]}])
    assert summary["semantic_selected_turns"] == 1
    assert summary["llm_judged_turns"] == 0
    assert summary["llm_pass_rate"] is None
    assert summary["fallback_turns"] == 1
    assert not summary["llm_judge_effective"]


def test_not_evaluable_llm_turn_is_excluded_from_pass_rate_and_badcases():
    turn = {
        "turn_index": 1,
        "message": "微信",
        "reply": "请说明具体需求",
        "hard_pass": True,
        "hard_failure_types": [],
        "scope_gate": {"in_scope": True, "domain": "after_sales_rag"},
        "semantic_judge": {
            "judge_mode": "llm",
            "evaluable": False,
            "pass": False,
            "failure_type": "not_evaluable",
            "summary": "只依赖未注入的人工客服提问",
        },
        "evaluation_evidence": {},
        "latency_ms": 1,
    }
    records = [{"conversation_id": "counterfactual-c1", "turns": [turn]}]

    assert not replay_turn_is_badcase(turn)
    assert badcase_turns(records) == []
    summary = summarize_replay(records)
    assert summary["llm_judged_turns"] == 1
    assert summary["llm_evaluable_turns"] == 0
    assert summary["llm_not_evaluable_turns"] == 1
    assert summary["llm_pass_rate"] is None
    assert summary["fallback_turns"] == 0
    assert summary["llm_judge_effective"]


def test_real_llm_failure_is_counted_and_clustered_with_evidence():
    semantic = {
        "judge_mode": "llm",
        "pass": False,
        "failure_type": "context_miss",
        "summary": "没有理解上一轮所指订单",
        "context_use": 0,
    }
    turn = {
        "turn_index": 2,
        "message": "这个呢",
        "reply": "请重新描述需求",
        "hard_pass": True,
        "hard_failure_types": [],
        "semantic_judge": semantic,
        "evaluation_evidence": {
            "hard_check": {"checker": "program_hard_check", "pass": True},
            "llm_judge": llm_judge_evidence(
                semantic, selected=True, requested=True
            ),
        },
        "latency_ms": 1,
    }
    records = [{"conversation_id": "c1", "turns": [turn]}]
    summary = summarize_replay(records)
    assert summary["llm_judged_turns"] == 1
    assert summary["llm_turns_passed"] == 0
    assert summary["semantic_failure_counts"] == {"context_miss": 1}
    assert replay_turn_is_badcase(turn)
    assert badcase_turns(records)[0]["llm_judge_summary"] == "没有理解上一轮所指订单"
    clusters = cluster_badcases(records)
    assert clusters[0]["evaluator"] == "llm_judge"
    assert clusters[0]["failure_type"] == "context_miss"
    assert clusters[0]["turn_count"] == 1
    assert "signal=unclassified_fragment" in clusters[0]["routing_signature"]
    assert "tools=no_tool" in clusters[0]["routing_signature"]


def test_badcase_clusters_split_same_judge_label_by_harness_signature():
    def failed_turn(message, intent, tools, source, reply):
        return {
            "turn_index": 1,
            "message": message,
            "reply": reply,
            "intents": [intent],
            "tools": tools,
            "trace": [
                {
                    "node": "intent_analyzer",
                    "tasks": [{"intent": intent, "source": source}],
                }
            ],
            "hard_pass": True,
            "hard_failure_types": [],
            "semantic_judge": {
                "judge_mode": "llm",
                "pass": False,
                "failure_type": "off_topic",
                "summary": "答非所问",
            },
        }

    records = [
        {
            "conversation_id": "c1",
            "turns": [
                failed_turn(
                    "配送太慢了",
                    "search",
                    ["catalog_search"],
                    "rules",
                    "我还不能确定具体商品类型。",
                )
            ],
        },
        {
            "conversation_id": "c2",
            "turns": [
                failed_turn(
                    "我要发票",
                    "service",
                    ["policy_retrieve"],
                    "符合完好标准的商品支持退货。",
                    "rules:current_turn_service",
                )
            ],
        },
    ]
    clusters = cluster_badcases(records)

    assert len(clusters) == 2
    assert {item["current_business_signal"] for item in clusters} == {"delivery", "invoice"}
    assert {item["tool_signature"] for item in clusters} == {
        "catalog_search",
        "policy_retrieve",
    }


def test_retry_replay_judge_only_replaces_fallback_turns(monkeypatch):
    calls = []

    def fake_judge(message, result, *, history, use_llm):
        calls.append((message, result["reply"], list(history), use_llm))
        return {
            "judge_mode": "llm",
            "pass": True,
            "failure_type": "none",
            "summary": "重试成功",
            "intent_fit": 2,
            "requirement_coverage": 2,
            "context_use": 2,
            "relevance": 2,
            "concise_and_non_repetitive": 2,
        }

    monkeypatch.setattr(retry_module, "judge_customer_turn", fake_judge)
    records = [
        {
            "conversation_id": "c1",
            "turns": [
                {
                    "message": "我要发票",
                    "reply": "请提供订单号",
                    "intents": ["service"],
                    "semantic_judge": {"judge_mode": "not_selected", "pass": None},
                },
                {
                    "message": "这个能重开吗",
                    "reply": "请补充信息",
                    "intents": ["service"],
                    "semantic_judge": {
                        "judge_mode": "deterministic_fallback",
                        "pass": True,
                    },
                    "evaluation_evidence": {},
                },
            ],
        }
    ]

    stats = retry_module.retry_fallback_turns(
        records, max_passes=2, delay_seconds=0
    )

    assert stats == {"attempted": 1, "recovered": 1, "unresolved": 0}
    assert len(calls) == 1
    assert calls[0][0] == "这个能重开吗"
    assert calls[0][2][-1] == {"role": "assistant", "content": "请提供订单号"}
    assert records[0]["turns"][0]["semantic_judge"]["judge_mode"] == "not_selected"
    assert records[0]["turns"][1]["semantic_judge"]["judge_mode"] == "llm"
    assert records[0]["turns"][1]["evaluation_evidence"]["llm_judge"][
        "counted_as_llm_judge"
    ]
