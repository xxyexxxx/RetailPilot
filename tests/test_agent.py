import RetailPilot.agent as agent_module
from RetailPilot.agent import chat


def _after_sales_state(text: str = "查一下退款规则") -> dict:
    return {
        "active_task": {
            "task_id": "task-1",
            "task_index": 1,
            "intent": "service",
            "text": text,
            "dialogue_frame": {},
            "decision": {"domain": "after_sales", "action": "policy", "tool": "policy_retrieve"},
        },
        "user_id": "demo-user",
        "context": {},
        "history_window": [],
    }


def test_runtime_graph_has_context_compaction_maintenance_node():
    assert set(agent_module.graph.nodes) - {"__start__"} == {
        "load_context",
        "compact_context",
        "route_and_extract",
        "execute_tool",
        "build_response",
        "verify_grounding",
        "verify_answer_faithfulness",
        "persist",
    }


def test_verify_grounding_fails_closed():
    result = agent_module.verify_grounding(
        {
            "grounding": {
                "pass": False,
                "claims": 1,
                "issues": [{"type": "unsupported_claim"}],
            }
        }
    )

    assert result["products"] == []
    assert result["citations"] == []
    assert "证据校验没有通过" in result["response"]


def test_ambiguous_business_sentence_reaches_intent_router(monkeypatch):
    """Unknown text is ambiguous, not high-confidence chitchat."""

    calls = []

    def fake_analyze(text, context=None, history=None):
        calls.append((text, context, history))
        return [
            {
                "intent": "service",
                "text": text,
                "query_text": text,
                "confidence": 0.96,
                "source": "llm:test",
            }
        ]

    monkeypatch.setattr(agent_module, "analyze_intents", fake_analyze)
    result = chat(
        "test-ambiguous-business-reaches-router",
        "我的洗衣机到现在都没有解决",
        "new-user",
    )

    assert calls
    assert result["intents"] == ["service"]
    assert result["intent_details"][0]["source"] == "llm:test"
    assert "你好，我在" not in result["reply"]


def test_active_service_fragment_reaches_router_instead_of_chitchat(monkeypatch):
    session = "test-active-service-fragment-router"
    chat(session, "发票怎么开", "new-user")

    def fake_analyze(text, context=None, history=None):
        return [
            {
                "intent": "service",
                "text": text,
                "query_text": text,
                "confidence": 0.9,
                "source": "llm:test",
            }
        ]

    monkeypatch.setattr(agent_module, "analyze_intents", fake_analyze)
    result = chat(session, "可我并未收到钱", "new-user")

    assert result["intents"] == ["service"]
    assert result["intent_details"][0]["source"] == "llm:test"
    assert "当前会话仍围绕" not in result["reply"]


def test_unknown_long_sentence_does_not_authorize_catalog_search():
    messages = (
        "我买的耳机用一段时间有一个不好使了",
        "我前几天买的，就降价了",
        "不知道为何评价提交不上",
        "请问如何取消每次下单的京东短信提醒呢",
    )
    for index, message in enumerate(messages):
        result = chat(f"test-no-length-search-fallback-{index}", message, "new-user")
        tools = {
            event.get("tool") for event in result["trace"] if event.get("node") == "tool_execution"
        }
        assert result["intents"] != ["search"]
        assert "catalog_search" not in tools


def test_service_duration_question_is_not_treated_as_a_location():
    session = "test-service-duration-not-location"
    chat(session, "售后维修点在哪里", "new-user")
    result = chat(session, "一定要这么久的吗", "new-user")

    assert "已收到您提供的地点" not in result["reply"]


def test_after_sales_tool_failure_is_terminal_and_reaches_reply(monkeypatch):
    calls: list[str] = []

    def fake_plan_after_sales(text, *, context=None, history=None):
        return {
            "route": "policy_question",
            "topic": "退款",
            "steps": [
                {"step_id": "order", "tool": "order_query", "args": {"order_id": None}},
                {
                    "step_id": "policy",
                    "tool": "policy_retrieve",
                    "args": {"product_context": "$order.product_context"},
                },
            ],
        }

    def fake_execute(name, scope, arguments, *, state=None, deadline=None):
        calls.append(name)
        return {
            "ok": False,
            "tool": name,
            "data": None,
            "observation": None,
            "error": {"type": "timeout", "message": "order_query timed out"},
            "trace": {
                "node": "tool_execution",
                "tool": name,
                "attempts": 2,
                "status": "error",
                "error_type": "timeout",
            },
        }

    monkeypatch.setattr(agent_module, "plan_after_sales", fake_plan_after_sales)
    monkeypatch.setattr(agent_module.registry, "execute", fake_execute)

    result = agent_module.after_sales_branch(_after_sales_state())
    payload = result["branch_outputs"][0]

    assert calls == ["order_query"]
    assert payload["tool_failure"] == {
        "type": "timeout",
        "message": "order_query timed out",
        "tool": "order_query",
        "step_id": "order",
        "attempts": 2,
    }
    assert "订单查询暂时失败" in payload["response"]
    assert payload["handoff"] is None
    assert payload["escalated"] is False


def test_after_sales_unresolved_reference_stops_dependent_step(monkeypatch):
    calls: list[str] = []

    def fake_plan_after_sales(text, *, context=None, history=None):
        return {
            "route": "policy_question",
            "topic": "退款",
            "steps": [
                {"step_id": "order", "tool": "order_query", "args": {"order_id": None}},
                {
                    "step_id": "policy",
                    "tool": "policy_retrieve",
                    "args": {"product_context": "$order.product_context"},
                },
            ],
        }

    def fake_execute(name, scope, arguments, *, state=None, deadline=None):
        calls.append(name)
        return {
            "ok": True,
            "tool": name,
            "data": {"order": None},
            "observation": {"order": None},
            "error": None,
            "trace": {
                "node": "tool_execution",
                "tool": name,
                "attempts": 1,
                "status": "success",
            },
        }

    monkeypatch.setattr(agent_module, "plan_after_sales", fake_plan_after_sales)
    monkeypatch.setattr(agent_module.registry, "execute", fake_execute)

    result = agent_module.after_sales_branch(_after_sales_state())
    payload = result["branch_outputs"][0]

    assert calls == ["order_query"]
    assert payload["tool_failure"]["type"] == "missing_dependency"
    assert "缺少上一步返回的订单或商品信息" in payload["response"]
    assert payload["required_slots"] == ["product_or_order"]


def test_after_sales_capability_boundary_does_not_handoff(monkeypatch):
    def fake_plan_after_sales(text, *, context=None, history=None):
        return {"route": "human_handoff", "steps": []}

    monkeypatch.setattr(agent_module, "plan_after_sales", fake_plan_after_sales)

    result = agent_module.after_sales_branch(_after_sales_state("我要投诉并赔偿"))
    payload = result["branch_outputs"][0]

    assert payload["handoff"] is None
    assert payload["escalated"] is False
    assert "人工转接" not in payload["response"]


def test_after_sales_planner_error_fails_closed_without_policy_lookup(monkeypatch):
    def fake_plan_after_sales(text, *, context=None, history=None):
        return {"route": "planner_error", "topic": None, "steps": []}

    def forbidden_execute(*args, **kwargs):
        raise AssertionError("planner failure must not execute a tool")

    monkeypatch.setattr(agent_module, "plan_after_sales", fake_plan_after_sales)
    monkeypatch.setattr(agent_module.registry, "execute", forbidden_execute)

    result = agent_module.after_sales_branch(_after_sales_state("耳机坏了怎么处理"))
    payload = result["branch_outputs"][0]

    assert "未执行政策查询" in payload["response"]
    assert payload["policy"] is None
    assert payload["tool_failure"] is None


def test_after_sales_rejects_weak_policy_evidence(monkeypatch):
    def fake_plan_after_sales(text, *, context=None, history=None):
        return {
            "route": "policy_question",
            "topic": "退款",
            "standalone_query": "退款多久到账",
            "steps": [
                {"step_id": "policy", "tool": "policy_retrieve", "args": {}}
            ],
        }

    document = {
        "doc_id": "policy-1",
        "chunk_id": "chunk-1",
        "content": "仅说明退款原路退回。",
    }

    def fake_execute(name, scope, arguments, *, state=None, deadline=None):
        return {
            "ok": True,
            "tool": name,
            "data": {"topic": "退款", "documents": [document]},
            "observation": None,
            "error": None,
            "trace": {"node": "tool_execution", "tool": name, "attempts": 1},
        }

    monkeypatch.setattr(agent_module, "plan_after_sales", fake_plan_after_sales)
    monkeypatch.setattr(agent_module.registry, "execute", fake_execute)
    monkeypatch.setattr(
        agent_module,
        "assess_policy_evidence",
        lambda *args, **kwargs: {
            "decision": "clarify",
            "sufficient": False,
            "conflict": False,
            "reason": "missing arrival-time evidence",
            "clarification_question": "请补充支付方式。",
        },
    )

    result = agent_module.after_sales_branch(_after_sales_state("退款多久到账"))
    payload = result["branch_outputs"][0]

    assert payload["response"] == "请补充支付方式。"
    assert payload["citations"] == []
    assert payload["policy"]["evidence_assessment"]["sufficient"] is False


def test_after_sales_product_order_lookup_asks_when_multiple_orders_match(monkeypatch):
    calls: list[str] = []

    def fake_plan_after_sales(text, *, context=None, history=None):
        return {
            "route": "policy_question",
            "topic": "退款",
            "order_scope": "none",
            "steps": [
                {
                    "step_id": "order",
                    "tool": "order_query",
                    "args": {"product_query": "鞋", "limit": 5},
                },
                {
                    "step_id": "policy",
                    "tool": "policy_retrieve",
                    "args": {"product_context": "$order.product_context"},
                },
            ],
        }

    def fake_execute(name, scope, arguments, *, state=None, deadline=None):
        calls.append(name)
        return {
            "ok": True,
            "tool": name,
            "data": {
                "order": None,
                "orders": [
                    {
                        "order_id": "O-1",
                        "sku": "sku-1",
                        "title": "黑色运动鞋",
                        "created_at": "2026-08-01T00:00:00Z",
                        "status": "已签收",
                    },
                    {
                        "order_id": "O-2",
                        "sku": "sku-2",
                        "title": "白色休闲鞋",
                        "created_at": "2026-07-01T00:00:00Z",
                        "status": "已签收",
                    },
                ],
            },
            "observation": None,
            "error": None,
            "trace": {"node": "tool_execution", "tool": name, "attempts": 1},
        }

    monkeypatch.setattr(agent_module, "plan_after_sales", fake_plan_after_sales)
    monkeypatch.setattr(agent_module.registry, "execute", fake_execute)

    result = agent_module.after_sales_branch(_after_sales_state("之前买的鞋，退款政策"))
    payload = result["branch_outputs"][0]

    assert calls == ["order_query"]
    assert payload["tool_failure"]["type"] == "ambiguous_order"
    assert payload["required_slots"] == ["order_id"]
    assert "找到多笔可能的订单" in payload["response"]
