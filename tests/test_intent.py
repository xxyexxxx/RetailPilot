from __future__ import annotations

import json
from types import SimpleNamespace

import RetailPilot.llm as llm_module
from RetailPilot.config import settings
from RetailPilot.llm import (
    analyze_intents,
    assess_policy_evidence,
    maybe_refine_query,
    plan_after_sales,
)
from RetailPilot.query import parse_query


class _FakeModel:
    def __init__(self, payload: dict):
        self.payload = payload
        self.prompt = ""

    def invoke(self, prompt: str):
        self.prompt = prompt
        return SimpleNamespace(content=json.dumps(self.payload, ensure_ascii=False))


def test_llm_intent_analyzer_accepts_repeated_intents_as_separate_tasks(monkeypatch):
    model = _FakeModel(
        {
            "tasks": [
                {
                    "intent": "search",
                    "text": "推荐电竞鼠标",
                    "confidence": 0.98,
                    "target_terms": ["鼠标", "电竞鼠标"],
                },
                {
                    "intent": "search",
                    "text": "推荐机械键盘",
                    "confidence": 0.97,
                    "target_terms": ["键盘", "机械键盘"],
                },
            ]
        }
    )
    monkeypatch.setattr(settings, "intent_mode", "llm")
    monkeypatch.setattr(
        llm_module, "get_model", lambda **kwargs: model
    )

    tasks = analyze_intents("推荐鼠标和键盘")

    assert [task["intent"] for task in tasks] == ["search", "search"]
    assert [task["text"] for task in tasks] == ["推荐电竞鼠标", "推荐机械键盘"]
    assert all(task["source"] == "llm" for task in tasks)
    assert "多个相同 intent" in model.prompt[0].content[0]["text"]
    assert "不能合并成一个" in model.prompt[0].content[0]["text"]


def test_invalid_llm_output_returns_empty(monkeypatch):
    # LLM-only routing: an unavailable/unusable LLM yields no intent tasks
    # (the harness replies with a default message) instead of rule fallback.
    def _boom_model(temperature=0):
        raise RuntimeError("simulated LLM failure")

    monkeypatch.setattr(settings, "intent_mode", "llm")
    monkeypatch.setattr(llm_module, "get_model", _boom_model)

    tasks = analyze_intents("查订单 JD20260801，顺便推荐耳机")

    assert tasks == []


def test_search_task_with_category_but_no_terms_is_dropped(monkeypatch):
    # 护栏：search 任务只给白名单类目、不给检索词 → 该任务被丢弃（防类目宽召回）。
    model = _FakeModel(
        {
            "tasks": [
                {
                    "intent": "search",
                    "text": "想买书",
                    "category": "休闲娱乐文教",
                    "target_terms": [],
                }
            ]
        }
    )
    monkeypatch.setattr(settings, "intent_mode", "llm")
    monkeypatch.setattr(
        llm_module, "get_model", lambda **kwargs: model
    )

    tasks = analyze_intents("想买书")

    assert tasks == []


def test_search_task_with_terms_but_no_category_is_kept(monkeypatch):
    # 只有检索词没有类目时本地词典可补类目，护栏不拦截。
    model = _FakeModel(
        {
            "tasks": [
                {
                    "intent": "search",
                    "text": "推荐鼠标",
                    "category": None,
                    "target_terms": ["鼠标"],
                }
            ]
        }
    )
    monkeypatch.setattr(settings, "intent_mode", "llm")
    monkeypatch.setattr(
        llm_module, "get_model", lambda **kwargs: model
    )

    tasks = analyze_intents("推荐鼠标")

    assert len(tasks) == 1
    assert tasks[0]["intent"] == "search"
    assert tasks[0]["target_terms"] == ["鼠标"]



def test_query_refinement_only_accepts_evidence_grounded_terms(monkeypatch):
    parsed = parse_query("想买露营灯")
    original_rewrite = parsed.rewritten_query
    model = _FakeModel(
        {
            "category": "Sports & Outdoors",
            "category_evidence": "露营灯",
            "target_terms": [{"term": "camping lantern", "evidence": "露营灯"}],
            "concepts": [{"term": "camping", "evidence": "露营"}],
            "rewritten_query": "popular outdoor equipment",
            "confidence": 0.92,
        }
    )
    monkeypatch.setattr(settings, "llm_mode", "fallback")
    monkeypatch.setattr(
        llm_module, "get_model", lambda **kwargs: model
    )

    refined = maybe_refine_query(parsed)

    # 类目白名单已改纯中文，英文类目不在白名单 → 不采纳。
    assert refined.category is None
    assert refined.grounded_target_terms == ["camping lantern"]
    assert refined.llm_concepts == ["camping"]
    assert refined.concept_evidence == {
        "camping lantern": "露营灯",
        "camping": "露营",
    }
    assert refined.rewritten_query == original_rewrite
    assert refined.clarification_slot is None


def test_query_refinement_rejects_terms_without_verbatim_evidence(monkeypatch):
    parsed = parse_query("给我推荐一个")
    model = _FakeModel(
        {
            "category": "Computers",
            "category_evidence": "电脑",
            "target_terms": [{"term": "gaming laptop", "evidence": "游戏本"}],
            "concepts": [{"term": "high performance", "evidence": "高性能"}],
            "confidence": 0.99,
        }
    )
    monkeypatch.setattr(settings, "llm_mode", "fallback")
    monkeypatch.setattr(
        llm_module, "get_model", lambda **kwargs: model
    )

    refined = maybe_refine_query(parsed)

    assert refined.category is None
    assert refined.grounded_target_terms == []
    assert refined.llm_concepts == []
    assert refined.clarification_slot == "category"


def test_after_sales_planner_owns_topic_and_standalone_query(monkeypatch):
    model = _FakeModel(
        {
            "route": "policy_question",
            "topic": "退款",
            "standalone_query": "信用卡退款通常多久能够到账",
            "steps": [
                {
                    "step_id": "policy",
                    "tool": "policy_retrieve",
                    "args": {"query": "错误重复值", "topic": "发票"},
                }
            ],
        }
    )
    monkeypatch.setattr(
        llm_module, "get_model", lambda **kwargs: model
    )

    plan = plan_after_sales("这个多久到", history=[{"role": "user", "content": "信用卡退款"}])

    assert plan["topic"] == "退款"
    assert plan["standalone_query"] == "信用卡退款通常多久能够到账"
    assert plan["steps"][0]["args"] | {
        "topic": "退款",
        "query": "信用卡退款通常多久能够到账",
    } == plan["steps"][0]["args"]


def test_after_sales_planner_fails_closed_without_standalone_query(monkeypatch):
    model = _FakeModel(
        {
            "route": "policy_question",
            "topic": "退款",
            "steps": [
                {"step_id": "policy", "tool": "policy_retrieve", "args": {}}
            ],
        }
    )
    monkeypatch.setattr(
        llm_module, "get_model", lambda **kwargs: model
    )

    plan = plan_after_sales("退款呢")

    assert plan["route"] == "planner_error"
    assert plan["topic"] is None
    assert plan["steps"] == []


def test_policy_evidence_quality_is_decided_by_prompt(monkeypatch):
    captured = {}

    def fake_invoke(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        captured["prompt_kind"] = kwargs.get("prompt_kind")
        return json.dumps(
            {
                "decision": "refuse",
                "sufficient": False,
                "conflict": True,
                "reason": "同一适用范围的证据冲突",
                "clarification_question": None,
            },
            ensure_ascii=False,
        )

    monkeypatch.setattr(llm_module, "_invoke_json", fake_invoke)
    result = assess_policy_evidence(
        "能否退款",
        topic="退款",
        documents=[
            {
                "doc_id": "d1",
                "chunk_id": "c1",
                "title": "退款规则",
                "content": "支持退款。",
                "version": "1.0",
            }
        ],
    )

    assert result["conflict"] is True
    assert captured["prompt_kind"] == "policy_evidence"
    assert "不使用召回分数" in captured["prompt"][0].content[0]["text"]



