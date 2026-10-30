from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from RetailPilot import customer_eval, llm_config
from RetailPilot.llm_config import (
    JudgeUnavailable,
    agent_max_tokens,
    agent_model_status,
    extract_llm_usage,
    get_model,
    get_judge_config,
    invoke_judge,
    invoke_judge_stream,
    invoke_with_telemetry,
    judge_config_status,
)


JUDGE_ENV_NAMES = (
    "JUDGE_API_KEY",
    "JUDGE_BASE_URL",
    "JUDGE_MODEL",
    "JUDGE_TIMEOUT_SECONDS",
    "JUDGE_MAX_RETRIES",
    "JUDGE_MAX_TOKENS",
    "JUDGE_JSON_MODE",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_RETRIES",
    "LLM_THINKING",
)


def _clear_judge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in JUDGE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_judge_config_prefers_isolated_settings_and_redacts_key(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "runtime-secret")
    monkeypatch.setenv("JUDGE_API_KEY", "judge-secret")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example.test/v1")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_MAX_TOKENS", "321")

    config = get_judge_config()
    status = judge_config_status()

    assert config.api_key == "judge-secret"
    assert config.model == "judge-model"
    assert config.max_tokens == 321
    assert status["endpoint_host"] == "judge.example.test"
    assert status["credential_source"] == "JUDGE_API_KEY"
    assert "api_key" not in status
    assert "judge-secret" not in json.dumps(status)


def test_judge_default_budget_covers_hidden_reasoning_tokens(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "judge-secret")

    assert get_judge_config().max_tokens == 3000


def test_invoke_judge_uses_bounded_openai_compatible_json_request(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "secret")
    monkeypatch.setenv("JUDGE_BASE_URL", "https://judge.example.test/v1")
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("JUDGE_MAX_TOKENS", "123")
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("RetailPilot.llm_config.requests.post", fake_post)

    assert invoke_judge("return json") == '{"ok":true}'
    assert captured["url"] == "https://judge.example.test/v1/chat/completions"
    assert captured["json"]["max_tokens"] == 123
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["headers"]["Authorization"] == "Bearer secret"


def test_invoke_judge_can_disable_thinking_for_batch_classification(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "secret")
    captured = {}

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"items":[]}'}}]}

    def fake_post(url, **kwargs):
        captured.update(kwargs)
        return Response()

    monkeypatch.setattr("RetailPilot.llm_config.requests.post", fake_post)
    invoke_judge("classify", thinking=False, max_tokens=777)

    assert captured["json"]["thinking"] == {"type": "disabled"}
    assert captured["json"]["max_tokens"] == 777


def test_agent_model_disables_thinking_and_exposes_secret_free_status(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("LLM_API_KEY", "agent-secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://agent.example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "qwen3.5-flash")
    monkeypatch.setenv("LLM_THINKING", "false")
    captured = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(llm_config, "ChatOpenAI", FakeChatOpenAI)

    get_model(temperature=0, max_tokens=321)
    status = agent_model_status()

    assert captured["model"] == "qwen3.5-flash"
    assert captured["max_tokens"] == 321
    assert captured["extra_body"] == {"thinking": {"type": "disabled"}}
    assert status == {
        "configured": True,
        "model": "qwen3.5-flash",
        "endpoint_host": "agent.example.test",
        "thinking_enabled": False,
    }
    assert "agent-secret" not in json.dumps(status)


def test_agent_output_caps_are_per_scene_and_configurable(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS_QUERY_REFINE", "321")

    assert agent_max_tokens("query_refine") == 321
    assert agent_max_tokens("chitchat") == 256
    assert agent_max_tokens("reply_policy") == 800


def test_extract_usage_includes_cached_and_cache_creation_tokens():
    response = SimpleNamespace(
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 120,
            "total_tokens": 1120,
            "input_token_details": {"cache_read": 600, "cache_creation": 100},
        },
        response_metadata={},
    )

    assert extract_llm_usage(response) == {
        "input_tokens": 1000,
        "output_tokens": 120,
        "total_tokens": 1120,
        "cached_tokens": 600,
        "cache_creation_tokens": 100,
        "cache_miss_tokens": 400,
    }


def test_runtime_llm_call_persists_usage_and_cost(monkeypatch):
    recorded = []
    monkeypatch.setattr("RetailPilot.database.record_llm_usage", recorded.append)
    monkeypatch.setenv("LLM_MODEL", "qwen3.5-flash")
    monkeypatch.setenv("LLM_INPUT_COST_PER_1M", "1")
    monkeypatch.setenv("LLM_CACHED_INPUT_COST_PER_1M", "0.2")
    monkeypatch.setenv("LLM_OUTPUT_COST_PER_1M", "2")

    class Model:
        @staticmethod
        def invoke(prompt):
            return SimpleNamespace(
                content="ok",
                usage_metadata={
                    "input_tokens": 1000,
                    "output_tokens": 100,
                    "total_tokens": 1100,
                    "input_token_details": {"cache_read": 500},
                },
                response_metadata={},
            )

    response = invoke_with_telemetry(
        Model(), "prompt", prompt_kind="intent_router"
    )

    assert response.content == "ok"
    assert recorded[0]["cached_tokens"] == 500
    assert recorded[0]["cache_miss_tokens"] == 500
    assert recorded[0]["estimated_cost"] == pytest.approx(0.0008)


def test_invoke_judge_accepts_reasoning_content_when_content_empty(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "secret")
    monkeypatch.setenv("JUDGE_MAX_RETRIES", "0")

    class Response:
        status_code = 200

        @staticmethod
        def json():
            return {
                "choices": [
                    {"message": {"content": "", "reasoning_content": '{"ok":true}'}}
                ]
            }

    monkeypatch.setattr(
        "RetailPilot.llm_config.requests.post", lambda *args, **kwargs: Response()
    )
    assert invoke_judge("return json") == '{"ok":true}'


def test_invoke_judge_stream_joins_openai_sse_content(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "secret")
    captured = {}

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def iter_lines(decode_unicode=True):
            del decode_unicode
            yield 'data: {"choices":[{"delta":{"content":"{\\"ok\\":"}}]}'
            yield 'data: {"choices":[{"delta":{"content":"true}"}}]}'
            yield "data: [DONE]"

    def fake_post(url, **kwargs):
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("RetailPilot.llm_config.requests.post", fake_post)

    assert invoke_judge_stream("return json", thinking=False) == '{"ok":true}'
    assert captured["json"]["stream"] is True
    assert captured["json"]["thinking"] == {"type": "disabled"}


def test_invoke_judge_stream_decodes_chinese_sse_as_utf8(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "secret")

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def iter_lines(decode_unicode=True):
            assert decode_unicode is False
            yield 'data: {"choices":[{"delta":{"content":"用户意图"}}]}'.encode(
                "utf-8"
            )
            yield b"data: [DONE]"

    monkeypatch.setattr(
        "RetailPilot.llm_config.requests.post",
        lambda *args, **kwargs: Response(),
    )

    assert invoke_judge_stream("return json", thinking=False) == "用户意图"


def test_invoke_judge_stream_reassembles_split_sse_json_event(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "secret")

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def iter_lines(decode_unicode=True):
            del decode_unicode
            yield 'data: {"choices":[{"delta":{"content":"{\\"ok\\"'
            yield 'data: :true}"}}]}'
            yield "data: [DONE]"

    monkeypatch.setattr(
        "RetailPilot.llm_config.requests.post",
        lambda *args, **kwargs: Response(),
    )

    assert invoke_judge_stream("return json", thinking=False) == '{"ok":true}'


def test_invoke_judge_stream_accepts_unprefixed_event_continuation(monkeypatch):
    _clear_judge_env(monkeypatch)
    monkeypatch.setenv("JUDGE_API_KEY", "secret")

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @staticmethod
        def iter_lines(decode_unicode=True):
            del decode_unicode
            yield 'data: {"choices":[{"delta":'
            yield '{"content":"{\\"ok\\":true}"}}]}'
            yield "data: [DONE]"

    monkeypatch.setattr(
        "RetailPilot.llm_config.requests.post",
        lambda *args, **kwargs: Response(),
    )

    assert invoke_judge_stream("return json", thinking=False) == '{"ok":true}'


def test_missing_judge_credentials_are_explicit(monkeypatch):
    _clear_judge_env(monkeypatch)
    with pytest.raises(JudgeUnavailable, match="No independent judge credential"):
        get_judge_config()
    assert judge_config_status()["configured"] is False


def test_customer_turn_marks_llm_only_after_valid_schema(monkeypatch):
    monkeypatch.setattr(
        customer_eval,
        "invoke_judge",
        lambda prompt, **kwargs: json.dumps(
            {
                "pass": True,
                "intent_fit": 2,
                "requirement_coverage": 2,
                "context_use": 2,
                "relevance": 2,
                "concise_and_non_repetitive": 2,
                "failure_type": "none",
                "summary": "理解了用户并正确作答",
            }
        ),
    )
    verdict = customer_eval.judge_customer_turn(
        "查订单",
        {"reply": "请提供订单号", "intents": ["order"], "task_results": []},
        use_llm=True,
    )
    assert verdict["judge_mode"] == "llm"
    assert verdict["pass"] is True
    assert verdict["judge_attempts"] == 1
    assert verdict["output_repaired"] is False


def test_customer_turn_accepts_explicit_not_evaluable_verdict(monkeypatch):
    monkeypatch.setattr(
        customer_eval,
        "invoke_judge",
        lambda prompt, **kwargs: json.dumps(
            {
                "evaluable": False,
                "pass": False,
                "intent_fit": 1,
                "requirement_coverage": 1,
                "context_use": 0,
                "relevance": 1,
                "concise_and_non_repetitive": 2,
                "failure_type": "not_evaluable",
                "summary": "当前碎片只依赖未注入的原客服提问",
            }
        ),
    )

    verdict = customer_eval.judge_customer_turn(
        "微信",
        {"reply": "请说明具体需求", "intents": ["chitchat"], "task_results": []},
        history=[],
        source_reference=[
            {"role": "assistant_reference", "content": "您是怎样支付的呢"}
        ],
        use_llm=True,
    )

    assert verdict["judge_mode"] == "llm"
    assert verdict["evaluable"] is False
    assert verdict["failure_type"] == "not_evaluable"


def test_customer_turn_invalid_llm_output_is_never_marked_llm(monkeypatch):
    monkeypatch.setattr(
        customer_eval, "invoke_judge", lambda prompt, **kwargs: "not-json"
    )
    verdict = customer_eval.judge_customer_turn(
        "查订单",
        {"reply": "请提供订单号", "intents": ["order"], "task_results": []},
        use_llm=True,
    )
    assert verdict["judge_mode"] == "deterministic_fallback"
    assert "fallback_reason" in verdict


@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {
            "pass": True,
            "intent_fit": 0.9,
            "requirement_coverage": 0.8,
            "context_use": 1.0,
            "relevance": 0.9,
            "concise_and_non_repetitive": 0.8,
            "failure_type": None,
            "summary": "used a 0-1 scale",
        },
        {
            "pass": False,
            "intent_fit": 1,
            "requirement_coverage": 0,
            "context_use": 0,
            "relevance": 1,
            "concise_and_non_repetitive": 2,
            "failure_type": None,
            "summary": "missing failure category",
        },
    ],
)
def test_customer_turn_repairs_invalid_contract_with_second_real_call(
    monkeypatch, invalid
):
    repaired = {
        "pass": False,
        "intent_fit": 1,
        "requirement_coverage": 1,
        "context_use": 0,
        "relevance": 1,
        "concise_and_non_repetitive": 2,
        "failure_type": "context_miss",
        "summary": "没有使用上一轮订单上下文",
    }
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return json.dumps(invalid if len(calls) == 1 else repaired)

    monkeypatch.setattr(customer_eval, "invoke_judge", fake_invoke)
    verdict = customer_eval.judge_customer_turn(
        "这个订单呢",
        {"reply": "请提供订单号", "intents": ["order"], "task_results": []},
        history=[{"role": "user", "content": "订单 JD1 到哪了"}],
        use_llm=True,
    )

    assert len(calls) == 2
    assert calls[0][1]["system"] == calls[1][1]["system"]
    assert "独立重新评审" in calls[1][0]
    assert "上次无效输出" not in calls[1][0]
    assert "ValidationError" not in calls[1][0]
    assert "这个订单呢" in calls[1][0]
    assert verdict["judge_mode"] == "llm"
    assert verdict["judge_attempts"] == 2
    assert verdict["output_repaired"] is True
    assert verdict["failure_type"] == "context_miss"


def test_customer_turn_failed_repair_remains_fallback(monkeypatch):
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append(prompt)
        return "{}"

    monkeypatch.setattr(customer_eval, "invoke_judge", fake_invoke)
    verdict = customer_eval.judge_customer_turn(
        "查订单",
        {"reply": "请提供订单号", "intents": ["order"], "task_results": []},
        use_llm=True,
    )

    assert len(calls) == 2
    assert verdict["judge_mode"] == "deterministic_fallback"
    assert "fallback_reason" in verdict


def test_customer_turn_losslessly_normalizes_null_failure_for_pass(monkeypatch):
    payload = {
        "pass": True,
        "intent_fit": 2,
        "requirement_coverage": 2,
        "context_use": 2,
        "relevance": 2,
        "concise_and_non_repetitive": 2,
        "failure_type": None,
        "summary": "回复正确",
    }
    calls = []

    def fake_invoke(prompt, **kwargs):
        calls.append(prompt)
        return json.dumps(payload)

    monkeypatch.setattr(customer_eval, "invoke_judge", fake_invoke)
    verdict = customer_eval.judge_customer_turn(
        "查订单",
        {"reply": "请提供订单号", "intents": ["order"], "task_results": []},
        use_llm=True,
    )

    assert len(calls) == 1
    assert verdict["judge_mode"] == "llm"
    assert verdict["failure_type"] == "none"
