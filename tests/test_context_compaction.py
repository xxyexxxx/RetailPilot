from __future__ import annotations

import RetailPilot.agent as agent_module
import RetailPilot.llm as llm_module
from RetailPilot.context import bounded_history
from RetailPilot.llm import _history_block, _history_payload


def _messages(count: int) -> list[dict[str, object]]:
    return [
        {"id": index, "role": "user", "content": f"第{index}轮用户消息"}
        for index in range(1, count + 1)
    ]


def test_history_is_append_only_until_context_ratio_triggers(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "100000")

    history, budget = bounded_history(
        _messages(20),
        current_message="继续",
    )

    assert len(history) == 20
    assert budget["history_truncated"] is False
    assert budget["needs_compaction"] is False


def test_context_ratio_is_the_compaction_trigger(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "20")

    _history, budget = bounded_history(
        _messages(4),
        current_message="继续查",
    )

    assert budget["context_usage_ratio"] >= 0.8
    assert budget["needs_compaction"] is True


def test_compaction_node_persists_structured_summary_and_anchor(monkeypatch):
    saved: list[dict] = []
    monkeypatch.setattr(
        agent_module,
        "summarize_history",
        lambda history, *, existing_summary=None, context=None: {
            "version": 1,
            "facts": ["用户预算500元"],
            "decisions": ["保留蓝色款"],
            "open_loops": ["等待订单号"],
            "focus": "退款",
        },
    )
    monkeypatch.setattr(
        agent_module,
        "save_session",
        lambda session_id, user_id, context: saved.append(context),
    )
    monkeypatch.setenv("LLM_CONTEXT_WINDOW", "100000")

    state = {
        "session_id": "compact-test",
        "user_id": "user-1",
        "message": "继续",
        "session": {"messages": _messages(12)},
        "context": {},
        "context_budget": {"needs_compaction": True, "context_usage_ratio": 0.81},
    }

    result = agent_module.compact_context(state)

    assert saved
    assert result["context"]["history_summary"]["facts"] == ["用户预算500元"]
    assert result["context"]["history_summary_through_id"] == 4
    assert result["history_window"][0]["role"] == "summary"
    assert len(result["history_window"]) == 9  # summary + the newest 8 raw messages
    assert result["context_budget"]["compaction_applied"] is True


def test_llm_receives_prepared_history_without_second_fixed_six_turn_cap():
    history = [
        {"role": "user", "content": f"消息{i}"}
        for i in range(8)
    ]

    assert len(_history_payload(history)) == 8
    assert "消息0" in _history_block(history)


def test_first_history_compaction_uses_empty_json_objects(monkeypatch):
    captured = {}

    def fake_invoke(prompt, *args, **kwargs):
        captured["prompt"] = prompt
        return '{"facts":["用户要查退款"],"decisions":[],"open_loops":[],"focus":"退款"}'

    monkeypatch.setattr(llm_module, "_invoke_json", fake_invoke)
    result = llm_module.summarize_history(
        [{"role": "user", "content": "我要查退款"}],
        existing_summary=None,
        context=None,
    )

    assert result["facts"] == ["用户要查退款"]
    assert "已有摘要：{}" in captured["prompt"]
    assert "结构化会话状态：{}" in captured["prompt"]
