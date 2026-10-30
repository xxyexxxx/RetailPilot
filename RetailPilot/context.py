from __future__ import annotations

from copy import deepcopy
import json
import math
import os
from typing import Any

from .text import compact_text


CONTEXT_VERSION = "customer-context-v3"
CONTEXT_COMPACTION_TRIGGER_RATIO = 0.80
DEFAULT_CONTEXT_WINDOW_TOKENS = 128_000
STATE_EVENT_VERSION = 1


def append_state_event(
    context: dict[str, Any],
    event_type: str,
    data: dict[str, Any],
) -> None:
    """Append one immutable state-change event to the session event stream.

    Provider prefix caches key on a byte-exact prefix.  A mutable snapshot
    (``context["pre_sales"]["goal"] = ...``) rewrites bytes in the middle of
    the prompt every turn, so nothing after the first mutated field can be
    reused.  An append-only event stream instead grows strictly at the tail:
    turn N's stream is a byte prefix of turn N+1's, so the whole serialized
    history stays cacheable.

    Events are frozen (deep-copied, no later mutation) and carry a monotonic
    ``seq`` so the stream is verifiable and replayable.  The snapshot fields
    are still maintained alongside for readers that have not migrated.
    """

    events = context.setdefault("state_events", [])
    if not isinstance(events, list):
        events = []
        context["state_events"] = events
    events.append(
        {
            "seq": len(events),
            "v": STATE_EVENT_VERSION,
            "type": event_type,
            "data": deepcopy(data),
        }
    )


def normalize_state_events(raw: Any) -> list[dict[str, Any]]:
    """Repair a persisted event stream without ever reordering or dropping it."""

    if not isinstance(raw, list):
        return []
    events: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        data = item.get("data")
        events.append(
            {
                "seq": index,
                "v": int(item.get("v") or STATE_EVENT_VERSION),
                "type": str(item.get("type") or "unknown"),
                "data": dict(data) if isinstance(data, dict) else {},
            }
        )
    return events


def context_window_tokens() -> int:
    """Return the configured provider context capacity.

    The harness makes decisions in terms of occupancy ratio.  Tokenization is
    only an implementation detail needed to calculate that ratio, so the
    model-specific capacity is configured once rather than exposing token
    budgets to the rest of the application.
    """

    raw = os.getenv("LLM_CONTEXT_WINDOW", str(DEFAULT_CONTEXT_WINDOW_TOKENS))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_WINDOW_TOKENS


def _estimate_tokens(value: Any) -> int:
    """Conservative provider-independent estimate for serialized context.

    A real tokenizer is not available for every OpenAI-compatible gateway used
    by this project.  Counting serialized characters conservatively is enough
    for a preflight guard; the provider remains the final authority at call
    time.  The estimate deliberately errs high to avoid crossing the limit.
    """

    serialized = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    if not serialized:
        return 0
    ascii_chars = sum(char.isascii() for char in serialized)
    non_ascii_chars = len(serialized) - ascii_chars
    return max(1, math.ceil(ascii_chars / 3 + non_ascii_chars))


def _summary_list(value: Any, limit: int = 8, item_limit: int = 180) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = compact_text(item, item_limit)
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def normalize_history_summary(value: Any) -> dict[str, Any]:
    """Normalize the structured summary returned by the compaction LLM."""

    if isinstance(value, str):
        text = compact_text(value, 600)
        return {"version": 1, "facts": [text]} if text else {}
    if not isinstance(value, dict):
        return {}
    return {
        "version": 1,
        "facts": _summary_list(value.get("facts") or value.get("confirmed_facts")),
        "decisions": _summary_list(value.get("decisions")),
        "open_loops": _summary_list(
            value.get("open_loops") or value.get("open_questions")
        ),
        "focus": compact_text(
            value.get("focus") or value.get("current_focus") or "", 180
        ),
    }


def render_history_summary(value: Any) -> str:
    normalized = normalize_history_summary(value)
    if not normalized:
        return ""
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def _list(value: Any, limit: int = 20) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[:limit]


def normalize_customer_context(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Upgrade the legacy flat session payload to a skill-isolated state.

    Compatibility aliases are deliberately retained because query parsing is a
    separate bounded component. New code should read the namespaced fields.
    """

    source = deepcopy(raw or {})
    pre_sales = dict(source.get("pre_sales") or {})
    after_sales = dict(source.get("after_sales") or {})
    references = dict(source.get("references") or {})

    pre_sales.setdefault("goal", source.get("current_goal") or "")
    pre_sales.setdefault("hard_constraints", dict(source.get("hard_constraints") or {}))
    pre_sales.setdefault("soft_preferences", dict(source.get("soft_preferences") or {}))
    pre_sales.setdefault("last_parsed_query", source.get("last_parsed_query") or {})
    pre_sales.setdefault("last_result_skus", _list(source.get("last_result_skus")))
    pre_sales.setdefault("last_result_prices", _list(source.get("last_result_prices")))

    references.setdefault("last_result_skus", _list(pre_sales.get("last_result_skus")))
    references.setdefault("focused_skus", _list(source.get("focused_skus"), 4))
    references.setdefault("order_id", after_sales.get("order_id") or source.get("order_id") or "")
    references.setdefault("last_identifier", after_sales.get("last_identifier") or "")

    after_sales.setdefault("topic", "")
    after_sales.setdefault("last_intent", "")
    after_sales.setdefault("last_request", "")
    after_sales.setdefault("product_skus", _list(references.get("focused_skus"), 4))
    after_sales.setdefault("order_id", references.get("order_id") or "")
    after_sales.setdefault("domain", "")
    after_sales.setdefault("action", "")
    after_sales.setdefault("last_question", "")
    after_sales.setdefault("required_slots", [])
    after_sales.setdefault("last_identifier", references.get("last_identifier") or "")

    dialogue = dict(source.get("dialogue") or {})
    dialogue.setdefault("act", "")
    dialogue.setdefault("domain", after_sales.get("domain") or "")
    dialogue.setdefault("action", after_sales.get("action") or "")
    dialogue.setdefault("last_question", after_sales.get("last_question") or "")
    dialogue.setdefault("required_slots", _list(after_sales.get("required_slots"), 8))

    raw_summary = source.get("history_summary")
    history_summary = normalize_history_summary(raw_summary)
    if not history_summary and isinstance(raw_summary, str):
        # Preserve the old compact string format as a valid structured fact.
        history_summary = normalize_history_summary(raw_summary)

    result = {
        "context_version": CONTEXT_VERSION,
        "active_skill": str(source.get("active_skill") or "none"),
        "pre_sales": pre_sales,
        "after_sales": after_sales,
        "references": references,
        "dialogue": dialogue,
        "pending_slots": _list(source.get("pending_slots"), 8),
        "history_summary": history_summary,
        "history_summary_through_id": max(
            0, int(source.get("history_summary_through_id") or 0)
        ),
        # Append-only change log.  This is the cacheable view of session state;
        # the snapshot fields above are the mutable view kept for existing
        # readers.  Never reorder or rewrite this list.
        "state_events": normalize_state_events(source.get("state_events")),
    }
    return _with_compatibility_aliases(result)


def _with_compatibility_aliases(context: dict[str, Any]) -> dict[str, Any]:
    pre_sales = context.get("pre_sales") or {}
    references = context.get("references") or {}
    # These aliases keep parse_query and the web diagnostics stable while the
    # source of truth lives in pre_sales/references.
    context["last_parsed_query"] = dict(pre_sales.get("last_parsed_query") or {})
    context["last_result_skus"] = _list(
        references.get("last_result_skus") or pre_sales.get("last_result_skus")
    )
    context["last_result_prices"] = _list(pre_sales.get("last_result_prices"))
    return context


def context_for_skill(context: dict[str, Any], skill: str) -> dict[str, Any]:
    """Return the smallest context view that one skill is allowed to consume."""

    normalized = normalize_customer_context(context)
    shared = {
        "context_version": normalized["context_version"],
        "active_skill": normalized["active_skill"],
        "references": deepcopy(normalized["references"]),
        "dialogue": deepcopy(normalized["dialogue"]),
        "pending_slots": list(normalized["pending_slots"]),
    }
    if skill == "pre_sales":
        shared["pre_sales"] = deepcopy(normalized["pre_sales"])
        return _with_compatibility_aliases(shared)
    if skill == "after_sales":
        shared["after_sales"] = deepcopy(normalized["after_sales"])
        # After-sales may resolve "this product" but cannot consume search
        # ranking preferences as policy facts.
        shared["last_result_skus"] = list(normalized["last_result_skus"])
        return shared
    return shared


def bounded_history(
    messages: list[dict[str, Any]] | None,
    *,
    context: dict[str, Any] | None = None,
    current_message: str = "",
    context_window: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Build an append-only session view and report context occupancy.

    This function deliberately does not slide to a fixed number of messages.
    It returns the whole current session, optionally anchored by a previously
    persisted structured summary.  The compaction graph node decides what to
    summarize when ``needs_compaction`` is true.
    """

    normalized_context = normalize_customer_context(context or {})
    through_id = int(normalized_context.get("history_summary_through_id") or 0)
    selected: list[dict[str, str]] = []
    raw_messages = messages or []
    for index, item in enumerate(raw_messages):
        role = str(item.get("role") or "").strip()
        content = compact_text(item.get("content") or item.get("message"), 12000)
        if not role or not content:
            continue
        message_id = int(item.get("id") or 0)
        if through_id and (
            (message_id and message_id <= through_id)
            or (not message_id and index < through_id)
        ):
            continue
        selected.append({"role": role, "content": content})

    summary = normalized_context.get("history_summary")
    if summary:
        # The summary replaces everything up to ``history_summary_through_id``,
        # which are messages the ``continue`` above already dropped.  That is
        # what makes this position prefix-stable: the summary occupies the slot
        # of the messages it replaces (always earlier than any new message), so
        # it does not insert bytes ahead of retained history.  It is emitted
        # first only because nothing retained precedes it.
        selected.insert(
            0,
            {
                "role": "summary",
                "content": render_history_summary(summary),
            },
        )

    window = max(1, int(context_window or context_window_tokens()))
    occupancy_payload = {
        "history": selected,
        "current_message": current_message or "",
        "context": normalized_context,
    }
    estimated_tokens = _estimate_tokens(occupancy_payload)
    ratio = estimated_tokens / window
    total = len(raw_messages)
    selected_raw = sum(1 for item in selected if item.get("role") != "summary")
    history_chars = sum(len(item.get("content") or "") for item in selected)
    return selected, {
        "history_messages_total": total,
        "history_messages_selected": selected_raw,
        "history_chars": history_chars,
        # Backwards-compatible diagnostic field.  The control decision is the
        # ratio below; this legacy char ceiling is intentionally generous.
        "history_char_budget": window * 4,
        "history_truncated": bool(through_id),
        "history_summary_applied": bool(summary),
        "context_window_tokens": window,
        "context_usage_tokens": estimated_tokens,
        "context_usage_ratio": round(ratio, 6),
        "context_compaction_trigger_ratio": CONTEXT_COMPACTION_TRIGGER_RATIO,
        "needs_compaction": ratio >= CONTEXT_COMPACTION_TRIGGER_RATIO,
    }


def _constraints(parsed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    hard = {
        key: parsed.get(key)
        for key in (
            "category",
            "price_min",
            "price_max",
            "brands",
            "negative_brands",
            "must_have_groups",
            "negative_terms",
        )
        if parsed.get(key) not in (None, [], {})
    }
    soft = {
        key: parsed.get(key)
        for key in ("preferred_brands", "concepts", "sort")
        if parsed.get(key) not in (None, [], {}, "relevance")
    }
    return hard, soft


_ACCUMULATE_MARKERS = (
    "还要", "另外", "也", "加上", "再来", "以及", "顺便", "还有", "再加",
)


def _accumulate_marker(raw: str) -> bool:
    """Detect additive follow-ups like 还要/另外/也 which must merge, not replace."""
    return any(marker in raw for marker in _ACCUMULATE_MARKERS)


def _merge_constraints(prev: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Merge a new constraint batch into the previous one.

    List-valued fields (brands/concepts/negative_*) accumulate and de-duplicate;
    scalar fields keep the new value, falling back to the previous value so a
    partial follow-up (e.g. 只改价格) does not wipe unchanged constraints.
    """
    merged = dict(prev)
    for key, value in current.items():
        if isinstance(value, list):
            merged[key] = list(dict.fromkeys(_list(prev.get(key)) + _list(value)))[:20]
        elif value not in (None, "", [], {}):
            merged[key] = value
    return merged


def update_customer_context(
    current: dict[str, Any] | None,
    *,
    branch_outputs: list[dict[str, Any]],
    parsed_queries: list[dict[str, Any]],
    products: list[dict[str, Any]],
    pending: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist only structured, independently inspectable customer state.

    Two views are produced from the same turn:

    * the mutable snapshot (``pre_sales`` / ``dialogue`` / ...) that existing
      readers consume, and
    * an append-only ``state_events`` stream used for prompt construction.

    Only the event stream is prefix-stable across turns, which is what lets
    the provider prompt cache reuse the serialized session state.
    """

    context = normalize_customer_context(current)
    branches = {str(item.get("branch")) for item in branch_outputs if item.get("branch")}
    durable_branches = branches - {"chitchat"}
    # A greeting/acknowledgement is an interjection, not a skill switch.  Keep
    # the prior pre/after-sales state so the next terse customer fragment can
    # still be routed in context.
    if durable_branches:
        active_skill = (
            next(iter(durable_branches))
            if len(durable_branches) == 1
            else "mixed"
        )
        if active_skill != context["active_skill"]:
            append_state_event(context, "skill", {"active_skill": active_skill})
        context["active_skill"] = active_skill

    if products:
        skus = [str(item["sku"]) for item in products[:12]]
        prices = [float(item["price"]) for item in products[:12]]
        if skus != context["references"].get("last_result_skus"):
            append_state_event(
                context,
                "results",
                {"skus": skus, "prices": prices},
            )
        context["pre_sales"]["last_result_skus"] = skus
        context["pre_sales"]["last_result_prices"] = prices
        context["references"]["last_result_skus"] = skus

    if parsed_queries:
        parsed = dict(parsed_queries[-1])
        context["pre_sales"]["last_parsed_query"] = parsed
        hard, soft = _constraints(parsed)
        raw = str(parsed.get("raw_query") or "")
        prev_hard = dict(context["pre_sales"].get("hard_constraints") or {})
        prev_soft = dict(context["pre_sales"].get("soft_preferences") or {})

        # 换目标重置：本轮明确换类目时，旧目标的品牌/价格/属性约束不再适用，
        # 只保留新类目，避免把"手机"的预算误套到"耳机"上。
        new_category = parsed.get("category")
        old_category = prev_hard.get("category")
        category_source = str((parsed.get("sources") or {}).get("category") or "")
        topic_switched = bool(
            new_category
            and old_category
            and new_category != old_category
            and category_source.startswith(("query", "llm"))
        )
        if topic_switched:
            hard = {key: value for key, value in hard.items() if key == "category"}
            soft = {}

        # 累加语义：本轮带"还要/另外/也"时，上一轮同类约束保留并合并，而不是被覆盖。
        if _accumulate_marker(raw) and not topic_switched:
            hard = _merge_constraints(prev_hard, hard)
            soft = _merge_constraints(prev_soft, soft)

        if hard != context["pre_sales"].get("hard_constraints") or soft != context[
            "pre_sales"
        ].get("soft_preferences"):
            append_state_event(context, "constraints", {"hard": hard, "soft": soft})
        context["pre_sales"]["hard_constraints"] = hard
        context["pre_sales"]["soft_preferences"] = soft
        goal_terms = parsed.get("grounded_target_terms") or []
        new_goal = " ".join(str(item) for item in goal_terms[:4])
        # 历史摘要：目标变化时把旧目标归档，供长对话补全与漂移判断。
        # The archive is appended (never a head insert) so the prompt prefix
        # stays stable; the derived snapshot below keeps last-8 for readers.
        previous_goal = context["pre_sales"].get("goal") or ""
        if new_goal and previous_goal and new_goal != previous_goal:
            append_state_event(context, "goal_archive", {"goal": previous_goal})
            summary = normalize_history_summary(context.get("history_summary"))
            facts = list(summary.get("facts") or [])
            marker = f"此前目标：{compact_text(previous_goal, 160)}"
            if marker not in facts:
                facts.append(marker)
            summary["facts"] = facts[-8:]
            context["history_summary"] = summary
        if new_goal != (context["pre_sales"].get("goal") or ""):
            append_state_event(context, "goal", {"goal": new_goal})
        context["pre_sales"]["goal"] = new_goal

    for output in branch_outputs:
        frame = dict(output.get("dialogue_frame") or {})
        if frame:
            dialogue_event: dict[str, Any] = {"act": str(frame.get("act") or "")}
            context["dialogue"]["act"] = str(frame.get("act") or "")
            if output.get("branch") == "chitchat" and frame.get("act") in {
                "closing", "stop"
            }:
                if context["active_skill"] != "none":
                    append_state_event(context, "skill", {"active_skill": "none"})
                context["active_skill"] = "none"
            interjection = output.get("branch") == "chitchat" and frame.get("act") in {
                "greeting", "acknowledgement"
            }
            if not interjection:
                context["dialogue"]["domain"] = str(frame.get("domain") or "")
                context["dialogue"]["action"] = str(frame.get("action") or "")
                requested_slots = [
                    str(item)
                    for item in _list(frame.get("requested_slots"), 8)
                    if item
                ]
                if requested_slots:
                    context["dialogue"]["requested_slots"] = requested_slots
                required_slots = [
                    str(item)
                    for item in _list(
                        output.get("required_slots") or frame.get("required_slots"), 8
                    )
                    if item
                ]
                context["dialogue"]["required_slots"] = required_slots
                response = compact_text(output.get("response"), 240)
                last_question = compact_text(frame.get("last_question"), 240)
                if not last_question and required_slots and response:
                    last_question = response
                if last_question:
                    last_question = compact_text(last_question, 240)
                    context["dialogue"]["last_question"] = last_question
                    dialogue_event["last_question"] = last_question
                identifier = compact_text(frame.get("identifier"), 100)
                if identifier:
                    context["references"]["last_identifier"] = identifier
                    dialogue_event["identifier"] = identifier
                dialogue_event["domain"] = str(frame.get("domain") or "")
                dialogue_event["action"] = str(frame.get("action") or "")
                if requested_slots:
                    dialogue_event["requested_slots"] = requested_slots
                dialogue_event["required_slots"] = required_slots
            append_state_event(context, "dialogue", dialogue_event)

        # 用户指代"第一款/它/这个"由意图 LLM 识别，本地序数词规则已删除。
        # 聚焦商品改为在 detail 分支执行后写入：用户刚看过的那一件就是聚焦商品，
        # 供后续指代与售后定位使用。
        if output.get("intent") == "detail" and output.get("products"):
            first_sku = str((output["products"][0] or {}).get("sku") or "")
            if first_sku and [first_sku] != context["references"].get("focused_skus"):
                append_state_event(context, "focus", {"focused_skus": [first_sku]})
                context["references"]["focused_skus"] = [first_sku]

        if output.get("branch") != "after_sales":
            continue
        policy = output.get("policy") or {}
        order = output.get("order") or {}
        after_event: dict[str, Any] = {
            "intent": str(output.get("intent") or ""),
            "request": compact_text(output.get("task_text") or "", 240),
        }
        context["after_sales"]["last_intent"] = str(output.get("intent") or "")
        context["after_sales"]["last_request"] = compact_text(
            output.get("task_text") or "", 240
        )
        if frame:
            context["after_sales"]["domain"] = str(frame.get("domain") or "")
            context["after_sales"]["action"] = str(frame.get("action") or "")
            context["after_sales"]["required_slots"] = list(
                context["dialogue"]["required_slots"]
            )
            after_event["domain"] = str(frame.get("domain") or "")
            after_event["action"] = str(frame.get("action") or "")
            if last_question:
                context["after_sales"]["last_question"] = compact_text(
                    last_question, 240
                )
                after_event["last_question"] = compact_text(last_question, 240)
            identifier = compact_text(frame.get("identifier"), 100)
            if identifier:
                context["after_sales"]["last_identifier"] = identifier
                after_event["identifier"] = identifier
        if policy.get("topic"):
            context["after_sales"]["topic"] = policy["topic"]
            after_event["topic"] = policy["topic"]
        product_skus = _list(policy.get("product_skus"), 4)
        if product_skus:
            context["after_sales"]["product_skus"] = product_skus
            context["references"]["focused_skus"] = product_skus
            after_event["product_skus"] = product_skus
        if order.get("order_id"):
            context["after_sales"]["order_id"] = order["order_id"]
            context["references"]["order_id"] = order["order_id"]
            after_event["order_id"] = order["order_id"]
        append_state_event(context, "after_sales", after_event)

    clarification_slot = str((pending or {}).get("slot") or "")
    if clarification_slot:
        context["dialogue"]["required_slots"] = [clarification_slot]
    context["pending_slots"] = (
        [clarification_slot]
        if clarification_slot
        else list(context["dialogue"].get("required_slots") or [])
    )
    return _with_compatibility_aliases(context)
