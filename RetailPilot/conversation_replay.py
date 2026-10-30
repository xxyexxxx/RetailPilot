from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


REPLAY_VERSION = "conversation-shadow-replay-v3"
SCOPE_GATE_VERSION = "project-scope-gate-v1"
LLM_SCOPE_GATE_VERSION = "project-scope-llm-v1"


def _trusted_preclassified_scope(turn: dict[str, Any]) -> dict[str, Any] | None:
    """Return an upstream LLM scope verdict only when its contract is complete.

    The LLM-scope corpus is already split at excluded turns. Re-running the
    older lexical gate here would silently change that audited corpus before
    Agent replay and make the later badcase denominator inconsistent.
    """

    scope = turn.get("scope_gate")
    if not isinstance(scope, dict):
        return None
    if scope.get("scope_version") != LLM_SCOPE_GATE_VERSION:
        return None
    if scope.get("classifier_mode") != "llm" or scope.get("in_scope") is not True:
        return None
    if scope.get("domain") not in {"pre_sales", "after_sales_rag", "order_read"}:
        return None
    return dict(scope)


_SCOPE_THIRD_PARTY_MARKERS = ("火车票", "机票", "酒店", "打车", "外卖", "话费", "游戏充值")
_SCOPE_ACCOUNT_MARKERS = ("额度", "白条", "账户冻结", "账号冻结", "对公转账", "银行卡")
_SCOPE_UI_SUPPORT_MARKERS = (
    "购物车白屏", "购物车白底", "购物车进不去", "购物车打不开",
    "短信提醒", "评价提交不上", "评价不了",
)
_SCOPE_PRICE_PROTECTION_MARKERS = ("价保", "保价", "降价补差", "补差申请", "退差价")
_SCOPE_TRANSACTION_MARKERS = (
    "帮我下单", "直接下单", "加入购物车", "加购物车", "替我付款", "帮我付款",
    "修改支付", "改支付方式", "库存锁定",
)
_SCOPE_SMALLTALK_ONLY = re.compile(
    r"^(?:你?好|您好|在吗|哈喽|hello|hi|闲聊|你是什么模型|今天星期几|谢谢|再见|拜拜|"
    r"我草你妈的|操你妈|傻逼)[啊呀哦吧哈\s,.，。!！?？]*$",
    re.IGNORECASE,
)
_SCOPE_ACK_ONLY = re.compile(
    r"^(?:是|是的|对|对的|嗯+|哦+|好|好的|可以|行|ok|okay|知道了|明白了)"
    r"[啊呀哦吧哈\s,.，。!！?？]*$",
    re.IGNORECASE,
)
_SCOPE_PRE_SALES_ACTIONS = (
    "推荐", "想买", "要买", "购买建议", "怎么选", "选购", "对比", "比较", "区别",
    "哪个好", "哪款", "详情", "参数", "价格", "多少钱", "便宜一点", "贵一点",
)
_SCOPE_PRODUCT_MARKERS = (
    "商品", "电脑", "笔记本", "手机", "耳机", "键盘", "鼠标", "显示器", "相机",
    "书", "书籍", "笔", "护肤", "化妆品", "衣服", "鞋", "水果", "食品", "饮料",
    "家电", "冰箱", "空调", "洗衣机", "平板", "手表", "音箱", "充电器", "路由器",
)
_SCOPE_POLICY_MARKERS = (
    "退货", "换货", "退款", "退换货", "七天无理由", "保修", "维修", "质保",
    "发票", "开票", "运费", "配送规则", "质量问题", "破损", "少件", "漏发",
)
_SCOPE_ORDER_READ_MARKERS = (
    "订单", "物流", "快递", "发货", "到哪", "什么时候到", "签收", "派送", "配送进度",
    "订单状态", "订单进度", "完整物流", "买的什么", "买了什么",
)
_SCOPE_CONTEXT_FOLLOWUPS = (
    "这个", "这款", "第一款", "第二款", "前两款", "刚才的", "再便宜", "再贵",
    "还有吗", "能退吗", "怎么办", "怎么处理", "为什么", "然后呢", "下一步",
    "流程", "哪里", "多久", "快点", "没开", "没到", "还没",
)
_SCOPE_PRE_SALES_FOLLOWUPS = (
    "再", "更", "最好", "预算", "以内", "以下", "左右", "便宜", "贵", "无线", "有线",
    "静音", "降噪", "颜色", "品牌", "尺寸", "容量", "不要", "排除", "换一个", "其他款",
)
_SCOPE_ORDER_ID = re.compile(
    r"\[(?:orderid|订单id|订单号)_[^\]]+\]|\b(?:jd|a)[a-z0-9_-]{5,24}\b",
    re.IGNORECASE,
)


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalized_reply(value: Any) -> str:
    return re.sub(r"[\W_]+", "", _clean_text(value).casefold())


def project_scope_gate(message: Any, *, active_domain: str | None = None) -> dict[str, Any]:
    """Classify replay eligibility against the project's declared product scope.

    The gate intentionally does not consume the Agent's predicted intent or tool
    trace. Otherwise a routing bug could incorrectly decide whether that same
    turn is evaluated. Only explicit query evidence and a separately maintained
    in-scope conversation frame are used.
    """

    text = _clean_text(message)
    lowered = text.casefold()
    base = {"scope_version": SCOPE_GATE_VERSION, "message": text[:500]}
    if not text:
        return {**base, "in_scope": False, "domain": None, "reason": "empty_query", "inherited": False}

    filtered_groups = (
        ("third_party_service", _SCOPE_THIRD_PARTY_MARKERS),
        ("account_or_credit", _SCOPE_ACCOUNT_MARKERS),
        ("commerce_ui_support", _SCOPE_UI_SUPPORT_MARKERS),
        ("price_protection", _SCOPE_PRICE_PROTECTION_MARKERS),
        ("transaction_execution", _SCOPE_TRANSACTION_MARKERS),
    )
    for reason, markers in filtered_groups:
        if any(marker in lowered for marker in markers):
            return {**base, "in_scope": False, "domain": None, "reason": reason, "inherited": False}
    if "降价" in lowered and any(marker in lowered for marker in ("买了", "买的", "下单", "前几天")):
        return {**base, "in_scope": False, "domain": None, "reason": "price_protection", "inherited": False}

    if "忽略之前" in lowered and ("提示" in lowered or "指令" in lowered):
        return {**base, "in_scope": False, "domain": None, "reason": "prompt_injection", "inherited": False}

    if _SCOPE_ORDER_ID.search(text) or any(marker in lowered for marker in _SCOPE_ORDER_READ_MARKERS):
        return {**base, "in_scope": True, "domain": "order_read", "reason": "explicit_order_or_logistics", "inherited": False}
    if any(marker in lowered for marker in _SCOPE_POLICY_MARKERS):
        return {**base, "in_scope": True, "domain": "after_sales_rag", "reason": "explicit_after_sales_topic", "inherited": False}

    pre_sales_action = any(marker in lowered for marker in _SCOPE_PRE_SALES_ACTIONS)
    product_evidence = any(marker in lowered for marker in _SCOPE_PRODUCT_MARKERS)
    if product_evidence or pre_sales_action and re.search(r"[\u4e00-\u9fffA-Za-z0-9]", text):
        return {**base, "in_scope": True, "domain": "pre_sales", "reason": "explicit_product_or_shopping_need", "inherited": False}

    contextual = bool(
        _SCOPE_ACK_ONLY.fullmatch(text)
        or any(marker in lowered for marker in _SCOPE_CONTEXT_FOLLOWUPS)
        or active_domain == "pre_sales"
        and any(marker in lowered for marker in _SCOPE_PRE_SALES_FOLLOWUPS)
        or re.fullmatch(r"(?:它|有吗|不是|不行|没有|其他.{0,6}|\[[^\]]+\]|\d{4,})", text)
    )
    if active_domain in {"pre_sales", "after_sales_rag", "order_read"} and contextual:
        return {
            **base,
            "in_scope": True,
            "domain": active_domain,
            "reason": "contextual_followup_of_in_scope_frame",
            "inherited": True,
        }

    if _SCOPE_SMALLTALK_ONLY.fullmatch(text):
        return {**base, "in_scope": False, "domain": None, "reason": "smalltalk_or_meta", "inherited": False}
    return {
        **base,
        "in_scope": False,
        "domain": None,
        "reason": "insufficient_in_scope_evidence",
        "inherited": False,
    }


def _source_satisfaction(value: str) -> list[int]:
    scores: list[int] = []
    for item in str(value or "").split(","):
        try:
            scores.append(int(item))
        except ValueError:
            continue
    return scores


def iter_jddc_uss(path: Path | str) -> Iterator[dict[str, Any]]:
    """Parse the USS JDDC.txt format without loading the corpus into memory."""

    source = Path(path)
    session: list[dict[str, Any]] = []
    session_index = 0

    def materialize() -> dict[str, Any] | None:
        nonlocal session_index
        if not session:
            return None
        session_index += 1
        return {
            "conversation_id": f"jddc-uss-{session_index:06d}",
            "source": "JDDC-derived USS",
            "turns": list(session),
            "metadata": {
                "source_path": source.name,
                "shadow_replay": True,
                "human_agent_responses_are_reference_only": True,
            },
        }

    with source.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                record = materialize()
                if record:
                    yield record
                session = []
                continue
            columns = line.split("\t")
            role = _clean_text(columns[0]).upper() if columns else ""
            text = _clean_text(columns[1]) if len(columns) > 1 else ""
            if role not in {"USER", "SYSTEM"} or not text:
                continue
            # USS appends an OVERALL rating row as USER; it is annotation, not a query.
            if role == "USER" and text.upper() == "OVERALL":
                continue
            session.append(
                {
                    "role": "user" if role == "USER" else "assistant_reference",
                    "content": text,
                    "source_action": _clean_text(columns[2]) if len(columns) > 2 else "",
                    "source_satisfaction": (
                        _source_satisfaction(columns[3]) if len(columns) > 3 else []
                    ),
                    "source_explanation": _clean_text(columns[4]) if len(columns) > 4 else "",
                }
            )
    record = materialize()
    if record:
        yield record


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    """Read generic one-conversation-per-line JSONL replay input."""

    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            turns = payload.get("turns") or payload.get("messages") or []
            normalized_turns = []
            for turn in turns:
                role = str(turn.get("role") or turn.get("speaker") or "").casefold()
                content = _clean_text(
                    turn.get("content") or turn.get("text") or turn.get("utterance")
                )
                if not content:
                    continue
                normalized_turns.append(
                    {
                        **turn,
                        "role": (
                            "user"
                            if role in {"user", "customer", "q"}
                            else "assistant_reference"
                        ),
                        "content": content,
                    }
                )
            yield {
                "conversation_id": str(
                    payload.get("conversation_id")
                    or payload.get("dialogue_id")
                    or payload.get("session_id")
                    or f"jsonl-{index:06d}"
                ),
                "source": str(payload.get("source") or "custom-jsonl"),
                "turns": normalized_turns,
                "metadata": payload.get("metadata") or {},
            }


def load_conversations(
    path: Path | str,
    *,
    input_format: str,
    limit: int | None = None,
    min_user_turns: int = 2,
) -> list[dict[str, Any]]:
    iterator: Iterable[dict[str, Any]]
    if input_format == "jddc-uss":
        iterator = iter_jddc_uss(path)
    elif input_format == "jsonl":
        iterator = iter_jsonl(path)
    else:
        raise ValueError(f"unsupported input format: {input_format}")
    selected: list[dict[str, Any]] = []
    for conversation in iterator:
        user_turns = sum(1 for turn in conversation["turns"] if turn["role"] == "user")
        if user_turns < min_user_turns:
            continue
        selected.append(conversation)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def filter_conversations_by_project_scope(
    conversations: list[dict[str, Any]],
    *,
    max_user_turns: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Materialize a replay-only corpus containing contiguous in-scope segments.

    A filtered turn closes the current segment. This prevents an unrelated or
    unevaluable query from silently connecting two business turns and changing
    the Agent's session context during the later replay.
    """

    eligible: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    scope_versions: Counter[str] = Counter()
    original_conversations_with_scope: set[str] = set()
    total_user_turns = 0
    eligible_user_turns = 0

    for conversation in conversations:
        original_id = str(conversation["conversation_id"])
        active_domain: str | None = None
        segment: list[dict[str, Any]] = []
        segment_index = 0
        user_turn_index = 0
        keep_following_reference = False

        def flush_segment() -> None:
            nonlocal segment, segment_index
            if not segment:
                return
            segment_index += 1
            eligible.append(
                {
                    "conversation_id": f"{original_id}::scope-{segment_index:03d}",
                    "source": conversation.get("source") or "",
                    "turns": segment,
                    "metadata": {
                        **(conversation.get("metadata") or {}),
                        "scope_filtered_corpus": True,
                        "scope_gate_version": (
                            (conversation.get("metadata") or {}).get(
                                "scope_gate_version"
                            )
                            or SCOPE_GATE_VERSION
                        ),
                        "original_conversation_id": original_id,
                        "scope_segment_index": segment_index,
                    },
                }
            )
            segment = []

        for source_turn in conversation.get("turns") or []:
            if source_turn.get("role") != "user":
                if keep_following_reference and segment:
                    segment.append(dict(source_turn))
                continue
            user_turn_index += 1
            if max_user_turns is not None and user_turn_index > max_user_turns:
                break
            total_user_turns += 1
            scope = _trusted_preclassified_scope(source_turn) or project_scope_gate(
                source_turn.get("content"), active_domain=active_domain
            )
            if scope["in_scope"]:
                scope_versions[str(scope.get("scope_version") or SCOPE_GATE_VERSION)] += 1
                active_domain = str(scope["domain"])
                eligible_user_turns += 1
                original_conversations_with_scope.add(original_id)
                domain_counts[active_domain] += 1
                segment.append(
                    {
                        **source_turn,
                        "source_turn_index": user_turn_index,
                        "scope_gate": scope,
                    }
                )
                keep_following_reference = True
                continue

            flush_segment()
            active_domain = None
            keep_following_reference = False
            reason = str(scope.get("reason") or "unknown")
            reason_counts[reason] += 1
            filtered.append(
                {
                    "conversation_id": original_id,
                    "turn_index": user_turn_index,
                    "message": source_turn.get("content"),
                    "reason": reason,
                    "scope_version": SCOPE_GATE_VERSION,
                    "excluded_from_agent_replay": True,
                    "excluded_from_llm_metrics": True,
                    "excluded_from_badcases": True,
                }
            )
        flush_segment()

    summary = {
        "scope_gate_version": (
            next(iter(scope_versions))
            if len(scope_versions) == 1
            else "mixed"
        ),
        "scope_gate_versions": dict(scope_versions),
        "source_conversations": len(conversations),
        "source_user_turns": total_user_turns,
        "eligible_original_conversations": len(original_conversations_with_scope),
        "eligible_segments": len(eligible),
        "eligible_user_turns": eligible_user_turns,
        "scope_filtered_turns": len(filtered),
        "eligible_rate": round(eligible_user_turns / max(1, total_user_turns), 4),
        "eligible_domain_counts": dict(domain_counts.most_common()),
        "scope_filter_counts": dict(reason_counts.most_common()),
    }
    return eligible, filtered, summary


def judge_selection(
    conversations: list[dict[str, Any]],
    *,
    judge_limit: int,
    seed: int,
    max_user_turns: int | None = None,
) -> set[tuple[str, int]]:
    candidates: list[tuple[str, int, str]] = []
    for conversation in conversations:
        user_index = 0
        active_domain: str | None = None
        for turn in conversation["turns"]:
            if turn["role"] != "user":
                continue
            user_index += 1
            if max_user_turns is not None and user_index > max_user_turns:
                break
            scope = _trusted_preclassified_scope(turn) or project_scope_gate(
                turn.get("content"), active_domain=active_domain
            )
            if not scope["in_scope"]:
                active_domain = None
                continue
            active_domain = str(scope["domain"])
            key = f"{seed}:{conversation['conversation_id']}:{user_index}"
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            candidates.append((conversation["conversation_id"], user_index, digest))
    candidates.sort(key=lambda item: item[2])
    if judge_limit <= 0:
        return {(conversation_id, turn_index) for conversation_id, turn_index, _ in candidates}
    return {
        (conversation_id, turn_index)
        for conversation_id, turn_index, _ in candidates[:judge_limit]
    }


def apply_project_scope(records: list[dict[str, Any]]) -> None:
    """Attach an auditable scope verdict to new or legacy replay records."""

    for conversation in records:
        active_domain: str | None = None
        for turn in conversation.get("turns") or []:
            scope = _trusted_preclassified_scope(turn) or project_scope_gate(
                turn.get("message"), active_domain=active_domain
            )
            turn["scope_gate"] = scope
            turn.setdefault("evaluation_evidence", {})["scope_gate"] = scope
            if scope["in_scope"]:
                active_domain = str(scope["domain"])
            else:
                active_domain = None


def scope_filtered_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for conversation in records:
        for turn in conversation.get("turns") or []:
            scope = turn.get("scope_gate") or {}
            if scope.get("in_scope") is not False:
                continue
            output.append(
                {
                    "conversation_id": conversation.get("conversation_id"),
                    "turn_index": turn.get("turn_index"),
                    "message": turn.get("message"),
                    "reason": scope.get("reason"),
                    "scope_version": scope.get("scope_version"),
                    "excluded_from_llm_metrics": True,
                    "excluded_from_badcases": True,
                }
            )
    return output


def grade_replay_turn(
    result: dict[str, Any], *, previous_reply: str = "",
    previous_message: str = "", current_message: str = "",
) -> tuple[bool, list[str]]:
    detail = grade_replay_turn_detailed(
        result,
        previous_reply=previous_reply,
        previous_message=previous_message,
        current_message=current_message,
    )
    return bool(detail["pass"]), list(detail["failure_types"])


def grade_replay_turn_detailed(
    result: dict[str, Any], *, previous_reply: str = "",
    previous_message: str = "", current_message: str = "",
) -> dict[str, Any]:
    """Return deterministic hard-check verdict plus auditable evidence.

    These checks are program assertions, not LLM opinions. Keeping their evidence
    beside every replay turn makes a badcase reproducible without re-running a
    model or reading the full Agent trace.
    """

    failures: list[str] = []
    checks: list[dict[str, Any]] = []
    reply = _clean_text(result.get("reply"))
    reply_present = bool(reply)
    checks.append(
        {
            "name": "non_empty_reply",
            "pass": reply_present,
            "evidence": {"reply_chars": len(reply)},
        }
    )
    if not reply_present:
        failures.append("empty_reply")

    same_customer_input = bool(previous_message and current_message) and (
        _normalized_reply(previous_message) == _normalized_reply(current_message)
    )
    duplicate_reply = (
        bool(previous_reply)
        and not same_customer_input
        and _normalized_reply(reply) == _normalized_reply(previous_reply)
    )
    checks.append(
        {
            "name": "not_duplicate_previous_reply",
            "pass": not duplicate_reply,
            "applicable": bool(previous_reply),
            "evidence": {
                "previous_reply_chars": len(_clean_text(previous_reply)),
                "normalized_equal": duplicate_reply,
                "same_customer_input": same_customer_input,
            },
        }
    )
    if duplicate_reply:
        failures.append("duplicate_reply")

    task_results = result.get("task_results") or []
    unique_tasks = {
        (item.get("intent"), item.get("task_text")) for item in task_results
    }
    duplicate_task = len(unique_tasks) != len(task_results)
    checks.append(
        {
            "name": "unique_agent_tasks",
            "pass": not duplicate_task,
            "evidence": {
                "task_count": len(task_results),
                "unique_task_count": len(unique_tasks),
            },
        }
    )
    if duplicate_task:
        failures.append("duplicate_task")

    grounding = result.get("grounding") or {}
    grounding_pass = not grounding or bool(grounding.get("pass"))
    checks.append(
        {
            "name": "grounding_contract",
            "pass": grounding_pass,
            "applicable": bool(grounding),
            "evidence": {
                "grounding_present": bool(grounding),
                "reported_pass": grounding.get("pass") if grounding else None,
                "reported_failure_types": list(
                    grounding.get("failure_types") or []
                ),
            },
        }
    )
    if not grounding_pass:
        failures.extend(
            str(item) for item in grounding.get("failure_types") or ["grounding_failure"]
        )
    unique_failures = list(dict.fromkeys(failures))
    return {
        "checker": "program_hard_check",
        "pass": not unique_failures,
        "failure_types": unique_failures,
        "checks": checks,
    }


def llm_judge_evidence(
    verdict: dict[str, Any] | None,
    *,
    selected: bool,
    requested: bool,
) -> dict[str, Any]:
    """Normalize a sidecar verdict without presenting fallback as LLM output."""

    payload = verdict or {}
    mode = str(payload.get("judge_mode") or "unknown")
    executed = mode == "llm"
    evaluable = payload.get("evaluable") is not False
    score_names = (
        "intent_fit",
        "requirement_coverage",
        "context_use",
        "relevance",
        "concise_and_non_repetitive",
    )
    evidence: dict[str, Any] = {
        "selected": bool(selected),
        "requested": bool(requested),
        "executed": executed,
        "evaluable": evaluable if executed else None,
        "counted_as_llm_judge": executed,
        "mode": mode,
        # A deterministic/fallback verdict may be useful diagnostically, but it
        # must never be surfaced as an LLM pass/fail result.
        "pass": payload.get("pass") if executed and evaluable else None,
        "failure_type": payload.get("failure_type") if executed else None,
        "summary": payload.get("summary") if executed else None,
        "scores": (
            {name: payload.get(name) for name in score_names if name in payload}
            if executed
            else {}
        ),
    }
    if mode == "deterministic_fallback":
        evidence["fallback_reason"] = payload.get("fallback_reason") or "unknown"
        evidence["fallback_diagnostic"] = {
            "pass": payload.get("pass"),
            "failure_type": payload.get("failure_type"),
            "summary": payload.get("summary"),
        }
    elif mode == "deterministic":
        evidence["deterministic_diagnostic"] = {
            "pass": payload.get("pass"),
            "failure_type": payload.get("failure_type"),
            "summary": payload.get("summary"),
        }
    return evidence


def replay_turn_is_badcase(turn: dict[str, Any]) -> bool:
    """A badcase is a hard-check failure or a *real* LLM Judge failure."""

    if (turn.get("scope_gate") or {}).get("in_scope") is False:
        return False
    if not turn.get("hard_pass"):
        return True
    semantic = turn.get("semantic_judge") or {}
    return (
        semantic.get("judge_mode") == "llm"
        and semantic.get("evaluable") is not False
        and semantic.get("pass") is False
    )


def replay_conversation_is_badcase(record: dict[str, Any]) -> bool:
    return any(replay_turn_is_badcase(turn) for turn in record.get("turns") or [])


def _intent_sources(turn: dict[str, Any]) -> list[str]:
    sources: list[str] = []
    for event in turn.get("trace") or []:
        if event.get("node") != "intent_analyzer":
            continue
        for task in event.get("tasks") or []:
            source = str(task.get("source") or "unknown")
            if source not in sources:
                sources.append(source)
    return sources or ["unknown"]


def _current_business_signal(message: Any) -> str:
    text = _clean_text(message).casefold()
    groups = (
        ("human_service_request", ("转人工", "人工客服", "人工服务", "转客服", "店铺客服")),
        ("close_or_stop", ("再见", "拜拜", "不需要了", "不用了", "不要处理", "放弃了", "没有了")),
        ("acknowledgement", ("好的", "是的", "对的", "知道了", "明白了")),
        ("invoice", ("发票", "开票", "重开", "补开")),
        ("refund_or_return", ("退款", "退货", "换货", "拒签", "取件")),
        ("repair", ("维修", "保修", "坏了", "不好使", "不能用", "售后点", "维修点")),
        ("delivery", ("物流", "快递", "配送", "派送", "发货", "到货", "送到", "超时")),
        ("account_or_payment", ("额度", "白条", "账户", "支付", "转账")),
        ("price_protection", ("价保", "保价", "降价", "差价")),
        ("page_or_setting", ("购物车", "白屏", "白底", "短信提醒", "评价提交")),
        ("catalog", ("推荐", "想买", "选购", "搜", "找一款", "有没有")),
    )
    for signal, markers in groups:
        if any(marker in text for marker in markers):
            return signal
    if re.fullmatch(r"[\W_]*(?:嗯+|哦+|好|是|对|ok)[\W_]*", text, re.IGNORECASE):
        return "acknowledgement"
    if re.search(r"\[(?:orderid|订单号|数字)[^\]]*\]|\b(?:jd|a)\d{5,16}\b", text):
        return "identifier"
    return "unclassified_fragment"


def _response_template(reply: Any) -> str:
    text = _clean_text(reply)
    templates = (
        ("catalog_clarification", "我还不能确定具体商品类型"),
        ("catalog_recommendation", "为您筛出"),
        ("order_not_found", "没有查到对应订单"),
        ("order_not_found_repeat", "这次仍未查到"),
        ("generic_context_prompt", "您希望我据此帮您查订单"),
        ("generic_ack", "好的，已了解"),
        ("policy_no_evidence", "当前知识库没有找到"),
        ("human_review", "人工审核"),
        ("capability_boundary", "当前工具不能"),
        ("policy_evidence", "来源："),
    )
    for name, marker in templates:
        if marker in text:
            return name
    normalized = re.sub(r"\[[^\]]+\]|\d+(?:\.\d+)?", "<value>", text)
    normalized = re.sub(r"\s+", "", normalized)
    return f"other:{normalized[:36]}" if normalized else "empty"


def badcase_turns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten failed turns with compact evidence for triage and code fixes."""

    output: list[dict[str, Any]] = []
    for conversation in records:
        for turn in conversation.get("turns") or []:
            if not replay_turn_is_badcase(turn):
                continue
            semantic = turn.get("semantic_judge") or {}
            llm_failed = (
                semantic.get("judge_mode") == "llm"
                and semantic.get("evaluable") is not False
                and semantic.get("pass") is False
            )
            output.append(
                {
                    "conversation_id": conversation.get("conversation_id"),
                    "turn_index": turn.get("turn_index"),
                    "message": turn.get("message"),
                    "reply": turn.get("reply"),
                    "intents": turn.get("intents") or [],
                    "tools": turn.get("tools") or [],
                    "intent_sources": _intent_sources(turn),
                    "current_business_signal": _current_business_signal(turn.get("message")),
                    "tool_signature": "->".join(str(item) for item in turn.get("tools") or []) or "no_tool",
                    "response_template": _response_template(turn.get("reply")),
                    "hard_failure_types": turn.get("hard_failure_types") or [],
                    "llm_failure_type": (
                        semantic.get("failure_type") if llm_failed else None
                    ),
                    "llm_judge_mode": semantic.get("judge_mode") or "unknown",
                    "llm_judge_summary": semantic.get("summary") if llm_failed else None,
                    "evaluation_evidence": turn.get("evaluation_evidence") or {},
                    "source_context_tail": turn.get("source_context_tail") or [],
                }
            )
    return output


def cluster_badcases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster failures by evaluator plus reproducible Harness signature."""

    clusters: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in badcase_turns(records):
        failure_keys = [
            ("hard_check", str(failure))
            for failure in item.get("hard_failure_types") or []
        ]
        if item.get("llm_failure_type"):
            failure_keys.append(("llm_judge", str(item["llm_failure_type"])))
        for source, failure_type in failure_keys:
            routing_signature = (
                f"signal={item['current_business_signal']}|"
                f"intent={','.join(item['intents']) or 'none'}|"
                f"source={','.join(item['intent_sources'])}|"
                f"tools={item['tool_signature']}|"
                f"response={item['response_template']}"
            )
            key = (source, failure_type, routing_signature)
            cluster = clusters.setdefault(
                key,
                {
                    "evaluator": source,
                    "failure_type": failure_type,
                    "routing_signature": routing_signature,
                    "current_business_signal": item["current_business_signal"],
                    "intents": item["intents"],
                    "intent_sources": item["intent_sources"],
                    "tool_signature": item["tool_signature"],
                    "response_template": item["response_template"],
                    "turn_count": 0,
                    "conversation_ids": set(),
                    "examples": [],
                },
            )
            cluster["turn_count"] += 1
            cluster["conversation_ids"].add(str(item["conversation_id"]))
            if len(cluster["examples"]) < 5:
                cluster["examples"].append(
                    {
                        "conversation_id": item["conversation_id"],
                        "turn_index": item["turn_index"],
                        "message": item["message"],
                        "reply": item["reply"],
                        "hard_failure_types": item["hard_failure_types"],
                        "llm_judge_summary": item["llm_judge_summary"],
                        "routing_signature": routing_signature,
                        "evaluation_evidence": item["evaluation_evidence"],
                    }
                )
    output: list[dict[str, Any]] = []
    for cluster in clusters.values():
        conversation_ids = sorted(cluster.pop("conversation_ids"))
        cluster["conversation_count"] = len(conversation_ids)
        cluster["conversation_ids"] = conversation_ids
        output.append(cluster)
    return sorted(
        output,
        key=lambda item: (-item["turn_count"], item["evaluator"], item["failure_type"]),
    )


def summarize_replay(records: list[dict[str, Any]]) -> dict[str, Any]:
    hard_failures: Counter[str] = Counter()
    judge_failures: Counter[str] = Counter()
    judge_modes: Counter[str] = Counter()
    intents: Counter[str] = Counter()
    tool_calls: Counter[str] = Counter()
    turns = 0
    in_scope_turns = 0
    scope_filtered_turns = 0
    scope_filter_reasons: Counter[str] = Counter()
    hard_passed = 0
    selected_for_semantic = 0
    llm_judged = 0
    llm_evaluable = 0
    llm_not_evaluable = 0
    llm_passed = 0
    badcase_conversations = 0
    latencies: list[float] = []
    for conversation in records:
        conversation_bad = False
        for turn in conversation.get("turns") or []:
            turns += 1
            latencies.append(float(turn.get("latency_ms") or 0.0))
            scope = turn.get("scope_gate") or {"in_scope": True, "reason": "legacy_unscoped"}
            if scope.get("in_scope") is False:
                scope_filtered_turns += 1
                scope_filter_reasons[str(scope.get("reason") or "unknown")] += 1
                judge_modes["scope_filtered"] += 1
                continue
            in_scope_turns += 1
            for intent in turn.get("intents") or []:
                intents[str(intent)] += 1
            for tool in turn.get("tools") or []:
                tool_calls[str(tool)] += 1
            if turn.get("hard_pass"):
                hard_passed += 1
            else:
                conversation_bad = True
                hard_failures.update(turn.get("hard_failure_types") or ["unknown"])
            semantic = turn.get("semantic_judge") or {}
            mode = str(semantic.get("judge_mode") or "unknown")
            judge_modes[mode] += 1
            if mode not in {"unknown", "not_selected"}:
                selected_for_semantic += 1
            if mode == "llm":
                llm_judged += 1
                if semantic.get("evaluable") is False:
                    llm_not_evaluable += 1
                    continue
                llm_evaluable += 1
                if semantic.get("pass"):
                    llm_passed += 1
                else:
                    conversation_bad = True
                    judge_failures[str(semantic.get("failure_type") or "unknown")] += 1
        badcase_conversations += int(conversation_bad)
    latencies.sort()

    def percentile(fraction: float) -> float:
        if not latencies:
            return 0.0
        return latencies[min(len(latencies) - 1, round((len(latencies) - 1) * fraction))]

    return {
        "conversations": len(records),
        "turns": turns,
        "in_scope_turns": in_scope_turns,
        "scope_filtered_turns": scope_filtered_turns,
        "scope_filter_counts": dict(scope_filter_reasons.most_common()),
        "hard_turns_passed": hard_passed,
        "hard_pass_rate": round(hard_passed / max(1, in_scope_turns), 4),
        "badcase_conversations": badcase_conversations,
        "hard_failure_counts": dict(hard_failures.most_common()),
        "semantic_selected_turns": selected_for_semantic,
        # Backwards-compatible semantic_* keys now mean real LLM Judge only.
        "semantic_judged_turns": llm_judged,
        "semantic_evaluable_turns": llm_evaluable,
        "semantic_not_evaluable_turns": llm_not_evaluable,
        "semantic_turns_passed": llm_passed,
        "semantic_pass_rate": (
            round(llm_passed / llm_evaluable, 4) if llm_evaluable else None
        ),
        "llm_judged_turns": llm_judged,
        "llm_evaluable_turns": llm_evaluable,
        "llm_not_evaluable_turns": llm_not_evaluable,
        "llm_turns_passed": llm_passed,
        "llm_pass_rate": (
            round(llm_passed / llm_evaluable, 4) if llm_evaluable else None
        ),
        "fallback_turns": judge_modes.get("deterministic_fallback", 0),
        "deterministic_sidecar_turns": judge_modes.get("deterministic", 0),
        "llm_judge_effective": llm_judged > 0,
        "judge_mode_counts": dict(judge_modes.most_common()),
        "semantic_failure_counts": dict(judge_failures.most_common()),
        "intent_counts": dict(intents.most_common()),
        "tool_call_counts": dict(tool_calls.most_common()),
        "latency_p50_ms": round(percentile(0.5), 2),
        "latency_p95_ms": round(percentile(0.95), 2),
    }
