from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from datetime import date
from typing import Any

from langgraph.graph import END, START, StateGraph

from .answer_faithfulness import render_answer_sections, verify_answer_faithfulness
from .catalog import (
    add_message,
    format_price,
    get_session,
    list_brands,
    save_session,
    persist_turn,
)
from .config import settings
from .context import (
    bounded_history,
    context_for_skill,
    normalize_customer_context,
    update_customer_context,
)
from .decision import build_decision
from .grounding import (
    attach_citation_ids,
    claim,
    policy_citation,
    validate_claim_citations,
)
from .llm import (
    ALLOWED_CATEGORIES,
    analyze_intents,
    assess_policy_evidence,
    compose_reply,
    plan_after_sales,
    summarize_history,
)
from .llm_config import agent_max_tokens, get_model, invoke_with_telemetry
from .memory import remember_turn
from .policy_followup import run_policy_followup
from .query import (
    extract_order_id,
    parse_query,
)
from .retrieval import search_engine
from .schemas import ParsedQuery
from .state import AgentState
from .taxonomy import CATEGORY_ALIASES, category_label
from .text import compact_text
from .tools import registry


def _trace(node: str, **payload: Any) -> list[dict[str, Any]]:
    return [{"node": node, **payload}]


_PLACEHOLDER_ORDER_ID = re.compile(
    r"\[(?:ORDERID|订单ID|订单号)_[A-Za-z0-9_-]+\]",
    re.IGNORECASE,
)
_STOP_MARKERS = (
    "不要处理了",
    "不用处理了",
    "先不处理",
    "不需要了",
    "没有了，放弃了",
    "没有了,放弃了",
    "放弃了",
    "算了吧",
    "不弄了",
)
_STOP_ONLY = re.compile(
    r"^(?:没有了|没了|不需要|不用了|算了|放弃了|不弄了|先这样)"
    r"[啊呀哦吧哈\s,.，。!！?？]*$"
)
_CLOSING_MARKERS = ("再见", "拜拜", "谢谢", "谢谢啦", "谢了", "么么哒")
_CLOSING_ONLY = re.compile(
    r"^(?:(?:没有了|没了)[，,\s]*)?(?:再见|拜拜|谢谢(?:啦)?|謝謝|谢了|么么哒)"
    r"[啊呀哦吧哈\s,.，。!！?？^\[\]数字x]*$"
)
_GREETING_ONLY = re.compile(
    r"^(?:你?好|您好|在吗|在\??|亲|哈喽|hello|hi)[啊呀哦吧哈\s?？!！]*$",
    re.IGNORECASE,
)
_ACK_ONLY = re.compile(
    r"^(?:是|是的|对|对的|嗯+|哦+|喔+|好|好的|可以|行|ok|okay|知道了|明白了)"
    r"[啊呀哦吧哈\s,.，。!！?？]*$",
    re.IGNORECASE,
)
_PROGRESS_ONLY = re.compile(
    r"^(?:好了没|好了吗|处理好了吗|处理到哪了|有进度吗|进度呢|能不能快点(?:帮我)?处理|"
    r"麻烦快点|尽快处理|好慢|太慢了?|我没时间浪费|但是我现在急用呀?)[啊呀哦吧哈\s,.，。!！?？]*$"
)
_PROCESS_QUESTION = re.compile(
    r"^(?:流程是什么|什么流程|怎么申请|怎么处理|接下来怎么办|下一步呢)"
    r"[啊呀哦吧哈\s,.，。!！?？]*$"
)
_PRICE_QUESTION = re.compile(
    r"(?:多少钱|什么价|价格|价钱|报价|费用|收费|降价|便宜|预算|\d+\s*(?:元|块|美元))",
    re.IGNORECASE,
)
_CATEGORY_RELATION = re.compile(
    r"^\s*(?P<subject>.+?)(?:算在|算是|属于)(?P<target>.+?)[吗么?？。!！]*$"
)
_DELIVERY_RESOLVED = re.compile(
    r"^(?:(?:已经|现在)?(?:完成了?|解决了?)[，,\s]*)?(?:已经)?(?:收到货了?|收到了?|签收了?)"
    r"[啊呀哦吧哈\s,.，。!！?？]*$"
)
_COMPOUND_CLOSE = re.compile(
    r"^(?:暂时)?(?:没有了|没了|不需要了).{0,12}(?:谢谢|感谢|再见|拜拜)|"
    r"^(?:谢谢|感谢).{0,12}(?:没有了|没了|再见|拜拜)"
)
_STANDALONE_IDENTIFIER = re.compile(
    r"^(?=.{3,100}$)(?:[#A-Za-z0-9_-]+|\[[^\]]{1,80}\]|"
    r"[#A-Za-z0-9_-]*\[[^\]]{1,80}\][#A-Za-z0-9_\-\[\]]*)$"
)


def _placeholder_order_id(text: str) -> str | None:
    match = _PLACEHOLDER_ORDER_ID.search(text)
    return match.group(0).upper() if match else None


def _standalone_identifier(text: str) -> str | None:
    candidate = compact_text(text, 100).strip()
    if _STANDALONE_IDENTIFIER.fullmatch(candidate):
        return candidate
    return None


def _catalog_taxonomy_answer(message: str) -> str | None:
    """Answer category-membership questions from the local taxonomy, not search."""

    match = _CATEGORY_RELATION.fullmatch(compact_text(message, 120))
    if not match:
        return None
    subject = re.sub(r"(?:里面|里|这一类|类别|类)$", "", match.group("subject")).strip()
    target = re.sub(r"(?:里面|里|这一类|类别|类)$", "", match.group("target")).strip()
    subject_category = CATEGORY_ALIASES.get(subject)
    target_category = CATEGORY_ALIASES.get(target)
    if not subject_category or not target_category:
        return None
    subject_label = category_label(subject_category)
    target_label = category_label(target_category)
    if subject_category == target_category:
        return f"按当前本地商品目录，{subject}归在“{subject_label}”类。"
    return (
        f"按当前本地商品目录，{subject}归在“{subject_label}”类，"
        f"不属于“{target_label}”类。"
    )


def _business_followup_kind(message: str) -> str | None:
    text = compact_text(message, 120)
    if _GREETING_ONLY.fullmatch(text):
        return "greeting"
    if _ACK_ONLY.fullmatch(text):
        return "acknowledgement"
    if _PROGRESS_ONLY.fullmatch(text):
        return "progress"
    if _PROCESS_QUESTION.fullmatch(text):
        return "process"
    if _PRICE_QUESTION.search(text.casefold()):
        return "price_question"
    return None


_REQUESTED_SLOT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("price", ("多少钱", "什么价", "价位", "报价", "费用", "收费", "降价", "便宜")),
    ("stock", ("有货", "库存", "缺货", "现货", "有没有货", "有货吗")),
    (
        "spec",
        ("参数", "配置", "规格", "内存", "容量", "屏幕", "尺寸", "处理器", "芯片"),
    ),
    ("shipping", ("发货", "物流", "几天到", "多久到", "运费", "包邮", "送达")),
    ("brand", ("牌子", "品牌", "哪个牌子")),
    ("warranty", ("保修", "质保", "售后", "维修")),
    ("detail", ("详情", "介绍", "性能", "功能", "支持")),
)
_COLOR_QUESTION = re.compile(
    r"颜色|什么色|有哪些色|某色|色系|[红橙黄绿青蓝紫黑白金银粉灰棕褐]色"
)


# 提取用户想知道哪些信息
def _requested_slots(message: str) -> list[str]:
    """Extract slots the user is asking about (MultiWOZ-style requested_slots).

    Unlike required_slots (what the system still needs from the user), these
    are attributes the user explicitly requests from the system in this turn.
    """
    text = compact_text(message, 240).casefold()
    slots: list[str] = []
    for slot, markers in _REQUESTED_SLOT_RULES:
        if any(marker in text for marker in markers) and slot not in slots:
            slots.append(slot)
    if _COLOR_QUESTION.search(text) and "color" not in slots:
        slots.append("color")
    return slots


def _dialogue_frame(
    message: str,
    intent: str,
    *,
    followup_kind: str | None,
) -> dict[str, Any]:
    """
    给对话打标签 通过规则分析一些补充信息
    Create the current-turn dialogue metadata carried with an intent task.

    The frame intentionally contains no inferred business facts. It records
    dialogue act and lightweight references for trace/UI continuity. The
    supervisor Decision does not consume it to extract domain arguments.
    """

    text = compact_text(message, 240)
    lowered = text.casefold()
    identifier = (
        extract_order_id(text)
        or _placeholder_order_id(text)
        or _standalone_identifier(text)
        or ""
    )
    if _COMPOUND_CLOSE.search(text) or _CLOSING_ONLY.fullmatch(text):
        act = "closing"
    elif any(marker in lowered for marker in _STOP_MARKERS) or _STOP_ONLY.fullmatch(
        text
    ):
        act = "stop"
    elif followup_kind:
        act = followup_kind
    else:
        act = "request"

    if intent in {"search", "detail", "compare"}:
        domain = "pre_sales"
    elif intent == "order":
        domain = "order_read"
    elif intent == "service":
        domain = "after_sales_rag"
    else:
        domain = "conversation"

    # 售后/订单域的语义路由（能力边界/隐私/网点/查单/查政策）由售后域规划器
    # plan_after_sales 决定；这里只保留 act/domain/identifier 等结构信息，
    # 不再做任何关键词语义分类。
    if intent in {"search", "detail", "compare"}:
        action = {
            "detail": "product_capability",
            "compare": "compare",
            "search": "catalog_search",
        }[intent]
    else:
        action = act
    return {
        "act": act,
        "domain": domain,
        "action": action,
        "identifier": identifier,
        "required_slots": [],
        "requested_slots": _requested_slots(text),
        "last_question": "",
    }


def _task_fields(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "task_index": task.get("task_index"),
        "task_text": task.get("text"),
        "dialogue_frame": dict(task.get("dialogue_frame") or {}),
        "decision": dict(task.get("decision") or {}),
    }


class _PlanDependencyError(ValueError):
    """A planned step referenced data that an earlier step did not produce."""


# Fixed responses for routes that never touch a tool. The planner picks the
# route; these templates only guarantee honest, non-hallucinated wording.
_ROUTE_REPLIES: dict[str, tuple[str, list[str]]] = {
    "planner_error": (
        "售后语义规划失败，未执行政策查询。请稍后重试。",
        [],
    ),
    "human_handoff": (
        "当前系统只支持商品、订单和售后规则的自动查询，"
        "不处理争议赔偿等超出查询范围的事项。请补充具体订单或商品信息，"
        "我可以继续核对已有规则。",
        [],
    ),
    "cancel_request": (
        "当前工具只支持订单和售后规则查询，暂不支持取消、修改订单或创建售后申请。"
        "如需核对相关规则，请提供订单号或商品信息。",
        ["order_or_service_id"],
    ),
    "service_location": (
        "当前知识库没有可核验的具体售后网点，不能直接编造地址；"
        "请提供商品品牌和所在区县，或到品牌官方售后渠道查询服务网点。",
        ["brand", "location"],
    ),
    "privacy_block": (
        "当前只读订单工具不会返回收件人姓名、手机号或完整地址等隐私字段，"
        "因此我不能替您查看或猜测这些信息。若要核对商品、订单状态或物流，"
        "请提供订单号。",
        ["order_id"],
    ),
    "out_of_scope": (
        "这个客服 Agent 目前只处理商城商品、订单和售后；"
        "火车票、机票、酒店等第三方服务不在当前工具范围内。"
        "请到对应订单渠道办理。",
        [],
    ),
}


from .graphs.routing import (
    load_context,
    understand,
    route_and_extract,
    execute_tool,
    compact_context,
)
from .graphs.pre_sales import (
    _format_product_line,
    _citation_for,
    _ground_output,
    pre_sales_branch,
)
from .graphs.after_sales import (
    _resolve_plan_value,
    _resolve_plan_args,
    _after_sales_failure_reply,
    _order_fallback_reply,
    after_sales_branch,
)
from .graphs.response import (
    _recent_dialogue,
    _render_business_outcome_followup,
    _llm_chitchat_reply,
    _offline_contextual_chitchat,
    _contextual_chitchat_reply,
    chitchat_branch,
    _normalized_response,
    _repair_repeated_response,
    build_response,
    verify_grounding,
)
from .graphs.persistence import persist
from .graphs.builder import build_graph
import sys

graph = build_graph(sys.modules[__name__])


def chat(session_id: str, message: str, user_id: str = "demo-user") -> dict[str, Any]:
    result = graph.invoke(
        {
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
            "branch_outputs": [],
            "trace": [],
        }
    )
    return result["final"]
