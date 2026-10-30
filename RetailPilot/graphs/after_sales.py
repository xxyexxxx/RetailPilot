"""Equivalent graph nodes; dependencies resolve through the compatibility facade."""

from __future__ import annotations

from .. import agent as runtime


def _resolve_plan_value(
    value: runtime.Any, observations: dict[str, dict[str, runtime.Any]]
) -> runtime.Any:
    """Resolve a "$step.field" reference against earlier step observations."""

    if isinstance(value, str) and value.startswith("$"):
        parts = value[1:].split(".")
        data: runtime.Any = observations.get(parts[0])
        for field_name in parts[1:]:
            if not isinstance(data, dict):
                return None
            data = data.get(field_name)
        return data
    return value


def _resolve_plan_args(
    args: dict[str, runtime.Any],
    observations: dict[str, dict[str, runtime.Any]],
) -> dict[str, runtime.Any]:
    """Resolve every $ref inside one planned step's arguments."""

    def resolve(value: runtime.Any, path: str) -> runtime.Any:
        if isinstance(value, str) and value.startswith("$"):
            resolved = runtime._resolve_plan_value(value, observations)
            if resolved in (None, ""):
                raise runtime._PlanDependencyError(
                    f"unresolved plan reference {value} at {path}"
                )
            return resolved
        if isinstance(value, list):
            return [
                resolved
                for index, item in enumerate(value)
                if (resolved := resolve(item, f"{path}[{index}]")) not in (None, "")
            ]
        if isinstance(value, dict):
            return {
                key: resolved
                for key, item in value.items()
                if (resolved := resolve(item, f"{path}.{key}")) not in (None, "")
            }
        return value

    resolved: dict[str, runtime.Any] = {}
    for key, value in args.items():
        resolved_value = resolve(value, key)
        if isinstance(value, (list, dict)):
            resolved[key] = resolved_value
        elif resolved_value not in (None, ""):
            resolved[key] = resolved_value
    return resolved


def _after_sales_failure_reply(
    route: str,
    failure: dict[str, runtime.Any],
) -> str:
    """Render a truthful terminal response after tool retries are exhausted."""

    failure_type = str(failure.get("type") or "tool_error")
    tool = str(failure.get("tool") or "")
    if failure_type == "ambiguous_order":
        candidates = failure.get("candidates") or []
        options = []
        for item in candidates[:5]:
            title = runtime.compact_text(item.get("title") or "商品", 36)
            created = runtime.compact_text(item.get("created_at") or "", 10)
            order_id = str(item.get("order_id") or "")
            suffix = order_id[-4:] if order_id else "未知"
            options.append(f"{title}（{created or '日期未知'}，订单尾号{suffix}）")
        return "找到多笔可能的订单：" + "；".join(options) + "。请指定其中一笔。"
    if failure_type == "order_not_found":
        return "没有找到匹配的历史订单，请补充订单号或更具体的商品/时间。"
    if failure_type == "missing_dependency":
        return (
            "当前售后查询缺少上一步返回的订单或商品信息，"
            "因此无法继续核对。请补充订单号或商品信息后重试。"
        )
    if failure_type == "invalid_arguments":
        return "当前售后查询参数不足或格式不正确，请补充订单号或商品信息后重试。"
    if tool == "order_query" or route == "order_fact":
        return "订单查询暂时失败，请稍后重试。"
    if tool == "policy_retrieve" or route == "policy_question":
        return "售后规则查询暂时失败，请稍后重试。"
    return "当前售后查询暂时失败，请稍后重试。"


def _order_fallback_reply(
    order_payload: dict[str, runtime.Any],
    request: str,
    latest: dict[str, runtime.Any],
) -> str:
    """Deterministic order reply used when compose_reply has no LLM."""

    if request == "item":
        return f"订单 {order_payload['order_id']} 购买的是：{order_payload['title']}。"
    if request == "logistics":
        events = order_payload.get("events") or []
        return "订单完整物流：\n" + "\n".join(
            f"- {event.get('event_time', '-')}｜{event.get('location', '-')}｜"
            f"{event.get('description', event.get('status', '-'))}"
            for event in events
        )
    return (
        f"订单 {order_payload['order_id']} 购买的是 {order_payload['title']}，"
        f"当前状态为“{order_payload['status']}”。"
        f"最新物流：{latest.get('description', '暂无物流节点')}"
        f"（{latest.get('location', '-')}，{latest.get('event_time', '-')}）。"
    )


def after_sales_branch(state: runtime.AgentState) -> dict[str, runtime.Any]:
    """Domain agent for orders and after-sales: Plan-and-Execute.

    The domain planner (LLM) decides the business route and the ordered tool
    chain; this node only executes the plan, resolves $refs between steps,
    grounds every claim against tool output and renders the reply. No local
    keyword/rule semantic classification happens here anymore.
    """

    task = state["active_task"]
    text = task["text"]
    skill_context = runtime.context_for_skill(state.get("context") or {}, "after_sales")
    history = state.get("history_window") or []

    plan = runtime.plan_after_sales(
        text,
        context=state.get("context") or {},
        history=history,
    )
    route = str(plan.get("route") or "clarify")

    required_slots: list[str] = []
    tool_traces: list[dict[str, runtime.Any]] = [
        {
            "node": "after_sales.plan",
            "route": route,
            "topic": plan.get("topic"),
            "standalone_query": plan.get("standalone_query"),
            "order_scope": plan.get("order_scope"),
            "steps": [
                {
                    "step_id": step.get("step_id"),
                    "tool": step.get("tool"),
                    "purpose": step.get("purpose"),
                }
                for step in plan.get("steps") or []
            ],
            "reason": plan.get("reason"),
        }
    ]
    claims: list[dict[str, runtime.Any]] = []
    citations: list[dict[str, runtime.Any]] = []
    order_payload: dict[str, runtime.Any] | None = None
    policy_payload: dict[str, runtime.Any] | None = None
    grounding_documents: list[dict[str, runtime.Any]] = []
    handoff_payload: dict[str, runtime.Any] | None = None
    chained_order: dict[str, runtime.Any] | None = None
    execution_failure: dict[str, runtime.Any] | None = None
    evidence_assessment: dict[str, runtime.Any] | None = None
    evidence_rejected = False
    policy_arguments: dict[str, runtime.Any] = {}
    policy_started: float | None = None
    followup_result: dict[str, runtime.Any] | None = None

    if route in runtime._ROUTE_REPLIES:
        response, required_slots = runtime._ROUTE_REPLIES[route]
    elif route == "clarify":
        question = str(plan.get("clarification_question") or "").strip()
        missing = [str(slot) for slot in plan.get("missing_slots") or []]
        if question:
            response = question
        elif missing:
            response = (
                f"我理解这是售后问题“{runtime.compact_text(text, 80)}”，"
                f"但当前还缺少：{'、'.join(missing)}。补充后我再继续核对。"
            )
        else:
            response = (
                f"我理解这是售后问题“{runtime.compact_text(text, 80)}”，"
                "但当前信息不足以形成可执行的查询计划。"
                "请说明要核对退换货、退款、保修维修、发票还是配送运费，"
                "并提供订单号或商品信息。"
            )
        required_slots = missing or ["product_or_order"]
    elif route == "progress_followup":
        after_state = skill_context.get("after_sales") or {}
        topic = str(after_state.get("topic") or "售后事项")
        pending = list(
            (skill_context.get("dialogue") or {}).get("required_slots") or []
        )
        if pending:
            slot_hint = "、".join(pending[:3])
            response = f"关于{topic}，目前还缺少：{slot_hint}；补充后我才能继续核对。"
            required_slots = pending
        else:
            response = (
                f"关于{topic}，目前这里只能确认已进入处理环节，"
                "当前系统不会创建外部工单。如需核对规则或订单状态，请直接说明。"
            )
    else:
        # order_fact / policy_question: run the planned tool chain in order.
        observations: dict[str, dict[str, runtime.Any]] = {}
        executed_steps: list[dict[str, runtime.Any]] = []
        for step in plan.get("steps") or []:
            tool_name = str(step.get("tool") or "")
            if tool_name not in {"order_query", "policy_retrieve"}:
                continue
            step_id = str(step.get("step_id") or tool_name)
            try:
                args = runtime._resolve_plan_args(
                    dict(step.get("args") or {}), observations
                )
            except runtime._PlanDependencyError as exc:
                execution_failure = {
                    "type": "missing_dependency",
                    "message": str(exc),
                    "tool": tool_name,
                    "step_id": step_id,
                }
                observations[step_id] = {
                    "ok": False,
                    "error": execution_failure,
                }
                executed_steps.append(
                    {
                        "step_id": step_id,
                        "tool": tool_name,
                        "ok": False,
                        "error": execution_failure,
                    }
                )
                break
            args.setdefault("user_id", state["user_id"])
            if tool_name == "policy_retrieve":
                args["query"] = plan.get("standalone_query") or text
                args["platform"] = runtime.settings.commerce_platform
                args["as_of"] = runtime.date.today()
                args["limit"] = 6
                if plan.get("topic"):
                    args["topic"] = plan["topic"]
            if tool_name == "policy_retrieve":
                policy_arguments = runtime.deepcopy(args)
                policy_started = runtime.time.perf_counter()
            # The initial retrieval shares the same deadline as followups.
            # Non-policy tools retain their original execution semantics.
            result = runtime.registry.execute(
                tool_name,
                "after_sales",
                args,
                **(
                    {
                        "deadline": policy_started
                        + runtime.settings.policy_followup_timeout_seconds
                    }
                    if tool_name == "policy_retrieve" and policy_started is not None
                    else {}
                ),
            )
            tool_traces.append(result["trace"])
            executed_steps.append(
                {
                    "step_id": step_id,
                    "tool": tool_name,
                    "ok": bool(result["ok"]),
                }
            )
            if not result["ok"]:
                execution_failure = {
                    "type": str(
                        (result.get("error") or {}).get("type") or "tool_error"
                    ),
                    "message": str((result.get("error") or {}).get("message") or ""),
                    "tool": tool_name,
                    "step_id": step_id,
                    "attempts": (result.get("trace") or {}).get("attempts", 0),
                }
                observations[step_id] = {
                    "ok": False,
                    "error": result.get("error"),
                }
                executed_steps[-1]["error"] = execution_failure
                break
            data = result["data"] or {}
            if tool_name == "order_query":
                matched_orders = list(data.get("orders") or [])
                order_obj = data.get("order")
                if not order_obj and len(matched_orders) == 1:
                    order_obj = matched_orders[0]
                if len(matched_orders) > 1:
                    execution_failure = {
                        "type": "ambiguous_order",
                        "message": "multiple order candidates",
                        "tool": tool_name,
                        "step_id": step_id,
                        "candidates": [
                            {
                                key: item.get(key)
                                for key in (
                                    "order_id",
                                    "title",
                                    "created_at",
                                    "status",
                                    "sku",
                                )
                            }
                            for item in matched_orders[:5]
                        ],
                    }
                    observations[step_id] = {
                        "ok": False,
                        "error": execution_failure,
                    }
                    executed_steps[-1]["error"] = execution_failure
                    break
                if not matched_orders and not order_obj and "orders" in data:
                    execution_failure = {
                        "type": "order_not_found",
                        "message": "no matching order",
                        "tool": tool_name,
                        "step_id": step_id,
                    }
                    observations[step_id] = {
                        "ok": False,
                        "error": execution_failure,
                    }
                    executed_steps[-1]["error"] = execution_failure
                    break
                if not matched_orders and not order_obj:
                    # Compatibility with older tool adapters that only return
                    # {"order": None}; dependent policy steps will surface a
                    # missing-dependency guard below.
                    observations[step_id] = {}
                    continue
                if order_obj:
                    chained_order = order_obj
                    order_payload = order_obj
                    citations.extend(data.get("citations") or [])
                    # Flatten the order object so "$<step>.sku" refs resolve.
                    observations[step_id] = {**data, **order_obj}
                else:
                    observations[step_id] = {}
            else:
                policy_payload = data
                grounding_documents = data.get("documents") or []
                observations[step_id] = data
        tool_traces.append(
            {"node": "after_sales.steps_executed", "steps": executed_steps}
        )

        if not execution_failure and route == "policy_question":
            # Freeze authoritative scope returned by the first tool. The grader
            # may propose topics/queries, never platform/category/order changes.
            policy_arguments["product_context"] = runtime.deepcopy(
                (policy_payload or {}).get("product_context")
                or policy_arguments.get("product_context")
                or {}
            )
            followup_result = runtime.run_policy_followup(
                query=str(plan.get("standalone_query") or text),
                topic=str(plan.get("topic") or ""),
                initial_payload=policy_payload or {},
                retrieval_arguments=policy_arguments,
                retrieve=lambda arguments: runtime.registry.execute(
                    "policy_retrieve",
                    "after_sales",
                    arguments,
                    deadline=(policy_started or runtime.time.perf_counter())
                    + runtime.settings.policy_followup_timeout_seconds,
                ),
                assess=runtime.assess_policy_evidence,
                started_at=policy_started,
            )
            evidence_assessment = followup_result["assessment"]
            grounding_documents = followup_result["documents"]
            policy_payload = {
                **(policy_payload or {}),
                "documents": grounding_documents,
            }
            evidence_rejected = evidence_assessment["next_action"] != "answer"
            tool_traces.extend(
                {
                    **entry,
                    "task_id": task.get("task_id"),
                    "task_index": task.get("task_index"),
                }
                for entry in followup_result["trace"]
            )

        if execution_failure:
            response = runtime._after_sales_failure_reply(route, execution_failure)
            required_slots = (
                ["product_or_order"]
                if execution_failure["type"] == "missing_dependency"
                else ["order_id"]
                if execution_failure["type"] in {"ambiguous_order", "order_not_found"}
                else []
            )
            order_payload = None
            policy_payload = None
            grounding_documents = []
            claims = []
            citations = []
        elif route == "order_fact":
            if order_payload:
                first_args = (plan.get("steps") or [{}])[0].get("args") or {}
                request = str(first_args.get("request") or "status")
                latest = (order_payload.get("events") or [{}])[-1]
                response = runtime.compose_reply(
                    "order",
                    user_text=text,
                    data={"order": order_payload, "request": request},
                    history=history,
                ) or runtime._order_fallback_reply(order_payload, request, latest)
                title_citation = runtime._citation_for(
                    citations, field="title", order_id=order_payload["order_id"]
                )
                status_citation = runtime._citation_for(
                    citations, field="status", order_id=order_payload["order_id"]
                )
                logistics_citation = runtime._citation_for(
                    citations,
                    field="latest_logistics",
                    order_id=order_payload["order_id"],
                )
                events_citation = runtime._citation_for(
                    citations,
                    field="logistics_events",
                    order_id=order_payload["order_id"],
                )
                if title_citation and request != "logistics":
                    claims.append(
                        runtime.claim(
                            f"订单购买的是 {order_payload['title']}",
                            title_citation,
                            claim_type="order_item",
                        )
                    )
                if status_citation and request == "status":
                    claims.append(
                        runtime.claim(
                            f"订单当前状态为 {order_payload['status']}",
                            status_citation,
                            claim_type="order_status",
                        )
                    )
                if logistics_citation and request == "status":
                    claims.append(
                        runtime.claim(
                            f"最新物流为 {latest.get('description', '')}",
                            logistics_citation,
                            claim_type="logistics_status",
                        )
                    )
                if events_citation and request == "logistics":
                    claims.append(
                        runtime.claim(
                            "订单完整物流节点",
                            events_citation,
                            claim_type="logistics_events",
                        )
                    )
            else:
                response = "没有查到对应订单。请确认订单号，或切换到下单时使用的账号。"
                required_slots = ["order_id"]
                order_payload = None
                citations = []
        elif evidence_rejected:  # policy_question without answerable evidence
            decision = str((evidence_assessment or {}).get("decision") or "refuse")
            question = str(
                (evidence_assessment or {}).get("clarification_question") or ""
            ).strip()
            if decision == "clarify":
                response = (
                    question or "现有政策证据不足。请补充具体商品或订单信息后再核对。"
                )
                required_slots = (evidence_assessment or {}).get(
                    "required_user_fields"
                ) or ["product_or_order"]
            else:
                response = "检索到的政策证据不足或存在冲突，暂不直接给出结论。"
            citations = []
            claims = []
            policy_payload = policy_payload or {}
        else:  # policy_question
            if grounding_documents:
                selected = grounding_documents
                evidence_lines = [
                    f"{item.get('context') or item['content']}（来源：{item['source']} / {item['title']} / {item['section_path']} / v{item['version']}）"
                    for item in selected
                ]
                response = runtime.compose_reply(
                    "policy",
                    user_text=text,
                    data={
                        "documents": [
                            {
                                "chunk_id": item["chunk_id"],
                                "doc_id": item["doc_id"],
                                "content": item["content"],
                                "evidence": item["content"],
                                "source": item["source"],
                                "title": item["title"],
                                "section_path": item["section_path"],
                                "version": item["version"],
                            }
                            for item in selected
                        ],
                        "aspect_evidence_map": (evidence_assessment or {}).get(
                            "aspect_evidence_map"
                        )
                        or {},
                        "needs_product_context": (policy_payload or {}).get(
                            "needs_product_context", False
                        ),
                        "order": (
                            {
                                "order_id": chained_order["order_id"],
                                "title": chained_order["title"],
                                "status": chained_order.get("status"),
                            }
                            if chained_order
                            else None
                        ),
                    },
                    history=history,
                ) or (
                    (
                        f"针对您最近购买的 {chained_order['title']}"
                        f"（订单 {chained_order['order_id']}），"
                        if chained_order
                        else f"针对您本轮补充的“{runtime.compact_text(text, 72)}”，"
                    )
                    + "当前知识库能核验的规则是：\n"
                    + "\n".join(evidence_lines)
                )
                if (policy_payload or {}).get("needs_product_context"):
                    response += (
                        "\n这是通用规则；请再提供具体商品或订单，以便核对类目例外。"
                    )
                    required_slots = ["product_or_order"]
                citations = [runtime.policy_citation(item) for item in selected]
                for item, citation in zip(selected, citations):
                    claims.append(
                        runtime.claim(
                            item["content"], citation, claim_type="policy_rule"
                        )
                    )
            else:
                response = "当前知识库没有找到能支持结论的规则，不建议直接猜测。"
                required_slots = ["product_or_order"]
                policy_payload = policy_payload or {}

    # This project is intentionally self-contained: no human handoff or
    # external ticket is created.  Keep the stable fields for API consumers,
    # but terminal failures are rendered directly in the reply.
    should_handoff = False

    payload = {
        **runtime._task_fields(task),
        "branch": "after_sales",
        "intent": task["intent"],
        "response": response,
        "products": [],
        "citations": citations,
        "suggestions": ["查看完整物流", "查询退换货规则"] if order_payload else [],
        "order": order_payload,
        "policy": (
            {
                "topic": (policy_payload or {}).get("topic") or plan.get("topic"),
                "query": (policy_payload or {}).get("query")
                or plan.get("standalone_query"),
                "product_context": (policy_payload or {}).get("product_context") or {},
                "product_skus": (policy_payload or {}).get("product_skus") or [],
                "retrieval_version": (policy_payload or {}).get("retrieval_version"),
                "document_ids": [item["doc_id"] for item in grounding_documents],
                "chunk_ids": [item["chunk_id"] for item in grounding_documents],
                "retrieval_trace": (policy_payload or {}).get("trace") or {},
                "needs_product_context": (policy_payload or {}).get(
                    "needs_product_context", False
                ),
                "evidence_assessment": evidence_assessment,
                "adaptive_summary": (followup_result or {}).get("summary"),
                "evidence_pool_chunk_ids": [
                    item["chunk_id"]
                    for item in (followup_result or {}).get("evidence_pool", [])
                ],
            }
            if policy_payload is not None
            else None
        ),
        "handoff": handoff_payload,
        "escalated": should_handoff,
        "required_slots": required_slots,
        "tool_failure": execution_failure,
    }

    payload = runtime._ground_output(
        payload, claims, policy_documents=grounding_documents
    )

    return {
        "branch_outputs": [payload],
        "policy_answer_evidence": (
            {
                str(task["task_id"]): {
                    "question": str(plan.get("standalone_query") or text),
                    "documents": runtime.deepcopy(grounding_documents),
                    "tool_results": {
                        "product_context": runtime.deepcopy(
                            (policy_payload or {}).get("product_context") or {}
                        ),
                        "order": (
                            {
                                **{
                                    key: runtime.deepcopy(order_payload.get(key))
                                    for key in (
                                        "order_id",
                                        "sku",
                                        "title",
                                        "status",
                                        "created_at",
                                    )
                                },
                                "events": [
                                    {
                                        key: event.get(key)
                                        for key in ("event_time", "status")
                                    }
                                    for event in order_payload.get("events") or []
                                ],
                            }
                            if order_payload
                            else None
                        ),
                    },
                }
            }
            if route == "policy_question"
            and not evidence_rejected
            and grounding_documents
            else {}
        ),
        "trace": [
            *tool_traces,
            {
                "node": "after_sales.grounded_response",
                "intent": task["intent"],
                "route": route,
                "policy_topic": (policy_payload or {}).get("topic"),
                "policy_documents": len(grounding_documents),
                "evidence_rejected": evidence_rejected,
                "escalated": should_handoff,
                "grounding_pass": payload["grounding"]["pass"],
            },
        ],
    }
