"""Equivalent graph nodes; dependencies resolve through the compatibility facade."""

from __future__ import annotations

from .. import agent as runtime


def _format_product_line(product: dict[str, runtime.Any], index: int) -> str:
    brand = f"{product['brand']} " if product["brand"] else ""
    return (
        f"{index}. {brand}{product['title']}，{runtime.format_price(product)}，"
        f"评分 {product['rating']:.1f}。{product['reason']}"
    )


def _citation_for(
    citations: list[dict[str, runtime.Any]],
    *,
    field: str,
    sku: str = "",
    doc_id: str = "",
    order_id: str = "",
) -> dict[str, runtime.Any] | None:
    return next(
        (
            item
            for item in citations
            if item.get("field") == field
            and (not sku or item.get("sku") == sku)
            and (not doc_id or item.get("doc_id") == doc_id)
            and (not order_id or item.get("order_id") == order_id)
        ),
        None,
    )


def _ground_output(
    payload: dict[str, runtime.Any],
    claims: list[dict[str, runtime.Any]],
    *,
    policy_documents: list[dict[str, runtime.Any]] | None = None,
) -> dict[str, runtime.Any]:
    citations = runtime.attach_citation_ids(payload.get("citations") or [])
    # Claims may have been created before citations were normalized. IDs are
    # deterministic, so validation remains stable across serialization.
    verdict = runtime.validate_claim_citations(
        claims,
        citations,
        products=payload.get("products") or [],
        policy_documents=policy_documents or [],
        order=payload.get("order"),
    )
    payload["citations"] = citations
    payload["claims"] = claims
    payload["grounding"] = verdict
    return payload


def pre_sales_branch(state: runtime.AgentState) -> dict[str, runtime.Any]:
    task = state["active_task"]
    skill_context = runtime.context_for_skill(state.get("context") or {}, "pre_sales")
    intent_category = task.get("category")
    intent_terms = [t for t in (task.get("target_terms") or []) if t][:12]
    # 检索信号全部由意图层同一次输出（category/target_terms/constraints），
    # 唯一权威来源，不再叠加本地 parse_query 解析（其输出会被 LLM 全量覆盖）。
    intent_c = task.get("constraints") or {}
    parsed = runtime.ParsedQuery(
        raw_query=task["text"],
        rewritten_query=task["text"],
        category=intent_category
        if intent_category in runtime.ALLOWED_CATEGORIES
        else None,
        grounded_target_terms=intent_terms,
        price_min=intent_c.get("price_min"),
        price_max=intent_c.get("price_max"),
        preferred_brands=[b for b in intent_c.get("brands") or [] if b][:4],
        negative_brands=[b for b in intent_c.get("negative_brands") or [] if b][:4],
        negative_terms=[t for t in intent_c.get("negative_terms") or [] if t][:6],
        concepts=[c for c in intent_c.get("concepts") or [] if c][:12],
        llm_concepts=[c for c in intent_c.get("concepts") or [] if c],
        intent=task["intent"],
    )
    parsed.sources["intent"] = str(task.get("intent_source") or "intent_analyzer")
    if intent_category in runtime.ALLOWED_CATEGORIES:
        parsed.sources["category"] = "llm:intent"
    if intent_terms:
        parsed.sources["grounded_target_terms"] = "llm:intent"
    if intent_c.get("brands"):
        parsed.sources["preferred_brands"] = "llm:intent"
    if intent_c.get("concepts"):
        parsed.sources["concepts"] = "llm:intent"
    # 是否澄清、澄清什么由意图 LLM 决定（clarification_slot/question 字段）。
    # 代码不做 case-by-case 澄清规则：检索信号残缺由 schema 层拒绝，而非本地补判断。
    if task.get("clarification_slot"):
        parsed.clarification_slot = task["clarification_slot"]
        parsed.clarification_question = (
            task.get("clarification_question")
            or "您想找什么类型的商品？可以说商品名称，比如“无线鼠标”“键盘”。"
        )
    # Tool contract: catalog_search requires a resolved product target. An
    # unknown category is a missing slot, not authorization for broad recall.
    if parsed.clarification_slot:
        return {
            "branch_outputs": [
                {
                    **runtime._task_fields(task),
                    "branch": "pre_sales",
                    "intent": task["intent"],
                    "response": parsed.clarification_question,
                    "clarification": {
                        "slot": parsed.clarification_slot,
                        "question": parsed.clarification_question,
                        "original_query": task["text"],
                    },
                    "products": [],
                    "citations": [],
                    "suggestions": [],
                    "parsed_query": parsed.to_dict(),
                }
            ],
            "trace": runtime._trace(
                "pre_sales.clarify",
                slot=parsed.clarification_slot,
                question=parsed.clarification_question,
            ),
        }

    if task["intent"] == "detail":
        # 用户指代"第一款/它/这个"时不走本地序数词规则，直接从会话状态取
        # 聚焦/上轮结果 SKU 由售前分支结合 session context 解析；路由层不
        # 把它们提前塞进 Decision arguments。
        context_skus = (skill_context.get("references") or {}).get("focused_skus") or []
        if not context_skus:
            context_skus = skill_context.get("last_result_skus") or []
        detail_skus = parsed.reference_skus or context_skus
        detail_tool = (
            runtime.registry.execute(
                "product_detail",
                "pre_sales",
                {"sku": detail_skus[0]},
            )
            if detail_skus
            else None
        )
        products = (
            [detail_tool["data"]]
            if detail_tool and detail_tool["ok"] and detail_tool["data"]
            else []
        )
        if not products:
            response = "您问的是哪一款商品？请先搜索后选择，或直接告诉我商品编号。"
            payload = {
                **runtime._task_fields(task),
                "branch": "pre_sales",
                "intent": "detail",
                "response": response,
                "products": [],
                "citations": [],
                "suggestions": ["重新搜索商品"],
                "parsed_query": parsed.to_dict(),
                "clarification": {
                    "slot": "product",
                    "question": response,
                    "original_query": task["text"],
                },
            }
        else:
            product = products[0]
            feature_text = (
                "；".join(product["features"][:4]) or product["description"][:240]
            )
            response = runtime.compose_reply(
                "detail",
                user_text=task["text"],
                data={"product": product, "feature_text": feature_text},
                history=state.get("history_window") or [],
            ) or (
                f"{product['title']} 当前价格 {runtime.format_price(product)}，评分 "
                f"{product['rating']:.1f}（{product['rating_count']} 条评价）。{feature_text}"
            )
            citations = [
                {
                    "sku": product["sku"],
                    "field": "price",
                    "value": product["price"],
                    "source": product["source"],
                },
                {
                    "sku": product["sku"],
                    "field": "rating",
                    "value": product["rating"],
                    "source": product["source"],
                },
            ]
            if feature_text:
                feature_source = (
                    {"field": "features", "value": product["features"][0]}
                    if product.get("features")
                    else {"field": "description", "value": feature_text}
                )
                citations.append(
                    {
                        "sku": product["sku"],
                        **feature_source,
                        "source": product["source"],
                    }
                )
            payload = {
                **runtime._task_fields(task),
                "branch": "pre_sales",
                "intent": "detail",
                "response": response,
                "products": products,
                "citations": citations,
                "suggestions": ["找相似款", "加入对比"],
                "parsed_query": parsed.to_dict(),
            }
            detail_claims = [
                runtime.claim(
                    f"{product['title']} 当前价格 {runtime.format_price(product)}",
                    citations[0],
                    claim_type="product_price",
                ),
                runtime.claim(
                    f"{product['title']} 评分 {product['rating']:.1f}",
                    citations[1],
                    claim_type="product_rating",
                ),
            ]
            if feature_text:
                detail_claims.append(
                    runtime.claim(
                        feature_text,
                        citations[2],
                        claim_type="product_feature",
                    )
                )
            payload = runtime._ground_output(payload, detail_claims)
        return {
            "branch_outputs": [payload],
            "trace": [
                *([detail_tool["trace"]] if detail_tool else []),
                {"node": "pre_sales.detail", "skus": parsed.reference_skus},
            ],
        }

    if task["intent"] == "compare":
        skus = parsed.reference_skus
        if len(skus) < 2:
            skus = skill_context.get("last_result_skus", [])[:2]
        detail_results = [
            runtime.registry.execute("product_detail", "pre_sales", {"sku": sku})
            for sku in skus[:4]
        ]
        products = [
            item["data"] for item in detail_results if item["ok"] and item.get("data")
        ]
        comparison = {
            "products": products,
            "fields": ["price", "rating", "stock", "features", "attributes"],
        }
        if len(products) < 2:
            response = "至少需要两款商品才能比较。请先搜索，再选择 2 到 4 款。"
        else:
            lines = [
                f"{index}. {product['title']}：{runtime.format_price(product)}，评分 {product['rating']:.1f}"
                for index, product in enumerate(products, start=1)
            ]
            cheapest = min(products, key=lambda item: item["price"])
            top_rated = max(products, key=lambda item: item["rating"])
            response = runtime.compose_reply(
                "compare",
                user_text=task["text"],
                data={
                    "products": products,
                    "cheapest_title": cheapest["title"],
                    "top_rated_title": top_rated["title"],
                },
                history=state.get("history_window") or [],
            ) or (
                "对比结果：\n"
                + "\n".join(lines)
                + f"\n价格更低的是 {cheapest['title']}；评分更高的是 {top_rated['title']}。"
            )
        payload = {
            **runtime._task_fields(task),
            "branch": "pre_sales",
            "intent": "compare",
            "response": response,
            "products": products,
            "comparison": comparison,
            "citations": [
                citation
                for product in products
                for citation in (
                    {
                        "sku": product["sku"],
                        "field": "price",
                        "value": product["price"],
                        "source": product["source"],
                    },
                    {
                        "sku": product["sku"],
                        "field": "rating",
                        "value": product["rating"],
                        "source": product["source"],
                    },
                )
            ],
            "suggestions": ["按价格排序", "看相似款"],
            "parsed_query": parsed.to_dict(),
        }
        comparison_claims: list[dict[str, runtime.Any]] = []
        for product in products:
            for field, claim_type in (
                ("price", "comparison_price"),
                ("rating", "comparison_rating"),
            ):
                citation = runtime._citation_for(
                    payload["citations"], field=field, sku=product["sku"]
                )
                if citation:
                    comparison_claims.append(
                        runtime.claim(
                            f"{product['title']} {field}={citation['value']}",
                            citation,
                            claim_type=claim_type,
                        )
                    )
        payload = runtime._ground_output(payload, comparison_claims)
        return {
            "branch_outputs": [payload],
            "trace": [
                *(item["trace"] for item in detail_results),
                {"node": "pre_sales.compare", "skus": skus[:4]},
            ],
        }

    tool_result = runtime.registry.execute(
        "catalog_search",
        "pre_sales",
        {
            "parsed_query": parsed.to_dict(),
            "user_id": state["user_id"],
            "limit": 8,
            "mode": "hybrid",
            "session_id": state["session_id"],
        },
    )
    if not tool_result["ok"]:
        return {
            "branch_outputs": [
                {
                    **runtime._task_fields(task),
                    "branch": "pre_sales",
                    "intent": "search",
                    "response": "商品搜索暂时失败，请稍后重试。",
                    "products": [],
                    "citations": [],
                    "claims": [],
                    "grounding": {"pass": True, "claims": 0, "issues": []},
                    "suggestions": [],
                    "parsed_query": parsed.to_dict(),
                    "tool_error": tool_result["error"],
                }
            ],
            "trace": [tool_result["trace"]],
        }
    result = tool_result["data"]
    parsed_payload = result.get("parsed_query") or parsed.to_dict()
    products = result["products"]
    if products:
        relaxation_note = (
            "严格属性组合下没有完全匹配，以下保留类目、商品类型、预算和排除项，"
            "给出最小放宽后的近似候选：\n"
            if result.get("relaxations")
            else ""
        )
        preferred = parsed.preferred_brands
        preferred_note = (
            f"没有找到同时满足硬条件的 {' / '.join(preferred)} 商品，"
            "以下优先保留预算、品类和属性约束推荐其他品牌：\n"
            if preferred
            and not any(product["brand"] in preferred for product in products)
            else ""
        )
        compose_products = products[:3]
        note = (relaxation_note + preferred_note).strip()
        response = runtime.compose_reply(
            "search",
            user_text=task["text"],
            data={
                "products": compose_products,
                "total_candidates": result["total_candidates"],
                "note": note,
            },
            history=state.get("history_window") or [],
        ) or (
            relaxation_note
            + preferred_note
            + f"为您筛出 {result['total_candidates']} 个候选，综合相关性、口碑和个人偏好后，"
            f"优先推荐这 {min(3, len(products))} 款：\n"
            + "\n".join(
                runtime._format_product_line(product, index)
                for index, product in enumerate(compose_products, start=1)
            )
        )
    elif parsed_payload.get("clarification_slot"):
        response = parsed_payload.get("clarification_question") or (
            "我还不能确定具体商品类型，请再补充一下用途或商品名称。"
        )
    else:
        response = (
            f"当前商品库里没有找到与“{parsed.raw_query}”足够相关的商品。"
            "这可能是目录覆盖不足，不是让您继续放宽无关条件。"
        )
    payload = {
        **runtime._task_fields(task),
        "branch": "pre_sales",
        "intent": "search",
        "response": response,
        "products": products,
        "citations": result.get("citations", []),
        "suggestions": result.get("suggestions", []),
        "parsed_query": parsed_payload,
        "search": {
            "request_id": result["request_id"],
            "total_candidates": result["total_candidates"],
            "facets": result["facets"],
            "trace": result["trace"],
            "latency_ms": result["latency_ms"],
            "judge": result["judge"],
            "relaxations": result.get("relaxations") or [],
        },
    }
    if not products and parsed_payload.get("clarification_slot"):
        payload["clarification"] = {
            "slot": parsed_payload["clarification_slot"],
            "question": response,
            "original_query": task["text"],
        }
    search_claims: list[dict[str, runtime.Any]] = []
    for product in products[:3]:
        for field, claim_type in (
            ("price", "product_price"),
            ("rating", "product_rating"),
        ):
            citation = runtime._citation_for(
                payload["citations"], field=field, sku=product["sku"]
            )
            if citation:
                search_claims.append(
                    runtime.claim(
                        f"{product['title']} {field}={citation['value']}",
                        citation,
                        claim_type=claim_type,
                    )
                )
    payload = runtime._ground_output(payload, search_claims)
    return {
        "branch_outputs": [payload],
        "trace": [
            tool_result["trace"],
            {
                "node": "pre_sales.search",
                "request_id": result["request_id"],
                "candidates": result["total_candidates"],
                "returned": len(products),
                "grounding_pass": payload["grounding"]["pass"],
            },
        ],
    }
