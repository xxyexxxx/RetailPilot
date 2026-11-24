"""Business adapters: immutable scope, one shared workflow, no extra indexes."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict

from .asset_evidence import asset_citation, requires_asset, requires_visual
from .catalog import get_products
from .database import db_session
from .evidence_policies import (
    OrderEvidencePolicy,
    PolicyEvidencePolicy,
    ProductEvidencePolicy,
    fingerprint,
    knowledge_ref,
    policy_bundle,
    product_bundle,
)
from .evidence_schema import EvidenceBundle, EvidenceRef, RequiredAspect
from .evidence_workflow import run_evidence_workflow
from .knowledge_evidence import hydrate
from .knowledge_retrieval import RetrievalFilters, RetrievalQuery, retrieve


def mode():
    from .evidence_shadow import mode_override

    value = mode_override.get() or os.getenv("RETAILPILOT_EVIDENCE_MODE", "legacy")
    if value not in {"legacy", "shadow", "unified"}:
        raise ValueError("invalid_evidence_mode")
    return value


def product_retrieve(proposal, bundle, scope):
    if proposal.domain != "product" or proposal.target_id not in scope.get(
        "product", []
    ):
        raise ValueError("product_scope")
    # _sql_filters is shared with initial catalog retrieval; no proposed filter accepted.
    from .retrieval import _sql_filters
    from .schemas import ParsedQuery

    parsed = ParsedQuery(
        raw_query=bundle.question,
        rewritten_query=bundle.question,
        **bundle.hard_constraints,
    )
    where, params = _sql_filters(parsed)
    for term in parsed.excluded_title_terms:
        where += " AND LOWER(p.title) NOT LIKE ?"
        params.append("%" + term.casefold() + "%")
    for term in parsed.negative_terms[:8]:
        where += " AND LOWER(p.title || ' ' || p.description || ' ' || p.attributes_text) NOT LIKE ?"
        params.append("%" + term.casefold() + "%")
    where += " AND p.sku=? AND p.stock>0"
    params.append(proposal.target_id)
    hits, _, _ = retrieve(
        RetrievalQuery(proposal.query, "product"),
        RetrievalFilters(product_where=where, product_params=params),
        limit=24,
    )
    return [ref for h in hits if (ref := knowledge_ref(h.unit_id))]


def evaluate_products(
    question, products, *, parsed=None, task_id="product", model=None
):
    if mode() == "legacy":
        return products, []
    if mode() == "shadow":
        from .evidence_shadow import schedule

        frozen = deepcopy(products)
        parsed_copy = deepcopy(parsed)

        def audit():
            checked, trace = evaluate_products(
                question, frozen, parsed=parsed_copy, task_id=task_id, model=model
            )
            return {"summary": {"products": len(checked), "trace": trace}}

        return products, [schedule(audit, task_id=task_id)]
    h = {}
    if parsed:
        h = {
            k: deepcopy(getattr(parsed, k))
            for k in (
                "price_min",
                "price_max",
                "brands",
                "negative_brands",
                "negative_terms",
                "excluded_title_terms",
                "category",
            )
        }
    if parsed and parsed.sources.get("price_constraint") == "soft:approximate":
        h["price_min"] = None
        if h.get("price_max") is not None:
            h["price_max"] = max(h["price_max"] * 1.35, h["price_max"] + 10.0)
    requirements = [" / ".join(g) for g in getattr(parsed, "must_have_groups", []) if g]
    complex_question = (
        requires_asset(question)
        or bool(requirements)
        or "比较" in question
        or "对比" in question
        or any(
            term in question
            for term in (
                "支持",
                "功能",
                "参数",
                "续航",
                "容量",
                "防水",
                "降噪",
                "怎么",
                "如何",
                "更换",
            )
        )
    )
    import time

    from .execution import call_deadline

    deadline = min(
        call_deadline.get() or float("inf"),
        time.perf_counter() + float(os.getenv("RETAILPILOT_EVIDENCE_TIMEOUT", "30")),
    )
    images_left = int(os.getenv("RETAILPILOT_EVIDENCE_MAX_IMAGES", "2"))
    tokens_left = int(os.getenv("RETAILPILOT_EVIDENCE_TOKEN_BUDGET", "80000"))
    traces = []
    accepted = []
    # A fixed top-N budget bounds request cost; simpler catalog-only requests
    # take the deterministic zero-model path.
    for product in products[:4] if complex_question else products:
        bundle = product_bundle(
            question,
            [product],
            task_id=task_id,
            constraints=h,
            requirements=requirements or None,
            deterministic=not complex_question,
        )
        result = run_evidence_workflow(
            initial_bundle=bundle,
            policy=ProductEvidencePolicy(),
            retrieve=product_retrieve,
            model=model,
            budget={
                "timeout": max(0, deadline - time.perf_counter()),
                "images": max(0, images_left),
                "tokens": max(0, tokens_left),
            },
        )
        images_left -= result["summary"].get("vision_input_count", 0)
        tokens_left -= result["summary"].get("llm_reserved_tokens", 0)
        entry = {
            "node": "evidence.product",
            "sku": product["sku"],
            **result["summary"],
            "assessment": result["verdict"],
        }
        traces.append(entry)
        if result["verdict"]["next_action"] != "answer":
            continue
        item = dict(product)
        item["evidence_assessment"] = result["verdict"]

        allowed = set(result["verdict"]["allowed_evidence_ids"])
        item["knowledge_evidence"] = [
            asdict(p)
            for ref in result["bundle"].evidence
            if ref.unit_id
            and ref.evidence_id in allowed
            and (p := hydrate(ref.unit_id))
        ]
        item["asset_evidence"] = [
            e for e in item["knowledge_evidence"] if e.get("asset_id")
        ]
        item["visual_observation"] = " ".join(
            o["observation"] for o in result["verdict"]["visual_observations"]
        )
        accepted.append(item)
    return accepted, traces


def policy_workflow(
    *,
    query,
    topic,
    initial_payload,
    retrieve: object,
    assess=None,
    retrieval_arguments,
    started_at=None,
    enabled=None,
    order_context=None,
    user_id=None,
    original_question=None,
):
    from .evidence_workflow import _run_adaptive

    legacy_kwargs = {
        "query": query,
        "topic": topic,
        "initial_payload": initial_payload,
        "retrieve": retrieve,
        "assess": assess,
        "retrieval_arguments": retrieval_arguments,
        "started_at": started_at,
        "enabled": enabled,
    }
    if mode() == "legacy":
        return _run_adaptive(**legacy_kwargs)
    if mode() == "shadow":
        from .evidence_shadow import schedule

        frozen = deepcopy(initial_payload)
        frozen_args = deepcopy(retrieval_arguments)
        legacy = _run_adaptive(**legacy_kwargs)
        entry = schedule(
            lambda: policy_workflow(
                query=query,
                topic=topic,
                initial_payload=frozen,
                retrieve=retrieve,
                assess=assess,
                retrieval_arguments=frozen_args,
                enabled=enabled,
                order_context=deepcopy(order_context),
                user_id=user_id,
                original_question=original_question,
            ),
            task_id="policy-shadow",
        )
        legacy["trace"].append(entry)
        return legacy
    args = deepcopy(retrieval_arguments)
    from .config import settings

    args.setdefault("platform", settings.commerce_platform)
    if args.get("as_of") is None:
        from datetime import date

        args["as_of"] = (
            (initial_payload.get("trace") or {}).get("hard_filter") or {}
        ).get("effective_on") or date.today().isoformat()
    pc = args.get("product_context") or {}
    sku = pc.get("sku")
    rows = get_products([sku]) if sku else []
    categories = [
        str(p.get(k))
        for p in rows
        for k in ("category", "category_label", "category_path")
        if p.get(k)
    ]
    args["product_categories"] = categories
    from .asset_evidence import _POLICY_RIGHTS

    assessment_question = original_question or query
    technical = requires_visual(assessment_question) and not _POLICY_RIGHTS.search(
        assessment_question
    )
    bundle = policy_bundle(
        assessment_question,
        initial_payload.get("documents") or [],
        topic=topic,
        arguments=args,
        technical=technical,
        extra_products=[sku] if sku else [],
    )
    if order_context and user_id:
        with db_session() as db:
            order = db.execute(
                "SELECT * FROM orders WHERE order_id=? AND user_id=?",
                (order_context.get("order_id"), user_id),
            ).fetchone()
        if order:
            facts = {**dict(order), "authorized": True}
            oid = order["order_id"]
            bundle.domains.append("order")
            bundle.evidence.append(
                EvidenceRef(
                    evidence_id="order:" + oid,
                    kind="order",
                    domain="order",
                    target_id=oid,
                    source_id="order_query",
                    version=fingerprint(facts),
                    authority="order_service",
                    evidence_type="order_fact",
                    text=str(facts),
                    facts=facts,
                    validated=True,
                    snapshot=fingerprint(facts),
                )
            )
            bundle.business_context["user_id"] = user_id
            bundle.retrieval_scope["order"] = [oid]
            bundle.aspects.append(
                RequiredAspect(aspect="订单适用事实", domain="order", target_id=oid)
            )
    # Cross-domain technical retrieval is authorized only by the resolved SKU
    # and an explicit technical intent in the original question.
    technical_requested = bool(
        sku and (requires_asset(assessment_question) or "更换" in assessment_question)
    )
    if technical_requested:
        if not _POLICY_RIGHTS.search(assessment_question):
            bundle.aspects = []
        else:
            bundle.aspects[0].aspect = "政策权益及适用条件"
        bundle.aspects.append(
            RequiredAspect(
                aspect="技术步骤", domain="product", target_id=sku, purpose="technical"
            )
        )

    def adapter(proposal, current, scope):
        if proposal.domain == "product":
            if not technical_requested:
                raise ValueError("technical_scope_denied")
            return product_retrieve(proposal, current, scope)
        if proposal.domain == "order" and proposal.target_id in scope.get("order", []):
            return [
                ref
                for ref in current.evidence
                if ref.kind == "order" and ref.target_id == proposal.target_id
            ]
        if proposal.domain != "policy":
            raise ValueError("domain_denied")
        # Original platform/product/date stay frozen; model only supplies text.
        result = retrieve(
            {
                **{
                    k: v
                    for k, v in args.items()
                    if k
                    in {
                        "platform",
                        "product_context",
                        "as_of",
                        "limit",
                        "jurisdiction",
                        "version",
                        "authority",
                    }
                },
                "query": proposal.query,
                "topic": proposal.target_id.removeprefix("topic:"),
            }
        )
        if not result.get("ok"):
            raise RuntimeError("policy_retrieval_failed")
        additions = []
        for doc in (result.get("data") or {}).get("documents", []):
            ref = knowledge_ref(doc.get("unit_id") or doc.get("chunk_id"))
            if ref:
                ref.record = dict(doc)
                additions.append(ref)
        return additions

    execution_budget = {"max_hops": 0} if enabled is False else {}
    if started_at is not None:
        import time

        execution_budget["timeout"] = max(
            0,
            started_at + settings.policy_followup_timeout_seconds - time.perf_counter(),
        )
    unified = run_evidence_workflow(
        initial_bundle=bundle,
        policy=PolicyEvidencePolicy(),
        retrieve=adapter,
        budget=execution_budget,
    )
    # Match the old output contract, with the final exact snapshot.
    documents = []
    technical_documents = []
    refs = {r.evidence_id: r for r in unified["bundle"].evidence}
    observations = {
        o["evidence_id"]: o for o in unified["verdict"]["visual_observations"]
    }
    for eid in unified["summary"]["allowed_evidence_ids"]:
        ref = refs[eid]
        if ref.domain == "order":
            continue
        if ref.domain == "product":
            technical_documents.append(ref.model_dump())
            continue
        doc = dict(ref.record)
        doc.update(content=ref.text, context=ref.text, evidence=ref.text)
        if ref.asset_id:
            p = hydrate(ref.unit_id)
            obs = observations.get(eid)
            value = obs["observation"] if obs else ref.text
            doc["asset_evidence"] = asset_citation(p, value, verified=bool(obs))
            doc.update(content=value, context=value, evidence=value)
        documents.append(doc)
    unified["documents"] = documents
    unified["technical_documents"] = technical_documents
    unified["assessment_documents"] = documents
    unified["assessment"]["supporting_asset_ids"] = [
        d["asset_evidence"]["asset_id"] for d in documents if d.get("asset_evidence")
    ]
    return unified


def order_workflow(question, order, user_id):
    # SQL verification is independent of the tool/model's assertions.
    with db_session() as db:
        row = db.execute(
            "SELECT * FROM orders WHERE order_id=? AND user_id=?",
            (order["order_id"], user_id),
        ).fetchone()
    facts = dict(row) if row else {}
    facts["authorized"] = bool(row)
    ref = EvidenceRef(
        evidence_id="order:" + order["order_id"],
        kind="order",
        domain="order",
        target_id=order["order_id"],
        source_id="order_query",
        version=fingerprint(facts),
        authority="order_service",
        evidence_type="order_fact",
        text=str(facts),
        facts=facts,
        validated=bool(row),
        snapshot=fingerprint(facts),
    )
    bundle = EvidenceBundle(
        task_id="order",
        question=question,
        domains=["order"],
        evidence=[ref],
        aspects=[
            RequiredAspect(
                aspect="order_facts", domain="order", target_id=ref.target_id
            )
        ],
        business_context={"user_id": user_id},
        retrieval_scope={"order": [ref.target_id], "user_fields": ["order_id"]},
    )
    return run_evidence_workflow(
        initial_bundle=bundle, policy=OrderEvidencePolicy(), retrieve=lambda *_: []
    )
