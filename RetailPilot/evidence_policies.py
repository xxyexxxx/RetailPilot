"""Domain rules: model proposals cannot change authoritative scope."""

from __future__ import annotations

import hashlib
import json
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict
from datetime import date

verification_cache = ContextVar("evidence_independent_verification", default=None)
from typing import Protocol

from .catalog import get_products
from .database import db_session
from .evidence_schema import (
    AspectAssessment,
    EvidenceAssessment,
    EvidenceBundle,
    EvidenceRef,
    RequiredAspect,
    validate_assessment,
)
from .knowledge_evidence import hydrate


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()


def knowledge_ref(unit_id):
    p = hydrate(unit_id, include_catalog=True)
    if not p:
        return None
    from .evidence_schema import PolicyApplicability

    applicability = None
    authority = "catalog"
    if p.doc_id:
        with db_session() as db:
            row = db.execute(
                "SELECT * FROM policy_documents WHERE doc_id=?", (p.doc_id,)
            ).fetchone()
            authority = row["authority"] if row else ""
            if row:
                applicability = PolicyApplicability(
                    **{k: row[k] for k in PolicyApplicability.model_fields}
                )
    return EvidenceRef(
        evidence_id=unit_id,
        kind="knowledge",
        domain="product" if p.sku else "policy",
        target_id=p.sku or p.doc_id,
        unit_id=unit_id,
        asset_id=p.asset_id,
        element_id=p.element_id,
        source_id=p.source_id,
        version=p.source_location["source_version"],
        authority=authority,
        evidence_type=p.evidence_type,
        text=p.text_content,
        facts=p.structured_content or {},
        location=p.source_location,
        validated=True,
        policy_metadata=applicability,
        snapshot=fingerprint(asdict(p)),
    )


class EvidencePolicy(Protocol):
    def validate_constraints(self, bundle): ...
    def build_assessment_context(self, bundle): ...
    def validate_verdict(self, verdict, bundle, **kwargs): ...
    def build_followup_scope(self, verdict, bundle): ...


class DomainEvidencePolicy:
    def validate_constraints(self, bundle):
        for ref in bundle.evidence:
            if not ref.validated:
                raise ValueError("unvalidated evidence")
            if ref.unit_id:
                fresh = knowledge_ref(ref.unit_id)
                fields = (
                    "kind",
                    "domain",
                    "target_id",
                    "unit_id",
                    "asset_id",
                    "element_id",
                    "source_id",
                    "version",
                    "authority",
                    "evidence_type",
                    "text",
                    "facts",
                    "location",
                    "snapshot",
                    "policy_metadata",
                )
                if not fresh or any(
                    getattr(fresh, k) != getattr(ref, k) for k in fields
                ):
                    raise ValueError("evidence_snapshot_changed")
            if ref.kind == "catalog":
                rows = get_products([ref.target_id])
                if (
                    not rows
                    or ref.text != json.dumps(ref.facts, ensure_ascii=False)
                    or any(rows[0].get(k) != v for k, v in ref.facts.items())
                ):
                    raise ValueError("catalog_snapshot_changed")
            if ref.domain == "product":
                rows = get_products([ref.target_id])
                if not rows:
                    raise ValueError("missing_product")
                p = rows[0]
                h = bundle.hard_constraints
                if (h.get("price_max") is not None and p["price"] > h["price_max"]) or (
                    h.get("price_min") is not None and p["price"] < h["price_min"]
                ):
                    raise ValueError("price_constraint")
                if (
                    p["brand"] in h.get("negative_brands", [])
                    or (h.get("category") and p["category"] != h["category"])
                    or p["stock"] <= 0
                ):
                    raise ValueError("product_hard_constraint")
                if h.get("brands") and p["brand"] not in h["brands"]:
                    raise ValueError("brand_constraint")
                text = (
                    p["title"]
                    + " "
                    + p.get("description", "")
                    + " "
                    + str(p.get("attributes", {}))
                ).casefold()
                if any(t.casefold() in text for t in h.get("negative_terms", [])):
                    raise ValueError("excluded_product")
                if any(
                    t.casefold() in p["title"].casefold()
                    for t in h.get("excluded_title_terms", [])
                ):
                    raise ValueError("excluded_title")
            if ref.kind == "order":
                user = bundle.business_context.get("user_id")
                with db_session() as db:
                    row = db.execute(
                        "SELECT * FROM orders WHERE order_id=? AND user_id=?",
                        (ref.target_id, user),
                    ).fetchone()
                if (
                    not row
                    or ref.text != str(ref.facts)
                    or any(
                        dict(row).get(k) != v
                        for k, v in ref.facts.items()
                        if k != "authorized"
                    )
                ):
                    raise ValueError("order_authorization")
            if ref.domain == "policy":
                with db_session() as db:
                    doc = db.execute(
                        "SELECT * FROM policy_documents WHERE doc_id=?",
                        (ref.target_id,),
                    ).fetchone()
                scope = bundle.retrieval_scope.get("policy_filters", {})
                when = scope.get("as_of") or date.today().isoformat()
                when = when.isoformat() if hasattr(when, "isoformat") else when
                if (
                    not doc
                    or doc["status"] != "active"
                    or doc["version"] != ref.version
                    or doc["effective_from"] > when
                    or (doc["effective_to"] and doc["effective_to"] < when)
                ):
                    raise ValueError("policy_version_or_date")
                if scope.get("topics") and doc["topic"] not in scope["topics"]:
                    raise ValueError("policy_topic")
                if scope.get("platform") and doc["platform"] not in {
                    "*",
                    scope["platform"],
                }:
                    raise ValueError("policy_platform")
                if doc["audience"] not in scope.get(
                    "audiences", ["public", "customer"]
                ):
                    raise ValueError("policy_audience")
                if doc["jurisdiction"] not in {"*", scope.get("jurisdiction", "*")}:
                    raise ValueError("policy_jurisdiction")
                if doc["category_scope"] != "*" and doc[
                    "category_scope"
                ].casefold() not in {s.casefold() for s in scope.get("categories", [])}:
                    raise ValueError("policy_category")
                if scope.get("version") and scope["version"] != doc["version"]:
                    raise ValueError("policy_version")
                if scope.get("authority") and scope["authority"] != doc["authority"]:
                    raise ValueError("policy_authority")
                if ref.authority != doc["authority"]:
                    raise ValueError("policy_authority_mismatch")
        return bundle

    def build_assessment_context(self, bundle):
        return bundle.model_dump()

    def validate_verdict(self, verdict, bundle, **kwargs):
        self.validate_constraints(bundle)
        checked = validate_assessment(verdict, bundle, **kwargs)
        import os

        if (
            os.getenv("RETAILPILOT_EVIDENCE_INDEPENDENT_VISION_CHECK", "false")
            == "true"
        ):
            from .asset_evidence import verify_image

            cache = verification_cache.get()
            if cache is None:
                cache = {}
            for obs in checked.visual_observations[:1]:
                key = (bundle.question, obs.evidence_id, obs.snapshot, obs.observation)
                if key not in cache:
                    if len(cache) >= 1:
                        raise EvidenceConflict("independent_visual_budget")
                    package = hydrate(obs.unit_id)
                    cache[key] = bool(
                        package
                        and verify_image(
                            package,
                            bundle.question
                            + "；独立核对这条观察是否成立："
                            + obs.observation,
                        )
                    )
                if not cache[key]:
                    raise EvidenceConflict("independent_visual_check_failed")
        return checked

    def build_followup_scope(self, verdict, bundle):
        return deepcopy(bundle.retrieval_scope)

    def fast_path(self, bundle):
        return None


class EvidenceConflict(ValueError):
    pass


class ProductEvidencePolicy(DomainEvidencePolicy):
    def validate_constraints(self, bundle):
        super().validate_constraints(bundle)
        from .asset_evidence import _conflict

        for ref in bundle.evidence:
            if ref.asset_id and ref.unit_id:
                package = hydrate(ref.unit_id, include_catalog=True)
                if package and _conflict(package, ref.text, bundle.question):
                    raise EvidenceConflict("catalog_asset_conflict")
        return bundle

    def fast_path(self, bundle):
        if not bundle.retrieval_scope.get("deterministic"):
            return None
        return deterministic_assessment(bundle, "catalog")


class PolicyEvidencePolicy(DomainEvidencePolicy):
    def validate_verdict(self, verdict, bundle, **kwargs):
        checked = super().validate_verdict(verdict, bundle, **kwargs)
        refs = {e.evidence_id: e for e in bundle.evidence}
        for aspect in checked.aspects:
            if aspect.status == "supported":
                for eid in aspect.supporting_evidence_ids:
                    ref = refs[eid]
                    if ref.domain == "policy" and ref.policy_metadata is None:
                        raise ValueError("missing authoritative policy applicability")
        return checked

    def build_followup_scope(self, verdict, bundle):
        scope = deepcopy(bundle.retrieval_scope)
        from .evidence_workflow import FOLLOWUP_TOPICS

        original = scope.get("topic", "")
        allowed = FOLLOWUP_TOPICS.get(original, {original})
        scope["policy"] = [
            target
            for target in scope.get("policy", [])
            if target.removeprefix("topic:") in allowed
        ]
        return scope


class OrderEvidencePolicy(DomainEvidencePolicy):
    def fast_path(self, bundle):
        return deterministic_assessment(bundle, "order")


def deterministic_assessment(bundle, kind):
    aspects = []
    for a in bundle.aspects:
        refs = [
            e.evidence_id
            for e in bundle.evidence
            if e.kind == kind
            and e.target_id == a.target_id
            and (not a.field or a.field in e.facts)
        ]
        if not refs:
            return None
        aspects.append(
            AspectAssessment(
                aspect=a.aspect,
                target_id=a.target_id,
                required=a.required,
                status="supported",
                supporting_evidence_ids=refs,
            )
        )
    return EvidenceAssessment(
        next_action="answer",
        coverage="complete",
        conflict=False,
        reason="deterministic_authoritative_facts",
        aspects=aspects,
        allowed_evidence_ids=list(
            dict.fromkeys(e for a in aspects for e in a.supporting_evidence_ids)
        ),
    )


def product_bundle(
    question,
    products,
    *,
    task_id="product",
    constraints=None,
    requirements=None,
    deterministic=False,
):
    evidence = []
    aspects = []
    for product in products:
        sku = product["sku"]
        authoritative = get_products([sku])
        if not authoritative:
            continue
        facts = {
            k: authoritative[0][k]
            for k in (
                "title",
                "brand",
                "price",
                "stock",
                "category",
                "features",
                "attributes",
            )
        }
        evidence.append(
            EvidenceRef(
                evidence_id="catalog:" + sku,
                kind="catalog",
                domain="product",
                target_id=sku,
                source_id="catalog:" + sku,
                version=fingerprint(facts),
                authority="catalog",
                evidence_type="catalog_verified",
                text=json.dumps(facts, ensure_ascii=False),
                facts=facts,
                validated=True,
                snapshot=fingerprint(facts),
            )
        )
        for item in product.get("knowledge_evidence", []):
            ref = knowledge_ref(item.get("unit_id"))
            if ref and ref.target_id == sku:
                evidence.append(ref)
        aspect_names = list(requirements or [])
        if not aspect_names:
            aspect_names = ["catalog_candidate" if deterministic else question[:160]]
        elif "完整问题的必要条件" not in aspect_names:
            aspect_names.append("完整问题的必要条件")
        for aspect in aspect_names:
            aspects.append(
                RequiredAspect(aspect=aspect, domain="product", target_id=sku)
            )
    unique = {e.evidence_id: e for e in evidence}
    return EvidenceBundle(
        task_id=task_id,
        question=question,
        domains=["product"],
        evidence=list(unique.values()),
        aspects=aspects,
        hard_constraints=constraints or {},
        business_context={"skus": [p["sku"] for p in products]},
        retrieval_scope={
            "product": [p["sku"] for p in products],
            "deterministic": deterministic,
            "user_fields": ["sku", "product_or_order"],
        },
    )


def policy_bundle(
    question,
    documents,
    *,
    topic,
    arguments,
    task_id="policy",
    technical=False,
    extra_products=(),
):
    refs = []
    for doc in documents:
        ref = knowledge_ref(doc.get("unit_id") or doc.get("chunk_id"))
        if ref and ref.domain == "policy":
            ref.record = dict(doc)
            refs.append(ref)
    # An empty retrieval still has a controlled policy-topic target for follow-up.
    target = "topic:" + topic
    aspects = [
        RequiredAspect(
            aspect=question[:160],
            domain="policy",
            target_id=target,
            purpose="technical" if technical else "rights",
        )
    ]
    from .evidence_workflow import FOLLOWUP_TOPICS

    allowed_topics = sorted(FOLLOWUP_TOPICS.get(topic, {topic}))
    scope = {
        "policy": ["topic:" + t for t in allowed_topics],
        "product": list(extra_products),
        "topic": topic,
        "user_fields": ["order_id", "sku", "purchased_at", "product_or_order"],
        "policy_filters": {
            **arguments,
            "topics": allowed_topics,
            "categories": arguments.get("product_categories", []),
        },
    }
    return EvidenceBundle(
        task_id=task_id,
        question=question,
        domains=["policy", *(["product"] if extra_products else [])],
        evidence=refs,
        aspects=aspects,
        retrieval_scope=scope,
        business_context=arguments.get("product_context", {}),
    )
