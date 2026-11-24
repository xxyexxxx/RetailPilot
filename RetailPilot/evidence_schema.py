"""Strict request-scoped evidence protocol. No new knowledge persistence."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Domain = Literal["product", "policy", "order"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class PolicyApplicability(StrictModel):
    doc_id: str
    policy_id: str
    topic: str
    platform: str
    jurisdiction: str
    audience: str
    category_scope: str
    effective_from: str
    effective_to: str
    status: str
    priority: int


class EvidenceRef(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=200)
    kind: Literal["knowledge", "catalog", "order", "visual"]
    domain: Domain
    target_id: str
    unit_id: str | None = None
    asset_id: str | None = None
    element_id: str | None = None
    source_id: str = ""
    version: str
    authority: str
    evidence_type: str
    text: str
    facts: dict = Field(default_factory=dict)
    location: dict = Field(default_factory=dict)
    validated: bool = False
    snapshot: str = ""
    # Trusted adapter snapshot used by existing generators/citation contracts.
    policy_metadata: PolicyApplicability | None = None
    record: dict = Field(default_factory=dict, exclude=True)

    @model_validator(mode="after")
    def source_identity(self):
        if self.kind in {"knowledge", "visual"} and not self.unit_id:
            raise ValueError("knowledge evidence requires a real unit id")
        if self.kind == "order" and self.unit_id:
            raise ValueError("orders are not knowledge units")
        if self.kind == "visual" and not self.asset_id:
            raise ValueError("visual evidence requires original asset")
        return self


class RequiredAspect(StrictModel):
    aspect: str = Field(min_length=1, max_length=160)
    domain: Domain
    target_id: str
    required: bool = True
    field: str | None = None
    purpose: Literal["fact", "technical", "rights"] = "fact"


class EvidenceBundle(StrictModel):
    task_id: str
    question: str
    domains: list[Domain]
    evidence: list[EvidenceRef]
    aspects: list[RequiredAspect]
    business_context: dict = Field(default_factory=dict)
    hard_constraints: dict = Field(default_factory=dict)
    retrieval_scope: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique(self):
        ids = [e.evidence_id for e in self.evidence]
        keys = [(a.aspect, a.target_id) for a in self.aspects]
        if len(ids) != len(set(ids)) or len(keys) != len(set(keys)):
            raise ValueError("duplicate evidence/aspect")
        if any(e.domain not in self.domains for e in self.evidence):
            raise ValueError("unauthorized evidence domain")
        return self


class VisualObservation(StrictModel):
    evidence_id: str
    unit_id: str
    asset_id: str
    target_id: str
    version: str
    snapshot: str
    observation: str = Field(min_length=1, max_length=1500)


class AspectAssessment(StrictModel):
    aspect: str
    target_id: str
    required: bool
    status: Literal["supported", "partially_supported", "unsupported", "conflicted"]
    supporting_evidence_ids: list[str] = Field(default_factory=list, max_length=24)


class FollowupQuery(StrictModel):
    aspect: str = Field(min_length=1, max_length=160)
    domain: Domain
    target_id: str
    query: str = Field(min_length=1, max_length=500)


class EvidenceAssessment(StrictModel):
    next_action: Literal["answer", "retrieve_more", "ask_user", "refuse"]
    coverage: Literal["complete", "qualified", "insufficient"]
    conflict: bool
    reason: str = Field(min_length=1, max_length=500)
    aspects: list[AspectAssessment] = Field(max_length=32)
    required_user_fields: list[str] = Field(default_factory=list, max_length=8)
    clarification_question: str | None = None
    suggested_queries: list[FollowupQuery] = Field(default_factory=list, max_length=4)
    visual_observations: list[VisualObservation] = Field(
        default_factory=list, max_length=5
    )
    allowed_evidence_ids: list[str] = Field(default_factory=list, max_length=32)
    unconfirmed_conditions: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def consistent(self):
        missing = [a for a in self.aspects if a.required and a.status != "supported"]
        if any(a.status == "conflicted" for a in self.aspects) and not self.conflict:
            raise ValueError("ignored evidence conflict")
        if self.conflict and self.next_action != "refuse":
            raise ValueError("unresolved conflict")
        if self.next_action == "answer" and (
            missing
            or self.coverage == "insufficient"
            or not self.allowed_evidence_ids
            or self.required_user_fields
        ):
            raise ValueError("unsupported necessary claim")
        if (
            self.coverage == "qualified"
            and self.next_action == "answer"
            and not self.unconfirmed_conditions
        ):
            raise ValueError("qualified answer requires explicit limits")
        if self.next_action == "retrieve_more" and (
            not missing or not self.suggested_queries or self.required_user_fields
        ):
            raise ValueError("retrieval requires concrete missing aspects")
        if self.next_action == "ask_user" and (
            not self.required_user_fields or not self.clarification_question
        ):
            raise ValueError("clarification requires missing user facts")
        if self.next_action != "ask_user" and self.clarification_question:
            raise ValueError("unexpected clarification")
        if self.next_action != "retrieve_more" and self.suggested_queries:
            raise ValueError("unexpected retrieval")
        return self


def validate_assessment(raw, bundle, *, sent_image_ids=()):
    verdict = (
        raw
        if isinstance(raw, EvidenceAssessment)
        else EvidenceAssessment.model_validate(raw)
    )
    refs = {e.evidence_id: e for e in bundle.evidence if e.validated}
    expected = {(a.aspect, a.target_id): a for a in bundle.aspects}
    actual = {(a.aspect, a.target_id): a for a in verdict.aspects}
    if len(actual) != len(verdict.aspects) or set(actual) != set(expected):
        raise ValueError("assessment omitted or invented aspects")
    observations = {}
    for o in verdict.visual_observations:
        ref = refs.get(o.evidence_id)
        if (
            not ref
            or o.evidence_id not in sent_image_ids
            or (o.unit_id, o.asset_id, o.target_id, o.version, o.snapshot)
            != (ref.unit_id, ref.asset_id, ref.target_id, ref.version, ref.snapshot)
        ):
            raise ValueError("invalid visual reference")
        observations[o.evidence_id] = o
    for key, a in actual.items():
        requirement = expected[key]
        if a.required != requirement.required:
            raise ValueError("model modified necessary condition")
        if len(a.supporting_evidence_ids) != len(set(a.supporting_evidence_ids)):
            raise ValueError("duplicate supporting evidence")
        if (
            a.status in {"supported", "partially_supported"}
            and not a.supporting_evidence_ids
        ):
            raise ValueError("unmapped aspect")
        for eid in a.supporting_evidence_ids:
            ref = refs.get(eid)
            if (
                not ref
                or (requirement.domain != "policy" and ref.target_id != a.target_id)
                or ref.domain != requirement.domain
            ):
                raise ValueError("cross-target or cross-domain evidence")
            if ref.domain == "policy" and (
                ref.authority not in {"platform", "manufacturer", "official", "seller"}
                or (
                    requirement.purpose != "technical"
                    and ref.evidence_type
                    in {
                        "seller_advertised",
                        "vlm_observed",
                        "image",
                        "visual_observation",
                    }
                )
            ):
                raise ValueError("non-authoritative policy evidence")
            if ref.kind == "order" and not ref.facts.get("authorized"):
                raise ValueError("unauthorized order")
            from .asset_evidence import requires_visual

            needs_original = (
                ref.asset_id
                and ref.evidence_type not in {"document_text", "structured_table"}
                and (
                    requirement.purpose == "technical"
                    or requires_visual(bundle.question)
                )
            )
            if (
                needs_original or ref.evidence_type in {"image", "vlm_observed"}
            ) and eid not in observations:
                raise ValueError("unverified visual inference")
    mapped = {
        eid
        for a in verdict.aspects
        if a.status == "supported"
        for eid in a.supporting_evidence_ids
    }
    if not set(verdict.allowed_evidence_ids) <= mapped:
        raise ValueError("allowed claims exceed supported evidence")
    if verdict.next_action == "answer" and not all(
        set(a.supporting_evidence_ids) & set(verdict.allowed_evidence_ids)
        for a in verdict.aspects
        if a.required
    ):
        raise ValueError("answer omitted necessary evidence")
    for q in verdict.suggested_queries:
        aspect = actual.get((q.aspect, q.target_id))
        if q.domain == "policy":
            aspect = next(
                (
                    a
                    for a in verdict.aspects
                    if a.aspect == q.aspect
                    and expected[(a.aspect, a.target_id)].domain == "policy"
                ),
                None,
            )
        if (
            not aspect
            or aspect.status == "supported"
            or q.domain != expected[(aspect.aspect, aspect.target_id)].domain
        ):
            raise ValueError("query must target an actual missing aspect")
        allowed = bundle.retrieval_scope.get(q.domain, [])
        if q.target_id not in allowed:
            raise ValueError("followup target outside initial authorization")
    if verdict.next_action == "ask_user" and not set(
        verdict.required_user_fields
    ) <= set(bundle.retrieval_scope.get("user_fields", [])):
        raise ValueError("uncontrolled user field request")
    if verdict.next_action == "ask_user":
        known = dict(bundle.business_context)
        if known.get("skus"):
            known["sku"] = known["skus"][0]
        if any(e.kind == "order" and e.validated for e in bundle.evidence):
            known["order_id"] = "authorized"
        if known.get("sku") or known.get("order_id"):
            known["product_or_order"] = "resolved"
        if any(known.get(field) for field in verdict.required_user_fields):
            raise ValueError("requested user fact already known")
    return verdict
