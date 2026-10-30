"""Validated semantic verdicts. Retrieval scores never authorize an answer."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .policy import PolicyTopic


class FollowupQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    aspect: str = Field(min_length=2, max_length=160)
    query: str = Field(min_length=2, max_length=500)
    topic: PolicyTopic


class PolicyEvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    decision: Literal["answer", "clarify", "refuse"]
    next_action: Literal["answer", "retrieve_more", "ask_user", "refuse"]
    sufficient: bool
    conflict: bool
    reason: str = Field(min_length=1, max_length=300)
    clarification_question: str | None = Field(default=None, max_length=300)
    missing_aspects: list[str] = Field(default_factory=list, max_length=8)
    covered_aspects: list[str] = Field(default_factory=list, max_length=8)
    required_user_fields: list[str] = Field(default_factory=list, max_length=8)
    supporting_chunk_ids: list[str] = Field(default_factory=list, max_length=24)
    aspect_evidence_map: dict[str, list[str]] = Field(default_factory=dict)
    suggested_queries: list[FollowupQuery] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def consistent(self) -> PolicyEvidenceAssessment:
        expected = {
            "answer": "answer",
            "retrieve_more": "clarify",
            "ask_user": "clarify",
            "refuse": "refuse",
        }
        if self.decision != expected[self.next_action]:
            raise ValueError("decision/next_action mismatch")
        if self.sufficient != (self.next_action == "answer") or (
            self.conflict and self.next_action != "refuse"
        ):
            raise ValueError("contradictory evidence verdict")
        for values in (
            self.missing_aspects,
            self.covered_aspects,
            self.required_user_fields,
            self.supporting_chunk_ids,
        ):
            if any(not value.strip() or len(value) > 160 for value in values) or len(
                values
            ) != len(set(values)):
                raise ValueError("invalid or duplicate evidence field")
        if set(self.covered_aspects) & set(self.missing_aspects):
            raise ValueError("aspect cannot be covered and missing")
        if len(self.aspect_evidence_map) > 8 or set(self.aspect_evidence_map) != set(
            self.covered_aspects
        ):
            raise ValueError("every covered aspect needs an evidence mapping")
        mapped = set()
        for ids in self.aspect_evidence_map.values():
            if not ids or len(ids) > 24:
                raise ValueError("empty or oversized aspect mapping")
            mapped.update(ids)
        if mapped != set(self.supporting_chunk_ids):
            raise ValueError("support ids must match aspect mappings")
        if self.next_action == "answer" and (
            self.missing_aspects or not mapped or self.required_user_fields
        ):
            raise ValueError("answer requires complete mapped evidence")
        if self.next_action == "retrieve_more" and (
            not self.missing_aspects or self.required_user_fields
        ):
            raise ValueError(
                "retrieval requires missing policy aspects, not personal facts"
            )
        if self.next_action == "ask_user" and (
            not self.required_user_fields or not self.clarification_question
        ):
            raise ValueError("ask_user requires explicit missing user information")
        if self.next_action != "ask_user" and self.clarification_question:
            raise ValueError("unexpected clarification")
        if self.next_action != "retrieve_more" and self.suggested_queries:
            raise ValueError("unexpected followup queries")
        if any(
            item.aspect not in self.missing_aspects for item in self.suggested_queries
        ):
            raise ValueError("query must target a missing aspect")
        return self


def refusal(reason: str, *, conflict: bool = False) -> dict[str, Any]:
    return PolicyEvidenceAssessment(
        decision="refuse",
        next_action="refuse",
        sufficient=False,
        conflict=conflict,
        reason=reason[:300],
    ).model_dump()


def validate_assessment(
    raw: dict[str, Any], documents: list[dict[str, Any]]
) -> dict[str, Any]:
    """Legacy non-answer verdicts remain safe; legacy answers need coverage."""
    data = dict(raw)
    if "next_action" not in data:
        if data.get("decision") == "clarify":
            data["next_action"] = "ask_user"
            data.setdefault("required_user_fields", ["product_or_order"])
        else:
            data["next_action"] = data.get("decision")
    verdict = PolicyEvidenceAssessment.model_validate(data)
    available = {item["chunk_id"] for item in documents if item.get("chunk_id")}
    if not set(verdict.supporting_chunk_ids) <= available:
        raise ValueError("unknown supporting chunk id")
    return verdict.model_dump()
