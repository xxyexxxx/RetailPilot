from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .text import tokenize

GROUNDING_VERSION = "claim-citation-v3"


def citation_key(citation: dict[str, Any]) -> str:
    identity = {
        key: citation.get(key)
        for key in (
            "sku",
            "chunk_id",
            "doc_id",
            "order_id",
            "field",
            "value",
            "source",
        )
        if citation.get(key) is not None
    }
    raw = json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def attach_citation_ids(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in citations:
        citation = dict(raw)
        citation["citation_id"] = citation.get("citation_id") or citation_key(citation)
        if citation["citation_id"] not in seen:
            seen.add(citation["citation_id"])
            result.append(citation)
    return result


def claim(
    text: str,
    citation: dict[str, Any],
    *,
    claim_type: str = "fact",
) -> dict[str, Any]:
    normalized = dict(citation)
    normalized["citation_id"] = normalized.get("citation_id") or citation_key(normalized)
    return {
        "claim": text,
        "claim_type": claim_type,
        "entity_id": (
            normalized.get("sku")
            or normalized.get("doc_id")
            or normalized.get("order_id")
            or ""
        ),
        "field": normalized.get("field"),
        "value": normalized.get("value"),
        "citation_id": normalized["citation_id"],
    }


def policy_citation(document: dict[str, Any]) -> dict[str, Any]:
    """Build policy citations from the actual snapshot, never model metadata."""
    return {
        "chunk_id": document["chunk_id"], "doc_id": document["doc_id"],
        "field": "content", "value": document["content"], "quoted_text": document["content"],
        "source": document.get("source", ""), "source_uri": document.get("source_uri", ""),
        "section_path": document.get("section_path", ""), "page_no": document.get("page_no"),
        "char_start": document.get("char_start"), "char_end": document.get("char_end"),
        "version": document.get("version", ""), "effective_from": document.get("effective_from", ""),
        "effective_to": document.get("effective_to", ""),
    }


def rebuild_policy_claims(
    statements: list[dict[str, Any]], documents: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Rebuild every claim/citation for repaired text, then run normal Grounding.

    Structural validation is necessary, but the semantic Faithfulness checker
    must still recheck the entire final answer after this function succeeds.
    """
    by_id = {document["chunk_id"]: document for document in documents}
    claims: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    lines: list[str] = []
    used: set[str] = set()
    for statement in statements:
        text = statement["text"].strip()
        ids = statement["chunk_ids"]
        if not text or not ids or len(ids) != len(set(ids)) or not set(ids) <= set(by_id):
            raise ValueError("invalid repair evidence mapping")
        lines.append(text)
        for cid in ids:
            citation = policy_citation(by_id[cid])
            citations.append(citation)
            claims.append(claim(text, citation, claim_type="policy_rule"))
            used.add(cid)
    if not lines or used != set(by_id):
        raise ValueError("repair omitted reviewed policy evidence")
    citations = attach_citation_ids(citations)
    verdict = validate_claim_citations(claims, citations, policy_documents=documents)
    return "\n".join(lines), claims, citations, verdict


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) <= 1e-6
    return left == right


def _normalized_text(value: Any) -> str:
    text = str(value or "").strip()
    if isinstance(value, float) and value.is_integer():
        text = str(int(value))
    return re.sub(r"\s+", " ", text).casefold()


def _claim_text_supported(claim_text: str, source_value: Any, field: str) -> bool:
    """Check that the generated claim text can be traced to the cited value.

    Structured fields must contain the cited value verbatim; long free-text
    fields are verified through shared lexical evidence instead of demanding
    the summary to quote an entire document.
    """

    claim = _normalized_text(claim_text)
    source = _normalized_text(source_value)
    if not claim or not source:
        return True
    if field not in {"description", "content", "latest_logistics"}:
        return source in claim
    source_tokens = {token for token in tokenize(source) if len(token) > 1 or token.isdigit()}
    claim_tokens = {token for token in tokenize(claim) if len(token) > 1 or token.isdigit()}
    return len(source_tokens & claim_tokens) >= 2


def validate_claim_citations(
    claims: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    *,
    products: list[dict[str, Any]] | None = None,
    policy_documents: list[dict[str, Any]] | None = None,
    order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Hard-check structured response facts against cited source records."""

    normalized = attach_citation_ids(citations)
    citation_by_id = {item["citation_id"]: item for item in normalized}
    product_by_sku = {str(item.get("sku")): item for item in products or []}
    document_by_id = {str(item.get("doc_id")): item for item in policy_documents or []}
    document_by_chunk = {
        str(item.get("chunk_id")): item
        for item in policy_documents or []
        if item.get("chunk_id")
    }
    issues: list[dict[str, Any]] = []

    for index, item in enumerate(claims):
        citation_id = str(item.get("citation_id") or "")
        citation = citation_by_id.get(citation_id)
        if not citation:
            issues.append({"index": index, "type": "unsupported_claim", "claim": item.get("claim")})
            continue
        if item.get("field") != citation.get("field") or not _same_value(
            item.get("value"), citation.get("value")
        ):
            issues.append({"index": index, "type": "citation_mismatch", "claim": item.get("claim")})
            continue

        field = str(citation.get("field") or "")
        value = citation.get("value")
        source_mismatch = False
        if citation.get("sku"):
            product = product_by_sku.get(str(citation["sku"]))
            if not product:
                issues.append({"index": index, "type": "citation_entity_missing", "entity": citation["sku"]})
                continue
            if field == "features" and value not in product.get("features", []):
                source_mismatch = True
            elif field == "description" and str(value) not in str(product.get("description") or ""):
                source_mismatch = True
            elif field not in {"features", "description"} and not _same_value(
                product.get(field), value
            ):
                source_mismatch = True
        elif citation.get("doc_id"):
            document = document_by_chunk.get(str(citation.get("chunk_id") or ""))
            if document is None and not citation.get("chunk_id"):
                document = document_by_id.get(str(citation["doc_id"]))
            if not document:
                issues.append({"index": index, "type": "citation_entity_missing", "entity": citation["doc_id"]})
                continue
            if str(document.get("doc_id")) != str(citation["doc_id"]):
                source_mismatch = True
            elif field == "content":
                if (
                    str(value) != str(document.get("content"))
                    or str(citation.get("quoted_text") or value) != str(value)
                ):
                    source_mismatch = True
            elif not _same_value(document.get(field), value):
                # Policy citations are no longer exempt from field/value checks.
                source_mismatch = True
        elif citation.get("order_id"):
            if not order or str(order.get("order_id")) != str(citation["order_id"]):
                issues.append({"index": index, "type": "citation_entity_missing", "entity": citation["order_id"]})
                continue
            if field == "status":
                source_mismatch = not _same_value(order.get("status"), value)
            elif field == "latest_logistics":
                events = order.get("events") or []
                source_mismatch = not events or not _same_value(
                    events[-1].get("description") or events[-1].get("status"), value
                )
            elif field == "logistics_events":
                source_mismatch = not _same_value(order.get("events") or [], value)
            else:
                source_mismatch = not _same_value(order.get(field), value)
        else:
            issues.append(
                {
                    "index": index,
                    "type": "citation_entity_missing",
                    "entity": str(citation.get("sku") or citation.get("doc_id") or citation.get("order_id") or ""),
                }
            )
            continue

        if source_mismatch:
            issues.append(
                {
                    "index": index,
                    "type": "citation_source_mismatch",
                    "field": field,
                    "claim": item.get("claim"),
                }
            )
        elif field not in {"logistics_events"} and not _claim_text_supported(
            str(item.get("claim") or ""), value, field
        ):
            issues.append(
                {
                    "index": index,
                    "type": "citation_source_mismatch",
                    "field": field,
                    "claim": item.get("claim"),
                    "reason": "claim text is not traceable to the cited source value",
                }
            )

    issue_types = sorted({str(item["type"]) for item in issues})
    return {
        "version": GROUNDING_VERSION,
        "pass": not issues,
        "claims": len(claims),
        "citations": len(normalized),
        "supported_claim_rate": round((len(claims) - len(issues)) / max(1, len(claims)), 4),
        "failure_types": issue_types,
        "issues": issues,
    }
