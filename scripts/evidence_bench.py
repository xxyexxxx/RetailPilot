"""Synthetic controller benchmark. No live models; no claims of semantic model quality."""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from statistics import mean
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from RetailPilot.asset_config import asset_settings
from RetailPilot.asset_evidence import product_asset_answer
from RetailPilot.asset_ingestion import process_asset, stage_asset
from RetailPilot.asset_parsing import TesseractOCR, VisionProvider
from RetailPilot.catalog import get_product
from RetailPilot.config import settings
from RetailPilot.database import db_session, init_db
from RetailPilot.evidence_policies import knowledge_ref
from RetailPilot.evidence_runtime import evaluate_products, policy_workflow
from RetailPilot.evidence_schema import (
    AspectAssessment,
    EvidenceAssessment,
    EvidenceBundle,
    FollowupQuery,
    VisualObservation,
)
from RetailPilot.evidence_workflow import run_evidence_workflow
from RetailPilot.grounding import claim, policy_citation, validate_claim_citations
from RetailPilot.policy import retrieve_policies
from RetailPilot.policy_ingestion import PolicyDocumentInput, ingest_policy_documents

CASES = [
    ("policy_complete", "保修期限", "policy", "answer"),
    ("policy_followup", "保修例外", "policy", "answer"),
    ("policy_user_fact", "我的订单是否符合保修", "policy", "ask_user"),
    ("policy_flowchart", "流程图的接头在哪一侧", "policy", "answer"),
    ("product_price", "当前价格", "product", "answer"),
    ("product_followup", "宣传图中的water-pocket", "product", "answer"),
    ("product_visual", "图片中的绿色外观", "product", "answer"),
]


class Oracle:
    """Scripted fixture oracle. Call counts are actual Python invocations, not API usage."""

    def __init__(self, case):
        self.case = case
        self.calls = 0
        self.images = 0

    def generate(self, **kwargs):
        self.calls += 1
        self.images += len(kwargs.get("images") or [])
        context = json.loads(kwargs["messages"][-1]["content"])
        context.pop("validated_prior_observations", None)
        b = EvidenceBundle.model_validate(context)
        need = (
            self.case == "policy_followup"
            and not any(e.kind == "knowledge" for e in b.evidence)
        ) or (
            self.case == "product_followup" and not any(e.asset_id for e in b.evidence)
        )
        action = (
            "ask_user"
            if self.case == "policy_user_fact"
            else "retrieve_more"
            if need
            else "answer"
        )
        aspects = [
            AspectAssessment(
                aspect=a.aspect,
                target_id=a.target_id,
                required=a.required,
                status="supported" if action == "answer" else "unsupported",
                supporting_evidence_ids=[
                    e.evidence_id
                    for e in b.evidence
                    if e.domain == a.domain
                    and (a.domain == "policy" or e.target_id == a.target_id)
                ]
                if action == "answer"
                else [],
            )
            for a in b.aspects
        ]
        obs = []
        sent = {image.unit_id for image in kwargs.get("images", [])}
        for e in b.evidence:
            if e.unit_id in sent:
                obs.append(
                    VisualObservation(
                        evidence_id=e.evidence_id,
                        unit_id=e.unit_id,
                        asset_id=e.asset_id,
                        version=e.version,
                        snapshot=e.snapshot,
                        target_id=e.target_id,
                        observation="接头在右侧，绿色外观",
                    )
                )
        return EvidenceAssessment(
            next_action=action,
            coverage="complete" if action == "answer" else "insufficient",
            conflict=False,
            reason="synthetic oracle; not a real model",
            aspects=aspects,
            allowed_evidence_ids=list(
                dict.fromkeys(i for a in aspects for i in a.supporting_evidence_ids)
            ),
            required_user_fields=["order_id"] if action == "ask_user" else [],
            clarification_question="请提供订单号。" if action == "ask_user" else None,
            suggested_queries=[
                FollowupQuery(
                    aspect=b.aspects[0].aspect,
                    domain=b.aspects[0].domain,
                    target_id=b.aspects[0].target_id,
                    query="water-pocket 保修期限",
                )
            ]
            if need
            else [],
            visual_observations=obs,
        )

    def legacy(self, question, *, documents, **_):
        self.calls += 1
        action = (
            "ask_user"
            if self.case == "policy_user_fact"
            else "answer"
            if documents
            else "retrieve_more"
        )
        mapping = (
            {"necessary condition": [d["chunk_id"] for d in documents]}
            if action == "answer"
            else {}
        )
        return {
            "decision": {
                "answer": "answer",
                "ask_user": "clarify",
                "retrieve_more": "clarify",
            }[action],
            "next_action": action,
            "sufficient": action == "answer",
            "conflict": False,
            "reason": "synthetic oracle",
            "covered_aspects": list(mapping),
            "missing_aspects": ["necessary condition"]
            if action == "retrieve_more"
            else [],
            "aspect_evidence_map": mapping,
            "supporting_chunk_ids": [i for ids in mapping.values() for i in ids],
            "required_user_fields": ["order_id"] if action == "ask_user" else [],
            "clarification_question": "请提供订单号。"
            if action == "ask_user"
            else None,
            "suggested_queries": [],
        }


def setup(root):
    settings.database_path = root / "db.sqlite"
    asset_settings.store = root / "assets"
    init_db()
    with db_session() as db:
        db.execute(
            "INSERT INTO products(sku,title,category,category_label,price,stock,source) VALUES('bench','Alpine','Sports & Outdoors','Sports & Outdoors',70,10,'synthetic')"
        )
    ingest_policy_documents(
        [
            PolicyDocumentInput(
                doc_id="bench-policy",
                policy_id="bench-policy",
                topic="保修",
                title="保修期限",
                content="保修期限24个月；质量问题按条款核验，需要订单凭证。",
                source_name="synthetic",
                platform="bench",
                authority="platform",
                version="1",
            )
        ]
    )
    image = Image.new("RGB", (32, 32), "green")
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def add_image(data, policy=False):
    aid, _ = stage_asset(
        data,
        owner_type="policy" if policy else "product",
        owner_id="bench-policy" if policy else "bench",
        kind="document_picture" if policy else "product_poster",
        source_key="fixture.png",
        source_version="1" if policy else "",
    )
    process_asset(aid)
    with db_session() as db:
        uid = db.execute(
            "SELECT unit_id FROM knowledge_units JOIN knowledge_sources USING(source_id) WHERE asset_id=?",
            (aid,),
        ).fetchone()[0]
    return knowledge_ref(uid)


def execute(root, mode, case, question, domain, expected):
    oracle = Oracle(case)
    vision_calls = []
    data = setup(root)
    original = run_evidence_workflow

    def engine(**kwargs):
        kwargs["model"] = oracle
        return original(**kwargs)

    def vision(*_):
        vision_calls.append(True)
        return "接头在右侧，绿色外观"

    def lookup(_args):
        return {
            "ok": True,
            "data": retrieve_policies(
                "保修期限", topic="保修", platform="bench", authority="platform"
            ),
        }

    product = get_product("bench")
    initial = []
    if case == "policy_complete":
        initial = lookup({})["data"]["documents"]
    elif case == "policy_flowchart":
        ref = add_image(data, policy=True)
        initial = retrieve_policies(
            "保修期限", topic="保修", platform="bench", authority="platform"
        )["documents"]
        # Frozen retrieval fixture: the hit is the diagram, not the Markdown policy.
        initial = [d for d in initial if d["chunk_id"] == ref.unit_id]
        if not initial:
            p = __import__(
                "RetailPilot.knowledge_evidence", fromlist=["hydrate"]
            ).hydrate(ref.unit_id)
            initial = [
                {
                    "unit_id": ref.unit_id,
                    "chunk_id": ref.unit_id,
                    "doc_id": "bench-policy",
                    "content": "",
                    "context": "",
                    "source": "synthetic",
                    "source_uri": p.source_location["source_uri"],
                    "title": "Diagram",
                    "section_path": "",
                    "version": "1",
                    "platform": "bench",
                    "category": "*",
                    "topic": "保修",
                    "audience": "customer",
                    "effective_from": "0001-01-01",
                    "effective_to": "",
                    "authority": "platform",
                    "chunk_metadata": {"asset_id": ref.asset_id},
                }
            ]
    if case == "product_visual":
        ref = add_image(data)
        product["knowledge_evidence"] = [
            {"unit_id": ref.unit_id, "asset_id": ref.asset_id}
        ]
    if case == "product_followup":
        with (
            patch.object(asset_settings, "ocr", "tesseract"),
            patch.object(TesseractOCR, "read", lambda *_: "water-pocket"),
        ):
            ref = add_image(data)
        product["knowledge_evidence"] = [
            {"unit_id": "profile:bench:0", "asset_id": None}
        ]
    from RetailPilot.evidence_runtime import product_retrieve as actual_product_lookup

    def product_lookup(q, bundle, scope):
        return actual_product_lookup(q, bundle, scope)

    started = time.perf_counter()
    validation = None
    from RetailPilot.evidence_shadow import audit_futures

    futures = []
    audit_token = audit_futures.set(futures)
    with (
        patch("RetailPilot.evidence_runtime.run_evidence_workflow", engine),
        patch("RetailPilot.evidence_runtime.product_retrieve", product_lookup),
        patch.object(VisionProvider, "verify", vision),
    ):
        if domain == "policy":
            result = policy_workflow(
                query=question,
                topic="保修",
                initial_payload={"documents": initial},
                retrieval_arguments={"platform": "bench", "authority": "platform"},
                retrieve=lookup,
                assess=oracle.legacy,
            )
            action = result["assessment"]["next_action"]
            hops = result["summary"]["extra_retrieval_calls"]
            docs = result["documents"]
            citations = [policy_citation(d) for d in docs]
            claims = [
                claim(d["content"], c, claim_type="policy")
                for d, c in zip(docs, citations)
            ]
            if docs:
                validation = validate_claim_citations(
                    claims, citations, policy_documents=docs
                )
        else:
            result = None
            if mode == "legacy":
                evidence = product_asset_answer(question, [product])
                action = (
                    "answer"
                    if case == "product_price" or evidence["lines"]
                    else "refuse"
                )
                hops = 0
                if evidence["citations"]:
                    validation = validate_claim_citations(
                        [
                            claim(c["value"], c, claim_type="asset_observation")
                            for c in evidence["citations"]
                        ],
                        evidence["citations"],
                        products=[product],
                        asset_sources=evidence["sources"],
                    )
            else:
                checked, trace = evaluate_products(question, [product], model=oracle)
                action = "answer" if checked else "refuse"
                hops = sum(t.get("extra_retrieval_calls", 0) for t in trace)
                if mode == "shadow":
                    old = product_asset_answer(question, [product])
                    action = (
                        "answer"
                        if case == "product_price" or old["lines"]
                        else "refuse"
                    )
                    if old["citations"]:
                        validation = validate_claim_citations(
                            [
                                claim(c["value"], c, claim_type="asset_observation")
                                for c in old["citations"]
                            ],
                            old["citations"],
                            products=[product],
                            asset_sources=old["sources"],
                        )
                elif checked:
                    evidence = product_asset_answer(question, checked)
                    if evidence["citations"]:
                        validation = validate_claim_citations(
                            [
                                claim(c["value"], c, claim_type="asset_observation")
                                for c in evidence["citations"]
                            ],
                            evidence["citations"],
                            products=checked,
                            asset_sources=evidence["sources"],
                        )
        decision_latency = (time.perf_counter() - started) * 1000
        audit_results = [future.result(timeout=30) for future in futures]
        for audit in audit_results:
            summary = audit.get("summary", {})
            hops += summary.get("extra_retrieval_calls", 0) + sum(
                t.get("extra_retrieval_calls", 0) for t in summary.get("trace", [])
            )
    audit_futures.reset(audit_token)
    return {
        "decision_latency_ms": decision_latency,
        "case": case,
        "domain": domain,
        "expected": expected,
        "decision": action,
        "correct": action == expected,
        "model_calls": oracle.calls + len(vision_calls),
        "vision_inputs": oracle.images + len(vision_calls),
        "retrieval_hops": hops,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "grounding": validation,
        "token_usage": None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = []
    with (
        tempfile.TemporaryDirectory() as temp,
        patch.object(asset_settings, "enabled", True),
        patch.object(asset_settings, "ocr", "off"),
        patch.object(asset_settings, "vlm_model", "synthetic"),
        patch.object(asset_settings, "vlm_url", "http://synthetic.invalid/v1"),
        patch.object(asset_settings, "visual_enabled", False),
        patch.object(
            VisionProvider,
            "describe",
            lambda *_: {
                "image_type": "image",
                "observed": "",
                "advertised": "",
                "unknown": "",
            },
        ),
    ):
        for mode in ("legacy", "shadow", "unified"):
            with patch.dict(
                os.environ,
                {
                    "RETAILPILOT_EVIDENCE_MODE": mode,
                    "RETAILPILOT_KNOWLEDGE_PRODUCT_MODEL": "hash",
                    "RETAILPILOT_KNOWLEDGE_POLICY_MODEL": "disabled",
                    "RETAILPILOT_REASONING_MODEL": "synthetic",
                    "RETAILPILOT_REASONING_BASE_URL": "http://synthetic.invalid/v1",
                    "RETAILPILOT_REASONING_API_KEY": "synthetic-not-a-secret",
                    "RETAILPILOT_REASONING_SUPPORTS_VISION": "true",
                    "RETAILPILOT_REASONING_ENABLED": "true",
                    "RETAILPILOT_EVIDENCE_MAX_EXTRA_HOPS": "2",
                    "RETAILPILOT_EVIDENCE_MAX_IMAGES": "2",
                    "RETAILPILOT_EVIDENCE_INDEPENDENT_VISION_CHECK": "false",
                },
            ):
                cases = []
                for case, question, domain, expected in CASES:
                    root = Path(temp) / mode / case
                    root.mkdir(parents=True)
                    cases.append(execute(root, mode, case, question, domain, expected))
                labelled = [r for r in cases if r["grounding"] is not None]
                presales = [
                    r
                    for r in cases
                    if r["domain"] == "product" and r["expected"] == "answer"
                ]
                aftersales = [
                    r
                    for r in cases
                    if r["domain"] == "policy" and r["expected"] != "answer"
                ]
                follow = [r for r in cases if "followup" in r["case"]]
                rows.append(
                    {
                        "mode": mode,
                        "evidence_sufficiency_accuracy": mean(
                            r["correct"] for r in cases
                        ),
                        "presales_false_refusal_rate": mean(
                            r["decision"] != "answer" for r in presales
                        ),
                        "aftersales_false_answer_rate": mean(
                            r["decision"] == "answer" for r in aftersales
                        ),
                        "followup_success_rate": mean(r["correct"] for r in follow),
                        "average_retrieval_hops": mean(
                            r["retrieval_hops"] for r in cases
                        ),
                        "average_model_calls": mean(r["model_calls"] for r in cases),
                        "vision_input_count": sum(r["vision_inputs"] for r in cases),
                        "grounding_pass_rate": mean(
                            r["grounding"]["pass"] for r in labelled
                        )
                        if labelled
                        else None,
                        "citation_accuracy": mean(
                            r["grounding"]["pass"] for r in labelled
                        )
                        if labelled
                        else None,
                        "unsupported_claim_rate": mean(
                            not r["grounding"]["pass"] for r in labelled
                        )
                        if labelled
                        else None,
                        "latency_ms": {
                            "p50": float(
                                np.percentile([r["latency_ms"] for r in cases], 50)
                            ),
                            "p95": float(
                                np.percentile([r["latency_ms"] for r in cases], 95)
                            ),
                        },
                        "token_usage": None,
                        "cases": cases,
                    }
                )
    report = {
        "external_api_calls": 0,
        "real_models": False,
        "human_evaluation": False,
        "scope": "synthetic controllers plus extracted-claim grounding; not end-to-end generative answer quality",
        "calls": "Actual scripted oracle/visual adapter invocations; no external API calls",
        "notes": [
            "Synthetic labels and oracle decisions do not validate semantic model capability.",
            "Token usage is unavailable for the scripted oracle, not fabricated as an API usage estimate.",
            "Shadow totals include both decisions and wait for background audits; decision_latency_ms records the legacy path separately.",
        ],
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
