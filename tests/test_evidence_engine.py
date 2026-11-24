from __future__ import annotations

import json
import time

import pytest
from test_multimodal import add, png
from test_multimodal import mm as mm  # noqa: PLC0414 - pytest fixture re-export

from RetailPilot.asset_config import asset_settings
from RetailPilot.asset_parsing import TesseractOCR
from RetailPilot.catalog import get_product
from RetailPilot.database import db_session
from RetailPilot.evidence_policies import (
    PolicyEvidencePolicy,
    ProductEvidencePolicy,
    knowledge_ref,
    product_bundle,
)
from RetailPilot.evidence_schema import (
    AspectAssessment,
    EvidenceAssessment,
    FollowupQuery,
    VisualObservation,
    validate_assessment,
)
from RetailPilot.evidence_workflow import run_evidence_workflow
from RetailPilot.multimodal_model import ImageInput, ModelError, gateway


@pytest.fixture(autouse=True)
def unified(monkeypatch, tmp_path):
    monkeypatch.setenv("RETAILPILOT_EVIDENCE_MODE", "unified")
    monkeypatch.setenv("RETAILPILOT_REASONING_MODEL", "one-vlm")
    monkeypatch.setenv("RETAILPILOT_REASONING_BASE_URL", "https://fixture/v1")
    monkeypatch.setenv("RETAILPILOT_REASONING_API_KEY", "fake")
    monkeypatch.setenv("RETAILPILOT_REASONING_SUPPORTS_VISION", "true")
    monkeypatch.setenv("RETAILPILOT_REASONING_SUPPORTS_STRUCTURED_OUTPUT", "true")
    from RetailPilot.multimodal_model import config, endpoint_fingerprint

    report = tmp_path / "mock-capabilities.json"
    report.write_text(
        json.dumps(
            {
                "model": "one-vlm",
                "endpoint_fingerprint": endpoint_fingerprint(config()),
                "vision_verified": True,
                "real_model": False,
            }
        )
    )
    monkeypatch.setenv("RETAILPILOT_REASONING_CAPABILITIES_FILE", str(report))
    monkeypatch.setattr(
        "RetailPilot.asset_parsing.VisionProvider.describe",
        lambda *_: {
            "image_type": "image",
            "observed": "",
            "advertised": "",
            "unknown": "",
        },
    )


def bundle(question="水杯袋", deterministic=False, products=None):
    return product_bundle(
        question,
        products or [get_product("mm-pack")],
        constraints={"price_max": 90, "negative_brands": ["Excluded"]},
        deterministic=deterministic,
    )


def verdict(b, *, action="answer", supports=None, observations=()):
    aspects = []
    for a in b.aspects:
        ids = (
            supports
            if supports is not None
            else [
                e.evidence_id
                for e in b.evidence
                if e.domain == a.domain
                and (a.domain == "policy" or e.target_id == a.target_id)
            ]
        )
        aspects.append(
            AspectAssessment(
                aspect=a.aspect,
                target_id=a.target_id,
                required=a.required,
                status="supported" if action == "answer" else "unsupported",
                supporting_evidence_ids=ids if action == "answer" else [],
            )
        )
    return EvidenceAssessment(
        next_action=action,
        coverage="complete" if action == "answer" else "insufficient",
        conflict=False,
        reason="synthetic fixture",
        aspects=aspects,
        allowed_evidence_ids=list(
            dict.fromkeys(e for a in aspects for e in a.supporting_evidence_ids)
        ),
        suggested_queries=[
            FollowupQuery(
                aspect=b.aspects[0].aspect,
                domain=b.aspects[0].domain,
                target_id=b.aspects[0].target_id,
                query="补充水杯袋",
            )
        ]
        if action == "retrieve_more"
        else [],
        visual_observations=list(observations),
    )


class Model:
    def __init__(self, actions=("answer",)):
        self.actions = iter(actions)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        from RetailPilot.evidence_schema import EvidenceBundle

        context = json.loads(kwargs["messages"][-1]["content"])
        context.pop("validated_prior_observations", None)
        b = EvidenceBundle.model_validate(context)
        action = next(self.actions)
        sent = {image.unit_id for image in kwargs.get("images") or []}
        observations = (
            [
                VisualObservation(
                    evidence_id=e.evidence_id,
                    unit_id=e.unit_id,
                    asset_id=e.asset_id,
                    target_id=e.target_id,
                    version=e.version,
                    snapshot=e.snapshot,
                    observation="水杯袋与绿色外观",
                )
                for e in b.evidence
                if e.unit_id in sent
            ]
            if action == "answer"
            else []
        )
        return verdict(b, action=action, observations=observations)


def test_fast_path_zero_calls_and_shared_executor(mm):
    model = Model()
    result = run_evidence_workflow(
        initial_bundle=bundle(deterministic=True),
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: pytest.fail("unnecessary retrieval"),
        model=model,
    )
    assert result["verdict"]["next_action"] == "answer"
    assert model.calls == []
    assert result["summary"]["engine"] == "unified"


def test_missing_feature_retrieves_reassesses_and_freezes_constraints(mm, monkeypatch):
    monkeypatch.setattr(asset_settings, "ocr", "tesseract")
    monkeypatch.setattr(TesseractOCR, "read", lambda *_: "水杯袋")
    q = []
    model = Model(("retrieve_more", "answer"))
    initial = bundle()

    def retrieve(proposal, current, scope):
        q.append((proposal, current.hard_constraints.copy(), scope.copy()))
        aid = add(png())
        with db_session() as db:
            uid = db.execute(
                "SELECT unit_id FROM knowledge_units JOIN knowledge_sources USING(source_id) WHERE asset_id=?",
                (aid,),
            ).fetchone()[0]
        return [knowledge_ref(uid)]

    result = run_evidence_workflow(
        initial_bundle=initial,
        policy=ProductEvidencePolicy(),
        retrieve=retrieve,
        model=model,
    )
    assert result["verdict"]["next_action"] == "answer", result["summary"]
    assert len(q) == 1 and len(model.calls) == 2
    assert q[0][1] == initial.hard_constraints
    assert len(result["documents"]) == 2
    assert result["summary"]["vision_input_count"] == 1


def test_no_new_information_and_budget_are_not_sufficient(mm):
    model = Model(("retrieve_more",))
    initial = bundle()
    result = run_evidence_workflow(
        initial_bundle=initial,
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: initial.evidence,
        model=model,
    )
    assert result["summary"]["stop_reason"] == "no_new_chunks"
    assert result["verdict"]["next_action"] == "refuse"
    result = run_evidence_workflow(
        initial_bundle=initial,
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: [],
        model=Model(("retrieve_more",)),
        budget={"max_hops": 0},
    )
    assert result["summary"]["stop_reason"] == "max_extra_hops"
    assert result["verdict"]["next_action"] == "refuse"


@pytest.mark.parametrize(
    "mutation", ["unknown", "sku", "required", "omitted", "qualified"]
)
def test_invalid_support_and_necessary_aspects_rejected(mm, mutation):
    b = bundle(products=[get_product("mm-pack"), get_product("mm-expensive")])
    raw = verdict(b).model_dump()
    if mutation == "unknown":
        raw["aspects"][0]["supporting_evidence_ids"] = ["nonexistent"]
    if mutation == "sku":
        raw["aspects"][0]["supporting_evidence_ids"] = ["catalog:mm-expensive"]
    if mutation == "required":
        raw["aspects"][0]["required"] = False
    if mutation == "omitted":
        raw["aspects"].pop()
    if mutation == "qualified":
        raw["coverage"] = "qualified"
        raw["unconfirmed_conditions"] = ["optional"]
        raw["aspects"][0]["status"] = "unsupported"
    with pytest.raises(ValueError):
        validate_assessment(raw, b)


def test_price_exclusion_and_outside_followup_denied(mm):
    b = bundle()
    b.hard_constraints["price_max"] = 1
    result = run_evidence_workflow(
        initial_bundle=b,
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: [],
        model=Model(),
    )
    assert result["verdict"]["next_action"] == "refuse"
    raw = verdict(bundle(), action="retrieve_more").model_dump()
    raw["suggested_queries"][0]["target_id"] = "mm-expensive"
    with pytest.raises(ValueError, match="missing aspect"):
        validate_assessment(raw, bundle())


def test_timeout_single_model_call_no_retry(mm):
    class Slow(Model):
        def generate(self, **kwargs):
            self.calls.append(kwargs)
            time.sleep(0.08)
            raise TimeoutError()

    m = Slow()
    result = run_evidence_workflow(
        initial_bundle=bundle(),
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: [],
        model=m,
        budget={"timeout": 0.02},
    )
    assert result["summary"]["stop_reason"] == "assessment_timeout"
    assert len(m.calls) == 1 and result["verdict"]["next_action"] == "refuse"


def test_multimodal_one_request_and_snapshot_binding(mm, monkeypatch):
    aid = add(png())
    with db_session() as db:
        uid = db.execute(
            "SELECT unit_id FROM knowledge_units JOIN knowledge_sources USING(source_id) WHERE asset_id=?",
            (aid,),
        ).fetchone()[0]
    ref = knowledge_ref(uid)
    b = bundle()
    b.evidence.append(ref)
    calls = []

    class Vision(Model):
        def generate(self, **kwargs):
            calls.append(kwargs)
            obs = VisualObservation(
                evidence_id=uid,
                unit_id=uid,
                asset_id=aid,
                target_id="mm-pack",
                version=ref.version,
                snapshot=ref.snapshot,
                observation="绿色外观",
            )
            return verdict(b, observations=[obs])

    result = run_evidence_workflow(
        initial_bundle=b,
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: [],
        model=Vision(),
    )
    assert result["verdict"]["next_action"] == "answer", result["summary"]
    assert len(calls) == 1 and calls[0]["images"][0].data.startswith(b"\x89PNG")
    raw = verdict(
        b,
        observations=[
            VisualObservation(
                evidence_id=uid,
                unit_id=uid,
                asset_id=aid,
                target_id="mm-pack",
                version="wrong",
                snapshot=ref.snapshot,
                observation="绿色外观",
            )
        ],
    )
    with pytest.raises(ValueError, match="visual reference"):
        validate_assessment(raw, b, sent_image_ids=[uid])


def test_gateway_text_and_image_same_model_and_strict_wire(mm, monkeypatch):
    captured = []

    class Response:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def json(self):
            return {
                "choices": [
                    {"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }

    def post(url, **kwargs):
        captured.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("RetailPilot.multimodal_model.requests.post", post)
    assert gateway.generate(
        task_type="intent_router", messages=[{"role": "user", "content": "text"}]
    )["ok"]
    assert gateway.generate(
        task_type="visual_verification",
        messages=[{"role": "user", "content": "image"}],
        images=[ImageInput(unit_id="unit", asset_id="asset", version="v", data=png())],
    )["ok"]
    assert [p["model"] for p in captured] == ["one-vlm", "one-vlm"]
    assert all(isinstance(m["content"], str) for m in captured[0]["messages"])
    assert any(
        part.get("image_url", {}).get("url", "").startswith("data:image/png;base64,")
        for msg in captured[1]["messages"]
        if isinstance(msg["content"], list)
        for part in msg["content"]
    )
    monkeypatch.setenv("RETAILPILOT_REASONING_SUPPORTS_VISION", "false")
    with pytest.raises(ModelError, match="unsupported_vision"):
        gateway.generate(
            task_type="evidence_assessment",
            messages=[],
            images=[ImageInput(unit_id="u", asset_id="a", version="v", data=png())],
        )
    assert len(captured) == 2
    with db_session() as db:
        usage = [
            tuple(row)
            for row in db.execute(
                "SELECT input_tokens,output_tokens,total_tokens,image_count FROM llm_usage_logs ORDER BY id"
            )
        ]
    assert usage == [(10, 5, 15, 0), (10, 5, 15, 1)]


def test_structured_wrong_schema_and_truncation_fail_closed(mm, monkeypatch):
    from RetailPilot.multimodal_model import GatewayModel

    class Response:
        status_code = 200
        finish = "length"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def json(self):
            return {
                "choices": [
                    {"message": {"content": "{}"}, "finish_reason": self.finish}
                ]
            }

    monkeypatch.setattr(
        "RetailPilot.multimodal_model.requests.post", lambda *_a, **_k: Response()
    )
    with pytest.raises(ModelError, match="truncated_output"):
        GatewayModel().invoke("prompt")
    Response.finish = "stop"
    with pytest.raises(ValueError):
        gateway.generate(
            task_type="evidence_assessment",
            messages=[],
            output_schema=EvidenceAssessment,
        )


def test_forged_content_and_conflict_cannot_be_self_certified(mm):
    b = bundle()
    b.evidence[0].text = "all products have a lifetime warranty"
    with pytest.raises(ValueError, match="snapshot"):
        ProductEvidencePolicy().validate_constraints(b)
    raw = verdict(bundle()).model_dump()
    raw["aspects"][0]["status"] = "conflicted"
    with pytest.raises(ValueError, match="conflict"):
        EvidenceAssessment.model_validate(raw)


def test_policy_shared_workflow_and_no_product_warranty_support(mm, monkeypatch):
    from RetailPilot.evidence_runtime import policy_workflow
    from RetailPilot.policy import retrieve_policies
    from RetailPilot.policy_ingestion import (
        PolicyDocumentInput,
        ingest_policy_documents,
    )

    ingest_policy_documents(
        [
            PolicyDocumentInput(
                doc_id="ep",
                policy_id="ep",
                topic="保修",
                title="保修",
                content="保修期限为24个月，须提交订单凭证。",
                source_name="fixture",
                platform="demo-mall",
                authority="platform",
            )
        ]
    )
    payload = retrieve_policies("保修期限", topic="保修", authority="platform")
    model = Model()
    from RetailPilot.evidence_policies import policy_bundle

    b0 = policy_bundle(
        "保修期限",
        payload["documents"],
        topic="保修",
        arguments={"platform": "demo-mall"},
    )
    PolicyEvidencePolicy().validate_constraints(b0)
    PolicyEvidencePolicy().validate_verdict(verdict(b0), b0)
    monkeypatch.setattr(
        "RetailPilot.evidence_runtime.run_evidence_workflow",
        lambda **kwargs: run_evidence_workflow(**kwargs, model=model),
    )
    result = policy_workflow(
        query="保修期限",
        topic="保修",
        initial_payload=payload,
        retrieval_arguments={"platform": "demo-mall"},
        retrieve=lambda *_: pytest.fail("unnecessary"),
    )
    assert result["assessment"]["next_action"] == "answer", result["summary"]
    assert result["documents"][0]["doc_id"] == "ep"
    assert len(model.calls) == 1
    b = __import__(
        "RetailPilot.evidence_policies", fromlist=["policy_bundle"]
    ).policy_bundle(
        "保修期限",
        payload["documents"],
        topic="保修",
        arguments={"platform": "demo-mall"},
    )
    foreign = bundle().evidence[0]
    b.domains.append("product")
    b.evidence.append(foreign)
    with pytest.raises(ValueError, match="cross-domain"):
        validate_assessment(verdict(b, supports=[foreign.evidence_id]), b)
    with db_session() as db:
        db.execute("UPDATE policy_documents SET version='new' WHERE doc_id='ep'")
    with pytest.raises(ValueError, match="snapshot"):
        PolicyEvidencePolicy().validate_constraints(b)


def test_product_runtime_fast_path_and_shadow_preserve_output(mm, monkeypatch):
    from RetailPilot.evidence_runtime import evaluate_products

    monkeypatch.setattr(
        gateway, "generate", lambda **_: pytest.fail("fast path called model")
    )
    products = [get_product("mm-pack"), get_product("mm-expensive")]
    checked, trace = evaluate_products("当前价格", products)
    assert [p["sku"] for p in checked] == [p["sku"] for p in products]
    assert all(p["evidence_assessment"]["next_action"] == "answer" for p in checked)
    assert all(t["llm_calls"] == 0 for t in trace)
    monkeypatch.setenv("RETAILPILOT_EVIDENCE_MODE", "shadow")
    original = [get_product("mm-pack") for _ in range(6)]
    from RetailPilot.evidence_shadow import audit_futures

    futures = []
    token = audit_futures.set(futures)
    try:
        checked, trace = evaluate_products(
            "宣传图中的功能", original, model=Model(("retrieve_more",) * 4)
        )
        assert (
            checked == original and len(trace) == 1 and trace[0]["status"] == "started"
        )
        audit = futures[0].result(timeout=5)
        assert len(audit["summary"]["trace"]) == 4
    finally:
        audit_futures.reset(token)


def test_order_authorization_zero_model_calls(mm, monkeypatch):
    from RetailPilot.evidence_runtime import order_workflow

    with db_session() as db:
        db.execute("INSERT OR IGNORE INTO users(user_id) VALUES('owner')")
        db.execute(
            "INSERT INTO orders(order_id,user_id,sku,amount,status,created_at) VALUES('e-order','owner','mm-pack',80,'paid','2026-11-01')"
        )
    monkeypatch.setattr(
        gateway, "generate", lambda **_: pytest.fail("deterministic order called model")
    )
    result = order_workflow("订单状态", {"order_id": "e-order"}, "owner")
    assert result["verdict"]["next_action"] == "answer"
    assert result["summary"]["llm_calls"] == 0
    assert result["bundle"].evidence[0].unit_id is None
    denied = order_workflow("订单状态", {"order_id": "e-order"}, "another-user")
    assert denied["verdict"]["next_action"] == "refuse"


def test_gateway_invalid_reference_schema_and_usage_logs(mm, monkeypatch):
    b = bundle()
    raw = verdict(b).model_dump()
    raw["aspects"][0]["supporting_evidence_ids"] = ["fake"]

    class Bad:
        def generate(self, **_):
            return EvidenceAssessment.model_validate(raw)

    result = run_evidence_workflow(
        initial_bundle=b,
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: [],
        model=Bad(),
    )
    assert result["verdict"]["next_action"] == "refuse"
    assert result["documents"] == []


def test_search_api_fast_path_and_original_response_contract(mm, monkeypatch):
    from fastapi.testclient import TestClient

    from RetailPilot.api import app

    monkeypatch.setattr(
        gateway, "generate", lambda **_: pytest.fail("simple search called VLM")
    )
    with TestClient(app) as client:
        response = client.post("/api/search", json={"query": "Alpine", "limit": 5})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["products"]
    assert result["products"][0]["evidence_assessment"]["next_action"] == "answer"
    assert {"sku", "price", "citations"} <= set(result["products"][0])


def test_cross_domain_technical_followup_keeps_policy_rights_separate(mm, monkeypatch):
    from test_multimodal import pdf

    from RetailPilot.asset_ingestion import process_asset, stage_asset
    from RetailPilot.evidence_runtime import policy_workflow
    from RetailPilot.evidence_schema import EvidenceBundle
    from RetailPilot.policy import retrieve_policies
    from RetailPilot.policy_ingestion import (
        PolicyDocumentInput,
        ingest_policy_documents,
    )

    ingest_policy_documents(
        [
            PolicyDocumentInput(
                doc_id="tech-policy",
                policy_id="tp",
                topic="保修",
                title="Warranty",
                content="Warranty for brush defects is 24 months.",
                source_name="fixture",
                platform="technical",
                authority="platform",
            )
        ]
    )
    aid, _ = stage_asset(
        pdf("Replace brush: disconnect power then remove cover"),
        owner_type="product",
        owner_id="mm-pack",
        kind="product_document",
        source_key="manual.pdf",
    )
    assert process_asset(aid)["status"] in {"ready", "partial"}
    initial = retrieve_policies(
        "Warranty brush", topic="保修", platform="technical", authority="platform"
    )

    class Cross:
        calls = 0

        def generate(self, **kwargs):
            self.calls += 1
            ctx = json.loads(kwargs["messages"][-1]["content"])
            ctx.pop("validated_prior_observations", None)
            b = EvidenceBundle.model_validate(ctx)
            if any(e.domain == "product" for e in b.evidence):
                return verdict(b)
            raw = verdict(b, action="retrieve_more").model_dump()
            tech = next(a for a in b.aspects if a.domain == "product")
            raw["suggested_queries"] = [
                {
                    "aspect": tech.aspect,
                    "target_id": tech.target_id,
                    "domain": "product",
                    "query": "Replace brush",
                }
            ]
            return EvidenceAssessment.model_validate(raw)

    model = Cross()
    monkeypatch.setattr(
        "RetailPilot.evidence_runtime.run_evidence_workflow",
        lambda **kw: run_evidence_workflow(**kw, model=model),
    )
    result = policy_workflow(
        query="如何更换滚刷，是否保修",
        topic="保修",
        initial_payload=initial,
        retrieval_arguments={
            "platform": "technical",
            "authority": "platform",
            "product_context": {"sku": "mm-pack"},
        },
        retrieve=lambda *_: pytest.fail("technical hop must use product index"),
    )
    assert result["assessment"]["next_action"] == "answer", result["summary"]
    assert model.calls == 2
    assert result["technical_documents"][0]["target_id"] == "mm-pack"
    assert all(d["doc_id"] == "tech-policy" for d in result["documents"])
    technical_only = Cross()
    monkeypatch.setattr(
        "RetailPilot.evidence_runtime.run_evidence_workflow",
        lambda **kw: run_evidence_workflow(**kw, model=technical_only),
    )
    result = policy_workflow(
        query="如何更换滚刷",
        topic="保修",
        initial_payload={"documents": []},
        retrieval_arguments={
            "platform": "technical",
            "product_context": {"sku": "mm-pack"},
        },
        retrieve=lambda *_: pytest.fail(
            "technical-only question must not require policy rights"
        ),
    )
    assert result["assessment"]["next_action"] == "answer"
    assert result["documents"] == [] and result["technical_documents"]


def test_capability_proof_and_strict_json_schema(mm, monkeypatch):
    from RetailPilot.multimodal_model import ready_for_vision, strict_schema

    assert ready_for_vision("evidence_assessment")
    monkeypatch.setenv("RETAILPILOT_REASONING_MODEL", "changed-model")
    assert not ready_for_vision("evidence_assessment")
    schema = strict_schema(EvidenceAssessment)
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert all(
        d.get("additionalProperties") is False
        for d in schema["$defs"].values()
        if d.get("type") == "object"
    )


def test_numeric_catalog_poster_conflict_refuses_without_model(mm, monkeypatch):
    with db_session() as db:
        db.execute(
            "UPDATE products SET attributes_json=?,attributes_text=? WHERE sku='mm-pack'",
            (json.dumps({"容量": "500 mL"}), "容量500mL"),
        )
    monkeypatch.setattr(asset_settings, "ocr", "tesseract")
    monkeypatch.setattr(TesseractOCR, "read", lambda *_: "容量 1000 mL")
    aid = add(png())
    with db_session() as db:
        uid = db.execute(
            "SELECT unit_id FROM knowledge_units JOIN knowledge_sources USING(source_id) WHERE asset_id=?",
            (aid,),
        ).fetchone()[0]
    b = bundle("宣传图中的容量是多少")
    b.evidence.append(knowledge_ref(uid))
    model = Model()
    result = run_evidence_workflow(
        initial_bundle=b,
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: [],
        model=model,
    )
    assert result["verdict"]["next_action"] == "refuse"
    assert result["verdict"]["conflict"] is True
    assert model.calls == []


def test_unified_visual_only_search_uses_single_assessment_call(mm, monkeypatch):
    from test_multimodal import ColorEncoder

    from RetailPilot import knowledge_index, knowledge_retrieval
    from RetailPilot.retrieval import search_engine
    from RetailPilot.schemas import ParsedQuery

    monkeypatch.setattr(asset_settings, "visual_enabled", True)
    encoder = ColorEncoder()
    monkeypatch.setattr(knowledge_index, "visual_encoder", lambda: encoder)
    monkeypatch.setattr(knowledge_retrieval, "visual_encoder", lambda: encoder)
    aid = add(png(), kind="product_main_image")
    model = Model()
    monkeypatch.setattr(gateway, "generate", model.generate)
    monkeypatch.setattr(
        "RetailPilot.retrieval.verify_image",
        lambda *_: pytest.fail("duplicate visual verification"),
    )
    result = search_engine.search(
        ParsedQuery(raw_query="绿色外观", rewritten_query="绿色外观")
    )
    assert [p["sku"] for p in result["products"]] == ["mm-pack"]
    assert len(model.calls) == 1
    assert result["products"][0]["asset_evidence"][0]["asset_id"] == aid
    assert result["products"][0]["evidence_assessment"]["visual_observations"]


def test_policy_expansion_is_relative_to_original_topic(mm):
    from RetailPilot.evidence_policies import policy_bundle

    b = policy_bundle("保修例外", [], topic="保修", arguments={"platform": "demo-mall"})
    raw = verdict(b, action="retrieve_more").model_dump()
    raw["suggested_queries"][0]["target_id"] = "topic:退换货"
    assert validate_assessment(raw, b).next_action == "retrieve_more"
    raw["suggested_queries"][0]["target_id"] = "topic:退款"
    with pytest.raises(ValueError, match="authorization"):
        validate_assessment(raw, b)


def test_optional_independent_visual_check_is_additional_and_bounded(mm, monkeypatch):
    aid = add(png())
    with db_session() as db:
        uid = db.execute(
            "SELECT unit_id FROM knowledge_units JOIN knowledge_sources USING(source_id) WHERE asset_id=?",
            (aid,),
        ).fetchone()[0]
    b = bundle("绿色外观")
    b.evidence.append(knowledge_ref(uid))
    monkeypatch.setenv("RETAILPILOT_EVIDENCE_INDEPENDENT_VISION_CHECK", "true")
    calls = []
    monkeypatch.setattr(
        "RetailPilot.asset_evidence.verify_image",
        lambda package, query: calls.append(query) or "绿色外观",
    )
    model = Model()
    result = run_evidence_workflow(
        initial_bundle=b,
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: [],
        model=model,
    )
    assert result["verdict"]["next_action"] == "answer"
    assert len(model.calls) == len(calls) == 1
    monkeypatch.setattr("RetailPilot.asset_evidence.verify_image", lambda *_: "")
    result = run_evidence_workflow(
        initial_bundle=b,
        policy=ProductEvidencePolicy(),
        retrieve=lambda *_: [],
        model=Model(),
    )
    assert result["verdict"]["next_action"] == "refuse"
