from __future__ import annotations

import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("PIL")
pytest.importorskip("pypdf")
from PIL import Image
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from RetailPilot import asset_ingestion, asset_retrieval
from RetailPilot.api import app
from RetailPilot.asset_config import asset_settings as config
from RetailPilot.asset_evidence import (
    hydrate,
    policy_asset_evidence,
    product_asset_answer,
)
from RetailPilot.asset_ingestion import (
    delete_asset,
    get_asset,
    process_asset,
    stage_asset,
)
from RetailPilot.asset_parsing import (
    DocumentElement,
    ParsedDocument,
    TesseractOCR,
    VisionProvider,
    enrich_context,
)
from RetailPilot.asset_qdrant import generation_status, sync_visual
from RetailPilot.asset_retrieval import aggregate_routes, retrieve_product_assets
from RetailPilot.asset_store import download_image, store
from RetailPilot.catalog import get_product
from RetailPilot.config import settings
from RetailPilot.database import db_session, init_db
from RetailPilot.grounding import claim, policy_citation, validate_claim_citations
from RetailPilot.policy import retrieve_policies
from RetailPilot.policy_ingestion import PolicyDocumentInput, ingest_policy_documents
from RetailPilot.retrieval import search_engine
from RetailPilot.schemas import ParsedQuery


def png(color="green", size=(32, 30)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, "PNG")
    return output.getvalue()


def pdf(text="Filter warranty is exactly 24 months"):
    writer = PdfWriter()
    page = writer.add_blank_page(width=500, height=700)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 650 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


@pytest.fixture
def mm(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "database_path", tmp_path / "mm.db")
    monkeypatch.setattr(config, "store", tmp_path / "blobs")
    monkeypatch.setattr(config, "enabled", True)
    monkeypatch.setattr(config, "visual_enabled", False)
    monkeypatch.setattr(config, "parser", "pypdf")
    monkeypatch.setattr(config, "ocr", "off")
    monkeypatch.setattr(config, "vlm_model", "")
    monkeypatch.setattr(config, "qdrant_url", "")
    monkeypatch.setattr(config, "text_model", "hash")
    init_db()
    with db_session() as db:
        db.executemany(
            """INSERT INTO products(sku,title,category,category_label,price,stock,brand,source)
               VALUES(?,?,'Sports & Outdoors','户外',?,?,?,'synthetic')""",
            [
                ("mm-pack", "Alpine", 70, 6, "Studio"),
                ("mm-expensive", "Alpine Pro", 190, 4, "Studio"),
                ("mm-stock", "Alpine Mini", 66, 0, "Other"),
            ],
        )
    ingest_policy_documents(
        [
            PolicyDocumentInput(
                doc_id="mm-doc",
                policy_id="mm-policy",
                topic="保修",
                title="Filter manual",
                content="Standard authority: proof of purchase is required.",
                source_name="Fixture",
                platform=settings.commerce_platform,
                version="7.1",
            )
        ]
    )


def add(data, *, sku="mm-pack", kind="product_poster", key="cover.png"):
    aid, queued = stage_asset(
        data, owner_type="product", owner_id=sku, kind=kind, source_key=key
    )
    if queued:
        result = process_asset(aid)
        assert result["status"] in {"ready", "partial"}, result
    return aid


def test_crud_atomic_mapping_content_dedup_and_update(mm, monkeypatch):
    monkeypatch.setattr(config, "ocr", "tesseract")
    calls = []
    monkeypatch.setattr(
        TesseractOCR, "read", lambda self, data: calls.append(data) or "主动降噪"
    )
    first = add(png())
    assert first == add(png())
    second = add(png(), key="another.png")
    assert first != second and len(calls) == 1
    assert get_asset(first)["storage_uri"] == get_asset(second)["storage_uri"]
    with pytest.raises(LookupError):
        add(png(), sku="not-catalog")
    with pytest.raises(ValueError):
        store().path("../catalog.db")
    routes, _ = retrieve_product_assets("主动降噪")
    ranks, evidence = aggregate_routes(routes)
    assert ranks["asset_bm25"] == ["mm-pack"]
    assert len(evidence["mm-pack"]) == 1
    cite_id = evidence["mm-pack"][0]["chunk_id"]
    assert hydrate(cite_id).text_content == "主动降噪"
    assert add(png("blue")) == first
    assert delete_asset(first) and delete_asset(second)
    assert hydrate(cite_id) is None
    with db_session() as db:
        for name in (
            "asset_chunks",
            "asset_vectors",
            "asset_chunk_elements",
            "asset_fts",
        ):
            assert db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] == 0


def test_caption_text_routes_hard_filters_and_grounded_poster(mm, monkeypatch):
    monkeypatch.setattr(config, "ocr", "tesseract")
    monkeypatch.setattr(config, "vlm_model", "fixture")
    monkeypatch.setattr(TesseractOCR, "read", lambda self, data: "水杯侧袋规格：500 mL")
    monkeypatch.setattr(
        VisionProvider,
        "describe",
        lambda self, image: {
            "image_type": "poster",
            "observed": "绿色徒步背包",
            "advertised": "水杯侧袋",
            "unknown": "材质",
        },
    )
    aid = add(png())
    add(png(), sku="mm-expensive")
    add(png(), sku="mm-stock")
    query = ParsedQuery(
        raw_query="水杯侧袋规格",
        rewritten_query="水杯侧袋规格",
        category="Sports & Outdoors",
        price_max=90,
        brands=["Studio"],
        must_have_groups=[["水杯侧袋"]],
    )
    result = search_engine.search(query, mode="hybrid")
    assert [p["sku"] for p in result["products"]] == ["mm-pack"]
    assert result["retrieval"]["routes"]["asset_bm25"] == ["mm-pack"]
    assert result["retrieval"]["routes"]["asset_text"] == ["mm-pack"]
    assert result["products"][0]["asset_evidence"][0]["asset_id"] == aid
    assert result["products"][0]["constraint_profile"]["must_have_match"] == 1
    answer = product_asset_answer("宣传图的水杯侧袋规格是多少？", result["products"])
    evidence = answer["citations"][0]
    assert evidence["value"] == "水杯侧袋规格：500 mL"
    assert evidence["evidence_type"] == "seller_advertised"
    claims = [claim(evidence["value"], evidence)]
    assert validate_claim_citations(
        claims, [evidence], products=result["products"], asset_sources=answer["sources"]
    )["pass"]
    forged = {**evidence, "value": "900 mL", "quoted_text": "900 mL"}
    assert not validate_claim_citations(
        [claim("900 mL", forged)],
        [forged],
        products=result["products"],
        asset_sources=[forged],
    )["pass"]
    monkeypatch.setattr(config, "enabled", False)
    assert not search_engine.search(query)["products"]


class ColorEncoder:
    def images(self, images):
        matrix = []
        for image in images:
            with Image.open(io.BytesIO(image)) as item:
                rgb = item.getpixel((0, 0))
            matrix.append([1.0, 0.0] if rgb[1] > rgb[2] else [0.0, 1.0])
        return np.asarray(matrix, dtype=np.float32)

    def texts(self, texts):
        return np.asarray(
            [[1.0, 0.0] if "绿" in text else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def test_visual_chinese_and_image_queries_mirror_and_model_failure(mm, monkeypatch):
    monkeypatch.setattr(config, "visual_enabled", True)
    encoder = ColorEncoder()
    monkeypatch.setattr(asset_ingestion, "visual_encoder", lambda: encoder)
    monkeypatch.setattr(asset_retrieval, "visual_encoder", lambda: encoder)
    green = add(png("green"), kind="product_main_image")
    blue = add(png("blue"), sku="mm-expensive", kind="product_main_image")
    routes, errors = retrieve_product_assets("绿色外观")
    assert not errors and [hit.asset_id for hit in routes["asset_visual"]] == [green]
    assert (
        retrieve_product_assets("", image=png("blue"))[0]["asset_visual"][0].asset_id
        == blue
    )

    qdrant = pytest.importorskip("qdrant_client")
    from RetailPilot import asset_qdrant

    client = qdrant.QdrantClient(":memory:")
    monkeypatch.setattr(config, "qdrant_url", "http://fixture")
    monkeypatch.setattr(asset_qdrant, "_client", lambda *_: client)
    assert sync_visual() == {"points": 2, "synchronized": True}
    assert retrieve_product_assets("绿色外观")[0]["asset_visual"][0].asset_id == green
    delete_asset(green)
    with db_session() as db:
        assert not generation_status(db)["synchronized"]
    assert sync_visual()["points"] == 1
    assert client.count(config.qdrant_collection, exact=True).count == 1

    def broken():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(asset_retrieval, "visual_encoder", broken)
    routes, errors = retrieve_product_assets("绿色外观")
    assert errors == {"visual": "RuntimeError"} and "asset_visual" not in routes


def test_visual_verification_real_image_transport_and_refusal(mm, monkeypatch):
    monkeypatch.setattr(config, "ocr", "tesseract")
    monkeypatch.setattr(config, "vlm_model", "fixture")
    monkeypatch.setattr(config, "vlm_url", "http://vision.fixture/v1")
    monkeypatch.setattr(TesseractOCR, "read", lambda self, data: "风扇示意图")
    monkeypatch.setattr(
        VisionProvider,
        "describe",
        lambda self, image: {
            "image_type": "diagram",
            "observed": "风扇示意图",
            "advertised": "",
            "unknown": "",
        },
    )
    add(png())
    request_log = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def raise_for_status(self):
            pass

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"supported":true,"observation":"接头在右侧"}'
                        }
                    }
                ]
            }

    def fake_post(url, **kwargs):
        request_log.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("RetailPilot.asset_parsing.requests.post", fake_post)
    answer = product_asset_answer(
        "风扇示意图的接头在哪一侧？", [get_product("mm-pack")]
    )
    assert "接头在右侧" in answer["lines"][0]
    assert answer["citations"][0]["verification_status"] == "image_verified"
    assert request_log[0]["messages"][1]["content"][1]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    monkeypatch.setattr(config, "vlm_url", "")
    answer = product_asset_answer(
        "风扇示意图的接头在哪一侧？", [get_product("mm-pack")]
    )
    assert answer["needed_visual"] and not answer["citations"]


def _cell(text, row, col, header=False, span=1):
    return {
        "text": text,
        "row_start": row,
        "row_end": row + 1,
        "col_start": col,
        "col_end": col + span,
        "column_header": header,
        "row_header": False,
    }


def test_real_pdf_subprocess_and_policy_version_fence(mm, monkeypatch):
    aid, _ = stage_asset(
        pdf(),
        owner_type="policy",
        owner_id="mm-doc",
        kind="document_page",
        source_key="manual.pdf",
    )
    result = process_asset(aid)
    assert result["status"] == "partial" and result["error_code"] == "pdf_text_only", (
        result
    )
    docs = retrieve_policies("filter warranty 24 months", topic="保修")["documents"]
    doc = next(item for item in docs if item["page_no"] == 1)
    assert "/content?revision=" in doc["asset_location"]["source_uri"]
    assert doc["version"] == "7.1"
    package = hydrate(doc["chunk_id"])
    assert package.doc_id == "mm-doc"
    assert not retrieve_policies("filter warranty 24 months", topic="退款")["documents"]
    monkeypatch.setattr(config, "enabled", False)
    assert all(
        not item["page_no"]
        for item in retrieve_policies("filter warranty", topic="保修")["documents"]
    )
    monkeypatch.setattr(config, "enabled", True)
    with db_session() as db:
        db.execute("UPDATE policy_documents SET version='8' WHERE doc_id='mm-doc'")
    assert hydrate(doc["chunk_id"]) is None
    assert delete_asset(aid)


def test_structured_table_and_context_plus_visual_policy(mm, monkeypatch):
    key = store().put(png())
    bbox = {"origin": "TOPLEFT", "unit": "pt", "coordinates": [1, 2, 100, 120]}
    elements = [
        DocumentElement("text", 0, "不同章节的秘密", "前言", 1),
        DocumentElement("text", 1, "安装步骤和蓝色状态灯", "安装", 2),
        DocumentElement(
            "image",
            2,
            section_path="安装",
            page_no=2,
            bbox=bbox,
            preview_key=key,
            metadata={"caption": "蓝色状态灯流程图"},
        ),
        DocumentElement(
            "table",
            3,
            section_path="安装",
            page_no=2,
            bbox=bbox,
            preview_key=key,
            metadata={
                "table": {
                    "rows": 10,
                    "columns": 2,
                    "cells": [
                        _cell("型号", 0, 0, True),
                        _cell("滤芯尺寸", 0, 1, True),
                        *[
                            cell
                            for i in range(1, 10)
                            for cell in (
                                _cell(f"X{i}", i, 0),
                                _cell(f"{i + 10} mm", i, 1),
                            )
                        ],
                    ],
                }
            },
        ),
        DocumentElement("page", 4, page_no=2, preview_key=key),
    ]
    parsed = ParsedDocument(elements, "parser-fixture")
    enrich_context(parsed, title="说明书")
    assert "蓝色状态灯" in elements[2].search_text
    assert "不同章节的秘密" not in elements[2].search_text
    assert elements[2].metadata["context_orders"] == [1]
    # The input remains a real PDF; the structured adapter output is deterministic.
    monkeypatch.setattr(asset_ingestion, "parse_asset", lambda *_: parsed)
    aid, _ = stage_asset(
        pdf(),
        owner_type="policy",
        owner_id="mm-doc",
        kind="document_page",
        source_key="layout.pdf",
    )
    assert process_asset(aid)["status"] == "ready"
    docs = retrieve_policies("X9 滤芯尺寸 19mm", topic="保修")["documents"]
    table = next(
        hydrate(item["chunk_id"]) for item in docs if "19 mm" in item["content"]
    )
    assert table.structured_content["cells"][0]["column_header"]
    assert (
        table.source_location["bbox"] == bbox
        and table.source_location["page_preview_url"]
    )
    assert len(table.structured_content["cells"]) == 4
    flow = retrieve_policies("蓝色状态灯流程图", topic="保修")["documents"]
    figure = next(item for item in flow if not item["content"])
    budget = {"remaining": 1, "deadline": time.perf_counter() + 4, "observations": {}}
    assert policy_asset_evidence(figure, "蓝色状态灯流程图", budget) is None
    budget["observations"].clear()
    monkeypatch.setattr(config, "vlm_model", "fixture")
    monkeypatch.setattr(VisionProvider, "verify", lambda *args: "绿灯分支指向继续安装")
    reviewed = policy_asset_evidence(figure, "蓝色状态灯流程图", budget)
    assert reviewed["content"].startswith("图示观察")
    assert policy_asset_evidence(figure, "流程图能证明退款政策？", budget) is None
    citation = policy_citation(reviewed)
    assert validate_claim_citations(
        [claim(reviewed["content"], citation)], [citation], policy_documents=[reviewed]
    )["pass"]
    with db_session() as db:
        assert (
            db.execute(
                "SELECT COUNT(*) FROM asset_chunk_elements WHERE chunk_id=?",
                (figure["chunk_id"],),
            ).fetchone()[0]
            == 2
        )


def test_long_image_regions_and_graph_grounding(mm, monkeypatch):
    from RetailPilot import agent

    monkeypatch.setattr(config, "ocr", "tesseract")
    monkeypatch.setattr(config, "tile_height", 400)
    monkeypatch.setattr(TesseractOCR, "read", lambda *_: "水杯侧袋规格：500 mL")
    aid = add(png(size=(300, 1600)))
    with db_session() as db:
        rows = db.execute(
            "SELECT bbox_json FROM asset_elements WHERE asset_id=? ORDER BY reading_order",
            (aid,),
        ).fetchall()
    assert len(rows) == 5 and '"coordinates":[0,360,300,760]' in rows[1][0].replace(
        " ", ""
    )

    def analyze(text, **kwargs):
        return [
            {
                "intent": "search",
                "text": text,
                "query_text": text,
                "category": "Sports & Outdoors",
                "target_terms": ["徒步背包"],
                "source": "fixture",
                "confidence": 1.0,
            }
        ]

    monkeypatch.setattr(agent, "analyze_intents", analyze)
    answer = agent.chat("mm-agent-test", "宣传图的水杯侧袋规格是多少？")
    assert answer["grounding"]["pass"], answer
    assert "500 mL" in answer["reply"]
    assert next(c for c in answer["citations"] if c.get("asset_id"))["asset_id"] == aid


def test_adaptive_policy_accepts_only_mapped_asset_chunks(mm):
    from RetailPilot.policy_followup import run_policy_followup

    aid, _ = stage_asset(
        pdf(),
        owner_type="policy",
        owner_id="mm-doc",
        kind="document_page",
        source_key="manual.pdf",
    )
    assert process_asset(aid)["status"] == "partial"
    query = "Filter warranty 24 months"
    initial = retrieve_policies(query, topic="保修")

    def assess(_query, documents, **kwargs):
        ids = [item["chunk_id"] for item in documents if "24 months" in item["content"]]
        return {
            "decision": "answer",
            "next_action": "answer",
            "sufficient": True,
            "conflict": False,
            "reason": "Fixture table/content match",
            "covered_aspects": ["term"],
            "supporting_chunk_ids": ids,
            "aspect_evidence_map": {"term": ids},
        }

    result = run_policy_followup(
        query=query,
        topic="保修",
        initial_payload=initial,
        retrieval_arguments={},
        assess=assess,
        retrieve=lambda _: pytest.fail("initial PDF evidence is sufficient"),
    )
    assert result["assessment"]["next_action"] == "answer"
    assert result["assessment"]["supporting_asset_ids"] == [aid]
    assert result["documents"][0]["asset_evidence"]["page_no"] == 1


def test_remote_allowlist_and_private_dns_fence(mm, monkeypatch):
    monkeypatch.setattr(config, "remote_hosts", "catalog.fixture")
    monkeypatch.setattr(
        "RetailPilot.asset_store.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError, match="remote_host_not_allowed"):
        download_image("https://unknown.fixture/item.png")
    with pytest.raises(ValueError, match="remote_address_not_allowed"):
        download_image("https://catalog.fixture/item.png")


def test_authenticated_upload_status_preview_and_denials(mm, monkeypatch):
    monkeypatch.setattr(settings, "allow_open_api", False)
    monkeypatch.setattr(settings, "api_key", "fixture-key")
    monkeypatch.setattr(settings, "api_user_id", "curator")
    monkeypatch.setattr(config, "curators", "curator")
    headers = {"X-API-Key": "fixture-key"}
    params = {
        "owner_type": "product",
        "owner_id": "mm-pack",
        "asset_type": "product_main_image",
        "source_key": "hero.png",
    }
    with TestClient(app) as client:
        assert client.get("/api/assets/capabilities").status_code == 401
        queued = client.post(
            "/api/assets",
            params=params,
            content=png(),
            headers={**headers, "Content-Type": "image/png"},
        )
        assert queued.status_code == 202, queued.json()
        payload = queued.json()
        for _ in range(50):
            status = client.get(payload["status_url"], headers=headers).json()
            if status["status"] in {"ready", "failed"}:
                break
            time.sleep(0.02)
        assert status["status"] == "ready", status
        assert not {"storage_uri", "content_hash", "processing_token"} & set(status)
        endpoint = status["elements"][0]["preview_url"]
        assert client.get(endpoint, headers=headers).content.startswith(b"\x89PNG")
        detail = client.get("/api/products/mm-pack", headers=headers).json()
        assert detail["assets"][0]["asset_id"] == payload["asset_id"]
        monkeypatch.setattr(settings, "api_user_id", "viewer")
        assert client.delete(payload["status_url"], headers=headers).status_code == 403
        assert client.get(status["source_uri"]).status_code == 401
        assert (
            client.get(
                endpoint.replace(status["elements"][0]["element_id"], "0" * 32),
                headers=headers,
            ).status_code
            == 404
        )
