from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from test_multimodal import ColorEncoder, add, pdf, png
from test_multimodal import mm as mm  # noqa: PLC0414 - pytest fixture re-export

from RetailPilot import knowledge_index, knowledge_retrieval
from RetailPilot.asset_config import asset_settings
from RetailPilot.asset_ingestion import (
    get_asset,
    process_asset,
    stage_asset,
)
from RetailPilot.asset_parsing import TesseractOCR
from RetailPilot.asset_store import store
from RetailPilot.config import settings
from RetailPilot.database import db_session, init_db
from RetailPilot.knowledge_evidence import hydrate
from RetailPilot.knowledge_index import build
from RetailPilot.knowledge_retrieval import (
    RetrievalFilters,
    RetrievalQuery,
    retrieve,
)
from RetailPilot.policy import retrieve_policies
from RetailPilot.policy_ingestion import PolicyDocumentInput, ingest_policy_documents
from RetailPilot.retrieval import search_engine
from RetailPilot.schemas import ParsedQuery


def test_one_corpus_one_method_and_catalog_field_provenance(mm, monkeypatch):
    monkeypatch.setattr(asset_settings, "ocr", "tesseract")
    monkeypatch.setattr(TesseractOCR, "read", lambda *_: "Alpine 水杯侧袋")
    aid = add(png())
    hits, routes, _ = retrieve(
        RetrievalQuery("Alpine 水杯侧袋", "product"), RetrievalFilters()
    )
    assert any(hit.metadata["asset_id"] == aid for hit in hits)
    assert set(routes) == {"knowledge_bm25", "knowledge_dense"}
    assert len({hit.unit_id for hit in hits}) == len(hits)
    assert {hit.metadata["source_format"] for hit in routes["knowledge_bm25"]} == {
        "catalog",
        "image",
    }
    with db_session() as db:
        tables = {
            row[0]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert not tables & {
            "product_fts",
            "product_vectors",
            "policy_chunks",
            "asset_chunks",
            "asset_vectors",
            "asset_fts",
        }
        profile = db.execute(
            "SELECT unit_id FROM knowledge_units JOIN knowledge_sources USING(source_id) WHERE sku='mm-pack' AND source_format='catalog'"
        ).fetchone()[0]
    with (
        db_session() as db,
        pytest.raises(sqlite3.IntegrityError, match="ownership is immutable"),
    ):
        db.execute("UPDATE assets SET sku='mm-expensive' WHERE asset_id=?", (aid,))
    evidence = hydrate(profile)
    assert evidence.source_id == "catalog:mm-pack"
    assert "products.attributes" in evidence.field_refs
    assert evidence.structured_content["sku"] == "mm-pack"


def test_captionless_visual_anchor_and_model_revision(mm, monkeypatch):
    monkeypatch.setattr(asset_settings, "visual_enabled", True)
    encoder = ColorEncoder()
    monkeypatch.setattr(knowledge_index, "visual_encoder", lambda: encoder)
    monkeypatch.setattr(knowledge_retrieval, "visual_encoder", lambda: encoder)
    aid = add(png("green"), kind="product_main_image")
    monkeypatch.setattr(
        "RetailPilot.retrieval.verify_image", lambda *_: "绿色外观的商品"
    )
    query = ParsedQuery(raw_query="绿色外观", rewritten_query="绿色外观")
    result = search_engine.search(query)
    assert result["products"][0]["sku"] == "mm-pack"
    assert result["products"][0]["asset_evidence"][0]["asset_id"] == aid
    assert (
        next(t for t in result["trace"] if t["node"] == "query_guard")["anchors"][
            "mm-pack"
        ]
        == "visual"
    )
    assert not hydrate(
        result["products"][0]["asset_evidence"][0]["unit_id"]
    ).text_content
    monkeypatch.setattr(asset_settings, "visual_revision", "new-revision")
    assert not retrieve(RetrievalQuery("绿色外观", "product"), RetrievalFilters())[1][
        "knowledge_visual"
    ]
    assert build("product", space="visual")["built"] == 1
    assert retrieve(RetrievalQuery("绿色外观", "product"), RetrievalFilters())[1][
        "knowledge_visual"
    ]


def test_profile_update_only_rebuilds_changed_units_and_cascade(mm, monkeypatch):
    monkeypatch.setattr(asset_settings, "ocr", "tesseract")
    monkeypatch.setattr(TesseractOCR, "read", lambda *_: "供电 28 Wh")
    aid = add(png())
    with db_session() as db:
        before = {
            row["unit_id"]: bytes(row["vector"])
            for row in db.execute("SELECT * FROM knowledge_vectors")
        }
        db.execute(
            "UPDATE products SET description='支持太阳能充电' WHERE sku='mm-pack'"
        )
    retrieve(RetrievalQuery("太阳能充电", "product"), RetrievalFilters())
    with db_session() as db:
        after = {
            row["unit_id"]: bytes(row["vector"])
            for row in db.execute("SELECT * FROM knowledge_vectors")
        }
        assert len(db.execute("SELECT * FROM knowledge_dirty").fetchall()) == 0
        asset_units = [
            row[0]
            for row in db.execute(
                "SELECT unit_id FROM knowledge_units JOIN knowledge_sources USING(source_id) WHERE asset_id=?",
                (aid,),
            )
        ]
        assert all(before[uid] == after[uid] for uid in asset_units)
        db.execute("DELETE FROM products WHERE sku='mm-pack'")
    assert get_asset(aid) is None
    with db_session() as db:
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        assert not db.execute(
            "SELECT 1 FROM knowledge_fts WHERE unit_id IN ({})".format(
                ",".join("?" for _ in asset_units)
            ),
            asset_units,
        ).fetchall()


def test_policy_sources_shared_and_all_applicability_fields(mm, monkeypatch):
    monkeypatch.setenv("RETAILPILOT_KNOWLEDGE_POLICY_MODEL", "hash")
    docs = [
        PolicyDocumentInput(
            doc_id="lawful",
            policy_id="p",
            topic="保修",
            title="保修期限",
            content="滤芯保修期限为24个月。",
            source_name="fixture",
            platform="f",
            jurisdiction="CN",
            authority="manufacturer",
        ),
        PolicyDocumentInput(
            doc_id="other",
            policy_id="q",
            topic="保修",
            title="其他",
            content="滤芯保修期限为24个月。",
            source_name="fixture",
            platform="f",
            jurisdiction="US",
            authority="seller",
        ),
    ]
    ingest_policy_documents(docs)
    result = retrieve_policies(
        "滤芯保修期限",
        topic="保修",
        platform="f",
        jurisdiction="CN",
        authority="manufacturer",
    )
    assert {item["doc_id"] for item in result["documents"]} == {"lawful"}
    md = result["documents"][0]
    assert hydrate(md["unit_id"]).source_location["char_end"] > 0
    aid, _ = stage_asset(
        pdf("Filter warranty 24 months"),
        owner_type="policy",
        owner_id="lawful",
        kind="document_page",
        source_key="terms.pdf",
    )
    process_asset(aid)
    hits, routes, _ = retrieve(
        RetrievalQuery("Filter warranty 保修期限", "policy"),
        RetrievalFilters(
            topics=["保修"], platform="f", jurisdiction="CN", authority="manufacturer"
        ),
    )
    assert {hit.metadata["source_format"] for hit in routes["knowledge_bm25"]} == {
        "markdown",
        "pdf",
    }
    assert all(hit.unit_id == hydrate(hit.unit_id).unit_id for hit in hits)
    with db_session() as db:
        db.execute("UPDATE policy_documents SET version='v2' WHERE doc_id='lawful'")
    assert not retrieve_policies(
        "保修期限", topic="保修", platform="f", jurisdiction="CN"
    )["documents"]


def test_failed_asset_retry_and_idempotent_build(mm):
    aid, _ = stage_asset(
        b"\x89PNG\r\n\x1a\ninvalid",
        owner_type="product",
        owner_id="mm-pack",
        kind="product_main_image",
        source_key="retry.png",
    )
    assert process_asset(aid)["status"] == "failed"
    aid2, _ = stage_asset(
        png(),
        owner_type="product",
        owner_id="mm-pack",
        kind="product_main_image",
        source_key="retry.png",
    )
    assert aid == aid2 and process_asset(aid)["status"] == "ready"
    assert build("product")["built"] == 0
    assert build("product")["unchanged"] >= 3


def test_real_previous_schema_migration_keeps_citation_ids(mm, monkeypatch, tmp_path):
    target = tmp_path / "legacy.db"
    with sqlite3.connect(target) as db:
        db.executescript(
            (Path(__file__).parent / "fixtures/legacy_schema_a18faa1.sql").read_text(
                encoding="utf-8"
            )
        )
        db.execute(
            "INSERT INTO products(sku,title,category,category_label,price,source) VALUES('old','Old profile','c','c',10,'fixture')"
        )
        db.execute("""INSERT INTO policy_documents(doc_id,policy_id,topic,title,content,source_name,checksum)
                    VALUES('old-doc','old-policy','保修','保修','保修24个月','fixture','hash')""")
        db.execute("""INSERT INTO policy_chunks(chunk_id,doc_id,chunk_type,ordinal,content,
                    contextualized_content,content_hash) VALUES('old-citation','old-doc','leaf',0,'保修24个月','保修24个月','hash')""")
        data = png()
        key = store().put(data)
        db.execute(
            """INSERT INTO assets(asset_id,sku,asset_type,source_uri,storage_uri,content_hash,mime_type,status,revision)
                    VALUES('old-asset','old','product_poster','source',?,?,'image/png','ready','rev')""",
            (key, key),
        )
        db.execute(
            """INSERT INTO asset_elements VALUES('old-element','old-asset','image',NULL,'{}','',0,'',
                   '供电28Wh',?,'{}','fixture','fixture')""",
            (key,),
        )
        db.execute(
            "INSERT INTO asset_chunks VALUES('old-image-citation','old-asset','old-element','ocr_observed','供电28Wh','供电28Wh','null','old')"
        )
        db.execute(
            "INSERT INTO asset_chunk_elements VALUES('old-image-citation','old-element')"
        )
        db.execute(
            """INSERT INTO assets(asset_id,doc_id,asset_type,source_uri,storage_uri,content_hash,
               mime_type,status,revision,source_version) VALUES(\'policy-picture\',\'old-doc\',
               \'document_picture\',\'source\',?,?,\'image/png\',\'ready\',\'rev\',\'1.0\')""",
            (key, key),
        )
        db.execute(
            """INSERT INTO asset_elements VALUES(\'policy-element\',\'policy-picture\',\'image\',1,
               \'{}\',\'保修\',0,\'保修24个月\',\'保修24个月\',?,\'{}\',\'fixture\',\'fixture\')""",
            (key,),
        )
        db.execute("""INSERT INTO asset_chunks VALUES(\'shared-policy-id\',\'policy-picture\',
            \'policy-element\',\'ocr_observed\',\'保修24个月\',\'保修24个月\',\'null\',\'old\')""")
        db.execute(
            "INSERT INTO asset_chunk_elements VALUES('shared-policy-id','policy-element')"
        )
        db.execute("""INSERT INTO policy_chunks(chunk_id,doc_id,chunk_type,ordinal,content,
            contextualized_content,content_hash) VALUES(\'shared-policy-id\',\'old-doc\',\'leaf\',1,
            \'保修24个月\',\'保修24个月\',\'hash\')""")
    monkeypatch.setattr(settings, "database_path", target)
    init_db()
    assert hydrate("old-citation").text_content == "保修24个月"
    assert hydrate("old-image-citation").text_content == "供电28Wh"
    projected = hydrate("shared-policy-id")
    assert projected.doc_id == "old-doc" and projected.asset_id == "policy-picture"
    assert projected.source_id == "asset:policy-picture"
    with db_session() as db:
        counts = [
            db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("knowledge_units", "knowledge_fts")
        ]
    init_db()
    with db_session() as db:
        assert counts == [
            db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("knowledge_units", "knowledge_fts")
        ]
        assert not db.execute("PRAGMA foreign_key_check").fetchall()

def test_explicit_browse_uses_business_candidates_without_semantic_votes(mm, monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("browse must not run semantic retrieval")

    monkeypatch.setattr("RetailPilot.retrieval.retrieve", unexpected)
    result = search_engine.search(ParsedQuery(raw_query="推荐热门商品", rewritten_query="推荐热门商品"))
    assert result["products"]
    assert result["retrieval"]["routes"] == {}
    assert result["retrieval"]["knowledge"]["business_only"]
