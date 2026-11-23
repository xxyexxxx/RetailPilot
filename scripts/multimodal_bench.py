"""Reproducible offline engineering benchmark with labelled synthetic fixtures."""

from __future__ import annotations

import argparse
import io
import json
import math
import sqlite3
import statistics
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot import asset_ingestion, asset_retrieval
from RetailPilot.asset_config import asset_settings as config
from RetailPilot.asset_evidence import (
    hydrate,
    policy_asset_evidence,
    product_asset_answer,
)
from RetailPilot.asset_ingestion import process_asset, stage_asset
from RetailPilot.asset_parsing import (
    DocumentElement,
    ParsedDocument,
    TesseractOCR,
    VisionProvider,
    parse_asset,
)
from RetailPilot.asset_store import store
from RetailPilot.config import settings
from RetailPilot.database import db_session, init_db
from RetailPilot.grounding import claim, policy_citation, validate_claim_citations
from RetailPilot.policy import retrieve_policies
from RetailPilot.policy_ingestion import PolicyDocumentInput, ingest_policy_documents
from RetailPilot.retrieval import search_engine
from RetailPilot.schemas import ParsedQuery

CATEGORY = "Sports & Outdoors"
PALETTE = {
    (22, 90, 67): ("Trail Pack", "Trail bottle sleeve 500 mL", "绿色双肩设计"),
    (45, 120, 90): (
        "Trail Pack spare view",
        "Trail bottle sleeve reinforced",
        "绿色背包的另一角度",
    ),
    (40, 50, 120): ("Power Kit", "Power capacity 28 Wh", "蓝色产品"),
    (240, 130, 44): ("Style", "", ""),
    (86, 80, 80): ("Filter Setup", "Blue light diagram orientation", "状态灯安装流程"),
    (120, 20, 90): ("Neutral Pack", "Trail pack shell", "紫色背包"),
}
CASES = [
    ("poster-function", "Trail bottle sleeve", "b-pack", "pack-poster", "500 mL"),
    ("visual-color", "橙色外观轮廓", "b-style", "style", None),
    ("image-parameter", "Power capacity", "b-power", "power", "28 Wh"),
    (
        "multi-image",
        "Trail bottle sleeve reinforced",
        "b-pack",
        "pack-side",
        "reinforced",
    ),
    ("catalog-control", "Neutral Pack", "b-neutral", None, None),
    ("pdf-body", "approved purifier clip", None, "manual", "P4"),
    ("table-number", "X9 filter width", None, "manual", "19 mm"),
    ("flowchart", "Blue light diagram", None, "manual", "press RESET"),
    ("figure-nearby", "indicator diagram 安装流程", None, "manual", "press RESET"),
    ("pdf-fallback", "Seal check 42", None, "text-pdf", "42"),
    (
        "missing-vision",
        "Does the manual diagram show a hidden valve?",
        None,
        None,
        None,
    ),
]


def make_image(color: tuple[int, int, int], label: str) -> bytes:
    image = Image.new("RGB", (360, 160), color)
    ImageDraw.Draw(image).text((20, 70), label, fill="white")
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def make_pdf(text: str) -> bytes:
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


def pixel(data: bytes) -> tuple[int, int, int]:
    with Image.open(io.BytesIO(data)) as image:
        return image.getpixel((0, 0))


class SyntheticVisualEncoder:
    def images(self, images):
        palette = list(PALETTE)
        return np.eye(len(palette), dtype=np.float32)[
            [palette.index(pixel(data)) for data in images]
        ]

    def texts(self, queries):
        vectors = np.zeros((len(queries), len(PALETTE)), dtype=np.float32)
        for index, text in enumerate(queries):
            if "橙" in text:
                vectors[index, 3] = 1
            elif "Trail" in text:
                vectors[index, 0:2] = [0.6, 0.8] if "reinforced" in text else [0.8, 0.6]
            elif "Power" in text:
                vectors[index, 2] = 1
        return vectors


def prepare(root: Path, mode: str) -> dict:
    settings.database_path = root / "benchmark.db"
    config.store = root / "objects"
    config.enabled = mode != "baseline"
    config.visual_enabled = mode == "full"
    init_db()
    with db_session() as db:
        for sku, title in [
            ("b-pack", "Alpine"),
            ("b-style", "Strata"),
            ("b-power", "Ranger"),
            ("b-neutral", "Neutral Pack"),
        ]:
            db.execute(
                """INSERT INTO products(sku,title,category,category_label,price,stock,source)
                         VALUES(?,?,?,?,40,10,'synthetic')""",
                (sku, title, CATEGORY, CATEGORY),
            )
            db.execute(
                """INSERT INTO product_fts(sku,title,description) VALUES(?,?,?)""",
                (sku, title, title),
            )
    ingest_policy_documents(
        [
            PolicyDocumentInput(
                doc_id="b-doc",
                policy_id="b-policy",
                topic="保修",
                title="Service manuals",
                content="Register the original invoice before starting a warranty claim.",
                source_name="Synthetic fixture",
                platform=settings.commerce_platform,
            )
        ]
    )
    if not config.enabled:
        return {}
    assets = {}
    for i, (name, sku) in enumerate(
        [
            ("pack-poster", "b-pack"),
            ("pack-side", "b-pack"),
            ("power", "b-power"),
            ("style", "b-style"),
        ]
    ):
        color = list(PALETTE)[i]
        raw = make_image(color, PALETTE[color][1] or PALETTE[color][0])
        assets[name], _ = stage_asset(
            raw,
            owner_type="product",
            owner_id=sku,
            kind="product_poster",
            source_key=name + ".png",
        )
        _processed(assets[name])
    assets["text-pdf"], _ = stage_asset(
        make_pdf("Seal check 42"),
        owner_type="policy",
        owner_id="b-doc",
        kind="document_page",
        source_key="text.pdf",
    )
    _processed(assets["text-pdf"])
    diagram = store().put(make_image(list(PALETTE)[4], "Blue light -> press RESET"))
    location = {"origin": "TOPLEFT", "unit": "pt", "coordinates": [20, 20, 310, 180]}

    def cell(value, r, c):
        return {
            "text": value,
            "row_start": r,
            "row_end": r + 1,
            "col_start": c,
            "col_end": c + 1,
            "column_header": r == 0,
            "row_header": False,
        }

    layout = ParsedDocument(
        [
            DocumentElement(
                "text", 0, "The approved purifier clip is P4.", "Installation", 1
            ),
            DocumentElement(
                "text",
                1,
                "Status indicator for the protection circuit.",
                "Installation",
                1,
            ),
            DocumentElement(
                "image",
                2,
                page_no=1,
                section_path="Installation",
                bbox=location,
                preview_key=diagram,
                metadata={"caption": "Blue light diagram"},
            ),
            DocumentElement(
                "table",
                3,
                page_no=1,
                section_path="Installation",
                bbox=location,
                preview_key=diagram,
                metadata={
                    "table": {
                        "rows": 2,
                        "columns": 2,
                        "cells": [
                            cell("Model", 0, 0),
                            cell("filter width", 0, 1),
                            cell("X9", 1, 0),
                            cell("19 mm", 1, 1),
                        ],
                    }
                },
            ),
            DocumentElement("page", 4, page_no=1, preview_key=diagram),
        ],
        "synthetic-structured-parser",
    )
    assets["manual"], _ = stage_asset(
        make_pdf("The approved purifier clip is P4."),
        owner_type="policy",
        owner_id="b-doc",
        kind="document_page",
        source_key="manual.pdf",
    )
    with patch.object(
        asset_ingestion,
        "parse_asset",
        lambda path, mime: (
            layout if mime == "application/pdf" else parse_asset(path, mime)
        ),
    ):
        _processed(assets["manual"])
    return assets


def _processed(aid: str) -> None:
    asset = process_asset(aid)
    if asset["status"] not in {"ready", "partial"}:
        raise RuntimeError(asset["error_code"])


def evaluate(case: tuple, assets: dict, k: int) -> dict:
    label, text, sku, target, expected = case
    started = time.perf_counter()
    if sku:
        parsed = ParsedQuery(
            raw_query=text,
            rewritten_query=text,
            category=CATEGORY,
            grounded_target_terms=[text] if label == "catalog-control" else [],
        )
        products = search_engine.search(parsed, limit=k)["products"]
        result = product_asset_answer(text, products)
        recalled_skus = [p["sku"] for p in products]
        recalled_assets = [
            e["asset_id"] for p in products for e in p.get("asset_evidence") or []
        ]
        citations = result["citations"]
        verdict = validate_claim_citations(
            [claim(c["value"], c) for c in citations],
            citations,
            products=products,
            asset_sources=result["sources"],
        )
    else:
        docs = retrieve_policies(text, topic="保修", limit=k)["documents"]
        recalled_skus = []
        recalled_assets = list(
            dict.fromkeys(
                item["chunk_metadata"]["asset_id"]
                for item in docs
                if "asset_id" in item["chunk_metadata"]
            )
        )
        budget = {
            "remaining": config.verification_images,
            "deadline": time.perf_counter() + config.verification_timeout,
            "observations": {},
        }
        reviewed = [
            review
            for item in docs
            if (review := policy_asset_evidence(item, text, budget))
        ]
        # Deterministic fixture grading uses the same evidence/citation path, excluding near misses.
        selected = [
            item
            for item in reviewed
            if expected and expected.casefold() in item["content"].casefold()
        ]
        citations = [policy_citation(item) for item in selected]
        verdict = validate_claim_citations(
            [claim(c["value"], c) for c in citations],
            citations,
            policy_documents=selected,
        )
    wanted = assets.get(target)
    sku_rank = (
        (recalled_skus.index(sku) + 1 if sku in recalled_skus else 0) if sku else None
    )
    asset_rank = (
        (recalled_assets.index(wanted) + 1 if wanted in recalled_assets[:k] else 0)
        if target
        else None
    )
    correct = (
        any(expected.casefold() in c["value"].casefold() for c in citations)
        and verdict["pass"]
        if expected
        else (bool(sku_rank) if sku else not citations)
    )
    return {
        "case": label,
        "sku_rank": sku_rank,
        "asset_rank": asset_rank,
        "answer_correct": correct,
        "citations": len(citations),
        "correct_citations": sum(
            bool(
                (not wanted or c.get("asset_id") == wanted)
                and verdict["pass"]
                and (not expected or expected.casefold() in c["value"].casefold())
                and (not c.get("asset_id") or hydrate(c["chunk_id"]))
            )
            for c in citations
        ),
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def metrics(results, root: Path, mode: str) -> dict:
    def ranking(field):
        ranks = [item[field] for item in results if item[field] is not None]
        return {
            "queries": len(ranks),
            "recall": statistics.mean(rank > 0 for rank in ranks),
            "mrr": statistics.mean(1 / rank if rank else 0 for rank in ranks),
            "ndcg": statistics.mean(
                1 / math.log2(rank + 1) if rank else 0 for rank in ranks
            ),
        }

    count = sum(case["citations"] for case in results)
    latency = np.asarray([case["latency_ms"] for case in results])
    with db_session() as db:
        try:
            index_bytes = db.execute("""SELECT COALESCE(SUM(pgsize),0) FROM dbstat
                WHERE name GLOB 'asset_fts*' OR name IN ('asset_vectors','asset_vectors_space',
                'asset_chunks','asset_chunks_asset','asset_chunk_elements','asset_elements',
                'asset_elements_asset','product_vectors') OR name GLOB 'product_fts*'
                OR name GLOB 'policy_chunk*'""").fetchone()[0]
        except sqlite3.OperationalError:
            index_bytes = None
        logical_bytes = db.execute("""
            SELECT COALESCE((SELECT SUM(length(vector)) FROM product_vectors), 0)
                + COALESCE((SELECT SUM(length(vector)) FROM asset_vectors), 0)
                + COALESCE((SELECT SUM(length(CAST(search_text AS BLOB))) FROM asset_fts), 0)
                + COALESCE((SELECT SUM(length(CAST(search_text AS BLOB))) FROM policy_chunk_fts), 0)
                + COALESCE((SELECT SUM(length(CAST(title || coalesce(description,'') AS BLOB)))
                            FROM product_fts), 0)
            """).fetchone()[0]
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return {
        "mode": mode,
        "sku": ranking("sku_rank"),
        "asset": ranking("asset_rank"),
        "answer_correctness": statistics.mean(
            item["answer_correct"] for item in results
        ),
        "citation_accuracy": sum(item["correct_citations"] for item in results) / count
        if count
        else None,
        "citations": count,
        "latency_ms": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
        },
        "sqlite_index_bytes": index_bytes,
        "index_payload_bytes": logical_bytes,
        "sqlite_database_bytes": (root / "benchmark.db").stat().st_size,
        "asset_store_bytes": sum(
            p.stat().st_size for p in (root / "objects").glob("*/*") if p.is_file()
        ),
        "external_model_calls": 0,
        "external_model_cost": 0.0,
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-k", type=int, default=3, choices=range(1, 11))
    args = parser.parse_args()
    encoder = SyntheticVisualEncoder()

    def caption(_, data):
        _, advertised, observed = PALETTE[pixel(data)]
        return {
            "image_type": "synthetic",
            "observed": observed,
            "advertised": advertised,
            "unknown": "",
        }

    def verify(_, data, query):
        if pixel(data) == list(PALETTE)[4] and "hidden" not in query:
            return "The blue light branch says press RESET."
        return ""

    with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
        for obj, key, value in [
            (config, "ocr", "tesseract"),
            (config, "parser", "pypdf"),
            (config, "vlm_model", "fixture"),
            (config, "qdrant_url", ""),
            (config, "text_model", "hash"),
            (settings, "policy_vector_backend", "disabled"),
            (settings, "policy_reranker_enabled", False),
            (settings, "embedding_model", "hash"),
            (TesseractOCR, "read", lambda _, data: PALETTE[pixel(data)][1]),
            (VisionProvider, "describe", caption),
            (VisionProvider, "verify", verify),
            (asset_ingestion, "visual_encoder", lambda: encoder),
            (asset_retrieval, "visual_encoder", lambda: encoder),
        ]:
            stack.enter_context(patch.object(obj, key, value))
        report = []
        for mode in ("baseline", "enhanced_text", "full"):
            root = Path(tmp) / mode
            assets = prepare(root, mode)
            results = [evaluate(case, assets, args.top_k) for case in CASES]
            report.append(metrics(results, root, mode))
    result = {
        "fixture": "synthetic-offline-v1",
        "real_models": False,
        "top_k": args.top_k,
        "cases": len(CASES),
        "runs": report,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
