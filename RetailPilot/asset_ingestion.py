"""Idempotent asset intake, modality-aware chunks and transactional publication."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
import secrets
import time
from dataclasses import asdict, dataclass
from threading import Lock
from urllib.parse import quote

from .asset_config import asset_settings as config
from .asset_parsing import (
    AssetParseError,
    DocumentElement,
    ParsedDocument,
    enrich_context,
    parse_asset,
)
from .asset_store import detect_mime, store
from .database import db_session, json_dumps, json_loads
from .execution import call_deadline
from .policy_ingestion import approximate_token_count
from .text import tokenize

ASSET_TYPES = frozenset(
    {
        "product_main_image",
        "product_detail_image",
        "product_poster",
        "product_comparison_image",
        "product_document",
        "document_picture",
        "document_table",
        "document_page",
    }
)
READY = ("ready", "partial")
INDEX_VERSION = "asset-chunks-v1"
_parse_lock = Lock()


def _id(*values: str) -> str:
    return hashlib.sha256(json_dumps(values).encode()).hexdigest()[:32]


def _distribution(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def parser_signature(mime: str) -> str:
    from .multimodal_model import config as model_config
    reasoning=model_config("image_description")
    from .multimodal_model import ready_for_vision
    vision_capable=ready_for_vision("image_description")
    parser = (
        (config.parser, _distribution(config.parser), config.docling_artifacts)
        if mime == "application/pdf"
        else (
            "pillow",
            _distribution("Pillow"),
        )
    )
    return _id(
        "asset-parser-v1",
        *parser,
        config.ocr,
        config.ocr_languages,
        config.vlm_url,
        config.vlm_model,
        reasoning.model, reasoning.base_url, str(reasoning.vision), str(vision_capable),
        str(config.tile_height),
        str(config.max_pixels),
        str(config.max_pages),
        str(config.verification_tokens),
    )


def _revision(digest: str, source_version: str, mime: str) -> str:
    return _id(
        digest,
        source_version,
        parser_signature(mime),
        INDEX_VERSION,
        str(config.context_tokens),
    )


def asset_url(asset_id: str, revision: str = "") -> str:
    url = f"/api/assets/{asset_id}/content"
    return f"{url}?revision={revision}" if revision else url


def get_asset(asset_id: str) -> dict | None:
    with db_session() as db:
        row = db.execute(
            "SELECT * FROM assets WHERE asset_id=?", (asset_id,)
        ).fetchone()
    return dict(row) if row else None


def stage_asset(
    data: bytes,
    *,
    owner_type: str,
    owner_id: str,
    kind: str,
    source_key: str,
    source_version: str = "",
) -> tuple[str, bool]:
    if not data or len(data) > config.max_bytes:
        raise ValueError("file_size_limit")
    mime = detect_mime(data)
    if owner_type not in {"product", "policy"} or kind not in ASSET_TYPES:
        raise ValueError("invalid_asset_type")
    if owner_type == "policy" and (
        not kind.startswith("document_")
        or (mime == "application/pdf" and kind != "document_page")
    ):
        raise ValueError("policy_asset_kind_mismatch")
    if owner_type == "product" and (
        (kind == "product_document") != (mime == "application/pdf")
        or not kind.startswith("product_")
    ):
        raise ValueError("asset_kind_mime_mismatch")
    if not re.fullmatch(r"[\w.-]{1,160}", source_key) or source_key in {".", ".."}:
        raise ValueError("invalid_source_key")
    if not owner_id or len(owner_id) > 160 or len(source_version) > 80:
        raise ValueError("invalid_asset_owner")
    asset_id = _id(owner_type, owner_id, kind, source_key)
    column, table = (
        ("sku", "products")
        if owner_type == "product"
        else ("doc_id", "policy_documents")
    )
    digest = hashlib.sha256(data).hexdigest()
    with db_session() as db:
        db.execute("BEGIN IMMEDIATE")
        owner = db.execute(
            f"SELECT * FROM {table} WHERE {column}=?", (owner_id,)
        ).fetchone()
        if not owner:
            raise LookupError("asset_owner_not_found")
        if owner_type == "policy":
            if source_version and source_version != owner["version"]:
                raise ValueError("policy_version_mismatch")
            source_version = owner["version"]
        revision = _revision(digest, source_version, mime)
        old = db.execute(
            "SELECT * FROM assets WHERE asset_id=?", (asset_id,)
        ).fetchone()
        if (
            old
            and old["revision"] == revision
            and old["status"] in READY
            and store().path(digest).is_file()
        ):
            return asset_id, False
        if old and old["status"] == "processing":
            active = db.execute(
                "SELECT updated_at > datetime('now', ?) FROM assets WHERE asset_id=?",
                (f"-{config.processing_timeout + 30} seconds", asset_id),
            ).fetchone()[0]
            if active:
                raise RuntimeError("asset_processing")
        storage_key = store().put(data)
        db.execute(
            f"""INSERT INTO assets(asset_id,{column},asset_type,source_uri,storage_uri,
                                   content_hash,mime_type,status,revision,source_version)
                VALUES(?,?,?,?,?,?,?,'queued',?,?)
                ON CONFLICT(asset_id) DO UPDATE SET storage_uri=excluded.storage_uri,
                  content_hash=excluded.content_hash,mime_type=excluded.mime_type,
                  status='queued',revision=excluded.revision,source_version=excluded.source_version,
                  error_code='',updated_at=CURRENT_TIMESTAMP""",
            (
                asset_id,
                owner_id,
                kind,
                f"asset-source:{quote(source_key)}",
                storage_key,
                digest,
                mime,
                revision,
                source_version,
            ),
        )
    return asset_id, True


@dataclass
class AssetChunk:
    element_id: str
    evidence_type: str
    content: str
    search_text: str
    structured: dict | None = None


def _text_windows(text: str, maximum: int = 560) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sections = re.split(r"(?<=[。！？])\s*|(?<=[.!?])\s+|\n+", text)
    pieces = []
    for section in sections:
        while approximate_token_count(section) > maximum:
            end = min(len(section) - 1, maximum)
            while approximate_token_count(section[:end]) > maximum:
                end //= 2
            pieces.append(section[:end])
            section = section[end:]
        if section:
            pieces.append(section)
    windows: list[str] = []
    for piece in pieces:
        if windows and approximate_token_count(windows[-1] + "\n" + piece) <= maximum:
            windows[-1] += "\n" + piece
        else:
            windows.append(piece)
    return windows


def _table_chunks(element_id: str, element: DocumentElement) -> list[AssetChunk]:
    table = element.metadata["table"]
    cells = table["cells"]
    headers = [cell for cell in cells if cell["column_header"]]
    first_row = max((cell["row_end"] for cell in headers), default=0)
    chunks = []
    for start in range(first_row, max(first_row + 1, table["rows"]), 8):
        end = min(table["rows"], start + 8)
        segment = [
            cell
            for cell in cells
            if not cell["column_header"]
            and cell["row_start"] < end
            and cell["row_end"] > start
        ]
        lines = []
        for row in range(start, end):
            entries = []
            for cell in sorted(segment, key=lambda entry: entry["col_start"]):
                if cell["row_start"] <= row < cell["row_end"] and cell["text"].strip():
                    labels = [
                        header["text"]
                        for header in headers
                        if header["col_start"] < cell["col_end"]
                        and header["col_end"] > cell["col_start"]
                        and header["text"].strip()
                    ]
                    label = (
                        " / ".join(dict.fromkeys(labels))
                        or f"列 {cell['col_start'] + 1}"
                    )
                    entries.append(f"{label} = {cell['text']}")
            if entries:
                lines.append(" | ".join(entries))
        raw = "\n".join(lines) or " | ".join(cell["text"] for cell in headers)
        if raw.strip():
            chunks.append(
                AssetChunk(
                    element_id,
                    "structured_table",
                    raw,
                    f"{raw}\n{element.search_text}",
                    {
                        "rows": table["rows"],
                        "columns": table["columns"],
                        "row_start": start,
                        "row_end": end,
                        "cells": headers + segment,
                    },
                )
            )
    return chunks


def build_chunks(element_id: str, element: DocumentElement) -> list[AssetChunk]:
    if element.element_type == "table":
        return _table_chunks(element_id, element)
    if element.element_type == "image":
        meta = element.metadata
        raw = (meta.get("ocr_observed") or "").strip()
        provenance = "ocr_observed" if raw else "image"
        return [AssetChunk(element_id, provenance, raw, element.search_text)]
    if element.element_type == "page":
        return []
    return [
        AssetChunk(element_id, "document_text", raw, raw)
        for raw in _text_windows(element.raw_content)
    ]


def _parse_cached(asset: dict) -> ParsedDocument:
    key = _id(asset["content_hash"], parser_signature(asset["mime_type"]))
    # The processing pool is intentionally a single worker per process.
    with _parse_lock:
        with db_session() as db:
            row = db.execute(
                "SELECT payload FROM asset_parse_cache WHERE cache_key=?", (key,)
            ).fetchone()
        if row:
            data = json_loads(row["payload"], {})
            elements = [DocumentElement(**element) for element in data["elements"]]
            if all(
                not item.preview_key or store().path(item.preview_key).is_file()
                for item in elements
            ):
                return ParsedDocument(
                    elements, data["parser_version"], data["warnings"]
                )
        parsed = parse_asset(store().path(asset["storage_uri"]), asset["mime_type"])
        if not any(warning != "pdf_text_only" for warning in parsed.warnings):
            with db_session() as db:
                db.execute(
                    "INSERT OR REPLACE INTO asset_parse_cache VALUES(?,?)",
                    (key, json_dumps(asdict(parsed))),
                )
        return parsed


def terms(text: str) -> str:
    return " ".join(dict.fromkeys(tokenize(text, include_expansion=False)))


def process_asset(asset_id: str, *, retry: bool = False) -> dict | None:
    token = secrets.token_hex(16)
    with db_session() as db:
        db.execute("BEGIN IMMEDIATE")
        permitted = (
            "('queued')" if not retry else "('queued','failed','partial','ready')"
        )
        changed = db.execute(
            f"""UPDATE assets SET status='processing', updated_at=CURRENT_TIMESTAMP,
                    error_code=?, processing_token=? WHERE asset_id=? AND
                (status IN {permitted} OR (status='processing' AND updated_at <= datetime('now',?)))""",
            ("", token, asset_id, f"-{config.processing_timeout + 30} seconds"),
        )
        asset = db.execute(
            "SELECT * FROM assets WHERE asset_id=?", (asset_id,)
        ).fetchone()
    if not changed.rowcount:
        return dict(asset) if asset else None
    asset = dict(asset)
    asset["revision"] = _revision(
        asset["content_hash"], asset["source_version"], asset["mime_type"]
    )
    deadline = time.perf_counter() + config.processing_timeout
    inherited = call_deadline.get()
    context_token = call_deadline.set(
        min(deadline, inherited) if inherited else deadline
    )
    try:
        parsed = _parse_cached(asset)
        parsed.doc_id = asset["doc_id"] or asset_id
        with db_session() as db:
            owner = (
                db.execute(
                    "SELECT title, version FROM policy_documents WHERE doc_id=?",
                    (asset["doc_id"],),
                ).fetchone()
                if asset["doc_id"]
                else None
            )
        if asset["doc_id"] and (
            not owner or owner["version"] != asset["source_version"]
        ):
            raise ValueError("policy_version_mismatch")
        enrich_context(parsed, title=owner["title"] if owner else "")
        ids = {
            element.reading_order: _id(
                asset_id, asset["revision"], str(element.reading_order)
            )
            for element in parsed.elements
        }
        elements = {ids[item.reading_order]: item for item in parsed.elements}
        chunks = [
            (_id(element_id, str(i), _id(chunk.content, chunk.search_text)), chunk)
            for element_id, element in elements.items()
            for i, chunk in enumerate(build_chunks(element_id, element))
        ]
        if not chunks:
            raise ValueError("no_searchable_elements")
        warnings = []
        expired = time.perf_counter() >= call_deadline.get()
        if expired:
            raise TimeoutError("asset_deadline")
        warnings = sorted(set(parsed.warnings + warnings))
        with db_session() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT processing_token FROM assets WHERE asset_id=?", (asset_id,)
            ).fetchone()
            if not current or current["processing_token"] != token:
                raise RuntimeError("asset_processing_superseded")
            from .knowledge import put_unit, source

            sid = f"asset:{asset_id}"
            source(
                db,
                source_id=sid,
                domain="product" if asset["sku"] else "policy",
                owner_id=asset["sku"] or asset["doc_id"],
                asset_id=asset_id,
                source_format="pdf"
                if asset["mime_type"] == "application/pdf"
                else "image",
                source_uri=f"asset:{asset_id}",
                version=asset["source_version"] or asset["revision"],
                content_hash=asset["revision"],
            )
            db.execute("DELETE FROM knowledge_units WHERE source_id=?", (sid,))
            db.execute("DELETE FROM asset_elements WHERE asset_id=?", (asset_id,))
            db.executemany(
                """INSERT INTO asset_elements VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                [
                    (
                        eid,
                        asset_id,
                        el.element_type,
                        el.page_no,
                        json_dumps(el.bbox),
                        el.section_path,
                        el.reading_order,
                        el.raw_content,
                        el.search_text,
                        el.preview_key,
                        json_dumps(
                            {
                                **el.metadata,
                                "context_element_ids": [
                                    ids[order]
                                    for order in el.metadata.get("context_orders", [])
                                ],
                            }
                        ),
                        parsed.parser_version,
                        parser_signature(asset["mime_type"]),
                    )
                    for eid, el in elements.items()
                ],
            )
            for cid, chunk in chunks:
                el = elements[chunk.element_id]
                put_unit(
                    db,
                    unit_id=cid,
                    source_id=sid,
                    element_id=chunk.element_id,
                    modality=el.element_type,
                    evidence_type=chunk.evidence_type,
                    evidence_text=chunk.content,
                    search_text=chunk.search_text,
                    structured=chunk.structured,
                    section_path=el.section_path,
                    page_no=el.page_no,
                    ordinal=el.reading_order,
                    parser_version=parsed.parser_version,
                    element_ids=[ids[o] for o in el.metadata.get("context_orders", [])],
                )
            db.execute(
                """UPDATE assets SET status=?, error_code=?, processing_token='', revision=?,
                   updated_at=CURRENT_TIMESTAMP WHERE asset_id=?""",
                (
                    "partial" if warnings else "ready",
                    ",".join(warnings),
                    asset["revision"],
                    asset_id,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - status must record any worker failure
        # DB writes after an expired deadline must still record a safe error code.
        call_deadline.reset(context_token)
        context_token = None
        with db_session() as db:
            db.execute(
                """UPDATE assets SET status='failed',error_code=?,processing_token='',
                   updated_at=CURRENT_TIMESTAMP WHERE asset_id=? AND processing_token=?""",
                (
                    str(exc)
                    if isinstance(exc, AssetParseError)
                    else type(exc).__name__,
                    asset_id,
                    token,
                ),
            )
    finally:
        if context_token is not None:
            call_deadline.reset(context_token)
    result = get_asset(asset_id)
    if result and result["status"] in READY:
        from .knowledge_index import build

        domain = "product" if result["sku"] else "policy"
        # Neural text and visual vectors are optional derived indexes.
        reports = [build(domain, source_id=f"asset:{asset_id}")]
        if config.visual_enabled and result["sku"]:
            reports.append(build(domain, space="visual", source_id=f"asset:{asset_id}"))
        if any(report["failed"] for report in reports):
            with db_session() as db:
                db.execute(
                    "UPDATE assets SET status='partial',error_code=error_code||',index_unavailable' WHERE asset_id=?",
                    (asset_id,),
                )
    return get_asset(asset_id)


def delete_asset(asset_id: str) -> bool:
    with db_session() as db:
        return bool(
            db.execute("DELETE FROM assets WHERE asset_id=?", (asset_id,)).rowcount
        )
