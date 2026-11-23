"""One Unit-ID hydration entry point for catalog, Markdown and binary assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .asset_config import asset_settings
from .asset_store import store
from .database import db_session, json_loads
from .knowledge import get_unit, unit_rows


@dataclass
class EvidencePackage:
    evidence_id: str
    chunk_id: str
    asset_id: str | None
    element_id: str | None
    sku: str | None
    doc_id: str | None
    evidence_type: str
    text_content: str
    structured_content: dict | None
    source_location: dict
    provenance: dict
    confidence: float | None = None
    verification_status: str = "unverified"
    catalog_fields: dict = field(default_factory=dict)
    unit_id: str = ""
    source_id: str = ""
    field_refs: list = field(default_factory=list)


def hydrate(unit_id, *, include_catalog=False, db=None):
    if db is None:
        with db_session() as connection:
            return hydrate(unit_id, include_catalog=include_catalog, db=connection)
    records = unit_rows(db, [unit_id])
    if not records:
        return None
    row = dict(records[0])
    unit = get_unit(db, unit_id)
    meta = json_loads(row["metadata_json"], {})
    loc = {
        "source_uri": row["source_uri"],
        "source_version": row["version"],
        "section_path": row["section_path"],
        "page_no": row["page_no"],
        "char_start": row["char_start"],
        "char_end": row["char_end"],
        "bbox": None,
        "preview_url": None,
        "page_preview_url": None,
    }
    evidence_type = row["evidence_type"]
    provenance = {
        "source_id": row["source_id"],
        "source_format": row["source_format"],
        "element_type": row["modality"],
        "source": "catalog" if row["sku"] else "policy",
        "parser_version": row["parser_version"],
        **meta,
    }
    if row["doc_id"]:
        doc = db.execute(
            "SELECT * FROM policy_documents WHERE doc_id=?", (row["doc_id"],)
        ).fetchone()
        if not doc or doc["status"] != "active" or doc["version"] != row["version"]:
            return None
        provenance["source"] = doc["source_name"]
    if row["asset_id"]:
        if not asset_settings.enabled:
            return None
        asset = db.execute(
            "SELECT * FROM assets WHERE asset_id=?", (row["asset_id"],)
        ).fetchone()
        if (
            not asset
            or asset["status"] not in {"ready", "partial"}
            or not store().path(asset["storage_uri"]).is_file()
        ):
            return None
        if row["doc_id"] and asset["source_version"] != row["version"]:
            return None
        element = db.execute(
            "SELECT * FROM asset_elements WHERE element_id=? AND asset_id=?",
            (row["element_id"], row["asset_id"]),
        ).fetchone()
        if not element:
            return None
        from .asset_ingestion import asset_url

        emeta = json_loads(element["metadata_json"], {})
        provenance.update(
            emeta,
            asset_type=asset["asset_type"],
            source=provenance["source"],
            enrichment_version=element["enrichment_version"],
        )
        loc.update(
            source_uri=asset_url(asset["asset_id"], asset["revision"]),
            asset_revision=asset["revision"],
            bbox=json_loads(element["bbox_json"], None),
        )
        preview = lambda eid: f"/api/assets/{asset['asset_id']}/elements/{eid}/preview"
        if element["preview_key"] and store().path(element["preview_key"]).is_file():
            loc["preview_url"] = preview(element["element_id"])
        page = db.execute(
            """SELECT element_id FROM asset_elements WHERE asset_id=? AND page_no=?
            AND element_type='page' AND preview_key IS NOT NULL LIMIT 1""",
            (asset["asset_id"], row["page_no"]),
        ).fetchone()
        if page:
            loc["page_preview_url"] = preview(page[0])
        if (
            row["sku"]
            and evidence_type == "ocr_observed"
            and asset["asset_type"] in {"product_poster", "product_comparison_image"}
        ):
            evidence_type = "seller_advertised"
    catalog = {}
    if row["sku"] and (include_catalog or row["source_format"] == "catalog"):
        product = db.execute(
            "SELECT * FROM products WHERE sku=?", (row["sku"],)
        ).fetchone()
        if not product:
            return None
        catalog = {key: product[key] for key in ("title", "brand", "stock", "price")}
        catalog.update(
            attributes=json_loads(product["attributes_json"], {}),
            features=json_loads(product["features_json"], []),
        )
    return EvidencePackage(
        row["unit_id"],
        row["unit_id"],
        row["asset_id"],
        row["element_id"],
        row["sku"],
        row["doc_id"],
        evidence_type,
        row["evidence_text"],
        (
            {"sku": row["sku"], **catalog}
            if row["source_format"] == "catalog"
            else unit.structured_data
        ),
        loc,
        provenance,
        verification_status="source_text"
        if row["modality"] in {"text", "table"}
        else "unverified",
        catalog_fields=catalog,
        unit_id=row["unit_id"],
        source_id=row["source_id"],
        field_refs=json_loads(row["field_refs_json"], []),
    )


def public_hit(hit, *, db=None):
    package = hydrate(hit.unit_id, db=db)
    if not package:
        return None
    payload = asdict(package)
    meta = payload["provenance"]
    payload["matched_description"] = (
        meta.get("seller_advertised")
        or meta.get("vlm_observed")
        or meta.get("caption")
        or package.text_content
    )[:480]
    payload["text_content"] = package.text_content[:650]
    payload["routes"] = hit.metadata.get("source_ranks", {})
    return payload
