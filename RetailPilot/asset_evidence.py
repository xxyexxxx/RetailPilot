"""Hydration and query-scoped verification of original asset evidence."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .asset_config import asset_settings as config
from .asset_ingestion import asset_url
from .asset_retrieval import aggregate_routes, retrieve_product_assets
from .asset_store import store
from .database import db_session, json_loads
from .execution import BoundedPool, call_deadline
from .text import tokenize

_vision_pool = BoundedPool("asset-vision", 2, 4)
_VISUAL_QUESTION = re.compile(
    r"流程图|示意图|结构图|哪条分支|哪个分支|箭头|左右|位置关系|长什么样|外观|颜色|"
    r"哪一侧|哪边|图中|图里|宣传图|详情图|海报|参数图|对比图|水杯袋|水杯侧袋|滤芯结构|故障灯|指示灯闪|"
    r"\b(?:diagram|flowchart|looks? like|image|picture|appearance|colour|color)\b",
    re.IGNORECASE,
)
_POLICY_RIGHTS = re.compile(
    r"退款|退货|换货|赔偿|金额|有效期|期限|几天|政策|义务|返还|保修多长|refund|return policy|days",
    re.IGNORECASE,
)


@dataclass
class EvidencePackage:
    evidence_id: str
    chunk_id: str
    asset_id: str
    element_id: str
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


def _preview_url(asset_id: str, element_id: str) -> str:
    return f"/api/assets/{asset_id}/elements/{element_id}/preview"


def hydrate(chunk_id: str, *, include_catalog: bool = False) -> EvidencePackage | None:
    if not config.enabled:
        return None
    with db_session() as db:
        row = db.execute(
            """
            SELECT ch.*,a.sku,a.doc_id,a.asset_type,a.source_version,a.source_uri,a.storage_uri,
                   a.revision,e.page_no,e.bbox_json,e.section_path,e.preview_key,
                   e.element_type,e.metadata_json,e.parser_version,e.enrichment_version,
                   d.version AS current_version,d.status AS document_status,d.source_name
            FROM asset_chunks ch JOIN assets a USING(asset_id) JOIN asset_elements e USING(element_id)
            LEFT JOIN policy_documents d ON d.doc_id=a.doc_id
            WHERE ch.chunk_id=? AND a.status IN ('ready','partial')
              AND (a.doc_id IS NULL OR (a.source_version=d.version AND d.status='active'))
            """,
            (chunk_id,),
        ).fetchone()
        if not row:
            return None
        page = db.execute(
            """SELECT element_id FROM asset_elements
            WHERE asset_id=? AND element_type='page' AND page_no=? AND preview_key IS NOT NULL
            LIMIT 1""",
            (row["asset_id"], row["page_no"]),
        ).fetchone()
    if not store().path(row["storage_uri"]).is_file():
        return None
    meta = json_loads(row["metadata_json"], {})
    preview = row["preview_key"] and store().path(row["preview_key"]).is_file()
    catalog = {}
    if include_catalog and row["sku"]:
        from .catalog import get_products

        products = get_products([row["sku"]])
        catalog = products[0] if products else {}
        catalog = {
            key: catalog.get(key)
            for key in ("title", "brand", "stock", "price", "attributes", "features")
        }
    evidence_type = row["evidence_type"]
    if (
        row["sku"]
        and evidence_type == "ocr_observed"
        and row["asset_type"] in {"product_poster", "product_comparison_image"}
    ):
        evidence_type = "seller_advertised"
    return EvidencePackage(
        row["chunk_id"],
        row["chunk_id"],
        row["asset_id"],
        row["element_id"],
        row["sku"],
        row["doc_id"],
        evidence_type,
        row["content"],
        json_loads(row["structured_json"], None),
        {
            "page_no": row["page_no"],
            "bbox": json_loads(row["bbox_json"], None),
            "section_path": row["section_path"],
            "source_uri": asset_url(row["asset_id"], row["revision"]),
            "asset_revision": row["revision"],
            "source_version": row["source_version"] or row["revision"],
            "preview_url": _preview_url(row["asset_id"], row["element_id"])
            if preview
            else None,
            "page_preview_url": _preview_url(row["asset_id"], page["element_id"])
            if page
            else None,
        },
        {
            "source": row["source_name"] or "product_asset",
            "asset_type": row["asset_type"],
            "element_type": row["element_type"],
            "parser_version": row["parser_version"],
            "enrichment_version": row["enrichment_version"],
            **{
                key: meta[key]
                for key in (
                    "source_ref",
                    "provenance",
                    "context_element_ids",
                    "caption",
                    "ocr_observed",
                    "vlm_observed",
                    "seller_advertised",
                    "image_type",
                    "unknown",
                )
                if key in meta
            },
        },
        verification_status=(
            "source_text"
            if row["element_type"] in {"text", "heading", "table"}
            else "unverified"
        ),
        catalog_fields=catalog,
    )


def hydrate_asset(asset_id: str, *, limit: int = 20) -> list[EvidencePackage]:
    with db_session() as db:
        ids = [
            row["chunk_id"]
            for row in db.execute(
                "SELECT chunk_id FROM asset_chunks WHERE asset_id=? ORDER BY rowid LIMIT ?",
                (asset_id, limit),
            )
        ]
    return [evidence for cid in ids if (evidence := hydrate(cid))]


def public_evidence(hit: dict) -> dict | None:
    package = hydrate(hit["chunk_id"])
    if (
        package is None
        or package.sku != hit["sku"]
        or package.asset_id != hit["asset_id"]
    ):
        return None
    payload = asdict(package)
    # Captions can support discovery. They do not serve as answer text.
    meta = payload["provenance"]
    payload["matched_description"] = (
        meta.get("seller_advertised")
        or meta.get("vlm_observed")
        or meta.get("caption")
        or package.text_content
    )[:480]
    payload["text_content"] = package.text_content[:650]
    payload["routes"] = hit["routes"]
    return payload


def requires_visual(query: str) -> bool:
    return bool(_VISUAL_QUESTION.search(query))


def requires_asset(query: str) -> bool:
    return requires_visual(query) or bool(
        re.search(
            r"说明书|怎么安装|如何安装|安装步骤|安装方法|manual|installation",
            query,
            re.IGNORECASE,
        )
    )


def verify_image(package: EvidencePackage, query: str) -> str:
    if not package.source_location.get("preview_url") or not config.vlm_model:
        return ""
    from .asset_parsing import VisionProvider

    with db_session() as db:
        row = db.execute(
            "SELECT preview_key FROM asset_elements WHERE element_id=? AND asset_id=?",
            (package.element_id, package.asset_id),
        ).fetchone()
    if not row or not row["preview_key"]:
        return ""
    deadline = time.perf_counter() + config.verification_timeout
    inherited = call_deadline.get()
    if inherited:
        deadline = min(deadline, inherited)
    try:
        result = _vision_pool.call(
            lambda: VisionProvider().verify(store().read(row["preview_key"]), query),
            deadline,
        )
        live = hydrate(package.chunk_id)
        return (
            result if live and live.source_location == package.source_location else ""
        )
    except Exception:  # noqa: BLE001 - the optional provider fails closed
        return ""


def asset_citation(
    package: EvidencePackage, quote: str, *, verified: bool = False
) -> dict:
    return {
        "asset_id": package.asset_id,
        "element_id": package.element_id,
        "chunk_id": package.chunk_id,
        "sku": package.sku,
        "doc_id": package.doc_id,
        "field": "asset_content",
        "value": quote,
        "quoted_text": quote,
        "source": package.provenance["source"],
        **package.source_location,
        "evidence_type": "visual_observation" if verified else package.evidence_type,
        "verification_status": "image_verified"
        if verified
        else package.verification_status,
    }


def meaningful_overlap(query: str, value: str) -> set[str]:
    ignored = {
        "商品",
        "产品",
        "图片",
        "什么",
        "多少",
        "哪款",
        "这款",
        "这个",
        "怎么",
        "有没有",
        "具有",
        "宣传",
        "可以",
        "能否",
        "支持",
        "请问",
        "是否",
        "能",
        "有",
        "图",
    }
    tokens = lambda text: {
        token
        for token in tokenize(text, include_expansion=False)
        if token not in ignored and (len(token) > 1 or token.isdecimal())
    }
    return tokens(query) & tokens(value)


def _extract(query: str, raw: str) -> str:
    lines = [line.strip() for line in raw.splitlines() if 1 < len(line.strip()) <= 560]
    lines.sort(key=lambda line: -len(meaningful_overlap(query, line)))
    return lines[0] if lines and meaningful_overlap(query, lines[0]) else ""


def _conflict(package: EvidencePackage, quote: str, query: str) -> bool:
    attrs = package.catalog_fields.get("attributes") or {}
    # Flag conflicting exact specifications when the source actually names the field.
    unit = r"\d+(?:[.,]\d+)?\s*(?:mah|wh|kg|mm|cm|hz|db|gb|tb|ml|小时|分钟|克|米|瓦|升|w|v)\b|\d+(?=小时|分钟|克|米|瓦|升)"
    exact = lambda value: {
        re.sub(r"\s+", "", x).lower()
        for x in re.findall(unit, str(value), re.IGNORECASE)
    }
    for key, value in attrs.items():
        name = str(key).strip().casefold()
        if (
            len(name) < 2
            or name not in quote.casefold()
            or name not in query.casefold()
        ):
            continue
        left, right = exact(value), exact(quote)
        if left and right and not (left & right):
            return True
    return False


def product_asset_answer(query: str, products: list[dict]) -> dict:
    if not config.enabled or not products:
        return {
            "lines": [],
            "citations": [],
            "sources": [],
            "needed_visual": config.enabled and requires_asset(query),
        }
    skus = [product["sku"] for product in products[:4]]
    hits, _ = retrieve_product_assets(
        query,
        where=f"p.sku IN ({','.join('?' for _ in skus)})",
        params=skus,
        in_stock=False,
    )
    _, results = aggregate_routes(hits)
    sources, lines, citations = [], [], []
    calls = 0
    used_images: set[tuple[str, int | None, str]] = set()
    verified_at = time.perf_counter() + config.verification_timeout
    needs_vision = requires_visual(query)
    for product in products[:4]:
        for hit in results.get(product["sku"], []):
            package = hydrate(hit["chunk_id"], include_catalog=True)
            if not package or package.sku != product["sku"]:
                continue
            is_image = package.provenance["element_type"] == "image"
            spatial = bool(
                re.search(
                    r"流程图|示意图|哪一侧|哪边|箭头|分支|结构关系|diagram|flowchart",
                    query,
                    re.IGNORECASE,
                )
            )
            value = (
                "" if is_image and spatial else _extract(query, package.text_content)
            )
            verified = False
            image_key = (
                package.asset_id,
                package.source_location["page_no"],
                str(package.source_location["bbox"]),
            )
            if (
                is_image
                and not value
                and needs_vision
                and config.vlm_model
                and calls < config.verification_images
                and image_key not in used_images
                and time.perf_counter() < verified_at
            ):
                calls += 1
                used_images.add(image_key)
                timeout_token = call_deadline.set(
                    min(call_deadline.get() or verified_at, verified_at)
                )
                try:
                    value = verify_image(package, query)
                    verified = bool(value)
                finally:
                    call_deadline.reset(timeout_token)
            if not value:
                continue
            if _conflict(package, value, query):
                lines.append(
                    f"{product['title']}：商品参数与资产内容有冲突，暂无法确认该参数。"
                )
                break
            label = (
                "图片核验观察"
                if verified
                else {
                    "seller_advertised": "商家宣传图文字（OCR）",
                    "ocr_observed": "图片文字（OCR）",
                    "structured_table": "资料表格",
                }.get(package.evidence_type, "商品资料原文")
            )
            citation = asset_citation(package, value, verified=verified)
            lines.append(f"{product['title']} · {label}：「{value}」")
            citations.append(citation)
            sources.append(citation)
            break
    return {
        "lines": lines,
        "citations": citations,
        "sources": sources,
        "needed_visual": requires_asset(query),
        "verification_calls": calls,
    }


def policy_asset_evidence(
    document: dict, query: str, budget: dict[str, Any]
) -> dict | None:
    """A visual observation can support diagram instructions, never policy rights."""
    if not (document.get("chunk_metadata") or {}).get("asset_id"):
        return document
    package = hydrate(document["chunk_id"])
    if not package or package.doc_id != document["doc_id"]:
        return None
    if package.provenance["element_type"] != "image":
        return {
            **document,
            "asset_evidence": asset_citation(package, package.text_content),
            "contextualized_content": package.text_content,
        }
    if not requires_visual(query) or _POLICY_RIGHTS.search(query):
        return None
    cache = budget["observations"]
    if package.chunk_id not in cache:
        cache[package.chunk_id] = ""
        if (
            config.vlm_model
            and budget["remaining"] > 0
            and time.perf_counter() < budget["deadline"]
        ):
            budget["remaining"] -= 1
            token = call_deadline.set(
                min(call_deadline.get() or budget["deadline"], budget["deadline"])
            )
            try:
                cache[package.chunk_id] = verify_image(package, query)
            finally:
                call_deadline.reset(token)
    value = cache[package.chunk_id]
    if not value:
        return None
    content = "图示观察：" + value
    return {
        **document,
        "content": content,
        "context": content,
        "evidence": content,
        "contextualized_content": content,
        "asset_evidence": asset_citation(package, content, verified=True),
    }


def valid_asset_citation(citation: dict, trusted: list[dict]) -> bool:
    package = hydrate(str(citation.get("chunk_id") or ""))
    if not package or not any(
        all(
            source.get(key) == citation.get(key)
            for key in (
                "asset_id",
                "chunk_id",
                "element_id",
                "sku",
                "doc_id",
                "value",
                "verification_status",
            )
        )
        for source in trusted
    ):
        return False
    quote = str(citation.get("value") or "")
    if not quote or citation.get("quoted_text") != quote:
        return False
    verified = citation["verification_status"] == "image_verified"
    if not verified and quote not in package.text_content:
        return False
    if verified and package.provenance["element_type"] != "image":
        return False
    expected = asset_citation(package, quote, verified=verified)
    return all(
        citation.get(key) == value
        for key, value in expected.items()
        if key != "field" or citation.get("field") != "content"
    )
