"""Hydration and query-scoped verification of original asset evidence."""

from __future__ import annotations

import re
import time
from typing import Any

from .asset_config import asset_settings as config
from .asset_store import store
from .database import db_session
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
    r"退款|退货|换货|赔偿|赔付|金额|费用|收费|有效期|期限|几天|政策|义务|返还|保修|refund|warranty|return policy|days|eligibility",
    re.IGNORECASE,
)


from .knowledge_evidence import EvidencePackage, hydrate
from .knowledge_retrieval import (
    RetrievalFilters,
    RetrievalQuery,
    aggregate_skus,
    retrieve,
)


def hydrate_asset(asset_id: str, *, limit: int = 20) -> list[EvidencePackage]:
    with db_session() as db:
        ids = [
            row["chunk_id"]
            for row in db.execute(
                "SELECT unit_id AS chunk_id FROM knowledge_units JOIN knowledge_sources USING(source_id) WHERE asset_id=? ORDER BY ordinal LIMIT ?",
                (asset_id, limit),
            )
        ]
    return [evidence for cid in ids if (evidence := hydrate(cid))]


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
    from .multimodal_model import ready_for_vision
    if not package.source_location.get("preview_url") or not ready_for_vision():
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
        "unit_id": package.unit_id,
        "source_id": package.source_id,
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
    results = {}
    for product in products[:4]:
        if product.get("knowledge_evidence"):
            results[product["sku"]] = [
                {"chunk_id": item["unit_id"]}
                for item in product["knowledge_evidence"]
                if item.get("asset_id")
            ]
    if not results:
        hits, _, _ = retrieve(
            RetrievalQuery(query, "product"),
            RetrievalFilters(
                product_where=f"p.sku IN ({','.join('?' for _ in skus)})",
                product_params=skus,
            ),
        )
        results = {
            sku: [
                {"chunk_id": hit.unit_id}
                for hit in group["hits"]
                if hit.metadata.get("asset_id")
            ]
            for sku, group in aggregate_skus(hits).items()
        }
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
            from .evidence_runtime import mode as evidence_mode
            if evidence_mode()=="unified":
                from .evidence_policies import knowledge_ref
                live=knowledge_ref(package.unit_id)
                observations=(product.get("evidence_assessment") or {}).get("visual_observations",[])
                observation=next((o for o in observations if live and o["evidence_id"]==live.evidence_id and o["snapshot"]==live.snapshot and o["version"]==live.version),None)
                if observation:
                    value=observation["observation"];verified=True
            image_key = (
                package.asset_id,
                package.source_location["page_no"],
                str(package.source_location["bbox"]),
            )
            if (
                is_image
                and evidence_mode()!="unified"
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
