"""Authenticated asset publication and restricted original/preview access."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from datetime import date
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from . import auth
from .asset_config import asset_settings as config
from .asset_evidence import hydrate
from .asset_ingestion import (
    READY,
    asset_url,
    delete_asset,
    get_asset,
    process_asset,
    stage_asset,
)
from .asset_store import detect_mime, store
from .config import settings
from .database import db_session, json_loads
from .execution import BoundedPool

router = APIRouter(prefix="/api/assets", tags=["assets"])
_workers = BoundedPool("asset-ingest", 1, 8)
_uploads = asyncio.Semaphore(4)


def _authenticated(request: Request) -> auth.Principal:
    if not config.enabled:
        raise HTTPException(404, "assets disabled")
    principal = auth.authenticate(request)
    if principal.mode == "demo":
        raise HTTPException(403, "asset access requires a login or service credential")
    return principal


def _can_publish(principal: auth.Principal) -> bool:
    return principal.user_id in {
        user.strip() for user in config.curators.split(",") if user.strip()
    }


def _publisher(
    principal: Annotated[auth.Principal, Depends(_authenticated)],
) -> auth.Principal:
    if not _can_publish(principal):
        raise HTTPException(403, "asset curator permission required")
    return principal


@router.get("/capabilities")
def capabilities(principal: Annotated[auth.Principal, Depends(_authenticated)]) -> dict:
    return {
        "can_publish": _can_publish(principal),
        "visual": config.visual_enabled,
        "vision_configured": bool(config.vlm_url and config.vlm_model),
    }


def _visible(asset: dict) -> bool:
    if asset["status"] not in READY:
        return False
    with db_session() as db:
        if asset["sku"]:
            return (
                db.execute(
                    """
                SELECT 1 FROM products p LEFT JOIN product_commerce c USING(sku)
                WHERE p.sku=? AND COALESCE(c.dataset_split,'')!='final'
                AND p.source!='Amazon Reviews 2023'
                """,
                    (asset["sku"],),
                ).fetchone()
                is not None
            )
        today = date.today().isoformat()
        return (
            db.execute(
                """
            SELECT 1 FROM policy_documents WHERE doc_id=? AND version=?
            AND status='active' AND audience IN ('public','customer')
            AND platform IN ('*',?) AND effective_from<=?
            AND (effective_to='' OR effective_to>=?)
            """,
                (
                    asset["doc_id"],
                    asset["source_version"],
                    settings.commerce_platform,
                    today,
                    today,
                ),
            ).fetchone()
            is not None
        )


def _public(asset: dict, *, inspect: bool = False) -> dict:
    allowed = (
        "asset_id",
        "sku",
        "doc_id",
        "asset_type",
        "source_version",
        "mime_type",
        "status",
        "error_code",
        "created_at",
        "updated_at",
    )
    value = {key: asset[key] for key in allowed}
    if _visible(asset):
        value["source_uri"] = asset_url(asset["asset_id"], asset["revision"])
    if inspect and asset["status"] in READY:
        with db_session() as db:
            rows = db.execute(
                """
                SELECT element_id,element_type,page_no,bbox_json,section_path,preview_key
                FROM asset_elements WHERE asset_id=? ORDER BY reading_order LIMIT 100
                """,
                (asset["asset_id"],),
            )
            value["elements"] = [
                {
                    "element_id": row["element_id"],
                    "element_type": row["element_type"],
                    "page_no": row["page_no"],
                    "bbox": json_loads(row["bbox_json"], None),
                    "section_path": row["section_path"],
                    "preview_url": (
                        f"/api/assets/{asset['asset_id']}/elements/{row['element_id']}/preview"
                        if row["preview_key"]
                        else None
                    ),
                }
                for row in rows
            ]
    return value


def product_assets(sku: str) -> list[dict]:
    with db_session() as db:
        rows = db.execute(
            """SELECT * FROM assets WHERE sku=? AND status IN ('ready','partial')
               ORDER BY asset_type, asset_id LIMIT 40""",
            (sku,),
        ).fetchall()
    return [_public(dict(row), inspect=True) for row in rows if _visible(dict(row))]


@router.get("")
def list_assets(
    _: Annotated[auth.Principal, Depends(_authenticated)],
    sku: Annotated[str | None, Query(max_length=160)] = None,
    doc_id: Annotated[str | None, Query(max_length=160)] = None,
) -> dict:
    if bool(sku) == bool(doc_id):
        raise HTTPException(422, "specify exactly one sku or doc_id")
    if sku:
        return {"assets": product_assets(sku)}
    with db_session() as db:
        rows = db.execute(
            "SELECT * FROM assets WHERE doc_id=? AND status IN ('ready','partial') ORDER BY asset_id LIMIT 40",
            (doc_id,),
        ).fetchall()
    return {
        "assets": [
            _public(dict(row), inspect=True) for row in rows if _visible(dict(row))
        ]
    }


def _submit(asset_id: str, *, retry: bool = False) -> None:
    deadline = time.perf_counter() + config.processing_timeout * 9
    try:
        _workers.submit(lambda: process_asset(asset_id, retry=retry), deadline=deadline)
    except RuntimeError as exc:
        raise HTTPException(
            429, "index queue full; retry this asset or upload again"
        ) from exc


@router.post("", status_code=202)
async def upload(
    request: Request,
    _: Annotated[auth.Principal, Depends(_publisher)],
    owner_type: str = Query(pattern="^(product|policy)$"),
    owner_id: str = Query(min_length=1, max_length=160),
    asset_type: str = Query(min_length=1, max_length=40),
    source_key: str = Query(min_length=1, max_length=160),
    source_version: str = Query(default="", max_length=80),
) -> dict:
    if _uploads.locked():
        raise HTTPException(429, "upload capacity exhausted")
    async with _uploads:
        try:
            payload = bytearray()
            async with asyncio.timeout(20):
                async for block in request.stream():
                    if len(payload) + len(block) > config.max_bytes:
                        raise HTTPException(413, "file_size_limit")
                    payload.extend(block)
            if request.headers.get("content-type", "").split(";")[
                0
            ].lower() != detect_mime(payload):
                raise ValueError("mime_mismatch")
            aid, scheduled = await run_in_threadpool(
                stage_asset,
                bytes(payload),
                owner_type=owner_type,
                owner_id=owner_id,
                kind=asset_type,
                source_key=source_key,
                source_version=source_version,
            )
        except TimeoutError as exc:
            raise HTTPException(408, "upload_timeout") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
    if scheduled:
        _submit(aid)
    return {
        "asset_id": aid,
        "status": "queued" if scheduled else "unchanged",
        "status_url": f"/api/assets/{aid}",
    }


def _asset_or_404(asset_id: str) -> dict:
    asset = get_asset(asset_id) if len(asset_id) == 32 else None
    if not asset:
        raise HTTPException(404, "asset not found")
    return asset


@router.get("/{asset_id}")
def status(
    asset_id: str, principal: Annotated[auth.Principal, Depends(_authenticated)]
) -> dict:
    asset = _asset_or_404(asset_id)
    if not _can_publish(principal) and not _visible(asset):
        raise HTTPException(404, "asset not found")
    return _public(asset, inspect=True)


@router.post("/{asset_id}/retry", status_code=202)
def retry_asset(
    asset_id: str, _: Annotated[auth.Principal, Depends(_publisher)]
) -> dict:
    _asset_or_404(asset_id)
    _submit(asset_id, retry=True)
    return {"asset_id": asset_id, "status": "queued"}


@router.delete("/{asset_id}", status_code=204)
def remove_asset(
    asset_id: str, _: Annotated[auth.Principal, Depends(_publisher)]
) -> None:
    if not delete_asset(asset_id):
        raise HTTPException(404, "asset not found")


def _download(key: str, mime: str, *, attachment: bool = False) -> FileResponse:
    path = store().path(key)
    if not path.is_file():
        raise HTTPException(404, "asset content unavailable")
    suffix = {
        "application/pdf": "pdf",
        "image/png": "png",
        "image/webp": "webp",
        "image/jpeg": "jpg",
    }[mime]
    return FileResponse(
        path,
        media_type=mime,
        filename=f"asset.{suffix}" if attachment else None,
        headers={
            "X-Content-Type-Options": "nosniff",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        },
    )


@router.get("/{asset_id}/evidence/{chunk_id}")
def evidence(
    asset_id: str, chunk_id: str, _: Annotated[auth.Principal, Depends(_authenticated)]
) -> dict:
    asset = _asset_or_404(asset_id)
    package = hydrate(chunk_id) if _visible(asset) and len(chunk_id) == 32 else None
    if not package or package.asset_id != asset_id:
        raise HTTPException(404, "asset evidence unavailable")
    return asdict(package)


@router.get("/{asset_id}/content")
def original(
    asset_id: str,
    _: Annotated[auth.Principal, Depends(_authenticated)],
    revision: Annotated[str, Query(pattern=r"^[a-f0-9]{32}$")] | None = None,
) -> FileResponse:
    asset = _asset_or_404(asset_id)
    if not _visible(asset) or (revision and revision != asset["revision"]):
        raise HTTPException(404, "asset not found")
    return _download(
        asset["storage_uri"],
        asset["mime_type"],
        attachment=asset["mime_type"] == "application/pdf",
    )


@router.get("/{asset_id}/elements/{element_id}/preview")
def preview(
    asset_id: str,
    element_id: str,
    _: Annotated[auth.Principal, Depends(_authenticated)],
) -> FileResponse:
    asset = _asset_or_404(asset_id)
    if not _visible(asset):
        raise HTTPException(404, "asset not found")
    with db_session() as db:
        row = db.execute(
            """SELECT preview_key FROM asset_elements WHERE asset_id=? AND element_id=?""",
            (asset_id, element_id),
        ).fetchone()
    if not row or not row["preview_key"]:
        raise HTTPException(404, "asset preview unavailable")
    return _download(row["preview_key"], "image/png")
