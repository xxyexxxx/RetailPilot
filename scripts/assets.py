"""Manage RetailPilot's curated assets and indexes from the repository root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.asset_config import asset_settings as config
from RetailPilot.asset_embeddings import text_version, visual_version
from RetailPilot.asset_evidence import public_evidence
from RetailPilot.asset_ingestion import (
    ASSET_TYPES,
    delete_asset,
    get_asset,
    process_asset,
    stage_asset,
)
from RetailPilot.asset_qdrant import generation_status, sync_visual
from RetailPilot.asset_retrieval import aggregate_routes, retrieve_product_assets
from RetailPilot.asset_store import download_image, store
from RetailPilot.catalog import get_product
from RetailPilot.database import db_session, init_db
from RetailPilot.policy import TOPIC_DOCUMENT_COMPATIBILITY
from RetailPilot.policy_ingestion import PolicyDocumentInput, ingest_policy_documents


def status(asset: dict | None) -> dict:
    if asset is None:
        raise LookupError("asset_not_found")
    return {
        key: value
        for key, value in asset.items()
        if key not in {"storage_uri", "processing_token"}
    }


def import_asset(
    *,
    path=None,
    url=None,
    owner_type,
    owner_id,
    kind,
    source_key=None,
    source_version="",
) -> dict:
    if bool(path) == bool(url):
        raise ValueError("select exactly one path or URL")
    if path:
        local = Path(path)
        with local.open("rb") as stream:
            data = stream.read(config.max_bytes + 1)
        source_key = source_key or local.name
    else:
        data, _ = download_image(url)
        source_key = source_key or Path(urlsplit(url).path).name
    aid, queued = stage_asset(
        data,
        owner_type=owner_type,
        owner_id=owner_id,
        kind=kind,
        source_key=source_key,
        source_version=source_version,
    )
    return status(process_asset(aid) if queued else get_asset(aid))


def check() -> dict:
    queries = {
        "assets_by_status": "SELECT status,COUNT(*) AS count FROM assets GROUP BY status",
        "missing_fts": """SELECT COUNT(*) FROM asset_chunks ch JOIN assets a USING(asset_id)
                          WHERE ch.search_text!='' AND a.sku IS NOT NULL
                          AND NOT EXISTS(SELECT 1 FROM asset_fts f WHERE f.chunk_id=ch.chunk_id)""",
        "orphan_fts": """SELECT COUNT(*) FROM asset_fts f WHERE NOT EXISTS(
                         SELECT 1 FROM asset_chunks ch WHERE ch.chunk_id=f.chunk_id)""",
        "duplicate_fts": "SELECT COUNT(*)-COUNT(DISTINCT chunk_id) FROM asset_fts",
        "missing_policy_chunks": """SELECT COUNT(*) FROM asset_chunks ch JOIN assets a USING(asset_id)
                          WHERE a.doc_id IS NOT NULL AND NOT EXISTS(
                          SELECT 1 FROM policy_chunks p WHERE p.chunk_id=ch.chunk_id)""",
        "stale_policy_versions": """SELECT COUNT(*) FROM assets a JOIN policy_documents d USING(doc_id)
                                  WHERE a.source_version!=d.version""",
        "bad_vectors": "SELECT COUNT(*) FROM asset_vectors WHERE dim<=0 OR length(vector)!=dim*4",
        "vector_versions": "SELECT space,model_version,dim,COUNT(*) AS count FROM asset_vectors GROUP BY 1,2,3",
    }
    with db_session() as db:
        result = {}
        for key, sql in queries.items():
            rows = db.execute(sql).fetchall()
            result[key] = (
                rows[0][0]
                if len(rows) == 1 and len(rows[0]) == 1
                else [dict(row) for row in rows]
            )
        result["missing_text_vectors"] = db.execute(
            """
            SELECT COUNT(*) FROM asset_chunks ch JOIN assets a USING(asset_id)
            WHERE a.sku IS NOT NULL AND a.status IN ('ready','partial') AND ch.search_text!=''
              AND NOT EXISTS(SELECT 1 FROM asset_vectors v WHERE v.chunk_id=ch.chunk_id
                             AND v.space='text' AND v.model_version=?)
            """,
            (text_version(),),
        ).fetchone()[0]
        result["missing_visual_vectors"] = (
            db.execute(
                """
            SELECT COUNT(*) FROM asset_chunks ch JOIN assets a USING(asset_id)
            JOIN asset_elements e ON e.element_id=ch.element_id
            WHERE a.sku IS NOT NULL AND a.status IN ('ready','partial') AND e.element_type='image'
              AND NOT EXISTS(SELECT 1 FROM asset_vectors v WHERE v.chunk_id=ch.chunk_id
                             AND v.space='visual' AND v.model_version=?)
            """,
                (visual_version(),),
            ).fetchone()[0]
            if config.visual_enabled
            else None
        )
        keys = [
            row[0]
            for row in db.execute("""
            SELECT storage_uri FROM assets UNION SELECT preview_key FROM asset_elements
            WHERE preview_key IS NOT NULL
            """)
        ]
        result["missing_blobs"] = sum(not store().path(key).is_file() for key in keys)
        result["visual_mirror"] = generation_status(db)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init")
    policy = commands.add_parser("register-policy")
    policy.add_argument("metadata", type=Path)
    policy.add_argument("--markdown", type=Path)
    add = commands.add_parser("import")
    sources = add.add_mutually_exclusive_group(required=True)
    for source in ("path", "url", "catalog-main"):
        sources.add_argument("--" + source)
    owners = add.add_mutually_exclusive_group(required=True)
    owners.add_argument("--sku")
    owners.add_argument("--doc-id")
    add.add_argument("--type", required=True, choices=sorted(ASSET_TYPES))
    add.add_argument("--source-key")
    add.add_argument("--source-version", default="")
    batch = commands.add_parser("batch")
    batch.add_argument(
        "manifest", type=Path, help="JSON array of import_asset arguments"
    )
    build = commands.add_parser("build")
    build.add_argument(
        "--all", action="store_true", help="Refresh existing ready assets too"
    )
    build.add_argument("--sync-visual", action="store_true")
    for command in ("status", "delete", "retry"):
        commands.add_parser(command).add_argument("asset_id")
    commands.add_parser("check")
    commands.add_parser("sync-visual")
    search = commands.add_parser("search")
    search.add_argument("query")
    search.add_argument("--sku")
    search.add_argument("--image", type=Path, help="Optional image-to-image query")
    search.add_argument("--no-visual", action="store_true")
    search.add_argument("--no-dense", action="store_true")
    args = parser.parse_args()
    init_db()
    if args.command == "init":
        result = {"initialized": True}
    elif args.command == "register-policy":
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        if metadata.get("topic") not in TOPIC_DOCUMENT_COMPATIBILITY:
            parser.error("invalid canonical policy topic")
        if args.markdown:
            metadata["content"] = args.markdown.read_text(encoding="utf-8")
        else:
            metadata.setdefault("content", "")
        result = ingest_policy_documents([PolicyDocumentInput(**metadata)])
    elif args.command == "import":
        url = args.url
        if args.catalog_main:
            if args.catalog_main != args.sku:
                parser.error("--catalog-main must match --sku")
            product = get_product(args.sku)
            url = product.get("image_url") if product else None
            if not url:
                parser.error("catalog image URL unavailable")
        result = import_asset(
            path=args.path,
            url=url,
            owner_type="product" if args.sku else "policy",
            owner_id=args.sku or args.doc_id,
            kind=args.type,
            source_key=args.source_key,
            source_version=args.source_version,
        )
    elif args.command == "batch":
        items = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            parser.error("manifest must be an array")
        result = []
        for index, item in enumerate(items):
            try:
                result.append(import_asset(**item))
            except (
                KeyError,
                ValueError,
                LookupError,
                OSError,
                TypeError,
                RuntimeError,
            ) as exc:
                result.append(
                    {"row": index + 1, "status": "failed", "error": type(exc).__name__}
                )
    elif args.command == "build":
        with db_session() as db:
            ids = [
                row[0]
                for row in db.execute(
                    """
                SELECT asset_id FROM assets WHERE status!='ready' OR ?
                ORDER BY updated_at,asset_id
                """,
                    (args.all,),
                )
            ]
        result = {"assets": [status(process_asset(aid, retry=True)) for aid in ids]}
        if args.sync_visual:
            result["qdrant"] = sync_visual()
    elif args.command == "status":
        result = status(get_asset(args.asset_id))
    elif args.command == "delete":
        result = {"deleted": delete_asset(args.asset_id)}
    elif args.command == "retry":
        result = status(process_asset(args.asset_id, retry=True))
    elif args.command == "check":
        result = check()
    elif args.command == "sync-visual":
        result = sync_visual()
    else:
        image = None
        if args.image:
            from io import BytesIO

            from PIL import Image

            from RetailPilot.asset_parsing import image_png

            with args.image.open("rb") as stream:
                raw = stream.read(config.max_bytes + 1)
            if len(raw) > config.max_bytes:
                parser.error("image query exceeds byte limit")
            with Image.open(BytesIO(raw)) as decoded:
                if decoded.width * decoded.height > config.max_pixels:
                    parser.error("image query exceeds pixel limit")
                image = image_png(decoded)
        params = {"where": "p.sku=?", "params": [args.sku]} if args.sku else {}
        hits, errors = retrieve_product_assets(
            args.query,
            dense=not args.no_dense,
            visual=not args.no_visual,
            image=image,
            **params,
        )
        routes, evidence = aggregate_routes(hits)
        result = {
            "routes": routes,
            "degraded": errors,
            "evidence": {
                sku: [pkg for hit in group if (pkg := public_evidence(hit))]
                for sku, group in evidence.items()
            },
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    values = result.get("assets", [result]) if isinstance(result, dict) else result
    if any(item.get("status") == "failed" for item in values):
        sys.exit(1)


if __name__ == "__main__":
    main()
