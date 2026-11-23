"""Migrate, build, inspect and query the canonical knowledge index."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RetailPilot.config import settings
from RetailPilot.database import db_session, init_db
from RetailPilot.knowledge import sync_products
from RetailPilot.knowledge_evidence import public_hit
from RetailPilot.knowledge_index import build
from RetailPilot.knowledge_qdrant import status, sync
from RetailPilot.knowledge_retrieval import RetrievalFilters, RetrievalQuery, retrieve


def backup(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or target.exists():
        raise ValueError("backup_target_must_be_new")
    target.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as origin, sqlite3.connect(target) as output:
        origin.backup(output)


def check():
    with db_session() as db:
        sync_products(db)
        result = {
            "units": [
                dict(row)
                for row in db.execute("""SELECT domain,source_format,modality,COUNT(*) AS count
                       FROM knowledge_units JOIN knowledge_sources USING(source_id) GROUP BY 1,2,3""")
            ],
            "foreign_key_errors": [
                tuple(row) for row in db.execute("PRAGMA foreign_key_check")
            ],
            "missing_fts": db.execute("""SELECT COUNT(*) FROM knowledge_units u WHERE NOT EXISTS(
                        SELECT 1 FROM knowledge_fts f WHERE f.unit_id=u.unit_id)""").fetchone()[
                0
            ],
            "orphan_fts": db.execute("""SELECT COUNT(*) FROM knowledge_fts f WHERE NOT EXISTS(
                        SELECT 1 FROM knowledge_units u WHERE u.unit_id=f.unit_id)""").fetchone()[
                0
            ],
            "duplicate_fts": db.execute(
                "SELECT COUNT(*)-COUNT(DISTINCT unit_id) FROM knowledge_fts"
            ).fetchone()[0],
            "stale_policy_sources": db.execute("""SELECT COUNT(*) FROM knowledge_sources s
                JOIN policy_documents d USING(doc_id) WHERE s.version!=d.version""").fetchone()[
                0
            ],
            "stale_vectors": db.execute("""SELECT COUNT(*) FROM knowledge_vectors v JOIN knowledge_units u USING(unit_id)
                           WHERE v.content_hash!=u.content_hash""").fetchone()[0],
            "vectors": [
                dict(row)
                for row in db.execute("""SELECT space,model_version,dim,COUNT(*) AS count
                       FROM knowledge_vectors GROUP BY 1,2,3""")
            ],
            "qdrant": status(db),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    migrate = commands.add_parser("migrate")
    migrate.add_argument("--backup", type=Path, required=True)
    build_command = commands.add_parser("build")
    build_command.add_argument(
        "--domain", choices=("product", "policy", "all"), default="all"
    )
    build_command.add_argument(
        "--space", choices=("text", "visual", "all"), default="text"
    )
    build_command.add_argument("--force", action="store_true")
    commands.add_parser("check")
    commands.add_parser("sync")
    query = commands.add_parser("query")
    query.add_argument("text")
    query.add_argument("--domain", choices=("product", "policy"), default="product")
    query.add_argument("--topic", default="保修")
    query.add_argument("--platform", default=settings.commerce_platform)
    query.add_argument("--limit", type=int, default=10)
    restore = commands.add_parser("restore")
    restore.add_argument("--backup", type=Path, required=True)
    restore.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Create a rollback database at a new path",
    )
    args = parser.parse_args()
    if args.command == "restore":
        backup(args.backup, args.output)
        print(
            json.dumps(
                {
                    "rollback_database": str(args.output.resolve()),
                    "requires_code_revision": "a18faa1",
                }
            )
        )
        return
    if args.command == "migrate":
        backup(settings.database_path, args.backup)
    init_db()
    if args.command == "build":
        if args.force:
            with db_session() as db:
                sync_products(
                    db, [row[0] for row in db.execute("SELECT sku FROM products")]
                )
        domains = ("product", "policy") if args.domain == "all" else (args.domain,)
        spaces = ("text", "visual") if args.space == "all" else (args.space,)
        result = [
            build(domain, space=space, force=args.force)
            for domain in domains
            for space in spaces
        ]
    elif args.command == "sync":
        result = sync()
    elif args.command == "query":
        filters = RetrievalFilters(topics=[args.topic], platform=args.platform)
        hits, routes, trace = retrieve(
            RetrievalQuery(args.text, args.domain), filters, limit=args.limit
        )
        result = {
            "hits": [public_hit(hit) for hit in hits],
            "routes": {
                key: [hit.unit_id for hit in group] for key, group in routes.items()
            },
            "trace": trace,
        }
    else:
        result = check()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
