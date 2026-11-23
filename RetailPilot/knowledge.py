"""KnowledgeUnit builder, repository and one-way legacy migration."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

from .database import json_dumps, json_loads
from .text import tokenize


@dataclass(frozen=True)
class KnowledgeUnit:
    unit_id: str
    domain: Literal["product", "policy"]
    owner_id: str
    source_id: str
    source_format: str
    modality: str
    search_text: str
    evidence_text: str
    version: str
    structured_data: dict | None = None
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def digest(*values) -> str:
    return hashlib.sha256(json_dumps(values).encode()).hexdigest()[:32]


def terms(text: str) -> str:
    return " ".join(dict.fromkeys(tokenize(text, include_expansion=False)))


def source(
    db,
    *,
    source_id,
    domain,
    owner_id,
    source_format,
    source_uri,
    version,
    content_hash,
    asset_id=None,
):
    db.execute(
        """INSERT INTO knowledge_sources VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(source_id) DO UPDATE SET version=excluded.version,
          content_hash=excluded.content_hash,source_uri=excluded.source_uri,
          source_format=excluded.source_format""",
        (
            source_id,
            domain,
            owner_id if domain == "product" else None,
            owner_id if domain == "policy" else None,
            asset_id,
            source_format,
            source_uri,
            version,
            content_hash,
        ),
    )


def put_unit(
    db,
    *,
    unit_id,
    source_id,
    modality,
    evidence_type,
    search_text,
    evidence_text,
    structured=None,
    field_refs=None,
    element_id=None,
    section_path="",
    page_no=None,
    char_start=0,
    char_end=0,
    ordinal=0,
    context_id=None,
    parser_version="knowledge-v1",
    metadata=None,
    element_ids=(),
    fields=None,
):
    content_hash = digest(search_text, evidence_text, structured, parser_version)
    previous = db.execute(
        "SELECT content_hash FROM knowledge_units WHERE unit_id=?", (unit_id,)
    ).fetchone()
    if previous and previous[0] == content_hash:
        return False
    db.execute("DELETE FROM knowledge_units WHERE unit_id=?", (unit_id,))
    db.execute(
        """INSERT INTO knowledge_units VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            unit_id,
            source_id,
            element_id,
            modality,
            evidence_type,
            search_text,
            evidence_text,
            json_dumps(structured),
            json_dumps(field_refs or []),
            section_path,
            page_no,
            char_start,
            char_end,
            ordinal,
            context_id,
            content_hash,
            parser_version,
            "knowledge-v1",
            json_dumps(metadata or {}),
        ),
    )
    db.executemany(
        "INSERT INTO knowledge_unit_elements VALUES(?,?)",
        [
            (unit_id, eid)
            for eid in dict.fromkeys(
                [*([element_id] if element_id else []), *element_ids]
            )
        ],
    )
    identity, title, brand, category = fields or ("", "", "", "")
    db.execute(
        "INSERT INTO knowledge_fts VALUES(?,?,?,?,?,?)",
        (
            unit_id,
            terms(identity),
            terms(title),
            terms(brand),
            terms(category),
            terms(search_text),
        ),
    )
    # Hash is an explicitly configured offline index. Neural models are built in batches by CLI.
    from .knowledge_index import encode_text, model_spec, version

    domain = db.execute(
        "SELECT domain FROM knowledge_sources WHERE source_id=?", (source_id,)
    ).fetchone()[0]
    if model_spec(domain)[0] == "hash" and search_text.strip():
        from .text import vector_to_blob

        vector = encode_text([search_text], domain)[0]
        db.execute(
            "INSERT INTO knowledge_vectors VALUES(?,?,?,?,?,?)",
            (
                unit_id,
                "text",
                version(domain),
                content_hash,
                len(vector),
                vector_to_blob(vector),
            ),
        )
    return True


def sync_products(db, skus=None):
    if skus is None:
        skus = [row[0] for row in db.execute("SELECT sku FROM knowledge_dirty")]
    for sku in skus:
        row = db.execute("SELECT * FROM products WHERE sku=?", (sku,)).fetchone()
        if row is None:
            continue
        p = dict(row)
        values = {
            key: p[key]
            for key in (
                "sku",
                "title",
                "brand",
                "category",
                "category_label",
                "description",
            )
        }
        values.update(
            features=json_loads(p["features_json"], []),
            attributes=json_loads(p["attributes_json"], {}),
            tags=json_loads(p["tags_json"], []),
        )
        sid = f"catalog:{sku}"
        content = "\n".join(
            [
                *[str(values[key]) for key in values],
                p["tags_text"],
                p["attributes_text"],
            ]
        )
        revision = digest(content, p["tags_text"], p["attributes_text"])
        source(
            db,
            source_id=sid,
            domain="product",
            owner_id=sku,
            source_format="catalog",
            source_uri=f"catalog:{sku}",
            version=revision,
            content_hash=revision,
        )
        # Bounded text windows retain field references; SKU voting is capped after retrieval.
        from .asset_ingestion import _text_windows

        windows = _text_windows(content, 560) or [p["title"]]
        used = []
        for index, text in enumerate(windows):
            uid = f"profile:{sku}:{index}"
            used.append(uid)
            put_unit(
                db,
                unit_id=uid,
                source_id=sid,
                modality="text",
                evidence_type="catalog_verified",
                search_text=" ".join([p["title"], p["brand"], text]),
                evidence_text=text,
                structured=None,
                field_refs=[f"products.{key}" for key in values],
                ordinal=index,
                parser_version="catalog-profile-v1",
                fields=(
                    sku,
                    p["title"],
                    p["brand"],
                    p["category"] + " " + p["category_label"],
                ),
            )
        db.execute(
            f"DELETE FROM knowledge_units WHERE source_id=? AND unit_id NOT IN ({','.join('?' for _ in used)})",
            [sid, *used],
        )
        db.execute("DELETE FROM knowledge_dirty WHERE sku=?", (sku,))


def write_markdown(db, document, chunks):
    sid = f"markdown:{document.doc_id}"
    old = db.execute(
        "SELECT version,content_hash FROM knowledge_sources WHERE source_id=?", (sid,)
    ).fetchone()
    if (
        old
        and old["version"] == document.version
        and old["content_hash"] == digest(document.content)
    ):
        return
    source(
        db,
        source_id=sid,
        domain="policy",
        owner_id=document.doc_id,
        source_format="markdown",
        source_uri=document.source_uri or sid,
        version=document.version,
        content_hash=digest(document.content),
    )
    db.execute("DELETE FROM knowledge_units WHERE source_id=?", (sid,))
    db.execute("DELETE FROM knowledge_contexts WHERE source_id=?", (sid,))
    for item in chunks:
        if item["chunk_type"] == "parent":
            db.execute(
                "INSERT INTO knowledge_contexts VALUES(?,?,?,?)",
                (item["chunk_id"], sid, item["content"], item["section_path"]),
            )
    from .asset_ingestion import build_chunks
    from .asset_parsing import DocumentElement, ParsedDocument

    # Section windows preserve established Markdown IDs and source offsets.
    parsed = ParsedDocument(
        [
            DocumentElement(
                "text",
                item["ordinal"],
                item["content"],
                item["section_path"],
                metadata={"window": item},
                search_text=item["contextualized_content"],
            )
            for item in chunks
            if item["chunk_type"] != "parent"
        ],
        "markdown-sections-v1",
        doc_id=document.doc_id,
    )
    for element in parsed.elements:
        item = element.metadata["window"]
        built = build_chunks(item["chunk_id"], element)
        if len(built) != 1:
            raise ValueError("Markdown section window exceeds shared builder limit")
        put_unit(
            db,
            unit_id=item["chunk_id"],
            source_id=sid,
            modality="text",
            evidence_type="document_text",
            evidence_text=built[0].content,
            search_text=element.search_text,
            section_path=item["section_path"],
            char_start=item["char_start"],
            char_end=item["char_end"],
            page_no=item["page_no"],
            ordinal=item["ordinal"],
            context_id=item["parent_chunk_id"],
            fields=(document.doc_id, document.title, "", document.topic),
        )


def migrate(db):
    """Preserve external chunk IDs while retiring duplicate physical tables."""
    done = db.execute(
        "SELECT value FROM metadata WHERE key='knowledge_schema'"
    ).fetchone()
    if done:
        return
    tables = {
        row[0]
        for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    db.execute("INSERT OR IGNORE INTO knowledge_dirty SELECT sku FROM products")
    sync_products(db)
    asset_ids = set()
    if "asset_chunks" in tables:
        for row in db.execute("""SELECT ch.*,a.sku,a.doc_id,a.mime_type,a.revision,a.source_version,
                                      e.element_type,e.section_path,e.page_no,e.parser_version
                                FROM asset_chunks ch JOIN assets a USING(asset_id)
                                JOIN asset_elements e USING(element_id)""").fetchall():
            item = dict(row)
            asset_ids.add(item["chunk_id"])
            sid = f"asset:{item['asset_id']}"
            source(
                db,
                source_id=sid,
                domain="product" if item["sku"] else "policy",
                owner_id=item["sku"] or item["doc_id"],
                asset_id=item["asset_id"],
                source_format="pdf"
                if item["mime_type"] == "application/pdf"
                else "image",
                source_uri=f"asset:{item['asset_id']}",
                version=item["source_version"] or item["revision"],
                content_hash=item["revision"],
            )
            refs = [
                r[0]
                for r in db.execute(
                    "SELECT element_id FROM asset_chunk_elements WHERE chunk_id=?",
                    (item["chunk_id"],),
                )
            ]
            put_unit(
                db,
                unit_id=item["chunk_id"],
                source_id=sid,
                element_id=item["element_id"],
                modality=item["element_type"]
                if item["element_type"] in {"image", "table"}
                else "text",
                evidence_type=item["evidence_type"],
                evidence_text=item["content"],
                search_text=item["search_text"],
                structured=json_loads(item["structured_json"], None),
                section_path=item["section_path"],
                page_no=item["page_no"],
                parser_version=item["parser_version"],
                element_ids=refs,
            )
        # Reuse only visual vectors from the exact configured visual model. Text vectors are rebuilt.
        from .asset_embeddings import visual_version

        for row in db.execute(
            "SELECT * FROM asset_vectors WHERE space='visual' AND model_version=?",
            (visual_version(),),
        ).fetchall():
            unit = db.execute(
                "SELECT content_hash FROM knowledge_units WHERE unit_id=?",
                (row["chunk_id"],),
            ).fetchone()
            if unit:
                db.execute(
                    "INSERT OR REPLACE INTO knowledge_vectors VALUES(?,?,?,?,?,?)",
                    (
                        row["chunk_id"],
                        "visual",
                        row["model_version"],
                        unit[0],
                        row["dim"],
                        row["vector"],
                    ),
                )
    if "policy_chunks" in tables:
        from .policy_ingestion import PolicyDocumentInput

        for row in db.execute("SELECT * FROM policy_documents").fetchall():
            p = dict(row)
            document = PolicyDocumentInput(
                doc_id=p["doc_id"],
                policy_id=p["policy_id"],
                topic=p["topic"],
                title=p["title"],
                content=p["content"],
                source_name=p["source_name"],
                source_uri=p["source_uri"],
                version=p["version"],
            )
            chunks = [
                dict(r)
                for r in db.execute(
                    "SELECT * FROM policy_chunks WHERE doc_id=? ORDER BY ordinal",
                    (p["doc_id"],),
                )
                if r["chunk_id"] not in asset_ids
            ]
            if chunks:
                write_markdown(db, document, chunks)
    db.execute("DROP TRIGGER IF EXISTS asset_chunks_delete")
    for name in ("visual_insert", "visual_delete", "visual_assets_queued"):
        db.execute(f"DROP TRIGGER IF EXISTS {name}")
    for table in (
        "asset_vectors",
        "asset_chunk_elements",
        "asset_chunks",
        "asset_fts",
        "asset_vector_state",
        "policy_chunk_fts",
        "policy_chunks",
        "product_vectors",
        "product_fts",
    ):
        if table in tables:
            db.execute(f"DROP TABLE {table}")
    db.execute("INSERT INTO metadata(key,value) VALUES('knowledge_schema','1')")


def unit_rows(db, ids):
    if not ids:
        return []
    return db.execute(
        f"""SELECT u.*,s.domain,s.sku,s.doc_id,s.asset_id,s.source_format,s.version,
        s.source_uri,s.content_hash AS source_hash FROM knowledge_units u
        JOIN knowledge_sources s USING(source_id) WHERE unit_id IN ({",".join("?" for _ in ids)})""",
        ids,
    ).fetchall()


def get_unit(db, unit_id):
    rows = unit_rows(db, [unit_id])
    if not rows:
        return None
    row = dict(rows[0])
    refs = [
        r[0]
        for r in db.execute(
            "SELECT element_id FROM knowledge_unit_elements WHERE unit_id=?", (unit_id,)
        )
    ]
    return KnowledgeUnit(
        unit_id=row["unit_id"],
        domain=row["domain"],
        owner_id=row["sku"] or row["doc_id"],
        source_id=row["source_id"],
        source_format=row["source_format"],
        modality=row["modality"],
        search_text=row["search_text"],
        evidence_text=row["evidence_text"],
        version=row["version"],
        structured_data=json_loads(row["structured_json"], None),
        evidence_refs=refs or json_loads(row["field_refs_json"], []),
        metadata={
            "parser_version": row["parser_version"],
            "section_path": row["section_path"],
            "page_no": row["page_no"],
        },
    )
