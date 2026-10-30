from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from .config import ensure_directories, settings

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    brand TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL,
    category_label TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    price REAL NOT NULL CHECK(price >= 0),
    rating REAL NOT NULL DEFAULT 0,
    rating_count INTEGER NOT NULL DEFAULT 0,
    stock INTEGER NOT NULL DEFAULT 0,
    features_json TEXT NOT NULL DEFAULT '[]',
    attributes_json TEXT NOT NULL DEFAULT '{}',
    tags_json TEXT NOT NULL DEFAULT '[]',
    tags_text TEXT NOT NULL DEFAULT '',
    attributes_text TEXT NOT NULL DEFAULT '',
    image_url TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    source_url TEXT NOT NULL DEFAULT '',
    quality_score REAL NOT NULL DEFAULT 0,
    popularity_score REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_products_category_price
ON products(category, price);
CREATE INDEX IF NOT EXISTS idx_products_brand
ON products(brand);
CREATE INDEX IF NOT EXISTS idx_products_quality
ON products(quality_score DESC, popularity_score DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS product_fts USING fts5(
    sku UNINDEXED,
    title,
    brand,
    category,
    tags,
    description,
    attributes,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS product_vectors (
    sku TEXT PRIMARY KEY REFERENCES products(sku) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    profile_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    preference_type TEXT NOT NULL,
    preference_key TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0,
    evidence TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(user_id, preference_type, preference_key)
);

CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    sku TEXT NOT NULL REFERENCES products(sku) ON DELETE CASCADE,
    event_type TEXT NOT NULL,
    weight REAL NOT NULL,
    query TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_interactions_user_time
ON interactions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_interactions_sku
ON interactions(sku, event_type);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    memory_type TEXT NOT NULL,
    memory_key TEXT NOT NULL,
    memory_value TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1,
    evidence TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, memory_type, memory_key)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    context_json TEXT NOT NULL DEFAULT '{}',
    pending_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session
ON messages(session_id, id DESC);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    sku TEXT NOT NULL REFERENCES products(sku),
    amount REAL NOT NULL,
    status TEXT NOT NULL,
    tracking_no TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS logistics_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    event_time TEXT NOT NULL,
    status TEXT NOT NULL,
    location TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faq_documents (
    doc_id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    platform TEXT NOT NULL DEFAULT 'demo-mall',
    category TEXT NOT NULL DEFAULT '*',
    effective_from TEXT NOT NULL DEFAULT '2026-01-01',
    effective_to TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '1.0',
    priority INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS policy_documents (
    doc_id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_uri TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL DEFAULT 'demo-mall',
    jurisdiction TEXT NOT NULL DEFAULT '*',
    language TEXT NOT NULL DEFAULT 'zh-CN',
    category_scope TEXT NOT NULL DEFAULT '*',
    seller_scope TEXT NOT NULL DEFAULT '*',
    product_type_scope TEXT NOT NULL DEFAULT '*',
    audience TEXT NOT NULL DEFAULT 'customer',
    effective_from TEXT NOT NULL DEFAULT '0001-01-01',
    effective_to TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '1.0',
    status TEXT NOT NULL DEFAULT 'active',
    supersedes_doc_id TEXT NOT NULL DEFAULT '',
    authority TEXT NOT NULL DEFAULT 'demo',
    priority INTEGER NOT NULL DEFAULT 0,
    checksum TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_policy_documents_filter
ON policy_documents(
    status, audience, platform, topic, category_scope,
    effective_from, effective_to, priority DESC
);

CREATE TABLE IF NOT EXISTS policy_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES policy_documents(doc_id) ON DELETE CASCADE,
    parent_chunk_id TEXT,
    chunk_type TEXT NOT NULL CHECK(chunk_type IN ('leaf', 'child', 'parent')),
    ordinal INTEGER NOT NULL,
    section_path TEXT NOT NULL DEFAULT '',
    page_no INTEGER,
    char_start INTEGER NOT NULL DEFAULT 0,
    char_end INTEGER NOT NULL DEFAULT 0,
    content TEXT NOT NULL,
    contextualized_content TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    embedding_model TEXT NOT NULL DEFAULT '',
    embedding_version TEXT NOT NULL DEFAULT '',
    index_version TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(doc_id, ordinal, chunk_type, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_policy_chunks_document
ON policy_chunks(doc_id, chunk_type, ordinal);
CREATE INDEX IF NOT EXISTS idx_policy_chunks_parent
ON policy_chunks(parent_chunk_id, ordinal);

-- FTS stores pre-tokenized Chinese/English terms. Parent chunks are context
-- expansion records and are deliberately not inserted into this recall index.
CREATE VIRTUAL TABLE IF NOT EXISTS policy_chunk_fts USING fts5(
    chunk_id UNINDEXED,
    search_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS policy_ingestion_runs (
    run_id TEXT PRIMARY KEY,
    source_name TEXT NOT NULL,
    source_version TEXT NOT NULL DEFAULT '',
    document_count INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS search_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    parsed_json TEXT NOT NULL,
    result_skus_json TEXT NOT NULL,
    trace_json TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_search_logs_created
ON search_logs(created_at DESC);

CREATE TABLE IF NOT EXISTS eval_cases (
    case_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    query TEXT NOT NULL,
    context_json TEXT NOT NULL DEFAULT '{}',
    expected_json TEXT NOT NULL,
    tags_json TEXT NOT NULL DEFAULT '[]',
    enabled INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS eval_qrels (
    case_id TEXT NOT NULL REFERENCES eval_cases(case_id) ON DELETE CASCADE,
    sku TEXT NOT NULL REFERENCES products(sku) ON DELETE CASCADE,
    relevance INTEGER NOT NULL CHECK(relevance BETWEEN 0 AND 3),
    source TEXT NOT NULL DEFAULT 'heuristic',
    note TEXT NOT NULL DEFAULT '',
    PRIMARY KEY(case_id, sku)
);
CREATE INDEX IF NOT EXISTS idx_eval_qrels_case_relevance
ON eval_qrels(case_id, relevance DESC);

CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    report_path TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS search_route_logs (
    request_id TEXT NOT NULL,
    route TEXT NOT NULL,
    rank INTEGER NOT NULL,
    sku TEXT NOT NULL,
    score REAL,
    model_version TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(request_id, route, rank)
);
CREATE INDEX IF NOT EXISTS idx_search_route_logs_request
ON search_route_logs(request_id, route);

CREATE TABLE IF NOT EXISTS search_result_impressions (
    request_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    query TEXT NOT NULL,
    sku TEXT NOT NULL,
    rank INTEGER NOT NULL,
    score REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(request_id, rank)
);
CREATE INDEX IF NOT EXISTS idx_search_impressions_sku
ON search_result_impressions(sku, created_at DESC);

CREATE TABLE IF NOT EXISTS evaluation_samples (
    sample_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    sampling_reasons_json TEXT NOT NULL DEFAULT '[]',
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    judge_json TEXT NOT NULL DEFAULT '{}',
    cluster_label TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_evaluation_samples_status
ON evaluation_samples(status, created_at);

CREATE TABLE IF NOT EXISTS llm_judge_cache (
    cache_key TEXT PRIMARY KEY,
    rubric_version TEXT NOT NULL,
    verdict_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_usage_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_kind TEXT NOT NULL,
    prompt_version TEXT NOT NULL DEFAULT 'v1',
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cached_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_miss_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms REAL NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'success',
    error_type TEXT NOT NULL DEFAULT '',
    estimated_cost REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_llm_usage_logs_kind_created
ON llm_usage_logs(prompt_kind, created_at DESC);

-- Optional commerce fields live in a side table so older Amazon databases can
-- be upgraded in-place without rebuilding the catalog.
CREATE TABLE IF NOT EXISTS product_commerce (
    sku TEXT PRIMARY KEY REFERENCES products(sku) ON DELETE CASCADE,
    platform TEXT NOT NULL DEFAULT '',
    platform_product_id TEXT NOT NULL DEFAULT '',
    spu_id TEXT NOT NULL DEFAULT '',
    seller_name TEXT NOT NULL DEFAULT '',
    currency TEXT NOT NULL DEFAULT 'USD',
    category_path TEXT NOT NULL DEFAULT '',
    variants_json TEXT NOT NULL DEFAULT '{}',
    services_json TEXT NOT NULL DEFAULT '[]',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    dataset_split TEXT NOT NULL DEFAULT '',
    license_note TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_product_commerce_platform_product
ON product_commerce(platform, platform_product_id);
CREATE INDEX IF NOT EXISTS idx_product_commerce_spu
ON product_commerce(spu_id);

CREATE TABLE IF NOT EXISTS catalog_imports (
    import_id TEXT PRIMARY KEY,
    adapter TEXT NOT NULL,
    source_path TEXT NOT NULL DEFAULT '',
    dataset_version TEXT NOT NULL DEFAULT '',
    product_count INTEGER NOT NULL DEFAULT 0,
    task_count INTEGER NOT NULL DEFAULT 0,
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

"""
# 以下表已随 improvement 模块移除而废弃，schema 不再创建：
# badcase_clusters / badcase_members / regression_cases（见 git 历史，improvement/ 已删除）


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    import time

    from .execution import call_deadline

    ensure_directories()
    deadline = call_deadline.get()
    remaining = (
        5.0
        if deadline is None
        else max(0.001, min(5.0, deadline - time.perf_counter()))
    )
    connection = sqlite3.connect(
        str(path or settings.database_path), timeout=remaining, check_same_thread=False
    )
    connection.row_factory = sqlite3.Row
    connection.execute(f"PRAGMA busy_timeout = {max(1, int(remaining * 1000))}")
    if deadline is not None:
        connection.set_progress_handler(
            lambda: int(time.perf_counter() >= deadline), 1000
        )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


@contextmanager
def db_session(path: Path | str | None = None) -> Iterator[sqlite3.Connection]:
    connection = connect(path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(path: Path | str | None = None) -> None:
    with db_session(path) as connection:
        connection.executescript(SCHEMA)
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(sessions)")
        }
        if "version" not in columns:
            connection.execute(
                "ALTER TABLE sessions ADD COLUMN version INTEGER NOT NULL DEFAULT 0"
            )
        connection.execute("""CREATE TABLE IF NOT EXISTS web_sessions (
            token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL, csrf_token TEXT NOT NULL, expires_at REAL NOT NULL)""")
        _migrate_faq_documents(connection)
        _migrate_legacy_policy_documents(connection)
        refresh_policy_fts(connection)


def _migrate_faq_documents(connection: sqlite3.Connection) -> None:
    """Add policy metadata to databases created before customer RAG v1."""

    columns = {
        row["name"] for row in connection.execute("PRAGMA table_info(faq_documents)")
    }
    additions = {
        "platform": "TEXT NOT NULL DEFAULT 'demo-mall'",
        "category": "TEXT NOT NULL DEFAULT '*'",
        "effective_from": "TEXT NOT NULL DEFAULT '2026-01-01'",
        "effective_to": "TEXT NOT NULL DEFAULT ''",
        "version": "TEXT NOT NULL DEFAULT '1.0'",
        "priority": "INTEGER NOT NULL DEFAULT 0",
        "status": "TEXT NOT NULL DEFAULT 'active'",
        "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
    }
    for name, declaration in additions.items():
        if name not in columns:
            connection.execute(
                f"ALTER TABLE faq_documents ADD COLUMN {name} {declaration}"
            )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_faq_documents_filter
        ON faq_documents(status, platform, topic, category, priority DESC)
        """
    )


def _migrate_legacy_policy_documents(connection: sqlite3.Connection) -> None:
    """Expose the original short FAQs through the chunk-level policy schema."""

    connection.execute(
        """
        INSERT INTO policy_documents(
            doc_id, policy_id, topic, title, content, source_name, source_uri,
            platform, jurisdiction, language, category_scope, audience,
            effective_from, effective_to, version, status, authority, priority,
            checksum, metadata_json, updated_at
        )
        SELECT
            doc_id, doc_id, topic, title, content, source, '', platform, '*',
            'zh-CN', category, 'customer', effective_from, effective_to,
            version, status, 'demo', priority, 'legacy:' || doc_id,
            metadata_json, updated_at
        FROM faq_documents
        WHERE 1
        ON CONFLICT(doc_id) DO UPDATE SET
            topic=excluded.topic,
            title=excluded.title,
            content=excluded.content,
            source_name=excluded.source_name,
            platform=excluded.platform,
            category_scope=excluded.category_scope,
            effective_from=excluded.effective_from,
            effective_to=excluded.effective_to,
            version=excluded.version,
            status=excluded.status,
            priority=excluded.priority,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        """
    )
    connection.execute(
        """
        INSERT INTO policy_chunks(
            chunk_id, doc_id, parent_chunk_id, chunk_type, ordinal,
            section_path, char_start, char_end, content,
            contextualized_content, token_count, content_hash, metadata_json
        )
        SELECT
            'legacy:' || doc_id, doc_id, NULL, 'leaf', 0, title, 0,
            LENGTH(content), content,
            topic || ' | ' || title || ' | ' || content,
            LENGTH(content), 'legacy:' || doc_id, '{}'
        FROM faq_documents
        WHERE 1
        ON CONFLICT(chunk_id) DO UPDATE SET
            section_path=excluded.section_path,
            char_end=excluded.char_end,
            content=excluded.content,
            contextualized_content=excluded.contextualized_content,
            token_count=excluded.token_count
        """
    )


def refresh_policy_fts(
    connection: sqlite3.Connection, doc_ids: list[str] | None = None
) -> None:
    """Synchronize pre-tokenized searchable chunks with the FTS5 table."""

    from .text import tokenize

    if doc_ids:
        placeholders = ",".join("?" for _ in doc_ids)
        chunk_rows = connection.execute(
            f"""
            SELECT chunk_id, contextualized_content
            FROM policy_chunks
            WHERE chunk_type != 'parent' AND doc_id IN ({placeholders})
            ORDER BY doc_id, ordinal
            """,
            doc_ids,
        ).fetchall()
        chunk_ids = [str(row["chunk_id"]) for row in chunk_rows]
        if chunk_ids:
            id_placeholders = ",".join("?" for _ in chunk_ids)
            connection.execute(
                f"DELETE FROM policy_chunk_fts WHERE chunk_id IN ({id_placeholders})",
                chunk_ids,
            )
    else:
        connection.execute("DELETE FROM policy_chunk_fts")
        chunk_rows = connection.execute(
            """
            SELECT chunk_id, contextualized_content
            FROM policy_chunks
            WHERE chunk_type != 'parent'
            ORDER BY doc_id, ordinal
            """
        ).fetchall()
    connection.executemany(
        "INSERT INTO policy_chunk_fts(chunk_id, search_text) VALUES(?, ?)",
        [
            (
                row["chunk_id"],
                " ".join(
                    dict.fromkeys(
                        tokenize(
                            str(row["contextualized_content"]),
                            include_expansion=False,
                        )
                    )
                ),
            )
            for row in chunk_rows
        ],
    )


def record_llm_usage(payload: dict[str, Any]) -> None:
    """Persist one runtime LLM call without running schema initialization."""

    with db_session() as connection:
        connection.execute(
            """
            INSERT INTO llm_usage_logs(
                prompt_kind, prompt_version, provider, model,
                input_tokens, output_tokens, total_tokens, cached_tokens,
                cache_creation_tokens, cache_miss_tokens, latency_ms,
                retry_count, status, error_type, estimated_cost
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload.get("prompt_kind") or "unknown"),
                str(payload.get("prompt_version") or "v1"),
                str(payload.get("provider") or ""),
                str(payload.get("model") or ""),
                int(payload.get("input_tokens") or 0),
                int(payload.get("output_tokens") or 0),
                int(payload.get("total_tokens") or 0),
                int(payload.get("cached_tokens") or 0),
                int(payload.get("cache_creation_tokens") or 0),
                int(payload.get("cache_miss_tokens") or 0),
                float(payload.get("latency_ms") or 0),
                int(payload.get("retry_count") or 0),
                str(payload.get("status") or "success"),
                str(payload.get("error_type") or ""),
                (
                    float(payload["estimated_cost"])
                    if payload.get("estimated_cost") is not None
                    else None
                ),
            ),
        )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def json_loads(value: str | bytes | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def set_metadata(key: str, value: Any, path: Path | str | None = None) -> None:
    with db_session(path) as connection:
        connection.execute(
            """
            INSERT INTO metadata(key, value, updated_at)
            VALUES(?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, json_dumps(value) if not isinstance(value, str) else value, utc_now()),
        )


def get_metadata(key: str, default: Any = None, path: Path | str | None = None) -> Any:
    with db_session(path) as connection:
        row = connection.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    value = row["value"]
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def catalog_ready(path: Path | str | None = None) -> bool:
    init_db(path)
    with db_session(path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    return count > 0
