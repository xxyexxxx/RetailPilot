"""Additive SQLite migration for shared, curated retrieval assets."""

ASSET_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    sku TEXT REFERENCES products(sku) ON DELETE CASCADE,
    doc_id TEXT REFERENCES policy_documents(doc_id) ON DELETE CASCADE,
    asset_type TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    storage_uri TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('queued','processing','ready','partial','failed')),
    revision TEXT NOT NULL,
    source_version TEXT NOT NULL DEFAULT '',
    error_code TEXT NOT NULL DEFAULT '',
    processing_token TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK ((sku IS NULL) != (doc_id IS NULL))
);
CREATE INDEX IF NOT EXISTS assets_sku ON assets(sku, status);
CREATE INDEX IF NOT EXISTS assets_doc ON assets(doc_id, status);
CREATE TABLE IF NOT EXISTS asset_parse_cache (
    cache_key TEXT PRIMARY KEY,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS asset_elements (
    element_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    element_type TEXT NOT NULL,
    page_no INTEGER,
    bbox_json TEXT,
    section_path TEXT NOT NULL,
    reading_order INTEGER NOT NULL,
    raw_content TEXT NOT NULL,
    search_text TEXT NOT NULL,
    preview_key TEXT,
    metadata_json TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    enrichment_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS asset_elements_asset ON asset_elements(asset_id, reading_order);

"""
