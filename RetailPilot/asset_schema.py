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
CREATE TABLE IF NOT EXISTS asset_chunks (
    chunk_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
    element_id TEXT NOT NULL REFERENCES asset_elements(element_id) ON DELETE CASCADE,
    evidence_type TEXT NOT NULL,
    content TEXT NOT NULL,
    search_text TEXT NOT NULL,
    structured_json TEXT,
    index_version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS asset_chunks_asset ON asset_chunks(asset_id);
CREATE TABLE IF NOT EXISTS asset_chunk_elements (
    chunk_id TEXT NOT NULL REFERENCES asset_chunks(chunk_id) ON DELETE CASCADE,
    element_id TEXT NOT NULL REFERENCES asset_elements(element_id) ON DELETE CASCADE,
    PRIMARY KEY(chunk_id, element_id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS asset_fts USING fts5(
    chunk_id UNINDEXED, search_text, tokenize='unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS asset_vectors (
    chunk_id TEXT NOT NULL REFERENCES asset_chunks(chunk_id) ON DELETE CASCADE,
    space TEXT NOT NULL CHECK(space IN ('text','visual')),
    model_version TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY(chunk_id, space)
);
CREATE INDEX IF NOT EXISTS asset_vectors_space ON asset_vectors(space, model_version);
CREATE TABLE IF NOT EXISTS asset_vector_state (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    generation INTEGER NOT NULL DEFAULT 0,
    target TEXT NOT NULL DEFAULT '',
    synced_generation INTEGER NOT NULL DEFAULT -1,
    sync_token TEXT NOT NULL DEFAULT '',
    sync_expires INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO asset_vector_state(singleton) VALUES(1);
CREATE TRIGGER IF NOT EXISTS visual_insert AFTER INSERT ON asset_vectors
WHEN NEW.space='visual' BEGIN
    UPDATE asset_vector_state SET generation=generation+1;
END;
CREATE TRIGGER IF NOT EXISTS visual_delete AFTER DELETE ON asset_vectors
WHEN OLD.space='visual' BEGIN
    UPDATE asset_vector_state SET generation=generation+1;
END;
CREATE TRIGGER IF NOT EXISTS visual_assets_queued AFTER UPDATE OF status ON assets
WHEN OLD.status IN ('ready','partial') AND NEW.status NOT IN ('ready','partial')
AND OLD.sku IS NOT NULL BEGIN
    UPDATE asset_vector_state SET generation=generation+1;
END;
CREATE TRIGGER IF NOT EXISTS asset_chunks_delete AFTER DELETE ON asset_chunks BEGIN
    DELETE FROM asset_fts WHERE chunk_id=OLD.chunk_id;
    DELETE FROM policy_chunk_fts WHERE chunk_id=OLD.chunk_id;
    DELETE FROM policy_chunks WHERE chunk_id=OLD.chunk_id;
END;
"""
