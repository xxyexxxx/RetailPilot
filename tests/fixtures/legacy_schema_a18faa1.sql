-- Exact SQLite schema extracted from release a18faa1; independent of Git history in CI.

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
