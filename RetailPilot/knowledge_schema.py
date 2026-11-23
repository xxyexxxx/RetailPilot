"""Canonical retrieval records and derived indexes. Sources retain business ownership."""

KNOWLEDGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS knowledge_sources (
    source_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL CHECK(domain IN ('product','policy')),
    sku TEXT REFERENCES products(sku) ON DELETE CASCADE,
    doc_id TEXT REFERENCES policy_documents(doc_id) ON DELETE CASCADE,
    asset_id TEXT UNIQUE REFERENCES assets(asset_id) ON DELETE CASCADE,
    source_format TEXT NOT NULL CHECK(source_format IN ('catalog','markdown','pdf','image')),
    source_uri TEXT NOT NULL,
    version TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    CHECK((domain='product' AND sku IS NOT NULL AND doc_id IS NULL) OR
          (domain='policy' AND doc_id IS NOT NULL AND sku IS NULL))
);
CREATE TRIGGER IF NOT EXISTS knowledge_source_owner BEFORE INSERT ON knowledge_sources
WHEN NEW.asset_id IS NOT NULL BEGIN
    SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM assets a WHERE a.asset_id=NEW.asset_id
        AND ((a.sku=NEW.sku AND NEW.doc_id IS NULL) OR (a.doc_id=NEW.doc_id AND NEW.sku IS NULL)))
        THEN RAISE(ABORT,'knowledge source owner mismatch') END;
END;
CREATE TRIGGER IF NOT EXISTS knowledge_source_owner_update BEFORE UPDATE ON knowledge_sources
WHEN NEW.asset_id IS NOT NULL BEGIN
    SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM assets a WHERE a.asset_id=NEW.asset_id
        AND ((a.sku=NEW.sku AND NEW.doc_id IS NULL) OR (a.doc_id=NEW.doc_id AND NEW.sku IS NULL)))
        THEN RAISE(ABORT,\'knowledge source owner mismatch\') END;
END;
CREATE TRIGGER IF NOT EXISTS knowledge_asset_owner_update BEFORE UPDATE OF sku,doc_id ON assets
WHEN EXISTS(SELECT 1 FROM knowledge_sources s WHERE s.asset_id=OLD.asset_id
    AND (s.sku IS NOT NEW.sku OR s.doc_id IS NOT NEW.doc_id)) BEGIN
    SELECT RAISE(ABORT,\'indexed asset ownership is immutable\');
END;
CREATE INDEX IF NOT EXISTS knowledge_sources_owner ON knowledge_sources(domain,sku,doc_id);
CREATE TABLE IF NOT EXISTS knowledge_contexts (
    context_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES knowledge_sources(source_id) ON DELETE CASCADE,
    content TEXT NOT NULL, section_path TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS knowledge_units (
    unit_id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES knowledge_sources(source_id) ON DELETE CASCADE,
    element_id TEXT REFERENCES asset_elements(element_id) ON DELETE CASCADE,
    modality TEXT NOT NULL CHECK(modality IN ('text','table','image')),
    evidence_type TEXT NOT NULL,
    search_text TEXT NOT NULL,
    evidence_text TEXT NOT NULL,
    structured_json TEXT,
    field_refs_json TEXT NOT NULL DEFAULT '[]',
    section_path TEXT NOT NULL DEFAULT '',
    page_no INTEGER,
    char_start INTEGER NOT NULL DEFAULT 0,
    char_end INTEGER NOT NULL DEFAULT 0,
    ordinal INTEGER NOT NULL DEFAULT 0,
    context_id TEXT REFERENCES knowledge_contexts(context_id) ON DELETE SET NULL,
    content_hash TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    index_version TEXT NOT NULL DEFAULT 'knowledge-v1',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TRIGGER IF NOT EXISTS knowledge_unit_element BEFORE INSERT ON knowledge_units
WHEN NEW.element_id IS NOT NULL BEGIN
    SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM asset_elements e JOIN knowledge_sources s
        ON s.asset_id=e.asset_id WHERE s.source_id=NEW.source_id AND e.element_id=NEW.element_id)
        THEN RAISE(ABORT,'knowledge element source mismatch') END;
END;
CREATE INDEX IF NOT EXISTS knowledge_units_source ON knowledge_units(source_id,ordinal);
CREATE TABLE IF NOT EXISTS knowledge_unit_elements (
    unit_id TEXT NOT NULL REFERENCES knowledge_units(unit_id) ON DELETE CASCADE,
    element_id TEXT NOT NULL REFERENCES asset_elements(element_id) ON DELETE CASCADE,
    PRIMARY KEY(unit_id,element_id)
);
CREATE TRIGGER IF NOT EXISTS knowledge_context_element BEFORE INSERT ON knowledge_unit_elements BEGIN
    SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM knowledge_units u JOIN knowledge_sources s USING(source_id)
        JOIN asset_elements e ON e.asset_id=s.asset_id
        WHERE u.unit_id=NEW.unit_id AND e.element_id=NEW.element_id)
        THEN RAISE(ABORT,'knowledge context element owner mismatch') END;
END;
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    unit_id UNINDEXED, identity, title, brand, category, body,
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TABLE IF NOT EXISTS knowledge_vectors (
    unit_id TEXT NOT NULL REFERENCES knowledge_units(unit_id) ON DELETE CASCADE,
    space TEXT NOT NULL CHECK(space IN ('text','visual')),
    model_version TEXT NOT NULL, content_hash TEXT NOT NULL,
    dim INTEGER NOT NULL CHECK(dim>0), vector BLOB NOT NULL,
    PRIMARY KEY(unit_id,space),
    CHECK(length(vector)=dim*4)
);
CREATE INDEX IF NOT EXISTS knowledge_vectors_model ON knowledge_vectors(space,model_version);
CREATE TABLE IF NOT EXISTS knowledge_index_state (
    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
    generation INTEGER NOT NULL DEFAULT 0,
    target TEXT NOT NULL DEFAULT '',
    synced_generation INTEGER NOT NULL DEFAULT -1,
    sync_token TEXT NOT NULL DEFAULT '', sync_expires INTEGER NOT NULL DEFAULT 0
);
INSERT OR IGNORE INTO knowledge_index_state(singleton) VALUES(1);
CREATE TABLE IF NOT EXISTS knowledge_dirty (
    sku TEXT PRIMARY KEY REFERENCES products(sku) ON DELETE CASCADE
);
CREATE TRIGGER IF NOT EXISTS knowledge_product_insert AFTER INSERT ON products BEGIN
    INSERT OR IGNORE INTO knowledge_dirty VALUES(NEW.sku);
END;
CREATE TRIGGER IF NOT EXISTS knowledge_product_update AFTER UPDATE OF
    title,brand,category,category_label,description,features_json,attributes_json,tags_json,tags_text,attributes_text
ON products BEGIN
    INSERT OR IGNORE INTO knowledge_dirty VALUES(NEW.sku);
END;
CREATE TRIGGER IF NOT EXISTS knowledge_asset_state AFTER UPDATE OF status ON assets BEGIN
    UPDATE knowledge_index_state SET generation=generation+1;
END;
CREATE TRIGGER IF NOT EXISTS knowledge_policy_version AFTER UPDATE OF version ON policy_documents BEGIN
    UPDATE knowledge_index_state SET generation=generation+1;
END;
CREATE TRIGGER IF NOT EXISTS knowledge_unit_delete AFTER DELETE ON knowledge_units BEGIN
    DELETE FROM knowledge_fts WHERE unit_id=OLD.unit_id;
    UPDATE knowledge_index_state SET generation=generation+1;
END;
CREATE TRIGGER IF NOT EXISTS knowledge_vector_insert AFTER INSERT ON knowledge_vectors BEGIN
    UPDATE knowledge_index_state SET generation=generation+1;
END;
CREATE TRIGGER IF NOT EXISTS knowledge_vector_delete AFTER DELETE ON knowledge_vectors BEGIN
    UPDATE knowledge_index_state SET generation=generation+1;
END;
"""
