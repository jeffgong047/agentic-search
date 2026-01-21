-- Initialize PostgreSQL schema for knowledge graph storage

-- Entities table
CREATE TABLE IF NOT EXISTS entities (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(500) NOT NULL,
    type VARCHAR(100) NOT NULL,
    attributes JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_entity_type ON entities(type);
CREATE INDEX idx_entity_name ON entities(name);

-- Relations table
CREATE TABLE IF NOT EXISTS relations (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(255) NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    target_id VARCHAR(255) NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    relation_type VARCHAR(100) NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_relation_source ON relations(source_id);
CREATE INDEX idx_relation_target ON relations(target_id);
CREATE INDEX idx_relation_type ON relations(relation_type);

-- Document-Entity mapping
CREATE TABLE IF NOT EXISTS document_entities (
    document_id VARCHAR(255) NOT NULL,
    entity_id VARCHAR(255) NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (document_id, entity_id)
);

CREATE INDEX idx_doc_entity_doc ON document_entities(document_id);
CREATE INDEX idx_doc_entity_entity ON document_entities(entity_id);

-- Sample data for testing
INSERT INTO entities (id, name, type, attributes) VALUES
    ('mickey_mouse_meta', 'Mickey Mouse', 'person', '{"org": "Meta", "role": "Researcher"}'),
    ('meta_platforms', 'Meta Platforms Inc.', 'organization', '{"industry": "Tech"}'),
    ('mickey_mouse_shanghai', 'Mickey Mouse', 'person', '{"org": "Shanghai Law Firm", "role": "Lawyer"}')
ON CONFLICT (id) DO NOTHING;

INSERT INTO relations (source_id, target_id, relation_type) VALUES
    ('mickey_mouse_meta', 'meta_platforms', 'employed_by'),
    ('mickey_mouse_shanghai', 'meta_platforms', 'unrelated_to')
ON CONFLICT DO NOTHING;
