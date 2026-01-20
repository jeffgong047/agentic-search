-- Legal Document Corpus Schema
-- Optimized for case-based organization with knowledge graph

-- Cases table (top-level organization)
CREATE TABLE IF NOT EXISTS cases (
    case_id VARCHAR(255) PRIMARY KEY,
    case_number VARCHAR(100),
    plaintiff VARCHAR(500),
    defendant VARCHAR(500),
    injury_type VARCHAR(100),
    jurisdiction VARCHAR(100),
    filing_date DATE,
    settlement_amount DECIMAL(12, 2),
    status VARCHAR(50),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_case_injury ON cases(injury_type);
CREATE INDEX idx_case_jurisdiction ON cases(jurisdiction);
CREATE INDEX idx_case_status ON cases(status);

-- Documents table (linked to cases)
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(255) PRIMARY KEY,
    case_id VARCHAR(255) NOT NULL REFERENCES cases(case_id) ON DELETE CASCADE,
    doc_type VARCHAR(100) NOT NULL,  -- deposition, demand, medical_record, settlement, etc.
    title VARCHAR(500),
    content TEXT NOT NULL,
    page_count INTEGER,
    file_path VARCHAR(1000),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_doc_case ON documents(case_id);
CREATE INDEX idx_doc_type ON documents(doc_type);
CREATE INDEX idx_doc_content_fts ON documents USING gin(to_tsvector('english', content));

-- Entities table (people, organizations, injuries, treatments)
CREATE TABLE IF NOT EXISTS entities (
    entity_id VARCHAR(255) PRIMARY KEY,
    entity_type VARCHAR(100) NOT NULL,  -- plaintiff, defendant, expert, injury, treatment, etc.
    name VARCHAR(500) NOT NULL,
    attributes JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_entity_type ON entities(entity_type);
CREATE INDEX idx_entity_name ON entities(name);
CREATE INDEX idx_entity_attrs ON entities USING gin(attributes);

-- Relations table (knowledge graph edges)
CREATE TABLE IF NOT EXISTS relations (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(255) NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_id VARCHAR(255) NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation_type VARCHAR(100) NOT NULL,  -- sustained, treated_by, examined_by, represented_by, etc.
    case_id VARCHAR(255) REFERENCES cases(case_id) ON DELETE CASCADE,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_relation_source ON relations(source_id);
CREATE INDEX idx_relation_target ON relations(target_id);
CREATE INDEX idx_relation_type ON relations(relation_type);
CREATE INDEX idx_relation_case ON relations(case_id);

-- Document-Entity mapping (which entities appear in which documents)
CREATE TABLE IF NOT EXISTS document_entities (
    document_id VARCHAR(255) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    entity_id VARCHAR(255) NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    mention_count INTEGER DEFAULT 1,
    PRIMARY KEY (document_id, entity_id)
);

CREATE INDEX idx_docent_doc ON document_entities(document_id);
CREATE INDEX idx_docent_entity ON document_entities(entity_id);

-- Sample data for testing
INSERT INTO cases (case_id, case_number, plaintiff, defendant, injury_type, jurisdiction, settlement_amount, status) VALUES
    ('case_12345', 'CV-2024-001', 'John Doe', 'ABC Corporation', 'spinal_injury', 'California', 500000.00, 'settled'),
    ('case_12346', 'CV-2024-002', 'Jane Smith', 'XYZ Industries', 'brain_injury', 'California', 750000.00, 'active'),
    ('case_12347', 'CV-2024-003', 'Robert Johnson', 'DEF Company', 'spinal_injury', 'New York', 450000.00, 'settled')
ON CONFLICT (case_id) DO NOTHING;

INSERT INTO entities (entity_id, entity_type, name, attributes) VALUES
    ('plaintiff_john_doe', 'plaintiff', 'John Doe', '{"age": 45, "occupation": "truck_driver"}'),
    ('injury_spinal_001', 'injury', 'L4-L5 Herniated Disc', '{"severity": "severe", "requires_surgery": true}'),
    ('expert_dr_smith', 'medical_expert', 'Dr. Sarah Smith', '{"specialty": "orthopedic_surgery", "years_experience": 20}'),
    ('plaintiff_jane_smith', 'plaintiff', 'Jane Smith', '{"age": 32, "occupation": "nurse"}'),
    ('injury_brain_001', 'injury', 'Traumatic Brain Injury', '{"severity": "moderate", "glasgow_score": 12}')
ON CONFLICT (entity_id) DO NOTHING;

INSERT INTO relations (source_id, target_id, relation_type, case_id) VALUES
    ('plaintiff_john_doe', 'injury_spinal_001', 'sustained', 'case_12345'),
    ('expert_dr_smith', 'plaintiff_john_doe', 'examined', 'case_12345'),
    ('plaintiff_jane_smith', 'injury_brain_001', 'sustained', 'case_12346')
ON CONFLICT DO NOTHING;

INSERT INTO documents (document_id, case_id, doc_type, title, content) VALUES
    ('case_12345_depo_1', 'case_12345', 'deposition', 'Plaintiff Deposition',
     'Deposition of John Doe. Q: Please describe the accident. A: I was driving my truck on I-5 when the defendant''s vehicle ran a red light...'),
    ('case_12345_demand_1', 'case_12345', 'demand_letter', 'Demand Letter',
     'Demand for settlement in the amount of $500,000 for spinal injury sustained by John Doe...'),
    ('case_12346_medical_1', 'case_12346', 'medical_record', 'ER Report',
     'Patient Jane Smith presented to ER with traumatic brain injury. Glasgow Coma Score: 12...')
ON CONFLICT (document_id) DO NOTHING;

INSERT INTO document_entities (document_id, entity_id, mention_count) VALUES
    ('case_12345_depo_1', 'plaintiff_john_doe', 5),
    ('case_12345_depo_1', 'injury_spinal_001', 3),
    ('case_12345_demand_1', 'plaintiff_john_doe', 2),
    ('case_12345_demand_1', 'injury_spinal_001', 4)
ON CONFLICT DO NOTHING;

-- Useful views for common queries

-- Case summary view
CREATE OR REPLACE VIEW case_summary AS
SELECT
    c.case_id,
    c.case_number,
    c.plaintiff,
    c.injury_type,
    c.jurisdiction,
    c.settlement_amount,
    COUNT(DISTINCT d.document_id) as document_count,
    COUNT(DISTINCT de.entity_id) as entity_count
FROM cases c
LEFT JOIN documents d ON c.case_id = d.case_id
LEFT JOIN document_entities de ON d.document_id = de.document_id
GROUP BY c.case_id, c.case_number, c.plaintiff, c.injury_type, c.jurisdiction, c.settlement_amount;

-- Document with entities view
CREATE OR REPLACE VIEW documents_with_entities AS
SELECT
    d.document_id,
    d.case_id,
    d.doc_type,
    d.title,
    d.content,
    json_agg(json_build_object(
        'entity_id', e.entity_id,
        'entity_type', e.entity_type,
        'name', e.name
    )) as entities
FROM documents d
LEFT JOIN document_entities de ON d.document_id = de.document_id
LEFT JOIN entities e ON de.entity_id = e.entity_id
GROUP BY d.document_id, d.case_id, d.doc_type, d.title, d.content;
