-- LegalBench Evaluation Database Schema
-- Initialized automatically by docker-compose on first startup

-- ========================================
-- Tasks Table: LegalBench Queries
-- ========================================
CREATE TABLE IF NOT EXISTS tasks (
    task_id VARCHAR(50) PRIMARY KEY,
    query TEXT NOT NULL,
    category VARCHAR(50),
    domain VARCHAR(50),
    difficulty VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE tasks IS 'LegalBench evaluation tasks (162 legal reasoning queries)';
COMMENT ON COLUMN tasks.category IS 'One of 6 reasoning types: issue_spotting, rule_recall, rule_application, rule_conclusion, interpretation, rhetorical_understanding';
COMMENT ON COLUMN tasks.domain IS 'Legal domain: contract_law, tort_law, criminal_law, corporate_law, ip_law';
COMMENT ON COLUMN tasks.difficulty IS 'Difficulty level: easy, medium, hard';

-- ========================================
-- Documents Table: Legal Document Corpus
-- ========================================
CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR(100) PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    chunk_count INTEGER DEFAULT 0,
    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE documents IS 'Legal document corpus for retrieval';
COMMENT ON COLUMN documents.metadata IS 'JSON metadata: {domain, doc_type, jurisdiction, year, entities}';

-- ========================================
-- Relevance Judgments (Qrels)
-- ========================================
CREATE TABLE IF NOT EXISTS qrels (
    task_id VARCHAR(50) REFERENCES tasks(task_id) ON DELETE CASCADE,
    doc_id VARCHAR(100) REFERENCES documents(doc_id) ON DELETE CASCADE,
    relevance_grade INTEGER CHECK (relevance_grade >= 0 AND relevance_grade <= 3),
    judgment_source VARCHAR(50) DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, doc_id)
);

COMMENT ON TABLE qrels IS 'Query-document relevance judgments for evaluation';
COMMENT ON COLUMN qrels.relevance_grade IS '0=not relevant, 1=marginally relevant, 2=relevant, 3=highly relevant';
COMMENT ON COLUMN qrels.judgment_source IS 'Source of judgment: manual, llm_judge, pseudo, automatic';

-- ========================================
-- Evaluation Results
-- ========================================
CREATE TABLE IF NOT EXISTS evaluation_results (
    eval_id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) NOT NULL,
    task_id VARCHAR(50) REFERENCES tasks(task_id) ON DELETE CASCADE,
    retrieved_ids TEXT[],
    retrieved_scores FLOAT[],
    latency_ms FLOAT,
    recall_at_5 FLOAT,
    recall_at_10 FLOAT,
    recall_at_20 FLOAT,
    precision_at_5 FLOAT,
    precision_at_10 FLOAT,
    precision_at_20 FLOAT,
    ndcg_at_10 FLOAT,
    num_relevant INTEGER DEFAULT 0,
    num_retrieved INTEGER DEFAULT 0,
    circuit_breaker_stopped BOOLEAN DEFAULT FALSE,
    novelty_score FLOAT,
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE evaluation_results IS 'Per-query evaluation results';
COMMENT ON COLUMN evaluation_results.run_id IS 'Unique identifier for evaluation run (e.g., timestamp or config hash)';
COMMENT ON COLUMN evaluation_results.retrieved_ids IS 'Array of document IDs retrieved (ordered by rank)';
COMMENT ON COLUMN evaluation_results.retrieved_scores IS 'Array of retrieval scores (parallel to retrieved_ids)';

-- ========================================
-- Aggregate Metrics (Per Run)
-- ========================================
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id VARCHAR(100) PRIMARY KEY,
    embedding_model VARCHAR(100),
    total_queries INTEGER,
    mean_recall_at_10 FLOAT,
    mean_precision_at_10 FLOAT,
    mean_ndcg_at_10 FLOAT,
    mean_latency_ms FLOAT,
    median_latency_ms FLOAT,
    p95_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    throughput_qps FLOAT,
    early_stop_rate FLOAT,
    mean_novelty FLOAT,
    total_runtime_sec FLOAT,
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

COMMENT ON TABLE run_metrics IS 'Aggregate metrics for each evaluation run';
COMMENT ON COLUMN run_metrics.config IS 'JSON configuration: {top_k, use_hyde, use_filters, max_iterations}';

-- ========================================
-- Indexes for Performance
-- ========================================
CREATE INDEX IF NOT EXISTS idx_qrels_task ON qrels(task_id);
CREATE INDEX IF NOT EXISTS idx_qrels_doc ON qrels(doc_id);
CREATE INDEX IF NOT EXISTS idx_qrels_grade ON qrels(relevance_grade);
CREATE INDEX IF NOT EXISTS idx_eval_results_run ON evaluation_results(run_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_task ON evaluation_results(task_id);
CREATE INDEX IF NOT EXISTS idx_eval_results_recall ON evaluation_results(recall_at_10);
CREATE INDEX IF NOT EXISTS idx_tasks_category ON tasks(category);
CREATE INDEX IF NOT EXISTS idx_tasks_domain ON tasks(domain);
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN(metadata);

-- ========================================
-- Views for Analysis
-- ========================================

-- View: Metrics by Category
CREATE OR REPLACE VIEW metrics_by_category AS
SELECT
    t.category,
    COUNT(DISTINCT e.task_id) AS num_queries,
    AVG(e.recall_at_10) AS avg_recall_at_10,
    AVG(e.precision_at_10) AS avg_precision_at_10,
    AVG(e.ndcg_at_10) AS avg_ndcg_at_10,
    AVG(e.latency_ms) AS avg_latency_ms
FROM evaluation_results e
JOIN tasks t ON e.task_id = t.task_id
GROUP BY t.category;

COMMENT ON VIEW metrics_by_category IS 'Aggregate metrics grouped by task category';

-- View: Metrics by Domain
CREATE OR REPLACE VIEW metrics_by_domain AS
SELECT
    t.domain,
    COUNT(DISTINCT e.task_id) AS num_queries,
    AVG(e.recall_at_10) AS avg_recall_at_10,
    AVG(e.precision_at_10) AS avg_precision_at_10,
    AVG(e.ndcg_at_10) AS avg_ndcg_at_10
FROM evaluation_results e
JOIN tasks t ON e.task_id = t.task_id
GROUP BY t.domain;

-- View: Latest Run Summary
CREATE OR REPLACE VIEW latest_run_summary AS
SELECT
    run_id,
    embedding_model,
    total_queries,
    mean_recall_at_10,
    mean_ndcg_at_10,
    mean_latency_ms,
    throughput_qps,
    created_at
FROM run_metrics
ORDER BY created_at DESC
LIMIT 10;

-- ========================================
-- Helper Functions
-- ========================================

-- Function: Calculate average precision for a query
CREATE OR REPLACE FUNCTION calculate_avg_precision(
    p_task_id VARCHAR,
    p_retrieved_ids TEXT[]
) RETURNS FLOAT AS $$
DECLARE
    relevant_docs TEXT[];
    avg_prec FLOAT := 0.0;
    num_relevant_found INTEGER := 0;
    i INTEGER;
BEGIN
    -- Get relevant document IDs for this task
    SELECT ARRAY_AGG(doc_id) INTO relevant_docs
    FROM qrels
    WHERE task_id = p_task_id AND relevance_grade >= 2;

    IF relevant_docs IS NULL OR ARRAY_LENGTH(relevant_docs, 1) = 0 THEN
        RETURN 0.0;
    END IF;

    -- Calculate average precision
    FOR i IN 1..ARRAY_LENGTH(p_retrieved_ids, 1) LOOP
        IF p_retrieved_ids[i] = ANY(relevant_docs) THEN
            num_relevant_found := num_relevant_found + 1;
            avg_prec := avg_prec + (num_relevant_found::FLOAT / i::FLOAT);
        END IF;
    END LOOP;

    RETURN avg_prec / ARRAY_LENGTH(relevant_docs, 1);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION calculate_avg_precision IS 'Calculate Average Precision for a single query';

-- ========================================
-- Sample Data (for testing)
-- ========================================

-- Insert sample task
INSERT INTO tasks (task_id, query, category, domain, difficulty)
VALUES
    ('lb_sample_001', 'Does this employment contract contain a non-compete clause?', 'issue_spotting', 'contract_law', 'easy')
ON CONFLICT (task_id) DO NOTHING;

-- Insert sample document
INSERT INTO documents (doc_id, content, metadata)
VALUES
    ('doc_sample_001',
     'EMPLOYMENT AGREEMENT\n\nThis Employment Agreement is entered into between Employer and Employee. Employee agrees that during employment and for a period of two years following termination, Employee shall not engage in any competitive business activities within a 50-mile radius of Company headquarters.',
     '{"domain": "contract_law", "doc_type": "employment_agreement", "jurisdiction": "CA", "year": 2023}'::jsonb)
ON CONFLICT (doc_id) DO NOTHING;

-- Insert sample qrel
INSERT INTO qrels (task_id, doc_id, relevance_grade, judgment_source)
VALUES
    ('lb_sample_001', 'doc_sample_001', 3, 'manual')
ON CONFLICT (task_id, doc_id) DO NOTHING;

-- ========================================
-- Completion Message
-- ========================================
DO $$
BEGIN
    RAISE NOTICE '========================================';
    RAISE NOTICE 'LegalBench Database Initialized Successfully';
    RAISE NOTICE '========================================';
    RAISE NOTICE 'Tables created: tasks, documents, qrels, evaluation_results, run_metrics';
    RAISE NOTICE 'Views created: metrics_by_category, metrics_by_domain, latest_run_summary';
    RAISE NOTICE 'Sample data inserted: 1 task, 1 document, 1 qrel';
    RAISE NOTICE '';
    RAISE NOTICE 'Connect: psql -h localhost -p 5433 -U benchuser -d legalbench_db';
    RAISE NOTICE 'Password: benchpass';
    RAISE NOTICE '========================================';
END $$;
