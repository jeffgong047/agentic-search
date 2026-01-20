# Agent Intelligence Diagnosis & Fix

## Root Cause Found ✅

### The Problem

**Agent suggests non-existent filters:**
- Query: "What is the statute of limitations for breach of contract?"
- Agent infers: `doc_type="statute"` or `doc_type="legal_analysis"`
- Reality: **These don't exist in the corpus!**

**Actual doc_types in index:**
- `employment_agreement` (200 docs)
- `lease_agreement` (100 docs)
- `sales_contract` (100 docs)
- `nda` (50 docs)
- `settlement_agreement` (30 docs)
- `demand_letter` (20 docs)

### Why It Happens

1. **LLM hallucinates**: Agent suggests doc_types based on query semantics, not index reality
2. **No validation**: System doesn't check if doc_type exists before applying filter
3. **No fallback**: If filter returns 0 results, query simply fails

### Impact

- **19/50 queries (38%)** return 0 results due to invalid filters
- Baseline returns 100% success because it doesn't filter at all

---

## The Fix: Validate + Fallback

### Strategy 1: Pre-Query Validation (Recommended)

```python
# In elasticsearch_hybrid_backend.py

def _get_available_doc_types(self, es, index_name):
    """Cache available doc_types from index"""
    agg_query = {
        "size": 0,
        "aggs": {
            "doc_types": {
                "terms": {"field": "metadata.doc_type.keyword", "size": 100}
            }
        }
    }
    response = es.search(index=index_name, body=agg_query)
    return {bucket['key'] for bucket in response['aggregations']['doc_types']['buckets']}

def validate_and_fix_filters(self, filters, available_doc_types):
    """Validate doc_type exists, otherwise remove filter"""
    if 'doc_type' in filters:
        if filters['doc_type'] not in available_doc_types:
            print(f"[Warning] doc_type '{filters['doc_type']}' not found in index")
            print(f"[Warning] Available types: {available_doc_types}")
            print(f"[Warning] Removing filter to avoid 0 results")
            del filters['doc_type']
    return filters
```

### Strategy 2: Post-Query Fallback (Complementary)

```python
def hybrid_search_with_fallback(self, query):
    """Try with filters first, fallback if 0 results"""
    
    # Attempt 1: With filters
    results = self.hybrid_search(query)
    
    # If 0 results and we had filters, retry without them
    if len(results) == 0 and query.filters:
        print(f"[Fallback] Got 0 results with filters, retrying without...")
        query_no_filter = query.copy()
        query_no_filter.filters = {}
        results = self.hybrid_search(query_no_filter)
        print(f"[Fallback] Retrieved {len(results)} docs without filters")
    
    return results
```

### Strategy 3: LLM Constraint (Preventive)

```python
# In orchestrator.py DSPy prompt:

filter_constraints: str = dspy.OutputField(
    desc=f"""Metadata filters as JSON dict. 
    ONLY use these EXACT doc_types that exist in the index: 
    {available_doc_types}
    
    Example: {{'doc_type': 'employment_agreement', 'year': 2023}}
    
    If query doesn't match any doc_type, return {{}}"""
)
```

---

## Recommendation: Implement All 3

**Why**:
1. **Strategy 1 (Validation)**: Catches hallucinations before wasting ES query
2. **Strategy 2 (Fallback)**: Safety net if validation misses edge cases
3. **Strategy 3 (LLM Constraint)**: Teaches agent the actual schema

**Expected Improvement**:
- Zero-result rate: 38% → ~5% (some queries may legitimately have no matches)
- Agent now **at least as good as baseline** (can always fall back)

---

## Implementation Priority

**Immediate** (10 min):
- Add Strategy 2 (fallback) to `elasticsearch_hybrid_backend.py`

**Short-term** (30 min):
- Add Strategy 1 (validation) with doc_type caching

**Medium-term** (1 hour):
- Add Strategy 3 (constrain LLM prompt with actual schema)
- Add observability (log when fallback triggers)

---

## Why This Matters

**Current**: Agent intelligently infers semantics but breaks on execution  
**After Fix**: Agent infers semantics AND gracefully degrades to baseline when needed

**Result**: Agent will **never be worse than baseline**, only equal or better.
