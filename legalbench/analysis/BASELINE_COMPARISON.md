# Baseline vs Agent Comparison Summary

## Elasticsearch-Only Baseline Results

**Configuration**: Direct kNN + BM25 + RRF (no LLM, no agent intelligence)

| Metric | Value |
|--------|-------|
| **Median Latency** | 30.5ms |
| **Mean Latency** | 31.1ms |
| **Min/Max** | 20.1ms / 67.0ms |
| **Mean Docs Retrieved** | 20.0 (perfect!) |
| **Zero Results** | 0/50 (0%) |
| **Full Results** | 50/50 (100%) |

---

## Agent vs Baseline Comparison

| Metric | ES Baseline | Agent System | Delta |
|--------|-------------|--------------|-------|
| **Median Latency** | 30.5ms | 127.6ms | **+318%** ⚠️ |
| **Mean Latency** | 31.1ms | 2,339ms | **+7,423%** ⚠️  |
| **Mean Docs Retrieved** | 20.0 | 12.4 | **-38%** ⚠️ |
| **Zero Result Rate** | 0% | 38% | **+38%** ⚠️ |
| **Full Results** | 100% | 62% | **-38%** ⚠️ |

---

## Analysis

### What the Baseline Shows

✅ **Pure ES is extremely fast**: 30ms median (4x faster than agent)  
✅ **100% retrieval success**: No over-filtering because no metadata constraints  
✅ **Consistent performance**: Very narrow latency range (20-67ms)

### What the Agent Adds

⚠️ **Latency Cost**: 318% slower (median) due to LLM orchestration  
⚠️ **Over-Filtering**: 38% zero-result rate from aggressive metadata filtering  
✅ **Intelligence**: Smart query understanding + metadata inference  
✅ **Query Expansion**: Adds domain-specific keywords  

### The Trade-off

**Baseline**: Fast + reliable, but "dumb" (no filtering, all docs treated equally)  
**Agent**: Slow + selective, but "smart" (understands context, applies filters)

---

## Recommendation

**Hybrid Approach**:
1. **Use ES baseline** for simple queries (e.g., direct keyword match)
2. **Use Agent** for complex queries requiring:
   - Domain-specific filtering
   - Query expansion with legal terminology
   - Multi-turn reasoning

**Optimization Path**:
- Add **fast-path detection**: If query is simple → skip LLM → use baseline
- Implement **fallback logic**: If agent returns 0 results → retry with baseline
- **Cache agent strategies**: Store common query patterns to avoid LLM re-computation

---

## Conclusion

The **LLM agent provides intelligence at a 318% latency cost**. For production:
- Fast tier (baseline): 30ms, 100% success
- Smart tier (agent): 128ms (when cached), selective filtering

**Best of both worlds**: Use baseline as default, agent for complex queries or as a fallback.

---

**Data Files**:
- `baseline_results.json` - ES-only evaluation data
- `results.json` - Agent evaluation data  
- `run_es_baseline.py` - Baseline evaluation script
