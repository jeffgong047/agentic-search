#!/usr/bin/env python3
"""
Elasticsearch-Only Baseline Evaluation
No LLM, no agent - pure vector + keyword search with static RRF ranking

Purpose: Quantify the value-add of the LLM agent by comparing:
- Agent system: DSPy orchestration + metadata filtering + query expansion
- Baseline: Direct ES hybrid search with no intelligence

Usage:
    python3 run_es_baseline.py --es-url http://localhost:9200 --max-queries 50
"""

import argparse
import json
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch

def load_tasks(tasks_path="legalbench/data/tasks.jsonl"):
    """Load evaluation tasks"""
    tasks = []
    with open(tasks_path) as f:
        for line in f:
            tasks.append(json.loads(line))
    return tasks

def simple_hybrid_search(es, index_name, query_text, query_vector, top_k=20):
    """
    Direct Elasticsearch hybrid search - NO LLM, NO AGENT
    Just: kNN + BM25 + RRF (no metadata filtering, no query expansion)
    """
    body = {
        "knn": {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 5  # Standard 5:1 ratio
        },
        "query": {
            "multi_match": {
                "query": query_text,  # Raw query, no expansion
                "fields": ["content"],
                "type": "best_fields"
            }
        },
        "size": top_k,
        "rank": {
            "rrf": {
                "window_size": 50
            }
        }
    }
    
    start = time.time()
    response = es.search(index=index_name, body=body)
    latency_ms = (time.time() - start) * 1000
    
    # Extract doc IDs
    hits = response['hits']['hits']
    doc_ids = [hit['_id'] for hit in hits]
    
    return {
        'doc_ids': doc_ids,
        'count': len(doc_ids),
        'latency_ms': latency_ms
    }

def run_baseline_evaluation(es_url, index_name, max_queries=50):
    """Run pure ES baseline (no agent)"""
    print("=" * 70)
    print("ELASTICSEARCH-ONLY BASELINE EVALUATION")
    print("=" * 70)
    print(f"ES URL: {es_url}")
    print(f"Index: {index_name}")
    print(f"Max Queries: {max_queries}")
    print()
    
    # Initialize
    es = Elasticsearch([es_url])
    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # Load tasks
    tasks = load_tasks()[:max_queries]
    print(f"Loaded {len(tasks)} tasks")
    print()
    
    # Run evaluation
    results = []
    total_latency = 0
    total_retrieved = 0
    
    for task in tqdm(tasks, desc="Baseline evaluation"):
        query_text = task['query']
        
        # Generate embedding
        query_vector = embedding_model.encode(query_text).tolist()
        
        # Simple hybrid search (NO agent intelligence)
        result = simple_hybrid_search(es, index_name, query_text, query_vector)
        
        total_latency += result['latency_ms']
        total_retrieved += result['count']
        
        results.append({
            'task_id': task['task_id'],
            'query': query_text,
            'category': task['category'],
            'domain': task['domain'],
            'latency_ms': result['latency_ms'],
            'num_retrieved': result['count'],
            'retrieved_ids': result['doc_ids']
        })
    
    # Calculate metrics
    mean_latency = total_latency / len(results)
    latencies = [r['latency_ms'] for r in results]
    retrieval_counts = [r['num_retrieved'] for r in results]
    
    print()
    print("=" * 70)
    print("BASELINE RESULTS")
    print("=" * 70)
    print(f"Total Queries: {len(results)}")
    print()
    print("--- LATENCY METRICS ---")
    print(f"Mean: {mean_latency:.1f} ms")
    print(f"Median: {sorted(latencies)[len(latencies)//2]:.1f} ms")
    print(f"Min: {min(latencies):.1f} ms")
    print(f"Max: {max(latencies):.1f} ms")
    print()
    print("--- RETRIEVAL METRICS ---")
    print(f"Mean docs retrieved: {sum(retrieval_counts)/len(retrieval_counts):.1f}")
    print(f"Zero results: {sum(1 for c in retrieval_counts if c == 0)}/{len(results)}")
    print(f"Full results (20): {sum(1 for c in retrieval_counts if c == 20)}/{len(results)}")
    print()
    
    # Save results
    output = {
        'config': {
            'es_url': es_url,
            'index': index_name,
            'max_queries': max_queries,
            'method': 'ES_BASELINE_NO_AGENT'
        },
        'summary': {
            'total_queries': len(results),
            'mean_latency_ms': mean_latency,
            'median_latency_ms': sorted(latencies)[len(latencies)//2],
            'mean_docs_retrieved': sum(retrieval_counts)/len(retrieval_counts),
            'zero_result_rate': sum(1 for c in retrieval_counts if c == 0) / len(results)
        },
        'per_query_results': results
    }
    
    output_path = "legalbench/results/baseline_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    return output

def compare_to_agent(baseline_path="legalbench/results/baseline_results.json",
                     agent_path="legalbench/results/results.json"):
    """Compare baseline to agent results"""
    with open(baseline_path) as f:
        baseline = json.load(f)
    with open(agent_path) as f:
        agent = json.load(f)
    
    print()
    print("=" * 70)
    print("BASELINE vs AGENT COMPARISON")
    print("=" * 70)
    print()
    
    print(f"{'Metric':<30} {'ES Baseline':<15} {'Agent System':<15} {'Delta':<15}")
    print("-" * 70)
    
    # Latency
    b_lat = baseline['summary']['mean_latency_ms']
    a_lat = agent['summary']['latency_metrics']['mean_ms']
    delta_lat = ((a_lat - b_lat) / b_lat) * 100
    print(f"{'Mean Latency (ms)':<30} {b_lat:<15.1f} {a_lat:<15.1f} {delta_lat:+.1f}%")
    
    b_med = baseline['summary']['median_latency_ms']
    a_med = agent['summary']['latency_metrics']['median_ms']
    delta_med = ((a_med - b_med) / b_med) * 100
    print(f"{'Median Latency (ms)':<30} {b_med:<15.1f} {a_med:<15.1f} {delta_med:+.1f}%")
    
    # Retrieval
    b_docs = baseline['summary']['mean_docs_retrieved']
    a_counts = [q.get('num_retrieved', 0) for q in agent['per_query_results']]
    a_docs = sum(a_counts) / len(a_counts)
    delta_docs = ((a_docs - b_docs) / b_docs) * 100 if b_docs > 0 else 0
    print(f"{'Mean Docs Retrieved':<30} {b_docs:<15.1f} {a_docs:<15.1f} {delta_docs:+.1f}%")
    
    b_zero = baseline['summary']['zero_result_rate']
    a_zero = sum(1 for c in a_counts if c == 0) / len(a_counts)
    delta_zero = ((a_zero - b_zero) / b_zero) * 100 if b_zero > 0 else 0
    print(f"{'Zero Result Rate':<30} {b_zero*100:<15.1f}% {a_zero*100:<15.1f}% {delta_zero:+.1f}%")
    
    print()
    print("INTERPRETATION:")
    if delta_lat > 0:
        print(f"  ⚠️  Agent is {delta_lat:.0f}% SLOWER (LLM overhead)")
    else:
        print(f"  ✅ Agent is {abs(delta_lat):.0f}% FASTER")
    
    if delta_docs > 0:
        print(f"  ✅ Agent retrieves {delta_docs:.0f}% MORE docs")
    else:
        print(f"  ⚠️  Agent retrieves {abs(delta_docs):.0f}% FEWER docs")
    
    if delta_zero < 0:
        print(f"  ✅ Agent has {abs(delta_zero):.0f}% FEWER zero-result queries")
    else:
        print(f"  ⚠️  Agent has {abs(delta_zero):.0f}% MORE zero-result queries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ES-only baseline evaluation")
    parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL")
    parser.add_argument("--index", default="legalbench_documents", help="Index name")
    parser.add_argument("--max-queries", type=int, default=50, help="Max queries to evaluate")
    parser.add_argument("--compare", action="store_true", help="Compare to agent results")
    args = parser.parse_args()
    
    # Run baseline
    run_baseline_evaluation(args.es_url, args.index, args.max_queries)
    
    # Optionally compare
    if args.compare:
        compare_to_agent()
