
import json
import os
import argparse
import dspy
from typing import List, Dict, Any
from core.config import get_config, update_config
from core.graph_engine import AgenticSearchEngine
from core.backends import create_best_backend
from backends_wrapper import BackendAgenticSearchEngine

def setup_dspy(model_name: str, api_key: str):
    """Configure DSPy with the specified model"""
    print(f"[LongBench] Configuring DSPy with: {model_name}")
    # Note: Using dspy.LM for modern DSPy versions
    lm = dspy.LM(model=f"anthropic/{model_name}", api_key=api_key)
    dspy.configure(lm=lm)

def load_longbench_cite(data_path: str) -> List[Dict[str, Any]]:
    """Load LongBench Cite dataset from JSONL"""
    if not os.path.exists(data_path):
        print(f"Warning: Data path {data_path} not found. Using placeholder sample.")
        return [
            {
                "query": "Who is Mickey Mouse and what is their affiliation with Meta?",
                "context": ["Mickey Mouse is a researcher at Meta AI.", "Mickey Mouse is a lawyer in Shanghai."],
                "answer": "Mickey Mouse is a Senior Research Scientist at Meta Platforms Inc."
            }
        ]
    
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def run_eval(args):
    config = get_config()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment.")
        return

    # Setup DSPy
    setup_dspy(args.model, api_key)
    
    # Initialize backend
    backend = create_best_backend(backend_type="elasticsearch")
    engine = BackendAgenticSearchEngine(backend)
    
    # Load data
    dataset = load_longbench_cite(args.data_path)
    print(f"[LongBench] Loaded {len(dataset)} queries.")
    
    results = []
    for i, item in enumerate(dataset[:args.limit]):
        print(f"\n[{i+1}/{args.limit}] Query: {item['query']}")
        
        # Run search
        state = engine.search(item['query'])
        docs = state.get("retrieved_docs", [])
        
        # In LongBench Cite, we care about whether the retrieved docs support the answer
        # and if the agent can correctly identify the citations.
        results.append({
            "query": item['query'],
            "retrieved_count": len(docs),
            "top_doc_id": docs[0].id if docs else None,
            "top_doc_content": docs[0].content[:200] if docs else None
        })
        
    # Save results
    output_path = "legalbench/results/longbench_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[LongBench] Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LongBench Cite Evaluation")
    parser.add_argument("--data_path", type=str, default="data/longbench/cite.jsonl")
    parser.add_argument("--model", type=str, default="claude-3-haiku-20240307")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    
    run_eval(args)
