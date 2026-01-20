import json
import glob
import os
import sys

def load_qrels(filepath="legalbench/data/qrels.tsv"):
    qrels = {}
    with open(filepath, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                task_id = parts[0]
                doc_id = parts[1]
                grade = int(parts[2])
                if grade >= 2: # Relevant
                    if task_id not in qrels: qrels[task_id] = set()
                    qrels[task_id].add(doc_id)
    return qrels

def audit_traces(trace_dir="legalbench/results/traces"):
    files = glob.glob(os.path.join(trace_dir, "*.jsonl"))
    if not files: return
    latest_file = max(files, key=os.path.getctime)
    
    qrels = load_qrels()
    
    print(f"Auditing Trace: {latest_file}")
    
    with open(latest_file, 'r') as f:
        events = [json.loads(line) for line in f]
        
    # Group by task? The trace logger doesn't explicitly log task_id in every step, 
    # but run_evaluation.py logs "Critic Score" to stdout. 
    # Trace logs capture inputs/outputs.
    
    # We need to correlate Trace events to Task IDs.
    # The Eval script runs sequentially.
    # Let's look for "critic_input" which contains "query". 
    # We can try to match the query to the task query.
    
    # Actually, the TraceLogger logs are generic.
    # BUT, the `run_evaluation` script printed the Trace ID to stdout.
    # Let's inspect the `legalbench/results/logs/reflexion_run.log` if possible?
    # Or just rely on the query text.
    
    tasks = []
    with open("legalbench/data/tasks.jsonl", 'r') as f:
        tasks = [json.loads(line) for line in f]
    
    query_to_task = {t['query']: t['task_id'] for t in tasks}
    
    # Track progression
    audit_log = {} # task_id -> list of attempts
    
    # We need to find "reflector_input" events to see the FAILED attempt query
    # and "reflector_output" to see the NEW query.
    
    for e in events:
        if e['step'] == 'critic_input':
            # Input: {query, doc_count}
            # Or formatted string?
            inp = e.get('input', {})
            if isinstance(inp, dict):
                q = inp.get('query')
                if q in query_to_task:
                    tid = query_to_task[q]
                    if tid not in audit_log: audit_log[tid] = []
                    # We don't have the docs here, just counts.
            
        # The best way to audit is to look at the "reflector_input" which contains 'critique'.
        # And we need to know WHICH docs were retrieved.
        # The TraceLogger logged "doc_count", not doc IDs in critic_input.
        # Wait, run_evaluation.py:
        # self.tracer.log("critic_input", {"query": query, "doc_count": len(docs)})
        # It did NOT log the doc content or IDs to the jsonl.
        
        # UT OH. I cannot verify *which* docs were found from the trace.jsonl alone.
        # I only logged doc *count*.
        
        pass

    # Verification fallback: The Stdout Log.
    # The Log contained: "[Trace lb_xyz] Critic Score: ... Success!"
    # I should parse the STDOUT log.
    
    print("Cannot perform deep audit on JSONL because doc IDs were not logged.")
    print("Please analyze the stdout log 'legalbench/results/logs/reflexion_run.log'.")

if __name__ == "__main__":
    audit_traces()
