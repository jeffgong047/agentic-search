import json
import glob
import os
import sys

def analyze_traces(trace_dir="legalbench/results/traces"):
    files = glob.glob(os.path.join(trace_dir, "*.jsonl"))
    if not files:
        print("No trace files found.")
        return

    # Use the most recent trace file
    latest_file = max(files, key=os.path.getctime)
    print(f"Analyzing trace file: {latest_file}")

    events = []
    with open(latest_file, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    print(f"Total events: {len(events)}")
    
    critics = [e for e in events if e['step'] == 'critic_output']
    reflections = [e for e in events if e['step'] == 'reflector_output']
    
    print(f"Critic Calls: {len(critics)}")
    print(f"Reflection Calls: {len(reflections)}")
    
    # Calculate pass rate
    scores = []
    for c in critics:
        try:
            # Handle DSPy output format
            data = c['output']
            if isinstance(data, dict):
                s = float(data.get('relevance_score', 0))
            else:
                s = 0.0 # Parse error fallback
            scores.append(s)
        except:
            pass
            
    avg_score = sum(scores) / len(scores) if scores else 0
    print(f"Average Critic Score: {avg_score:.2f}")
    
    # Show examples
    if reflections:
        print("\n--- SAMPLE REFLECTION ---")
        ex = reflections[0]
        
        # Debug: Print types
        print(f"Input Type: {type(ex.get('input'))}")
        print(f"Output Type: {type(ex.get('output'))}")
        print(f"Raw Input: {ex.get('input')}")

        try:
            inp = ex.get('input', {})
            if isinstance(inp, str):
                 print("Warning: Input is string, treating as raw message.")
                 critique_txt = inp
            else:
                 critique_txt = inp.get('critique', 'N/A')

            out = ex.get('output', {})
            if isinstance(out, str):
                 out_dict = {} # Can't parse string output easily without regex
            else:
                 out_dict = out
            
            print(f"Critique Input: {critique_txt}")
            print(f"Reflection: {out_dict.get('reflection', 'N/A')}")
            print(f"Improved Query: {out_dict.get('improved_query', 'N/A')}")
        except Exception as e:
            print(f"Error parsing sample: {e}")

if __name__ == "__main__":
    analyze_traces()
