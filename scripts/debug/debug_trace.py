"""
Debug Trace Script
Runs specific legal reasoning examples using Claude Opus via DSPy.
Captures and displays the Chain-of-Thought reasoning traces.
"""


print("DEBUG: STARTING", flush=True)

import os
import dspy
import json
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from core.orchestrator import LegalOrchestrator, AgentState

def setup_dspy_opus():
    """
    Configure DSPy to use Claude Opus.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  WARNING: ANTHROPIC_API_KEY not found in environment.")
    
    
    # Model Priority List (Attempting "Opus 4.5" / Best Available)
    models_to_try = [
        'anthropic/claude-3-opus-20240229',      # Opus (User Request)
        'anthropic/claude-3-5-sonnet-20240620',  # Sonnet 3.5 (Best Value/Power)
        'anthropic/claude-3-haiku-20240307',     # Haiku (Fallback)
        'claude-3-haiku-20240307'                # Haiku (Direct no-prefix)
    ]
    
    lm = None
    for model_id in models_to_try:
        try:
            print(f"🔄 Attempting to load model: {model_id}...", end="", flush=True)
            lm = dspy.LM(model=model_id, api_key=api_key, max_tokens=2000)
            # Test connection with a dummy call
            lm("Test connection")
            print(" ✅ SUCCESS!")
            break
        except Exception as e:
            print(f" ❌ FAILED ({str(e)[:100]}...)")
            lm = None
            
    if not lm:
        raise RuntimeError("Could not initialize ANY Anthropic model. Check API Key permissions.")

    dspy.configure(lm=lm)
    return lm

def print_trace(history):
    """
    Pretty print the last trace from DSPy history.
    """
    print("\n" + "="*80)
    print("🧠  CLAUDE OPUS REASONING TRACE")
    print("="*80)
    
    last_call = history[-1]
    
    # DSPy signature fields
    print(f"\n📝 INPUT (User Query + Context):")
    for k, v in last_call['kwargs'].items():
        if k != 'signature':
            print(f"  • {k}: {v}")
            
    # The 'response' usually contains the chain of thought if enabled
    # In newer DSPy, the CoT is often in the 'rationale' or implicit in the output
    
    # Try to extract chain of thought
    print(f"\n🤔 CHAIN OF THOUGHT:")
    # Depending on how DSPy stores it, it might be in different places.
    # We inspect the raw response/messages if possible.
    # But usually, it's just the 'rationale' field if using ChainOfThought module.
    
    # Let's inspect the outputs
    print(f"\n📤 OUTPUT (Structured Plan):")
    
    # In DSPy 3.x+ with dspy.LM, the 'response' might be the raw ModelResponse or the Prediction
    # The 'last_call' is from history. 
    # Usually history stores {'kwargs': ..., 'response': ...}
    
    response = last_call.get('response') 
    
    if response:
        # If it's a Pydantic model (Prediction or ModelResponse), convert to dict
        if hasattr(response, 'model_dump'):
            data = response.model_dump()
        elif hasattr(response, 'to_dict'):
            data = response.to_dict()
        elif isinstance(response, dict):
            data = response
        else:
             # Fallback: try iteration or str
             try:
                 data = dict(response)
             except:
                 data = {"raw": str(response)}

        # Try to find rationale
        # In dspy.LM history, we often just get the raw completion.
        # We need to look into choices -> message -> content
        content = None
        if 'choices' in data:
            try:
                content = data['choices'][0]['message']['content']
                print(f"\n📜 RAW RESPONSE CONTENT:\n{content}")
            except:
                pass
        
        # Print fields
        for k, v in data.items():
            if k != 'rationale' and k != 'choices' and k != 'usage':
                print(f"  • {k}: {v}")


    print("\n" + "="*80 + "\n")

def run_tests():
    print("🚀 Initializing Legal Orchestrator with Claude Haiku...")
    try:
        lm = setup_dspy_opus()
    except Exception as e:
        print(f"❌ Failed to setup Claude Opus: {e}")
        return

    orchestrator = LegalOrchestrator()
    
    test_cases = [
        {
            "query": "What are the requirements for adverse possession in California?",
            "verified_facts": [],
            "negative_cache": [],
            "desc": "Statute/Case Law Lookup"
        },
        {
            "query": "Find cases involving 'meta' and 'privacy' in 2024",
            "verified_facts": ["Meta Platforms Inc. is the primary entity."],
            "negative_cache": [{"entity": "Meta (prefix)", "reason": "Not the company"}],
            "desc": "Entity + Filter Constraints"
        },
        {
            "query": "Explain the doctrine of fair use for AI training",
            "verified_facts": [],
            "negative_cache": [],
            "desc": "Broad Conceptual Research"
        },
        {
            "query": "Compare statutes of limitation for fraud in NY vs CA",
            "verified_facts": [],
            "negative_cache": [],
            "desc": "Comparative Analysis (Multi-Jurisdiction)"
        },
        {
            "query": "What are the procedural steps to file a motion to dismiss in federal court?",
            "verified_facts": [],
            "negative_cache": [],
            "desc": "Procedural/Process Query"
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n🔹 TEST CASE {i+1}: {test['desc']}")

        print(f"   Query: {test['query']}")
        
        # Build mock state
        state = AgentState(
            query=test['query'],
            verified_facts=test['verified_facts'],
            negative_cache=test['negative_cache'],
            search_plan=None,
            results=[],
            final_report=""
        )
        
        # Run orchestrator
        try:
            plan = orchestrator.forward(state)
            
            # Print the trace
            print_trace(lm.history)
            
            # Verify structure
            print(f"✅ Created Plan: {plan.primary_intent} with {len(plan.search_queries)} queries")
            
        except Exception as e:
            print(f"❌ Error running test case: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_tests()
