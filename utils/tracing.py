import json
import uuid
import os
from datetime import datetime
from typing import Any, Dict, Optional

class TraceLogger:
    def __init__(self, base_dir: str = "legalbench/results/traces"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.run_id = str(uuid.uuid4())
        self.trace_file = os.path.join(self.base_dir, f"trace_{self.run_id}.jsonl")
        print(f"[TraceLogger] Initialized run: {self.run_id}")

    def log(self, step: str, input_data: Any = None, output_data: Any = None, metadata: Optional[Dict] = None):
        """
        Log a single step in the agent's execution.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "run_id": self.run_id,
            "step": step,
            "input": self._sanitize(input_data),
            "output": self._sanitize(output_data),
            "metadata": metadata or {}
        }
        
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _sanitize(self, data: Any) -> Any:
        """Helper to ensure data is JSON serializable."""
        if isinstance(data, (str, int, float, bool, type(None))):
            return data
        if isinstance(data, (list, tuple)):
            return [self._sanitize(x) for x in data]
        if isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        # Fallback for objects (like DSPy modules)
        return str(data)

# Singleton instance for easy import if needed, 
# but usually instantiated per-run in the Orchestrator.
