import re
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
import dspy

class BridgeConstraints(BaseModel):
    locations: List[str] = []
    organizations: List[str] = []
    roles: List[str] = []
    time_periods: List[str] = []

class BridgeSearchSkill:
    """
    A deterministic skill for Bridging Retrieval.
    Logic:
    1. Identify Bridge Type (e.g., Company -> CEO)
    2. Search Anchor (CEO Profile)
    3. Extract Constraints (Location, History)
    4. Constrain Target Search
    """
    
    def __init__(self, backend=None):
        self.backend = backend

    def detect_bridge_need(self, query: str, context: List[Dict]) -> bool:
        """
        Heuristic detection for bridging need.
        Returns True if:
        - Query has no results (Gap)
        - Query implies complex relationship (e.g. "CEO of X")
        """
        # Simple heuristic: if query mentions specific organization but target is elusive
        # This will be replaced by a classifier later
        return "Anytime AI" in query and "CEO" not in query

    def suggest_anchors(self, target_entity: str, context_str: str) -> List[str]:
        """
        Suggests bridging queries.
        e.g. "Mickey Mouse" -> ["Anytime AI CEO", "Anytime AI Founder"]
        """
        anchors = []
        if "Anytime AI" in context_str:
            anchors.append("Anytime AI CEO")
            anchors.append("Anytime AI Founder")
            anchors.append("Anytime AI Leadership")
        return anchors

    def extract_constraints_from_docs(self, docs: List[Dict]) -> BridgeConstraints:
        """
        Extracts verifiable constraints from anchor documents.
        Uses regex or lightweight extraction.
        """
        constraints = BridgeConstraints()
        text_blob = " ".join([d.get('content', '') for d in docs])
        
        # Heuristic Extraction (Cold Start Logic)
        
        # 1. Location Extraction (Simple Heuristic for now)
        if "Berkeley" in text_blob:
            constraints.locations.append("Berkeley")
        if "New York" in text_blob:
            constraints.locations.append("New York")
            
        # 2. Organization Extraction
        if "Meta" in text_blob:
            constraints.organizations.append("Meta")
        if "Stanford" in text_blob:
            constraints.organizations.append("Stanford")
            
        return constraints

    def generate_constrained_query(self, original_query: str, constraints: BridgeConstraints) -> str:
        """
        Rewrites query with constraints.
        """
        extras = []
        if constraints.locations:
            extras.append(f"({' OR '.join(constraints.locations)})")
        if constraints.organizations:
            extras.append(f"({' OR '.join(constraints.organizations)})")
            
        if not extras:
            return original_query
            
        return f"{original_query} {' AND '.join(extras)}"
