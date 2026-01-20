import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from core.skills.bridging import BridgeSearchSkill, BridgeConstraints

class TestBridgeSearchSkill(unittest.TestCase):
    def setUp(self):
        self.skill = BridgeSearchSkill()

    def test_detect_bridge_need(self):
        # Scenario: Gap in info for "Anytime AI" without CEO mentioned
        self.assertTrue(self.skill.detect_bridge_need("Who is Qian Chen at Anytime AI?", []))
        # Scenario: CEO already queried, no bridge needed (naive heuristic check)
        self.assertFalse(self.skill.detect_bridge_need("Who is the CEO of Anytime AI?", []))

    def test_suggest_anchors(self):
        anchors = self.skill.suggest_anchors("Qian Chen", "User asks about Anytime AI")
        self.assertIn("Anytime AI CEO", anchors)
        self.assertIn("Anytime AI Founder", anchors)

    def test_extract_constraints(self):
        # Mock docs from "Anytime AI CEO" search
        mock_docs = [
            {"content": "Richard Wang is the CEO of Anytime AI. He previously worked at Meta and studied at Berkeley."},
            {"content": "Anytime AI is based in New York and focuses on legal AI."}
        ]
        constraints = self.skill.extract_constraints_from_docs(mock_docs)
        
        self.assertIn("Berkeley", constraints.locations)
        self.assertIn("New York", constraints.locations)
        self.assertIn("Meta", constraints.organizations)

    def test_generate_constrained_query(self):
        constraints = BridgeConstraints(
            locations=["Berkeley", "New York"],
            organizations=["Meta"]
        )
        query = self.skill.generate_constrained_query("Qian Chen", constraints)
        
        # Verify query construction
        self.assertIn("Qian Chen", query)
        self.assertIn("(Berkeley OR New York)", query)
        self.assertIn("(Meta)", query)
        self.assertIn(" AND ", query)

if __name__ == '__main__':
    unittest.main()
