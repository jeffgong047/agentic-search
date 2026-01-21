"""
Mock Data Generator: Creates the "Mickey Mouse" test dataset
Generates ambiguous entities to test the disambiguation capabilities
"""

from typing import List, Dict, Any
import uuid


class MockDataGenerator:
    """
    Generates mock legal/research documents with entity collision scenarios.
    Creates multiple "Mickey Mouse" entities to test disambiguation.
    """

    def __init__(self):
        """Initialize the mock data generator"""
        pass

    def generate_mickey_mouse_dataset(self) -> List[Dict[str, Any]]:
        """
        Generate the "Mickey Mouse" test dataset.

        Returns:
            List of document dicts with id, content, metadata, entities, relations
        """
        documents = []

        # === GROUP 1: Mickey Mouse - Meta Researcher (TARGET) ===
        documents.extend(self._generate_meta_researcher_docs())

        # === GROUP 2: Mickey Mouse - Shanghai Finance Lawyer (DISTRACTOR) ===
        documents.extend(self._generate_shanghai_lawyer_docs())

        # === GROUP 3: Mickey Mouse - Generic/Student (DISTRACTOR) ===
        documents.extend(self._generate_student_docs())

        # === GROUP 4: Anytime AI - High-Difficulty Scenario ===
        documents.extend(self._generate_anytime_ai_docs())

        # === GROUP 5: Related entities and noise ===
        documents.extend(self._generate_related_docs())

        return documents

    def _generate_meta_researcher_docs(self) -> List[Dict[str, Any]]:
        """Generate documents for Mickey Mouse at Meta (the target entity)"""
        docs = []

        # Doc 1: Employment Agreement
        docs.append({
            "id": f"meta_mickey_mouse_{uuid.uuid4().hex[:8]}",
            "content": """
            Employment Agreement - Meta Platforms Inc.

            Employee: Mickey Mouse
            Position: Senior Research Scientist
            Department: AI Research Lab
            Start Date: January 15, 2023
            Location: Menlo Park, California

            Non-Compete Clause (Section 8):
            Employee agrees not to engage in competitive activities with Meta Platforms
            for a period of 12 months following termination of employment. This includes
            work on social media platforms, VR/AR technologies, and AI-driven content
            recommendation systems.

            Scope: This agreement is governed by California law, which generally disfavors
            non-compete agreements. The restriction is limited to direct competitive
            activities and does not prevent Employee from working in the broader tech industry.

            Employee Signature: Mickey Mouse
            Date: January 10, 2023
            """,
            "metadata": {
                "org": "Meta",
                "year": 2023,
                "type": "Employment Agreement",
                "entities": [
                    {"id": "mickey_mouse_meta", "name": "Mickey Mouse", "type": "person"},
                    {"id": "meta_platforms", "name": "Meta Platforms Inc.", "type": "organization"}
                ],
                "relations": [
                    {"source": "mickey_mouse_meta", "target": "meta_platforms", "type": "employed_by"}
                ]
            }
        })

        # Doc 2: Research Publication
        docs.append({
            "id": f"meta_mickey_mouse_pub_{uuid.uuid4().hex[:8]}",
            "content": """
            Research Publication - Meta AI Research

            Title: "Advances in Multi-Modal Content Understanding"
            Authors: Mickey Mouse, Sarah Johnson, Michael Wong
            Affiliation: Meta AI Research Lab, Menlo Park, CA
            Published: March 2024

            Abstract: This paper presents novel approaches to multi-modal content understanding
            using transformer-based architectures. Our work builds on Meta's foundation models
            to achieve state-of-the-art results on visual question answering and image captioning
            tasks.

            Author Bio: Mickey Mouse is a Senior Research Scientist at Meta Platforms, specializing
            in computer vision and natural language processing. Prior to Meta, Dr. Chen completed
            their PhD at Stanford University.
            """,
            "metadata": {
                "org": "Meta",
                "year": 2024,
                "type": "Research Publication",
                "entities": [
                    {"id": "mickey_mouse_meta", "name": "Mickey Mouse", "type": "person"},
                    {"id": "meta_platforms", "name": "Meta Platforms Inc.", "type": "organization"},
                    {"id": "sarah_johnson", "name": "Sarah Johnson", "type": "person"},
                    {"id": "michael_wong", "name": "Michael Wong", "type": "person"}
                ],
                "relations": [
                    {"source": "mickey_mouse_meta", "target": "meta_platforms", "type": "affiliated_with"},
                    {"source": "mickey_mouse_meta", "target": "sarah_johnson", "type": "co_author"},
                    {"source": "mickey_mouse_meta", "target": "michael_wong", "type": "co_author"}
                ]
            }
        })

        # Doc 3: Internal Memo
        docs.append({
            "id": f"meta_mickey_mouse_memo_{uuid.uuid4().hex[:8]}",
            "content": """
            Internal Memo - Confidential

            To: AI Research Team
            From: Mickey Mouse, Senior Research Scientist
            Date: June 5, 2024
            Re: Non-Compete Policy Clarification

            Team,

            I've received questions about our non-compete agreements. As you know, California
            law (Business and Professions Code Section 16600) generally prohibits non-compete
            clauses except in limited circumstances (sale of business, dissolution of partnership).

            However, our employment agreements do include reasonable restrictions on:
            - Trade secret protection
            - Confidential information
            - Customer non-solicitation

            These provisions are enforceable under California law. Please consult with Legal
            if you have specific questions.

            Best,
            Mickey Mouse
            Meta AI Research Lab
            """,
            "metadata": {
                "org": "Meta",
                "year": 2024,
                "type": "Internal Memo",
                "entities": [
                    {"id": "mickey_mouse_meta", "name": "Mickey Mouse", "type": "person"},
                    {"id": "meta_platforms", "name": "Meta Platforms Inc.", "type": "organization"}
                ],
                "relations": [
                    {"source": "mickey_mouse_meta", "target": "meta_platforms", "type": "employed_by"}
                ]
            }
        })

        return docs

    def _generate_shanghai_lawyer_docs(self) -> List[Dict[str, Any]]:
        """Generate documents for Mickey Mouse - Shanghai Finance Lawyer (distractor)"""
        docs = []

        # Doc 1: Legal Opinion
        docs.append({
            "id": f"shanghai_mickey_mouse_{uuid.uuid4().hex[:8]}",
            "content": """
            Legal Opinion - Shanghai Financial District Law Firm

            Attorney: Mickey Mouse (陈谦)
            Practice Area: Corporate Finance & Securities Law
            Bar Admission: Shanghai Bar Association (2018)

            Re: Non-Compete Enforcement in Chinese Employment Law

            This opinion addresses the enforceability of non-compete clauses under Chinese
            Labor Contract Law (Article 23-24). In China, non-compete agreements are generally
            enforceable if they meet statutory requirements:

            1. Limited duration (maximum 2 years)
            2. Reasonable compensation (minimum 30% of average salary)
            3. Specified geographic scope

            Unlike California law, Chinese law explicitly permits and regulates non-compete
            agreements. Our firm has successfully enforced such agreements in Shanghai courts.

            Respectfully submitted,
            Mickey Mouse, Esq.
            Shanghai Financial District Law Firm
            """,
            "metadata": {
                "org": "Shanghai Financial District Law Firm",
                "year": 2024,
                "type": "Legal Opinion",
                "location": "Shanghai",
                "jurisdiction": "China",
                "entities": [
                    {"id": "mickey_mouse_shanghai", "name": "Mickey Mouse", "type": "person"},
                    {"id": "shanghai_law_firm", "name": "Shanghai Financial District Law Firm", "type": "organization"}
                ],
                "relations": [
                    {"source": "mickey_mouse_shanghai", "target": "shanghai_law_firm", "type": "employed_by"}
                ]
            }
        })

        # Doc 2: Case Brief
        docs.append({
            "id": f"shanghai_mickey_mouse_case_{uuid.uuid4().hex[:8]}",
            "content": """
            Case Brief - Shanghai Intermediate Court (2023)

            Case No.: (2023) 沪01民终1234号
            Attorney for Plaintiff: Mickey Mouse
            Matter: Non-Compete Agreement Enforcement

            Attorney Mickey Mouse represented a major Chinese bank in enforcing a non-compete
            agreement against a former executive who joined a competitor. The court ruled
            in favor of our client, awarding damages and injunctive relief.

            This case demonstrates the robustness of non-compete enforcement in Shanghai's
            financial sector.

            Counsel: Mickey Mouse, Shanghai Financial District Law Firm
            """,
            "metadata": {
                "org": "Shanghai Financial District Law Firm",
                "year": 2023,
                "type": "Case Brief",
                "location": "Shanghai",
                "jurisdiction": "China",
                "entities": [
                    {"id": "mickey_mouse_shanghai", "name": "Mickey Mouse", "type": "person"},
                    {"id": "shanghai_law_firm", "name": "Shanghai Financial District Law Firm", "type": "organization"}
                ],
                "relations": [
                    {"source": "mickey_mouse_shanghai", "target": "shanghai_law_firm", "type": "employed_by"}
                ]
            }
        })

        return docs

    def _generate_student_docs(self) -> List[Dict[str, Any]]:
        """Generate documents for Mickey Mouse - Student (distractor)"""
        docs = []

        docs.append({
            "id": f"student_mickey_mouse_{uuid.uuid4().hex[:8]}",
            "content": """
            Student Research Paper - UC Berkeley School of Law

            Author: Mickey Mouse
            Course: Employment Law Seminar
            Professor: Dr. Lisa Martinez
            Date: May 2022

            Title: "The Enforceability of Non-Compete Agreements: A Comparative Analysis"

            Abstract: This paper examines non-compete agreement enforcement across different
            jurisdictions, with particular focus on California's prohibition under Business
            and Professions Code Section 16600. The paper compares California's approach
            to states like New York and Massachusetts that permit reasonable non-competes.

            This is a student work-product and does not constitute legal advice.

            Student ID: 20245678
            Email: qchen@berkeley.edu
            """,
            "metadata": {
                "org": "UC Berkeley",
                "year": 2022,
                "type": "Student Paper",
                "entities": [
                    {"id": "mickey_mouse_student", "name": "Mickey Mouse", "type": "person"},
                    {"id": "uc_berkeley", "name": "UC Berkeley", "type": "organization"}
                ],
                "relations": [
                    {"source": "mickey_mouse_student", "target": "uc_berkeley", "type": "student_at"}
                ]
            }
        })

        return docs

    def _generate_related_docs(self) -> List[Dict[str, Any]]:
        """Generate related documents (co-workers, similar topics)"""
        docs = []

        # Meta co-worker
        docs.append({
            "id": f"meta_coworker_{uuid.uuid4().hex[:8]}",
            "content": """
            Employment Agreement - Meta Platforms Inc.

            Employee: Sarah Johnson
            Position: Research Scientist
            Department: AI Research Lab

            Sarah Johnson works closely with Mickey Mouse on multi-modal AI research projects.
            Both are subject to Meta's standard employment agreements including confidentiality
            and limited non-compete provisions.

            Note: While California generally prohibits non-competes, Meta's agreements focus
            on trade secret protection and customer non-solicitation, which are enforceable.
            """,
            "metadata": {
                "org": "Meta",
                "year": 2023,
                "type": "Employment Agreement",
                "entities": [
                    {"id": "sarah_johnson", "name": "Sarah Johnson", "type": "person"},
                    {"id": "meta_platforms", "name": "Meta Platforms Inc.", "type": "organization"},
                    {"id": "mickey_mouse_meta", "name": "Mickey Mouse", "type": "person"}
                ],
                "relations": [
                    {"source": "sarah_johnson", "target": "meta_platforms", "type": "employed_by"},
                    {"source": "sarah_johnson", "target": "mickey_mouse_meta", "type": "colleague"}
                ]
            }
        })

        return docs
    def _generate_anytime_ai_docs(self) -> List[Dict[str, Any]]:
        """Generate documents for Anytime AI and its CEO (Relational Bridging Scenario)"""
        docs = []

        # Doc 1: Anytime AI Press Release
        docs.append({
            "id": f"anytime_ai_launch_{uuid.uuid4().hex[:8]}",
            "content": """
            Anytime AI Announces Seed Funding for Legal GenAI
            
            Anytime AI, a startup founded by Stanford researchers, has raised $5M in seed funding.
            The company is led by CEO Dr. Richard Wang, former VP of Engineering at a major tech firm.
            Joining Dr. Wang is Dr. Mickey Mouse as Chief Scientist. 
            
            The team specializes in agentic search and retrieval for legal professionals.
            """,
            "metadata": {
                "org": "Anytime AI",
                "year": 2024,
                "type": "Press Release",
                "entities": [
                    {"id": "anytime_ai", "name": "Anytime AI", "type": "organization"},
                    {"id": "richard_wang", "name": "Richard Wang", "type": "person"},
                    {"id": "mickey_mouse_anytime", "name": "Mickey Mouse", "type": "person"}
                ],
                "relations": [
                    {"source": "mickey_mouse_anytime", "target": "anytime_ai", "type": "employed_by"},
                    {"source": "richard_wang", "target": "anytime_ai", "type": "ceo_of"}
                ]
            }
        })

        # Doc 2: CEO Richard Wang Bio (The "Bridging" Document)
        docs.append({
            "id": f"richard_wang_bio_{uuid.uuid4().hex[:8]}",
            "content": """
            Dr. Richard Wang - CEO & Founder of Anytime AI
            
            Richard Wang is a veteran of the AI industry. Before founding Anytime AI, he spent 
            five years as a Director of Engineering at Meta Platforms Inc. where he worked 
            closely with the AI Research Lab. 
            
            Richard holds a PhD from Stanford University, where he co-authored several papers 
            with Mickey Mouse on early transformer models.
            """,
            "metadata": {
                "org": "Personal Bio",
                "year": 2024,
                "type": "Biography",
                "entities": [
                    {"id": "richard_wang", "name": "Richard Wang", "type": "person"},
                    {"id": "meta_platforms", "name": "Meta Platforms Inc.", "type": "organization"},
                    {"id": "mickey_mouse_anytime", "name": "Mickey Mouse", "type": "person"},
                    {"id": "stanford", "name": "Stanford University", "type": "organization"}
                ],
                "relations": [
                    {"source": "richard_wang", "target": "meta_platforms", "type": "formerly_employed_by"},
                    {"source": "richard_wang", "target": "mickey_mouse_anytime", "type": "colleague"},
                    {"source": "richard_wang", "target": "stanford", "type": "alumnus_of"}
                ]
            }
        })

        return docs

    def generate_general_legal_docs(self, count: int = 20) -> List[Dict[str, Any]]:
        """
        Generate general legal documents (noise) for testing.

        Args:
            count: Number of noise documents to generate

        Returns:
            List of general legal documents
        """
        docs = []

        topics = [
            ("California Non-Compete Law Overview", "California", "Statute"),
            ("New York Non-Compete Requirements", "New York", "Statute"),
            ("Trade Secret Protection Act", "Federal", "Statute"),
            ("Customer Non-Solicitation Agreements", "California", "Case Law"),
            ("Employee Mobility and Innovation", "Academic", "Research")
        ]

        for i in range(min(count, len(topics) * 4)):
            topic, jurisdiction, doc_type = topics[i % len(topics)]

            docs.append({
                "id": f"general_legal_{uuid.uuid4().hex[:8]}",
                "content": f"""
                {doc_type} - {jurisdiction}

                Title: {topic}

                This document discusses {topic.lower()} in the context of {jurisdiction}
                employment law. It covers key legal principles, recent case law developments,
                and practical considerations for employers and employees.

                Key Points:
                - Legal framework and requirements
                - Enforcement mechanisms
                - Recent judicial interpretations
                - Best practices for compliance

                This is a general reference document and does not pertain to any specific
                individual or case.
                """,
                "metadata": {
                    "org": "Legal Research Database",
                    "year": 2024,
                    "type": doc_type,
                    "jurisdiction": jurisdiction,
                    "entities": []
                }
            })

        return docs


# === HELPER FUNCTION ===

def get_mock_dataset() -> List[Dict[str, Any]]:
    """
    Get the complete mock dataset for testing.

    Returns:
        List of all mock documents
    """
    generator = MockDataGenerator()

    # Generate Mickey Mouse collision dataset
    mickey_mouse_docs = generator.generate_mickey_mouse_dataset()

    # Generate general legal docs for noise
    general_docs = generator.generate_general_legal_docs(count=20)

    return mickey_mouse_docs + general_docs


if __name__ == "__main__":
    # Test the data generator
    dataset = get_mock_dataset()
    print(f"Generated {len(dataset)} mock documents")
    print(f"\nSample document:")
    print(f"ID: {dataset[0]['id']}")
    print(f"Type: {dataset[0]['metadata']['type']}")
    print(f"Entities: {len(dataset[0]['metadata'].get('entities', []))}")
