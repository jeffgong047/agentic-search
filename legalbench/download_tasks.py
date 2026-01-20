"""
LegalBench Task Downloader

Downloads or generates LegalBench tasks (162 legal reasoning queries)
from arxiv:2308.11462.

The paper describes 6 types of legal reasoning:
1. Issue-spotting: Identify legal issues in text
2. Rule-recall: Recall relevant legal rules
3. Rule-application: Apply rules to facts
4. Rule-conclusion: Determine legal conclusions
5. Interpretation: Interpret legal language
6. Rhetorical-understanding: Understand legal argumentation
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path
import requests
from tqdm import tqdm


class LegalBenchDownloader:
    """Download or generate LegalBench tasks"""

    # LegalBench categories from the paper
    CATEGORIES = [
        "issue_spotting",
        "rule_recall",
        "rule_application",
        "rule_conclusion",
        "interpretation",
        "rhetorical_understanding"
    ]

    # Legal domains
    DOMAINS = [
        "contract_law",
        "tort_law",
        "criminal_law",
        "corporate_law",
        "ip_law",
        "civil_procedure"
    ]

    def __init__(self, cache_dir: str = "legalbench/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download_tasks(self, output_dir: str, source: str = "synthetic") -> List[Dict]:
        """
        Download LegalBench tasks

        Args:
            output_dir: Directory to save tasks
            source: Source of tasks:
                - "synthetic": Generate synthetic tasks (default, fastest)
                - "huggingface": Download from HuggingFace (if available)
                - "github": Download from official GitHub repo (if available)

        Returns:
            List of task dictionaries
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"[Download] Source: {source}")

        if source == "synthetic":
            tasks = self._generate_synthetic_tasks()
        elif source == "huggingface":
            tasks = self._download_from_huggingface()
        elif source == "github":
            tasks = self._download_from_github()
        else:
            raise ValueError(f"Unknown source: {source}")

        # Save to JSONL
        output_file = output_path / "tasks.jsonl"
        self.save_tasks_jsonl(tasks, str(output_file))

        print(f"✓ Saved {len(tasks)} tasks to {output_file}")
        return tasks

    def _generate_synthetic_tasks(self, num_per_category: int = 27) -> List[Dict]:
        """
        Generate synthetic LegalBench tasks

        Args:
            num_per_category: Number of tasks per category (27 * 6 = 162 total)

        Returns:
            List of 162 synthetic tasks
        """
        print("[Synthetic] Generating tasks based on LegalBench categories...")

        tasks = []
        task_counter = 1

        # Task templates for each category
        templates = {
            "issue_spotting": [
                "Does this {doc_type} contain a {clause_type}?",
                "Identify all potential legal issues in this {doc_type}.",
                "Is there a {issue_type} issue present in this agreement?",
                "What legal concerns should be raised regarding this {doc_type}?",
                "Does this contract have provisions for {topic}?",
            ],
            "rule_recall": [
                "What is the statute of limitations for {claim_type} in {jurisdiction}?",
                "What are the elements of {tort_type} under common law?",
                "What rule applies to {legal_situation}?",
                "Under the {statute_name}, what are the requirements for {action}?",
                "What is the legal standard for {legal_concept}?",
            ],
            "rule_application": [
                "Does the defendant's conduct constitute {violation_type}?",
                "Are the facts sufficient to establish {legal_element}?",
                "Would this scenario qualify as {legal_doctrine}?",
                "Does the plaintiff have standing to bring a claim for {claim_type}?",
                "Is there sufficient evidence of {element} in this case?",
            ],
            "rule_conclusion": [
                "What is the likely outcome of this {case_type} case?",
                "Will the court grant the motion for {motion_type}?",
                "Is the defendant liable for {claim}?",
                "Should damages be awarded in this case?",
                "Will the {contractual_term} be enforced by a court?",
            ],
            "interpretation": [
                "How should the term '{legal_term}' be interpreted in this contract?",
                "What does the phrase '{contract_phrase}' mean in this context?",
                "Is the language '{ambiguous_clause}' ambiguous?",
                "What is the plain meaning of '{statute_text}'?",
                "How would a court construe '{contract_provision}'?",
            ],
            "rhetorical_understanding": [
                "What is the main argument in this legal memorandum?",
                "What legal theory supports the plaintiff's position?",
                "What counterargument does the defense raise?",
                "What policy rationale justifies this rule?",
                "What is the purpose of this legal provision?",
            ]
        }

        # Fill-in values for templates
        fill_ins = {
            "doc_type": ["employment contract", "lease agreement", "sales contract", "NDA", "shareholder agreement"],
            "clause_type": ["non-compete clause", "arbitration clause", "confidentiality provision", "indemnification clause", "force majeure clause"],
            "issue_type": ["breach of contract", "fraud", "misrepresentation", "unconscionability", "illegality"],
            "topic": ["termination", "intellectual property", "dispute resolution", "limitation of liability", "assignment"],
            "claim_type": ["breach of contract", "negligence", "fraud", "defamation", "medical malpractice"],
            "jurisdiction": ["California", "New York", "Texas", "federal law", "Delaware"],
            "tort_type": ["negligence", "intentional infliction of emotional distress", "defamation", "false imprisonment", "trespass"],
            "legal_situation": ["contract formation", "agency relationships", "corporate governance", "securities fraud", "employment discrimination"],
            "statute_name": ["Uniform Commercial Code", "Securities Exchange Act", "Fair Labor Standards Act", "Americans with Disabilities Act", "Sherman Antitrust Act"],
            "action": ["filing a derivative suit", "piercing the corporate veil", "claiming qualified immunity", "obtaining injunctive relief", "enforcing a foreign judgment"],
            "legal_concept": ["probable cause", "reasonable doubt", "good faith", "commercial reasonableness", "material breach"],
            "violation_type": ["breach of fiduciary duty", "securities fraud", "antitrust violation", "copyright infringement", "breach of contract"],
            "legal_element": ["causation", "duty of care", "reliance", "damages", "offer and acceptance"],
            "legal_doctrine": ["promissory estoppel", "res judicata", "unclean hands", "assumption of risk", "piercing the corporate veil"],
            "case_type": ["breach of contract", "personal injury", "employment discrimination", "securities fraud", "patent infringement"],
            "motion_type": ["summary judgment", "dismissal", "preliminary injunction", "class certification", "sanctions"],
            "claim": ["negligence", "breach of warranty", "unjust enrichment", "conversion", "intentional interference"],
            "contractual_term": ["non-compete clause", "liquidated damages provision", "forum selection clause", "exculpatory clause", "penalty clause"],
            "legal_term": ["reasonable time", "best efforts", "material adverse effect", "commercially reasonable", "good faith"],
            "contract_phrase": ["subject to regulatory approval", "time is of the essence", "as is", "sole discretion", "jointly and severally"],
            "ambiguous_clause": ["reasonable notice", "material breach", "substantial performance", "good cause", "best judgment"],
            "statute_text": ["knowingly and willfully", "without undue delay", "in the interest of justice", "reasonably necessary", "substantial evidence"],
            "contract_provision": ["the warranty disclaimer", "the limitation of liability", "the termination clause", "the assignment provision", "the notice requirement"],
        }

        for category in self.CATEGORIES:
            category_templates = templates[category]

            for i in range(num_per_category):
                # Cycle through templates
                template = category_templates[i % len(category_templates)]

                # Fill in template with random values
                query = template
                for placeholder, values in fill_ins.items():
                    if f"{{{placeholder}}}" in query:
                        query = query.replace(f"{{{placeholder}}}", values[i % len(values)])

                # Assign domain (cycle through domains)
                domain = self.DOMAINS[task_counter % len(self.DOMAINS)]

                # Assign difficulty
                if i < num_per_category // 3:
                    difficulty = "easy"
                elif i < 2 * num_per_category // 3:
                    difficulty = "medium"
                else:
                    difficulty = "hard"

                task = {
                    "task_id": f"lb_{task_counter:03d}",
                    "query": query,
                    "category": category,
                    "domain": domain,
                    "difficulty": difficulty
                }

                tasks.append(task)
                task_counter += 1

        print(f"✓ Generated {len(tasks)} synthetic tasks")
        return tasks

    def _download_from_huggingface(self) -> List[Dict]:
        """
        Download from HuggingFace datasets (if available)

        Note: As of implementation, LegalBench may not be on HuggingFace.
        This is a placeholder for future integration.
        """
        try:
            from datasets import load_dataset

            print("[HuggingFace] Attempting to download LegalBench dataset...")
            dataset = load_dataset("nguha/legalbench")  # Hypothetical path

            tasks = []
            for item in dataset["test"]:
                tasks.append({
                    "task_id": item["task_id"],
                    "query": item["text"],
                    "category": item["task_type"],
                    "domain": item.get("domain", "general"),
                    "difficulty": item.get("difficulty", "medium")
                })

            print(f"✓ Downloaded {len(tasks)} tasks from HuggingFace")
            return tasks

        except Exception as e:
            print(f"✗ Failed to download from HuggingFace: {e}")
            print("[Fallback] Generating synthetic tasks instead...")
            return self._generate_synthetic_tasks()

    def _download_from_github(self) -> List[Dict]:
        """
        Download from official GitHub repository (if available)

        Note: Replace with actual GitHub URL when available.
        """
        github_url = "https://raw.githubusercontent.com/HazyResearch/legalbench/main/tasks.json"

        try:
            print(f"[GitHub] Downloading from {github_url}...")
            response = requests.get(github_url, timeout=30)
            response.raise_for_status()

            data = response.json()
            tasks = []

            for item in data:
                tasks.append({
                    "task_id": item["id"],
                    "query": item["question"],
                    "category": item["type"],
                    "domain": item.get("domain", "general"),
                    "difficulty": item.get("difficulty", "medium")
                })

            print(f"✓ Downloaded {len(tasks)} tasks from GitHub")
            return tasks

        except Exception as e:
            print(f"✗ Failed to download from GitHub: {e}")
            print("[Fallback] Generating synthetic tasks instead...")
            return self._generate_synthetic_tasks()

    def save_tasks_jsonl(self, tasks: List[Dict], filepath: str):
        """Save tasks in JSONL format"""
        with open(filepath, 'w') as f:
            for task in tasks:
                f.write(json.dumps(task) + '\n')

    def load_tasks_jsonl(self, filepath: str) -> List[Dict]:
        """Load tasks from JSONL file"""
        tasks = []
        with open(filepath, 'r') as f:
            for line in f:
                tasks.append(json.loads(line))
        return tasks

    def parse_task_categories(self, tasks: List[Dict]) -> Dict[str, List[str]]:
        """
        Group tasks by category

        Returns:
            Dict mapping category name to list of task IDs
        """
        categories = {cat: [] for cat in self.CATEGORIES}

        for task in tasks:
            category = task["category"]
            if category in categories:
                categories[category].append(task["task_id"])

        return categories

    def get_task_statistics(self, tasks: List[Dict]) -> Dict:
        """Get statistics about the task distribution"""
        stats = {
            "total_tasks": len(tasks),
            "by_category": {},
            "by_domain": {},
            "by_difficulty": {}
        }

        for task in tasks:
            # Count by category
            cat = task["category"]
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

            # Count by domain
            domain = task["domain"]
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1

            # Count by difficulty
            diff = task["difficulty"]
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1

        return stats


if __name__ == "__main__":
    # Test script
    downloader = LegalBenchDownloader()

    # Generate synthetic tasks
    tasks = downloader.download_tasks("legalbench/data", source="synthetic")

    # Print statistics
    stats = downloader.get_task_statistics(tasks)
    print("\n" + "="*60)
    print("TASK STATISTICS")
    print("="*60)
    print(f"Total tasks: {stats['total_tasks']}")
    print(f"\nBy Category:")
    for cat, count in stats["by_category"].items():
        print(f"  {cat}: {count}")
    print(f"\nBy Domain:")
    for domain, count in stats["by_domain"].items():
        print(f"  {domain}: {count}")
    print(f"\nBy Difficulty:")
    for diff, count in stats["by_difficulty"].items():
        print(f"  {diff}: {count}")
    print("="*60)

    # Sample tasks
    print("\nSample Tasks:")
    for i, task in enumerate(tasks[:5], 1):
        print(f"\n{i}. [{task['category']}] {task['query']}")
        print(f"   Domain: {task['domain']}, Difficulty: {task['difficulty']}")
