"""
Legal Corpus Manager

Loads or generates a legal document corpus for LegalBench evaluation.

Options:
1. Synthetic: Generate realistic legal documents (fastest)
2. CUAD: Download Contract Understanding Atticus Dataset
3. Custom: Load from custom JSONL file
"""

import json
import os
from typing import List, Dict, Optional
from pathlib import Path
import random
from tqdm import tqdm


class LegalCorpusManager:
    """Manage legal document corpus for LegalBench evaluation"""

    # Document templates for synthetic generation
    DOCUMENT_TEMPLATES = {
        "employment_agreement": """EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into as of {date} between {company} ("Employer") and {employee} ("Employee").

1. POSITION AND DUTIES
Employee shall serve as {position} and perform duties as assigned by Employer.

2. COMPENSATION
Employee shall receive an annual salary of ${salary}, payable in accordance with Employer's standard payroll practices.

3. TERM
This Agreement shall commence on {start_date} and continue until terminated by either party with {notice_period} days written notice.

4. {special_clause}

5. CONFIDENTIALITY
Employee agrees to maintain strict confidentiality of all proprietary information and trade secrets.

6. GOVERNING LAW
This Agreement shall be governed by the laws of {jurisdiction}.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first written above.
""",
        "non_compete": """NON-COMPETE AGREEMENT

Employee agrees that during employment and for a period of {duration} following termination, Employee shall not:

(a) Engage in any business that competes with Employer within a {radius}-mile radius of {location};
(b) Solicit any clients or customers of Employer;
(c) Recruit or hire any employees of Employer.

This restriction is necessary to protect Employer's legitimate business interests including customer relationships, trade secrets, and confidential information.

Employee acknowledges that this restriction is reasonable in scope and duration.
""",
        "lease_agreement": """RESIDENTIAL LEASE AGREEMENT

This Lease Agreement is made on {date} between {landlord} ("Landlord") and {tenant} ("Tenant") for the property located at {address}.

1. TERM
The lease term shall be {lease_term}, commencing on {start_date}.

2. RENT
Tenant shall pay monthly rent of ${rent} due on the {due_day} of each month.

3. SECURITY DEPOSIT
Tenant has paid a security deposit of ${deposit}, to be held in accordance with {jurisdiction} law.

4. USE OF PREMISES
The premises shall be used solely for residential purposes.

5. {lease_clause}

6. TERMINATION
Either party may terminate with {notice_period} days written notice.
""",
        "sales_contract": """SALES CONTRACT

This Sales Contract is entered into on {date} between {seller} ("Seller") and {buyer} ("Buyer").

1. GOODS
Seller agrees to sell and Buyer agrees to purchase {goods_description}.

2. PRICE
The total purchase price is ${price}, payable as follows: {payment_terms}.

3. DELIVERY
Delivery shall occur on or before {delivery_date} at {delivery_location}.

4. WARRANTY
Seller warrants that the goods are {warranty_terms}.

5. RISK OF LOSS
Risk of loss passes to Buyer upon {risk_transfer_point}.

6. REMEDIES
In the event of breach, the non-breaching party shall be entitled to {remedies}.
""",
        "nda": """NON-DISCLOSURE AGREEMENT

This Non-Disclosure Agreement ("NDA") is entered into as of {date} between {party1} and {party2}.

1. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means all non-public information disclosed by either party including {info_types}.

2. OBLIGATIONS
The receiving party agrees to:
(a) Maintain confidentiality of all Confidential Information;
(b) Use Confidential Information solely for {purpose};
(c) Not disclose to third parties without prior written consent.

3. TERM
This Agreement shall remain in effect for {duration} from the date of disclosure.

4. EXCLUSIONS
Confidential Information does not include information that {exclusions}.

5. RETURN OF MATERIALS
Upon termination, all Confidential Information shall be returned or destroyed.
""",
        "settlement_agreement": """SETTLEMENT AGREEMENT AND RELEASE

This Settlement Agreement is entered into on {date} between {plaintiff} ("Plaintiff") and {defendant} ("Defendant") to resolve all claims arising from {incident_description}.

1. RECITALS
WHEREAS, Plaintiff alleges {claim_type} resulting in {damages_description};
WHEREAS, Defendant denies all liability but wishes to avoid litigation costs;

2. SETTLEMENT PAYMENT
Defendant shall pay Plaintiff the sum of ${settlement_amount} in full settlement of all claims.

3. RELEASE
Plaintiff hereby releases Defendant from any and all claims related to {incident_description}.

4. CONFIDENTIALITY
The parties agree that the terms of this settlement shall remain confidential.

5. NO ADMISSION
This Agreement shall not be construed as an admission of liability by Defendant.
""",
        "demand_letter": """DEMAND LETTER

{date}

{defendant}
{defendant_address}

Re: Demand for Payment - {case_description}

Dear {defendant}:

This office represents {plaintiff} in connection with {incident_description}.

On {incident_date}, {fact_pattern}.

As a result of your {action_type}, our client suffered {damages_list}.

DEMAND: We demand payment of ${demand_amount} within {deadline} days to resolve this matter without litigation.

If we do not receive payment by {deadline_date}, we will have no choice but to file suit seeking damages, costs, and attorney's fees.

Please contact me immediately to discuss settlement.

Sincerely,
{attorney_name}
Attorney for Plaintiff
""",
    }

    # Fill-in values for templates
    FILL_VALUES = {
        "company": ["TechCorp Inc.", "Meta Platforms Inc.", "Global Solutions LLC", "Innovation Labs Corp.", "DataSystems Inc."],
        "employee": ["Qian Chen", "John Smith", "Maria Garcia", "David Kim", "Sarah Johnson"],
        "position": ["Senior Software Engineer", "Research Scientist", "Product Manager", "Data Analyst", "Legal Counsel"],
        "salary": ["120,000", "150,000", "95,000", "180,000", "110,000"],
        "date": ["January 15, 2023", "March 1, 2023", "June 10, 2023", "September 5, 2023", "December 1, 2023"],
        "start_date": ["February 1, 2023", "April 1, 2023", "July 1, 2023", "October 1, 2023", "January 1, 2024"],
        "notice_period": ["30", "60", "90", "14", "45"],
        "jurisdiction": ["California", "New York", "Delaware", "Texas", "Washington"],
        "duration": ["two (2) years", "one (1) year", "eighteen (18) months", "three (3) years", "six (6) months"],
        "radius": ["50", "100", "25", "75", "150"],
        "location": ["San Francisco", "New York City", "Austin", "Seattle", "Boston"],
        "landlord": ["Property Management LLC", "Real Estate Holdings Inc.", "ABC Apartments", "Metro Housing Corp.", "City Rentals LLC"],
        "tenant": ["Alice Williams", "Robert Brown", "Jennifer Lee", "Michael Davis", "Lisa Anderson"],
        "address": ["123 Main St, Apt 4B", "456 Oak Avenue", "789 Elm Street", "321 Pine Road", "654 Maple Drive"],
        "lease_term": ["twelve (12) months", "six (6) months", "twenty-four (24) months", "month-to-month", "eighteen (18) months"],
        "rent": ["2,500", "1,800", "3,200", "2,000", "2,800"],
        "deposit": ["5,000", "3,600", "6,400", "4,000", "5,600"],
        "due_day": ["1st", "5th", "15th", "1st", "10th"],
        "seller": ["Manufacturing Co.", "Wholesale Distributors Inc.", "Equipment Sales LLC", "Supply Company", "Industrial Products Corp."],
        "buyer": ["Retail Operations Inc.", "Consumer Goods LLC", "Distribution Center Corp.", "Purchasing Group", "Commercial Buyers Inc."],
        "goods_description": ["500 units of Product X", "manufacturing equipment", "office furniture", "computer systems", "raw materials"],
        "price": ["50,000", "125,000", "75,000", "200,000", "95,000"],
        "payment_terms": ["50% deposit, 50% on delivery", "net 30 days", "upon delivery", "installments over 6 months", "wire transfer"],
        "delivery_date": ["March 15, 2024", "April 30, 2024", "May 20, 2024", "June 10, 2024", "July 5, 2024"],
        "delivery_location": ["Buyer's warehouse", "FOB shipping point", "Seller's facility", "designated carrier", "Buyer's premises"],
        "warranty_terms": ["free from defects", "merchantable and fit for purpose", "in conformance with specifications", "free from liens", "as described"],
        "risk_transfer_point": ["delivery to carrier", "delivery to Buyer", "inspection and acceptance", "shipment", "receipt"],
        "remedies": ["specific performance and damages", "refund of purchase price", "replacement or repair", "consequential damages", "injunctive relief"],
        "party1": ["Acme Corporation", "Innovation Labs Inc.", "Research Institute", "Technology Partners LLC", "Development Group"],
        "party2": ["Beta Solutions Inc.", "Consulting Group LLC", "Strategic Advisors", "Business Partners Corp.", "Collaborative Ventures"],
        "info_types": ["technical data, business plans, and customer information", "proprietary algorithms and trade secrets", "financial data and strategic plans", "research findings and methodologies", "product designs and specifications"],
        "purpose": ["evaluating potential business relationship", "collaborative research", "contract negotiations", "due diligence", "joint venture discussions"],
        "exclusions": ["is publicly available, was independently developed, or was rightfully received from third parties", "becomes public through no fault of recipient", "was known prior to disclosure", "is independently developed without use of Confidential Information", "must be disclosed by law"],
        "plaintiff": ["James Wilson", "Corporate Plaintiff Inc.", "Estate of John Doe", "Consumer Protection Group", "Injured Party"],
        "defendant": ["XYZ Corporation", "Individual Defendant", "Insurance Company", "Product Manufacturer", "Service Provider"],
        "incident_description": ["the motor vehicle accident on Highway 101", "breach of the employment contract", "product liability claim", "medical malpractice", "premises liability incident"],
        "claim_type": ["negligence and personal injury", "breach of contract and fraud", "product defect causing injury", "professional malpractice", "violation of consumer protection laws"],
        "damages_description": ["medical expenses of $50,000 and lost wages", "economic damages and emotional distress", "permanent injury and disability", "property damage and consequential losses", "pain and suffering"],
        "settlement_amount": ["75,000", "125,000", "250,000", "50,000", "180,000"],
        "incident_date": ["January 15, 2023", "March 20, 2023", "May 10, 2023", "August 5, 2023", "November 1, 2023"],
        "fact_pattern": ["your vehicle struck our client's car causing serious injuries", "you breached the contract by failing to deliver goods", "your defective product caused injury to our client", "your negligent conduct resulted in damages", "you failed to fulfill your obligations"],
        "action_type": ["negligent driving", "breach of contract", "negligence", "failure to warn", "wrongful conduct"],
        "damages_list": ["medical expenses, lost wages, and pain and suffering", "economic losses and reputational harm", "permanent injury and future medical costs", "property damage and consequential damages", "mental anguish and loss of enjoyment of life"],
        "demand_amount": ["100,000", "75,000", "250,000", "50,000", "150,000"],
        "deadline": ["thirty (30)", "twenty-one (21)", "fourteen (14)", "forty-five (45)", "ten (10)"],
        "deadline_date": ["March 15, 2024", "April 1, 2024", "February 28, 2024", "May 1, 2024", "March 30, 2024"],
        "attorney_name": ["Jane Attorney, Esq.", "John Lawyer, Esq.", "Mary Counsel, Esq.", "Robert Legal, Esq.", "Susan Advocate, Esq."],
        "case_description": ["Personal Injury Claim", "Breach of Contract", "Product Liability", "Professional Malpractice", "Property Damage"],
        "defendant_address": ["123 Business Blvd, Suite 500", "456 Corporate Way", "789 Industrial Park", "321 Commerce Street", "654 Trade Center"],
    }

    # Special clauses for employment agreements
    SPECIAL_CLAUSES = [
        "NON-COMPETE: Employee agrees not to engage in competitive business for 2 years within 50 miles",
        "INTELLECTUAL PROPERTY: All inventions and works created during employment belong to Employer",
        "ARBITRATION: Any disputes shall be resolved through binding arbitration",
        "AT-WILL EMPLOYMENT: Either party may terminate this employment relationship at any time",
        "BENEFITS: Employee shall be entitled to health insurance, 401(k), and paid time off per company policy",
    ]

    LEASE_CLAUSES = [
        "PETS: No pets allowed without prior written consent of Landlord",
        "MAINTENANCE: Tenant responsible for routine maintenance; Landlord responsible for major repairs",
        "SUBLETTING: Tenant may not sublet without Landlord's written consent",
        "UTILITIES: Tenant responsible for electricity, gas, and water; Landlord pays trash collection",
        "PARKING: One parking space included; additional spaces available for $50/month",
    ]

    def __init__(self, cache_dir: str = "legalbench/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_or_generate_corpus(self, source: str = "synthetic", num_docs: int = 500, output_dir: str = "legalbench/data") -> List[Dict]:
        """
        Load corpus from source or generate synthetic documents

        Args:
            source: "synthetic" | "cuad" | "custom"
            num_docs: Number of documents to generate (for synthetic)
            output_dir: Directory to save corpus

        Returns:
            List of document dictionaries
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"[Corpus] Source: {source}, Target: {num_docs} documents")

        if source == "synthetic":
            corpus = self.generate_synthetic_documents(num_docs)
        elif source == "cuad":
            corpus = self.download_cuad_dataset()
        elif source == "custom":
            corpus = self.load_custom_corpus()
        else:
            raise ValueError(f"Unknown source: {source}")

        # Save to JSONL
        output_file = output_path / "corpus.jsonl"
        self.save_corpus_jsonl(corpus, str(output_file))

        print(f"✓ Saved {len(corpus)} documents to {output_file}")
        return corpus

    def generate_synthetic_documents(self, num_docs: int = 500) -> List[Dict]:
        """
        Generate synthetic legal documents

        Distribution:
        - 200 employment agreements (40%)
        - 100 lease agreements (20%)
        - 100 sales contracts (20%)
        - 50 NDAs (10%)
        - 30 settlement agreements (6%)
        - 20 demand letters (4%)
        """
        print("[Synthetic] Generating legal documents...")

        documents = []
        doc_id_counter = 1

        # Distribution
        doc_types = [
            ("employment_agreement", 200),
            ("lease_agreement", 100),
            ("sales_contract", 100),
            ("nda", 50),
            ("settlement_agreement", 30),
            ("demand_letter", 20),
        ]

        # Ensure we generate exactly num_docs
        if num_docs != sum(count for _, count in doc_types):
            # Scale proportionally
            scale = num_docs / sum(count for _, count in doc_types)
            doc_types = [(dtype, max(1, int(count * scale))) for dtype, count in doc_types]

        for doc_type, count in doc_types:
            template = self.DOCUMENT_TEMPLATES[doc_type]

            for i in range(count):
                # Fill in template
                content = template
                fill_values = {}

                # Extract placeholders
                import re
                placeholders = re.findall(r'\{(\w+)\}', template)

                for placeholder in placeholders:
                    if placeholder in self.FILL_VALUES:
                        value = random.choice(self.FILL_VALUES[placeholder])
                        fill_values[placeholder] = value
                    elif placeholder == "special_clause":
                        fill_values[placeholder] = random.choice(self.SPECIAL_CLAUSES)
                    elif placeholder == "lease_clause":
                        fill_values[placeholder] = random.choice(self.LEASE_CLAUSES)
                    else:
                        fill_values[placeholder] = f"[{placeholder}]"

                content = content.format(**fill_values)

                # Determine domain
                if doc_type in ["employment_agreement", "non_compete"]:
                    domain = "contract_law"
                elif doc_type in ["lease_agreement"]:
                    domain = "contract_law"
                elif doc_type in ["sales_contract"]:
                    domain = "contract_law"
                elif doc_type in ["nda"]:
                    domain = "contract_law"
                elif doc_type in ["settlement_agreement", "demand_letter"]:
                    domain = "tort_law"
                else:
                    domain = "general"

                # Get jurisdiction
                jurisdiction = fill_values.get("jurisdiction", random.choice(self.FILL_VALUES["jurisdiction"]))

                doc = {
                    "doc_id": f"doc_{doc_id_counter:04d}",
                    "content": content,
                    "metadata": {
                        "domain": domain,
                        "doc_type": doc_type,
                        "jurisdiction": jurisdiction,
                        "year": random.choice([2021, 2022, 2023, 2024]),
                        "length": len(content),
                        "source": "synthetic"
                    }
                }

                documents.append(doc)
                doc_id_counter += 1

        print(f"✓ Generated {len(documents)} synthetic documents")
        return documents

    def download_cuad_dataset(self) -> List[Dict]:
        """
        Download CUAD dataset from HuggingFace

        CUAD: Contract Understanding Atticus Dataset
        500+ contracts with expert annotations
        """
        try:
            from datasets import load_dataset

            print("[CUAD] Downloading from HuggingFace...")
            dataset = load_dataset("cuad/CUAD")

            documents = []
            for idx, item in enumerate(dataset["train"]):
                doc = {
                    "doc_id": f"cuad_{idx:04d}",
                    "content": item["text"],
                    "metadata": {
                        "domain": "contract_law",
                        "doc_type": "contract",
                        "jurisdiction": "US",
                        "year": 2021,
                        "length": len(item["text"]),
                        "source": "cuad"
                    }
                }
                documents.append(doc)

            print(f"✓ Downloaded {len(documents)} documents from CUAD")
            return documents

        except Exception as e:
            print(f"✗ Failed to download CUAD: {e}")
            print("[Fallback] Generating synthetic documents instead...")
            return self.generate_synthetic_documents()

    def load_custom_corpus(self, filepath: str = "legalbench/data/custom_corpus.jsonl") -> List[Dict]:
        """Load corpus from custom JSONL file"""
        documents = []
        with open(filepath, 'r') as f:
            for line in f:
                documents.append(json.loads(line))
        return documents

    def save_corpus_jsonl(self, corpus: List[Dict], filepath: str):
        """Save corpus in JSONL format"""
        with open(filepath, 'w') as f:
            for doc in corpus:
                f.write(json.dumps(doc) + '\n')

    def load_corpus_jsonl(self, filepath: str) -> List[Dict]:
        """Load corpus from JSONL file"""
        corpus = []
        with open(filepath, 'r') as f:
            for line in f:
                corpus.append(json.loads(line))
        return corpus

    def get_corpus_statistics(self, corpus: List[Dict]) -> Dict:
        """Get statistics about the corpus"""
        stats = {
            "total_docs": len(corpus),
            "by_doc_type": {},
            "by_domain": {},
            "by_jurisdiction": {},
            "total_length": 0,
            "avg_length": 0
        }

        for doc in corpus:
            metadata = doc.get("metadata", {})

            # Count by doc type
            doc_type = metadata.get("doc_type", "unknown")
            stats["by_doc_type"][doc_type] = stats["by_doc_type"].get(doc_type, 0) + 1

            # Count by domain
            domain = metadata.get("domain", "unknown")
            stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1

            # Count by jurisdiction
            jurisdiction = metadata.get("jurisdiction", "unknown")
            stats["by_jurisdiction"][jurisdiction] = stats["by_jurisdiction"].get(jurisdiction, 0) + 1

            # Track length
            stats["total_length"] += len(doc["content"])

        stats["avg_length"] = stats["total_length"] / len(corpus) if corpus else 0

        return stats


if __name__ == "__main__":
    # Test script
    manager = LegalCorpusManager()

    # Generate synthetic corpus
    corpus = manager.load_or_generate_corpus(source="synthetic", num_docs=500)

    # Print statistics
    stats = manager.get_corpus_statistics(corpus)
    print("\n" + "="*60)
    print("CORPUS STATISTICS")
    print("="*60)
    print(f"Total documents: {stats['total_docs']}")
    print(f"Average length: {stats['avg_length']:.0f} characters")
    print(f"\nBy Document Type:")
    for doc_type, count in stats["by_doc_type"].items():
        print(f"  {doc_type}: {count}")
    print(f"\nBy Domain:")
    for domain, count in stats["by_domain"].items():
        print(f"  {domain}: {count}")
    print(f"\nBy Jurisdiction:")
    for jurisdiction, count in stats["by_jurisdiction"].items():
        print(f"  {jurisdiction}: {count}")
    print("="*60)

    # Sample documents
    print("\nSample Document:")
    print("-" * 60)
    print(corpus[0]["content"][:500] + "...")
    print(f"\nMetadata: {corpus[0]['metadata']}")
