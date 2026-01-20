import dspy

class RetrievalReflectorSignature(dspy.Signature):
    """
    Reflect on a failed retrieval attempt and propose an improved query.
    Use the critique to identify missing concepts or misunderstandings.
    """
    query = dspy.InputField(desc="The user's original query")
    retrieved_docs = dspy.InputField(desc="The irrelevant documents that were retrieved")
    critique = dspy.InputField(desc="The critic's explanation of why they were irrelevant")
    history = dspy.InputField(desc="List of previous queries attempted")
    
    reflection = dspy.OutputField(desc="Analysis of the failure (e.g., 'I prioritized generic terms instead of...')")
    improved_query = dspy.OutputField(desc="A new, refined query to try next")

class RetrievalReflector(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(RetrievalReflectorSignature)
    
    def forward(self, query, retrieved_docs, critique, history):
        docs_str = "\n".join([f"- {str(d)[:200]}..." for d in retrieved_docs[:3]]) # Brief context
        history_str = str(history)
        return self.prog(query=query, retrieved_docs=docs_str, critique=critique, history=history_str)
