import dspy

class RetrievalCriticSignature(dspy.Signature):
    """
    Evaluate the relevance of retrieved documents to the user query.
    Act as a strict judge. Scores below 0.7 indicate failure.
    """
    query = dspy.InputField(desc="The user's original legal query")
    retrieved_docs = dspy.InputField(desc="The list of documents retrieved by the search engine")
    
    critique = dspy.OutputField(desc="Explanation of why the documents are relevant or irrelevant")
    relevance_score = dspy.OutputField(desc="A score between 0.0 and 1.0 indicating relevance")

class RetrievalCritic(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(RetrievalCriticSignature)
    
    def forward(self, query, retrieved_docs):
        # Convert docs list to string for the prompt
        docs_str = "\n".join([f"- {d}" for d in retrieved_docs[:5]]) # Only critique top 5
        return self.prog(query=query, retrieved_docs=docs_str)
