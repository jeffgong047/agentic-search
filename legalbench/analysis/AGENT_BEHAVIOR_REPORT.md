# Agent Behavior Analysis

## Observations from Traces
I intercepted the internal Chains-of-Thought for representative queries (e.g., `lb_082`) to understand the Agent's decision-making.

### 1. The "Good Student" Effect (Constraint Relaxation) 🎓
- **Observation**: For queries like *"What is the likely outcome of this breach of contract case?"*, the agent generated `Filters: {}` (Empty).
- **Reason**: The User Manual (Schema) explicitly instructed: *"Only use `doc_type` if it matches [employment_agreement, ...]. If not in list, DO NOT USE."*
- **Success**: Since "breach case" is not a valid `doc_type` in our index, the Agent **correctly** refrained from hallucinating filters like `doc_type="case_law"`.
- **Impact**: Zero crashes. 100% retrieval yield.

### 2. The "Generic Query" Problem 🌫️
- **Observation**: The queries often refer to specific instances implicitly (*"this contract"*, *"this case"*) without providing unique entities (Company Name, Date).
- **Agent Response**:
  - **HyDE**: Hallucinates a *generic* legal explanation (*"In a breach case, courts look at materiality..."*).
  - **Keywords**: Extracts broad terms (*"breach", "damages", "outcome"*).
- **Result**: The search retreives generic legal documents or random case law about those topics.
- **Failure**: Without unique identifiers in the query, the agent cannot distinguish the *target* document from thousands of similar documents.

## Summary of Patterns

| Pattern | Outcome | Why |
| :--- | :--- | :--- |
| **Schema Compliance** | ✅ Success | Agent respects the "User Manual" and avoids invalid filters, fixing the crash bug. |
| **Topic Retrieval** | ✅ Success | Agent finds documents regarding correct *topics* (e.g. correctly identifies 'damages' issue). |
| **Instance Retrieval** | ❌ Failure | Queries lack specific entities ("this agreement"), making targeted retrieval impossible without metadata filters. |

## Recommendation for Improvement
To improve beyond 18% Recall, we need to **bridge the context gap**:
1. **Interactive Clarification**: If the user asks about "this contract", the Agent should detect ambiguity and ask: *"Which party or date are you referring to?"*.
2. **Reranking**: Use a Cross-Encoder to re-sort the generic results based on subtle semantic matches with the query's intent (even if keywords are broad).
