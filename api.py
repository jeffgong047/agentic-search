"""
FastAPI Mini-Service for Agentic Search
Provides REST API endpoints for the retrieval system
"""

import os
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import get_config, update_config
from core.data_structures import SearchResult
from core.backends import create_best_backend
from core.backends_wrapper import BackendAgenticSearchEngine
from core.mock_data import get_mock_dataset

import dspy


# ====================================================================
# Request/Response Models
# ====================================================================

class SearchRequest(BaseModel):
    """Search request payload"""
    query: str = Field(..., description="The search query", min_length=1)
    top_k: Optional[int] = Field(5, description="Number of results to return", ge=1, le=50)
    use_dspy: Optional[bool] = Field(True, description="Use DSPy signatures")
    use_novelty_circuit: Optional[bool] = Field(True, description="Use novelty circuit breaker")
    use_memory: Optional[bool] = Field(True, description="Use negative memory evolution")


class SearchResultResponse(BaseModel):
    """Individual search result"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source_index: str


class SearchResponse(BaseModel):
    """Search response payload"""
    query: str
    results: List[SearchResultResponse]
    total_results: int
    iterations: int
    novelty_score: float


class IndexRequest(BaseModel):
    """Document indexing request"""
    documents: List[Dict[str, Any]] = Field(..., description="Documents to index")


class IndexResponse(BaseModel):
    """Indexing response"""
    indexed_count: int
    message: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    backend_type: str
    document_count: int
    config: Dict[str, Any]


class ConfigUpdateRequest(BaseModel):
    """Configuration update request"""
    use_dspy_signatures: Optional[bool] = None
    use_novelty_circuit: Optional[bool] = None
    use_negative_memory: Optional[bool] = None
    use_cascade_recall: Optional[bool] = None
    novelty_epsilon: Optional[float] = None
    max_iterations: Optional[int] = None


# ====================================================================
# Application State
# ====================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.backend = None
        self.search_engine = None
        self.is_initialized = False


app_state = AppState()


# ====================================================================
# Lifespan Management
# ====================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - initialize on startup"""
    print("[API] Starting Agentic Search Service...")
    
    # Check for OpenAI API key
    if "OPENAI_API_KEY" not in os.environ:
        print("[API] Warning: OPENAI_API_KEY not set. Some features will be limited.")
    else:
        # Initialize DSPy
        config = get_config()
        lm = dspy.OpenAI(model=config.LLM_MODEL, max_tokens=1000)
        dspy.settings.configure(lm=lm)
    
    # Create backend
    es_config = None
    if os.environ.get("ES_HOST"):
        es_config = {
            "host": os.environ.get("ES_HOST", "localhost"),
            "port": int(os.environ.get("ES_PORT", 9200)),
            "index": os.environ.get("ES_INDEX", "documents")
        }
    
    app_state.backend = create_best_backend(es_config=es_config)
    
    # Load mock data if AUTO_LOAD_MOCK is set
    if os.environ.get("AUTO_LOAD_MOCK", "true").lower() == "true":
        print("[API] Loading mock dataset...")
        documents = get_mock_dataset()
        app_state.backend.index_documents(documents)
        app_state.search_engine = BackendAgenticSearchEngine(app_state.backend)
        app_state.is_initialized = True
        print(f"[API] Loaded {len(documents)} documents")
    
    print("[API] Service ready!")
    
    yield
    
    # Cleanup
    print("[API] Shutting down...")


# ====================================================================
# FastAPI Application
# ====================================================================

app = FastAPI(
    title="Agentic Search API",
    description="High-SNR Agentic RAG with DSPy + LangGraph",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================================================================
# API Endpoints
# ====================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "service": "Agentic Search API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    config = get_config()
    
    stats = app_state.backend.get_stats() if app_state.backend else {}
    
    return HealthResponse(
        status="healthy" if app_state.is_initialized else "initializing",
        backend_type=stats.get("backend_type", "unknown"),
        document_count=stats.get("vector_size", 0) or stats.get("document_count", 0),
        config={
            "use_dspy_signatures": config.USE_DSPY_SIGNATURES,
            "use_novelty_circuit": config.USE_NOVELTY_CIRCUIT,
            "use_negative_memory": config.USE_NEGATIVE_MEMORY,
            "use_cascade_recall": config.USE_CASCADE_RECALL,
            "novelty_epsilon": config.NOVELTY_EPSILON,
            "max_iterations": config.MAX_ITERATIONS
        }
    )


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Execute agentic search
    
    This runs the full agentic pipeline:
    1. DSPy orchestrator generates search plan
    2. Parallel tri-index search (Vector + BM25 + Graph)
    3. Cascade filtering and reranking
    4. Novelty-based loop control
    """
    if not app_state.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Call POST /index first."
        )
    
    # Apply per-request config overrides
    config = get_config()
    original_config = {
        "USE_DSPY_SIGNATURES": config.USE_DSPY_SIGNATURES,
        "USE_NOVELTY_CIRCUIT": config.USE_NOVELTY_CIRCUIT,
        "USE_NEGATIVE_MEMORY": config.USE_NEGATIVE_MEMORY
    }
    
    try:
        if request.use_dspy is not None:
            update_config(USE_DSPY_SIGNATURES=request.use_dspy)
        if request.use_novelty_circuit is not None:
            update_config(USE_NOVELTY_CIRCUIT=request.use_novelty_circuit)
        if request.use_memory is not None:
            update_config(USE_NEGATIVE_MEMORY=request.use_memory)
        
        # Execute search
        final_state = app_state.search_engine.search(request.query)
        
        # Build response
        results = [
            SearchResultResponse(
                id=r.id,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
                source_index=r.source_index
            )
            for r in final_state["retrieved_docs"][:request.top_k]
        ]
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            iterations=final_state["step_count"],
            novelty_score=final_state["novelty_score"]
        )
    
    finally:
        # Restore original config
        for key, value in original_config.items():
            update_config(**{key: value})


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """
    Index documents into the search backend
    
    Documents should have:
    - id: Unique identifier
    - content: Text content
    - metadata: Dict with optional fields (org, type, year, entities, relations)
    """
    if not app_state.backend:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    try:
        app_state.backend.index_documents(request.documents)
        
        # Initialize search engine if not already done
        if not app_state.search_engine:
            app_state.search_engine = BackendAgenticSearchEngine(app_state.backend)
        
        app_state.is_initialized = True
        
        return IndexResponse(
            indexed_count=len(request.documents),
            message=f"Successfully indexed {len(request.documents)} documents"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/index/mock", response_model=IndexResponse)
async def index_mock_data():
    """Load the mock dataset (Mickey Mouse disambiguation test)"""
    if not app_state.backend:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    documents = get_mock_dataset()
    app_state.backend.index_documents(documents)
    
    if not app_state.search_engine:
        app_state.search_engine = BackendAgenticSearchEngine(app_state.backend)
    
    app_state.is_initialized = True
    
    return IndexResponse(
        indexed_count=len(documents),
        message=f"Loaded mock dataset with {len(documents)} documents"
    )


@app.put("/config", response_model=Dict[str, Any])
async def update_configuration(request: ConfigUpdateRequest):
    """Update system configuration (feature flags, parameters)"""
    updates = {}
    
    if request.use_dspy_signatures is not None:
        update_config(USE_DSPY_SIGNATURES=request.use_dspy_signatures)
        updates["USE_DSPY_SIGNATURES"] = request.use_dspy_signatures
    
    if request.use_novelty_circuit is not None:
        update_config(USE_NOVELTY_CIRCUIT=request.use_novelty_circuit)
        updates["USE_NOVELTY_CIRCUIT"] = request.use_novelty_circuit
    
    if request.use_negative_memory is not None:
        update_config(USE_NEGATIVE_MEMORY=request.use_negative_memory)
        updates["USE_NEGATIVE_MEMORY"] = request.use_negative_memory
    
    if request.use_cascade_recall is not None:
        update_config(USE_CASCADE_RECALL=request.use_cascade_recall)
        updates["USE_CASCADE_RECALL"] = request.use_cascade_recall
    
    if request.novelty_epsilon is not None:
        update_config(NOVELTY_EPSILON=request.novelty_epsilon)
        updates["NOVELTY_EPSILON"] = request.novelty_epsilon
    
    if request.max_iterations is not None:
        update_config(MAX_ITERATIONS=request.max_iterations)
        updates["MAX_ITERATIONS"] = request.max_iterations
    
    return {"updated": updates, "message": "Configuration updated"}


@app.get("/stats", response_model=Dict[str, Any])
async def get_stats():
    """Get backend statistics"""
    if not app_state.backend:
        raise HTTPException(status_code=503, detail="Backend not initialized")
    
    return app_state.backend.get_stats()


@app.post("/test/mickey-mouse", response_model=Dict[str, Any])
async def run_disambiguation_test():
    """
    Run the canonical Mickey Mouse disambiguation test
    
    Expected: Should retrieve ONLY documents about Mickey Mouse (Meta Researcher)
    Should AVOID: Mickey Mouse (Shanghai Lawyer), Mickey Mouse (Student)
    """
    if not app_state.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service not initialized. Call POST /index/mock first."
        )
    
    query = "Did Mickey Mouse at Meta sign a non-compete agreement?"
    final_state = app_state.search_engine.search(query)
    
    results = final_state["retrieved_docs"]
    
    # Analyze results
    meta_count = sum(1 for r in results if r.metadata.get("org") == "Meta")
    shanghai_count = sum(1 for r in results if "Shanghai" in r.metadata.get("org", ""))
    student_count = sum(1 for r in results if r.metadata.get("org") == "UC Berkeley")
    
    precision = meta_count / len(results) if results else 0
    
    return {
        "query": query,
        "total_results": len(results),
        "iterations": final_state["step_count"],
        "novelty_score": final_state["novelty_score"],
        "breakdown": {
            "meta_target": meta_count,
            "shanghai_distractor": shanghai_count,
            "student_distractor": student_count,
            "other": len(results) - meta_count - shanghai_count - student_count
        },
        "precision": precision,
        "test_passed": precision >= 0.8,
        "results": [
            {
                "id": r.id,
                "org": r.metadata.get("org"),
                "score": r.score,
                "preview": r.content[:100] + "..."
            }
            for r in results
        ]
    }


# ====================================================================
# Main Entry Point
# ====================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=os.environ.get("RELOAD", "false").lower() == "true"
    )

