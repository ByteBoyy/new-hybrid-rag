import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import uvicorn

from src.models_fanout import (
    QueryRequest, QueryResponse, HealthResponse, ConfigResponse,
    Source, RetrievedDocument
)
from src.config import get_config, DEFAULT_MODE, FINAL_K
from src.retrieval.fanout_retriever import get_fanout_retriever
from src.evaluation.relevance_filter import get_relevance_filter
from src.generation.answer_generator import get_answer_generator
from src.generation.ollama_answer_generator import get_ollama_generator


# ============================================================================
# INITIALIZE APP
# ============================================================================

app = FastAPI(
    title="Wikipedia RAG API with Fan-Out Queries",
    description=(
        "Fast semantic search over 7M Wikipedia articles with fan-out query support. "
        "Dual-mode support: standard (full quality) or ultra (50% smaller)."
    ),
    version="2.1.0"
)


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_STANDARD_RETRIEVER = None
_ULTRA_RETRIEVER = None
_RELEVANCE_FILTER = None
_ANSWER_GENERATOR = None
_OLLAMA_GENERATOR = None


@app.on_event("startup")
async def startup():
    """Pre-load models at startup for fast queries."""
    global _STANDARD_RETRIEVER, _ULTRA_RETRIEVER, _RELEVANCE_FILTER, _ANSWER_GENERATOR, _OLLAMA_GENERATOR
    
    print("="*70)
    print("🚀 Starting Wikipedia RAG API with Fan-Out Queries")
    print("="*70)
    
    # Pre-load both modes (if available)
    print("\n📦 Pre-loading retrieval modes...")
    
    # Try standard
    try:
        print("\n1️⃣  Loading STANDARD mode with fan-out...")
        _STANDARD_RETRIEVER = get_fanout_retriever('standard')
    except FileNotFoundError as e:
        print(f"   ⚠️  Standard mode not available: {e}")
        _STANDARD_RETRIEVER = None
    
    # Try ultra
    try:
        print("\n2️⃣  Loading ULTRA mode with fan-out...")
        _ULTRA_RETRIEVER = get_fanout_retriever('ultra')
    except FileNotFoundError as e:
        print(f"   ⚠️  Ultra mode not available: {e}")
        _ULTRA_RETRIEVER = None
    
    # Check if at least one mode is available
    if _STANDARD_RETRIEVER is None and _ULTRA_RETRIEVER is None:
        raise RuntimeError("No retrieval mode available! Check your index paths.")
    
    # Load filter and generators
    print("\n3️⃣  Loading relevance filter...")
    _RELEVANCE_FILTER = get_relevance_filter()
    
    print("\n4️⃣  Loading OpenAI answer generator...")
    try:
        _ANSWER_GENERATOR = get_answer_generator()
    except ValueError as e:
        print(f"   ⚠️  OpenAI generator not available: {e}")
        _ANSWER_GENERATOR = None
    
    print("\n5️⃣  Loading Ollama answer generator...")
    try:
        _OLLAMA_GENERATOR = get_ollama_generator(model="qwen2.5:0.5b")
    except Exception as e:
        print(f"   ⚠️  Ollama generator not available: {e}")
        _OLLAMA_GENERATOR = None
    
    print("\n" + "="*70)
    print("✅ API Ready!")
    modes = []
    if _STANDARD_RETRIEVER: modes.append("standard")
    if _ULTRA_RETRIEVER: modes.append("ultra")
    print(f"   Retrieval modes: {', '.join(modes)}")
    print(f"   Default mode: {DEFAULT_MODE}")
    generators = []
    if _ANSWER_GENERATOR: generators.append("OpenAI")
    if _OLLAMA_GENERATOR: generators.append("Ollama")
    print(f"   Generators: {', '.join(generators) if generators else 'None'}")
    print(f"   🎯 Fan-out queries: Enabled (auto-detect)")
    print("="*70 + "\n")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Redirect to docs."""
    return RedirectResponse(url="/docs")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the Wikipedia knowledge base with optional fan-out.
    
    **New Feature: Fan-Out Queries**
    - Automatically decomposes complex queries into specialized sub-queries
    - Retrieves documents for each sub-query
    - Ranks by consensus (articles appearing in multiple queries)
    - Best for: "how-to", troubleshooting, complex multi-aspect queries
    - Not used for: simple factual queries like "what is X"
    
    **Example:**
    ```json
    {
        "query": "how do i make caramel at home",
        "mode": "ultra",
        "top_k": 5,
        "use_fanout": null  // auto-detect
    }
    ```
    
    Fan-out will generate sub-queries like:
    - "caramelization chemical process" (understanding)
    - "caramel recipe home cooking" (procedure)
    - "sugar cooking safety temperature" (safety)
    """
    # Get retriever
    mode = request.mode or DEFAULT_MODE
    
    if mode == 'standard':
        if _STANDARD_RETRIEVER is None:
            raise HTTPException(
                status_code=503,
                detail="Standard mode not available. Use 'ultra' mode or check index paths."
            )
        retriever = _STANDARD_RETRIEVER
    else:
        if _ULTRA_RETRIEVER is None:
            raise HTTPException(
                status_code=503,
                detail="Ultra mode not available. Use 'standard' mode or check index paths."
            )
        retriever = _ULTRA_RETRIEVER
    
    print(f"\n{'='*60}")
    print(f"Query: {request.query}")
    print(f"Mode: {mode.upper()}")
    print(f"Use lexical: {request.use_lexical}")
    print(f"Use fan-out: {request.use_fanout if request.use_fanout is not None else 'auto-detect'}")
    print(f"{'='*60}")
    
    try:
        # 1. Retrieve documents (with fan-out if applicable)
        retrieval_result = retriever.retrieve(
            query=request.query,
            top_k=request.top_k * 2,  # Get more for filtering
            use_lexical=request.use_lexical
            # use_fanout is auto-detected by FanOutRetriever
        )
        
        documents = retrieval_result['documents']
        fanout_info = retrieval_result.get('fanout_info', {})
        
        if not documents:
            return QueryResponse(
                answer="No relevant information found.",
                sources=[],
                confidence="low",
                retrieved=[],
                metadata={
                    'mode': mode,
                    'num_retrieved': 0,
                    'use_lexical': request.use_lexical,
                    'fanout_info': fanout_info
                }
            )
        
        print(f"  📥 Retrieved {len(documents)} documents")
        if fanout_info.get('used_fanout'):
            print(f"  🎯 Fan-out used: {fanout_info['num_fanouts']} queries")
            print(f"  🎯 Consensus articles: {fanout_info.get('consensus_articles', 0)}")
        
        # 2. Filter by relevance
        filtered = _RELEVANCE_FILTER.filter_results(request.query, documents)
        filtered = filtered[:FINAL_K]  # Keep top K
        
        print(f"  ✅ Filtered to {len(filtered)} relevant documents")
        
        # 3. Generate answer
        print(f"  🤖 Generating answer...")
        result = _ANSWER_GENERATOR.generate(request.query, filtered)
        
        # 4. Format response
        retrieved_docs = [
            RetrievedDocument(
                id=doc['id'],
                title=doc['title'],
                text=doc['text'],
                summary=doc['summary'],
                url=doc['url'],
                score=doc.get('final_score', doc.get('score', 0)),
                relevance=doc.get('relevance', 'unknown'),
                appearances=doc.get('appearances'),
                purposes=doc.get('purposes'),
                from_original=doc.get('from_original')
            )
            for doc in filtered
        ]
        
        response = QueryResponse(
            answer=result['answer'],
            sources=[
                Source(
                    title=s['title'],
                    url=s['url'],
                    score=s['score']
                )
                for s in result['sources']
            ],
            confidence=result['confidence'],
            retrieved=retrieved_docs,
            metadata={
                'mode': mode,
                'num_retrieved': len(documents),
                'num_filtered': len(filtered),
                'use_lexical': request.use_lexical,
                'reasoning': result['reasoning'],
                'fanout_info': fanout_info
            }
        )
        
        print(f"  ✅ Response ready (confidence: {result['confidence']})")
        print(f"{'='*60}\n")
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/ollama", response_model=QueryResponse)
async def query_ollama(request: QueryRequest):
    """
    Query using Ollama local model with fan-out support.
    
    Same as `/query` but uses free Ollama models.
    Fan-out queries are also generated using Ollama (completely free!).
    """
    # Check if Ollama generator is available
    if _OLLAMA_GENERATOR is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Ollama generator not available. "
                "Make sure Ollama is running: ollama serve"
            )
        )
    
    # Get retriever (same logic as /query)
    mode = request.mode or DEFAULT_MODE
    
    if mode == 'standard':
        if _STANDARD_RETRIEVER is None:
            raise HTTPException(
                status_code=503,
                detail="Standard mode not available. Use 'ultra' mode or check index paths."
            )
        retriever = _STANDARD_RETRIEVER
    else:
        if _ULTRA_RETRIEVER is None:
            raise HTTPException(
                status_code=503,
                detail="Ultra mode not available. Use 'standard' mode or check index paths."
            )
        retriever = _ULTRA_RETRIEVER
    
    print(f"\n{'='*60}")
    print(f"Query (Ollama): {request.query}")
    print(f"Mode: {mode.upper()}")
    print(f"Use lexical: {request.use_lexical}")
    print(f"Use fan-out: {request.use_fanout if request.use_fanout is not None else 'auto-detect'}")
    print(f"{'='*60}")
    
    try:
        # 1. Retrieve documents (same as /query)
        retrieval_result = retriever.retrieve(
            query=request.query,
            top_k=request.top_k * 2,
            use_lexical=request.use_lexical
        )
        
        documents = retrieval_result['documents']
        fanout_info = retrieval_result.get('fanout_info', {})
        
        if not documents:
            return QueryResponse(
                answer="No relevant information found.",
                sources=[],
                confidence="low",
                retrieved=[],
                metadata={
                    'mode': mode,
                    'num_retrieved': 0,
                    'use_lexical': request.use_lexical,
                    'generator': 'ollama',
                    'fanout_info': fanout_info
                }
            )
        
        print(f"  📥 Retrieved {len(documents)} documents")
        if fanout_info.get('used_fanout'):
            print(f"  🎯 Fan-out used: {fanout_info['num_fanouts']} queries")
            print(f"  🎯 Consensus articles: {fanout_info.get('consensus_articles', 0)}")
        
        # 2. Filter by relevance (same as /query)
        filtered = _RELEVANCE_FILTER.filter_results(request.query, documents)
        filtered = filtered[:FINAL_K]
        
        print(f"  ✅ Filtered to {len(filtered)} relevant documents")
        
        # 3. Generate answer with OLLAMA
        print(f"  🤖 Generating answer with Ollama...")
        result = _OLLAMA_GENERATOR.generate(request.query, filtered)
        
        # 4. Format response (same as /query)
        retrieved_docs = [
            RetrievedDocument(
                id=doc['id'],
                title=doc['title'],
                text=doc['text'],
                summary=doc['summary'],
                url=doc['url'],
                score=doc.get('final_score', doc.get('score', 0)),
                relevance=doc.get('relevance', 'unknown'),
                appearances=doc.get('appearances'),
                purposes=doc.get('purposes'),
                from_original=doc.get('from_original')
            )
            for doc in filtered
        ]
        
        response = QueryResponse(
            answer=result['answer'],
            sources=[
                Source(
                    title=s['title'],
                    url=s['url'],
                    score=s['score']
                )
                for s in result['sources']
            ],
            confidence=result['confidence'],
            retrieved=retrieved_docs,
            metadata={
                'mode': mode,
                'num_retrieved': len(documents),
                'num_filtered': len(filtered),
                'use_lexical': request.use_lexical,
                'reasoning': result['reasoning'],
                'generator': 'ollama',
                'model': 'qwen2.5:0.5b',
                'fanout_info': fanout_info
            }
        )
        
        print(f"  ✅ Response ready (confidence: {result['confidence']})")
        print(f"{'='*60}\n")
        
        return response
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    # Count total documents
    total_docs = 0
    if _STANDARD_RETRIEVER:
        total_docs = _STANDARD_RETRIEVER.base_retriever.semantic.index.ntotal
    elif _ULTRA_RETRIEVER:
        total_docs = _ULTRA_RETRIEVER.base_retriever.semantic.index.ntotal
    
    return HealthResponse(
        status="healthy",
        mode=DEFAULT_MODE,
        index_available={
            'standard': _STANDARD_RETRIEVER is not None,
            'ultra': _ULTRA_RETRIEVER is not None
        },
        total_documents=total_docs
    )


@app.get("/config", response_model=ConfigResponse)
async def config():
    """Get current configuration."""
    cfg = get_config(DEFAULT_MODE)
    
    return ConfigResponse(
        mode=cfg['mode'],
        dimensions=cfg['dimensions'],
        compressed=cfg['compressed'],
        top_k=FINAL_K,
        use_lexical_default=False,
        use_fanout_default=True  # Auto-detect by default
    )


@app.get("/stats")
async def stats():
    """Get detailed statistics."""
    stats = {
        'default_mode': DEFAULT_MODE,
        'modes': {}
    }
    
    if _STANDARD_RETRIEVER:
        stats['modes']['standard'] = _STANDARD_RETRIEVER.get_stats()
    
    if _ULTRA_RETRIEVER:
        stats['modes']['ultra'] = _ULTRA_RETRIEVER.get_stats()
    
    return stats


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "src.api.app_fanout:app",
        host=host,
        port=port,
        reload=reload
    )