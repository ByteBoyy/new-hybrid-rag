# src/api/app_with_conversation.py
"""
Updated FastAPI app with conversation memory and query rephrasing.
"""
import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import uvicorn
from typing import Optional

# Import updated models
from src.models_updated import (
    QueryRequest, QueryResponse, HealthResponse, ConfigResponse,
    Source, RetrievedDocument, ConversationMessage,
    ConversationHistoryRequest, ConversationHistoryResponse,
    ClearHistoryRequest, SessionSummaryResponse
)
from src.config import get_config, DEFAULT_MODE, FINAL_K
from src.retrieval.hybrid_retriever import get_hybrid_retriever
from src.evaluation.relevance_filter import get_relevance_filter
from src.generation.answer_generator import get_answer_generator
from src.generation.ollama_answer_generator import get_ollama_generator

# Import new conversation components
from src.conversation.memory import get_conversation_memory
from src.conversation.query_rephraser import get_query_rephraser


# ============================================================================
# INITIALIZE APP
# ============================================================================

app = FastAPI(
    title="Wikipedia RAG API with Conversation Memory",
    description=(
        "Fast semantic search over 7M Wikipedia articles with conversation memory and query rephrasing. "
        "Dual-mode support: standard (full quality) or ultra (50% smaller)."
    ),
    version="3.0.0"
)


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

_STANDARD_RETRIEVER = None
_ULTRA_RETRIEVER = None
_RELEVANCE_FILTER = None
_ANSWER_GENERATOR = None
_OLLAMA_GENERATOR = None
_CONVERSATION_MEMORY = None
_QUERY_REPHRASER = None
_OLLAMA_QUERY_REPHRASER = None
_OLLAMA_QUERY_REPHRASER = None


@app.on_event("startup")
async def startup():
    """Pre-load models at startup."""
    global _STANDARD_RETRIEVER, _ULTRA_RETRIEVER, _RELEVANCE_FILTER
    global _ANSWER_GENERATOR, _OLLAMA_GENERATOR
    global _CONVERSATION_MEMORY, _QUERY_REPHRASER, _OLLAMA_QUERY_REPHRASER, _OLLAMA_QUERY_REPHRASER
    
    print("="*70)
    print("🚀 Starting Wikipedia RAG API with Conversation Memory")
    print("="*70)
    
    # Load retrievers
    print("\n📦 Pre-loading retrieval modes...")
    
    try:
        print("\n1️⃣  Loading STANDARD mode (768D, full quality)...")
        _STANDARD_RETRIEVER = get_hybrid_retriever('standard')
    except FileNotFoundError as e:
        print(f"   ⚠️  Standard mode not available: {e}")
        _STANDARD_RETRIEVER = None
    
    try:
        print("\n2️⃣  Loading ULTRA mode (192D, 50% smaller)...")
        _ULTRA_RETRIEVER = get_hybrid_retriever('ultra')
    except FileNotFoundError as e:
        print(f"   ⚠️  Ultra mode not available: {e}")
        _ULTRA_RETRIEVER = None
    
    if _STANDARD_RETRIEVER is None and _ULTRA_RETRIEVER is None:
        raise RuntimeError("No retrieval mode available! Check your index paths.")
    
    # Load filter
    print("\n3️⃣  Loading relevance filter...")
    _RELEVANCE_FILTER = get_relevance_filter()
    
    # Load generators
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
    
    # Load conversation components
    print("\n6️⃣  Loading conversation memory...")
    _CONVERSATION_MEMORY = get_conversation_memory(max_history=10)
    print("   ✅ Conversation memory ready")
    
    print("\n7️⃣  Loading query rephraser...")
    _QUERY_REPHRASER = get_query_rephraser()
    print("   ✅ Query rephraser ready ")
    
    
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
    print(f"   🧠 Conversation memory: Enabled")
    print(f"   🔄 Query rephrasing (OpenAI): {'Enabled' if _QUERY_REPHRASER else 'Disabled'}")
    print(f"   🔄 Query rephrasing (Ollama): {'Enabled' if _OLLAMA_QUERY_REPHRASER else 'Disabled'}")
    print("="*70 + "\n")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_or_create_session_id(session_id: Optional[str]) -> str:
    """Get existing session ID or create new one."""
    if session_id:
        return session_id
    return str(uuid.uuid4())


def process_query_with_conversation(
    request: QueryRequest,
    retriever,
    generator,
    generator_type: str = "openai",
    rephraser_type: str = "openai"
) -> QueryResponse:
    """
    Process query with conversation memory and rephrasing.
    
    Args:
        request: Query request
        retriever: Hybrid retriever instance
        generator: Answer generator instance
        generator_type: 'openai' or 'ollama'
        rephraser_type: 'openai' or 'ollama' (which rephraser to use)
    
    Returns:
        QueryResponse with conversation support
    """
    # Get or create session ID
    session_id = get_or_create_session_id(request.session_id)
    
    # Get conversation history
    conversation_history = _CONVERSATION_MEMORY.get_context_window(
        session_id, 
        window_size=4
    )
    
    # Rephrase query if needed
    rephrasing_info = None
    query_to_use = request.query
    
    if request.use_rephrasing and conversation_history:
        # Choose rephraser
        if rephraser_type == "ollama" and _OLLAMA_QUERY_REPHRASER:
            rephraser = _OLLAMA_QUERY_REPHRASER
            print(f"  🔄 Rephrasing with Ollama...")
        elif _QUERY_REPHRASER:
            rephraser = _QUERY_REPHRASER
            print(f"  🔄 Rephrasing...")
        else:
            # No rephraser available
            print(f"  ⚠️  No rephraser available, using original query")
            rephraser = None
        
        if rephraser:
            rephrasing_result = rephraser.rephrase(
                request.query,
                conversation_history
            )
            
            rephrasing_info = rephrasing_result
            query_to_use = rephrasing_result['rephrased_query']
            
            if rephrasing_result['needs_context']:
                print(f"     Original: {request.query}")
                print(f"     Rephrased: {query_to_use}")
                print(f"     Method: {rephrasing_result['method']}")
    
    # Store user query in memory
    _CONVERSATION_MEMORY.add_message(
        session_id=session_id,
        role='user',
        content=request.query,
        metadata={'rephrased_query': query_to_use} if rephrasing_info else None
    )
    
    # Retrieve documents
    documents = retriever.retrieve(
        query=query_to_use,  # Use rephrased query
        top_k=request.top_k * 2,
        use_lexical=request.use_lexical
    )
    
    if not documents:
        answer = "No relevant information found."
        response_data = QueryResponse(
            answer=answer,
            sources=[],
            confidence="low",
            retrieved=[],
            metadata={
                'mode': request.mode or DEFAULT_MODE,
                'num_retrieved': 0,
                'use_lexical': request.use_lexical,
                'generator': generator_type
            },
            session_id=session_id,
            rephrasing_info=rephrasing_info
        )
        
        # Store assistant response
        _CONVERSATION_MEMORY.add_message(
            session_id=session_id,
            role='assistant',
            content=answer
        )
        
        return response_data
    
    print(f"  📥 Retrieved {len(documents)} documents")
    
    # Filter by relevance
    filtered = _RELEVANCE_FILTER.filter_results(query_to_use, documents)
    filtered = filtered[:FINAL_K]
    
    print(f"  ✅ Filtered to {len(filtered)} relevant documents")
    
    # Generate answer
    print(f"  🤖 Generating answer with {generator_type}...")
    result = generator.generate(query_to_use, filtered)
    
    # Format retrieved documents
    retrieved_docs = [
        RetrievedDocument(
            id=doc['id'],
            title=doc['title'],
            text=doc['text'],
            summary=doc['summary'],
            url=doc['url'],
            score=doc.get('final_score', doc.get('score', 0)),
            relevance=doc.get('relevance', 'unknown')
        )
        for doc in filtered
    ]
    
    # Build response
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
            'mode': request.mode or DEFAULT_MODE,
            'num_retrieved': len(documents),
            'num_filtered': len(filtered),
            'use_lexical': request.use_lexical,
            'reasoning': result['reasoning'],
            'generator': generator_type
        },
        session_id=session_id,
        rephrasing_info=rephrasing_info
    )
    
    # Add conversation history if requested
    if request.include_history:
        history = _CONVERSATION_MEMORY.get_history(session_id)
        response.conversation_history = [
            ConversationMessage(
                role=msg['role'],
                content=msg['content'],
                timestamp=msg['timestamp'],
                metadata=msg.get('metadata')
            )
            for msg in history
        ]
    
    # Store assistant response in memory
    _CONVERSATION_MEMORY.add_message(
        session_id=session_id,
        role='assistant',
        content=result['answer'],
        metadata={
            'confidence': result['confidence'],
            'sources': result['sources']
        }
    )
    
    print(f"  ✅ Response ready (confidence: {result['confidence']})")
    print(f"  💾 Session: {session_id}")
    
    return response


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
    Query with OpenAI and conversation memory.
    
    **New Features:**
    - 🧠 Conversation memory: Maintains chat history per session
    - 🔄 Query rephrasing: Converts follow-up questions to standalone queries
    - 📝 Session tracking: Automatic session ID generation
    
    **Example:**
    ```json
    {
        "query": "What is photosynthesis?",
        "session_id": "optional-session-id",
        "mode": "ultra",
        "top_k": 5,
        "use_rephrasing": true,
        "include_history": false
    }
    ```
    
    **Follow-up Example:**
    ```json
    {
        "query": "How does it work in plants?",
        "session_id": "same-session-id",
        "use_rephrasing": true
    }
    ```
    → Will be rephrased to: "How does photosynthesis work in plants?"
    """
    if _ANSWER_GENERATOR is None:
        raise HTTPException(
            status_code=503,
            detail="OpenAI generator not available. Check OPENAI_API_KEY."
        )
    
    mode = request.mode or DEFAULT_MODE
    retriever = _STANDARD_RETRIEVER if mode == 'standard' else _ULTRA_RETRIEVER
    
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail=f"{mode.capitalize()} mode not available."
        )
    
    print(f"\n{'='*60}")
    print(f"Query: {request.query}")
    print(f"Session: {request.session_id or 'new'}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*60}")
    
    try:
        return process_query_with_conversation(
            request, 
            retriever, 
            _ANSWER_GENERATOR,
            generator_type="openai",
            rephraser_type="openai"  # Use OpenAI for rephrasing
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/ollama", response_model=QueryResponse)
async def query_ollama(request: QueryRequest):
    """
    Query with Ollama and conversation memory.
    
    Same as `/query` but uses free local Ollama models.
    Supports all conversation features (memory + rephrasing).
    """
    if _OLLAMA_GENERATOR is None:
        raise HTTPException(
            status_code=503,
            detail="Ollama generator not available. Make sure Ollama is running."
        )
    
    mode = request.mode or DEFAULT_MODE
    retriever = _STANDARD_RETRIEVER if mode == 'standard' else _ULTRA_RETRIEVER
    
    if retriever is None:
        raise HTTPException(
            status_code=503,
            detail=f"{mode.capitalize()} mode not available."
        )
    
    print(f"\n{'='*60}")
    print(f"Query (Ollama): {request.query}")
    print(f"Session: {request.session_id or 'new'}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*60}")
    
    try:
        return process_query_with_conversation(
            request, 
            retriever, 
            _OLLAMA_GENERATOR,
            generator_type="ollama",
            rephraser_type="ollama"  # Use Ollama for rephrasing (FREE!)
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(request: ConversationHistoryRequest):
    """
    Get conversation history for a session.
    
    **Example:**
    ```json
    {
        "session_id": "your-session-id",
        "last_n": 10
    }
    ```
    """
    history = _CONVERSATION_MEMORY.get_history(
        request.session_id,
        last_n=request.last_n
    )
    
    return ConversationHistoryResponse(
        session_id=request.session_id,
        history=[
            ConversationMessage(
                role=msg['role'],
                content=msg['content'],
                timestamp=msg['timestamp'],
                metadata=msg.get('metadata')
            )
            for msg in history
        ],
        message_count=len(history)
    )


@app.post("/conversation/clear")
async def clear_conversation(request: ClearHistoryRequest):
    """
    Clear conversation history for a session.
    
    **Example:**
    ```json
    {
        "session_id": "your-session-id"
    }
    ```
    """
    _CONVERSATION_MEMORY.clear_session(request.session_id)
    
    return {
        "status": "success",
        "message": f"Cleared history for session {request.session_id}"
    }


@app.get("/conversation/sessions")
async def list_sessions():
    """Get all active session IDs."""
    sessions = _CONVERSATION_MEMORY.get_all_sessions()
    
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }


@app.get("/conversation/session/{session_id}", response_model=SessionSummaryResponse)
async def get_session_summary(session_id: str):
    """Get summary of a session."""
    summary = _CONVERSATION_MEMORY.get_session_summary(session_id)
    
    return SessionSummaryResponse(
        session_id=summary['session_id'],
        message_count=summary['message_count'],
        started_at=summary['started_at'],
        last_activity=summary['last_activity'],
        user_messages=summary.get('user_messages', 0),
        assistant_messages=summary.get('assistant_messages', 0)
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check with conversation features."""
    total_docs = 0
    if _STANDARD_RETRIEVER:
        total_docs = _STANDARD_RETRIEVER.semantic.index.ntotal
    elif _ULTRA_RETRIEVER:
        total_docs = _ULTRA_RETRIEVER.semantic.index.ntotal
    
    return HealthResponse(
        status="healthy",
        mode=DEFAULT_MODE,
        index_available={
            'standard': _STANDARD_RETRIEVER is not None,
            'ultra': _ULTRA_RETRIEVER is not None
        },
        total_documents=total_docs,
        conversation_memory=_CONVERSATION_MEMORY is not None,
        query_rephrasing=_QUERY_REPHRASER is not None
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
        conversation_enabled=True,
        rephrasing_enabled=True
    )


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "src.api.app_with_conversation:app",
        host=host,
        port=port,
        reload=reload
    )