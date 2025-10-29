from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# ============================================================================
# REQUEST MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Query request with conversation support."""
    query: str = Field(..., description="Search query", min_length=1)
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation memory (auto-generated if not provided)"
    )
    mode: Optional[str] = Field(
        None, 
        description="Retrieval mode: 'standard' (full quality) or 'ultra' (50% smaller)"
    )
    top_k: Optional[int] = Field(
        5, 
        ge=1, 
        le=20, 
        description="Number of results to return"
    )
    use_lexical: Optional[bool] = Field(
        False, 
        description="Add BM25 boost for keyword-heavy queries (slower but better for specific terms)"
    )
    use_rephrasing: Optional[bool] = Field(
        True,
        description="Enable query rephrasing using conversation context"
    )
    include_history: Optional[bool] = Field(
        False,
        description="Include conversation history in response"
    )


class ConversationHistoryRequest(BaseModel):
    """Request to get conversation history."""
    session_id: str = Field(..., description="Session ID")
    last_n: Optional[int] = Field(
        None,
        description="Return only last N messages"
    )


class ClearHistoryRequest(BaseModel):
    """Request to clear conversation history."""
    session_id: str = Field(..., description="Session ID to clear")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class Source(BaseModel):
    """Source information."""
    title: str
    url: str
    score: float


class RetrievedDocument(BaseModel):
    """Full retrieved document with text."""
    id: str
    title: str
    text: str
    summary: str
    url: str
    score: float
    relevance: Optional[str] = None  # "high", "medium", "low"


class ConversationMessage(BaseModel):
    """Single conversation message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    metadata: Optional[Dict] = None


class QueryResponse(BaseModel):
    """Complete query response with conversation support."""
    answer: str
    sources: List[Source]
    confidence: str  # "high", "medium", "low"
    retrieved: List[RetrievedDocument]
    metadata: Dict
    session_id: Optional[str] = None
    rephrasing_info: Optional[Dict] = None
    conversation_history: Optional[List[ConversationMessage]] = None


class ConversationHistoryResponse(BaseModel):
    """Conversation history response."""
    session_id: str
    history: List[ConversationMessage]
    message_count: int


class SessionSummaryResponse(BaseModel):
    """Session summary response."""
    session_id: str
    message_count: int
    started_at: Optional[str]
    last_activity: Optional[str]
    user_messages: int
    assistant_messages: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    mode: str
    index_available: Dict[str, bool]
    total_documents: int
    conversation_memory: bool
    query_rephrasing: bool


class ConfigResponse(BaseModel):
    """Current configuration."""
    mode: str
    dimensions: int
    compressed: bool
    top_k: int
    use_lexical_default: bool
    conversation_enabled: bool
    rephrasing_enabled: bool