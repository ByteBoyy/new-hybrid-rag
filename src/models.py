# src/models.py
"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

# ============================================================================
# REQUEST MODELS
# ============================================================================

class QueryRequest(BaseModel):
    """Query request with simple options."""
    query: str = Field(..., description="Search query", min_length=1)
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


class QueryResponse(BaseModel):
    """Complete query response."""
    answer: str
    sources: List[Source]
    confidence: str  # "high", "medium", "low"
    retrieved: List[RetrievedDocument]
    metadata: Dict


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    mode: str
    index_available: Dict[str, bool]
    total_documents: int


class ConfigResponse(BaseModel):
    """Current configuration."""
    mode: str
    dimensions: int
    compressed: bool
    top_k: int
    use_lexical_default: bool
