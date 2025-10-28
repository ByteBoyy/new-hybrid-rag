# src/config.py
"""
Simplified configuration for clean RAG system.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "index"

# Standard mode (full quality, 2.2 GB)
STANDARD_INDEX = DATA_DIR / "wikipedia.index"
STANDARD_METADATA = DATA_DIR / "wikipedia.pkl"

# Ultra-compressed mode (50% smaller, 1.1 GB)
ULTRA_DIR = Path("/Users/omarashraf/Downloads/hybrid-rag/ultra_compressed_208k_p6")
ULTRA_INDEX = ULTRA_DIR / "wikipedia_ultra.index"
ULTRA_METADATA = ULTRA_DIR / "metadata_ultra_enhanced.pkl"
ULTRA_PCA = ULTRA_DIR / "pca_192.pkl"

# ============================================================================
# MODELS
# ============================================================================

EMBEDDING_MODEL = "BAAI/bge-base-en"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================================
# RETRIEVAL SETTINGS
# ============================================================================

# Default mode ('standard' or 'ultra')
DEFAULT_MODE = os.getenv("RAG_MODE", "ultra")

# Number of results to retrieve
TOP_K = 10  # Retrieve more, return best 5 after filtering

# Final results to return
FINAL_K = 5

# Semantic search settings
SEMANTIC_WEIGHT = 1.0  # Primary method

# Optional lexical boost (only when use_lexical=True)
LEXICAL_WEIGHT = 0.3   # Light boost for keyword matches
LEXICAL_TOP_K = 20     # How many BM25 results to consider

# Ultra-compressed settings
ULTRA_NPROBE = 20  # Higher = better quality, slower (10-50)

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Lightweight relevance filtering
MIN_RELEVANCE_SCORE = 0.2  # Lower threshold (we do less evaluation)
USE_RELEVANCE_FILTER = True  # Quick relevance check

# ============================================================================
# GENERATION SETTINGS
# ============================================================================

TEMPERATURE = 0.2
MAX_TOKENS = 1024

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(mode: str = None) -> dict:
    """
    Get configuration for specified mode.
    
    Args:
        mode: 'standard', 'ultra', or None (uses DEFAULT_MODE)
    
    Returns:
        Configuration dictionary
    """
    mode = mode or DEFAULT_MODE
    
    if mode == 'ultra':
        return {
            'mode': 'ultra',
            'index_path': str(ULTRA_INDEX),
            'metadata_path': str(ULTRA_METADATA),
            'pca_path': str(ULTRA_PCA),
            'dimensions': 192,
            'nprobe': ULTRA_NPROBE,
            'compressed': True
        }
    else:
        return {
            'mode': 'standard',
            'index_path': str(STANDARD_INDEX),
            'metadata_path': str(STANDARD_METADATA),
            'dimensions': 768,
            'compressed': False
        }


def get_retrieval_config() -> dict:
    """Get retrieval-specific configuration."""
    return {
        'top_k': TOP_K,
        'final_k': FINAL_K,
        'semantic_weight': SEMANTIC_WEIGHT,
        'lexical_weight': LEXICAL_WEIGHT,
        'lexical_top_k': LEXICAL_TOP_K,
        'min_relevance': MIN_RELEVANCE_SCORE,
        'use_filter': USE_RELEVANCE_FILTER
    }


def get_generation_config() -> dict:
    """Get generation-specific configuration."""
    return {
        'model': OPENAI_MODEL,
        'api_key': OPENAI_API_KEY,
        'temperature': TEMPERATURE,
        'max_tokens': MAX_TOKENS
    }


# Quick validation on import
if DEFAULT_MODE == 'ultra' and not ULTRA_INDEX.exists():
    print(f"⚠️  Warning: Ultra mode selected but index not found at {ULTRA_INDEX}")
    print(f"   Falling back to standard mode")
    DEFAULT_MODE = 'standard'
