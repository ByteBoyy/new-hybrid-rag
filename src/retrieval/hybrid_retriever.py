# src/retrieval/hybrid_retriever.py
"""
Simple hybrid retriever: semantic-first with optional lexical boost.
Auto-detects when to use BM25 for proper nouns and technical terms.
"""
import re
from typing import List, Dict
from collections import defaultdict

from src.retrieval.semantic_retriever import get_retriever as get_semantic
from src.retrieval.lexical_retriever import get_lexical_retriever
from src.config import get_retrieval_config


class HybridRetriever:
    """
    Clean hybrid retrieval with auto-detection:
    - Primary: Semantic search (FAISS)
    - Auto-detect: Use BM25 for proper nouns, acronyms, technical terms
    - Optional: Manual BM25 boost for keyword-heavy queries
    
    Much simpler than before - no complex RRF, just score fusion.
    """
    
    def __init__(self, mode: str = 'ultra'):
        """
        Initialize hybrid retriever.
        
        Args:
            mode: 'standard' or 'ultra'
        """
        self.mode = mode
        self.cfg = get_retrieval_config()
        
        # Get retrievers (cached)
        self.semantic = get_semantic(mode)
        self.lexical = None  # Lazy load (only if needed)
        
        print(f"Hybrid Retriever initialized in {mode.upper()} mode")
        print(f"  Auto-detection: Enabled (proper nouns, acronyms, numbers)")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = None,
        use_lexical: bool = None
    ) -> List[Dict]:
        """
        Retrieve documents with optional auto-detection of BM25 usage.
        
        Args:
            query: Search query
            top_k: Number of results (default from config)
            use_lexical: Add BM25 boost (None = auto-detect, True = force, False = disable)
        
        Returns:
            List of documents sorted by relevance
        """
        top_k = top_k or self.cfg['top_k']
        
        # AUTO-DETECT if not specified
        if use_lexical is None:
            use_lexical = self._should_use_bm25(query)
            if use_lexical:
                print(f"  🎯 Auto-enabled BM25 (detected: proper nouns/technical terms)")
        
        if not use_lexical:
            # Fast path: semantic only
            print(f"🔍 Semantic search only (fast)")
            results = self.semantic.retrieve(query, top_k)
            return self._normalize_scores(results)
        
        # Hybrid: semantic + lexical
        print(f"🔍 Hybrid search: semantic + BM25 boost")
        
        # Semantic results
        sem_results = self.semantic.retrieve(query, top_k)
        
        # Lexical results
        if self.lexical is None:
            self.lexical = get_lexical_retriever(self.mode)
        
        lex_results = self.lexical.retrieve(query, self.cfg['lexical_top_k'])
        
        # Combine with simple fusion
        combined = self._fuse_results(sem_results, lex_results, top_k)
        
        return combined
    
    def _should_use_bm25(self, query: str) -> bool:
        """
        Auto-detect if query would benefit from BM25 boost.
        
        Triggers for:
        - Proper nouns (capitalized words)
        - Acronyms (2+ consecutive capitals)
        - Numbers/versions
        - Technical long words (>10 chars)
        
        Returns:
            True if BM25 should be used
        """
        # Check for proper nouns (capitalized words after first word)
        # Example: "EF Johnson Technologies", "Barack Obama"
        words = query.split()
        capital_words = sum(1 for w in words[1:] if w and w[0].isupper())
        if capital_words >= 1:
            return True
        
        # Check for acronyms (2+ consecutive capitals)
        # Example: "NASA", "IBM", "HTTP", "COVID-19"
        if re.search(r'\b[A-Z]{2,}\b', query):
            return True
        
        # Check for numbers/versions
        # Example: "Python 3.9", "iPhone 15", "COVID-19"
        if re.search(r'\b\d+', query):
            return True
        
        # Check for technical terms (long words often technical)
        # Example: "photosynthesis", "semiconductor"
        if any(len(w) > 10 for w in words):
            return True
        
        return False
    
    def _fuse_results(
        self, 
        sem_results: List[Dict], 
        lex_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Simple fusion: weighted sum of normalized scores.
        Much simpler than RRF.
        """
        sem_weight = self.cfg['semantic_weight']
        lex_weight = self.cfg['lexical_weight']
        
        # Normalize scores
        sem_results = self._normalize_scores(sem_results)
        lex_results = self._normalize_scores(lex_results)
        
        # Create score map
        score_map = defaultdict(lambda: {'sem': 0.0, 'lex': 0.0, 'doc': None})
        
        for doc in sem_results:
            doc_id = doc['id']
            score_map[doc_id]['sem'] = doc['score']
            score_map[doc_id]['doc'] = doc
        
        for doc in lex_results:
            doc_id = doc['id']
            score_map[doc_id]['lex'] = doc['score']
            if score_map[doc_id]['doc'] is None:
                score_map[doc_id]['doc'] = doc
        
        # Calculate combined scores
        combined = []
        for doc_id, data in score_map.items():
            doc = data['doc']
            
            # Weighted fusion
            combined_score = (
                sem_weight * data['sem'] + 
                lex_weight * data['lex']
            )
            
            doc['score'] = combined_score
            doc['sem_score'] = data['sem']
            doc['lex_score'] = data['lex']
            
            combined.append(doc)
        
        # Sort and return top_k
        combined.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"  ✅ Combined {len(combined)} unique documents")
        
        return combined[:top_k]
    
    def _normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """Min-max normalize scores to [0, 1]."""
        if not results:
            return results
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for r in results:
                r['score'] = 1.0
        else:
            for r in results:
                r['score'] = (r['score'] - min_score) / (max_score - min_score)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            'mode': self.mode,
            'semantic_stats': self.semantic.get_stats(),
            'lexical_loaded': self.lexical is not None,
            'auto_bm25_detection': True
        }


# ============================================================================
# GLOBAL CACHE
# ============================================================================

_HYBRID_CACHE = {}

def get_hybrid_retriever(mode: str = 'ultra') -> HybridRetriever:
    """Get cached hybrid retriever."""
    if mode not in _HYBRID_CACHE:
        _HYBRID_CACHE[mode] = HybridRetriever(mode)
    
    return _HYBRID_CACHE[mode]