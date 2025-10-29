
from typing import List, Dict
import re


class RelevanceFilter:
    """
    Simple relevance filter that works when summaries are empty.
    
    Scoring:
    1. Keyword overlap in title + text (40%)
    2. Semantic similarity score from retriever (60%)
    
    Trusts the semantic model more since summaries are unavailable.
    """
    
    def __init__(self, min_score: float = 0.3):
        """
        Initialize filter.
        
        Args:
            min_score: Minimum score to keep (0-1)
        """
        self.min_score = min_score
        
        # Stop words to ignore
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'will', 'with',
            'what', 'who', 'when', 'where', 'why', 'how', 'does', 'do'
        }
    
    def filter_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Filter results by relevance.
        
        Args:
            query: Search query
            results: Retrieved documents
        
        Returns:
            Filtered documents with relevance scores
        """
        if not results:
            return results
        
        # Score each document
        for doc in results:
            relevance_score = self._calculate_relevance(query, doc)
            doc['relevance_score'] = relevance_score
            doc['relevance'] = self._get_relevance_level(relevance_score)
        
        # Filter by threshold
        filtered = [
            doc for doc in results 
            if doc['relevance_score'] >= self.min_score
        ]
        
        # Sort by combined score (retrieval + relevance)
        for doc in filtered:
            doc['final_score'] = (
                0.5 * doc['score'] +          # Retrieval score (semantic + reranking)
                0.5 * doc['relevance_score']  # Relevance score
            )
        
        filtered.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"  ✅ Relevance filter: {len(filtered)}/{len(results)} passed (threshold: {self.min_score})")
        
        return filtered
    
    def _calculate_relevance(self, query: str, doc: Dict) -> float:
        """
        Calculate relevance score (0-1).
        
        Two components:
        1. Keyword overlap (40% weight)
        2. Semantic similarity from retriever (60% weight)
        
        Trusts semantic model more since 95% of docs have no summary.
        """
        # Get semantic score from retriever (already 0-1)
        semantic_score = doc.get('score', 0.5)
        
        # Calculate keyword overlap
        keyword_score = self._keyword_overlap(query, doc)
        
        # Weighted combination (trust semantic model more)
        return 0.4 * keyword_score + 0.6 * semantic_score
    
    def _keyword_overlap(self, query: str, doc: Dict) -> float:
        """
        Check if query terms appear in title and text.
        Ignores summaries since 95% are empty.
        """
        # Extract query terms (remove stop words)
        query_terms = set(
            term.lower() 
            for term in re.findall(r'\b\w+\b', query)
            if term.lower() not in self.stop_words and len(term) > 2
        )
        
        if not query_terms:
            return 0.5  # Neutral if no valid terms
        
        # Check in title (very high weight - titles are usually relevant)
        title = doc.get('title', '').lower()
        title_matches = sum(1 for term in query_terms if term in title)
        title_ratio = title_matches / len(query_terms)
        
        # Check in text (use more text since no summaries available)
        text = doc.get('text', '')[:2000].lower()  # First 2000 chars
        text_matches = sum(1 for term in query_terms if term in text)
        text_ratio = text_matches / len(query_terms)
        
        # Combined score
        # Title match is worth more (0.6) than text match (0.4)
        score = 0.6 * title_ratio + 0.4 * text_ratio
        
        return min(score, 1.0)
    
    def _get_relevance_level(self, score: float) -> str:
        """Convert score to level."""
        if score >= 0.7:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_FILTER = None

def get_relevance_filter(min_score: float = 0.3) -> RelevanceFilter:
    """Get cached filter instance."""
    global _FILTER
    
    if _FILTER is None:
        _FILTER = RelevanceFilter(min_score)
    
    return _FILTER