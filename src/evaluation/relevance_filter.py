# src/evaluation/relevance_filter.py
"""
Lightweight relevance filtering.
Much faster than the old 9-dimensional evaluation.

Only checks 2 things:
1. Keyword overlap (is the query in the document?)
2. Content quality (is the document substantial?)
"""
from typing import List, Dict
import re


class RelevanceFilter:
    """
    Simple 2-check relevance filter.
    
    Old system: 9 dimensions, 2-3 seconds
    New system: 2 checks, 50ms
    """
    
    def __init__(self, min_score: float = 0.2):
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
            'what', 'who', 'when', 'where', 'why', 'how'
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
                0.6 * doc['score'] +          # Retrieval score
                0.4 * doc['relevance_score']  # Relevance score
            )
        
        filtered.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"  ✅ Relevance filter: {len(filtered)}/{len(results)} passed (threshold: {self.min_score})")
        
        return filtered
    
    def _calculate_relevance(self, query: str, doc: Dict) -> float:
        """
        Calculate relevance score (0-1).
        
        Two simple checks:
        1. Keyword overlap (0.6 weight)
        2. Content quality (0.4 weight)
        """
        # Check 1: Keyword overlap
        keyword_score = self._keyword_overlap(query, doc)
        
        # Check 2: Content quality
        quality_score = self._content_quality(doc)
        
        # Combined score
        return 0.6 * keyword_score + 0.4 * quality_score
    
    def _keyword_overlap(self, query: str, doc: Dict) -> float:
        """
        Check if query terms appear in document.
        Simple but effective.
        """
        # Extract query terms (remove stop words)
        query_terms = set(
            term.lower() 
            for term in re.findall(r'\b\w+\b', query)
            if term.lower() not in self.stop_words and len(term) > 2
        )
        
        if not query_terms:
            return 0.5  # Neutral if no valid terms
        
        # Check in title (highest weight)
        title = doc.get('title', '').lower()
        title_matches = sum(1 for term in query_terms if term in title)
        title_score = (title_matches / len(query_terms)) * 0.5
        
        # Check in summary (medium weight)
        summary = doc.get('summary', '').lower()
        summary_matches = sum(1 for term in query_terms if term in summary)
        summary_score = (summary_matches / len(query_terms)) * 0.3
        
        # Check in text preview (lower weight)
        text = doc.get('text', '')[:500].lower()
        text_matches = sum(1 for term in query_terms if term in text)
        text_score = (text_matches / len(query_terms)) * 0.2
        
        return min(title_score + summary_score + text_score, 1.0)
    
    def _content_quality(self, doc: Dict) -> float:
        """
        Check if document has substantial content.
        """
        text = doc.get('text', '')
        summary = doc.get('summary', '')
        
        text_len = len(text)
        summary_len = len(summary)
        
        score = 0.5  # Base score
        
        # Good text length (500-10000 chars)
        if 500 <= text_len <= 10000:
            score += 0.3
        elif 200 <= text_len < 500:
            score += 0.2
        elif text_len > 10000:
            score += 0.2
        
        # Has summary
        if summary_len > 100:
            score += 0.2
        elif summary_len > 50:
            score += 0.1
        
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

def get_relevance_filter(min_score: float = 0.2) -> RelevanceFilter:
    """Get cached filter instance."""
    global _FILTER
    
    if _FILTER is None:
        _FILTER = RelevanceFilter(min_score)
    
    return _FILTER
