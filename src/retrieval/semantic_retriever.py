import faiss
import pickle
import bz2
import lzma
import zlib
import numpy as np
import re
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict

from src.config import get_config, EMBEDDING_MODEL, ENABLE_RERANKING


class SemanticRetriever:
    """
    Fast semantic retriever with dual-mode support and reranking.
    
    Modes:
        - standard: Full 768D embeddings (2.2 GB, 100% quality)
        - ultra: 192D PCA compressed (1.1 GB, 97%+ quality)
    """
    
    # Shared embedder (loaded once globally)
    _embedder = None
    
    def __init__(self, mode: str = 'ultra'):
        """
        Initialize retriever in specified mode.
        
        Args:
            mode: 'standard' or 'ultra'
        """
        self.mode = mode
        cfg = get_config(mode)
        
        print(f"Loading {mode.upper()} semantic retriever...")
        
        # Load FAISS index
        index_path = Path(cfg['index_path'])
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Configure search parameters
        if mode == 'ultra':
            self.index.nprobe = cfg.get('nprobe', 20)
            
            # Load PCA model for compression
            pca_path = Path(cfg['pca_path'])
            with open(pca_path, 'rb') as f:
                self.pca_model = pickle.load(f)
            
            print(f"  Dimensions: {cfg['dimensions']}D (PCA compressed)")
            print(f"  nprobe: {self.index.nprobe}")
        else:
            self.pca_model = None
            print(f"  Dimensions: {cfg['dimensions']}D (full)")
        
        # Load metadata
        self._load_metadata(cfg['metadata_path'], mode)
        
        # Load embedder (shared across all instances)
        if SemanticRetriever._embedder is None:
            print(f"  Loading embedding model: {EMBEDDING_MODEL}")
            SemanticRetriever._embedder = SentenceTransformer(EMBEDDING_MODEL)
        
        self.embedder = SemanticRetriever._embedder
        
        print(f"✅ {mode.upper()} retriever ready: {self.index.ntotal:,} vectors, {len(self.docs):,} documents\n")
    
    def _load_metadata(self, metadata_path: str, mode: str):
        """Load and parse metadata based on mode."""
        metadata_path = Path(metadata_path)
        
        if mode == 'ultra':
            # Ultra: compressed with BZ2
            with bz2.open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            
            self.docs = data['d']  # Documents
            self.title_dict = data.get('td', [])
            self.url_dict = data.get('ud', [])
            
            print(f"  Metadata: {len(self.docs):,} docs, {len(self.title_dict):,} titles, {len(self.url_dict):,} URLs")
        else:
            # Standard: regular pickle
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            
            self.docs = data['id_map']
            self.title_dict = None
            self.url_dict = None
            
            print(f"  Metadata: {len(self.docs):,} documents")
    
    def retrieve(self, query: str, top_k: int = 10, retrieve_k: int = None, enable_reranking: bool = None) -> List[Dict]:
        """
        Retrieve documents using semantic search with optional reranking.
        
        Two-stage approach:
        1. Retrieve more candidates (retrieve_k)
        2. Rerank with multiple signals and return top_k
        
        Args:
            query: Search query
            top_k: Number of results to return
            retrieve_k: Number to retrieve initially (default: top_k * 5)
            enable_reranking: Use multi-signal reranking (default: from config)
        
        Returns:
            List of documents with text, summary, metadata
        """
        # Use config default if not specified
        if enable_reranking is None:
            enable_reranking = ENABLE_RERANKING
        
        # Retrieve more candidates for reranking
        retrieve_k = retrieve_k or (top_k * 5)
        
        if enable_reranking:
            print(f"  📥 Retrieving {retrieve_k} candidates for reranking...")
        
        # Encode query
        qvec = self.embedder.encode(
            [query], 
            normalize_embeddings=True, 
            convert_to_numpy=True
        )
        
        # Apply PCA if ultra mode
        if self.pca_model is not None:
            qvec = self._apply_pca(qvec)
        
        # Search
        D, I = self.index.search(qvec, retrieve_k)
        
        # Build results
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or str(idx) not in self.docs:
                continue
            
            doc = self._get_document(idx)
            
            if doc:
                doc['semantic_score'] = float(score)  # Keep original semantic score
                doc['score'] = float(score)  # Initial score (will be updated by reranking)
                results.append(doc)
        
        # Stage 2: Rerank if enabled
        if enable_reranking and results:
            print(f"  🎯 Reranking with title + TF + quality signals...")
            results = self._rerank_results(query, results)
        
        # Return top_k
        print(f"  ✅ Returning top {min(top_k, len(results))} results")
        return results[:top_k]
    
    def _rerank_results(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Rerank results with multiple signals:
        1. Title term matching (strongest - 35% max boost)
        2. Term frequency in text (15% max boost)
        3. Document quality (8% max boost)
        
        This significantly improves relevance compared to semantic similarity alone.
        """
        # Extract meaningful query terms
        query_terms = self._extract_query_terms(query)
        
        if not query_terms:
            return results  # No reranking possible
        
        for doc in results:
            semantic_score = doc['semantic_score']
            
            # Signal 1: Title term presence (STRONGEST)
            title_boost = self._calculate_title_boost(query_terms, doc['title'])
            
            # Signal 2: Term frequency in text
            tf_boost = self._calculate_tf_boost(query_terms, doc['text'])
            
            # Signal 3: Document quality
            quality_boost = self._calculate_quality_boost(doc['text'])
            
            # Combine signals
            doc['score'] = semantic_score + title_boost + tf_boost + quality_boost
            
            # Store components for debugging (optional)
            doc['_rerank_components'] = {
                'semantic': semantic_score,
                'title': title_boost,
                'tf': tf_boost,
                'quality': quality_boost
            }
        
        # Sort by new score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results
    
    def _extract_query_terms(self, query: str) -> set:
        """
        Extract meaningful terms from query.
        Removes stop words and short terms.
        """
        stop_words = {
            'what', 'is', 'the', 'a', 'an', 'how', 'does', 'do', 'in', 
            'of', 'to', 'for', 'are', 'was', 'were', 'been', 'be',
            'have', 'has', 'had', 'with', 'by', 'from', 'about', 'at'
        }
        
        terms = set(
            term.lower() 
            for term in re.findall(r'\b\w+\b', query)
            if term.lower() not in stop_words and len(term) >= 3
        )
        
        return terms
    
    def _calculate_title_boost(self, query_terms: set, title: str) -> float:
        """
        Calculate boost from title matching.
        
        Title is the STRONGEST signal - if query terms appear in title,
        the document is almost always relevant.
        
        Returns: 0 to 0.35 boost
        """
        if not title:
            return 0
        
        title_lower = title.lower()
        title_words = set(re.findall(r'\b\w+\b', title_lower))
        
        # Count exact matches
        exact_matches = len(query_terms & title_words)
        
        # Check for partial/stem matches
        # e.g., "technology" in query, "technologies" in title
        partial_matches = 0
        for qterm in query_terms:
            if qterm not in title_words:
                # Check substring matches (stem matching)
                if any(qterm in tword or tword in qterm for tword in title_words if len(qterm) > 3):
                    partial_matches += 1
        
        # Calculate match ratio
        total_matches = exact_matches + (partial_matches * 0.5)
        match_ratio = total_matches / len(query_terms) if query_terms else 0
        
        # Aggressive boosting for title matches
        if match_ratio >= 0.8:
            return 0.35  # Almost all terms in title!
        elif match_ratio >= 0.6:
            return 0.25  # Most terms in title
        elif match_ratio >= 0.4:
            return 0.15  # Several terms in title
        elif match_ratio > 0:
            return 0.08  # At least one term in title
        else:
            return 0
    
    def _calculate_tf_boost(self, query_terms: set, text: str) -> float:
        """
        Calculate boost from term frequency.
        
        Counts how often query terms appear in the early part of the text
        (first 1500 chars where key info usually appears).
        
        Returns: 0 to 0.15 boost
        """
        if not text:
            return 0
        
        # Focus on early text (most important info is usually there)
        text_early = text[:1500].lower()
        
        # Count occurrences with word boundaries
        total_occurrences = 0
        for term in query_terms:
            count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_early))
            total_occurrences += count
        
        # Normalize: 
        # - 1 occurrence per term = decent (score ~0.05)
        # - 3+ occurrences per term = excellent (score ~0.15)
        # Cap to prevent keyword stuffing bias
        normalized_score = min(total_occurrences / (len(query_terms) * 10), 0.15)
        
        return normalized_score
    
    def _calculate_quality_boost(self, text: str) -> float:
        """
        Calculate boost from document quality (based on length).
        
        Well-developed articles are usually 1000-10000 characters.
        
        Returns: 0 to 0.08 boost
        """
        length = len(text)
        
        if 1000 <= length <= 10000:
            return 0.08  # Sweet spot - well-developed article
        elif 500 <= length < 1000:
            return 0.04  # Decent article
        elif length > 10000:
            return 0.05  # Comprehensive but might be too broad
        else:
            return 0  # Too short (stub article)
    
    def _apply_pca(self, embedding: np.ndarray) -> np.ndarray:
        """Apply PCA transformation for ultra mode."""
        return np.dot(
            embedding - self.pca_model['mean'],
            self.pca_model['components'].T
        ).astype(np.float32)
    
    def _get_document(self, idx: int) -> Dict:
        """Extract document with text/summary based on mode."""
        doc = self.docs.get(str(idx))
        if not doc:
            return None
        
        if self.mode == 'ultra':
            return self._parse_ultra_document(doc, idx)
        else:
            return self._parse_standard_document(doc, idx)
    
    def _parse_ultra_document(self, doc: Dict, idx: int) -> Dict:
        """Parse ultra-compressed document."""
        # Get title
        title_idx = doc.get('ti', -1)
        if title_idx >= 0 and title_idx < len(self.title_dict):
            title = self.title_dict[title_idx]
        else:
            title = doc.get('t', 'Unknown')
        
        # Get URL
        url_idx = doc.get('ui', -1)
        if url_idx >= 0 and url_idx < len(self.url_dict):
            url = self.url_dict[url_idx]
        else:
            url = doc.get('u', '')
        
        # Decompress text
        text = self._decompress_text(doc.get('t', ''), True)
        summary = self._decompress_text(doc.get('s', ''), True)
        
        return {
            'id': str(idx),
            'title': title,
            'text': text,
            'summary': summary,
            'url': url,
            'domain': doc.get('d', 'general')
        }
    
    def _parse_standard_document(self, doc: Dict, idx: int) -> Dict:
        """Parse standard document."""
        return {
            'id': str(idx),
            'title': doc.get('title', 'Unknown'),
            'text': doc.get('text', ''),
            'summary': doc.get('summary', ''),
            'url': doc.get('url', ''),
            'domain': doc.get('primary_domain', 'general')
        }
    
    def _decompress_text(self, data, method):
        """Decompress text based on compression method."""
        if not data or not isinstance(data, bytes):
            return data if isinstance(data, str) else ''
        
        try:
            return lzma.decompress(data).decode('utf-8')
        except Exception as e:
            print(f"⚠️  Decompression error: {e}")
            return ""
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            'mode': self.mode,
            'total_vectors': self.index.ntotal,
            'total_documents': len(self.docs),
            'dimensions': self.pca_model['n_components'] if self.pca_model else 768,
            'compressed': self.mode == 'ultra',
            'reranking_enabled': ENABLE_RERANKING
        }


# ============================================================================
# GLOBAL CACHE FOR FAST INITIALIZATION
# ============================================================================

_RETRIEVER_CACHE = {}

def get_retriever(mode: str = 'ultra') -> SemanticRetriever:
    """
    Get cached retriever instance (fast).
    
    Args:
        mode: 'standard' or 'ultra'
    
    Returns:
        Cached SemanticRetriever instance
    """
    if mode not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[mode] = SemanticRetriever(mode)
    
    return _RETRIEVER_CACHE[mode]