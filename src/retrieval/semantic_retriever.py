# src/retrieval/semantic_retriever.py
"""
Semantic retriever using FAISS.
Supports both standard (768D) and ultra-compressed (192D) modes.
"""
import faiss
import pickle
import bz2
import lzma
import zlib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import List, Dict

from src.config import get_config, EMBEDDING_MODEL


class SemanticRetriever:
    """
    Fast semantic retriever with dual-mode support.
    
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
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve documents using semantic search.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of documents with text, summary, metadata
        """
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
        D, I = self.index.search(qvec, top_k)
        
        # Build results
        results = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0 or str(idx) not in self.docs:
                continue
            
            doc = self._get_document(idx)
            
            if doc:
                doc['score'] = float(score)
                results.append(doc)
        
        return results
    
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
        # """Decompress text based on compression method."""
        if not data or not isinstance(data, bytes):
            return data if isinstance(data, str) else ''
        
        # try:
        #     if method == 1:  # zlib
        #         return zlib.decompress(data).decode('utf-8')
        #     elif method == 2:  # LZMA
        #         return lzma.decompress(data).decode('utf-8')
        #     else:
        #         # Auto-detect
        #         try:
        #             return zlib.decompress(data).decode('utf-8')
        #         except:
        #             return lzma.decompress(data).decode('utf-8')
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
            'compressed': self.mode == 'ultra'
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
