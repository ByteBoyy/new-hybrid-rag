# src/retrieval/lexical_retriever.py
"""
Optional BM25 lexical retriever for keyword-heavy queries.
Use sparingly - semantic search is usually better.
"""
import bz2
import pickle
import lzma
import zlib
from rank_bm25 import BM25Okapi
from typing import List, Dict

from src.config import get_config


class LexicalRetriever:
    """
    Simple BM25 retriever for keyword matching.
    Only use when semantic search needs a boost.
    """
    
    # Shared cache
    _cache = {}
    
    def __init__(self, mode: str = 'ultra'):
        """
        Initialize BM25 retriever.
        
        Args:
            mode: 'standard' or 'ultra' (uses same metadata)
        """
        self.mode = mode
        cfg = get_config(mode)
        
        print(f"Loading BM25 retriever for {mode.upper()} mode...")
        
        # Load metadata
        self._load_documents(cfg['metadata_path'], mode)
        
        # Build BM25 index
        print(f"  Building BM25 index from {len(self.docs):,} documents...")
        self.corpus_tokens = []
        self.doc_ids = []
        
        for idx, doc in self.docs.items():
            # Get text and summary
            text = self._get_text(doc)
            summary = self._get_summary(doc)
            
            # Tokenize
            combined = f"{summary} {text}"
            tokens = combined.lower().split()[:500]  # Limit tokens
            
            self.corpus_tokens.append(tokens)
            self.doc_ids.append(idx)
        
        self.bm25 = BM25Okapi(self.corpus_tokens)
        
        print(f"✅ BM25 ready: {len(self.doc_ids):,} documents\n")
    
    def _load_documents(self, metadata_path: str, mode: str):
        """Load documents based on mode."""
        if mode == 'ultra':
            with bz2.open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            self.docs = data['d']
            self.title_dict = data.get('td', [])
            self.url_dict = data.get('ud', [])
        else:
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
            self.docs = data['id_map']
            self.title_dict = None
            self.url_dict = None
    
    def _get_text(self, doc: Dict) -> str:
        """Extract text from document."""
        if self.mode == 'ultra':
            text = doc.get('t', '')
            return self._decompress(text, True)
        else:
            return doc.get('text', '')
    
    def _get_summary(self, doc: Dict) -> str:
        """Extract summary from document."""
        if self.mode == 'ultra':
            summary = doc.get('s', '')
            method = doc.get('_sc')
            return self._decompress(summary, method)
        else:
            return doc.get('summary', '')
    
    def _decompress(self, data, method):
        """Decompress text."""
        if not data or not isinstance(data, bytes):
            return data if isinstance(data, str) else ''
        
        try:
            if method == 1:
                return zlib.decompress(data).decode('utf-8')
            elif method == 2:
                return lzma.decompress(data).decode('utf-8')
            else:
                try:
                    return zlib.decompress(data).decode('utf-8')
                except:
                    return lzma.decompress(data).decode('utf-8')
        except:
            return ""
    
    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        """
        Retrieve using BM25.
        
        Args:
            query: Search query
            top_k: Number of results
        
        Returns:
            List of documents with BM25 scores
        """
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        
        # Get top results
        top_indices = scores.argsort()[-top_k:][::-1]
        
        results = []
        for pos in top_indices:
            doc_id = self.doc_ids[pos]
            doc = self.docs.get(doc_id)
            
            if not doc:
                continue
            
            # Parse document
            if self.mode == 'ultra':
                title_idx = doc.get('ti', -1)
                title = self.title_dict[title_idx] if title_idx >= 0 else 'Unknown'
                
                url_idx = doc.get('ui', -1)
                url = self.url_dict[url_idx] if url_idx >= 0 else ''
            else:
                title = doc.get('title', 'Unknown')
                url = doc.get('url', '')
            
            results.append({
                'id': str(doc_id),
                'title': title,
                'text': self._get_text(doc),
                'summary': self._get_summary(doc),
                'url': url,
                'score': float(scores[pos]),
                'domain': doc.get('d', 'general') if self.mode == 'ultra' else doc.get('primary_domain', 'general')
            })
        
        return results


def get_lexical_retriever(mode: str = 'ultra') -> LexicalRetriever:
    """Get cached lexical retriever."""
    if mode not in LexicalRetriever._cache:
        LexicalRetriever._cache[mode] = LexicalRetriever(mode)
    
    return LexicalRetriever._cache[mode]
