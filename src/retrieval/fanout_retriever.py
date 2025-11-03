from typing import List, Dict, Set
from collections import defaultdict, Counter

from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.fanout_generator import get_fanout_generator


class FanOutRetriever:
    """
    Retrieves documents using fan-out queries with consensus scoring.
    
    Strategy:
    1. Generate 3-5 specialized queries from user query
    2. Retrieve top-k documents for each query
    3. Identify consensus articles (appear in multiple fan-outs)
    4. Boost consensus articles + keep top semantic matches
    5. Deduplicate and rank by combined score
    """
    
    def __init__(self, base_retriever: HybridRetriever):
        """
        Initialize fan-out retriever.
        
        Args:
            base_retriever: HybridRetriever instance to use for each query
        """
        self.base_retriever = base_retriever
        self.fanout_generator = get_fanout_generator()
        
        # Scoring weights
        self.semantic_weight = 0.6
        self.consensus_weight = 0.4
        self.consensus_boost = 1.5  # Boost for articles in 3+ fan-outs
        
        print(f"Fan-out retriever initialized")
        print(f"  Consensus boost: {self.consensus_boost}x for 3+ appearances")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_lexical: bool = None,
        fanout_k: int = 15,
        max_fanouts: int = 5
    ) -> Dict:
        """
        Retrieve documents using fan-out queries.
        
        Args:
            query: User query
            top_k: Final number of results to return
            use_lexical: Use BM25 boost (passed to base retriever)
            fanout_k: Documents to retrieve per fan-out query
            max_fanouts: Maximum fan-out queries to generate
        
        Returns:
            Dictionary with:
                - documents: Ranked documents
                - fanout_info: Metadata about fan-out process
        """
        # Generate fan-out queries
        fanouts = self.fanout_generator.generate_fanout_queries(
            query, 
            max_fanouts=max_fanouts
        )
        
        # Fallback to direct retrieval if no fan-outs
        if not fanouts:
            print(f"  ℹ️  No fan-out needed, using direct retrieval")
            documents = self.base_retriever.retrieve(
                query,
                top_k=top_k,
                use_lexical=use_lexical
            )
            return {
                'documents': documents,
                'fanout_info': {
                    'used_fanout': False,
                    'num_fanouts': 0,
                    'fanout_queries': []
                }
            }
        
        # Retrieve documents for each fan-out
        print(f"  🔍 Retrieving {fanout_k} docs per fan-out...")
        fanout_results = []
        
        for i, fanout in enumerate(fanouts, 1):
            results = self.base_retriever.retrieve(
                query=fanout['query'],
                top_k=fanout_k,
                use_lexical=use_lexical
            )
            
            fanout_results.append({
                'query': fanout['query'],
                'purpose': fanout['purpose'],
                'results': results
            })
            
            print(f"     Fan-out {i}/{len(fanouts)}: {len(results)} docs")
        
        # Add original query results (important!)
        print(f"  🔍 Retrieving original query...")
        original_results = self.base_retriever.retrieve(
            query=query,
            top_k=fanout_k,
            use_lexical=use_lexical
        )
        
        # Combine and score
        combined = self._combine_with_consensus(
            original_query=query,
            original_results=original_results,
            fanout_results=fanout_results,
            top_k=top_k
        )
        
        return {
            'documents': combined,
            'fanout_info': {
                'used_fanout': True,
                'num_fanouts': len(fanouts),
                'fanout_queries': [
                    {'query': fo['query'], 'purpose': fo['purpose']}
                    for fo in fanouts
                ],
                'consensus_articles': self._get_consensus_count(fanout_results)
            }
        }
    
    def _combine_with_consensus(
        self,
        original_query: str,
        original_results: List[Dict],
        fanout_results: List[Dict],
        top_k: int
    ) -> List[Dict]:
        """
        Combine fan-out results with consensus scoring.
        
        Strategy:
        1. Track which articles appear in multiple fan-outs
        2. Boost consensus articles (appear in 3+ queries)
        3. Keep top semantic matches from original query
        4. Deduplicate and rank
        """
        # Track document appearances and scores
        doc_info = defaultdict(lambda: {
            'doc': None,
            'appearances': 0,
            'purposes': set(),
            'max_semantic_score': 0.0,
            'avg_semantic_score': 0.0,
            'scores': [],
            'from_original': False
        })
        
        # Process original query results (highest priority)
        for doc in original_results:
            doc_id = doc['id']
            doc_info[doc_id]['doc'] = doc
            doc_info[doc_id]['appearances'] += 1
            doc_info[doc_id]['from_original'] = True
            doc_info[doc_id]['max_semantic_score'] = doc['score']
            doc_info[doc_id]['scores'].append(doc['score'])
        
        # Process fan-out results
        for fanout in fanout_results:
            purpose = fanout['purpose']
            
            for doc in fanout['results']:
                doc_id = doc['id']
                
                # First time seeing this document?
                if doc_info[doc_id]['doc'] is None:
                    doc_info[doc_id]['doc'] = doc
                
                doc_info[doc_id]['appearances'] += 1
                doc_info[doc_id]['purposes'].add(purpose)
                doc_info[doc_id]['scores'].append(doc['score'])
                
                # Track max semantic score
                if doc['score'] > doc_info[doc_id]['max_semantic_score']:
                    doc_info[doc_id]['max_semantic_score'] = doc['score']
        
        # Calculate final scores
        scored_docs = []
        
        for doc_id, info in doc_info.items():
            if info['doc'] is None:
                continue
            
            # Average semantic score
            avg_score = sum(info['scores']) / len(info['scores'])
            info['avg_semantic_score'] = avg_score
            
            # Consensus score (normalized by number of queries)
            total_queries = 1 + len(fanout_results)  # +1 for original
            consensus_score = info['appearances'] / total_queries
            
            # Apply consensus boost for articles in 3+ queries
            if info['appearances'] >= 3:
                consensus_score *= self.consensus_boost
            
            # Boost for original query results
            if info['from_original']:
                consensus_score *= 1.2
            
            # Combined score
            final_score = (
                self.semantic_weight * info['max_semantic_score'] +
                self.consensus_weight * min(consensus_score, 1.0)
            )
            
            # Update document
            doc = info['doc'].copy()
            doc['score'] = final_score
            doc['fanout_score'] = final_score
            doc['semantic_score'] = info['max_semantic_score']
            doc['consensus_score'] = consensus_score
            doc['appearances'] = info['appearances']
            doc['purposes'] = list(info['purposes'])
            doc['from_original'] = info['from_original']
            
            scored_docs.append(doc)
        
        # Sort by final score
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Log statistics
        consensus_docs = [d for d in scored_docs if d['appearances'] >= 3]
        print(f"  ✅ Combined results:")
        print(f"     Total unique: {len(scored_docs)}")
        print(f"     Consensus (3+): {len(consensus_docs)}")
        print(f"     From original: {sum(1 for d in scored_docs if d['from_original'])}")
        
        return scored_docs[:top_k]
    
    def _get_consensus_count(self, fanout_results: List[Dict]) -> int:
        """Count documents appearing in 3+ fan-outs."""
        doc_counts = Counter()
        
        for fanout in fanout_results:
            for doc in fanout['results']:
                doc_counts[doc['id']] += 1
        
        return sum(1 for count in doc_counts.values() if count >= 3)
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            'mode': self.base_retriever.mode,
            'fanout_enabled': self.fanout_generator.available,
            'semantic_weight': self.semantic_weight,
            'consensus_weight': self.consensus_weight,
            'consensus_boost': self.consensus_boost
        }


# ============================================================================
# GLOBAL CACHE
# ============================================================================

_FANOUT_CACHE = {}

def get_fanout_retriever(mode: str = 'ultra') -> FanOutRetriever:
    """Get cached fan-out retriever."""
    if mode not in _FANOUT_CACHE:
        from src.retrieval.hybrid_retriever import get_hybrid_retriever
        base_retriever = get_hybrid_retriever(mode)
        _FANOUT_CACHE[mode] = FanOutRetriever(base_retriever)
    
    return _FANOUT_CACHE[mode]