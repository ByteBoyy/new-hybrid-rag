# src/generation/answer_generator.py
"""
LLM-based answer generation using OpenAI.
"""
import os
from openai import OpenAI
from typing import List, Dict

from src.config import get_generation_config


class AnswerGenerator:
    """Generate answers using LLM with retrieved context."""
    
    def __init__(self):
        """Initialize generator."""
        cfg = get_generation_config()
        
        self.api_key = cfg['api_key']
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = cfg['model']
        self.temperature = cfg['temperature']
        self.max_tokens = cfg['max_tokens']
    
    def generate(self, query: str, documents: List[Dict]) -> Dict:
        """
        Generate answer from query and retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved documents with text/summary
        
        Returns:
            Dictionary with answer, sources, confidence
        """
        if not documents:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'confidence': 'low',
                'reasoning': "No documents retrieved"
            }
        
        # Build context
        context = self._build_context(documents)
        
        # Generate answer
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        'role': 'system',
                        'content': self._get_system_prompt()
                    },
                    {
                        'role': 'user',
                        'content': f"Question: {query}\n\n{context}\n\nPlease answer the question."
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content.strip()
            
            # Extract sources
            sources = [
                {
                    'title': doc['title'],
                    'url': doc['url'],
                    'score': doc.get('final_score', doc.get('score', 0))
                }
                for doc in documents[:5]  # Top 5 sources
            ]
            
            # Determine confidence
            confidence = self._determine_confidence(documents, answer)
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'reasoning': f"Answer based on {len(documents)} sources with avg relevance {self._avg_relevance(documents):.2f}"
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'confidence': 'low',
                'reasoning': f"Generation error: {str(e)}"
            }
    
    def _build_context(self, documents: List[Dict]) -> str:
        """Build context from documents."""
        context_parts = []
        
        for i, doc in enumerate(documents[:5], 1):  # Top 5
            summary = doc.get('summary', '').strip()
            text = doc.get('text', '').strip()
            
            # Use summary + first part of text
            if summary:
                content = f"{summary}\n\n{text[:1000]}" if text else summary
            else:
                content = text[:1500] if text else "No content"
            
            context_parts.append(
                f"[Source {i}: {doc['title']}]\n{content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for answer generation."""
        return """You are a helpful Wikipedia assistant. Your task is to:

1. Answer questions accurately using the provided sources
2. Synthesize information from multiple sources when helpful
3. Be concise but comprehensive (2-4 paragraphs)
4. Cite sources naturally (e.g., "According to Source 1...")
5. If information is incomplete, acknowledge this

Guidelines:
- Focus on directly answering the question
- Use clear, accessible language
- Don't add information not in the sources
- Be honest about limitations in the provided context"""
    
    def _determine_confidence(self, documents: List[Dict], answer: str) -> str:
        """Determine confidence level."""
        if not documents:
            return 'low'
        
        # Check average relevance
        avg_rel = self._avg_relevance(documents)
        
        # Check number of sources
        num_sources = len(documents)
        
        # Check answer length (too short might indicate insufficient info)
        answer_len = len(answer)
        
        if avg_rel >= 0.7 and num_sources >= 3 and answer_len > 200:
            return 'high'
        elif avg_rel >= 0.5 and num_sources >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _avg_relevance(self, documents: List[Dict]) -> float:
        """Calculate average relevance score."""
        if not documents:
            return 0.0
        
        scores = [
            doc.get('relevance_score', doc.get('final_score', doc.get('score', 0)))
            for doc in documents
        ]
        
        return sum(scores) / len(scores)


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_GENERATOR = None

def get_answer_generator() -> AnswerGenerator:
    """Get cached generator instance."""
    global _GENERATOR
    
    if _GENERATOR is None:
        _GENERATOR = AnswerGenerator()
    
    return _GENERATOR
