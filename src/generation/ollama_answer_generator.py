import requests
from typing import List, Dict


class OllamaGenerator:
    
    def __init__(
        self, 
        model: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434"
    ):

        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✅ Ollama connected: {base_url}")
                print(f"   Using model: {model}")
            else:
                print(f"⚠️  Ollama connection warning: status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Warning: Cannot connect to Ollama at {base_url}")
            print(f"   Make sure Ollama is running: ollama serve")
            print(f"   Error: {e}")
    
    def generate(self, query: str, documents: List[Dict]) -> Dict:
        if not documents:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'confidence': 'low',
                'reasoning': "No documents retrieved"
            }
        
        
        if not documents:
            return {
                'answer': "I couldn't find relevant information to answer your question.",
                'sources': [],
                'confidence': 'low',
                'reasoning': "No documents retrieved"
            }
        
        context = self._build_context(documents)
        
        prompt = self._build_prompt(query, context)
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 512,  # Max tokens
                    }
                },
                timeout=30
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            answer = result.get('response', '').strip()
            
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
                'reasoning': f"Answer based on {len(documents)} sources using {self.model}"
            }
            
        except requests.exceptions.Timeout:
            return {
                'answer': "Request timed out. The model may be taking too long to respond.",
                'sources': [],
                'confidence': 'low',
                'reasoning': "Timeout error"
            }
        except Exception as e:
            print(f"Error generating answer with Ollama: {e}")
            return {
                'answer': f"Error generating answer: {str(e)}",
                'sources': [],
                'confidence': 'low',
                'reasoning': f"Generation error: {str(e)}"
            }
    
    def _build_context(self, documents: List[Dict]) -> str:
        context_parts = []
        
        for i, doc in enumerate(documents[:5], 1):  # Top 5
            summary = doc.get('summary', '').strip()
            text = doc.get('text', '').strip()
            
            # Use summary + first part of text
            if summary:
                content = f"{summary}\n\n{text[:800]}" if text else summary
            else:
                content = text[:1200] if text else "No content"
            
            context_parts.append(
                f"[Source {i}: {doc['title']}]\n{content}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        return f"""You are a helpful Wikipedia assistant. Answer the question using only the provided sources.

Sources:
{context}

Question: {query}

Instructions:
1. Answer directly and concisely (2-3 paragraphs)
2. Use information from the sources above
3. Mention which source number you used (e.g., "According to Source 1...")
4. If information is incomplete, say so
5. Do not add information not in the sources

Answer:"""
    
    def _determine_confidence(self, documents: List[Dict], answer: str) -> str:
        if not documents:
            return 'low'
        
        avg_rel = self._avg_relevance(documents)
        
        num_sources = len(documents)
        
        answer_len = len(answer)
        
        if avg_rel >= 0.7 and num_sources >= 3 and answer_len > 150:
            return 'high'
        elif avg_rel >= 0.5 and num_sources >= 2 and answer_len > 80:
            return 'medium'
        else:
            return 'low'
    
    def _avg_relevance(self, documents: List[Dict]) -> float:
        if not documents:
            return 0.0
        
        scores = [
            doc.get('relevance_score', doc.get('final_score', doc.get('score', 0)))
            for doc in documents
        ]
        
        return sum(scores) / len(scores)



_OLLAMA_GENERATOR = None

def get_ollama_generator(
    model: str = "qwen2.5:0.5b",
    base_url: str = "http://localhost:11434"
) -> OllamaGenerator:

    global _OLLAMA_GENERATOR
    
    if _OLLAMA_GENERATOR is None:
        _OLLAMA_GENERATOR = OllamaGenerator(model, base_url)
    
    return _OLLAMA_GENERATOR