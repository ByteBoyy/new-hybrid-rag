# src/conversation/ollama_query_rephraser.py
"""
Ollama-based query rephrasing - completely free alternative to OpenAI.
Uses local models like qwen2.5 for query reformulation.
"""
import requests
from typing import List, Dict, Optional


class QueryRephraser:
    """
    Query rephraser using local Ollama models.
    
    Benefits:
    - 🆓 Completely free (no API costs)
    - 🚀 Fast (local inference)
    - 🔒 Private (no data sent to external APIs)
    - 💪 Works offline
    
    Handles:
    - Follow-up questions
    - Pronoun resolution
    - Context-aware rephrasing
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize Ollama-based rephraser.
        
        Args:
            model: Ollama model to use (qwen2.5:0.5b, llama2, etc.)
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✅ Ollama query rephraser initialized")
                print(f"   Model: {model}")
                print(f"   Server: {base_url}")
            else:
                print(f"⚠️  Ollama connection warning: status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Warning: Cannot connect to Ollama at {base_url}")
            print(f"   Make sure Ollama is running: ollama serve")
            print(f"   Falling back to rule-based rephrasing")
            self.ollama_available = False
            return
        
        self.ollama_available = True
    
    def rephrase(
        self, 
        query: str, 
        conversation_history: List[Dict],
        force_rephrase: bool = False
    ) -> Dict:
        """
        Rephrase query using conversation context.
        
        Args:
            query: Current user query
            conversation_history: Previous messages
            force_rephrase: Always rephrase (even if not needed)
        
        Returns:
            Dictionary with:
                - original_query: Original query
                - rephrased_query: Rephrased query
                - needs_context: Whether context was needed
                - method: 'ollama' or 'rule-based' or 'none'
        """
        # Check if rephrasing is needed
        if not force_rephrase and not self._needs_rephrasing(query, conversation_history):
            return {
                'original_query': query,
                'rephrased_query': query,
                'needs_context': False,
                'method': 'none'
            }
        
        # Try Ollama-based rephrasing
        if self.ollama_available:
            try:
                rephrased = self._ollama_rephrase(query, conversation_history)
                return {
                    'original_query': query,
                    'rephrased_query': rephrased,
                    'needs_context': True,
                    'method': 'ollama'
                }
            except Exception as e:
                print(f"⚠️  Ollama rephrasing failed: {e}, falling back to rule-based")
        
        # Fallback: rule-based rephrasing
        rephrased = self._rule_based_rephrase(query, conversation_history)
        return {
            'original_query': query,
            'rephrased_query': rephrased,
            'needs_context': True,
            'method': 'rule-based'
        }
    
    def _needs_rephrasing(self, query: str, history: List[Dict]) -> bool:
        """Check if query needs rephrasing."""
        if not history:
            return False
        
        query_lower = query.lower()
        
        # Check for pronouns
        pronouns = ['it', 'its', 'that', 'this', 'these', 'those', 'they', 'them', 'their', 'he', 'she']
        if any(f' {p} ' in f' {query_lower} ' for p in pronouns):
            return True
        
        # Check for follow-up phrases
        follow_up_phrases = [
            'what about', 'how about', 'tell me more', 'more about',
            'explain that', 'and what', 'also', 'additionally',
            'can you', 'could you', 'what else', 'any other'
        ]
        if any(query_lower.startswith(phrase) for phrase in follow_up_phrases):
            return True
        
        # Very short queries
        if len(query) < 10 and len(query.split()) <= 2:
            return True
        
        return False
    
    def _ollama_rephrase(self, query: str, history: List[Dict]) -> str:
        """Use Ollama to rephrase query."""
        # Build context (last 4 messages)
        context_messages = history[-4:] if len(history) > 4 else history
        
        context_str = ""
        for msg in context_messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            content = msg['content'][:300]
            context_str += f"{role}: {content}\n"
        
        # Optimized prompt for small models
        prompt = f"""Your job is to rewrite the follow-up question so it is completely clear even if read alone.

Steps:
1. Identify what the user is talking about (e.g., "WiFi keeps cutting out").
2. Replace vague words like "it", "this", "that", "they" with those nouns.
3. Combine both pieces of information into one clear question.
4. Output only the rewritten question.

Example 2:
Previous conversation:
User: I’ve been having headaches almost every morning lately.
Follow-up question: Could it be because of my new glasses?
Rewritten question: Could my new glasses be causing my headaches every morning?

Example 3:
Previous conversation:
User: I’m trying to lose weight but nothing seems to work.
Follow-up question: Should I cut carbs completely?
Rewritten question: Should I cut carbs completely to lose weight since nothing else seems to work?

Example 4:
Previous conversation:
User: I applied for the marketing job at Meta last week.
Follow-up question: When do you think they’ll respond?
Rewritten question: When do you think Meta will respond to my marketing job application I submitted last week?

Example 5:
Previous conversation:
User: My laptop battery drains from 100% to 50% in less than an hour.
Follow-up question: Do I need to replace it?
Rewritten question: Do I need to replace my laptop battery since it drains from 100% to 50% in less than an hour?

Example 6:
Previous conversation:
User: I just started a new workout plan and my muscles are super sore.
Follow-up question: Is that normal?
Rewritten question: Is it normal for my muscles to feel very sore after starting a new workout plan?

Example 7:
Previous conversation:
User: I think my cat might be sick — she’s been hiding all day and not eating much.
Follow-up question: Should I take her to the vet?
Rewritten question: Should I take my cat to the vet since she’s been hiding all day and not eating much?

Example 8:
Previous conversation:
User: I recently switched to a standing desk but my back hurts more now.
Follow-up question: Am I using it wrong?
Rewritten question: Am I using my standing desk incorrectly since my back hurts more after switching to it?

Example 9:
Previous conversation:
User: My phone keeps overheating whenever I play games for more than 10 minutes.
Follow-up question: Is that dangerous?
Rewritten question: Is it dangerous that my phone overheats whenever I play games for more than 10 minutes?

Example 10:
Previous conversation:
User: I’ve been feeling super tired even after sleeping 8 hours a night.
Follow-up question: Could it be stress?
Rewritten question: Could stress be the reason I feel super tired even after sleeping 8 hours a night?

        
        
        
Previous conversation:
{context_str}

Follow-up question: {query}

Rewritten question:"""
        
        # Call Ollama
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 50,
                        "stop": ["\n", "Follow-up", "Instructions:", "Task:"]
                    }
                },
                timeout=10
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            rephrased = result.get('response', '').strip()
            
            # Clean up
            rephrased = self._clean_rephrased_query(rephrased, query)
            
            return rephrased
            
        except Exception as e:
            print(f"⚠️  Ollama error: {e}")
            return query
    
    def _clean_rephrased_query(self, rephrased: str, original: str) -> str:
        """Clean up rephrased query."""
        # Remove common prefixes
        prefixes = [
            "rewritten question:",
            "standalone question:",
            "the question is:",
            "answer:",
            "question:",
        ]
        
        rephrased_lower = rephrased.lower()
        for prefix in prefixes:
            if rephrased_lower.startswith(prefix):
                rephrased = rephrased[len(prefix):].strip()
        
        # Remove quotes
        if rephrased.startswith('"') and rephrased.endswith('"'):
            rephrased = rephrased[1:-1]
        if rephrased.startswith("'") and rephrased.endswith("'"):
            rephrased = rephrased[1:-1]
        
        # Validate
        if len(rephrased) < 5 or not rephrased[0].isupper():
            return original
        
        if '\n' in rephrased:
            rephrased = rephrased.split('\n')[0].strip()
        
        return rephrased
    
    def _rule_based_rephrase(self, query: str, history: List[Dict]) -> str:
        """Simple rule-based rephrasing (fallback)."""
        if not history:
            return query
        
        # Get last assistant message
        last_assistant = None
        for msg in reversed(history):
            if msg['role'] == 'assistant':
                last_assistant = msg['content']
                break
        
        if not last_assistant:
            return query
        
        # Extract topic
        first_sentence = last_assistant.split('.')[0][:200]
        words = first_sentence.split()
        topics = [w for w in words if w and w[0].isupper() and len(w) > 3]
        
        if not topics:
            return query
        
        topic = topics[0]
        query_lower = query.lower()
        
        # Replace pronouns
        if ' it ' in f' {query_lower} ' or query_lower.startswith('it '):
            query = query.replace(' it ', f' {topic} ')
            query = query.replace('It ', f'{topic} ')
            query = query.replace(' it?', f' {topic}?')
        
        if ' that ' in f' {query_lower} ' or query_lower.startswith('that '):
            query = query.replace(' that ', f' {topic} ')
            query = query.replace('That ', f'{topic} ')
        
        return query


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_OLLAMA_REPHRASER = None

def get_query_rephraser(
    model: str = "qwen2.5:0.5b",
    base_url: str = "http://localhost:11434"
) -> QueryRephraser:
    """Get cached Ollama query rephraser instance."""
    global _OLLAMA_REPHRASER
    
    if _OLLAMA_REPHRASER is None:
        _OLLAMA_REPHRASER = QueryRephraser(model, base_url)
    
    return _OLLAMA_REPHRASER