import requests
import json
import re
from typing import List, Dict, Optional


class FanOutGenerator:
    """
    Generates specialized fan-out queries for Wikipedia RAG.
    
    Decomposes complex queries into:
    - Core concept queries
    - Procedural/how-to queries
    - Prerequisite knowledge queries
    - Safety/troubleshooting queries
    """
    
    def __init__(
        self,
        model: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize fan-out generator.
        
        Args:
            model: Ollama model to use
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        
        # Test connection
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                print(f"✅ Fan-out generator initialized")
                print(f"   Model: {model}")
            else:
                print(f"⚠️  Ollama connection warning: status {response.status_code}")
            self.available = True
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Warning: Cannot connect to Ollama at {base_url}")
            print(f"   Fan-out queries disabled")
            self.available = False
    
    def should_use_fanout(self, query: str) -> bool:
        """
        Determine if query would benefit from fan-out.
        
        Args:
            query: User query
        
        Returns:
            True if fan-out should be used
        """
        query_lower = query.lower()
        
        # Skip simple factual queries
        simple_patterns = [
            r'^what is ',
            r'^who is ',
            r'^when (was|did|is)',
            r'^where (is|was)',
            r'^define ',
        ]
        if any(re.match(pattern, query_lower) for pattern in simple_patterns):
            return False
        
        # Use fan-out for complex queries
        complex_indicators = [
            'how do i', 'how to', 'how can i',
            'fix', 'repair', 'troubleshoot', 'diagnose',
            'build', 'create', 'make', 'develop',
            'learn', 'understand', 'explain',
            ' and ', ' or ',  # Multiple aspects
            'difference between', 'compare',
            'best way to', 'steps to'
        ]
        
        if any(indicator in query_lower for indicator in complex_indicators):
            return True
        
        # Use for longer queries (>8 words)
        if len(query.split()) > 8:
            return True
        
        return False
    
    def generate_fanout_queries(
        self, 
        query: str,
        max_fanouts: int = 5
    ) -> Optional[List[Dict[str, str]]]:
        """
        Generate fan-out queries for a complex query.
        
        Args:
            query: User query
            max_fanouts: Maximum number of fan-out queries
        
        Returns:
            List of {query, purpose} dicts or None if unavailable
        """
        if not self.available:
            return None
        
        # if not self.should_use_fanout(query):
        #     return None
        
        prompt = self._build_fanout_prompt(query, max_fanouts)
        
        try:
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 300,
                    }
                },
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"⚠️  Ollama API error: {response.status_code}")
                return None
            
            result = response.json()
            fanout_text = result.get('response', '').strip()
            
            # Parse fan-out queries
            fanouts = self._parse_fanout_response(fanout_text)
            
            if fanouts:
                print(f"  🎯 Generated {len(fanouts)} fan-out queries")
                for i, fo in enumerate(fanouts, 1):
                    print(f"     {i}. [{fo['purpose']}] {fo['query']}")
            
            return fanouts if fanouts else None
            
        except Exception as e:
            print(f"⚠️  Fan-out generation error: {e}")
            return None
    
    def _build_fanout_prompt(self, query: str, max_fanouts: int) -> str:
        """Build prompt for fan-out generation."""
        return f"""You are a query decomposer for Wikipedia search. Break down the user's query into {max_fanouts} specialized sub-queries.

Each sub-query should have:
- query: A specific search query (3-8 words)
- purpose: Why this query is needed (concept/procedure/prerequisite/safety/troubleshooting)

User query: "{query}"

Generate {max_fanouts} sub-queries in JSON format:
[
  {{"query": "specific search query", "purpose": "concept"}},
  {{"query": "another search query", "purpose": "procedure"}},
  ...
]

Guidelines:
- Focus on DIFFERENT aspects (don't repeat same query with different words)
- Use Wikipedia-friendly search terms
- Cover: core concepts, procedures, prerequisites, safety, troubleshooting
- Keep queries concise (3-8 words)
- Return ONLY the JSON array, no explanations

JSON:"""
    
    def _parse_fanout_response(self, text: str) -> List[Dict[str, str]]:
        """Parse fan-out queries from LLM response."""
        # Try to extract JSON
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if not json_match:
            return []
        
        json_str = json_match.group(0)
        
        try:
            fanouts = json.loads(json_str)
            
            # Validate structure
            valid_fanouts = []
            for fo in fanouts:
                if isinstance(fo, dict) and 'query' in fo and 'purpose' in fo:
                    query = fo['query'].strip()
                    purpose = fo['purpose'].strip()
                    
                    # Basic validation
                    if 3 <= len(query.split()) <= 12 and len(query) > 10:
                        valid_fanouts.append({
                            'query': query,
                            'purpose': purpose
                        })
            
            return valid_fanouts
            
        except json.JSONDecodeError:
            # Fallback: parse line by line
            return self._parse_fallback(text)
    
    def _parse_fallback(self, text: str) -> List[Dict[str, str]]:
        """Fallback parser for non-JSON responses."""
        fanouts = []
        
        # Look for patterns like "query: X, purpose: Y"
        pattern = r'["\']?query["\']?\s*:\s*["\']([^"\']+)["\'].*?["\']?purpose["\']?\s*:\s*["\']([^"\']+)["\']'
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        for query, purpose in matches:
            query = query.strip()
            purpose = purpose.strip()
            
            if 3 <= len(query.split()) <= 12:
                fanouts.append({
                    'query': query,
                    'purpose': purpose
                })
        
        return fanouts


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_FANOUT_GENERATOR = None

def get_fanout_generator(
    model: str = "qwen2.5:0.5b",
    base_url: str = "http://localhost:11434"
) -> FanOutGenerator:
    """Get cached fan-out generator instance."""
    global _FANOUT_GENERATOR
    
    if _FANOUT_GENERATOR is None:
        _FANOUT_GENERATOR = FanOutGenerator(model, base_url)
    
    return _FANOUT_GENERATOR