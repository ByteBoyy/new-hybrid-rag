# src/conversation/memory.py
"""
Conversation memory management for multi-turn chat.
Stores conversation history and provides context for query rephrasing.
"""
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
import json


class ConversationMemory:
    """
    In-memory conversation storage.
    
    For production, replace with Redis or PostgreSQL.
    """
    
    def __init__(self, max_history: int = 10):
        """
        Initialize conversation memory.
        
        Args:
            max_history: Maximum number of messages to keep per session
        """
        self.max_history = max_history
        self.sessions = defaultdict(list)  # session_id -> list of messages
    
    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str,
        metadata: Optional[Dict] = None
    ):
        """
        Add message to conversation history.
        
        Args:
            session_id: Unique session identifier
            role: 'user' or 'assistant'
            content: Message content
            metadata: Optional metadata (sources, confidence, etc.)
        """
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.utcnow().isoformat(),
            'metadata': metadata or {}
        }
        
        self.sessions[session_id].append(message)
        
        # Trim history if too long
        if len(self.sessions[session_id]) > self.max_history * 2:  # 2x for user+assistant pairs
            self.sessions[session_id] = self.sessions[session_id][-(self.max_history * 2):]
    
    def get_history(
        self, 
        session_id: str, 
        last_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history for session.
        
        Args:
            session_id: Session identifier
            last_n: Return only last N messages (default: all)
        
        Returns:
            List of messages in chronological order
        """
        history = self.sessions.get(session_id, [])
        
        if last_n:
            return history[-last_n:]
        
        return history
    
    def get_context_window(
        self, 
        session_id: str, 
        window_size: int = 4
    ) -> List[Dict]:
        """
        Get recent conversation context for query rephrasing.
        
        Args:
            session_id: Session identifier
            window_size: Number of recent messages to include
        
        Returns:
            List of recent messages
        """
        return self.get_history(session_id, last_n=window_size)
    
    def clear_session(self, session_id: str):
        """Clear conversation history for session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def get_all_sessions(self) -> List[str]:
        """Get all active session IDs."""
        return list(self.sessions.keys())
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of session."""
        history = self.sessions.get(session_id, [])
        
        if not history:
            return {
                'session_id': session_id,
                'message_count': 0,
                'started_at': None,
                'last_activity': None
            }
        
        return {
            'session_id': session_id,
            'message_count': len(history),
            'started_at': history[0]['timestamp'],
            'last_activity': history[-1]['timestamp'],
            'user_messages': sum(1 for m in history if m['role'] == 'user'),
            'assistant_messages': sum(1 for m in history if m['role'] == 'assistant')
        }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_MEMORY = None

def get_conversation_memory(max_history: int = 10) -> ConversationMemory:
    """Get cached conversation memory instance."""
    global _MEMORY
    
    if _MEMORY is None:
        _MEMORY = ConversationMemory(max_history)
    
    return _MEMORY