from datetime import datetime
from typing import Any, Dict, List, Optional

from config.config import config


class SharedState:
    def __init__(self):
        self.state: Dict[str, Any] = {
            # Existing keys
            config.StateKeys.CURRENT_TOOL: None,
            config.StateKeys.CURRENT_AGENT: None,
            config.StateKeys.RESPONSE: None,
            config.StateKeys.ERROR: None,
            config.StateKeys.CHAT_HISTORY: [],
            config.StateKeys.USER_INFO: {},
            config.StateKeys.MEMORY: {},
            # New StarkQA specific keys
            config.StateKeys.EMBEDDINGS: None,
            config.StateKeys.QUERY_EMBEDDINGS: None,
            config.StateKeys.NODE_EMBEDDINGS: None,
            config.StateKeys.EVALUATION_RESULTS: None,
            config.StateKeys.METRICS: None,
            config.StateKeys.KNOWLEDGE_GRAPH: None,
            config.StateKeys.BATCH_CONFIG: None,
            config.StateKeys.EVALUATION_CONFIG: None,
        }

    def get(self, key: str) -> Any:
        """Get value from state"""
        return self.state.get(key)

    def set(self, key: str, value: Any) -> None:
        """Set value in state"""
        self.state[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """Update multiple values in state"""
        self.state.update(updates)

    def clear_response(self) -> None:
        """Clear response and error fields"""
        self.state[config.StateKeys.RESPONSE] = None
        self.state[config.StateKeys.ERROR] = None

    def add_to_chat_history(self, role: str, content: str) -> None:
        """Add a message to chat history"""
        self.state[config.StateKeys.CHAT_HISTORY].append(
            {"role": role, "content": content, "timestamp": datetime.now().isoformat()}
        )

    def update_user_info(self, info: Dict[str, Any]) -> None:
        """Update user information"""
        self.state[config.StateKeys.USER_INFO].update(info)

    def get_chat_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent chat history"""
        history = self.state[config.StateKeys.CHAT_HISTORY]
        if limit:
            return history[-limit:]
        return history

    def update_memory(self, key: str, value: Any) -> None:
        """Update memory with key-value pair"""
        self.state[config.StateKeys.MEMORY][key] = value

    def get_from_memory(self, key: str) -> Any:
        """Get value from memory"""
        return self.state[config.StateKeys.MEMORY].get(key)

    def get_current_context(self) -> Dict[str, Any]:
        """Get current context information"""
        return {
            "current_agent": self.state[config.StateKeys.CURRENT_AGENT],
            "current_tool": self.state[config.StateKeys.CURRENT_TOOL],
            "user_info": self.state[config.StateKeys.USER_INFO],
            "recent_messages": self.get_chat_history(limit=5),
            # Add StarkQA specific context
            "evaluation_results": self.state[config.StateKeys.EVALUATION_RESULTS],
            "metrics": self.state[config.StateKeys.METRICS],
            "batch_config": self.state[config.StateKeys.BATCH_CONFIG],
        }


# Create a global instance
shared_state = SharedState()
