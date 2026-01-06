from __future__ import annotations

from .base import Environment
from typing import Dict, Any, List
from datetime import datetime

class InternetEnv(Environment):
    """
    Represents the observable state of internet interactions.
    """

    def __init__(self):
        self.current_url = None
        self.history: List[Dict[str, Any]] = []
        self.last_response = None
        self.blocked = False
        self._done = False

    def reset(self) -> Dict[str, Any]:
        self.current_url = None
        self.history.clear()
        self.last_response = None
        self.blocked = False
        self._done = False
        return self.observe()
    
    def observe(self) -> Dict[str, Any]:
        return {
            "current_url": self.current_url,
            "last_response": self.last_response,
            "history": self.history[-5:],  # keep window small
            "blocked": self.blocked,
        }
    
    def step(self, result) -> Dict[str, Any]:
        """
        Update environment based on HTTPTool result.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "url": result.url,
            "status": result.status,
            "success": result.success,
        }

        self.history.append(entry)

        if result.success:
            self.current_url = result.url
            self.last_response = {
                "status": result.status,
                "content_type": result.content_type,
                "content_summary": result.summary,  # pre-processed by tool
            }
        else:
            self.last_response = {
                "error": result.error
            }

        if result.status in (403, 429):
            self.blocked = True

        return self.observe()
    
    def is_done(self) -> bool:
        return self._done
