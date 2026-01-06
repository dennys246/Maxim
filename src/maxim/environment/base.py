# environment/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

class Environment(ABC):
    """
    Abstract environment interface.
    """

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to a clean state.
        Returns initial observation.
        """
        pass

    @abstractmethod
    def observe(self) -> Dict[str, Any]:
        """
        Return current observation of the environment.
        """
        pass

    @abstractmethod
    def step(self, event: Any) -> Dict[str, Any]:
        """
        Update environment state from an event (e.g., a tool result) and return an observation.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        """
        Signal terminal condition.
        """
        pass
