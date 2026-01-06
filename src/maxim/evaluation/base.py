from abc import ABC, abstractmethod
from typing import Any, Dict

class Evaluator(ABC):
    """
    Base class for all evaluation modules.
    """
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns a structured evaluation dictionary.
        """
        pass