"""Imagination system — real-time entity design from novel percept mentions.

When the agent encounters something unfamiliar in its environment (a novel
entity that has no existing SEM component), the imagination system:

1. Extracts entity-like noun phrases from percept text
2. Checks the ComponentIndex for known matches (alias or embedding)
3. If truly novel: gates on DN arousal + energy budget, then designs a
   temporary SEM entity via EntityDesigner
4. Registers the result as an ephemeral component (session-scoped)
5. Tags any resulting Episodes and CausalLinks with ``imagined=True``

Bio-metaphor: imagination fires during low-arousal idle states — you
don't daydream while fighting.
"""

from maxim.imagination.cache import ImaginationCache, ImaginationResult
from maxim.imagination.trigger import ImaginationTrigger, extract_entity_phrases

__all__ = [
    "ImaginationCache",
    "ImaginationResult",
    "ImaginationTrigger",
    "extract_entity_phrases",
]
