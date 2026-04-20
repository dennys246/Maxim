"""Imagination trigger — entity extraction + ComponentIndex lookup + design dispatch.

The trigger runs post-state.update() in the agent loop. For each
novel entity phrase that isn't in the ComponentIndex, it dispatches
to ImaginationDesigner (I2) for real-time entity generation.

Entity extraction uses lightweight NLP heuristics inspired by
memory/concept_extractor.py. No spaCy required — uses regex + POS
heuristics for SEM-relevant noun phrases (physical objects, creatures,
environments, weapons, vehicles, items).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import TYPE_CHECKING, Any

from maxim.imagination.cache import ImaginationCache, ImaginationResult

if TYPE_CHECKING:
    from maxim.default_network.network import DefaultNetwork
    from maxim.embodiment.component_index import ComponentIndex
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.imagination.designer import ImaginationDesigner
    from maxim.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Entity noun-phrase extraction
# ---------------------------------------------------------------------------

# Words that look entity-like but are NOT SEM-relevant (abstract concepts,
# furniture, clothing, body parts, etc.)
_STOP_WORDS = frozenset(
    {
        # Abstract / meta
        "thing",
        "something",
        "nothing",
        "everything",
        "anything",
        "way",
        "time",
        "place",
        "world",
        "story",
        "plan",
        "idea",
        "thought",
        "feeling",
        "sense",
        "moment",
        "reason",
        "question",
        "answer",
        "problem",
        "solution",
        "situation",
        "action",
        "option",
        "choice",
        "decision",
        "result",
        "effect",
        "level",
        "type",
        "kind",
        "sort",
        "part",
        "side",
        "end",
        "point",
        "fact",
        "case",
        "area",
        "turn",
        "step",
        # Body parts (not SEM entities)
        "hand",
        "hands",
        "arm",
        "arms",
        "leg",
        "legs",
        "head",
        "eye",
        "eyes",
        "face",
        "foot",
        "feet",
        "finger",
        "fingers",
        "body",
        "back",
        "shoulder",
        "shoulders",
        # Clothing (not SEM entities)
        "shirt",
        "pants",
        "boots",
        "cloak",
        "robe",
        "hat",
        "hood",
        "gloves",
        "belt",
        # Common furniture (not SEM-relevant unless environmental)
        "chair",
        "table",
        "bed",
        "bench",
        # Pronouns / determiners
        "you",
        "your",
        "it",
        "its",
        "this",
        "that",
        "one",
        "ones",
        "self",
    }
)

# SEM-relevant categories — phrases containing these words are entity candidates
_ENTITY_INDICATORS = frozenset(
    {
        # Creatures
        "wolf",
        "dragon",
        "spider",
        "snake",
        "bear",
        "rat",
        "bat",
        "goblin",
        "orc",
        "troll",
        "zombie",
        "skeleton",
        "ghost",
        "demon",
        "elemental",
        "golem",
        "beast",
        "creature",
        "monster",
        "animal",
        "insect",
        "drone",
        "robot",
        "android",
        "cyborg",
        "dog",
        "hound",
        "guard",
        "sentinel",
        # Weapons
        "sword",
        "axe",
        "bow",
        "staff",
        "dagger",
        "spear",
        "mace",
        "hammer",
        "blade",
        "crossbow",
        "gun",
        "rifle",
        "pistol",
        "laser",
        "cannon",
        "turret",
        # Items
        "potion",
        "scroll",
        "amulet",
        "ring",
        "gem",
        "crystal",
        "key",
        "flask",
        "vial",
        "lantern",
        "torch",
        "device",
        "gadget",
        "terminal",
        "console",
        "laptop",
        "chip",
        # Vehicles
        "cart",
        "wagon",
        "ship",
        "boat",
        "speeder",
        "hover",
        "vehicle",
        "bike",
        "motorcycle",
        "car",
        # Environmental features
        "door",
        "gate",
        "bridge",
        "lever",
        "switch",
        "trap",
        "pit",
        "altar",
        "shrine",
        "fountain",
        "well",
        "chest",
        "crate",
        "barrel",
        "pillar",
        "statue",
        "portal",
        "pedestal",
        "mechanism",
        "lock",
        "panel",
        # NPCs
        "merchant",
        "blacksmith",
        "healer",
        "priest",
        "mage",
        "wizard",
        "warrior",
        "knight",
        "thief",
        "rogue",
        "assassin",
        "archer",
        "captain",
        "commander",
        "king",
        "queen",
        "villager",
        "innkeeper",
        "bartender",
    }
)

# Sentence-level patterns that introduce entities: "you see X", "there is X",
# "a X stands", "X appears", "X blocks", etc.
# Use word boundary or common prepositions/conjunctions as terminators to
# avoid swallowing the rest of the sentence.
_INTRO_PATTERNS = [
    re.compile(
        r"(?:you\s+(?:see|notice|spot|observe|find|discover))\s+(?:a|an|the|some)\s+"
        r"(.+?)(?:\s+(?:and|on|in|at|near|by|from|with|behind|inside|lying|resting|sitting)\b|\.|,|;|$)",
        re.I,
    ),
    re.compile(
        r"(?:there\s+(?:is|are|stands?|sits?))\s+(?:a|an|the|some)\s+"
        r"(.+?)(?:\s+(?:and|on|in|at|near|by|from|with|behind|inside)\b|\.|,|;|$)",
        re.I,
    ),
    re.compile(r"(?:a|an|the)\s+(.+?)\s+(?:appears?|emerges?|blocks?|guards?|lurks?|waits?)", re.I),
    re.compile(r"(?:a|an|the)\s+(.+?)\s+(?:lies|rests?|hangs?|leans?|stands?)\s", re.I),
]


def extract_entity_phrases(text: str) -> list[str]:
    """Extract SEM-relevant entity noun phrases from percept text.

    Returns deduplicated, normalized phrases in order of first occurrence.
    Filters out abstract concepts, body parts, clothing — only returns
    physical objects, creatures, weapons, items, environmental features,
    vehicles, and NPCs that could be SEM entities.

    Args:
        text: Raw percept text (narration, scene description, etc.)

    Returns:
        List of candidate entity phrases (lowercase, stripped).
    """
    if not text or not text.strip():
        return []

    seen: set[str] = set()
    seen_heads: set[str] = set()
    results: list[str] = []

    def _maybe_add(phrase: str) -> None:
        """Validate and add a candidate phrase."""
        phrase = phrase.strip().lower()
        # Remove leading determiners
        phrase = re.sub(r"^(?:a|an|the|some)\s+", "", phrase)
        # Remove trailing punctuation
        phrase = re.sub(r"[.,;:!?]+$", "", phrase).strip()
        if not phrase or len(phrase) < 3:
            return
        words = phrase.split()
        if len(words) > 4 or len(words) == 0:
            return

        # Stem the head noun (avoid stripping "s" from words ending in "ss")
        raw_head = words[-1]
        head_noun = raw_head.rstrip("s") if not raw_head.endswith("ss") else raw_head
        if head_noun in _STOP_WORDS or phrase in _STOP_WORDS:
            return

        # Check if any word is an entity indicator
        has_indicator = any(
            (w.rstrip("s") if not w.endswith("ss") else w) in _ENTITY_INDICATORS or w in _ENTITY_INDICATORS
            for w in words
        )

        if not has_indicator:
            return

        # Deduplicate on head noun — "wolf", "large wolf", "the wolf" collapse
        if head_noun in seen_heads:
            return
        seen_heads.add(head_noun)

        if phrase not in seen:
            seen.add(phrase)
            results.append(phrase)

    # Strategy 1: Sentence-level intro patterns (highest confidence)
    for pat in _INTRO_PATTERNS:
        for m in pat.finditer(text):
            candidate = m.group(1).strip()
            # May contain multiple entities joined by "and"
            for part in re.split(r"\s+and\s+", candidate):
                _maybe_add(part)

    # Strategy 2: Scan for entity indicator words with preceding adjectives.
    # For each word that matches an entity indicator, collect up to 3
    # preceding adjectives to form a phrase.
    words = re.findall(r"[a-z]+-?[a-z]*", text.lower())
    for i, word in enumerate(words):
        stem = word.rstrip("s") if not word.endswith("ss") else word
        if stem not in _ENTITY_INDICATORS and word not in _ENTITY_INDICATORS:
            continue
        # Collect up to 3 preceding words as adjectives
        start = max(0, i - 3)
        candidate_words = words[start : i + 1]
        # Filter out stop words and determiners from the front
        while candidate_words and candidate_words[0] in (
            "a",
            "an",
            "the",
            "some",
            "this",
            "that",
            "its",
            "your",
            "my",
        ):
            candidate_words = candidate_words[1:]
        if candidate_words:
            _maybe_add(" ".join(candidate_words))

    return results


# ---------------------------------------------------------------------------
# Imagination trigger
# ---------------------------------------------------------------------------


class ImaginationTrigger:
    """Orchestrates entity extraction → cache → index → design pipeline.

    Wired into the agent loop post-state.update(). Thread-safe.

    Args:
        component_index: Semantic component lookup.
        component_registry: For ephemeral registration.
        designer: ImaginationDesigner for LLM-based entity design.
        cache: Session-scoped imagination cache.
        tool_registry: For registering affordance tools (I3 API).
        default_network: For arousal gating (optional — if None, arousal
            gate is always open).
        imagination_threshold: Number of mentions before triggering
            imagination for a novel phrase.
        enabled: Master enable/disable switch.
    """

    def __init__(
        self,
        component_index: ComponentIndex,
        component_registry: ComponentRegistry,
        designer: ImaginationDesigner | None = None,
        cache: ImaginationCache | None = None,
        tool_registry: ToolRegistry | None = None,
        default_network: DefaultNetwork | None = None,
        *,
        imagination_threshold: int = 2,
        enabled: bool = True,
    ) -> None:
        self._index = component_index
        self._registry = component_registry
        self._designer = designer
        self._cache = cache or ImaginationCache()
        self._tool_registry = tool_registry
        self._dn = default_network
        self._threshold = imagination_threshold
        self._enabled = enabled
        self._lock = threading.RLock()
        # Per-phrase design guard — prevents concurrent LLM calls for the
        # same phrase when AUT + orchestrator process simultaneously.
        self._designing: set[str] = set()

        # Track imagined entity refs for provenance tagging at session end
        self._imagined_refs: set[str] = set()

        # Stats
        self._phrases_extracted = 0
        self._cache_hits = 0
        self._index_hits = 0
        self._designs_attempted = 0
        self._designs_succeeded = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def cache(self) -> ImaginationCache:
        return self._cache

    @property
    def imagined_refs(self) -> frozenset[str]:
        """Return the set of imagined entity refs created this session.

        Used at session end to retroactively tag CausalLinks and Episodes
        involving these entities with ``imagined=True`` provenance.
        """
        with self._lock:
            return frozenset(self._imagined_refs)

    def process_percept(
        self,
        percept_text: str,
        scene_context: dict[str, Any] | None = None,
        scene_id: str | None = None,
    ) -> list[ImaginationResult]:
        """Run the full imagination pipeline on a percept text.

        1. Extract entity noun phrases.
        2. For each phrase: check cache → check index → design if novel.
        3. Register ephemeral entities + affordance tools.

        Returns list of ImaginationResults (both cached and newly resolved).
        """
        if not self._enabled:
            return []

        # Extract candidates
        phrases = extract_entity_phrases(percept_text)
        if not phrases:
            return []

        with self._lock:
            self._phrases_extracted += len(phrases)

        results: list[ImaginationResult] = []

        for phrase in phrases:
            result = self._resolve_phrase(phrase, scene_context, scene_id)
            if result is not None:
                results.append(result)

        return results

    def _resolve_phrase(
        self,
        phrase: str,
        scene_context: dict[str, Any] | None,
        scene_id: str | None,
    ) -> ImaginationResult | None:
        """Resolve a single entity phrase through cache → index → design."""
        # 1. Check cache first (avoids repeated embedding computations)
        cached = self._cache.get(phrase)
        if cached is not None:
            with self._lock:
                self._cache_hits += 1
            return cached

        # 2. Record mention and check threshold
        count = self._cache.record_mention(phrase)

        # 3. Check ComponentIndex for existing match
        match = self._index.find(phrase)
        if match is not None:
            result = ImaginationResult(
                phrase=phrase,
                ref=match.ref,
                imagined=False,
                score=match.score,
            )
            self._cache.put(result)
            with self._lock:
                self._index_hits += 1
            log.debug(
                "Imagination: '%s' matched existing component '%s' (score=%.2f, layer=%s)",
                phrase,
                match.ref,
                match.score,
                match.layer,
            )
            return result

        # 4. Below threshold → not enough mentions yet, skip
        if count < self._threshold:
            return None

        # 5. Check DN arousal gate — only imagine during low arousal
        if not self._is_arousal_allowed():
            log.debug("Imagination: arousal gate blocked design for '%s'", phrase)
            return None

        # 6. Check energy budget
        if not self._is_energy_available():
            log.debug("Imagination: energy gate blocked design for '%s'", phrase)
            return None

        # 7. Design the entity
        if self._designer is None:
            log.debug("Imagination: no designer available, skipping '%s'", phrase)
            return None

        # Per-phrase guard: prevent concurrent LLM design for the same phrase
        normalized = self._cache.normalize(phrase)
        with self._lock:
            if normalized in self._designing:
                return None  # Another thread is already designing this
            self._designing.add(normalized)
            self._designs_attempted += 1

        try:
            design_result = self._designer.imagine(phrase, scene_context or {})
        except Exception as e:
            log.warning("Imagination: design failed for '%s': %s", phrase, e, exc_info=True)
            return None
        finally:
            with self._lock:
                self._designing.discard(normalized)

        if design_result is None:
            return None

        spec = design_result.spec
        ref = design_result.ref

        # 8. Register ephemeral component
        self._registry.register_ephemeral(
            ref,
            spec,
            provenance="imagined",
        )

        # 9. Add to ComponentIndex for future lookups
        synonyms = design_result.synonyms
        self._index.add(ref, spec, synonyms=synonyms)

        # 10. Register affordance tools if tool_registry available
        if self._tool_registry is not None and scene_id is not None:
            try:
                from maxim.embodiment.tool_bridge import generate_tools_for_entity
                from maxim.embodiment.spec import _parse_entity

                entity = _parse_entity(spec.get("entity", spec))
                tools = generate_tools_for_entity(entity)
                if tools:
                    self._tool_registry.register_scene_tools(tools, scene_id)
                    log.info(
                        "Imagination: registered %d tools for imagined entity '%s'",
                        len(tools),
                        ref,
                    )
            except Exception as e:
                log.warning("Imagination: tool registration failed for '%s': %s", ref, e)

        result = ImaginationResult(
            phrase=phrase,
            ref=ref,
            imagined=True,
            score=1.0,
            spec=spec,
            validation_warnings=design_result.validation_warnings,
        )
        self._cache.put(result)

        with self._lock:
            self._designs_succeeded += 1
            self._imagined_refs.add(ref)

        log.info("Imagination: designed new entity '%s' from phrase '%s'", ref, phrase)
        return result

    def _is_arousal_allowed(self) -> bool:
        """Check DN arousal gate — imagination only during low arousal."""
        if self._dn is None:
            return True  # No DN → gate is open
        return self._dn.imagination_allowed()

    def _is_energy_available(self) -> bool:
        """Check energy budget — skip imagination if LLM energy is critical."""
        try:
            from maxim.energy.registry import get_global_registry

            registry = get_global_registry()
            if registry.is_critical_energy("llm"):
                return False
        except Exception:
            pass  # Energy system not available → allow
        return True

    def stats(self) -> dict[str, int]:
        """Return imagination trigger statistics."""
        with self._lock:
            return {
                "phrases_extracted": self._phrases_extracted,
                "cache_hits": self._cache_hits,
                "index_hits": self._index_hits,
                "designs_attempted": self._designs_attempted,
                "designs_succeeded": self._designs_succeeded,
                "cache_size": self._cache.size,
            }

    def clear_session(self) -> None:
        """Clear session state (called at session end)."""
        self._cache.clear()
        with self._lock:
            self._phrases_extracted = 0
            self._cache_hits = 0
            self._index_hits = 0
            self._designs_attempted = 0
            self._designs_succeeded = 0
