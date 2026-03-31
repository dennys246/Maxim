"""Contextual skill prompts — semantic matching for context-sensitive
prompt fragments that flow through the existing ExecAgent pipeline.

Skills are `.txt` files with an ``# intent:`` header that gets embedded
once at load time and compared against the current context via cosine
similarity, not substring matching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.agents.bus import StructuredContext
    from maxim.integration.memory_hub import MemoryHub

logger = logging.getLogger(__name__)


@dataclass
class SkillPrompt:
    """A loaded skill prompt with its intent embedding."""

    name: str
    intent: str
    intent_embedding: list[float]
    requires_mode: str | None
    prompt_text: str
    trusted: bool = True  # False for agent-generated skills


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a < 1e-15 or norm_b < 1e-15:
        return 0.0
    return dot / (norm_a * norm_b)


class SkillMatcher:
    """Matches skill prompts to the current context via semantic similarity.

    Skills are loaded from text files with header metadata:
        # intent: control smart home devices, lights, thermostat
        # mode: active

        When controlling home devices, use http_fetch to call the Home Assistant
        API. Lights: POST /api/services/light/turn_on ...
    """

    def __init__(
        self,
        skills_dir: str | None = None,
        agent_skills_dir: str | None = None,
        embedder: Any = None,
        memory_hub: "MemoryHub | None" = None,
    ) -> None:
        self.skills: list[SkillPrompt] = []
        self.embedder = embedder
        self._memory_hub = memory_hub

        if embedder is None:
            logger.debug("No embedder provided — skill matching disabled")
            return

        # Human-written skills (higher trust)
        if skills_dir:
            skills_path = Path(skills_dir)
            if skills_path.exists():
                for path in sorted(skills_path.glob("*.txt")):
                    self._load_skill(path, trusted=True)

        # Agent-generated skills (lower trust)
        if agent_skills_dir:
            agent_path = Path(agent_skills_dir)
            if agent_path.exists():
                for path in sorted(agent_path.glob("*.txt")):
                    self._load_skill(path, trusted=False)

    def _load_skill(self, path: Path, trusted: bool) -> None:
        """Parse a skill file and compute its intent embedding."""
        try:
            meta, body = self._parse_header(path)
        except Exception:
            logger.warning("Failed to parse skill file: %s", path)
            return

        intent = meta.get("intent", "")
        if not intent:
            logger.warning("Skill file %s has no intent declaration", path)
            return

        # Human skills override agent skills with the same name
        if not trusted and any(s.name == path.stem for s in self.skills):
            return

        intent_embedding = self.embedder.embed(intent)
        self.skills.append(
            SkillPrompt(
                name=path.stem,
                intent=intent,
                intent_embedding=intent_embedding,
                requires_mode=meta.get("mode"),
                prompt_text=body,
                trusted=trusted,
            )
        )

    def match(
        self,
        ctx: "StructuredContext",
        threshold: float = 0.65,
    ) -> list[SkillPrompt]:
        """Match skills by semantic similarity to current context.

        Uses ATL concept boost if available — skills that relate to
        known semantic concepts get a scoring boost.
        """
        if not self.skills or self.embedder is None:
            return []

        context_text = self._extract_searchable_text(ctx)
        if not context_text:
            return []

        context_embedding = self.embedder.embed(context_text)
        matched: list[tuple[SkillPrompt, float]] = []

        for skill in self.skills:
            if skill.requires_mode and skill.requires_mode != ctx.mode:
                continue

            similarity = _cosine_similarity(context_embedding, skill.intent_embedding)
            if similarity >= threshold:
                matched.append((skill, similarity))

        # ATL concept boost — reinforce matches with learned knowledge
        if self._memory_hub and ctx.knowledge_context:
            for i, (skill, score) in enumerate(matched):
                for concept in ctx.knowledge_context:
                    if self._concept_relates_to_skill(concept, skill):
                        matched[i] = (skill, score * 1.2)
                        break

        # Return top-3 by similarity
        matched.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in matched[:3]]

    def _extract_searchable_text(self, ctx: "StructuredContext") -> str:
        """Build a text representation of the context for embedding."""
        parts: list[str] = []
        if ctx.current_percept:
            if ctx.current_percept.content:
                parts.append(ctx.current_percept.content)
            if ctx.current_percept.transcript_chunk:
                parts.append(ctx.current_percept.transcript_chunk)
            if ctx.current_percept.cli_input:
                parts.append(ctx.current_percept.cli_input)
        if ctx.active_goal:
            parts.append(ctx.active_goal)
        return " ".join(parts)

    def _concept_relates_to_skill(
        self, concept: dict[str, Any], skill: SkillPrompt
    ) -> bool:
        """Check if an ATL concept relates to a skill's intent."""
        concept_name = concept.get("concept_name", "").lower()
        intent_words = set(skill.intent.lower().split())
        concept_words = set(concept_name.split())
        return bool(intent_words & concept_words)

    @staticmethod
    def _parse_header(path: Path) -> tuple[dict[str, str], str]:
        """Parse ``# key: value`` header lines and body from a skill file."""
        text = path.read_text()
        lines = text.split("\n")
        meta: dict[str, str] = {}
        body_start = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("# ") and ":" in stripped:
                key, _, value = stripped[2:].partition(":")
                meta[key.strip().lower()] = value.strip()
                body_start = i + 1
            elif stripped == "" and body_start == i:
                body_start = i + 1
            else:
                break

        body = "\n".join(lines[body_start:]).strip()
        return meta, body
