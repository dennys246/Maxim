"""Exp 63's EXECUTED ranker: ``hippocampus_retrieval._rank_by_relevance`` byte for byte as it was at 52653848, the
commit T1-16's evidence ran on (docs/plans/behavioral_graduation_candidates.md, the T1-16 row).

Session R's #1064 fix changed the shipped ranker, which fires T1-16's named re-run trigger. The pin
``tests/unit/test_o19_rerun.py::test_the_frozen_ranker_is_the_shipped_ranker`` is re-pointed here, at the ranker the
evidence was produced by, never deleted (roadmap_1_3_x.md §1.3.4, ledger note). Vendored, not read through ``git show``
at test time, so a shallow checkout still runs the pin; ``test_the_vendored_ranker_is_the_executed_commits`` checks the
bytes against the commit wherever its history is present. Do not edit this function.
"""

from __future__ import annotations

from maxim.memory.types import CompressedMemory, EpisodicMemory

EXECUTED_COMMIT = "52653848"


def _rank_by_relevance(
    memories: list[EpisodicMemory | CompressedMemory],
    query: str,
    limit: int,
) -> list[EpisodicMemory | CompressedMemory]:
    """Rank memories by keyword overlap with a free-text query.

    Scores each memory by how many query tokens appear in its goal,
    tool name, detected objects/people, and observation text.
    Results are sorted by score descending, with recency as tiebreaker.
    """
    query_tokens = set(query.lower().split())
    if not query_tokens:
        memories.sort(key=lambda m: m.timestamp, reverse=True)
        return memories[:limit]

    scored: list[tuple[float, float, EpisodicMemory | CompressedMemory]] = []
    for mem in memories:
        tokens: set[str] = set()

        # Extract searchable text from memory
        if isinstance(mem, CompressedMemory):
            tokens.update((mem.goal or "").lower().split())
            tokens.add((mem.tool_name or "").lower())
        else:
            # Full EpisodicMemory
            if mem.context and mem.context.active_goal:
                tokens.update(mem.context.active_goal.lower().split())
            if mem.action:
                tokens.add((mem.action.tool_name or "").lower())
            if mem.perception:
                for obj in mem.perception.detected_objects or []:
                    tokens.update(obj.lower().split())
                for person in mem.perception.detected_people or []:
                    tokens.update(person.lower().split())
                # Also check observation text
                obs_text = mem.perception.observations.get("text", "")
                if isinstance(obs_text, str):
                    tokens.update(obs_text.lower().split()[:50])  # Cap to avoid huge scans
                # Check decision_rationale
                rationale = getattr(mem.perception, "decision_rationale", "")
                if rationale:
                    tokens.update(rationale.lower().split())

        # Score = fraction of query tokens found
        overlap = query_tokens & tokens
        score = len(overlap) / len(query_tokens) if query_tokens else 0.0
        scored.append((score, mem.timestamp, mem))

    # Sort by score descending, then recency as tiebreaker
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [mem for _, _, mem in scored[:limit]]
