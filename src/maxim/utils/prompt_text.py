"""Stored text as a prompt may show it: the bound, the provenance and the untrusted-content frame (#1137, #1444, #1519).

Moved verbatim from ``memory/text.py`` (B2 edge, 2026-11-24): these helpers frame and bound text for a PROMPT and do
no memory work, and their home inside the ``maxim.memory`` package made every importer (``agents/prompt_builder.py``,
so the loop's intake leaf through ``agents/llm_worker.py``) load the whole memory package and numpy. Here they import
only the standard library and ``utils/content_safety``, so the bio-free import closure B2 needs holds
(``tests/unit/test_loop_intake_bio_free.py``). ``memory/text.py`` keeps the content keying (``normalize_tokens`` and
the keys built on it).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

# How much of what an episode heard is kept and rendered (#1444, #1137): ONE bound, used by enrichment's render and by
# compression's ``CompressedMemory.heard``, so a compressed record renders as its episode did. A reviewed constant: it
# holds a stated fact or a question; three recalled episodes stay well under the enrichment section's budget.
RECALLED_WORDS_MAX_CHARS = 240


def bounded_words(text: str, limit: int = RECALLED_WORDS_MAX_CHARS) -> str:
    """``text`` NFKC-normalized and whitespace-collapsed, THEN cut to ``limit`` (normalization first, so it cannot
    grow past the bound). A cut ends in "...", NFKC's own fixed point (a "…" expanded to "..." on the next pass and
    pushed a re-bound over the limit), so bounding is idempotent and a compressed record renders as its episode did."""
    import unicodedata

    text = " ".join(unicodedata.normalize("NFKC", text or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# ── Stored text shown to a model, by provenance (#1519) ─────────────────────────────────────────────────────────────

# The sources whose words are the OPERATOR's own: trusted, shown defanged, never framed (owner decision 2026-11-21,
# corrected the same day): ``cli`` only, typed input (a sim's narrator also arrives as ``cli`` and stays operator, by
# D5). NOT ``voice``: ``embodied_runtime/input_handlers.py::_handle_transcript_record`` forwards any room-mic Whisper
# line naming the wake word as ``pending_voice_input``, which ``loop_perception`` publishes as ``source="voice"``, so
# anyone in the room or on a speakerphone can produce it. NOT ``transcript`` (room-mic audio) for the same reason. Not
# ``speech``: no producer publishes it as a percept source (``speech`` is a comms channel, so ``comms:speech``), and an
# unknown source fails closed. This is a TRUST list, kept apart from ``runtime/loop_perception.py::ADDRESSED_SOURCES``
# (D5: what is always enriched), so widening one never widens the other. No ``comms:*`` source is ever in it (SMS,
# voice calls and the command API are outside parties: #1481, owner decision 2026-10-28).
OPERATOR_SOURCES: frozenset[str] = frozenset({"cli"})


class Provenance(str, Enum):
    """Where a stored text came from, decided ONCE where it is stored and carried with it (#1519), so a renderer
    never has to remember to ask. ``OPERATOR``: the operator's own words, shown defanged. ``EXTERNAL``: from outside
    the agent (inbound comms, the world, recalled records), shown only framed. ``INTERNAL``: the agent's own
    composed text (its enrichment, statuses, tool names), whose embedded frames are kept whole."""

    OPERATOR = "operator"
    EXTERNAL = "external"
    INTERNAL = "internal"


def provenance_of(source: Any) -> Provenance:
    """A percept source's provenance: an ``OPERATOR_SOURCES`` source is the operator, anything else (a ``comms:*``
    source, the world, a missing source: fail-closed) is external."""
    return Provenance.OPERATOR if isinstance(source, str) and source in OPERATOR_SOURCES else Provenance.EXTERNAL


def frame_external(label: str, text: str) -> str:
    """THE frame for stored text from outside the agent (#1444, #1519): ``frame_tool_output(..., external=True)``.
    Every renderer of recalled or stored outside text calls this one helper."""
    from maxim.utils.content_safety import frame_tool_output

    return frame_tool_output(label, text, external=True)


def _bounded_lines(text: str, limit: int) -> str:
    """``bounded_words`` that keeps line breaks (a multi-line message stays multi-line): NFKC first, then the cut."""
    import unicodedata

    text = "\n".join(" ".join(line.split()) for line in unicodedata.normalize("NFKC", text or "").strip().splitlines())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


# The labels enrichment writes before a frame (``(goal: ``, ``earlier input: ``, ``→ ``, ``Related concepts: ``), as a
# tail cut that drops the frame leaves them dangling (#1519).
_DANGLING_LABEL = re.compile(r"\s*(?:\(goal:|earlier input:|→|Related concepts:)\s*$")


def whole_frames(text: str) -> str:
    """``text`` cut somewhere, with any partial untrusted-content frame dropped (Session R, rule 2) and the label a
    dropped frame leaves at the end dropped with it (#1519). Text holding no frame marker is returned as it is."""
    if "TOOL_OUTPUT" not in text:
        return text
    from maxim.utils.content_safety import drop_unbalanced_frames

    kept, dropped = drop_unbalanced_frames(text)
    return _DANGLING_LABEL.sub("", kept) if dropped else kept


def shown_text(text: str, provenance: Provenance, *, label: str, limit: int, keep_lines: bool = False) -> str:
    """``text`` as a prompt may show it, by its provenance, bounded after normalization; "" for empty text.
    ``OPERATOR``: defanged in place (a pasted marker cannot open or close a frame). ``EXTERNAL``: framed
    (``frame_external``). ``INTERNAL``: cut, and any frame the cut broke dropped, so no frame body shows bare."""
    from maxim.utils.content_safety import defang_frame_markers

    if provenance is Provenance.INTERNAL:
        return whole_frames(str(text or "")[:limit])
    body = _bounded_lines(str(text or ""), limit) if keep_lines else bounded_words(str(text or ""), limit)
    if not body:
        return ""
    if provenance is Provenance.OPERATOR:
        return defang_frame_markers(body)[0]
    return frame_external(label, body)


def stored_percept_text(source: Any, text: str, *, limit: int = 100, keep_lines: bool = False) -> str:
    """A percept's text as a prompt may show it, by its SOURCE's provenance (``provenance_of``, ``shown_text``)."""
    label = f"percept_{source}" if isinstance(source, str) and source else "percept"
    return shown_text(text, provenance_of(source), label=label, limit=limit, keep_lines=keep_lines)


__all__ = [
    "bounded_words",
    "RECALLED_WORDS_MAX_CHARS",
    "OPERATOR_SOURCES",
    "Provenance",
    "provenance_of",
    "frame_external",
    "whole_frames",
    "shown_text",
    "stored_percept_text",
]
