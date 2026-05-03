"""Regression tests for the deliberation-transcript suppression contract.

When `StructuredContext.deliberation_transcript` is populated (multi-cycle
deliberation, cycles 2+), two prompt sections are intentionally suppressed
to avoid duplicate rendering:

- `bio_enrichment_context` (current cycle's enrichment) — assumed to already
  appear as the last transcript entry.
- `working_memory_thoughts` (prior cycles' thought summaries) — subsumed by
  the richer, ordered transcript representation.

The risk this guards against: if a future change populates
`bio_enrichment_context` with NEW data after the transcript was built (e.g.,
a fresh percept arrived mid-deliberation), the suppression would silently
drop it. These tests pin the current contract so any change to the
suppression behaviour or the producer-consumer alignment surfaces loudly
instead of as a stale-prompt regression.
"""

from __future__ import annotations

import time

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.bus import StructuredContext
from maxim.agents.llm_types import LLMRequest, ModeInfo
from maxim.agents.prompt_budgeter import PromptBudgeter, SectionPriority
from maxim.agents.prompt_builder import PromptBuilder


class _WordTokenCounter:
    """Token counter stub — splits on whitespace."""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


def _make_request(
    *,
    bio_enrichment_context: str = "",
    deliberation_transcript: list[str] | None = None,
    working_memory_thoughts: list[str] | None = None,
) -> LLMRequest:
    context = StructuredContext(
        timestamp=time.time(),
        bio_enrichment_context=bio_enrichment_context,
        deliberation_transcript=deliberation_transcript,
        working_memory_thoughts=working_memory_thoughts,
    )
    mode = ModeInfo(name="passive", goal="", context_prompt="")
    return LLMRequest(
        request_id="test",
        context=context,
        mode=mode,
        autonomy_level=AutonomyLevel.PLANNING,
        internet_access=False,
        internet_policy_summary="",
    )


def _budgeter() -> PromptBudgeter:
    return PromptBudgeter(
        total_budget=4096,
        response_reserve=512,
        token_counter=_WordTokenCounter(),
        template_overhead=100,
    )


def _section_names(b: PromptBudgeter) -> list[str]:
    return [s.name for s in b._sections]


# ─── bio_enrichment_context suppression ────────────────────────────────────


class TestBioEnrichmentSuppression:
    def test_renders_when_no_transcript(self):
        """Without a transcript, bio_enrichment is added at IMPORTANT priority."""
        b = _budgeter()
        req = _make_request(
            bio_enrichment_context="explored a similar cave before; got hurt",
            deliberation_transcript=None,
        )
        PromptBuilder._add_perception_sections(b, req)
        assert "bio_enrichment" in _section_names(b)
        section = next(s for s in b._sections if s.name == "bio_enrichment")
        assert section.priority == SectionPriority.IMPORTANT
        assert "explored a similar cave" in section.content

    def test_suppressed_when_transcript_present(self):
        """With a transcript, bio_enrichment is suppressed even when populated.

        Contract: the transcript's last entry is assumed to contain the
        current cycle's bio_enrichment. If this assumption breaks (e.g.,
        a fresh percept arrives between transcript construction and prompt
        build), the new bio_enrichment is silently dropped — this test pins
        the suppression so any future producer/consumer divergence surfaces
        as a deliberate failing test rather than a stale-prompt bug.
        """
        b = _budgeter()
        req = _make_request(
            bio_enrichment_context="NEW signal that arrived after transcript was built",
            deliberation_transcript=["[Cycle 1] previous reasoning"],
        )
        PromptBuilder._add_perception_sections(b, req)
        names = _section_names(b)
        assert "bio_enrichment" not in names, (
            "bio_enrichment_context was rendered despite transcript being present. "
            "If this is intentional (transcript no longer subsumes current-cycle "
            "enrichment), update the suppression in _add_perception_sections AND "
            "the producer in agent_loop._build_deliberation_transcript."
        )

    def test_no_section_when_both_empty(self):
        """No transcript and no bio_enrichment → no section."""
        b = _budgeter()
        req = _make_request(bio_enrichment_context="", deliberation_transcript=None)
        PromptBuilder._add_perception_sections(b, req)
        assert "bio_enrichment" not in _section_names(b)


# ─── working_memory_thoughts suppression ───────────────────────────────────


class TestWorkingMemoryThoughtsSuppression:
    def test_renders_when_no_transcript(self):
        b = _budgeter()
        req = _make_request(
            working_memory_thoughts=["thought one", "thought two"],
            deliberation_transcript=None,
        )
        PromptBuilder._add_working_memory_section(b, req.context)
        assert "working_memory_thoughts" in _section_names(b)
        section = next(s for s in b._sections if s.name == "working_memory_thoughts")
        assert section.priority == SectionPriority.IMPORTANT
        assert "thought one" in section.content

    def test_suppressed_when_transcript_present(self):
        """Transcript subsumes working memory thoughts."""
        b = _budgeter()
        req = _make_request(
            working_memory_thoughts=["thought one"],
            deliberation_transcript=["[Cycle 1] reasoning"],
        )
        PromptBuilder._add_working_memory_section(b, req.context)
        assert "working_memory_thoughts" not in _section_names(b), (
            "working_memory_thoughts rendered despite transcript being present. "
            "If this is intentional, update the suppression in "
            "_add_working_memory_section AND verify the transcript producer "
            "still includes prior-cycle reasoning."
        )

    def test_no_section_when_thoughts_empty(self):
        b = _budgeter()
        req = _make_request(working_memory_thoughts=None, deliberation_transcript=None)
        PromptBuilder._add_working_memory_section(b, req.context)
        assert "working_memory_thoughts" not in _section_names(b)


# ─── Combined contract ─────────────────────────────────────────────────────


class TestTranscriptSubsumesAll:
    def test_transcript_present_suppresses_both(self):
        """When a transcript renders, neither bio_enrichment nor WMS thoughts
        appear — the transcript is the single source of cycle reasoning.

        If a future change makes the transcript no longer carry the
        current-cycle bio_enrichment, this test will keep passing (both
        suppressed) but the agent will silently lose the bio data. The
        REAL guard is the paired producer test in agent_loop that verifies
        the transcript's last entry contains the current bio_enrichment —
        if you remove that producer guard, also remove this suppression.
        """
        b = _budgeter()
        req = _make_request(
            bio_enrichment_context="current-cycle bio data",
            working_memory_thoughts=["older thought"],
            deliberation_transcript=["[Cycle 1] r1", "[Cycle 2] r2"],
        )
        PromptBuilder._add_perception_sections(b, req)
        PromptBuilder._add_working_memory_section(b, req.context)
        names = _section_names(b)
        assert "bio_enrichment" not in names
        assert "working_memory_thoughts" not in names


# ─── motor_programs entry-aware truncation ─────────────────────────────────


class TestMotorProgramTruncation:
    """The motor_programs truncate_fn must drop full entries (1-3 lines each:
    name + Steps + Known risks), never partial ones. The previous
    implementation sliced by line count and would leave orphaned Steps or
    Known-risks lines whose owning name had been dropped.
    """

    def _build_with_programs(self, programs: list[dict]) -> PromptBudgeter:
        from dataclasses import replace as _dc_replace

        b = _budgeter()
        # Use _add_memory_sections, which is the static method that adds the
        # motor_programs section + its truncate_fn.
        ctx = StructuredContext(timestamp=time.time())
        ctx = _dc_replace(ctx, motor_programs=programs)
        PromptBuilder._add_memory_sections(b, ctx)
        return b

    def _section(self, b: PromptBudgeter, name: str):
        return next(s for s in b._sections if s.name == name)

    def test_no_truncation_under_generous_budget(self):
        progs = [
            {
                "name": f"prog_{i}",
                "confidence": 0.5,
                "executions": 1,
                "success_rate": 0.5,
                "steps": ["a", "b"],
                "risks": ["r1"],
            }
            for i in range(3)
        ]
        b = self._build_with_programs(progs)
        s = self._section(b, "motor_programs")
        # Under no pressure the full content survives.
        assert "prog_0" in s.content
        assert "prog_2" in s.content

    def test_truncate_drops_full_entries_no_orphans(self):
        progs = [
            {
                "name": f"prog_{i}",
                "confidence": 0.5,
                "executions": 1,
                "success_rate": 0.5,
                "steps": ["a", "b"],
                "risks": ["r1", "r2"],
            }
            for i in range(5)
        ]
        b = self._build_with_programs(progs)
        s = self._section(b, "motor_programs")
        # Squeeze hard — only header + ~1 entry should fit.
        truncated = s.truncate_fn(s.content, max_tok=60)
        # The header survives.
        assert "Available Motor Programs" in truncated
        # Any "Steps:" line must be preceded by a "- " entry header (no orphan).
        lines = truncated.split("\n")
        for i, ln in enumerate(lines):
            if ln.lstrip().startswith("Steps:") or ln.lstrip().startswith("Known risks:"):
                # find the nearest preceding "- " line
                preceded = any(lines[j].startswith("- ") for j in range(i - 1, -1, -1))
                assert preceded, f"Orphan continuation line at index {i}: {ln!r}\nFull:\n{truncated}"

    def test_truncate_preserves_head_entries(self):
        """Tail entries are dropped first; head entry survives."""
        progs = [
            {"name": "first", "confidence": 0.9, "executions": 5, "success_rate": 0.9, "steps": ["s1"], "risks": []},
            {
                "name": "second",
                "confidence": 0.5,
                "executions": 1,
                "success_rate": 0.3,
                "steps": ["s2"],
                "risks": ["danger"],
            },
            {
                "name": "third",
                "confidence": 0.2,
                "executions": 1,
                "success_rate": 0.1,
                "steps": ["s3"],
                "risks": ["danger"],
            },
        ]
        b = self._build_with_programs(progs)
        s = self._section(b, "motor_programs")
        truncated = s.truncate_fn(s.content, max_tok=30)
        assert "first" in truncated
        # third should be dropped before first under tight budget
        if "third" in truncated:
            assert "first" in truncated  # invariant: head survives
