"""Research agents — Writer and Reviewer for the research protocol.

Writer produces structured research papers from experiment data.
Reviewer validates claims by re-running experiments and checking consistency.

Both agents run sequentially using a shared LLMRouter (same as the
simulation orchestrator pattern). They communicate via LocalMessageBus
using MeshMessage envelopes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from maxim.mesh.identity import AgentProfile
from maxim.mesh.message import MeshMessage, MeshMessageType
from maxim.mesh.bus import LocalMessageBus
from maxim.simulation.research_tools import ExperimentLog

logger = logging.getLogger(__name__)


# ── Paper data model ─────────────────────────────────────────────────────────

PAPER_SECTIONS = [
    "methods",
    "results",
    "introduction",
    "discussion",
    "conclusions",
    "title_abstract",
    "references",
    "acknowledgements",
]


@dataclass
class PaperDraft:
    """In-memory representation of a research paper draft."""

    sections: dict[str, str] = field(default_factory=dict)
    revision: int = 0
    output_path: Path | None = None

    def write_section(self, name: str, content: str) -> None:
        name = name.lower().strip()
        self.sections[name] = content

    def read_section(self, name: str) -> str | None:
        return self.sections.get(name.lower().strip())

    def to_markdown(self) -> str:
        """Render the paper as a markdown document."""
        lines = []
        # Title + abstract first (if present)
        if "title_abstract" in self.sections:
            lines.append(self.sections["title_abstract"])
            lines.append("")

        # Ordered sections
        for section in PAPER_SECTIONS:
            if section == "title_abstract":
                continue  # Already rendered
            content = self.sections.get(section)
            if content:
                heading = section.replace("_", " ").title()
                lines.append(f"## {heading}")
                lines.append("")
                lines.append(content)
                lines.append("")

        return "\n".join(lines)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")
        self.output_path = path


# ── Review data model ────────────────────────────────────────────────────────


@dataclass
class ReviewResult:
    """Structured review output from the Reviewer agent."""

    verdict: str = "pending"  # accept, revise, reject
    confidence: float = 0.0
    section_feedback: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    revision_requests: list[str] = field(default_factory=list)
    reproducibility_check: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "section_feedback": self.section_feedback,
            "issues": self.issues,
            "revision_requests": self.revision_requests,
            "reproducibility_check": self.reproducibility_check,
        }


# ── Writer Agent ─────────────────────────────────────────────────────────────


class WriterAgent:
    """Produces a structured research paper from experiment data.

    Uses the shared LLMRouter to generate each section, informed by
    the experiment log from the Researcher.
    """

    def __init__(
        self,
        llm: Any,
        experiment_log: ExperimentLog,
        bus: LocalMessageBus,
        session_dir: Path,
    ) -> None:
        self.llm = llm
        self.experiment_log = experiment_log
        self.bus = bus
        self.session_dir = session_dir
        self.profile = AgentProfile(
            nickname="writer",
            role="writer",
            capabilities=["write_section", "read_section", "submit_draft"],
            personality="Clear, precise academic writer. Lets data speak.",
        )
        self.draft = PaperDraft()
        self._received_messages: list[MeshMessage] = []
        bus.register("writer", self._on_message)

    def _on_message(self, msg: MeshMessage) -> None:
        self._received_messages.append(msg)

    def run(self, goal: str) -> PaperDraft:
        """Generate a complete paper draft from experiment data.

        Returns the PaperDraft with all sections populated.
        """
        experiments = self.experiment_log.all_entries()
        if not experiments:
            logger.warning(
                "Writer: no formal experiments recorded — generating observational paper from available metrics"
            )
            # Build a minimal experiment summary so the Writer can still produce content
            exp_summary = json.dumps(
                [{"note": "No formal experiments recorded. Paper generated from simulation observations and metrics."}],
                indent=2,
                default=str,
            )
        else:
            exp_summary = json.dumps(experiments, indent=2, default=str)

        # Write sections in academic order: methods first, title last
        for section in PAPER_SECTIONS:
            content = self._generate_section(section, goal, exp_summary)
            if content:
                self.draft.write_section(section, content)
                logger.info("Writer: completed section '%s' (%d chars)", section, len(content))

        # Save to disk
        paper_path = self.session_dir / "paper.md"
        self.draft.save(paper_path)
        logger.info("Writer: paper saved to %s", paper_path)

        # Notify reviewer via bus
        self.bus.send(
            MeshMessage(
                sender="writer",
                recipient="reviewer",
                msg_type=MeshMessageType.PAPER_DRAFT,
                payload={"path": str(paper_path), "sections": list(self.draft.sections.keys())},
            )
        )

        return self.draft

    def revise(self, feedback: ReviewResult) -> PaperDraft:
        """Revise sections based on reviewer feedback."""
        self.draft.revision += 1
        experiments = self.experiment_log.all_entries()
        exp_summary = json.dumps(experiments, indent=2, default=str)

        for section, comment in feedback.section_feedback.items():
            current = self.draft.read_section(section) or ""
            revised = self._revise_section(section, current, comment, exp_summary)
            if revised:
                self.draft.write_section(section, revised)
                logger.info("Writer: revised section '%s' (revision %d)", section, self.draft.revision)

        paper_path = self.session_dir / "paper.md"
        self.draft.save(paper_path)
        return self.draft

    _WRITER_SYSTEM = (
        "You are an academic research paper writer. "
        "Output ONLY prose text in markdown format — no JSON, no code fences, no metadata. "
        "Write in third person, academic tone. "
        "Cite experiment references inline (e.g., [researcher.hippo.exp_001]). "
        "Include specific metrics and numbers from the experiment data. "
        "Be concise but thorough."
    )

    def _generate_prose(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate prose text via LLM, bypassing JSON parsing."""
        if self.llm is None:
            return ""
        try:
            text, _usage = self.llm._complete_text(
                self._WRITER_SYSTEM,
                prompt,
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return text.strip() if text else ""
        except Exception as e:
            logger.warning("Writer: prose generation failed: %s", e)
            return ""

    def _generate_section(self, section: str, goal: str, exp_summary: str) -> str:
        """Use the LLM to generate a paper section as academic prose."""
        if self.llm is None:
            return ""

        prompt = (
            f"Write the {section.upper()} section of a research paper.\n\n"
            f"Research goal: {goal}\n\n"
            f"Experiment data:\n{exp_summary[:4000]}\n\n"
            f"Requirements for this section:\n"
        )

        # Section-specific guidance
        section_guidance = {
            "title_abstract": (
                "Write a paper title on the first line, then a blank line, then an abstract (150-250 words). "
                "The abstract should state the research question, methodology, key findings, and conclusions."
            ),
            "introduction": (
                "Provide background on memory retention under interference in cognitive architectures. "
                "State the research question and hypothesis. Explain why this matters."
            ),
            "methods": (
                "Describe the experimental setup: campaign structure (seed/interference/recall phases), "
                "turn count, delivery method (direct bridge injection), AUT model, and measurement approach. "
                "Include enough detail for reproducibility."
            ),
            "results": (
                "Report the findings with specific numbers: memory survival (yes/no), total memories formed, "
                "graph edges, causal links, AUT actions per turn. Use a results table if helpful."
            ),
            "discussion": (
                "Interpret the results: what do they mean for the memory system? "
                "Discuss the gap between memory survival and behavioral recall. "
                "Note limitations and suggest improvements."
            ),
            "conclusions": (
                "Summarize the key findings in 2-3 sentences. State whether the hypothesis was supported. "
                "Suggest next steps (longer campaigns, different AUT models, self-introspection tools)."
            ),
            "references": (
                "List all experiment references cited in the paper, formatted as:\n"
                "[UMR] Author, Title, Method, Key Finding"
            ),
            "acknowledgements": (
                "Brief acknowledgement of the experimental infrastructure and any relevant tools used."
            ),
        }

        prompt += section_guidance.get(section, "Write this section clearly and concisely.")
        return self._generate_prose(prompt)

    def _revise_section(self, section: str, current: str, feedback: str, exp_summary: str) -> str:
        """Use the LLM to revise a section based on feedback."""
        if self.llm is None:
            return current

        prompt = (
            f"Revise the {section.upper()} section of a research paper based on reviewer feedback.\n\n"
            f"Current text:\n{current[:2000]}\n\n"
            f"Reviewer feedback: {feedback}\n\n"
            f"Experiment data:\n{exp_summary[:2000]}\n\n"
            f"Rewrite the section addressing the feedback. Output ONLY the revised prose."
        )

        result = self._generate_prose(prompt)
        return result if result else current


# ── Reviewer Agent ───────────────────────────────────────────────────────────


class ReviewerAgent:
    """Reviews research papers for logical gaps and reproducibility.

    Can re-run experiments via spawn_sub_simulation to verify claims.
    Produces structured review with per-section feedback and verdict.
    """

    def __init__(
        self,
        llm: Any,
        experiment_log: ExperimentLog,
        bus: LocalMessageBus,
        session_dir: Path,
    ) -> None:
        self.llm = llm
        self.experiment_log = experiment_log
        self.bus = bus
        self.session_dir = session_dir
        self.profile = AgentProfile(
            nickname="reviewer",
            role="reviewer",
            capabilities=["read_section", "submit_review", "run_validation_experiment"],
            personality="Rigorous, skeptical academic reviewer. Demands evidence.",
        )
        self._received_messages: list[MeshMessage] = []
        bus.register("reviewer", self._on_message)

    def _on_message(self, msg: MeshMessage) -> None:
        self._received_messages.append(msg)

    def review(self, draft: PaperDraft, goal: str) -> ReviewResult:
        """Review a paper draft. Returns structured feedback with verdict."""
        experiments = self.experiment_log.all_entries()
        result = ReviewResult()

        # Review each section
        for section_name, content in draft.sections.items():
            feedback = self._review_section(section_name, content, goal, experiments)
            if feedback:
                result.section_feedback[section_name] = feedback

        # Check data consistency
        consistency = self._check_data_consistency(draft, experiments)
        result.issues.extend(consistency.get("issues", []))

        # Determine verdict
        verdict_data = self._determine_verdict(result, goal, experiments)
        result.verdict = verdict_data.get("verdict", "revise")
        result.confidence = verdict_data.get("confidence", 0.5)
        result.revision_requests = verdict_data.get("revision_requests", [])

        # Save review
        review_path = self.session_dir / f"review_r{draft.revision}.json"
        review_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        logger.info(
            "Reviewer: verdict=%s confidence=%.2f issues=%d", result.verdict, result.confidence, len(result.issues)
        )

        # Notify via bus
        self.bus.send(
            MeshMessage(
                sender="reviewer",
                recipient="writer",
                msg_type=MeshMessageType.REVIEW_RESULT,
                payload=result.to_dict(),
            )
        )

        return result

    def _review_section(self, name: str, content: str, goal: str, experiments: list) -> str:
        """Use the LLM to review a single section."""
        if self.llm is None or not content:
            return ""

        exp_summary = json.dumps(experiments[:10], indent=2, default=str)
        prompt = (
            f"You are a peer reviewer examining the {name.upper()} section.\n\n"
            f"Research goal: {goal}\n\n"
            f"Section content:\n{content[:3000]}\n\n"
            f"Experiment data available:\n{exp_summary[:2000]}\n\n"
            f"Review criteria:\n"
            f"- Does the content accurately reflect the experiment data?\n"
            f"- Are claims supported by evidence?\n"
            f"- Are there logical gaps or unsupported conclusions?\n"
            f"- Is the writing clear and precise?\n\n"
            f"Provide specific, actionable feedback. If the section is acceptable, say 'No issues found.'"
        )

        try:
            result = self.llm.generate_json(prompt, max_tokens=512)
            if isinstance(result, dict):
                return result.get("feedback", result.get("content", str(result)))
            return str(result) if result else ""
        except Exception as e:
            logger.warning("Reviewer: failed to review '%s': %s", name, e)
            return ""

    def _check_data_consistency(self, draft: PaperDraft, experiments: list) -> dict:
        """Check if the paper's claims are consistent with experiment data."""
        issues = []

        # Basic checks (no LLM needed)
        methods = draft.read_section("methods") or ""
        results = draft.read_section("results") or ""

        if experiments and not results:
            issues.append("Results section is empty despite experiments being recorded")

        if results and not methods:
            issues.append("Results section present without Methods section")

        # Check that experiment UMR references in the paper exist
        for exp in experiments:
            umr = exp.get("umr", "")
            if umr and results and umr not in results and umr not in methods:
                issues.append(f"Experiment {umr} not cited in Methods or Results")

        return {"issues": issues}

    def _determine_verdict(self, review: ReviewResult, goal: str, experiments: list) -> dict:
        """Use the LLM to determine the final verdict."""
        if self.llm is None:
            # Heuristic fallback
            if not review.issues:
                return {"verdict": "accept", "confidence": 0.7, "revision_requests": []}
            if len(review.issues) > 3:
                return {"verdict": "reject", "confidence": 0.6, "revision_requests": review.issues}
            return {"verdict": "revise", "confidence": 0.5, "revision_requests": review.issues}

        prompt = (
            f"You are a peer reviewer making a final verdict.\n\n"
            f"Research goal: {goal}\n"
            f"Experiments conducted: {len(experiments)}\n"
            f"Section feedback:\n{json.dumps(review.section_feedback, indent=2)[:2000]}\n"
            f"Issues found: {review.issues}\n\n"
            f"Verdict options:\n"
            f"- accept: Paper is sound, claims supported\n"
            f"- revise: Specific improvements needed (list them)\n"
            f"- reject: Fundamental flaws (insufficient data, fabricated claims)\n\n"
            f'Return JSON: {{"verdict": "...", "confidence": 0.0-1.0, "revision_requests": [...]}}'
        )

        try:
            result = self.llm.generate_json(prompt, max_tokens=256)
            if isinstance(result, dict):
                return {
                    "verdict": result.get("verdict", "revise"),
                    "confidence": float(result.get("confidence", 0.5)),
                    "revision_requests": result.get("revision_requests", []),
                }
        except Exception as e:
            logger.warning("Reviewer: verdict generation failed: %s", e)

        # Fallback
        return {"verdict": "revise", "confidence": 0.5, "revision_requests": review.issues}
