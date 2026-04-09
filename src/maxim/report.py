"""Report — saveable research/analysis output.

A Report wraps the output of ``session.research()`` or any analysis
pipeline.  It can be saved to multiple formats.

Supported formats (v0.2.0):
- ``.md`` — Markdown (human-readable)
- ``.json`` — Structured JSON (machine-readable)

Future formats (planned):
- ``.pdf`` — Requires ``pymaxim[docs]`` extra (reportlab)
- ``.docx`` — Requires ``pymaxim[docs]`` extra (python-docx)
- ``.html`` — Standalone HTML with embedded styles

Example::

    report = session.research()
    report.save("findings.md")
    report.save("findings.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Formats that need optional dependencies (not yet shipped)
_OPTIONAL_FORMATS = {".pdf", ".docx", ".html"}


@dataclass
class Report:
    """A saveable research or analysis report.

    Attributes:
        title: Report title.
        goal: The research goal or question.
        session_id: Associated simulation session.
        content: The main report body (Markdown text).
        sections: Named sections for structured access.
        metrics: Quantitative results.
        review: Reviewer feedback (if applicable).
        review_verdict: ``"accept"``, ``"revise"``, or ``"reject"``.
        metadata: Arbitrary extra data.
    """

    title: str = ""
    goal: str = ""
    session_id: str = ""
    content: str = ""
    sections: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    review: str = ""
    review_verdict: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def save(self, path: str) -> str:
        """Save report to a file.  Format is inferred from extension.

        Supported: ``.md``, ``.json``.
        Future: ``.pdf``, ``.docx``, ``.html`` (will require extras).

        Args:
            path: Output file path.

        Returns:
            The resolved output path.

        Raises:
            ValueError: If the format is not supported.

        Example::

            report.save("findings.md")    # Markdown
            report.save("findings.json")  # Structured JSON
        """
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix == ".md":
            return self._save_markdown(p)
        elif suffix == ".json":
            return self._save_json(p)
        elif suffix in _OPTIONAL_FORMATS:
            raise ValueError(f"Format '{suffix}' requires pymaxim[docs] (not yet available). Use .md or .json for now.")
        else:
            raise ValueError(f"Unsupported format '{suffix}'. Supported: .md, .json. Future: .pdf, .docx, .html")

    def _save_markdown(self, path: Path) -> str:
        """Save as Markdown."""
        lines = []

        if self.title:
            lines.append(f"# {self.title}")
            lines.append("")

        if self.goal:
            lines.append(f"**Goal:** {self.goal}")
            lines.append("")

        if self.session_id:
            lines.append(f"**Session:** {self.session_id}")
            lines.append("")

        if self.content:
            lines.append(self.content)
            lines.append("")

        if self.sections:
            for section_name, section_content in self.sections.items():
                lines.append(f"## {section_name}")
                lines.append("")
                lines.append(section_content)
                lines.append("")

        if self.metrics:
            lines.append("## Metrics")
            lines.append("")
            for k, v in self.metrics.items():
                lines.append(f"- **{k}:** {v}")
            lines.append("")

        if self.review:
            lines.append("## Review")
            lines.append("")
            if self.review_verdict:
                lines.append(f"**Verdict:** {self.review_verdict}")
                lines.append("")
            lines.append(self.review)
            lines.append("")

        text = "\n".join(lines)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp file + os.replace to avoid partial writes
        import os
        import tempfile

        content = text.encode("utf-8")
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            os.write(fd, content)
            os.fsync(fd)
            os.close(fd)
            os.replace(tmp, str(path))
        except BaseException:
            os.close(fd)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        logger.info("Report saved: %s (%d chars)", path, len(text))
        return str(path)

    def _save_json(self, path: Path) -> str:
        """Save as structured JSON."""
        from maxim.utils.atomic_io import atomic_write_json

        data = {
            "title": self.title,
            "goal": self.goal,
            "session_id": self.session_id,
            "content": self.content,
            "sections": self.sections,
            "metrics": self.metrics,
            "review": self.review,
            "review_verdict": self.review_verdict,
            "metadata": self.metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(path), data)
        logger.info("Report saved: %s", path)
        return str(path)

    @classmethod
    def from_json(cls, path: str) -> "Report":
        """Load a Report from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        parts = [f"Report(title={self.title!r}"]
        if self.session_id:
            parts.append(f"session={self.session_id!r}")
        if self.review_verdict:
            parts.append(f"verdict={self.review_verdict!r}")
        content_len = len(self.content)
        if content_len:
            parts.append(f"content={content_len} chars")
        return ", ".join(parts) + ")"
