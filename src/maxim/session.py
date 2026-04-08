"""Session — persistent simulation container.

A Session wraps a SimulationResult with session identity, persistence,
and convenience methods for post-hoc analysis.  It is the primary return
type from ``maxim.imagine()`` and can be reloaded from disk via
``maxim.session(id)``.

Design principles:
- Delegates to SimulationResult for all sim data (backward compat)
- Adds session_id, observe(), research() as first-class operations
- Loadable from persisted session directories without re-running sims
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.simulation.orchestrator import SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """A persistent simulation session.

    Wraps a ``SimulationResult`` with session identity and convenience
    methods.  Properties delegate to the underlying result so existing
    code that reads ``result.turns``, ``result.duration_s``, etc. still
    works unchanged via ``session.turns``.

    Example::

        session = maxim.imagine(goal="test memory recall")
        print(session.id)              # "20260408_143022"
        print(session.turns)           # 12

        memories = session.observe("memory")
        report = session.research()

        # Reload later
        session = maxim.get_session("20260408")
    """

    id: str
    dir: str = ""
    goal: str = ""
    persona: str = ""
    model: str = ""
    _result: Any = field(default=None, repr=False)

    # ── Delegated properties from SimulationResult ─────────────────

    @property
    def turns(self) -> int:
        return getattr(self._result, "turns", 0)

    @property
    def total_actions(self) -> int:
        return getattr(self._result, "total_actions", 0)

    @property
    def blocked_actions(self) -> int:
        return getattr(self._result, "blocked_actions", 0)

    @property
    def duration_s(self) -> float:
        return getattr(self._result, "duration_s", 0.0)

    @property
    def finish_reason(self) -> str:
        return getattr(self._result, "finish_reason", "unknown")

    @property
    def summary(self) -> str:
        return getattr(self._result, "summary", "")

    @property
    def actions(self) -> list[dict[str, Any]]:
        return getattr(self._result, "actions", [])

    @property
    def campaign_analysis(self) -> dict[str, Any]:
        return getattr(self._result, "campaign_analysis", {})

    @property
    def result(self) -> Any:
        """Access the underlying SimulationResult directly."""
        return self._result

    # ── Session operations ─────────────────────────────────────────

    def observe(
        self,
        subsystem: str | None = None,
        *,
        keyword: str | None = None,
        limit: int = 10,
    ) -> dict[str, Any]:
        """Query bio-state from this session's persisted data.

        Args:
            subsystem: Which subsystem to query (``"memory"``, ``"causal"``,
                ``"concepts"``, ``"pain"``, ``"temporal"``, ``"energy"``).
                ``None`` returns a summary of all subsystems.
            keyword: Filter results by keyword.
            limit: Max results to return.

        Returns:
            Dict with subsystem-specific data.
        """
        session_path = Path(self.dir)
        if not session_path.is_dir():
            return {"error": f"Session directory not found: {self.dir}"}

        observer = _build_session_observer(session_path)
        if observer is None:
            return {"error": "No persisted state found in session", "session_dir": self.dir}

        dispatch = {
            None: lambda: observer.system_stats(),
            "memory": lambda: observer.memory_recall(keyword=keyword, limit=limit),
            "causal": lambda: observer.causal_links(event_signature=keyword),
            "concepts": lambda: observer.concept_query(name=keyword),
            "pain": lambda: observer.pain_history(limit=limit),
            "temporal": lambda: observer.temporal_patterns(),
            "energy": lambda: observer.energy_status(),
        }

        handler = dispatch.get(subsystem)
        if handler is None:
            return {
                "error": f"Unknown subsystem: {subsystem!r}",
                "available": [k for k in dispatch if k is not None],
            }
        return handler()

    def research(self, **kwargs: Any) -> Any:
        """Generate a research report from this session's accumulated data.

        Returns a ``Report`` object that can be saved to multiple formats
        (Markdown, JSON).  Full research pipeline wiring (Writer + Reviewer
        agents) ships when the research bugs (D-0a..D-0e) are fixed.

        Returns:
            ``Report`` with session data.  Call ``.save("report.md")``
            or ``.save("report.json")`` to persist.

        Example::

            report = session.research()
            report.save("findings.md")
            report.save("findings.json")
        """
        import warnings
        from maxim.report import Report

        logger.info("Research requested for session %s", self.id)

        # Build report from available session data
        report = Report(
            title=f"Research Report — {self.goal}" if self.goal else "Research Report",
            goal=self.goal,
            session_id=self.id,
            metadata={"persona": self.persona, "model": self.model},
        )

        # Populate with bio-state if available
        session_path = Path(self.dir) if self.dir else None
        if session_path and session_path.is_dir():
            observer = _build_session_observer(session_path)
            if observer is not None:
                try:
                    stats = observer.system_stats()
                    report.metrics = stats
                    report.sections["System State"] = json.dumps(stats, indent=2, default=str)
                except Exception:
                    pass
        else:
            warnings.warn(
                f"Session '{self.id}' has no persisted data on disk. "
                f"Report will be incomplete. Run a simulation first with maxim.imagine().",
                stacklevel=2,
            )

        return report

    # ── Persistence ────────────────────────────────────────────────

    def save(self) -> str:
        """Save session metadata to disk.

        Creates or updates the session directory with a ``session.json``
        containing session identity and metadata.  Does NOT re-save
        simulation results or bio-state (those are saved by the
        orchestrator at sim completion).

        Returns:
            Path to the session directory.

        Example::

            session = maxim.imagine(goal="test")
            session.save()  # Persist session metadata
        """
        session_dir = Path(self.dir)
        if not self.dir:
            from maxim.utils.paths import sim_reports
            session_dir = sim_reports() / self.id
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "session_id": self.id,
            "goal": self.goal,
            "persona": self.persona,
            "model": self.model,
            "turns": self.turns,
            "duration_s": self.duration_s,
            "finish_reason": self.finish_reason,
        }
        from maxim.utils.atomic_io import atomic_write_json
        atomic_write_json(str(session_dir / "session.json"), meta)

        self.dir = str(session_dir)
        return self.dir

    @classmethod
    def from_result(cls, result: "SimulationResult", model: str = "") -> "Session":
        """Create a Session from a fresh SimulationResult."""
        return cls(
            id=result.session_id,
            dir=result.session_dir,
            goal=result.goal,
            persona=result.persona,
            model=model,
            _result=result,
        )

    @classmethod
    def from_disk(cls, session_id: str) -> "Session":
        """Load a Session from a persisted session directory.

        Supports fuzzy matching on session_id prefix (e.g. ``"20260408"``
        matches ``"20260408_143022"``).

        Raises:
            FileNotFoundError: If no matching session is found.
        """
        from maxim.utils.paths import sim_reports

        reports_dir = sim_reports()
        match = _fuzzy_find_session(reports_dir, session_id)
        if match is None:
            raise FileNotFoundError(
                f"No session matching '{session_id}' found in {reports_dir}. "
                f"Run maxim.list_sessions() to see available sessions."
            )

        session_dir = reports_dir / match

        # Load report.json for metadata
        report_path = session_dir / "report.json"
        goal = ""
        persona = ""
        model = ""
        if report_path.exists():
            try:
                with open(report_path) as f:
                    data = json.load(f)
                goal = data.get("goal", "")
                persona = data.get("persona", "")
                model = data.get("language_model", "")
            except Exception as exc:
                logger.warning("Could not load report.json for session %s: %s", match, exc)

        return cls(
            id=match,
            dir=str(session_dir),
            goal=goal,
            persona=persona,
            model=model,
            _result=None,  # No live result for disk-loaded sessions
        )

    def __repr__(self) -> str:
        parts = [f"Session(id={self.id!r}"]
        if self.goal:
            goal_short = self.goal[:40] + "..." if len(self.goal) > 40 else self.goal
            parts.append(f"goal={goal_short!r}")
        if self.turns:
            parts.append(f"turns={self.turns}")
        if self.duration_s:
            parts.append(f"duration={self.duration_s:.1f}s")
        return ", ".join(parts) + ")"


# ── Module-level helpers ───────────────────────────────────────────────


def _fuzzy_find_session(reports_dir: Path, prefix: str) -> str | None:
    """Find a session directory by prefix match. Returns the full name or None."""
    if not reports_dir.is_dir():
        return None

    candidates = []
    for p in reports_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            candidates.append(p.name)

    if not candidates:
        return None

    # Return the most recent match (lexicographic sort = chronological for timestamps)
    candidates.sort(reverse=True)
    return candidates[0]


def _build_session_observer(session_dir: Path) -> Any:
    """Build an Observer from a session directory's persisted state."""
    from maxim.simulation.introspection import Observer

    hippocampus = None
    nac = None

    hippo_path = session_dir / "aut_hippocampus.json"
    if hippo_path.exists():
        try:
            from maxim.memory.hippocampus import Hippocampus

            hippocampus = Hippocampus()
            hippocampus.load(str(hippo_path))
        except Exception as exc:
            logger.warning("Could not load hippocampus from %s: %s", hippo_path, exc)
            hippocampus = None

    nac_path = session_dir / "aut_nac.json"
    if nac_path.exists():
        try:
            from maxim.decisions.nac import NAc

            nac = NAc()
            nac.load(str(nac_path))
        except Exception as exc:
            logger.warning("Could not load NAc from %s: %s", nac_path, exc)
            nac = None

    if hippocampus is None and nac is None:
        return None

    return Observer(hippocampus=hippocampus, nac=nac)


def list_sessions(*, limit: int = 20) -> list[Session]:
    """List recent simulation sessions from disk.

    Returns Session objects with metadata loaded from report.json.
    Most recent sessions first.
    """
    from maxim.utils.paths import sim_reports

    reports_dir = sim_reports()
    if not reports_dir.is_dir():
        return []

    # Collect session directories (sorted newest first)
    dirs = sorted(
        [p for p in reports_dir.iterdir() if p.is_dir() and (p / "report.json").exists()],
        key=lambda p: p.name,
        reverse=True,
    )[:limit]

    sessions = []
    for d in dirs:
        try:
            with open(d / "report.json") as f:
                data = json.load(f)
            sessions.append(
                Session(
                    id=d.name,
                    dir=str(d),
                    goal=data.get("goal", ""),
                    persona=data.get("persona", ""),
                    model=data.get("language_model", ""),
                )
            )
        except Exception:
            # Include even if report.json is unreadable
            sessions.append(Session(id=d.name, dir=str(d)))

    return sessions
