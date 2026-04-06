"""Research tools — experiment tracking for the researcher agent.

These tools extend the orchestrator's toolkit for structured research:
- record_experiment: Log a structured experiment entry
- query_experiments: Search past experiments

Experiment data is persisted to the session directory as experiments.jsonl.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

from maxim.mesh.naming import UMR
from maxim.tools.base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class ExperimentLog:
    """Thread-safe experiment log with JSONL persistence.

    Stores structured experiment entries and persists them incrementally
    to a JSONL file in the session directory.
    """

    def __init__(self, session_dir: str | Path, agent_nickname: str = "researcher") -> None:
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._nickname = agent_nickname
        self._counter = 0
        self._path = Path(session_dir) / "experiments.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Load existing entries if resuming
        if self._path.exists():
            self._load_existing()

    def _load_existing(self) -> None:
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry = json.loads(line)
                        self._entries.append(entry)
                        self._counter = max(self._counter, entry.get("index", 0))
            logger.info("Loaded %d existing experiments from %s", len(self._entries), self._path)
        except Exception as e:
            logger.warning("Failed to load experiments: %s", e)

    def record(
        self,
        hypothesis: str,
        method: str,
        result: str,
        conclusion: str,
        tags: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a structured experiment entry. Returns the entry with UMR."""
        with self._lock:
            self._counter += 1
            umr = UMR.build(self._nickname, "hippo", f"exp_{self._counter:03d}")
            entry = {
                "umr": str(umr),
                "index": self._counter,
                "timestamp": time.time(),
                "hypothesis": hypothesis,
                "method": method,
                "result": result,
                "conclusion": conclusion,
                "tags": tags or [],
                "metrics": metrics or {},
            }
            self._entries.append(entry)

        # Persist incrementally (append, not rewrite)
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning("Failed to persist experiment: %s", e)

        return entry

    def query(
        self,
        keyword: str = "",
        tag: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search experiments by keyword (in hypothesis/result/conclusion) or tag."""
        with self._lock:
            results = list(self._entries)

        if keyword:
            kw = keyword.lower()
            results = [
                e
                for e in results
                if kw in e.get("hypothesis", "").lower()
                or kw in e.get("result", "").lower()
                or kw in e.get("conclusion", "").lower()
                or kw in e.get("method", "").lower()
            ]

        if tag:
            results = [e for e in results if tag in e.get("tags", [])]

        return results[-limit:]

    def all_entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class RecordExperimentTool(Tool):
    """Log a structured experiment entry with hypothesis, method, result, conclusion."""

    name = "record_experiment"
    description = (
        "Record a structured experiment. Provide hypothesis (what you predicted), "
        "method (what you did — which tools, prompts, scenarios), result (what "
        "happened — include data), and conclusion (what it means). Optionally "
        "add tags and metrics. Returns the experiment entry with a UMR reference."
    )
    input_schema = {
        "hypothesis": str,
        "method": str,
        "result": str,
        "conclusion": str,
        "tags": (list, []),
        "metrics": (dict, {}),
    }

    def __init__(self, experiment_log: ExperimentLog) -> None:
        super().__init__()
        self._log = experiment_log

    def execute(self, **kwargs: Any) -> ToolOutput:
        hypothesis = kwargs.get("hypothesis", "")
        method = kwargs.get("method", "")
        result = kwargs.get("result", "")
        conclusion = kwargs.get("conclusion", "")

        if not hypothesis or not result:
            return ToolOutput(
                success=False,
                error="hypothesis and result are required",
            )

        tags = kwargs.get("tags") or []
        metrics = kwargs.get("metrics") or {}

        entry = self._log.record(
            hypothesis=hypothesis,
            method=method,
            result=result,
            conclusion=conclusion,
            tags=tags,
            metrics=metrics,
        )

        return ToolOutput(
            success=True,
            output={
                "recorded": True,
                "umr": entry["umr"],
                "index": entry["index"],
                "experiment": entry,
            },
        )


class QueryExperimentsTool(Tool):
    """Search past experiments by keyword or tag."""

    name = "query_experiments"
    description = (
        "Search the experiment log. Use keyword to search hypothesis, method, "
        "result, and conclusion fields. Use tag to filter by experiment tags. "
        "Returns matching experiments with their UMR references."
    )
    input_schema = {
        "keyword": (str, ""),
        "tag": (str, ""),
        "limit": (int, 20),
    }

    def __init__(self, experiment_log: ExperimentLog) -> None:
        super().__init__()
        self._log = experiment_log

    def execute(self, **kwargs: Any) -> ToolOutput:
        keyword = kwargs.get("keyword", "")
        tag = kwargs.get("tag", "")
        limit = int(kwargs.get("limit", 20))

        results = self._log.query(keyword=keyword, tag=tag, limit=limit)

        return ToolOutput(
            success=True,
            output={
                "total_experiments": len(self._log),
                "returned": len(results),
                "keyword": keyword,
                "tag": tag,
                "experiments": results,
            },
        )
