"""ProvenanceStore — session-aware, crash-safe provenance persistence.

Directory structure:
    data/provenance/
    +-- sessions.json         # Manifest of all sessions
    +-- {session_id}.jsonl    # Per-session traces + activities

Each JSONL line is either:
- {"type": "trace", ...}     -- completed cycle trace
- {"type": "activity", ...}  -- background activity entry
- {"type": "summary", ...}   -- session summary (written on shutdown)

Uses atomic writes for the sessions manifest (tmp + os.replace).
JSONL appends use flush for crash safety.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from maxim.provenance.types import ProvenanceEntry, ProvenanceTrace

logger = logging.getLogger(__name__)


class ProvenanceStore:
    """Session-aware, crash-safe provenance persistence."""

    def __init__(self, base_dir: str = "data/provenance") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self._base_dir / "sessions.json"
        self._current_file: Any = None  # TextIOWrapper when open
        self._current_session_id: str | None = None

    def _ensure_session_file(self, session_id: str) -> None:
        """Open or reuse the JSONL file for this session."""
        if self._current_session_id == session_id and self._current_file:
            return
        if self._current_file:
            try:
                self._current_file.close()
            except Exception:
                pass
        path = self._base_dir / f"{session_id}.jsonl"
        self._current_file = open(path, "a", encoding="utf-8")
        self._current_session_id = session_id

    def write_trace(self, trace: ProvenanceTrace) -> None:
        """Persist a completed cycle trace."""
        try:
            self._ensure_session_file(trace.session_id)
            line = json.dumps(trace.to_dict(), default=str) + "\n"
            self._current_file.write(line)
            self._current_file.flush()
        except Exception as e:
            logger.warning("Failed to persist trace: %s", e)

    def write_activity(
        self, entry: ProvenanceEntry, session_id: str
    ) -> None:
        """Persist a background activity entry."""
        try:
            self._ensure_session_file(session_id)
            data = entry.to_dict()
            data["type"] = "activity"
            data["session_id"] = session_id
            line = json.dumps(data, default=str) + "\n"
            self._current_file.write(line)
            self._current_file.flush()
        except Exception as e:
            logger.warning("Failed to persist activity: %s", e)

    def write_session_summary(
        self, session_id: str, stats: dict[str, Any]
    ) -> None:
        """Write session summary and update manifest."""
        try:
            self._ensure_session_file(session_id)
            summary = {
                "type": "summary",
                "session_id": session_id,
                "ended_at": time.time(),
                **stats,
            }
            self._current_file.write(
                json.dumps(summary, default=str) + "\n"
            )
            self._current_file.flush()
        except Exception as e:
            logger.warning("Failed to write session summary: %s", e)

        # Update manifest (atomic write)
        self._update_manifest(session_id, stats)

    def close(self) -> None:
        """Close the current session file. Safe to call multiple times."""
        if self._current_file:
            try:
                self._current_file.close()
            except Exception:
                pass
            self._current_file = None
            self._current_session_id = None

    def _update_manifest(
        self, session_id: str, stats: dict[str, Any]
    ) -> None:
        """Atomically update sessions.json manifest."""
        try:
            manifest = self._load_manifest()
            manifest[session_id] = {
                "ended_at": time.time(),
                "file": f"{session_id}.jsonl",
                **stats,
            }
            tmp = self._manifest_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)
            os.replace(tmp, self._manifest_path)
        except Exception as e:
            logger.warning("Failed to update manifest: %s", e)

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {}

    # ---- Cross-run queries ----

    def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Load all records (traces + activities) from a session."""
        path = self._base_dir / f"{session_id}.jsonl"
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def load_recent_sessions(self, limit: int = 10) -> list[str]:
        """Return session IDs ordered by most recent."""
        manifest = self._load_manifest()
        sessions = sorted(
            manifest.items(),
            key=lambda kv: kv[1].get("ended_at", 0),
            reverse=True,
        )
        return [sid for sid, _ in sessions[:limit]]

    def query_concept(
        self, concept_name: str, max_sessions: int = 20
    ) -> list[dict[str, Any]]:
        """Find all provenance records mentioning a concept across sessions."""
        results: list[dict[str, Any]] = []
        for session_id in self.load_recent_sessions(max_sessions):
            records = self.load_session(session_id)
            for record in records:
                if self._record_mentions_concept(record, concept_name):
                    record["session_id"] = session_id
                    results.append(record)
        return results

    def query(
        self,
        concept_name: str | None = None,
        tool_name: str | None = None,
        success: bool | None = None,
        max_sessions: int = 20,
    ) -> list[dict[str, Any]]:
        """General query across sessions."""
        results: list[dict[str, Any]] = []
        for session_id in self.load_recent_sessions(max_sessions):
            records = self.load_session(session_id)
            for record in records:
                if record.get("type") == "summary":
                    continue
                if self._matches(record, concept_name, tool_name, success):
                    record["session_id"] = session_id
                    results.append(record)
        return results

    @staticmethod
    def _record_mentions_concept(record: dict[str, Any], concept_name: str) -> bool:
        """Check if a record mentions a concept by name."""
        cn = concept_name.lower()
        if cn in record.get("action", "").lower():
            return True
        for source in record.get("sources", []):
            if cn in source.get("label", "").lower():
                return True
        for entry in record.get("entries", []):
            if cn in entry.get("action", "").lower():
                return True
            for source in entry.get("sources", []):
                if cn in source.get("label", "").lower():
                    return True
        return False

    @staticmethod
    def _matches(
        record: dict[str, Any],
        concept_name: str | None,
        tool_name: str | None,
        success: bool | None,
    ) -> bool:
        entries = record.get("entries", [])
        if record.get("type") == "activity":
            entries = [record]
        for entry in entries:
            if concept_name:
                if any(
                    concept_name.lower() in s.get("label", "").lower()
                    for s in entry.get("sources", [])
                ):
                    return True
            if tool_name and tool_name in entry.get("action", ""):
                return True
            if success is not None and f"Success={success}" in entry.get("action", ""):
                return True
        return False
