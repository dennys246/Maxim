"""BioTelemetryCollector — captures bio-system events for research reports.

Subscribes to the sim_logger's in-memory record stream and filters for
bio-system events.  At session end, persists to ``bio_telemetry.jsonl``
in the session directory.

The collector is intentionally lightweight: it hooks into the existing
``sim_log`` JSONL record stream (``_log_records``) rather than adding
new instrumentation points.  This means ANY event emitted via ``sim_log``
with a bio-system subsystem is automatically captured.

Usage::

    collector = BioTelemetryCollector()
    # ... run simulation ...
    path = collector.save(session_dir)
    # Returns Path to bio_telemetry.jsonl

Research agents can then read bio_telemetry.jsonl for detailed bio-system
analysis sections.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Subsystems considered "bio-system telemetry" — events from these
# subsystems are captured for research analysis.
BIO_TELEMETRY_SUBSYSTEMS = frozenset(
    {
        "HIPPOCAMPUS",
        "NAc",
        "ATL",
        "SCN",
        "CEREBELLUM",
        "SENSORY",
        "SENSOR",
        "BODY",
        "BODY_STATE",
        "FEAR",
        "PAIN",
        "REACTION",
        "THOUGHT",
        "DELIBERATION",
        "ENRICHMENT",
        "IMAGINATION",
        "DISCOVERY",
        "GATE",
        "MOTOR",
    }
)


class BioTelemetryCollector:
    """Collects bio-system events from sim_log records for research persistence.

    Thread-safe: reads a snapshot of ``_log_records`` at save time.
    """

    def save(self, session_dir: str | Path) -> Path | None:
        """Filter bio-system events from sim_log records and persist to JSONL.

        Args:
            session_dir: Session directory to write bio_telemetry.jsonl into.

        Returns:
            Path to the saved file, or None if no events were captured.
        """
        try:
            import maxim.simulation.sim_logger as _sim_mod
        except ImportError:
            return None

        # Use module reference (not by-name import) to avoid the mutable-
        # globals rebinding trap: enable_sim_logging() rebinds _log_records
        # to a new list, so a by-name import would capture a stale reference.
        # list() on a CPython list is GIL-atomic with respect to concurrent
        # .append() calls. Under free-threaded Python (3.13t+) this would
        # need a lock — acceptable since bio_telemetry is only called at
        # session end (not on the hot path).
        records = list(_sim_mod._log_records)

        bio_events = [r for r in records if r.get("subsystem") in BIO_TELEMETRY_SUBSYSTEMS]

        if not bio_events:
            return None

        session_path = Path(session_dir)
        session_path.mkdir(parents=True, exist_ok=True)
        telemetry_path = session_path / "bio_telemetry.jsonl"

        try:
            with open(str(telemetry_path), "w", encoding="utf-8") as f:
                for event in bio_events:
                    f.write(json.dumps(event, default=str) + "\n")
            logger.info(
                "Bio telemetry saved: %s (%d events)",
                telemetry_path,
                len(bio_events),
            )
            return telemetry_path
        except Exception as e:
            logger.warning("Failed to save bio telemetry: %s", e)
            return None

    @staticmethod
    def load(telemetry_path: str | Path) -> list[dict[str, Any]]:
        """Load bio telemetry events from a JSONL file.

        Args:
            telemetry_path: Path to bio_telemetry.jsonl.

        Returns:
            List of event dicts.
        """
        events: list[dict[str, Any]] = []
        path = Path(telemetry_path)
        if not path.exists():
            return events
        try:
            with open(str(path), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except Exception as e:
            logger.warning("Failed to load bio telemetry from %s: %s", path, e)
        return events

    @staticmethod
    def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
        """Produce a summary of bio telemetry events for research reports.

        Returns a dict with per-subsystem event counts + key metrics.
        """
        counts: dict[str, int] = {}
        for event in events:
            sub = event.get("subsystem", "UNKNOWN")
            counts[sub] = counts.get(sub, 0) + 1

        # Extract notable events
        nac_links: list[dict[str, Any]] = []
        pain_events: list[dict[str, Any]] = []
        imagination_events: list[dict[str, Any]] = []
        discovery_events: list[dict[str, Any]] = []
        enrichment_events: list[dict[str, Any]] = []

        for event in events:
            sub = event.get("subsystem", "")
            data = event.get("data", {})
            if sub == "NAc" and data.get("link_type"):
                nac_links.append(
                    {
                        "t": event.get("t"),
                        "event": data.get("event"),
                        "outcome": data.get("outcome"),
                        "confidence": data.get("confidence"),
                        "rpe": data.get("rpe"),
                    }
                )
            elif sub == "PAIN":
                pain_events.append({"t": event.get("t"), "message": event.get("message", "")})
            elif sub == "IMAGINATION":
                imagination_events.append(
                    {
                        "t": event.get("t"),
                        "phase": data.get("phase"),
                        "phrase": data.get("phrase"),
                        "result": data.get("result"),
                    }
                )
            elif sub == "DISCOVERY":
                discovery_events.append(
                    {
                        "t": event.get("t"),
                        "query": data.get("query"),
                        "matched": data.get("matched"),
                        "activated": data.get("activated"),
                    }
                )
            elif sub == "ENRICHMENT":
                enrichment_events.append(
                    {
                        "t": event.get("t"),
                        "system": data.get("system"),
                        "summary": data.get("summary"),
                    }
                )

        return {
            "total_events": len(events),
            "per_subsystem": counts,
            "nac_links_formed": len(nac_links),
            "nac_links": nac_links[:10],  # Top 10 for report
            "pain_events": len(pain_events),
            "pain_samples": pain_events[:5],
            "imagination_events": len(imagination_events),
            "imagination_samples": imagination_events[:5],
            "discovery_events": len(discovery_events),
            "discovery_samples": discovery_events[:5],
            "enrichment_events": len(enrichment_events),
            "enrichment_samples": enrichment_events[:5],
        }
