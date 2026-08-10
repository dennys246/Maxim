"""Append-only post-mortem record of every ``build_primary_router`` decision (P9).

Solves the "why did we pick this configuration?" question. Without this
log, debugging "my sim ran qwen2.5-14b instead of mistral-7b" requires
reading the source and guessing at env-var state at the time of the run.
With this log, the user runs::

    maxim doctor --last-decision
    # or
    tail -1 ~/.maxim/util/lane_decisions.jsonl | jq .

and gets the exact tier walk: which env vars were set, which lanes had
remote_url cleared, which probe outcomes the leader produced, which
profiles were auto-downloaded.

Storage shape:
* JSONL at ``$MAXIM_DATA_HOME/util/lane_decisions.jsonl``.
* Append-only (``open(..., "a", encoding="utf-8")``). On POSIX, writes
  smaller than ``PIPE_BUF`` (typically 4096 bytes) are atomic, so two
  parallel ``maxim`` processes can both append without locking. Each
  record fits well under that threshold (~600-800 bytes).
* Rotation: at ~1000 entries the oldest are dropped on the next append.
  We rewrite the file in place — only the appender rotates, so the
  rewrite race is bounded to "two appenders both decide to rotate" and
  the worst case is one record's loss, not corruption.

Redaction policy:
* API keys (``MAXIM_LANE_*_REMOTE_API_KEY``) are NEVER logged.
* Remote URLs are reduced to the hostname via ``urlparse().hostname``
  so paths and query strings stay off disk.
* Anything that looks like an env-var key with ``KEY``, ``TOKEN``, or
  ``SECRET`` in its name is masked.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Hard cap on the number of records kept on disk. Older records are
# dropped on the next append once the file crosses this threshold. The
# cap is intentionally generous — each record is ~700 bytes, so 1000
# records is well under 1 MB and easy to grep.
MAX_RECORDS = 1000

_REDACT_KEYWORDS = ("KEY", "TOKEN", "SECRET", "PASS")


@dataclass
class LaneDecisionRecord:
    """One ``build_primary_router`` invocation, suitable for JSONL serialization.

    Each field is plain JSON-serializable so the record can be diff'd
    in a normal text editor without a custom decoder.
    """

    timestamp: float
    pid: int
    maxim_version: str
    caps: dict[str, Any]
    env: dict[str, str]
    peer_config_loaded: bool
    tier_decisions: dict[str, dict[str, Any]]
    probe_results: dict[str, str] = field(default_factory=dict)
    auto_download_triggered: list[str] = field(default_factory=list)


def _decision_log_path() -> Path:
    """Return ``~/.maxim/util/lane_decisions.jsonl``. Resolved lazily."""
    from maxim.utils.paths import data_home

    return data_home() / "util" / "lane_decisions.jsonl"


def _redact_url(url: str | None) -> str:
    """Reduce a URL to its hostname so paths/query don't hit disk."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        return parsed.hostname or ""
    except Exception:
        return ""


def _capture_env_snapshot() -> dict[str, str]:
    """Snapshot the routing-relevant env vars with secrets redacted.

    Only includes the variables that actually steer routing decisions —
    not the entire process environment. Anything whose key matches a
    redact keyword is replaced with ``"<redacted>"``; URLs are reduced
    to hostnames.
    """
    interesting = (
        "MAXIM_LLM_PROFILE",
        "MAXIM_LLM_N_CTX",
        "MAXIM_AUTO_DOWNLOAD_MODELS",
        "MAXIM_AUTO_SPAWN_LLM_SERVER",
        "MAXIM_SKIP_REMOTE_PROBE",
        "MAXIM_LANE_LARGE_REMOTE_URL",
        "MAXIM_LANE_LARGE_REMOTE_MODEL",
        "MAXIM_LANE_LARGE_REMOTE_API_KEY",
        "MAXIM_LANE_MEDIUM_REMOTE_URL",
        "MAXIM_LANE_MEDIUM_REMOTE_MODEL",
        "MAXIM_LANE_MEDIUM_REMOTE_API_KEY",
        "MAXIM_LANE_SMALL_REMOTE_URL",
        "MAXIM_LANE_SMALL_REMOTE_MODEL",
        "MAXIM_LANE_SMALL_REMOTE_API_KEY",
    )
    snap: dict[str, str] = {}
    for key in interesting:
        raw = os.environ.get(key)
        if raw is None:
            continue
        if any(token in key for token in _REDACT_KEYWORDS):
            snap[key] = "<redacted>"
            continue
        if key.endswith("_REMOTE_URL"):
            snap[key] = _redact_url(raw)
            continue
        snap[key] = raw
    return snap


def _serialize_caps(caps: Any) -> dict[str, Any]:
    """Reduce RuntimeCapabilities (or any dataclass) to a JSON dict."""
    if caps is None:
        return {}
    try:
        if dataclasses.is_dataclass(caps):
            return dataclasses.asdict(caps)
    except Exception:
        pass
    return {
        "has_gpu": getattr(caps, "has_gpu", None),
        "gpu_type": getattr(caps, "gpu_type", None),
        "vram_gb": getattr(caps, "vram_gb", None),
        "ram_gb": getattr(caps, "ram_gb", None),
    }


def _serialize_lane(lane_cfg: Any, source: str) -> dict[str, Any]:
    """Pull the routing-relevant fields off a LaneConfig (with URL redaction)."""
    if lane_cfg is None:
        return {"source": source}
    return {
        "profile": getattr(lane_cfg, "model_profile", None),
        "n_ctx": getattr(lane_cfg, "n_ctx", None),
        # Estimate-vs-served split (n_ctx leg 3): n_ctx above is the local
        # hardware ESTIMATE; this is what the lane's server actually serves
        # when known — the self-auditing surface for budget-vs-server drift.
        "served_n_ctx": getattr(lane_cfg, "served_n_ctx", None),
        "kv_quant_mode": getattr(lane_cfg, "kv_quant_mode", "f16"),
        "device": getattr(lane_cfg, "device", "auto"),
        "remote_host": _redact_url(getattr(lane_cfg, "remote_url", None)),
        "remote_model": getattr(lane_cfg, "remote_model", None),
        "source": source,
    }


def _maxim_version() -> str:
    try:
        from maxim import __version__

        return str(__version__)
    except Exception:
        return "unknown"


def build_record(
    *,
    caps: Any,
    final_lane_configs: dict[str, Any],
    decision_sources: dict[str, str] | None = None,
    peer_config_loaded: bool = False,
    probe_results: dict[str, str] | None = None,
    auto_download_triggered: list[str] | None = None,
) -> LaneDecisionRecord:
    """Assemble a :class:`LaneDecisionRecord` from the build pipeline outputs.

    Pure function — no I/O, no env-var mutation. Tests use this directly
    to assert serialized shape without writing files.
    """
    sources = decision_sources or {}
    tier_decisions = {
        name: _serialize_lane(cfg, sources.get(name, "tier_table")) for name, cfg in final_lane_configs.items()
    }
    return LaneDecisionRecord(
        timestamp=time.time(),
        pid=os.getpid(),
        maxim_version=_maxim_version(),
        caps=_serialize_caps(caps),
        env=_capture_env_snapshot(),
        peer_config_loaded=peer_config_loaded,
        tier_decisions=tier_decisions,
        probe_results=dict(probe_results or {}),
        auto_download_triggered=list(auto_download_triggered or []),
    )


def record_to_json_line(record: LaneDecisionRecord) -> str:
    """Serialize a record to a single-line JSON string (newline included)."""
    payload = dataclasses.asdict(record)
    return json.dumps(payload, separators=(",", ":"), default=str) + "\n"


def append_record(record: LaneDecisionRecord) -> None:
    """Append a record to the on-disk JSONL log, rotating if needed.

    Best-effort: any I/O failure is logged at debug and swallowed. We
    never want telemetry to break the routing path.
    """
    path = _decision_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = record_to_json_line(record)
        # Append. POSIX guarantees small writes (well under PIPE_BUF) are
        # atomic, so two parallel processes won't interleave bytes.
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        logger.debug("decision_log append failed (%s)", e)
        return

    # Rotate if we crossed the cap. Cheap line count — the file is
    # bounded at MAX_RECORDS so re-reading it is at most ~700 KB.
    try:
        _maybe_rotate(path)
    except OSError as e:
        logger.debug("decision_log rotation failed (%s)", e)


def _maybe_rotate(path: Path) -> None:
    """If the log has more than MAX_RECORDS entries, drop the oldest."""
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()
    if len(lines) <= MAX_RECORDS:
        return
    # Keep the most recent MAX_RECORDS lines.
    keep = lines[-MAX_RECORDS:]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        f.writelines(keep)
    tmp.replace(path)


def read_last_record() -> LaneDecisionRecord | None:
    """Read the most recent record from the log, or None if missing/empty.

    Used by ``maxim doctor --last-decision``. Tolerates malformed
    trailing lines (returns None) so a partial write doesn't crash
    the doctor.
    """
    path = _decision_log_path()
    try:
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            return LaneDecisionRecord(**payload)
        except TypeError:
            return None
    return None


def format_record_human(record: LaneDecisionRecord) -> str:
    """Render a record as a multi-line human-readable summary."""
    import datetime as _dt

    when = _dt.datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"Last lane decision — {when} (pid {record.pid}, maxim {record.maxim_version})",
        f"  Capabilities: {record.caps}",
        f"  Peer config loaded: {record.peer_config_loaded}",
    ]
    if record.env:
        lines.append("  Env (redacted):")
        for k, v in sorted(record.env.items()):
            lines.append(f"    {k}={v}")
    lines.append("  Tiers:")
    for lane_name, info in sorted(record.tier_decisions.items()):
        profile = info.get("profile") or "(none)"
        source = info.get("source", "tier_table")
        n_ctx = info.get("n_ctx")
        host = info.get("remote_host") or ""
        suffix = []
        if n_ctx:
            suffix.append(f"n_ctx={n_ctx}")
        if host:
            suffix.append(f"remote={host}")
        suffix.append(f"src={source}")
        lines.append(f"    {lane_name}: {profile} [{', '.join(suffix)}]")
    if record.probe_results:
        lines.append("  Probe outcomes:")
        for host, outcome in sorted(record.probe_results.items()):
            lines.append(f"    {host}: {outcome}")
    if record.auto_download_triggered:
        lines.append(f"  Auto-downloaded: {', '.join(record.auto_download_triggered)}")
    return "\n".join(lines)


__all__ = [
    "MAX_RECORDS",
    "LaneDecisionRecord",
    "append_record",
    "build_record",
    "format_record_human",
    "read_last_record",
    "record_to_json_line",
]
