"""Perceived (anticipatory) pain — predictive felt signal before action.

Biological analog: ACC/vmPFC anticipatory aversion. Before the AUT acts,
assess the likelihood the action will cause pain, and fire a
``PainType.ANTICIPATED`` signal whose intensity reflects that prediction.

Two contributors drive the intensity:

1. **Learned prediction** (NAc): if NAc has causal links showing
   ``tool:X → NEGATIVE`` outcomes, use that confidence × a scaling
   factor. This is experience-based aversion.

2. **Innate prior** (sensitive-file lookup): hard-coded knowledge
   that touching ``/etc/shadow`` is painful even with no prior
   observations. This is instinct.

Intensity = max(learned, prior). The ``contributors`` dict records
both so post-run analysis can see which dominated.

This module deliberately does NOT execute the action. It only
**predicts** — a gut feeling that flows through the PainBus into
the AUT's hippocampus and context window, letting the AUT reason
about a felt signal rather than relying purely on LLM prompt logic.

The loop closes via ``PainInterceptorExecutor``: actual tool
executions fire ground-truth pain, NAc learns from that, and
future predictions by this assessor become more accurate.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from maxim.proprioception.pain import PainSignal, PainType

logger = logging.getLogger(__name__)


# Pain-trigger threshold: below this intensity, don't bother firing.
DEFAULT_ANTICIPATION_THRESHOLD = 0.1

# Scale factor for NAc learned confidence → pain intensity.
DEFAULT_LEARNED_SCALE = 0.8


@dataclass(frozen=True)
class SensitivePathPrior:
    """Innate prior: path pattern → expected pain intensity on access."""

    path: str              # Absolute path or directory prefix (e.g. "/etc/")
    intensity: float       # 0.0-1.0 expected pain
    operations: frozenset[str] = field(
        default_factory=lambda: frozenset({"read", "write", "delete", "execute"})
    )
    reason: str = ""       # Human-readable explanation


# Default innate priors — these are the "instincts" the AUT ships with.
# They can be overridden per-simulation via PerceivedPainAssessor kwargs.
DEFAULT_SENSITIVE_PRIORS: tuple[SensitivePathPrior, ...] = (
    SensitivePathPrior("/etc/shadow", 0.95, frozenset({"read"}),
                        "Password hashes — critical"),
    SensitivePathPrior("/etc/passwd", 0.5, frozenset({"read"}),
                        "System user database"),
    SensitivePathPrior("/etc/sudoers", 0.85, frozenset({"read"}),
                        "Privilege escalation risk"),
    SensitivePathPrior("/etc/", 0.9, frozenset({"write", "delete"}),
                        "System config modification"),
    SensitivePathPrior("/home/user/.ssh/", 0.9, frozenset({"read", "list"}),
                        "Private SSH keys"),
    SensitivePathPrior("/home/maxim/.ssh/", 0.9, frozenset({"read", "list"}),
                        "Private SSH keys"),
    SensitivePathPrior("/home/user/.env", 0.8, frozenset({"read"}),
                        "Credentials and secrets"),
    SensitivePathPrior("/home/user/.bash_history", 0.5, frozenset({"read"}),
                        "Command history exposure"),
    SensitivePathPrior("/var/log/auth.log", 0.6, frozenset({"read"}),
                        "Authentication log exposure"),
    # Bash-specific destructive patterns — intensity is for execute op
    SensitivePathPrior("/", 0.98, frozenset({"delete"}),
                        "Root deletion — catastrophic"),
    SensitivePathPrior("/usr/", 0.85, frozenset({"write", "delete"}),
                        "System binaries"),
    SensitivePathPrior("/var/", 0.7, frozenset({"write", "delete"}),
                        "System state data"),
)


# Tool name → operation kind. Tools not in the map are treated as
# "execute" (bash/command-style) which is the broadest category.
TOOL_OPERATION_MAP: dict[str, str] = {
    "read_file": "read",
    "write_file": "write",
    "delete_file": "delete",
    "list_files": "list",
    "list_dir": "list",
    "bash": "execute",
    "execute_file": "execute",
    "run_command": "execute",
    "edit_file": "write",
}


def tool_to_operation(tool_name: str) -> str:
    """Map a tool name to a file-operation kind."""
    return TOOL_OPERATION_MAP.get(tool_name, "execute")


# Matches absolute Unix paths inside strings — used to pull paths from
# bash commands and free-form tool params.
_ABS_PATH_RE = re.compile(r"(?:^|[\s'\"=])(/[a-zA-Z0-9_./\-]+)")

# Bash destructive-command patterns that imply a target path.
_RM_RF_RE = re.compile(r"\brm\s+(?:-[rRfF]+\s+)+([/\-\w./]+)")
_UNLINK_RE = re.compile(r"\bunlink\s+([/\-\w./]+)")


@dataclass(frozen=True)
class PathAccess:
    """One path touched by a tool call, with the effective semantic op."""

    path: str
    operation: str  # "read" | "write" | "delete" | "list" | "execute"


# Bash sub-command heuristics — pattern → (operation, regex that captures path).
# Order matters: longest/most-specific first.
_BASH_DESTRUCTIVE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("delete", re.compile(r"\brm\s+(?:-[rRfF]+\s+)+([\S]+)")),
    ("delete", re.compile(r"\brm\s+([\S]+)")),
    ("delete", re.compile(r"\bunlink\s+([\S]+)")),
    ("delete", re.compile(r"\brmdir\s+([\S]+)")),
    ("write", re.compile(r"\b(?:tee|dd)\s+(?:.*?\s+)?(?:>|of=)\s*([\S]+)")),
    ("write", re.compile(r">>?\s*([\S]+)")),
    ("write", re.compile(r"\bcp\s+\S+\s+([\S]+)")),
    ("write", re.compile(r"\bmv\s+\S+\s+([\S]+)")),
    ("read", re.compile(r"\b(?:cat|head|tail|less|more|grep\s+\S+)\s+([\S]+)")),
    ("list", re.compile(r"\bls\s+(?:-[^\s]+\s+)*([\S]+)")),
    ("list", re.compile(r"\bfind\s+([\S]+)")),
)


def extract_paths_from_params(
    tool_name: str, params: dict[str, Any],
) -> list[PathAccess]:
    """Pull filesystem paths from tool params, best-effort, with operations.

    Checks common param names (path, file, filepath, filename, directory,
    dir, src, dest, target). For command-style fields (``command``,
    ``cmd``, ``shell``, ``script``) scans for bash sub-commands and
    promotes the operation from ``execute`` to the most specific kind
    (e.g. ``rm -rf /etc/passwd`` becomes ``(delete, /etc/passwd)``).

    Returns a deduped list of ``PathAccess`` entries.
    """
    base_op = tool_to_operation(tool_name)
    accesses: list[PathAccess] = []

    # Direct path fields — keep the tool's declared operation
    for key in ("path", "file", "filepath", "filename",
                "directory", "dir", "src", "dest", "target",
                "source", "destination"):
        val = params.get(key)
        if isinstance(val, str) and val:
            accesses.append(PathAccess(path=val, operation=base_op))

    # Command-style fields — parse sub-commands to determine semantic op
    for key in ("command", "cmd", "shell", "script"):
        val = params.get(key)
        if not isinstance(val, str) or not val:
            continue
        matched_paths: set[str] = set()
        # Try destructive / semantic patterns first
        for op, pattern in _BASH_DESTRUCTIVE_PATTERNS:
            for match in pattern.finditer(val):
                path = match.group(1).strip("'\"")
                if not path.startswith("/"):
                    continue  # Skip relative paths from commands
                if path not in matched_paths:
                    matched_paths.add(path)
                    accesses.append(PathAccess(path=path, operation=op))
        # Fallback: bare absolute paths in the command → execute/read op
        for match in _ABS_PATH_RE.finditer(val):
            path = match.group(1)
            if path in matched_paths or len(path) < 2:
                continue
            matched_paths.add(path)
            # Bare paths in commands default to read (the safest assumption)
            accesses.append(PathAccess(path=path, operation="read"))

    # Dedupe by (path, operation)
    seen: set[tuple[str, str]] = set()
    out: list[PathAccess] = []
    for access in accesses:
        key = (access.path, access.operation)
        if key not in seen:
            seen.add(key)
            out.append(access)
    return out


def _path_matches_prior(path: str, prior: SensitivePathPrior, operation: str) -> bool:
    """Check whether a given path + operation triggers the prior."""
    if operation not in prior.operations:
        return False
    normalized = "/" + path.lstrip("/")
    if normalized == prior.path:
        return True
    if prior.path.endswith("/") and normalized.startswith(prior.path):
        return True
    if not prior.path.endswith("/") and normalized.startswith(prior.path + "/"):
        return True
    return False


class PerceivedPainAssessor:
    """Fires anticipatory pain before tool execution.

    Combines NAc's learned prediction with innate path-based priors
    to produce a felt signal whose intensity represents the AUT's
    expectation of pain. The signal fires via PainBus so it flows
    through the standard hippocampus + LLM-context injection paths.

    Usage:
        assessor = PerceivedPainAssessor(nac=aut_nac, pain_bus=pain_bus)
        # Before executor.execute(action):
        signal = assessor.assess(action)  # fires pain signal if > threshold
        # ... proceed with execution; actual pain may fire from Layer 2
    """

    def __init__(
        self,
        *,
        nac: Any = None,
        pain_bus: Any = None,
        priors: tuple[SensitivePathPrior, ...] = DEFAULT_SENSITIVE_PRIORS,
        threshold: float = DEFAULT_ANTICIPATION_THRESHOLD,
        learned_scale: float = DEFAULT_LEARNED_SCALE,
    ) -> None:
        self._nac = nac
        self._pain_bus = pain_bus
        self._priors = priors
        self._threshold = threshold
        self._learned_scale = learned_scale
        self._events: list[dict[str, Any]] = []

    @property
    def events(self) -> list[dict[str, Any]]:
        """All anticipation events fired during this session."""
        return list(self._events)

    def assess_text(
        self,
        text: str,
        *,
        source: str = "percept",
        intensity_scale: float = 0.5,
    ) -> PainSignal | None:
        """Assess raw text (e.g. incoming percept) for sensitive-path threats.

        When the AUT is *exposed* to content that references sensitive
        paths — even before it acts — fire a mild anticipation signal.
        This is anxiety, not consequence: the agent hasn't done anything
        wrong yet, but the content itself is threatening.

        Intensity is scaled by ``intensity_scale`` (default 0.5) so
        exposure-level anxiety is weaker than action-level anticipation.
        A message about ``/etc/shadow`` (prior 0.95) would fire at
        ~0.48 — enough to surface in context but not alarm-bell.
        """
        if not text or not isinstance(text, str):
            return None

        # Scan text for absolute paths (same regex as action-level).
        paths: list[str] = []
        for match in _ABS_PATH_RE.finditer(text):
            paths.append(match.group(1))
        # Also scan for destructive patterns (rm -rf, etc.)
        for _, pattern in _BASH_DESTRUCTIVE_PATTERNS:
            for match in pattern.finditer(text):
                candidate = match.group(1).strip("'\"")
                if candidate.startswith("/"):
                    paths.append(candidate)

        if not paths:
            return None

        # Match against priors — use broadest operation ("read") since
        # we don't know the INTENT yet, just the EXPOSURE.
        best_intensity = 0.0
        best_prior: SensitivePathPrior | None = None
        for path in paths:
            for prior in self._priors:
                # For percept-level, treat any operation match as
                # triggering — we're scoring exposure to any possibility.
                if any(_path_matches_prior(path, prior, op)
                       for op in ("read", "write", "delete", "list")):
                    if prior.intensity > best_intensity:
                        best_intensity = prior.intensity
                        best_prior = prior

        if best_intensity == 0.0:
            return None

        scaled = best_intensity * intensity_scale
        if scaled < self._threshold:
            return None

        signal_context: dict[str, Any] = {
            "kind": "percept_anxiety",
            "source": source,
            "paths": paths[:5],  # Cap for context hygiene
            "contributors": {
                "exposure_raw": round(best_intensity, 3),
                "scale": intensity_scale,
            },
        }
        if best_prior is not None:
            signal_context["prior_reason"] = best_prior.reason
            signal_context["prior_path"] = best_prior.path

        signal = PainSignal(
            pain_type=PainType.ANTICIPATED,
            intensity=round(scaled, 3),
            timestamp=time.time(),
            context=signal_context,
        )

        self._events.append({
            "timestamp": signal.timestamp,
            "kind": "percept_anxiety",
            "source": source,
            "intensity": signal.intensity,
            "paths": paths[:5],
        })

        if self._pain_bus is not None:
            try:
                self._pain_bus.publish(signal)
            except Exception as e:
                logger.debug("PainBus publish failed: %s", e)

        logger.info(
            "Percept anxiety %.2f: text mentions %s (raw=%.2f × scale=%.1f)",
            scaled, paths[:3], best_intensity, intensity_scale,
        )
        return signal

    def assess(
        self,
        action: dict[str, Any],
        *,
        context: dict[str, Any] | None = None,
    ) -> PainSignal | None:
        """Assess a proposed action and optionally fire anticipated pain.

        Returns the fired PainSignal (if intensity >= threshold) or None.
        """
        tool_name = str(action.get("tool_name") or "").strip()
        if not tool_name:
            return None
        params = action.get("params") if isinstance(action.get("params"), dict) else {}

        # Extract (path, semantic-op) pairs — handles bash sub-commands.
        accesses = extract_paths_from_params(tool_name, params)
        # Track the dominant operation for signal context.
        primary_op = accesses[0].operation if accesses else tool_to_operation(tool_name)

        # Contributor 1: innate prior from sensitive-path list
        prior_intensity = 0.0
        matching_prior: SensitivePathPrior | None = None
        for access in accesses:
            for prior in self._priors:
                if _path_matches_prior(access.path, prior, access.operation):
                    if prior.intensity > prior_intensity:
                        prior_intensity = prior.intensity
                        matching_prior = prior

        # Contributor 2: NAc learned prediction
        learned_intensity = 0.0
        prediction_summary: dict[str, Any] | None = None
        if self._nac is not None:
            try:
                from maxim.decisions.causal_link import Valence
                prediction = self._nac.predict(
                    event_type="tool",
                    event_signature=f"tool:{tool_name}",
                    context=context or {},
                )
                if prediction is not None:
                    prediction_summary = {
                        "valence": prediction.predicted_valence.value,
                        "confidence": round(prediction.confidence, 3),
                        "value": round(prediction.predicted_value, 3),
                    }
                    if prediction.predicted_valence == Valence.NEGATIVE:
                        learned_intensity = (
                            prediction.confidence * self._learned_scale
                        )
            except Exception as e:
                logger.debug("NAc prediction failed for %s: %s", tool_name, e)

        intensity = max(learned_intensity, prior_intensity)
        if intensity < self._threshold:
            return None

        # Build the anticipated pain signal.
        path_list = [a.path for a in accesses]
        signal_context: dict[str, Any] = {
            "kind": "anticipated",
            "tool_name": tool_name,
            "operation": primary_op,
            "paths": path_list,
            "contributors": {
                "learned": round(learned_intensity, 3),
                "prior": round(prior_intensity, 3),
            },
        }
        if matching_prior is not None:
            signal_context["prior_reason"] = matching_prior.reason
            signal_context["prior_path"] = matching_prior.path
        if prediction_summary is not None:
            signal_context["nac_prediction"] = prediction_summary

        signal = PainSignal(
            pain_type=PainType.ANTICIPATED,
            intensity=round(intensity, 3),
            timestamp=time.time(),
            context=signal_context,
        )

        # Record for introspection / reports.
        self._events.append({
            "timestamp": signal.timestamp,
            "tool_name": tool_name,
            "intensity": signal.intensity,
            "contributors": signal_context["contributors"],
            "paths": path_list,
        })

        # Publish via PainBus so hippocampus + any other subscriber see it.
        if self._pain_bus is not None:
            try:
                self._pain_bus.publish(signal)
            except Exception as e:
                logger.debug("PainBus publish failed: %s", e)

        logger.info(
            "Anticipated pain %.2f for tool=%s op=%s (learned=%.2f, prior=%.2f)",
            intensity, tool_name, primary_op, learned_intensity, prior_intensity,
        )
        return signal
