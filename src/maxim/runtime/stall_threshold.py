"""Canonical stall-threshold derivation, consuming per-tier ``timeout_s``.

Architectural invariant: this module is the single source of truth for
stall-threshold derivation. New stall detectors MUST consult
:func:`compute_stall_threshold` rather than defining their own hardcoded
thresholds. Enforced via the CI grep in ``.github/workflows/test.yml``
that blocks hardcoded 30s stall threshold literals outside this module
and its tests.

The derivation is dynamic: each call recomputes from the current env-var
state + the provided ``lane_timeout_s`` (which callers should read once
from the backend-facing provider cfg at construction, since sims don't
hot-reload). This makes per-cycle calls in a daemon thread cheap.

See [docs/plans/deferred/stall_detector_timeout_awareness.md] for the load-bearing
rationale and the consolidated review fold that produced the current
signature shape.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "DEFAULT_MAX_BYTE_SILENCE_S",
    "DEFAULT_STALL_FLOOR_S",
    "DEFAULT_STALL_MARGIN_S",
    "PlanningLivenessExhausted",
    "compute_stall_threshold",
    "max_byte_silence_threshold_s",
    "spinner_truth_message",
]


class PlanningLivenessExhausted(RuntimeError):
    """A sim's planning turns failed repeatedly and the bounded retry budget
    is spent (bugs ledger D13).

    Raised by ``run_agentic_loop`` AFTER its normal teardown (state persist +
    bio session end) so the sim aborts loudly instead of idling forever on a
    silently-dropped planning turn. Sim-mode only: an interactive session has
    a human percept source that can always re-arm the loop, so it logs and
    stops retrying instead of raising.
    """


DEFAULT_STALL_FLOOR_S = 30.0
"""Floor: shortest stall threshold regardless of lane configuration.

Picked to match the historical hardcoded threshold so cloud operators
(no lane_timeout_s configured) see no behavioral change. Self-hosted
operators with ``lanes.<tier>.timeout_s`` configured get the higher
``timeout_s + margin`` ceiling automatically."""

DEFAULT_STALL_MARGIN_S = 10.0
"""Slack between when an LLM call would itself time out (HTTP layer) and
when the stall detector declares the orchestrator stalled. The HTTP
layer should raise first; the margin gives that exception a chance to
unwind before the stall detector fires a corrupting nudge."""

DEFAULT_MAX_BYTE_SILENCE_S = 90.0
"""Wedged-connection threshold: seconds without bytes-on-wire before the
stall detector considers the connection dead, even if the call is still
nominally in flight per the registry. 3x PR #320's default keepalive
interval of 30s — a healthy in-flight call (TTFT or generation) emits
bytes well within this window via real tokens or keepalive frames."""


def compute_stall_threshold(
    *,
    lane_tier: str,
    lane_timeout_s: float | None = None,
    model: str | None = None,
    prompt_tokens: int | None = None,
    floor_env_var: str = "MAXIM_STALL_FLOOR_S",
    margin_env_var: str = "MAXIM_STALL_MARGIN_S",
    **future: Any,
) -> float:
    """Derive the effective stall threshold for the given lane tier.

    ``threshold = max(floor, lane_timeout_s + margin)``. When
    ``lane_timeout_s`` is None or non-positive (operator hasn't configured
    it), returns the floor unchanged — identical to pre-fix behavior, so
    cloud operators see no regression.

    ``lane_tier`` is consumed today only for diagnostics (it's logged when
    the threshold differs from the floor). ``model`` and ``prompt_tokens``
    are reserved for Stage 3's adaptive prediction (see
    [`llm_timeout_scalability.md`](llm_timeout_scalability.md) Stage 4)
    and currently ignored. ``**future`` keeps the signature
    forward-compatible: a future param addition won't require call-site
    updates.

    Both floor and margin honor a custom env-var name (``floor_env_var``
    / ``margin_env_var``) so different consumers can have their own
    floors. The heartbeat monitor uses ``MAXIM_HEARTBEAT_STALL_S`` as its
    floor (legacy compat); the orchestrator uses ``MAXIM_STALL_FLOOR_S``
    (with ``MAXIM_SIM_STALL_THRESHOLD_S`` as a deprecated alias read by
    the caller before invoking this function).
    """
    floor_s = _read_clamped_env(floor_env_var, DEFAULT_STALL_FLOOR_S, lo=5.0, hi=3600.0)
    margin_s = _read_clamped_env(margin_env_var, DEFAULT_STALL_MARGIN_S, lo=0.0, hi=120.0)

    if lane_timeout_s is None or lane_timeout_s <= 0:
        return floor_s

    return max(floor_s, float(lane_timeout_s) + margin_s)


def max_byte_silence_threshold_s() -> float:
    """Wedged-connection detection threshold (independent of total call age).

    A healthy in-flight call emits bytes well within this window via real
    tokens or PR #320 TTFT keepalive frames. Exceeding this threshold
    means the connection is wedged (upstream dead, network dropped,
    process hung), and the stall detector should fire as a stuck-call
    warning even if the registry still shows the call as in-flight.
    """
    return _read_clamped_env(
        "MAXIM_STALL_MAX_BYTE_SILENCE_S",
        DEFAULT_MAX_BYTE_SILENCE_S,
        lo=30.0,
        hi=600.0,
    )


def should_hard_abort(
    *,
    stall_duration_s: float,
    threshold_s: float,
    nudge_count: int,
    byte_silence_s: float | None,
    byte_silence_threshold_s: float,
) -> bool:
    """Hard-abort decision for a wedged orchestrator LLM call (bugs ledger D12).

    The stall detector's only actuator used to be the NUDGE — an injected
    percept telling the orchestrator LLM to get moving. Against a HUNG LLM
    call that is useless: the orchestrator thread is blocked inside the call
    and never reads the nudge (observed live 2026-08-18: 'planning first
    probe' at 8,624s and again at 3,286s, server healthy and idle both
    times). This function decides when nudging has provably failed and the
    sim must be terminated loudly instead.

    Two routes, both deliberately conservative (a hard abort kills a
    campaign run — false positives are expensive):

    1. KNOWN-wedged: an in-flight call's byte silence exceeded the
       keepalive-derived threshold (connection provably dead per the PR #320
       contract) AND the stall has additionally outlasted one full stall
       threshold of grace.
    2. PERSISTENT stall: >= 3 nudges have fired without any turn progress
       AND the stall has lasted >= max(3x threshold, threshold + 120s) —
       long enough that every recoverable cause (slow TTFT, one lost nudge,
       a transient provider error with retry) is exhausted.

    Pure function — fully unit-testable; the detector supplies the state.
    """
    if stall_duration_s <= 0 or threshold_s <= 0:
        return False
    if byte_silence_s is not None and byte_silence_s >= byte_silence_threshold_s:
        return stall_duration_s >= byte_silence_threshold_s + threshold_s
    return nudge_count >= 3 and stall_duration_s >= max(3.0 * threshold_s, threshold_s + 120.0)


def spinner_truth_message(
    *,
    between_turns: bool,
    in_flight: bool,
    stall_duration_s: float,
    threshold_s: float,
    nudge_count: int,
    byte_silence_s: float | None,
    byte_silence_threshold_s: float,
) -> str | None:
    """Status-line truth for the between-turns spinner (bugs ledger D14).

    The bridge sets "Orchestrator planning next probe..." when a turn ENDS
    and nothing updates it on any failure path, so during a dropped planning
    turn the display asserts work that py-spy proves is not happening — the
    defect that steered hours of diagnosis toward server/network theories.
    A status display must report OBSERVED state, never intent.

    Returns the corrected spinner text, or ``None`` when the default text is
    truthful (a call really is in flight, or the inter-call gap is still
    within the stall threshold) — the caller leaves the spinner alone.
    ``between_turns=False`` always returns ``None``: mid-turn the spinner
    belongs to the AUT exchange, not the orchestrator.

    Pure function — fully unit-testable; the stall detector supplies the
    same registry-derived state it already polls.
    """
    if not between_turns:
        return None
    if in_flight:
        if byte_silence_s is not None and byte_silence_s >= byte_silence_threshold_s:
            return f"⚠ planning call in flight but silent {int(byte_silence_s)}s (connection may be wedged)"
        return None
    if stall_duration_s >= threshold_s:
        nudge_note = f", {nudge_count} nudge(s) sent" if nudge_count > 0 else ""
        return (
            f"⚠ no LLM call in flight — planning turn may be lost "
            f"({int(stall_duration_s)}s since last turn{nudge_note})"
        )
    return None


_DEPRECATED_FLOOR_ALIASES = {
    "MAXIM_STALL_FLOOR_S": ("MAXIM_SIM_STALL_THRESHOLD_S",),
    # Future deprecations can be added here without touching call sites.
}


def _read_clamped_env(name: str, default: float, *, lo: float, hi: float) -> float:
    """Read a float env var, clamp to [lo, hi], fall back to default on
    parse failure. Negative / out-of-range silently clamps. Honors the
    deprecated-alias map: if ``name`` isn't set but an alias is, the
    alias is read. Keeps the deprecation handling centralized in the
    canonical module instead of forcing callers to mutate os.environ.

    Defensive against concurrent ``os.environ`` mutation by tests'
    monkeypatch fixtures: any RuntimeError ("dictionary changed size
    during iteration") falls back to the default.
    """
    try:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            for alias in _DEPRECATED_FLOOR_ALIASES.get(name, ()):
                raw = os.environ.get(alias)
                if raw is not None and raw != "":
                    break
            else:
                return default
    except (RuntimeError, KeyError):
        return default
    try:
        val = float(raw)
    except (ValueError, TypeError):
        return default
    return max(lo, min(hi, val))
