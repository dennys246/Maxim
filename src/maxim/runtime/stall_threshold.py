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
    "non_streaming_age_bound_s",
    "stall_suppression",
    "DEFAULT_MAX_BYTE_SILENCE_S",
    "DEFAULT_STALL_FLOOR_S",
    "DEFAULT_STALL_MARGIN_S",
    "compute_stall_threshold",
    "max_byte_silence_threshold_s",
]


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


def non_streaming_age_bound_s(lane_timeout_s: float | None = None, allowed_s: float | None = None) -> float:
    """How long a NON-streaming planning call may run before the stall detector judges it wedged by age (#1042):
    the time it was allowed — its own ``allowed_s`` (the worker's effective timeout, a timeout-retry's doubled
    allowance included) or, when the caller did not say, the worker's call timeout (``MAXIM_LLM_CALL_TIMEOUT_S``,
    default 300 s) — or a longer configured lane timeout, plus the stall margin."""
    from maxim.agents.llm_worker import _read_llm_call_timeout_env

    own = allowed_s if allowed_s is not None and allowed_s > 0 else _read_llm_call_timeout_env()
    allowed = max(own, float(lane_timeout_s or 0.0))
    return allowed + _read_clamped_env("MAXIM_STALL_MARGIN_S", DEFAULT_STALL_MARGIN_S, lo=0.0, hi=120.0)


def stall_suppression(
    *, turn_in_progress: bool, lane: str | None = None, lane_timeout_s: float | None = None
) -> tuple[bool, float | None]:
    """The simulation stall detector's suppression decision, as one callable (#1042). Returns ``(suppress,
    wedged_s)``:

    - ``(True, None)``: the narrator is BUSY, not stalled. Either it is waiting on the agent's turn
      (``turn_in_progress``: the bridge's ``send_and_wait``, bounded by its response timeout), or an LLM call on
      ``lane`` (default: the planning lane) is in flight and still alive: a STREAMING call that has received bytes
      within :func:`max_byte_silence_threshold_s`, or a NON-streaming call younger than the time it was allowed
      (:func:`non_streaming_age_bound_s`; it reports no bytes, so its age is its only evidence). Any planning-lane
      call counts, the agent-under-test's included (#1043 scopes it to the narrator's own calls).
    - ``(False, s)``: every in-flight call on ``lane`` is wedged — byte-silent ``s`` seconds (streaming) or ``s``
      seconds old (the youngest non-streaming call).
    - ``(False, None)``: nothing is in flight: a real stall candidate.
    """
    from maxim.agents.llm_types import PLANNING_LANE
    from maxim.runtime import llm_call_registry as reg

    if turn_in_progress:
        return True, None
    lane = lane or PLANNING_LANE
    if not reg.any_call_in_flight(tier=lane):
        return False, None
    silence = reg.oldest_byte_silence_s(tier=lane)
    if silence is not None and silence < max_byte_silence_threshold_s():
        return True, None
    calls = reg.non_streaming_calls(tier=lane)
    if any(age < non_streaming_age_bound_s(lane_timeout_s, allowed) for age, allowed in calls):
        return True, None
    youngest = min((age for age, _ in calls), default=None)
    return False, silence if silence is not None else youngest
