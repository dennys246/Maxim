# Stall Detector — Timeout & TTFT Awareness

**Status:** DRAFT v2 2026-06-04 (post-review fold). Triggered by an Exp 37 PR #5 re-run incident where the sim orchestrator's hardcoded 30s stall threshold fired during legitimate inference on Qwen2.5-32B (TTFT 60-120s), injecting `SYSTEM: REPEATED STALL — call send_message` nudges into the orchestrator's percept stream every 30s. The cumulative prompt growth pushed the next call past PR #321 Stage 3.5's context-overflow admission gate (`_check_context_admission`), surfacing as `peer_backend_failed` / `dispatch_exhausted` at the router layer.
**Scope:** Stage 1 ~250 LOC (combined threshold derivation + in-flight call registry + orchestrator integration — single PR per review consensus). Stage 2 ~80 LOC (heartbeat consults registry with opt-in transitional knob). Stage 3 ~60 LOC (consume `llm_timeout_scalability.md` Stage 4 adaptive prediction; 1.1).
**Target versions:** Stage 1 → 1.0 (blocks Exp 37 PR #5 today). Stage 2 → 1.0.x (behavioral change for heartbeat operators; opt-in for 1.0, default in 1.1). Stage 3 → 1.1.
**Gates:** Stage 1 gates itself on "does this break self-hosted leaders running models >24B in sim mode?" — yes today. Validated by the Exp 37 incident.
**Driving incident:** 2026-06-04. Exp 37 PR #5 post-fix re-run on Qwen2.5-32B via Mac Mini leader. Sim orchestrator's 30s stall fired at #3 (169.72s), #4 (199.78s), every 30s thereafter. The actual prompt-corruption mechanism is *cumulative nudge injection past the context-overflow admission gate*, not "the LLM call timed out." Each `inject_cli(SYSTEM: REPEATED STALL …)` appends a user-role message to the orchestrator's next dispatched prompt. With Qwen2.5-32B's typical `n_ctx=8192`, 4-6 nudges plus the growing scene/percept history push the next call past `_PROXY_CHAR_TO_TOKEN_RATIO * (prompt + max_tokens + overhead) > n_ctx`. The router catches HTTP 413, exhausts providers, raises `peer_backend_failed` at 00:37:18.

**Depends on:**
- [`runtime/config_loader.py::LaneTierConfig.timeout_s`](../../src/maxim/runtime/config_loader.py) — per-tier `timeout_s` field shipped via PR #321 ([`llm_timeout_scalability.md`](llm_timeout_scalability.md) Stage 2)
- [`runtime/lane_models.py::LaneConfig.remote_timeout_s`](../../src/maxim/runtime/lane_models.py) — env-var bridge path
- [`runtime/lane_backends.py`](../../src/maxim/runtime/lane_backends.py) — lines 985-1003 thread `cfg.remote_timeout_s` into `providers[provider_key]["timeout_s"]`. This is the **backend-facing surface** the stall detector consults — NOT `resolve_setting` (per Integration review B2: prevents silent drift if `_coerce_for_field` or future overrides diverge)
- [`models/language/router.py::LLMRouter._dispatch`](../../src/maxim/models/language/router.py) — the **canonical instrumentation site** (per consolidated review: covers the provider-failover loop that per-backend wrapping misses)
- [`models/language/maxim_peer_backend.py::_stream_response`](../../src/maxim/models/language/maxim_peer_backend.py) — updates `last_byte_at` on every chunk arrival, including PR #320's TTFT keepalive frames (`: keepalive\n\n`) — the SSE parser receives these as bytes-on-wire
- [`runtime/function_router.py::FunctionRouter.resolve`](../../src/maxim/runtime/function_router.py) — `resolve("sim_orchestrator")` returns the currently-routed tier including failover (large → medium). The stall detector consults this NOT a hardcoded tier
- [`simulation/orchestrator.py::_stall_detector`](../../src/maxim/simulation/orchestrator.py) — current implementation (lines 2200-2440). Primary fix surface
- [`runtime/heartbeat.py::HeartbeatMonitor`](../../src/maxim/runtime/heartbeat.py) — Stage 2 target. Different signal from orchestrator (warns, doesn't inject) — gets DIFFERENT integration (floor + in-flight check, NOT lane-timeout ceiling — per Executor review I3)

**Enables:**
- Sim runs against self-hosted leaders serving 24B+ models without false-positive stall nudges destroying inference
- Exp 37 PR #5 post-fix re-run on the apparatus
- Generalizes: any future operator running `maxim --sim` with a configured large-tier `timeout_s > 30s` gets correct stall behavior by default

---

## Front-gate scope pressure (Principle 3)

**Question:** does this need its own mechanism, or can it ride on existing infrastructure?

| Candidate | Why insufficient (or sufficient) |
|---|---|
| Raise `MAXIM_SIM_STALL_THRESHOLD_S` env var globally | **Insufficient.** Operators must know to set it (Exp 37's harness didn't). Per-tier semantics aren't expressible in one global. |
| Disable the stall detector entirely | **Insufficient.** Catches real stalls (ping-pong, dead-zone after `respond success=False`). Disabling regresses real bugs. |
| Per-tier env-vars (`MAXIM_SIM_STALL_THRESHOLD_LARGE_S`) | **Insufficient.** Duplicates `lanes.<tier>.timeout_s`. Two sources of truth drift. |
| Stall detector consults `cfg.remote_timeout_s` per-active-tier + threshold-only fix | **Insufficient on its own** per Integration B1. Solves the "30s baseline is wrong" half but NOT the "cumulative nudges → admission gate" half. A 70s inference with a 600s threshold still gets ONE nudge at 30s (from the floor) before the threshold-derived ceiling kicks in. One nudge can push a borderline prompt past `n_ctx`. |
| **NEW: cross-component in-flight LLM call registry + threshold derivation, landing together** | **Sufficient as Stage 1.** Honestly named: a new module-level mutable global (`runtime/llm_call_registry.py`) — exactly the shape CLAUDE.md's "Mutable globals + module extraction" lesson categorizes. Justified because (a) the orchestrator stall thread, (b) heartbeat monitor thread, and (c) `maxim doctor` all need read-side access to "is there an LLM call in flight right now?", and (d) the LLMRouter is the only single layer that knows both the start and end of every dispatched call. No existing surface (orchestrator's `_tools_attempted`, heartbeat's `set_loop_state_hook`, function_router's per-tier routing) provides the cross-component answer. The registry IS new architecture; the plan owns that framing. |
| Unify three stall detectors into one timing source | **Deferred to Stage 2.** The heartbeat monitor's signal is *agent-loop idle*, not *LLM-call latency* — these need different floors per Executor review I3. Real unification means consolidating the activity-timestamp sources too, which is Tier-2 work. Stage 2 ships the targeted heartbeat fix (consult `any_call_in_flight` for suppression; keep its own floor). |
| Adaptive prediction consumer | **Deferred to Stage 3 (1.1).** Reuses `llm_timeout_scalability.md` Stage 4. Composes cleanly on Stage 1's `**future`-accepting signature. |

**Verdict:** Stage 1 introduces ONE real architectural addition (cross-component LLM call registry) and one straightforward derivation function. The framing is honest: registry IS new infrastructure with a CC3-compliant shape; the threshold function consumes already-shipped per-tier `timeout_s`. Stage 2 is a behavioral fix to the heartbeat that needs a transitional opt-in knob. Stage 3 is 1.1 research. No mechanism is hidden as "additive layer."

---

## Failure-mode anatomy

**The actual damage mechanism is two-layered:**

| Layer | Signal | Mechanism | Pre-existing fix? |
|---|---|---|---|
| 1: stall fires false-positive | Static 30s threshold vs 60-120s TTFT on big models | First nudge at 30s injects `SYSTEM: REPEATED STALL` user-message into orchestrator percept stream | NO — this plan's Stage 1 derivation |
| 2: cumulative nudges break next call | Each `inject_cli` appends to next dispatched prompt | After N nudges, `prompt_tokens + max_tokens + overhead > n_ctx` → PR #321 Stage 3.5 admission gate rejects with HTTP 413 → router exhausts retries → `peer_backend_failed` | NO — this plan's Stage 1 registry suppresses nudges during in-flight calls, breaking the cumulative-injection chain |

**Why Stage 1 alone (without Stage 2) is required for 1.0:** the threshold derivation prevents the FIRST nudge from firing at 30s when the lane timeout is 600s. But a borderline-fit prompt + a 70s legitimate inference + a 30s floor would still produce one nudge at 30s (before the inference finishes) → admission gate cascade on the NEXT call. The registry's `any_call_in_flight(tier=X)` suppression closes that gap.

**Walk-through with Stage 1 applied:**

| t (s) | Event | Behavior pre-Stage 1 | Behavior post-Stage 1 |
|---|---|---|---|
| 0 | Orchestrator dispatches LLM call via LLMRouter._dispatch | — | `register_call_start(tier="large", call_id=uuid)` |
| 0 | TTFT begins on Qwen2.5-32B | — | — |
| 30 | Stall thread polls | Fires nudge #1 (30s > 30s threshold) | Checks `any_call_in_flight(tier="large")` → True; suppress |
| 60 | TTFT continues (PR #320 keepalive frame #1 emitted by leader) | Fires nudge #2 | Suppress (still in flight; `last_byte_at` updated) |
| 90 | First real token arrives | Nudges already corrupted prompt | Suppress (call still in flight) |
| 120 | LLM call returns | Cumulative damage already present | `register_call_end` → registry empty |
| 122 | Orchestrator advances turn | Turn advances despite damage | Turn advances cleanly |
| 122 | `_last_activity_time` resets | — | — |

Stage 1 prevents the entire chain.

---

## Stage 1 — Threshold derivation + in-flight call registry  *(target 1.0, single PR)*

### Module: `runtime/llm_call_registry.py` (new, ~80 LOC)

```python
"""Cross-component in-flight LLM call registry.

Architectural invariant: this module is the SINGLE source of truth for
"is there an LLM call in flight right now?" across the codebase. Three
consumers query it:
  - simulation/orchestrator.py::_stall_detector (Stage 1)
  - runtime/heartbeat.py::HeartbeatMonitor (Stage 2, opt-in)
  - doctor/checks.py::Derived Config rows (1.0.x)

The instrumentation site is models/language/router.py::LLMRouter._dispatch
— ONE register_call_start at dispatch entry, ONE register_call_end in
try/finally at dispatch exit. The wrap covers LLMRouter._try_provider's
provider-fallback loop, so the registry sees one continuous in-flight
window across provider retries.

SHAPE-FROZEN at 1.0 (CC3): _InFlightCall is shape-frozen. Adding optional
fields with defaults at the end is non-breaking. Adding required fields
post-1.0 is a major-version bump for downstream consumers of the snapshot
API.
"""

from __future__ import annotations
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class _InFlightCall:
    """SHAPE-FROZEN at 1.0 (CC3): registry value type."""
    call_id: str
    tier: str
    started_at: float
    last_byte_at: float  # updated by stream consumer on chunk arrival incl. keepalives

_registry: dict[str, _InFlightCall] = {}
_lock = threading.RLock()

# Defense against SIGKILL / unwrapped exit paths: entries older than
# this auto-expire on read. Floor is generous (covers worst-case big-model
# timeouts at 600s + 2× safety) so legitimate in-flight calls never get
# filtered out.
_STALE_ENTRY_TTL_S = 1800.0

def register_call_start(*, tier: str) -> str:
    """Register the start of an LLM dispatch. Returns the call_id to be
    passed to register_call_end / register_byte_received."""
    cid = str(uuid.uuid4())
    now = time.time()
    with _lock:
        _registry[cid] = _InFlightCall(call_id=cid, tier=tier, started_at=now, last_byte_at=now)
    return cid

def register_call_end(call_id: str) -> None:
    with _lock:
        _registry.pop(call_id, None)

def register_byte_received(call_id: str) -> None:
    """Update last_byte_at. Called from _stream_response chunk loop (incl.
    PR #320 TTFT keepalive frames). Silently no-ops if call_id is unknown."""
    with _lock:
        cur = _registry.get(call_id)
        if cur is not None:
            _registry[call_id] = _InFlightCall(
                call_id=cur.call_id,
                tier=cur.tier,
                started_at=cur.started_at,
                last_byte_at=time.time(),
            )

def _live_entries(now: float) -> dict[str, _InFlightCall]:
    return {k: v for k, v in _registry.items() if now - v.started_at < _STALE_ENTRY_TTL_S}

def any_call_in_flight(*, tier: str | None = None) -> bool:
    """True if any non-stale LLM call (optionally filtered by tier) is in flight."""
    now = time.time()
    with _lock:
        for v in _live_entries(now).values():
            if tier is None or v.tier == tier:
                return True
    return False

def oldest_byte_silence_s(*, tier: str | None = None) -> float | None:
    """Seconds since the most recent byte was received across in-flight calls
    in the given tier. None if no in-flight calls. Used by the stall detector's
    'wedged connection' safety net — distinguishes 'call is alive but slow'
    (bytes flowing via keepalives) from 'call is wedged' (no bytes for N s)."""
    now = time.time()
    with _lock:
        candidates = [
            now - v.last_byte_at for v in _live_entries(now).values()
            if tier is None or v.tier == tier
        ]
    return min(candidates) if candidates else None
```

### Module: `runtime/stall_threshold.py` (new, ~60 LOC)

```python
"""Dynamic stall-threshold derivation consuming per-tier timeout_s.

Architectural invariant: this module is the canonical stall-threshold
derivation site. New stall detectors MUST consult compute_stall_threshold()
rather than defining their own hardcoded thresholds. Enforced via CI grep
(see Regression guards section of stall_detector_timeout_awareness.md).
"""

from __future__ import annotations
import os
from typing import Any

DEFAULT_STALL_FLOOR_S = 30.0
DEFAULT_STALL_MARGIN_S = 10.0
DEFAULT_MAX_BYTE_SILENCE_S = 90.0  # 3× PR #320 default keepalive interval

def compute_stall_threshold(
    *,
    lane_tier: str,
    lane_timeout_s: float | None = None,
    model: str | None = None,  # reserved for Stage 3 adaptive prediction
    prompt_tokens: int | None = None,  # reserved for Stage 3
    floor_env_var: str = "MAXIM_STALL_FLOOR_S",
    margin_env_var: str = "MAXIM_STALL_MARGIN_S",
    **future: Any,
) -> float:
    """Derive the effective stall threshold for the given lane tier.

    Threshold = max(floor, lane_timeout_s + margin). When lane_timeout_s
    is None (operator hasn't configured it), returns floor unchanged —
    identical to pre-fix behavior, so cloud operators see no regression.

    lane_tier and model are accepted as keyword arguments now even though
    only lane_tier is consumed today; Stage 3's adaptive prediction will
    consume model + prompt_tokens without a signature change.
    """
    floor_s = _read_clamped_env(floor_env_var, DEFAULT_STALL_FLOOR_S, lo=5.0, hi=3600.0)
    margin_s = _read_clamped_env(margin_env_var, DEFAULT_STALL_MARGIN_S, lo=0.0, hi=120.0)
    if lane_timeout_s is None or lane_timeout_s <= 0:
        return floor_s
    return max(floor_s, float(lane_timeout_s) + margin_s)

def max_byte_silence_threshold_s() -> float:
    """The "wedged connection" detection threshold — seconds since the
    most-recent byte before the stall detector considers the connection
    dead (independent of total call age). Tied to PR #320's keepalive
    interval: 3× default of 30s = 90s. A healthy in-flight call (TTFT
    or generation) emits bytes well within this window."""
    return _read_clamped_env(
        "MAXIM_STALL_MAX_BYTE_SILENCE_S",
        DEFAULT_MAX_BYTE_SILENCE_S,
        lo=30.0,
        hi=600.0,
    )

def _read_clamped_env(name: str, default: float, *, lo: float, hi: float) -> float:
    try:
        val = float(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        val = default
    return max(lo, min(hi, val))
```

### LLMRouter integration (~30 LOC delta in `models/language/router.py`)

Wrap `_dispatch` (the top-level dispatch entry, covering `_try_provider`'s provider-fallback loop):

```python
def _dispatch(self, ...):
    from maxim.runtime.llm_call_registry import register_call_start, register_call_end
    # Tier is resolved here from the lane / function-router; use it.
    call_id = register_call_start(tier=resolved_tier)
    try:
        return self._dispatch_inner(...)  # existing body
    finally:
        register_call_end(call_id)
```

### `_MaximPeerBackend._stream_response` instrumentation (~10 LOC delta)

```python
def _stream_response(self, ..., call_id: str | None = None):
    from maxim.runtime.llm_call_registry import register_byte_received
    for chunk in iter_lines(...):
        if call_id is not None:
            register_byte_received(call_id)  # includes : keepalive\n\n frames
        ...
```

The `call_id` is threaded through from `LLMRouter._dispatch` via the existing per-call context. Same instrumentation parallel in `_OpenAIBackend` streaming path (cloud backends' keepalives are SSE-spec-compliant too).

### Orchestrator integration (~30 LOC delta in `simulation/orchestrator.py`)

```python
def _stall_detector() -> None:
    from maxim.runtime.stall_threshold import compute_stall_threshold, max_byte_silence_threshold_s
    from maxim.runtime.llm_call_registry import any_call_in_flight, oldest_byte_silence_s
    from maxim.runtime.function_router import FunctionRouter

    # Resolve tier at construction; consult router rather than hardcoding "large"
    # so failover paths (large → medium) yield the correct timeout consumer.
    _resolved_tier = FunctionRouter.resolve("sim_orchestrator") or "large"

    # Lane timeout from backend-facing surface (not resolve_setting; per
    # Integration review B2). Read once at construction; sims don't hot-reload.
    _lane_timeout_s = _get_lane_remote_timeout_s(_resolved_tier)  # helper

    def _current_threshold() -> float:
        return compute_stall_threshold(
            lane_tier=_resolved_tier,
            lane_timeout_s=_lane_timeout_s,
        )

    # ... existing loop body ...

    while not stop_event.is_set():
        # ... existing check_interval / max_turns / skip-while-paused logic ...

        # Suppress nudges entirely while an LLM call is legitimately in flight
        # AND its byte-silence is within the keepalive-derived budget.
        if any_call_in_flight(tier=_resolved_tier):
            silence_s = oldest_byte_silence_s(tier=_resolved_tier)
            if silence_s is None or silence_s < max_byte_silence_threshold_s():
                # Call alive, bytes flowing (or keepalives flowing) — suppress
                continue
            # No bytes for >2× keepalive interval — connection is wedged.
            # Fall through to fire the nudge as a stuck-call warning.

        # ... existing ping_pong / time_stalled / nudge-fire logic with
        #     stall_threshold_s replaced by _current_threshold() ...
```

### Knobs

| Env var | Default | Clamped | Notes |
|---|---|---|---|
| `MAXIM_STALL_FLOOR_S` | 30s | [5, 3600] | New unified floor; supersedes per-detector floors (with backward-compat aliases per below) |
| `MAXIM_STALL_MARGIN_S` | 10s | [0, 120] | Slack past lane timeout |
| `MAXIM_STALL_MAX_BYTE_SILENCE_S` | 90s | [30, 600] | Wedged-connection threshold (3× PR #320 default keepalive of 30s) |
| `MAXIM_SIM_STALL_THRESHOLD_S` | (none) | — | DEPRECATED alias for `MAXIM_STALL_FLOOR_S`. Read by `compute_stall_threshold` if `MAXIM_STALL_FLOOR_S` is unset. Removed in 1.1. |

### Regression guard

`tests/unit/test_stall_threshold.py` — pins:
- Floor returned when `lane_timeout_s` is None / 0 / negative
- Derived value when lane_timeout_s=600: max(30, 600+10) = 610
- Env override of floor + margin
- Clamps for malformed values
- `**future` kwargs accepted without TypeError (forward-compat for Stage 3)

`tests/unit/test_llm_call_registry.py` — pins:
- Register start → in_flight True; end → in_flight False
- Multiple concurrent calls track independently
- `last_byte_at` updates monotonically; oldest_byte_silence_s tracks correctly
- Tier filter isolates large/medium/small
- Stale entries (older than `_STALE_ENTRY_TTL_S`) auto-filter on read
- Thread safety under contention (parallel start/end/query)

`tests/unit/test_stall_detector_with_registry.py` — pins (with mocked orchestrator state):
- In-flight + recent bytes → suppress
- In-flight + byte silence > max_byte_silence_threshold_s → fire (wedged-call branch)
- No in-flight + ping_pong → fire (real stall)
- No in-flight + idle past threshold → fire (real stall)
- Tier filtering: large-tier in-flight + medium-tier query → ignored (orchestrator's tier-aware query is correct)

CI grep regression guard (in `.github/workflows/test.yml`):

```yaml
- name: No new hardcoded stall thresholds outside the canonical module
  run: |
    # Catches hardcoded 30-as-stall-threshold literals in code,
    # excluding the canonical module and its tests
    if grep -nE "STALL.*_?=.*30(\.0)?[^0-9]|stall_threshold.*_?=.*30(\.0)?[^0-9]" \
         src/maxim/ tests/ \
         --include="*.py" \
         | grep -vE "runtime/stall_threshold\.py|tests/.*test_stall" ; then
      echo "Found hardcoded stall threshold of 30 outside canonical module"
      exit 1
    fi
```

---

## Stage 2 — Heartbeat consults registry (opt-in)  *(target 1.0.x)*

**Per Executor review I3:** the heartbeat's signal is *agent-loop idle*, not *LLM-call latency*. Applying the lane-timeout ceiling to it conflates two distinct signals. Heartbeat should use FLOOR for stall detection AND use `any_call_in_flight` to suppress warnings during legitimate inference.

### Implementation

In `runtime/heartbeat.py::_check_stall`:

```python
def _check_stall(self) -> None:
    from maxim.runtime.llm_call_registry import any_call_in_flight
    use_lane_timeout = _read_env_bool("MAXIM_HEARTBEAT_USE_LANE_TIMEOUT", default=False)
    # Suppress when an LLM call is in flight regardless of lane-timeout opt-in;
    # this is the load-bearing fix (agent loop isn't stalled, it's awaiting inference)
    if any_call_in_flight():
        return
    if use_lane_timeout:
        # Opt-in: derive threshold from active large-tier timeout
        from maxim.runtime.stall_threshold import compute_stall_threshold
        threshold = compute_stall_threshold(lane_tier="large", lane_timeout_s=self._lane_timeout_s)
    else:
        threshold = self._stall_threshold  # existing floor
    # ... existing comparison + warn logic ...
```

### Transitional knob

`MAXIM_HEARTBEAT_USE_LANE_TIMEOUT=1` opt-in for 1.0. Default ON in 1.1. Documented in 1.0 release notes. Operators relying on the implicit 30s heartbeat floor see no change unless they opt in.

---

## Stage 3 — Adaptive prediction (1.1, blocked on sibling plan)

Once [`llm_timeout_scalability.md`](llm_timeout_scalability.md) Stage 4 ships an `predict_completion_s(tier, model, prompt_tokens)` helper, `compute_stall_threshold`'s reserved `model` + `prompt_tokens` kwargs become consumed:

```python
def compute_stall_threshold(..., model: str | None = None, prompt_tokens: int | None = None, **future):
    if _adaptive_enabled() and model and prompt_tokens:
        from maxim.runtime.lane_throughput import predict_completion_s
        predicted = predict_completion_s(lane_tier, model, prompt_tokens)
        if predicted is not None:
            return max(floor_s, 2 * predicted)  # 2× safety margin
    # ... existing static-timeout path ...
```

The 2× factor mirrors `llm_timeout_scalability.md` Stage 4's cold-start safety margin.

---

## Architectural invariant (to add to CLAUDE.md "Architectural invariants" section)

**[engineering] `runtime/llm_call_registry.py` is the canonical in-flight LLM call surface; `runtime/stall_threshold.py::compute_stall_threshold` is the canonical stall-threshold derivation.** The orchestrator stall detector + heartbeat monitor consult both modules; new stall detectors MUST also consult them (no hardcoded thresholds, no independent activity-tracking dicts). Registry instrumentation sits at `models/language/router.py::LLMRouter._dispatch` (one entry/exit per logical dispatch, covering the provider-fallback retry loop). Byte-arrival updates flow from per-backend `_stream_response` chunk loops via `register_byte_received(call_id)`. Adding optional fields to `_InFlightCall` at the end of the dataclass is non-breaking; reordering or removing fields is a major-version bump (SHAPE-FROZEN at 1.0 per CC3). Regression guards: [tests/unit/test_stall_threshold.py](../../tests/unit/test_stall_threshold.py) + [tests/unit/test_llm_call_registry.py](../../tests/unit/test_llm_call_registry.py) + [tests/unit/test_stall_detector_with_registry.py](../../tests/unit/test_stall_detector_with_registry.py) + CI grep in [.github/workflows/test.yml](../../.github/workflows/test.yml) blocking hardcoded 30s stall thresholds outside the canonical module.

---

## Roll-out order

1. **Stage 1 (target this PR, ~250 LOC + ~300 LOC tests):**
   - `runtime/stall_threshold.py` module
   - `runtime/llm_call_registry.py` module (frozen dataclass SHAPE-FROZEN at 1.0 CC3)
   - `LLMRouter._dispatch` wrap (register_call_start / register_call_end in try/finally)
   - `_MaximPeerBackend._stream_response` + `_OpenAIBackend._stream_response` instrumented with `register_byte_received`
   - `simulation/orchestrator.py::_stall_detector` consults `FunctionRouter.resolve("sim_orchestrator")` for tier + `compute_stall_threshold` for threshold + `any_call_in_flight` for suppression + `oldest_byte_silence_s` for wedged-connection safety net
   - Three new test files (registry, threshold, integrated stall behavior)
   - CI grep regression guard
   - CLAUDE.md Architectural invariant addition
   - `maxim doctor` "Derived Config" section (split from "Configured" — applies to stall threshold, admission gate, VRAM context-fit per Integration review I6; small additive section)
2. **Stage 2 (target 1.0.x, ~80 LOC + ~80 LOC tests):**
   - Heartbeat consults `any_call_in_flight` for suppression
   - Optional `MAXIM_HEARTBEAT_USE_LANE_TIMEOUT=1` opt-in for lane-timeout ceiling
   - 1.0 release notes documenting the behavioral change with the opt-in knob
3. **Stage 3 (target 1.1, ~60 LOC):**
   - Consume `llm_timeout_scalability.md` Stage 4 adaptive prediction
   - `compute_stall_threshold` already accepts `model` + `prompt_tokens` + `**future` — zero signature changes needed

---

## Open questions (none remaining after review fold)

All Q1-Q7 from the prior draft have been resolved by the parallel review:
- **Q1 (wrap site):** `LLMRouter._dispatch` (Executor + Integration converged)
- **Q2 (AUT vs orchestrator):** the orchestrator's `send_message` blocks on AUT response, so suppressing during AUT's tier="large" calls is correct (AUT call IS what orchestrator is waiting on)
- **Q3 (stale entries):** `_STALE_ENTRY_TTL_S = 1800.0` auto-filter on read; no GC thread
- **Q4 (module location):** `runtime/` (config-derived, multi-consumer)
- **Q5 (TTFT keepalive interaction):** consumed via `register_byte_received` from `_stream_response` chunk loop (keepalive frames arrive as bytes; SSE parser handles them)
- **Q6 (tier hardcoding):** consult `FunctionRouter.resolve("sim_orchestrator")` for active tier including failover (Integration I3 fix)
- **Q7 (heartbeat behavioral change):** opt-in via `MAXIM_HEARTBEAT_USE_LANE_TIMEOUT` for 1.0, default in 1.1 (Architecture I3 fix)
