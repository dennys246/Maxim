"""Recovery-time benchmark for the LLM peer path.

Phase D (2026-04-13) flagged that the sim workload muddies recovery-time
measurement because the orchestrator spends 30+s on local agent work
between LLM calls. That made the leader-restart "first successful call
after leader-ready" gate inconclusive — the peer was trivially
recoverable but the trace showed ~30s gaps between attempts simply
because the next attempt was waiting on the next agent turn, not on the
peer.

This module fires chat completions in a **tight loop** against the
configured peer URL, with zero local work between attempts. The result
is a rigorous recovery-time measurement: the gap between the last
pre-restart success and the first post-restart success is the actual
peer recovery latency, not a workload artifact.

**Design constraints:**

- **Uses ``_MaximPeerBackend`` directly** — same path as production
  inference. No router, no llm_worker, no _inference_lock contention.
  The harness is a minimal wrapper around
  ``_MaximPeerBackend.complete_with_usage`` that fires as fast as
  possible.
- **Emits the same JSONL event shape** as production
  (``peer_backend_call``/``peer_backend_failed``) so existing
  ``jq 'select(.e=="peer_backend_call")'`` queries work unchanged on
  the benchmark output. The only addition is a ``benchmark`` wrapper
  event with aggregate timing so the rerun runbook can extract a
  single recovery-time number without post-processing.
- **No retries on failure** — a failed call is recorded with its error
  class and latency, then the loop fires the next attempt immediately.
  This matches Plan 3's "fail fast, let the caller failover" contract
  and gives us the per-attempt latency distribution during a
  leader-down window.
- **Cooperative cancellation** via SIGINT — the loop polls a stop
  Event between attempts so ``Ctrl+C`` exits cleanly without losing
  the final JSONL record.
- **Does NOT touch ``_inference_lock``** — the backend is instantiated
  fresh and called directly. Multi-worker contention is not in scope
  for recovery-time; this harness measures the peer path, not the
  router path.

**Typical run:**

    maxim bench recovery-time \\
        --url https://maxim.dennyschaedig.com/v1 \\
        --duration 120 \\
        --output /tmp/recovery_bench.jsonl

Then in a second terminal:

    maxim peer restart

Wait until the bench exits, then:

    jq 'select(.e=="benchmark")' /tmp/recovery_bench.jsonl

The ``benchmark`` event carries ``recovery_time_s`` — the observed
gap between the last pre-restart success and the first post-restart
success, or ``null`` if no recovery transition was observed (e.g., the
leader never went down during the run).
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from maxim.models.language.maxim_peer_backend import _MaximPeerBackend
from maxim.models.language.types import (
    BackendAuthFailed,
    BackendDown,
    BackendError,
    BackendInferenceBroken,
    BackendModelMissing,
    BackendOverloaded,
    BackendTimeout,
)
from maxim.utils.http import (
    RequestContext,
    generate_request_id,
    reset_context,
    set_context,
)
from maxim.utils.structured_logging import log_structured

logger = logging.getLogger("maxim.bench.recovery_time")

# Bench agent_id is stable + grep-friendly — the operator can filter
# the JSONL log for just the benchmark calls while leaving sim events
# alone.
BENCH_AGENT_ID = "bench_recovery_time"

# Short prompt to keep the leader's decode time low (tight loop wants
# each call to be ~1s not ~10s). "say hi" is intentionally trivial —
# the benchmark measures the PEER path, not the model's generation
# speed.
BENCH_SYSTEM = "You are a benchmark endpoint. Reply with one word."
BENCH_USER = "say hi"


@dataclass
class BenchAttempt:
    """One call's timing + outcome. Pure data.

    Fields mirror the production ``peer_backend_call`` /
    ``peer_backend_failed`` structured log event shape exactly so the
    bench's JSONL output is wire-compatible — existing
    ``jq 'select(.e=="peer_backend_call") | .input_tokens'`` queries
    work unchanged on bench traces.

    Successes carry ``provider``, ``model``, ``input_tokens``,
    ``output_tokens`` from the returned ``LLMResponse``. Failures
    carry ``error`` (exception class name) and ``fix_hint`` (from
    ``BackendError.fix_hint`` or ``str(exc)`` as the fallback).
    """

    request_id: str
    submit_ts: float  # monotonic seconds since start
    complete_ts: float  # monotonic seconds since start
    latency_ms: float
    status: str  # "success" | "failure"
    outcome: str  # BackendError subclass name, or "ok"
    # Success-path fields (match production peer_backend_call shape)
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    # Failure-path fields (match production peer_backend_failed shape)
    error: str = ""  # exception class name (e.g., "BackendDown")
    fix_hint: str = ""  # human-readable hint from BackendError.fix_hint or str(exc)
    http_status: int | None = None  # exception.status if present


@dataclass
class BenchResult:
    """Aggregate result of a single recovery-time bench run."""

    duration_s: float
    total_attempts: int
    successes: int
    failures: int
    attempts: list[BenchAttempt] = field(default_factory=list)
    # Recovery-time analysis (computed post-run):
    last_success_before_failure: float | None = None  # ts of last pre-outage success
    first_failure: float | None = None  # ts of first failure after last success
    first_success_after_failure: float | None = None  # ts of first post-recovery success
    recovery_time_s: float | None = None  # first_success_after_failure - first_failure
    reason: str = ""  # explains recovery_time_s=None cases


def _classify_error(exc: Exception) -> tuple[str, str]:
    """Return (outcome_tag, fix_hint) for a raised exception.

    The tag matches the production ``peer_backend_failed`` ``outcome``
    field exactly so existing ``jq`` queries filter the same way on
    bench traces.

    The second element is the human-readable fix hint: for
    ``BackendError`` subclasses it's ``exc.fix_hint`` (the typed
    recovery suggestion); for anything else it's ``str(exc)``. This
    mirrors ``_MaximPeerBackend._log_failure``'s
    ``getattr(exc, "fix_hint", "") or str(exc)`` pattern exactly.
    """

    def _hint(e: Exception) -> str:
        return getattr(e, "fix_hint", "") or str(e)

    if isinstance(exc, BackendOverloaded):
        return "overloaded", _hint(exc)
    if isinstance(exc, BackendAuthFailed):
        return "auth_rejected", _hint(exc)
    if isinstance(exc, BackendModelMissing):
        return "model_missing", _hint(exc)
    if isinstance(exc, BackendInferenceBroken):
        return "inference_broken", _hint(exc)
    if isinstance(exc, BackendTimeout):
        return "timeout", _hint(exc)
    if isinstance(exc, BackendDown):
        return "down", _hint(exc)
    if isinstance(exc, BackendError):
        return "generic_backend_error", _hint(exc)
    return "unhandled_" + type(exc).__name__, str(exc)


def _analyse_recovery(attempts: list[BenchAttempt]) -> tuple[BenchResult, dict[str, Any]]:
    """Walk the attempt list and identify the first recovery transition.

    A recovery transition is:
    1. A run of successes
    2. Followed by at least one failure
    3. Followed by at least one success

    The recovery_time_s is ``first_success_after_failure - first_failure``.

    If the bench never saw a failure → no outage happened → recovery
    is reported as None with a "no_outage_observed" reason.

    If the bench saw failures but no subsequent success → the peer
    didn't recover within the run window → None with
    "did_not_recover" reason.

    If the bench saw only successes and then ended during a failure
    streak → None with "ended_during_outage".
    """
    result_dict: dict[str, Any] = {
        "last_success_before_failure": None,
        "first_failure": None,
        "first_success_after_failure": None,
        "recovery_time_s": None,
        "reason": "",
    }
    # Empty-attempt edge case (Plan 4 B pre-merge review finding #3):
    # distinguish "bench ran but leader never went down" from "bench
    # was killed before its first call". Without this branch the empty
    # list incorrectly labeled as "no_outage_observed" which implies
    # a successful observation window.
    if not attempts:
        result_dict["reason"] = "no_attempts"
        bench = BenchResult(
            duration_s=0.0,
            total_attempts=0,
            successes=0,
            failures=0,
            attempts=[],
            **result_dict,
        )
        return bench, result_dict
    state = "pre_outage"
    for attempt in attempts:
        if state == "pre_outage":
            if attempt.status == "success":
                result_dict["last_success_before_failure"] = attempt.complete_ts
            else:
                if result_dict["last_success_before_failure"] is None:
                    # First attempt in the run was a failure — we have
                    # no pre-outage baseline, so we can't compute
                    # recovery time.
                    result_dict["reason"] = "no_pre_outage_success"
                    state = "done"
                    break
                result_dict["first_failure"] = attempt.complete_ts
                state = "in_outage"
        elif state == "in_outage":
            if attempt.status == "success":
                result_dict["first_success_after_failure"] = attempt.complete_ts
                result_dict["recovery_time_s"] = round(
                    attempt.complete_ts - result_dict["first_failure"],
                    3,
                )
                result_dict["reason"] = "recovered"
                state = "done"
                break

    if state == "pre_outage":
        result_dict["reason"] = "no_outage_observed"
    elif state == "in_outage":
        result_dict["reason"] = "did_not_recover"

    bench = BenchResult(
        duration_s=0.0,  # filled in by caller
        total_attempts=len(attempts),
        successes=sum(1 for a in attempts if a.status == "success"),
        failures=sum(1 for a in attempts if a.status == "failure"),
        attempts=attempts,
        **result_dict,
    )
    return bench, result_dict


def run_recovery_benchmark(
    *,
    url: str,
    api_key: str | None,
    model: str | None = None,
    duration_s: float = 120.0,
    max_tokens: int = 8,
    pace_s: float = 0.0,
    stop_event: threading.Event | None = None,
    backend_factory: Any = None,
) -> BenchResult:
    """Fire chat completions in a tight loop and return a ``BenchResult``.

    Parameters
    ----------
    url
        Peer base URL — e.g. ``https://maxim.dennyschaedig.com/v1``.
    api_key
        Bearer token for the peer. ``None`` only works for localhost
        solo-mode leaders.
    model
        Model name to request; if ``None`` the backend's default (from
        ``/v1/models``) is used.
    duration_s
        Maximum wall-clock duration for the loop. The loop exits early
        if ``stop_event`` is set (Ctrl+C).
    max_tokens
        Cap on generated tokens per call. Kept small so each call
        finishes fast — the benchmark measures the peer path, not
        generation throughput.
    pace_s
        Optional delay between attempts. Default 0.0 means "as fast as
        possible" (true tight loop). Set to a small value (e.g. 0.1)
        if you want to avoid hammering a fragile leader.
    stop_event
        External cancellation Event. If set, the loop exits between
        attempts. Normally wired to a SIGINT handler by the CLI.
    backend_factory
        Test hook: a callable ``(url, api_key, model) -> backend``.
        If omitted, uses ``_MaximPeerBackend.for_url``. Used by unit
        tests to inject a fake backend that returns deterministic
        results without hitting the network.

    Returns
    -------
    BenchResult
        Full attempt log + recovery-time analysis.
    """
    if stop_event is None:
        stop_event = threading.Event()

    if backend_factory is None:
        backend_factory = _MaximPeerBackend.for_url
    backend = backend_factory(url, api_key=api_key, model=model)

    attempts: list[BenchAttempt] = []
    start = time.monotonic()
    log_structured(
        logger,
        logging.INFO,
        event="benchmark_start",
        data={
            "bench": "recovery_time",
            "url": url,
            "model": model,
            "duration_s": duration_s,
            "max_tokens": max_tokens,
            "pace_s": pace_s,
        },
    )

    # Provider key: what the production path sets on
    # ``_MaximPeerBackend._log_success`` data["provider"]. For the bench
    # we use a stable sentinel so operators can filter `jq
    # 'select(.provider=="bench_recovery_time")'`. This lands in the
    # production-shape ``provider`` field, not a bench-specific field.
    bench_provider_key = "bench_recovery_time"

    while not stop_event.is_set():
        now = time.monotonic() - start
        if now >= duration_s:
            break
        req_id = generate_request_id()
        # Plan 4 A.2: bind a fresh RequestContext per call so
        # ``utils.http._build_headers`` populates X-Maxim-* headers on
        # the outbound internal request. One binding per attempt — the
        # pre-merge review round flagged an earlier draft that bound a
        # stable outer context then shadowed it per-call, which was
        # dead code. The per-call binding is the only useful one.
        per_call_ctx = RequestContext(
            request_id=req_id,
            agent_id=BENCH_AGENT_ID,
            session_id=BENCH_AGENT_ID,
            lane="large",
        )
        per_call_token = set_context(per_call_ctx)
        attempt_submit = time.monotonic() - start
        t0 = time.monotonic()
        try:
            resp = backend.complete_with_usage(
                system=BENCH_SYSTEM,
                user=BENCH_USER,
                max_tokens=max_tokens,
                temperature=0.0,
                request_context={
                    "request_id": req_id,
                    "agent_id": BENCH_AGENT_ID,
                    "session_id": BENCH_AGENT_ID,
                    "lane": "large",
                },
            )
            latency_ms = (time.monotonic() - t0) * 1000.0
            # Capture the LLMResponse fields so bench JSONL matches
            # production peer_backend_call shape exactly.
            attempts.append(
                BenchAttempt(
                    request_id=req_id,
                    submit_ts=attempt_submit,
                    complete_ts=time.monotonic() - start,
                    latency_ms=round(latency_ms, 1),
                    status="success",
                    outcome="ok",
                    provider=getattr(resp, "provider", "") or bench_provider_key,
                    model=getattr(resp, "model", "") or (model or ""),
                    input_tokens=int(getattr(resp, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(resp, "output_tokens", 0) or 0),
                )
            )
        except Exception as exc:  # noqa: BLE001 — we classify below
            latency_ms = (time.monotonic() - t0) * 1000.0
            outcome, fix_hint = _classify_error(exc)
            attempts.append(
                BenchAttempt(
                    request_id=req_id,
                    submit_ts=attempt_submit,
                    complete_ts=time.monotonic() - start,
                    latency_ms=round(latency_ms, 1),
                    status="failure",
                    outcome=outcome,
                    provider=bench_provider_key,
                    model=model or "",
                    error=type(exc).__name__,
                    fix_hint=fix_hint[:500],
                    http_status=getattr(exc, "status", None),
                )
            )
        finally:
            reset_context(per_call_token)

        if pace_s > 0.0 and not stop_event.is_set():
            stop_event.wait(pace_s)

    duration = time.monotonic() - start
    bench_result, _ = _analyse_recovery(attempts)
    bench_result.duration_s = round(duration, 3)

    log_structured(
        logger,
        logging.INFO,
        event="benchmark",
        data={
            "bench": "recovery_time",
            "duration_s": bench_result.duration_s,
            "total_attempts": bench_result.total_attempts,
            "successes": bench_result.successes,
            "failures": bench_result.failures,
            "recovery_time_s": bench_result.recovery_time_s,
            "reason": bench_result.reason,
            "first_failure_ts": bench_result.first_failure,
            "first_success_after_failure_ts": bench_result.first_success_after_failure,
            "last_success_before_failure_ts": bench_result.last_success_before_failure,
        },
    )
    return bench_result


def install_sigint_handler(stop_event: threading.Event) -> Any:
    """Install a SIGINT handler that sets ``stop_event``.

    Returns the previous handler so the CLI can restore it on exit.
    """

    def _handler(signum: int, frame: Any) -> None:
        stop_event.set()

    return signal.signal(signal.SIGINT, _handler)
