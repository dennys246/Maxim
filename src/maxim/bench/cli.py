"""CLI dispatcher for ``maxim bench <subcommand>``.

Follows the same pattern as ``maxim doctor`` and ``maxim tunnel``:
positional subcommand after ``bench``, argparse-driven arg parsing
inside the handler, returns an int exit code. See
:mod:`maxim.bench.recovery_time` for the actual benchmark logic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from typing import Sequence

from maxim.bench.recovery_time import (
    BenchResult,
    install_sigint_handler,
    run_recovery_benchmark,
)

logger = logging.getLogger("maxim.bench.cli")


def _print_summary(result: BenchResult) -> None:
    """Human-readable summary written to stderr so stdout stays JSONL-clean."""
    print("─── maxim bench recovery-time ───", file=sys.stderr)
    print(f"  duration:         {result.duration_s:.1f}s", file=sys.stderr)
    print(f"  total attempts:   {result.total_attempts}", file=sys.stderr)
    print(f"  successes:        {result.successes}", file=sys.stderr)
    print(f"  failures:         {result.failures}", file=sys.stderr)
    if result.recovery_time_s is not None:
        print(
            f"  recovery time:    {result.recovery_time_s:.2f}s  ({result.reason})",
            file=sys.stderr,
        )
    else:
        print(f"  recovery time:    N/A  ({result.reason})", file=sys.stderr)


def _result_to_jsonl(result: BenchResult) -> list[dict]:
    """Serialize a BenchResult as a list of JSONL-ready dicts.

    Emits one ``peer_backend_call``/``peer_backend_failed`` event per
    attempt (matching production shape so existing ``jq`` queries work)
    plus one aggregate ``benchmark`` summary event at the end.

    **Wire-compat invariant** (Plan 4 B pre-merge review finding #2):
    every field that ``_MaximPeerBackend._log_success`` /
    ``_log_failure`` emit must appear here with the same name and type
    so existing ``jq 'select(.e=="peer_backend_call") | .provider'``
    queries work unchanged on bench traces. The bench-specific tag
    lives in a ``bench`` field; the ``submit_ts`` / ``complete_ts``
    fields are bench-only timing and do not collide with any
    production field.
    """
    # Plan 4 follow-up (2026-04-14): session_id is now a distinct
    # per-run value (e.g., "bench_20260414_152300") sourced from
    # ``BenchResult.session_id``. Older traces that reused
    # ``BENCH_AGENT_ID`` as the session_id are still readable — just
    # filter on ``agent_id == "bench_recovery_time"`` which remains
    # stable. Empty-string session_id (test-fixture path) degrades to
    # ``BENCH_AGENT_ID`` for backwards compat with existing regression
    # guards that constructed BenchResult directly without going
    # through ``run_recovery_benchmark``.
    session_id = result.session_id or "bench_recovery_time"

    events: list[dict] = []
    for attempt in result.attempts:
        if attempt.status == "success":
            events.append(
                {
                    "e": "peer_backend_call",
                    "bench": "recovery_time",
                    # Production shape (from _MaximPeerBackend._log_success)
                    "provider": attempt.provider,
                    "model": attempt.model,
                    "status": 200,
                    "latency_ms": attempt.latency_ms,
                    "input_tokens": attempt.input_tokens,
                    "output_tokens": attempt.output_tokens,
                    "request_id": attempt.request_id,
                    "agent_id": "bench_recovery_time",
                    "session_id": session_id,
                    "lane": "large",
                    # Bench-specific timing (not in production)
                    "submit_ts": attempt.submit_ts,
                    "complete_ts": attempt.complete_ts,
                }
            )
        else:
            events.append(
                {
                    "e": "peer_backend_failed",
                    "bench": "recovery_time",
                    # Production shape (from _MaximPeerBackend._log_failure)
                    "provider": attempt.provider,
                    "error": attempt.error,
                    "outcome": attempt.outcome,
                    "status": attempt.http_status,
                    "fix_hint": attempt.fix_hint,
                    "latency_ms": attempt.latency_ms,
                    "request_id": attempt.request_id,
                    "agent_id": "bench_recovery_time",
                    "session_id": session_id,
                    "lane": "large",
                    # Bench-specific timing (not in production)
                    "submit_ts": attempt.submit_ts,
                    "complete_ts": attempt.complete_ts,
                }
            )
    events.append(
        {
            "e": "benchmark",
            "bench": "recovery_time",
            "session_id": session_id,
            "duration_s": result.duration_s,
            "total_attempts": result.total_attempts,
            "successes": result.successes,
            "failures": result.failures,
            "recovery_time_s": result.recovery_time_s,
            "reason": result.reason,
            "last_success_before_failure_ts": result.last_success_before_failure,
            "first_failure_ts": result.first_failure,
            "first_success_after_failure_ts": result.first_success_after_failure,
        }
    )
    return events


def _run_recovery_time(argv: Sequence[str]) -> int:
    """Entry point for ``maxim bench recovery-time``."""
    parser = argparse.ArgumentParser(
        prog="maxim bench recovery-time",
        description=(
            "Fire chat completions in a tight loop against a peer URL "
            "to measure leader-restart recovery time. Run this, then "
            "trigger `maxim peer restart` in a second terminal; the "
            "bench will report the gap between the last pre-restart "
            "success and the first post-restart success as a single "
            "clean number."
        ),
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Peer base URL (e.g. https://maxim.example.com/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token for the peer (optional for localhost leaders)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to request (defaults to the peer's active model)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Maximum wall-clock duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Max tokens per call — kept low for tight-loop timing (default: 8)",
    )
    parser.add_argument(
        "--pace",
        type=float,
        default=0.0,
        help=(
            "Optional delay between attempts in seconds. Default 0.0 "
            "means 'as fast as possible'; set to 0.1 to avoid hammering "
            "a fragile leader."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Write per-attempt + summary events as JSONL to this path. "
            "If omitted, JSONL is written to stdout. The shape matches "
            "production peer_backend_call/peer_backend_failed events so "
            "existing jq queries work."
        ),
    )
    args = parser.parse_args(list(argv))

    stop_event = threading.Event()
    prev_handler = install_sigint_handler(stop_event)
    try:
        result = run_recovery_benchmark(
            url=args.url,
            api_key=args.api_key,
            model=args.model,
            duration_s=args.duration,
            max_tokens=args.max_tokens,
            pace_s=args.pace,
            stop_event=stop_event,
        )
    finally:
        import signal as _signal

        _signal.signal(_signal.SIGINT, prev_handler)

    events = _result_to_jsonl(result)
    if args.output:
        with open(args.output, "w") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")
    else:
        for ev in events:
            print(json.dumps(ev))

    _print_summary(result)
    return 0


def run_bench_subcommand(argv: Sequence[str]) -> int:
    """Dispatch ``maxim bench <subcommand> [args]``."""
    if not argv:
        print(
            "usage: maxim bench <subcommand> [args]\n\n"
            "subcommands:\n"
            "  recovery-time   tight-loop recovery-time benchmark\n",
            file=sys.stderr,
        )
        return 2

    subcommand, *rest = argv
    if subcommand in ("recovery-time", "recovery_time"):
        return _run_recovery_time(rest)
    print(
        f"unknown bench subcommand: {subcommand!r}\nknown subcommands: recovery-time",
        file=sys.stderr,
    )
    return 2
