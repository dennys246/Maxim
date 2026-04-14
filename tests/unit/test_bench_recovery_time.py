"""Unit tests for the Plan 4 B recovery-time benchmark harness.

Covers:
- Tight-loop correctness under a mocked backend that simulates a
  leader outage window
- Recovery-time analysis produces a single clean number
- Edge cases: no outage, never recovers, outage on first call
- CLI argument parsing + JSONL output shape
- SIGINT stop_event cooperates cleanly

All tests are offline — no real peer, no real HTTP, no real
_MaximPeerBackend. The ``backend_factory`` test hook injects a fake
backend that returns deterministic results.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from maxim.bench.recovery_time import (
    BENCH_AGENT_ID,
    BenchAttempt,
    _analyse_recovery,
    _classify_error,
    run_recovery_benchmark,
)
from maxim.models.language.types import (
    BackendAuthFailed,
    BackendDown,
    BackendError,
    BackendInferenceBroken,
    BackendModelMissing,
    BackendOverloaded,
    BackendTimeout,
    LLMResponse,
)


def _ok_response() -> LLMResponse:
    return LLMResponse(
        content="hi",
        provider="fake-peer",
        model="fake-model",
        input_tokens=5,
        output_tokens=1,
        latency_ms=10.0,
    )


def _fake_backend_factory(responses: list):
    """Build a backend_factory that returns a fake backend whose
    complete_with_usage pops from ``responses`` in order.

    Each entry in ``responses`` is either an ``LLMResponse`` (success)
    or an ``Exception`` instance (failure). If the list runs out, the
    backend keeps returning the last success value.
    """
    call_count = {"n": 0}

    def _factory(url: str, api_key: str | None = None, model: str | None = None):
        backend = MagicMock()

        def _complete_with_usage(**kwargs):
            idx = min(call_count["n"], len(responses) - 1)
            call_count["n"] += 1
            result = responses[idx]
            if isinstance(result, BaseException):
                raise result
            return result

        backend.complete_with_usage = _complete_with_usage
        return backend

    return _factory, call_count


class TestClassifyError:
    def test_overloaded(self):
        exc = BackendOverloaded("p", retry_after_s=5.0)
        assert _classify_error(exc)[0] == "overloaded"

    def test_auth_failed(self):
        assert _classify_error(BackendAuthFailed("p", status=401))[0] == "auth_rejected"

    def test_model_missing(self):
        assert _classify_error(BackendModelMissing("p"))[0] == "model_missing"

    def test_inference_broken(self):
        assert _classify_error(BackendInferenceBroken("p"))[0] == "inference_broken"

    def test_timeout(self):
        assert _classify_error(BackendTimeout("p", elapsed_s=30.0))[0] == "timeout"

    def test_down(self):
        assert _classify_error(BackendDown("p"))[0] == "down"

    def test_generic_backend_error(self):
        assert _classify_error(BackendError("p"))[0] == "generic_backend_error"

    def test_unhandled_exception(self):
        outcome, _ = _classify_error(RuntimeError("something weird"))
        assert outcome.startswith("unhandled_")


class TestAnalyseRecovery:
    """The analysis walks the attempt list and finds the first
    success→failure→success transition. These tests lock in the
    precedence rules."""

    def _attempt(self, ts: float, status: str, req_id: str = "r") -> BenchAttempt:
        return BenchAttempt(
            request_id=req_id,
            submit_ts=ts - 0.01,
            complete_ts=ts,
            latency_ms=10.0,
            status=status,
            outcome="ok" if status == "success" else "down",
        )

    def test_simple_recovery(self):
        attempts = [
            self._attempt(1.0, "success"),
            self._attempt(2.0, "success"),
            self._attempt(3.0, "failure"),
            self._attempt(4.0, "failure"),
            self._attempt(5.0, "success"),
        ]
        result, _ = _analyse_recovery(attempts)
        assert result.reason == "recovered"
        assert result.last_success_before_failure == 2.0
        assert result.first_failure == 3.0
        assert result.first_success_after_failure == 5.0
        # recovery_time = first_success - first_failure
        assert result.recovery_time_s == pytest.approx(2.0)

    def test_no_outage_observed_returns_none(self):
        attempts = [
            self._attempt(1.0, "success"),
            self._attempt(2.0, "success"),
            self._attempt(3.0, "success"),
        ]
        result, _ = _analyse_recovery(attempts)
        assert result.recovery_time_s is None
        assert result.reason == "no_outage_observed"
        assert result.successes == 3
        assert result.failures == 0

    def test_did_not_recover_returns_none(self):
        attempts = [
            self._attempt(1.0, "success"),
            self._attempt(2.0, "failure"),
            self._attempt(3.0, "failure"),
            self._attempt(4.0, "failure"),
        ]
        result, _ = _analyse_recovery(attempts)
        assert result.recovery_time_s is None
        assert result.reason == "did_not_recover"

    def test_no_pre_outage_success(self):
        """If the first attempt is a failure, there's no pre-outage
        baseline and recovery_time is not meaningful."""
        attempts = [
            self._attempt(1.0, "failure"),
            self._attempt(2.0, "success"),
        ]
        result, _ = _analyse_recovery(attempts)
        assert result.recovery_time_s is None
        assert result.reason == "no_pre_outage_success"

    def test_recovery_time_computed_from_first_failure_not_last(self):
        """Critical precision guard: recovery_time must use the FIRST
        failure timestamp as the denominator, not the last failure
        before success. The user needs to know 'how long after the
        peer went down did it come back', not 'how long after the
        last failed probe'."""
        attempts = [
            self._attempt(1.0, "success"),
            self._attempt(5.0, "failure"),  # first failure — outage start
            self._attempt(7.0, "failure"),
            self._attempt(9.0, "failure"),
            self._attempt(11.0, "success"),  # recovery
        ]
        result, _ = _analyse_recovery(attempts)
        assert result.first_failure == 5.0
        assert result.recovery_time_s == pytest.approx(6.0)  # 11 - 5, not 11 - 9


class TestRunRecoveryBenchmark:
    """End-to-end tests with a fake backend factory. No real HTTP."""

    def test_tight_loop_fires_many_calls_in_short_duration(self):
        """A tight loop with zero-latency fakes should fire far more
        than 1 call/sec. Regression guard against accidentally adding
        a sleep or pace in the hot path."""
        factory, call_count = _fake_backend_factory([_ok_response()])
        result = run_recovery_benchmark(
            url="http://fake/v1",
            api_key="k",
            duration_s=0.25,  # quarter-second window
            backend_factory=factory,
        )
        assert result.successes >= 10, f"only {result.successes} calls in 0.25s — is the loop paced?"
        assert result.failures == 0
        assert result.recovery_time_s is None
        assert result.reason == "no_outage_observed"

    def test_recovery_transition_measured_correctly(self):
        """Inject [success, success, failure, failure, success] and
        assert the analysis extracts a recovery_time_s."""
        responses: list[Any] = [
            _ok_response(),
            _ok_response(),
            BackendDown("fake-peer"),
            BackendDown("fake-peer"),
            _ok_response(),
            _ok_response(),
        ]
        factory, call_count = _fake_backend_factory(responses)
        result = run_recovery_benchmark(
            url="http://fake/v1",
            api_key="k",
            duration_s=2.0,
            pace_s=0.01,  # small pace to ensure monotonic ordering
            backend_factory=factory,
        )
        assert result.successes >= 3
        assert result.failures >= 2
        assert result.reason == "recovered"
        assert result.recovery_time_s is not None
        assert result.recovery_time_s > 0

    def test_stop_event_exits_early(self):
        """SIGINT → stop_event.set() → loop exits between attempts.
        Regression guard for cooperative cancellation."""
        stop = threading.Event()
        responses: list[Any] = [_ok_response()] * 100

        factory, call_count = _fake_backend_factory(responses)
        # Set stop_event immediately; the loop should exit after
        # at most one attempt.
        stop.set()
        result = run_recovery_benchmark(
            url="http://fake/v1",
            api_key="k",
            duration_s=60.0,
            stop_event=stop,
            backend_factory=factory,
        )
        assert result.total_attempts <= 1, f"stop_event did not exit the loop: {result.total_attempts} attempts"

    def test_bench_context_has_stable_agent_id(self):
        """All attempts must share BENCH_AGENT_ID so operators can
        filter the JSONL log by agent_id=bench_recovery_time."""
        captured: list[Any] = []

        def _capture_factory(url, api_key=None, model=None):
            backend = MagicMock()

            def _complete_with_usage(**kwargs):
                captured.append(kwargs.get("request_context"))
                return _ok_response()

            backend.complete_with_usage = _complete_with_usage
            return backend

        run_recovery_benchmark(
            url="http://fake/v1",
            api_key="k",
            duration_s=0.1,
            backend_factory=_capture_factory,
        )
        assert len(captured) >= 1
        for ctx_dict in captured:
            assert ctx_dict is not None
            assert ctx_dict.get("agent_id") == BENCH_AGENT_ID

    def test_contextvar_cleaned_up_after_run(self):
        """Regression guard: the bench binds set_context and must
        reset the binding so the caller's contextvar state is not
        polluted."""
        from maxim.utils.http import current_context

        assert current_context() is None, "precondition: contextvar must be empty"

        factory, _ = _fake_backend_factory([_ok_response()])
        run_recovery_benchmark(
            url="http://fake/v1",
            api_key="k",
            duration_s=0.1,
            backend_factory=factory,
        )
        assert current_context() is None, "bench leaked the contextvar binding into the caller"


class TestBenchCliOutput:
    """Tests for the CLI's JSONL output shape + dispatch."""

    def test_result_to_jsonl_emits_peer_backend_call_shape(self):
        from maxim.bench.cli import _result_to_jsonl
        from maxim.bench.recovery_time import BenchAttempt, BenchResult

        result = BenchResult(
            duration_s=1.0,
            total_attempts=2,
            successes=1,
            failures=1,
            attempts=[
                BenchAttempt(
                    request_id="r1",
                    submit_ts=0.0,
                    complete_ts=0.1,
                    latency_ms=100.0,
                    status="success",
                    outcome="ok",
                ),
                BenchAttempt(
                    request_id="r2",
                    submit_ts=0.2,
                    complete_ts=0.3,
                    latency_ms=100.0,
                    status="failure",
                    outcome="down",
                    error_message="Cloudflare 502",
                ),
            ],
            recovery_time_s=None,
            reason="no_outage_observed",
        )
        events = _result_to_jsonl(result)
        # Two attempts + one summary event
        assert len(events) == 3
        # Event shape matches production peer_backend_call
        assert events[0]["e"] == "peer_backend_call"
        assert events[0]["request_id"] == "r1"
        assert events[0]["agent_id"] == BENCH_AGENT_ID
        assert events[0]["status"] == 200
        # Failure event matches production peer_backend_failed
        assert events[1]["e"] == "peer_backend_failed"
        assert events[1]["outcome"] == "down"
        assert events[1]["error"] == "Cloudflare 502"
        # Summary event
        assert events[2]["e"] == "benchmark"
        assert events[2]["bench"] == "recovery_time"
        assert events[2]["recovery_time_s"] is None
        assert events[2]["reason"] == "no_outage_observed"

    def test_unknown_subcommand_returns_2(self):
        from maxim.bench.cli import run_bench_subcommand

        rc = run_bench_subcommand(["does-not-exist"])
        assert rc == 2

    def test_empty_argv_prints_usage_and_returns_2(self):
        from maxim.bench.cli import run_bench_subcommand

        rc = run_bench_subcommand([])
        assert rc == 2
