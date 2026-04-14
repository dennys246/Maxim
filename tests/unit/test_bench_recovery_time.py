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

    def test_every_backend_error_subclass_has_specific_mapping(self):
        """**Plan 4 B pre-merge review finding N6.** Regression guard:
        every concrete subclass of ``BackendError`` must map to a
        named outcome tag (NOT ``unhandled_*``). If a new subclass is
        added to ``models/language/types.py`` without updating
        ``_classify_error``, it silently falls through to
        ``unhandled_NewSubclass``, which contributes a misleading
        recovery signal to the bench output (the failure would be
        classified as a bug-in-the-harness rather than a legitimate
        typed backend failure).

        The test walks ``BackendError.__subclasses__()`` transitively
        and asserts each concrete subclass is mapped to a named
        outcome. New subclasses require a one-line update to
        ``_classify_error`` AND a test in this class — but this
        regression guard is the backstop.
        """

        def _all_subclasses(cls: type) -> list[type]:
            result: list[type] = []
            for sub in cls.__subclasses__():
                result.append(sub)
                result.extend(_all_subclasses(sub))
            return result

        all_subs = _all_subclasses(BackendError)
        # Must be non-empty — sanity check that the taxonomy isn't empty
        # due to an import order issue (the subclasses need to have been
        # defined at module-load time).
        assert len(all_subs) >= 6, (
            f"BackendError has only {len(all_subs)} subclasses — did "
            "a typo / import issue hide them from __subclasses__?"
        )

        for subcls in all_subs:
            # Construct a minimal instance. All BackendError subclasses
            # take at least `provider_key` as their first positional.
            try:
                exc = subcls("p")
            except TypeError:
                # Some subclasses have additional required kwargs
                # (e.g., BackendAuthFailed(status=...)). Try a few.
                try:
                    exc = subcls("p", status=500)  # type: ignore[call-arg]
                except TypeError:
                    try:
                        exc = subcls("p", elapsed_s=1.0)  # type: ignore[call-arg]
                    except TypeError:
                        try:
                            exc = subcls("p", retry_after_s=1.0)  # type: ignore[call-arg]
                        except TypeError:
                            pytest.fail(
                                f"Could not construct {subcls.__name__} with any known "
                                "kwarg combination — update this test if the signature "
                                "changed"
                            )
            outcome, _ = _classify_error(exc)
            assert not outcome.startswith("unhandled_"), (
                f"{subcls.__name__} falls through to generic unhandled_* in "
                f"_classify_error — add a dedicated branch. Current outcome: {outcome!r}"
            )


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

    def test_empty_attempts_returns_no_attempts_reason(self):
        """Plan 4 B pre-merge review finding #3: an empty attempt list
        must NOT be conflated with 'no outage observed' — the bench
        may have been SIGINT'd before firing its first call, and
        labeling that case as a successful observation window is
        semantically wrong. Dedicated ``no_attempts`` reason makes the
        distinction explicit."""
        result, _ = _analyse_recovery([])
        assert result.recovery_time_s is None
        assert result.reason == "no_attempts"
        assert result.total_attempts == 0
        assert result.successes == 0
        assert result.failures == 0

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
        a sleep or pace in the hot path.

        **Plan 4 B pre-merge review finding #4:** assert the per-second
        RATE rather than an absolute count, so a future hot-path
        regression that adds e.g. ``time.sleep(0.025)`` per call fails
        this loudly instead of sliding under an absolute threshold
        like ``>= 10``. The threshold below (>= 40 calls/sec) is
        comfortably above any realistic slow-path regression and
        comfortably below the MagicMock's achievable rate (>1000/sec).
        """
        factory, call_count = _fake_backend_factory([_ok_response()])
        result = run_recovery_benchmark(
            url="http://fake/v1",
            api_key="k",
            duration_s=0.25,  # quarter-second window
            backend_factory=factory,
        )
        # Rate-based assertion (finding #4)
        observed_rate = result.successes / max(result.duration_s, 0.001)
        assert observed_rate >= 40, (
            f"tight loop rate regression: {result.successes} calls in "
            f"{result.duration_s:.3f}s = {observed_rate:.1f}/sec — "
            "the hot path probably grew a sleep or blocking op"
        )
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

    def test_bench_run_has_distinct_per_run_session_id(self):
        """Plan 4 follow-up (2026-04-14): the bench's ``session_id``
        must be a distinct per-run value (not reuse ``BENCH_AGENT_ID``).
        Two back-to-back bench runs should be distinguishable by
        session_id even though they share the agent_id. The initial
        Plan 4 B ship reused agent_id as session_id which was
        semantically wrong per the pre-merge review.
        """
        captured: list[Any] = []

        def _capture_factory(url, api_key=None, model=None):
            backend = MagicMock()

            def _complete_with_usage(**kwargs):
                captured.append(kwargs.get("request_context"))
                return _ok_response()

            backend.complete_with_usage = _complete_with_usage
            return backend

        result = run_recovery_benchmark(
            url="http://fake/v1",
            api_key="k",
            duration_s=0.1,
            backend_factory=_capture_factory,
        )
        # Per-run session_id gets set on the BenchResult
        assert result.session_id != ""
        assert result.session_id != BENCH_AGENT_ID
        assert result.session_id.startswith("bench_")

        # Every attempt's request_context dict carries the same
        # per-run value
        assert len(captured) >= 1
        for ctx_dict in captured:
            assert ctx_dict.get("session_id") == result.session_id
            # And it is NOT the agent_id — the two must be distinct
            assert ctx_dict.get("session_id") != ctx_dict.get("agent_id")

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
        """Plan 4 B pre-merge review finding #2: bench JSONL must carry
        every field the production ``_log_success`` / ``_log_failure``
        emit so existing ``jq 'select(.e=="peer_backend_call") |
        .input_tokens'`` queries work unchanged."""
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
                    provider="bench_recovery_time",
                    model="qwen2.5-14b-instruct",
                    input_tokens=12,
                    output_tokens=3,
                ),
                BenchAttempt(
                    request_id="r2",
                    submit_ts=0.2,
                    complete_ts=0.3,
                    latency_ms=100.0,
                    status="failure",
                    outcome="down",
                    provider="bench_recovery_time",
                    model="qwen2.5-14b-instruct",
                    error="BackendDown",
                    fix_hint="Cloudflare 502",
                    http_status=502,
                ),
            ],
            recovery_time_s=None,
            reason="no_outage_observed",
        )
        events = _result_to_jsonl(result)
        # Two attempts + one summary event
        assert len(events) == 3
        # Success event: production peer_backend_call wire-compat
        s = events[0]
        assert s["e"] == "peer_backend_call"
        assert s["bench"] == "recovery_time"
        assert s["request_id"] == "r1"
        assert s["agent_id"] == BENCH_AGENT_ID
        assert s["status"] == 200
        # The critical wire-compat fields — if any of these are missing,
        # existing `jq` queries on production traces break on bench output
        assert s["provider"] == "bench_recovery_time"
        assert s["model"] == "qwen2.5-14b-instruct"
        assert s["input_tokens"] == 12
        assert s["output_tokens"] == 3
        assert s["latency_ms"] == 100.0
        # Session_id fallback: when BenchResult.session_id is unset
        # (this test fixture path), the CLI falls back to BENCH_AGENT_ID
        # for backward compat. The Plan 4 follow-up regression guard
        # that the CLI RESPECTS a set session_id lives in
        # test_cli_respects_per_run_session_id below.
        assert s["session_id"] == BENCH_AGENT_ID
        assert s["lane"] == "large"
        # Failure event: production peer_backend_failed wire-compat
        f = events[1]
        assert f["e"] == "peer_backend_failed"
        assert f["outcome"] == "down"
        assert f["provider"] == "bench_recovery_time"
        assert f["error"] == "BackendDown"  # exception class name (production: type(exc).__name__)
        assert f["fix_hint"] == "Cloudflare 502"  # human-readable (production: exc.fix_hint)
        assert f["status"] == 502  # HTTP status from exception
        assert f["latency_ms"] == 100.0
        # Summary event
        assert events[2]["e"] == "benchmark"
        assert events[2]["bench"] == "recovery_time"
        assert events[2]["recovery_time_s"] is None
        assert events[2]["reason"] == "no_outage_observed"

    def test_success_jsonl_field_parity_with_production_log_success(self):
        """Locks the bench success JSONL shape against every field the
        production ``_MaximPeerBackend._log_success`` emits. Regression
        guard: if production adds a new field, this test fails loudly
        and forces the bench to be updated in lockstep.
        """
        from maxim.bench.cli import _result_to_jsonl
        from maxim.bench.recovery_time import BenchAttempt, BenchResult

        # These field names are copied verbatim from
        # src/maxim/models/language/maxim_peer_backend.py::_log_success
        # at the time of Plan 4 B. Keep in sync.
        PRODUCTION_SUCCESS_FIELDS = {
            "provider",
            "model",
            "status",
            "latency_ms",
            "input_tokens",
            "output_tokens",
            "request_id",
            "agent_id",
            "session_id",
            "lane",
        }
        result = BenchResult(
            duration_s=1.0,
            total_attempts=1,
            successes=1,
            failures=0,
            attempts=[
                BenchAttempt(
                    request_id="r",
                    submit_ts=0.0,
                    complete_ts=0.1,
                    latency_ms=100.0,
                    status="success",
                    outcome="ok",
                    provider="p",
                    model="m",
                    input_tokens=1,
                    output_tokens=1,
                )
            ],
            reason="no_outage_observed",
        )
        event = _result_to_jsonl(result)[0]
        missing = PRODUCTION_SUCCESS_FIELDS - set(event.keys())
        assert not missing, f"bench success JSONL missing production fields: {missing}"

    def test_failure_jsonl_field_parity_with_production_log_failure(self):
        """Locks the bench failure JSONL shape against every field the
        production ``_MaximPeerBackend._log_failure`` emits."""
        from maxim.bench.cli import _result_to_jsonl
        from maxim.bench.recovery_time import BenchAttempt, BenchResult

        PRODUCTION_FAILURE_FIELDS = {
            "provider",
            "error",
            "outcome",
            "status",
            "fix_hint",
            "latency_ms",
            "request_id",
            "agent_id",
            "session_id",
            "lane",
        }
        result = BenchResult(
            duration_s=1.0,
            total_attempts=1,
            successes=0,
            failures=1,
            attempts=[
                BenchAttempt(
                    request_id="r",
                    submit_ts=0.0,
                    complete_ts=0.1,
                    latency_ms=100.0,
                    status="failure",
                    outcome="down",
                    provider="p",
                    error="BackendDown",
                    fix_hint="down",
                    http_status=502,
                )
            ],
            reason="did_not_recover",
        )
        event = _result_to_jsonl(result)[0]
        missing = PRODUCTION_FAILURE_FIELDS - set(event.keys())
        assert not missing, f"bench failure JSONL missing production fields: {missing}"

    def test_cli_respects_per_run_session_id(self):
        """Plan 4 follow-up (2026-04-14): when ``BenchResult.session_id``
        is set (the real run_recovery_benchmark path), the CLI JSONL
        emitter must use that value instead of the legacy hard-coded
        ``"bench_recovery_time"`` fallback. Back-to-back runs produce
        distinct JSONL session_id values so they can be filtered apart.
        """
        from maxim.bench.cli import _result_to_jsonl
        from maxim.bench.recovery_time import BenchAttempt, BenchResult

        result = BenchResult(
            duration_s=1.0,
            total_attempts=1,
            successes=1,
            failures=0,
            attempts=[
                BenchAttempt(
                    request_id="r1",
                    submit_ts=0.0,
                    complete_ts=0.1,
                    latency_ms=100.0,
                    status="success",
                    outcome="ok",
                    provider="bench_recovery_time",
                    model="qwen",
                    input_tokens=1,
                    output_tokens=1,
                )
            ],
            session_id="bench_20260414_152300",
            reason="no_outage_observed",
        )
        events = _result_to_jsonl(result)
        # Attempt event carries the per-run session_id
        assert events[0]["session_id"] == "bench_20260414_152300"
        # Summary event also carries it
        summary = events[-1]
        assert summary["e"] == "benchmark"
        assert summary["session_id"] == "bench_20260414_152300"
        # Agent_id stays stable (the distinguishing dimension)
        assert events[0]["agent_id"] == BENCH_AGENT_ID
        assert events[0]["session_id"] != events[0]["agent_id"]

    def test_unknown_subcommand_returns_2(self):
        from maxim.bench.cli import run_bench_subcommand

        rc = run_bench_subcommand(["does-not-exist"])
        assert rc == 2

    def test_empty_argv_prints_usage_and_returns_2(self):
        from maxim.bench.cli import run_bench_subcommand

        rc = run_bench_subcommand([])
        assert rc == 2
