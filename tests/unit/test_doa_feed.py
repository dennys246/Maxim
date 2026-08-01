"""Stage-2 regression guards: the live DoA → azimuth feed.

live_audio_orient_wiring.md Stage 2 — a poll thread (CaptureManager
pattern) that caches the latest speech-gated azimuth and world-sets the
body sensor. Pins:

- fresh gated reading → clamped write into the REAL reachy_mini body;
- silence → NO write (a stale direction is never re-written);
- ``drift_rate: 0`` keeps the world-set value across ``evaluate_failures``
  (no fabricated "centered" between re-measurements);
- teardown: the thread joins promptly on ``stop_event`` even mid-sampling
  (the interruptible ``gated_azimuth`` contract);
- the ``_maybe_start_doa_feed`` triple gate (reader present / azimuth
  sensor declared / robots.yaml opt-out).
"""

from __future__ import annotations

import logging
import math
import threading
import time


from maxim.embodiment.audio_localization import DoAFeed
from maxim.embodiment.body import Embodiment
from maxim.embodiment.component_registry import ComponentRegistry

_SPEECH_LEFT = (0.0, True)  # DoA 0 rad = full left → azimuth -1.0


def _reachy_embodiment():
    body = ComponentRegistry().instantiate("bodies/reachy_mini")
    return body, Embodiment(body)


class _ScriptedReader:
    """Yields a fixed burst of readings, then silence; optionally sets a
    stop event once the burst is exhausted so single-threaded tests exit."""

    def __init__(self, burst, stop_event=None):
        self._burst = list(burst)
        self._stop_event = stop_event
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self._burst:
            return self._burst.pop(0)
        if self._stop_event is not None:
            self._stop_event.set()
        return None


class TestDoAFeedWrites:
    def test_fresh_gated_reading_world_sets_the_real_body(self):
        body, emb = _reachy_embodiment()
        stop = threading.Event()
        reader = _ScriptedReader([_SPEECH_LEFT] * 3, stop_event=stop)
        feed = DoAFeed(reader, emb, stop_event=stop, sample_poll_s=0.0, sample_timeout_s=0.5)
        feed.run()  # single-threaded: burst → write, silence → stop → return
        assert body.vital_metrics["azimuth"] == -1.0
        assert feed.latest is not None
        az, ts = feed.latest
        assert az == -1.0
        assert ts > 0

    def test_out_of_convention_doa_angle_cannot_escape_the_range(self):
        # An angle below the XVF3800's 0..π convention is clamped by
        # doa_to_azimuth BEFORE the write, so the sensor can never leave
        # [-1, 1]. (The world_set clamp itself is pinned separately by
        # TestWorldSetAxis in test_audio_localization.py — this test covers
        # the feed-path composition, not the clamp in isolation.)
        body, emb = _reachy_embodiment()
        stop = threading.Event()
        reader = _ScriptedReader([(-1.0, True)] * 3, stop_event=stop)  # < 0 rad → below -1 pre-clamp
        feed = DoAFeed(reader, emb, stop_event=stop, sample_poll_s=0.0, sample_timeout_s=0.5)
        feed.run()
        assert body.vital_metrics["azimuth"] == -1.0

    def test_silence_never_writes(self):
        body, emb = _reachy_embodiment()
        body.vital_metrics["azimuth"] = 0.123  # sentinel
        stop = threading.Event()
        reader = _ScriptedReader([(0.0, False)] * 5, stop_event=stop)  # non-speech only
        feed = DoAFeed(reader, emb, stop_event=stop, sample_poll_s=0.0, sample_timeout_s=0.2)
        feed.run()
        assert body.vital_metrics["azimuth"] == 0.123
        assert feed.latest is None

    def test_world_set_value_survives_evaluate_failures(self):
        """drift_rate: 0 — the tick must not fabricate 'centered' between
        re-measurements (the reachy_mini.yaml load-bearing note)."""
        body, emb = _reachy_embodiment()
        stop = threading.Event()
        reader = _ScriptedReader([(math.pi, True)] * 3, stop_event=stop)  # right → +1.0
        feed = DoAFeed(reader, emb, stop_event=stop, sample_poll_s=0.0, sample_timeout_s=0.5)
        feed.run()
        assert body.vital_metrics["azimuth"] == 1.0
        emb.evaluate_failures()  # first call sets drift baseline
        time.sleep(0.05)
        emb.evaluate_failures()  # elapsed wall-clock drift applies
        assert body.vital_metrics["azimuth"] == 1.0

    def test_percept_sink_receives_fresh_readings_only(self):
        _, emb = _reachy_embodiment()
        stop = threading.Event()
        seen = []
        reader = _ScriptedReader([_SPEECH_LEFT] * 3 + [(0.0, False)] * 3, stop_event=stop)
        feed = DoAFeed(
            reader,
            emb,
            stop_event=stop,
            percept_sink=seen.append,
            agent_id="reachy-test",
            sample_poll_s=0.0,
            sample_timeout_s=0.2,
        )
        feed.run()
        assert len(seen) == 1  # one gated batch → one percept; silence → none
        assert seen[0].metadata["azimuth"] == -1.0
        assert seen[0].context.agent_id == "reachy-test"

    def test_sink_exception_does_not_kill_the_feed(self):
        body, emb = _reachy_embodiment()
        stop = threading.Event()

        def _boom(_p):
            raise RuntimeError("consumer bug")

        reader = _ScriptedReader([_SPEECH_LEFT] * 3, stop_event=stop)
        feed = DoAFeed(reader, emb, stop_event=stop, percept_sink=_boom, sample_poll_s=0.0, sample_timeout_s=0.5)
        feed.run()  # must return normally, not raise
        assert body.vital_metrics["azimuth"] == -1.0


class TestDoAFeedTeardown:
    def test_thread_joins_promptly_on_stop_event_mid_sampling(self):
        """The interruptible gated_azimuth contract: even with a LONG sample
        window (5 s), setting stop_event unblocks the thread fast."""
        _, emb = _reachy_embodiment()
        stop = threading.Event()
        feed = DoAFeed(lambda: None, emb, stop_event=stop, sample_timeout_s=5.0, sample_poll_s=0.15)
        thread = threading.Thread(target=feed.run, daemon=True)
        thread.start()
        time.sleep(0.1)  # let it park inside gated_azimuth
        t0 = time.monotonic()
        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "feed thread did not exit on stop_event"
        assert time.monotonic() - t0 < 1.0, "join took too long — stop_event not honored mid-sample"


class _FakeExecutor:
    def __init__(self, embodiment):
        self.embodiment = embodiment


class _FakeRobot:
    def __init__(self, reader):
        self._reader = reader

    def get_doa_reader(self):
        return self._reader


class _FakeRobotConfig:
    def __init__(self, config):
        self.config = config


def _make_runtime(robot, robot_config=None):
    from maxim.embodied_runtime.agentic_runtime import AgenticRuntimeMixin

    class _Runtime(AgenticRuntimeMixin):
        def __init__(self):
            self.log = logging.getLogger("test-doa-runtime")
            self._robot = robot
            if robot_config is not None:
                self._resolved_robot_config = robot_config

    return _Runtime()


class TestMaybeStartDoAFeedGates:
    """The Stage-2 triple gate in agentic_runtime._maybe_start_doa_feed."""

    def _join(self, runtime, stop):
        stop.set()
        thread = getattr(runtime, "_doa_thread", None)
        if thread is not None:
            thread.join(timeout=2.0)

    def test_all_gates_pass_starts_the_thread(self):
        _, emb = _reachy_embodiment()
        stop = threading.Event()
        runtime = _make_runtime(_FakeRobot(lambda: None))
        try:
            runtime._maybe_start_doa_feed(_FakeExecutor(emb), stop)
            assert getattr(runtime, "_doa_thread", None) is not None
            assert runtime._doa_thread.is_alive()
        finally:
            self._join(runtime, stop)

    def test_no_reader_no_thread(self):
        _, emb = _reachy_embodiment()
        stop = threading.Event()

        class _NoDoARobot:
            pass

        runtime = _make_runtime(_NoDoARobot())
        runtime._maybe_start_doa_feed(_FakeExecutor(emb), stop)
        assert getattr(runtime, "_doa_thread", None) is None

    def test_bodiless_executor_no_thread(self):
        stop = threading.Event()
        runtime = _make_runtime(_FakeRobot(lambda: None))
        runtime._maybe_start_doa_feed(_FakeExecutor(None), stop)
        assert getattr(runtime, "_doa_thread", None) is None

    def test_body_without_azimuth_no_thread(self):
        stop = threading.Event()
        body = ComponentRegistry().instantiate("bodies/base_humanoid")
        # base_humanoid DOES declare azimuth — strip it to simulate a mic-less body.
        body.sensors.pop("azimuth", None)
        runtime = _make_runtime(_FakeRobot(lambda: None))
        runtime._maybe_start_doa_feed(_FakeExecutor(Embodiment(body)), stop)
        assert getattr(runtime, "_doa_thread", None) is None

    def test_robots_yaml_opt_out_wins_over_capability(self):
        _, emb = _reachy_embodiment()
        stop = threading.Event()
        runtime = _make_runtime(
            _FakeRobot(lambda: None),
            robot_config=_FakeRobotConfig({"audio_localization": False}),
        )
        runtime._maybe_start_doa_feed(_FakeExecutor(emb), stop)
        assert getattr(runtime, "_doa_thread", None) is None

    def test_reader_returning_none_means_capability_absent(self):
        _, emb = _reachy_embodiment()
        stop = threading.Event()

        class _DisconnectedRobot:
            def get_doa_reader(self):
                return None

        runtime = _make_runtime(_DisconnectedRobot())
        runtime._maybe_start_doa_feed(_FakeExecutor(emb), stop)
        assert getattr(runtime, "_doa_thread", None) is None


class TestLiveWorldOwnedSensorFilter:
    """Pre-merge review fold (phantom credit mill): while a live DoA writer
    owns ``azimuth``, the modeled self_effect must neither write, credit,
    nor blame that sensor — a modeled shift the next reading reverts would
    book relief for actuation that never happened, repeatably."""

    def _turn(self, body, emb, affordance="turn_left"):
        from maxim.embodiment.tool_bridge import ModulatorAffordanceTool

        mod = body.modulators["orient"]
        tool = ModulatorAffordanceTool(body, mod, affordance, mod.affordances[affordance], affordance, embodiment=emb)
        return tool.execute()

    def test_doa_feed_claims_the_azimuth_sensor(self):
        _, emb = _reachy_embodiment()
        assert "azimuth" not in emb.live_world_set_sensors
        DoAFeed(lambda: None, emb, stop_event=threading.Event())
        assert "azimuth" in emb.live_world_set_sensors

    def test_live_owned_azimuth_gets_no_write_no_credit(self):
        body, emb = _reachy_embodiment()
        emb.live_world_set_sensors.add("azimuth")
        body.vital_metrics["azimuth"] = -0.5  # breached, as a live reading would set
        result = self._turn(body, emb)
        assert result.success
        # No modeled write: the world owns this sensor.
        assert body.vital_metrics["azimuth"] == -0.5
        # No phantom relief credit.
        se = result.side_effects or {}
        assert "drive_potential_diff" not in se
        # No B8 blame for a delta that was never applied: the lingering
        # azimuth breach is a bystander failure, filtered out.
        failures = se.get("embodiment_failures", [])
        assert not any("azimuth" in f.get("name", "") for f in failures)
        # Motor semantics preserved: head_yaw still written.
        assert body.vital_metrics["head_yaw"] == 0.3

    def test_sim_semantics_unchanged_when_set_empty(self):
        body, emb = _reachy_embodiment()
        body.vital_metrics["azimuth"] = -0.5
        result = self._turn(body, emb)
        assert result.side_effects["drive_potential_diff"] > 0  # Gate A intact in sim
        assert abs(body.vital_metrics["azimuth"] - (-0.33)) < 1e-9


class TestFeedRestartHygiene:
    def test_maybe_start_clears_stale_state_when_gates_fail(self):
        """Pre-merge review fold: a stale _doa_sim_adapter from a prior
        session must not replay last session's bearing into a restarted
        loop whose gates fail this time."""
        stop = threading.Event()

        class _NoDoARobot:
            pass

        runtime = _make_runtime(_NoDoARobot())
        runtime._doa_sim_adapter = object()  # leftovers from "session 1"
        runtime._doa_thread = object()
        runtime._doa_feed = object()
        runtime._maybe_start_doa_feed(_FakeExecutor(None), stop)
        assert runtime._doa_sim_adapter is None
        assert runtime._doa_thread is None
        assert runtime._doa_feed is None

    def test_opt_out_accepts_string_false(self):
        _, emb = _reachy_embodiment()
        stop = threading.Event()
        runtime = _make_runtime(
            _FakeRobot(lambda: None),
            robot_config=_FakeRobotConfig({"audio_localization": "false"}),
        )
        runtime._maybe_start_doa_feed(_FakeExecutor(emb), stop)
        assert getattr(runtime, "_doa_thread", None) is None


class TestAudioSalienceConfig:
    """robots.yaml audio_salience/audio_novelty reach the DoAFeed percepts
    (2026-08-01): with the passive defaults (0.5/0.3) a no_media,
    no-typed-input live session NEVER submits to the LLM — sound is
    perceived but below every > 0.5 escalation gate, so the head never
    moves. Raising audio_salience is the operator's escalation knob."""

    def _feed_percept_weights(self, runtime, emb):
        stop = threading.Event()
        try:
            runtime._maybe_start_doa_feed(_FakeExecutor(emb), stop)
            feed = runtime._doa_feed
            assert feed is not None
            return feed._salience, feed._novelty
        finally:
            stop.set()
            thread = getattr(runtime, "_doa_thread", None)
            if thread is not None:
                thread.join(timeout=2.0)

    def test_defaults_stay_passive(self):
        _, emb = _reachy_embodiment()
        runtime = _make_runtime(_FakeRobot(lambda: None))
        sal, nov = self._feed_percept_weights(runtime, emb)
        assert sal == 0.5 and nov == 0.3  # below/at every > 0.5 gate

    def test_configured_salience_reaches_the_feed(self):
        _, emb = _reachy_embodiment()
        runtime = _make_runtime(
            _FakeRobot(lambda: None),
            robot_config=_FakeRobotConfig({"audio_salience": 0.75, "audio_novelty": 0.6}),
        )
        sal, nov = self._feed_percept_weights(runtime, emb)
        assert sal == 0.75 and nov == 0.6

    def test_malformed_value_falls_back_with_default(self):
        _, emb = _reachy_embodiment()
        runtime = _make_runtime(
            _FakeRobot(lambda: None),
            robot_config=_FakeRobotConfig({"audio_salience": "loud", "audio_novelty": 7}),
        )
        sal, nov = self._feed_percept_weights(runtime, emb)
        assert sal == 0.5 and nov == 0.3  # never silently clamp misconfiguration

    def test_escalating_salience_emits_escalating_percepts(self):
        """End-to-end: a configured 0.75 percept passes is_audio_escalation."""
        from maxim.embodiment.audio_localization import is_audio_escalation

        _, emb = _reachy_embodiment()
        stop = threading.Event()
        seen = []
        reader = _ScriptedReader([_SPEECH_LEFT] * 3, stop_event=stop)
        feed = DoAFeed(
            reader,
            emb,
            stop_event=stop,
            percept_sink=seen.append,
            salience=0.75,
            sample_poll_s=0.0,
            sample_timeout_s=0.5,
        )
        feed.run()
        assert len(seen) == 1
        assert is_audio_escalation(seen[0].salience)
