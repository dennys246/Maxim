"""Behaviour of the code #1499 PR 3 moved out of ``MediaLoopMixin.live``: the observation step, the transcriber's stop,
and the live loop's thread setup and frame loop, driven with stubs (no robot, audio or real worker threads)."""

from __future__ import annotations

import logging
import os
import signal
import threading
import time
from types import SimpleNamespace

import pytest

import maxim.embodied_runtime.media_loop as media_loop
import maxim.embodied_runtime.workers as workers
from maxim.embodied_runtime.media_loop import MediaLoopMixin

LOGGER = "test.media_loop_live_coverage_1499"


class _Host(MediaLoopMixin):
    def __init__(self) -> None:
        self.log = logging.getLogger(LOGGER)
        self.verbosity = 0
        self.verbose = False
        self.current_epoch = 3
        self.mode = "active"
        self._observation_lock = threading.Lock()


# ---------------------------------------------------------------------------
# _observe_frame
# ---------------------------------------------------------------------------


class _CaptureManager:
    def __init__(self, captured) -> None:
        self.captured = captured

    def get_latest_frame(self):
        return self.captured


@pytest.fixture
def observation_calls(monkeypatch):
    calls: dict[str, list] = {"display": [], "passive": []}

    def _display(frame, detections, **kwargs):
        calls["display"].append((frame, detections, kwargs))
        return {"target": "from-display"}

    def _passive(host, photo, *, show):
        calls["passive"].append((host, photo, show))
        return {"target": "from-passive"}

    monkeypatch.setattr(media_loop, "display_detections", _display)
    monkeypatch.setattr(media_loop, "passive_observation", _passive)
    return calls


def _segmented():
    return SimpleNamespace(segmented=True, frame="seg-frame", detections=["person", "cup"])


def test_a_segmented_capture_frame_is_displayed_when_verbose(observation_calls, caplog):
    host = _Host()
    host.verbose, host.verbosity = True, 2
    host._capture_manager = _CaptureManager(_segmented())
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        host._observe_frame("photo")
    assert len(observation_calls["display"]) == 1 and not observation_calls["passive"]
    frame, detections, kwargs = observation_calls["display"][0]
    assert frame == "seg-frame" and detections == ["person", "cup"] and kwargs["segmenter"] is None
    assert host._last_detection_target == {"target": "from-display"}
    assert "Display frame from CaptureManager (epoch=3, detections=2)" in caplog.text


def test_a_segmented_capture_frame_is_not_displayed_when_quiet(observation_calls):
    host = _Host()
    host._capture_manager = _CaptureManager(_segmented())
    host._observe_frame("photo")
    assert not observation_calls["display"] and not observation_calls["passive"]
    assert not hasattr(host, "_last_detection_target")  # no target to store


@pytest.mark.parametrize("captured", [None, SimpleNamespace(segmented=False, frame="f", detections=[])])
def test_a_capture_manager_without_a_segmented_frame_falls_back_to_passive(observation_calls, caplog, captured):
    host = _Host()
    host.verbosity = 2
    host._capture_manager = _CaptureManager(captured)
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        host._observe_frame("photo")
    assert observation_calls["passive"] == [(host, "photo", False)] and not observation_calls["display"]
    assert host._last_detection_target == {"target": "from-passive"}
    assert f"captured={captured is not None}" in caplog.text


def test_no_capture_manager_uses_the_legacy_passive_observation(observation_calls, caplog):
    host = _Host()
    host.verbosity, host.verbose = 2, True
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        host._observe_frame("photo")
    assert observation_calls["passive"] == [(host, "photo", True)]
    assert "Display using legacy passive_observation (epoch=3)" in caplog.text


@pytest.mark.parametrize("verbosity", [0, 2])
def test_a_failing_observation_step_is_logged_not_raised(monkeypatch, caplog, verbosity):
    def _boom(*_a, **_k):
        raise RuntimeError("segmenter crashed")

    monkeypatch.setattr(media_loop, "passive_observation", _boom)
    host = _Host()
    host.verbosity = verbosity
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        host._observe_frame("photo")
    records = [r for r in caplog.records if "Observation step failed (mode=active)" in r.getMessage()]
    assert len(records) == 1
    if verbosity >= 2:
        assert records[0].exc_info is not None  # log.exception carries the traceback
    else:
        assert records[0].exc_info is None and "segmenter crashed" in records[0].getMessage()


# ---------------------------------------------------------------------------
# _stop_transcription_worker
# ---------------------------------------------------------------------------


class _Proc:
    """A transcriber stand-in that dies after ``alive_checks`` reads of ``is_alive`` that saw it alive."""

    pid = 424242

    def __init__(self, *, exits_on: str | None, kill_raises: bool = False, terminate_raises: bool = False) -> None:
        self.exits_on = exits_on  # "join", "terminate", "kill" or None (never)
        self.kill_raises = kill_raises
        self.terminate_raises = terminate_raises
        self.alive = True
        self.calls: list[str] = []

    def join(self, timeout=None):
        self.calls.append(f"join({timeout})")
        if self.exits_on == "join":
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.calls.append("terminate")
        if self.terminate_raises:
            raise OSError("terminate failed")
        if self.exits_on == "terminate":
            self.alive = False

    def kill(self):
        self.calls.append("kill")
        if self.kill_raises:
            raise OSError("kill failed")
        if self.exits_on == "kill":
            self.alive = False


@pytest.fixture
def os_kills(monkeypatch):
    sent: list[tuple[int, int]] = []

    def _kill(pid, sig):
        sent.append((pid, sig))
        if pid == _Proc.pid:
            raise ProcessLookupError(pid)  # the fake pid has no process; the stop must shrug it off

    monkeypatch.setattr(media_loop.os, "kill", _kill)
    return sent


def test_no_process_and_no_file_is_a_no_op(os_kills):
    _Host()._stop_transcription_worker(None, None)
    assert os_kills == []


def test_a_worker_that_exits_on_the_shutdown_file_gets_no_signal(tmp_path, os_kills):
    shutdown = tmp_path / ".shutdown"
    seen: list[str] = []
    proc = _Proc(exits_on="join")
    proc.join = lambda timeout=None: (seen.append(shutdown.read_text()), setattr(proc, "alive", False))  # type: ignore[method-assign]
    _Host()._stop_transcription_worker(proc, str(shutdown))
    assert seen == ["shutdown\n"]  # the file was written before the join
    assert proc.calls == [] and os_kills == []
    assert not shutdown.exists()  # cleaned up


def test_a_worker_that_needs_sigterm_is_terminated(tmp_path, os_kills, caplog):
    proc = _Proc(exits_on="terminate")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _Host()._stop_transcription_worker(proc, None)
    assert proc.calls == ["join(2.0)", "terminate", "join(1.0)"]
    assert os_kills == [(_Proc.pid, signal.SIGTERM)]
    assert "SIGKILL" not in caplog.text


@pytest.mark.parametrize("kill_raises", [False, True], ids=["kill", "kill-raises-os-kill-fallback"])
def test_a_worker_that_ignores_sigterm_is_killed(monkeypatch, caplog, kill_raises):
    proc = _Proc(exits_on=None if kill_raises else "kill", kill_raises=kill_raises, terminate_raises=True)
    sent: list[tuple[int, int]] = []

    def _kill(pid, sig):
        sent.append((pid, sig))
        if sig == signal.SIGKILL and kill_raises:
            proc.alive = False  # the direct os.kill fallback is what ends it
            return
        raise ProcessLookupError(pid)

    monkeypatch.setattr(media_loop.os, "kill", _kill)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _Host()._stop_transcription_worker(proc, None)
    assert proc.calls == ["join(2.0)", "terminate", "join(1.0)", "kill", "join(1.0)"]
    assert sent == [(_Proc.pid, signal.SIGTERM), (_Proc.pid, signal.SIGKILL)]
    assert "did not respond to SIGTERM, sending SIGKILL" in caplog.text
    assert "could not be terminated" not in caplog.text
    assert not proc.is_alive()


def test_a_worker_that_cannot_be_terminated_is_logged(os_kills, caplog):
    proc = _Proc(exits_on=None)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _Host()._stop_transcription_worker(proc, None)
    assert "Transcription process could not be terminated (pid=424242)" in caplog.text
    assert proc.is_alive()


def test_a_join_that_raises_does_not_skip_the_later_signals(os_kills, caplog):
    """Each join is guarded: a join that raises still leads to SIGTERM, SIGKILL and the final error."""
    proc = _Proc(exits_on=None)

    def _join(timeout=None):
        proc.calls.append(f"join({timeout})")
        raise RuntimeError("join failed")

    proc.join = _join  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        _Host()._stop_transcription_worker(proc, None)
    assert proc.calls == ["join(2.0)", "terminate", "join(1.0)", "kill", "join(1.0)"]
    assert os_kills == [(_Proc.pid, signal.SIGTERM), (_Proc.pid, signal.SIGKILL)]
    assert "could not be terminated (pid=424242)" in caplog.text


def test_a_shutdown_file_that_cannot_be_written_or_removed_does_not_raise(tmp_path, monkeypatch, os_kills):
    unwritable = tmp_path / "missing-dir" / ".shutdown"
    _Host()._stop_transcription_worker(None, str(unwritable))  # the open fails, quietly
    assert not unwritable.exists()

    shutdown = tmp_path / ".shutdown"
    removed: list[str] = []

    def _remove(path):
        removed.append(path)
        raise PermissionError(path)

    monkeypatch.setattr(media_loop.os, "remove", _remove)
    _Host()._stop_transcription_worker(None, str(shutdown))
    assert removed == [str(shutdown)]
    assert shutdown.read_text() == "shutdown\n"  # the removal failed, so the file stays


# ---------------------------------------------------------------------------
# live(): the thread setup and the frame loop
# ---------------------------------------------------------------------------


class _Robot:
    def __init__(self) -> None:
        self.stops = 0

    def stop_motion(self):
        self.stops += 1


class _LiveHost(_Host):
    def __init__(self, tmp_path) -> None:
        super().__init__()
        self._robot = _Robot()
        self.mini = None
        self.home_dir = str(tmp_path)
        self.epochs = 1
        self.current_epoch = 0
        self.observation_period = 1
        self.interactive = False
        self.audio = True
        self.audio_len = 1.0
        self._capabilities = None
        self.sleeping = False
        self._live_stopped = False
        self._motor_queue = None
        self.listener_threads: list[threading.Thread] = []
        self.observed: list = []
        self.shutdowns = 0

    def awaken(self, **_k):
        pass

    def _build_speech_output(self):
        pass

    def _start_media_audio(self):
        return 16000, 16000

    def _start_agentic_runtime(self):
        pass

    def _start_transcription_worker(self, **_k):
        return None, None

    def _release_media(self):
        pass

    def shutdown(self):
        self.shutdowns += 1

    def _listener(self, name):
        def _start(_stop):
            t = threading.Thread(target=lambda: None, name=name, daemon=True)
            self.listener_threads.append(t)
            return t

        return _start

    def _observe_frame(self, photo):
        self.observed.append(photo)


def test_live_starts_every_thread_observes_a_frame_and_joins_them(tmp_path, monkeypatch):
    monkeypatch.setattr(media_loop, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(media_loop, "TrainingSampleLogger", lambda *_a: SimpleNamespace(start=lambda: None))
    ran: list[str] = []

    def _capture(stop_event, _lock, _host, _save_q, obs_q):
        ran.append("capture")
        obs_q.put((time.time(), "frame-1"), timeout=2.0)
        stop_event.wait(timeout=5.0)

    def _worker(name):
        def _run(stop_event, *_a):
            ran.append(name)
            stop_event.wait(timeout=5.0)

        return _run

    def _motor(_queue, stop_event, _host):
        ran.append("motor_worker")
        stop_event.wait(timeout=5.0)

    monkeypatch.setattr(workers, "frame_capture_worker", _capture)
    monkeypatch.setattr(workers, "motor_worker", _motor)
    for name in ("video_writer_worker", "audio_capture_worker", "audio_writer_worker"):
        monkeypatch.setattr(workers, name, _worker(name))

    host = _LiveHost(tmp_path)
    host._start_cli_listener = host._listener("cli")
    host._start_key_listener = host._listener("key")
    host._start_transcript_listener = host._listener("transcript")
    started: list[threading.Thread] = []
    real_start = threading.Thread.start

    def _start(thread):
        started.append(thread)
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", _start)
    try:
        host.live(vision=True, motor=True, parallel=True, run_id="coverage-1499")
    finally:
        host._close_live_stop_event()

    names = [t.name for t in started]
    assert names == [
        "cli",
        "key",
        "transcript",
        "maxim.capture.video",
        "maxim.write.video",
        "maxim.motor",
        "maxim.capture.audio",
        "maxim.write.audio",
    ]  # each listener started once, by its own branch, not again by the worker loop
    assert host.observed == ["frame-1"]  # observation_period 1: the one frame is observed
    assert host.current_epoch == 1  # the epochs limit ended the loop
    assert sorted(ran) == sorted(
        ["capture", "video_writer_worker", "motor_worker", "audio_capture_worker", "audio_writer_worker"]
    )
    assert all(not t.is_alive() for t in started)  # the teardown's stop + join ended every thread
    assert host._robot.stops == 1 and host.shutdowns == 1
    assert os.path.isfile(os.path.join(str(tmp_path), "transcript", "reachy_transcript_coverage-1499.jsonl"))
