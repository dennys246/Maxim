"""The robot's voice input reaches the transcriber, or says loudly that it cannot (#1436).

`db274547` deleted the `_file_based_transcription` re-export shim; the robot's transcription start
(`MediaLoopMixin.live`) and the audio writer's hand-off to the transcriber
(`workers.py::audio_writer_worker`) still imported it lazily, so every robot session since 2026-04-03
ran deaf behind one warning. Design and owner decisions: docs/plans/lazy_import_lint_1436.md.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from maxim.embodied_runtime.workers import audio_writer_worker


class _Process:
    def __init__(self, alive: bool = True) -> None:
        self.alive = alive

    def is_alive(self) -> bool:
        return self.alive


def _run_writer(tmp_path, process, *, seconds: float = 2.0, rate: int = 16000):
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    maxim = SimpleNamespace(audio=True, audio_len=1.0, log=logging.getLogger("test"))
    samples = (np.sin(np.arange(int(rate * seconds)) / 10.0) * 1000).astype(np.int16)
    for block in np.array_split(samples, 8):
        q.put((0.0, block, rate))
    stop.set()
    audio_writer_worker(stop, maxim, q, str(tmp_path / "audio.wav"), str(tmp_path / "chunks"), rate, rate, process)
    return sorted((tmp_path / "chunks").glob("task_*.json"))


def test_the_audio_writer_hands_each_chunk_to_a_live_transcriber(tmp_path):
    tasks = _run_writer(tmp_path, _Process(alive=True))
    assert len(tasks) == 2, "two 1 s chunks from 2 s of audio, each queued for the transcriber"
    assert json.loads(tasks[0].read_text())["sample_rate"] == 16000


def test_a_dead_transcriber_is_reported_once_and_no_tasks_pile_up(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        tasks = _run_writer(tmp_path, _Process(alive=False), seconds=3.0)
    assert tasks == [], "no task files for a transcriber that is not running"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and "voice input" in r.getMessage().lower()]
    assert len(errors) == 1, "one ERROR naming voice input, not one per chunk"


# ── The robot's transcription start (MediaLoopMixin._start_transcription_worker) ────────────────────────────────


class _Runtime:
    """Just the attributes the start helper reads from the embodied runtime."""

    def __init__(self, *, audio: bool = True) -> None:
        self.audio = audio
        self._robot = object()
        self.verbosity = 0
        self.log = logging.getLogger("test.voice")


def _start(monkeypatch, tmp_path, *, installed=True, alive=True, config_error=None, audio=True, blackwell=False):
    import maxim.embodied_runtime.media_loop as ml

    spawned = []

    class _Ctx:
        def Process(self, target, args, kwargs, daemon):  # noqa: N802 - mirrors multiprocessing
            spawned.append(kwargs)
            proc = _Process(alive=alive)
            proc.start = lambda: None
            return proc

    def _config():
        if config_error is not None:
            raise config_error
        return SimpleNamespace(
            model="tiny.en",
            device="cpu",
            compute_type="int8",
            vad_filter=False,
            language="en",
            beam_size=1,
            cleanup_chunks=True,
        )

    monkeypatch.setattr(ml, "optional_dependency_installed", lambda name: installed and name == "faster_whisper")
    monkeypatch.setattr(ml, "load_whisper_config", _config)
    monkeypatch.setattr(ml.mp, "get_context", lambda kind: _Ctx())
    monkeypatch.setattr(ml.time, "sleep", lambda s: None)
    monkeypatch.setattr(ml._gpu_state, "blackwell_detected", blackwell)
    monkeypatch.delenv("MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY", raising=False)
    result = ml.MediaLoopMixin._start_transcription_worker(
        _Runtime(audio=audio),
        parallel=True,
        chunk_dir=str(tmp_path / "chunks"),
        transcript_path=str(tmp_path / "t.jsonl"),
        log_path=str(tmp_path / "w.log"),
    )
    return result, spawned


def test_a_robot_session_with_the_audio_extra_spawns_the_transcriber(monkeypatch, tmp_path):
    (proc, shutdown), spawned = _start(monkeypatch, tmp_path)
    assert proc is not None and shutdown.endswith(".shutdown")
    assert spawned and spawned[0]["model_size_or_path"] == "tiny.en"


def test_a_missing_audio_extra_is_one_error_naming_the_fix_and_the_session_continues(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        (proc, shutdown), spawned = _start(monkeypatch, tmp_path, installed=False)
    assert (proc, shutdown) == (None, None) and spawned == []
    (error,) = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert "Voice input is OFF" in error.getMessage() and "pymaxim[audio]" in error.getMessage()


@pytest.mark.parametrize("alive, config_error", [(False, None), (True, RuntimeError("bad whisper.json"))])
def test_any_other_start_failure_is_one_error_and_no_worker(monkeypatch, tmp_path, caplog, alive, config_error):
    with caplog.at_level(logging.ERROR):
        (proc, shutdown), _ = _start(monkeypatch, tmp_path, alive=alive, config_error=config_error)
    assert (proc, shutdown) == (None, None)
    assert len([r for r in caplog.records if "Voice input is OFF" in r.getMessage()]) == 1


def test_no_robot_audio_means_no_transcriber_and_no_noise(monkeypatch, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        (proc, shutdown), spawned = _start(monkeypatch, tmp_path, audio=False)
    assert (proc, shutdown, spawned, caplog.records) == (None, None, [], [])


def test_a_blackwell_gpu_switches_int8_to_float16(monkeypatch, tmp_path):
    _, spawned = _start(monkeypatch, tmp_path, blackwell=True)
    assert spawned[0]["compute_type"] == "float16"
    _, spawned = _start(monkeypatch, tmp_path, blackwell=False)
    assert spawned[0]["compute_type"] == "int8"


# ── The other dead imports the lint found ───────────────────────────────────────────────────────────────────────


def test_wiring_comms_with_a_nac_no_longer_imports_the_deleted_bridge():
    from maxim.agents.maxim_agent import MaximAgent

    agent = MaximAgent.__new__(MaximAgent)
    gateway = object()
    agent.wire_communication(gateway=gateway, nac=object(), memory_hub=object())
    assert agent._gateway is gateway


def test_the_spec_only_probe_never_imports_and_never_raises():
    import sys

    from maxim.utils.optional_deps import optional_dependency_installed

    assert optional_dependency_installed("json") is True
    assert optional_dependency_installed("zz_not_installed_1436") is False
    assert optional_dependency_installed("zz_not_installed_1436.sub") is False
    sys.modules.pop("colorsys", None)
    assert optional_dependency_installed("colorsys") is True
    assert "colorsys" not in sys.modules, "find_spec only: the probe must not import the package"


def test_a_task_pending_when_the_transcriber_dies_is_dropped_not_written(tmp_path, monkeypatch, caplog):
    """The flush half of S5: a task whose file failed to write is pending when the child dies; never write it."""
    import maxim.embodied_runtime.workers as workers

    proc = _Process(alive=True)
    real = workers.create_task_file
    calls = []

    def _fails_then_child_dies(**kwargs):
        calls.append(kwargs["chunk_index"])
        if len(calls) == 1:
            proc.alive = False  # the child dies while this first task is still pending
            return None
        return real(**kwargs)

    monkeypatch.setattr(workers, "create_task_file", _fails_then_child_dies)
    with caplog.at_level(logging.ERROR):
        tasks = _run_writer(tmp_path, proc, seconds=3.0)
    assert tasks == [] and calls == [0]
    assert len([r for r in caplog.records if "Voice input is OFF" in r.getMessage()]) == 1
