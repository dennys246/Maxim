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


def _run_writer(tmp_path, process, *, seconds: float = 2.0, rate: int = 16000, stopped_first: bool = False):
    """Feed the writer `seconds` of audio; the session stops only once the queue is drained (as in live()),
    unless `stopped_first`, which models a session already stopping."""
    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    maxim = SimpleNamespace(audio=True, audio_len=1.0, log=logging.getLogger("test"))
    samples = (np.sin(np.arange(int(rate * seconds)) / 10.0) * 1000).astype(np.int16)
    for block in np.array_split(samples, 8):
        q.put((0.0, block, rate))
    if stopped_first:
        stop.set()
    writer = threading.Thread(
        target=audio_writer_worker,
        args=(stop, maxim, q, str(tmp_path / "audio.wav"), str(tmp_path / "chunks"), rate, rate, process),
    )
    writer.start()
    q.join()
    stop.set()
    writer.join(timeout=5.0)
    assert not writer.is_alive()
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
            spawned.append({**kwargs, "_target": target, "_args": args})
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
    # setenv first so teardown removes what the helper writes (device=cpu sets it)
    monkeypatch.setenv("MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY", "")
    monkeypatch.delenv("MAXIM_TRANSCRIPTION_WORKER_CPU_ONLY")
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


def test_an_exited_transcriber_during_a_stop_is_not_reported(tmp_path, caplog):
    """A session that is stopping expects its transcriber to exit: no "Voice input is OFF" on an ordinary stop."""
    with caplog.at_level(logging.ERROR):
        _run_writer(tmp_path, _Process(alive=False), stopped_first=True)
    assert [r for r in caplog.records if "Voice input is OFF" in r.getMessage()] == []


def test_the_transcriber_runs_in_its_own_process_group(monkeypatch, tmp_path):
    """A terminal Ctrl-C reaches the foreground group; the child must leave it so only the parent stops it."""
    import maxim.utils.audio as audio_mod
    from maxim.inference.transcribe_audio import watch_and_transcribe

    (_, _), spawned = _start(monkeypatch, tmp_path)
    assert spawned[0]["_target"] is audio_mod.run_in_own_process_group
    assert spawned[0]["_args"][0] is watch_and_transcribe

    calls = []
    monkeypatch.setattr(audio_mod.os, "setpgrp", lambda: calls.append("setpgrp"), raising=False)
    assert audio_mod.run_in_own_process_group(lambda *a, **k: calls.append(("target", a, k)) or "ran", 1, x=2) == "ran"
    assert calls == ["setpgrp", ("target", (1,), {"x": 2})]


def test_live_feeds_the_helpers_worker_to_the_audio_writer_with_a_per_run_chunk_dir():
    """The seam #1436 broke: live() must start the transcriber through the helper, hand the worker it returns to
    audio_writer_worker, and keep this run's chunks in a directory of their own."""
    import ast
    import inspect
    import textwrap

    from maxim.embodied_runtime.media_loop import MediaLoopMixin

    tree = ast.parse(textwrap.dedent(inspect.getsource(MediaLoopMixin.live)))
    helper_targets = [
        [t.id for t in node.targets[0].elts]
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and getattr(node.value.func, "attr", "") == "_start_transcription_worker"
    ]
    assert helper_targets == [["transcribe_process", "transcribe_shutdown_file"]]
    writer_args = [
        [getattr(a, "id", None) for a in kw.value.elts]
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "Thread"
        for kw in node.keywords
        if kw.arg == "args"
        and isinstance(kw.value, ast.Tuple)
        and any(getattr(k.value, "id", None) == "audio_writer_worker" for k in node.keywords if k.arg == "target")
    ]
    assert writer_args and "transcribe_process" in writer_args[0] and "chunk_dir" in writer_args[0]
    chunk_dir = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "chunk_dir"
    ]
    assert len(chunk_dir) == 1 and "run_id" in ast.unparse(chunk_dir[0])


def test_importing_the_robot_media_modules_loads_no_speech_or_gpu_stack():
    """The hoisted imports must stay cheap: the parent hides its GPUs, and a robot without the `audio` extra must
    still import the runtime (transcribe_audio is Session V's file; a top-level faster_whisper there breaks this)."""
    import os
    import subprocess
    import sys

    code = (
        "import sys, maxim.embodied_runtime.media_loop, maxim.embodied_runtime.workers;"
        "print(sorted(m for m in ('faster_whisper', 'ctranslate2', 'torch') if m in sys.modules))"
    )
    from pathlib import Path

    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")}  # this checkout's maxim
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env=env)
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == "[]"


def test_an_orphaned_transcriber_exits_when_its_parent_dies_hard(tmp_path):
    """Leaving the terminal's group also drops its SIGHUP: a parent killed without its teardown must not leave the
    transcriber (and its model) running. A real spawn: the parent exits hard, the child must follow."""
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    script = tmp_path / "parent.py"
    script.write_text(
        "import multiprocessing as mp, os, time\n"
        "from maxim.utils.audio import run_in_own_process_group\n"
        "if __name__ == '__main__':\n"
        "    child = mp.get_context('spawn').Process(target=run_in_own_process_group, args=(time.sleep, 120))\n"
        "    child.start()\n"
        "    print(child.pid, flush=True)\n"
        "    os._exit(0)  # a hard death: no teardown, no join\n"
    )
    env = {**os.environ, "PYTHONPATH": str(src)}
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=60, env=env)
    pid = int(out.stdout.strip())
    deadline = time.time() + 30.0
    try:
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return  # the orphan exited
            time.sleep(0.2)
        pytest.fail("the orphaned transcriber outlived its parent")
    finally:
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            pass
