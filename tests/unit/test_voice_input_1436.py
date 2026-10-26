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


@pytest.mark.xfail(strict=True, reason="#1436: the writer imports the deleted _file_based_transcription shim")
def test_the_audio_writer_hands_each_chunk_to_a_live_transcriber(tmp_path):
    tasks = _run_writer(tmp_path, _Process(alive=True))
    assert len(tasks) == 2, "two 1 s chunks from 2 s of audio, each queued for the transcriber"
    assert json.loads(tasks[0].read_text())["sample_rate"] == 16000


@pytest.mark.xfail(strict=True, reason="#1436 S5: a transcriber that died after spawn is not noticed")
def test_a_dead_transcriber_is_reported_once_and_no_tasks_pile_up(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        tasks = _run_writer(tmp_path, _Process(alive=False), seconds=3.0)
    assert tasks == [], "no task files for a transcriber that is not running"
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR and "voice input" in r.getMessage().lower()]
    assert len(errors) == 1, "one ERROR naming voice input, not one per chunk"
