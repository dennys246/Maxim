"""The ``sim.stdin`` operator-line reader the run's stop ends (``simulation/stdin_lines.py``, #1541).

On an OS pipe (the selectable path a terminal takes) and on a stdin with no file descriptor (the blocking fallback).
The cross-run case is two runs reading the SAME stdin, as an in-process caller's runs read the process's one stdin:
the first run's reader, stopped, must leave the next line to the second.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.timeout(30)  # a reader the stop does not end hangs: fail it, never wait it out

from maxim.simulation import stdin_lines as SLN
from maxim.simulation.stdin_lines import OperatorLines, read_operator_lines, stop_reader


class _PipeStdin(io.TextIOBase):
    """A stdin over an OS pipe: ``fileno`` is the read end, the test writes the other."""

    def __init__(self) -> None:
        super().__init__()
        self.r, self.w = os.pipe()

    def fileno(self) -> int:
        return self.r

    def write_bytes(self, data: bytes) -> None:
        os.write(self.w, data)

    def close_write(self) -> None:
        os.close(self.w)

    def close(self) -> None:
        if not self.closed:
            for fd in (self.r, self.w):
                try:
                    os.close(fd)
                except OSError:
                    pass  # the test closed the write end already
        super().close()


class _BlockingStdin(io.TextIOBase):
    """A stdin with no file descriptor whose ``readline`` blocks until the test hands it a line."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []
        self.cv = threading.Condition()
        self.calls = 0

    def readline(self, size: int | None = -1) -> str:  # type: ignore[override]
        with self.cv:
            self.calls += 1
            self.cv.wait_for(lambda: bool(self.lines), timeout=10)
            return self.lines.pop(0) if self.lines else ""

    def give(self, line: str) -> None:
        with self.cv:
            self.lines.append(line)
            self.cv.notify_all()


@pytest.fixture
def pipe() -> Any:
    stdin = _PipeStdin()
    yield stdin
    stdin.close()


def _reader(lines: OperatorLines, stop: threading.Event, seen: list[str], **kw: Any) -> threading.Thread:
    def process(line: str) -> bool:
        seen.append(line)
        return line == "/cancel"

    kwargs: dict[str, Any] = {
        "lines": lines,
        "stop_event": stop,
        "operator_stop": threading.Event(),
        "sim_prompt_handler": None,
        "process_line": process,
        "stop_by_operator": lambda s, o: (o.set(), s.set()),
    }
    kwargs.update(kw)
    thread = threading.Thread(target=read_operator_lines, kwargs=kwargs, name="sim.stdin", daemon=True)
    thread.start()
    return thread


def _until(cond: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_the_stop_ends_a_read_in_progress_on_a_selectable_stdin(pipe: _PipeStdin) -> None:
    stop = threading.Event()
    lines = OperatorLines(pipe, stop)
    assert lines.interruptible
    seen: list[str] = []
    thread = _reader(lines, stop, seen)
    time.sleep(3 * SLN.POLL_S)  # blocked in the read: nothing typed yet
    stop.set()
    assert stop_reader(thread) is True and seen == []


def test_a_stopped_runs_reader_leaves_the_next_line_to_the_next_run(pipe: _PipeStdin) -> None:
    """Two runs on the same stdin: the first ended without ``/cancel`` (a finish), the second gets the line."""
    first_stop, second_stop = threading.Event(), threading.Event()
    first_seen: list[str] = []
    first = _reader(OperatorLines(pipe, first_stop), first_stop, first_seen)
    time.sleep(3 * SLN.POLL_S)
    first_stop.set()
    assert stop_reader(first)
    second_seen: list[str] = []
    second = _reader(OperatorLines(pipe, second_stop), second_stop, second_seen)
    pipe.write_bytes(b"/cancel\n")
    second.join(timeout=5)
    assert first_seen == [] and second_seen == ["/cancel"]


def test_lines_split_from_one_read_and_a_character_split_across_two(pipe: _PipeStdin) -> None:
    stop = threading.Event()
    lines = OperatorLines(pipe, stop)
    pipe.write_bytes(b"/status\nhello \xe2\x89")
    assert lines.next_line() == "/status"
    reader_got: list[str | None] = []
    thread = threading.Thread(target=lambda: reader_got.append(lines.next_line()), daemon=True)
    thread.start()
    time.sleep(2 * SLN.POLL_S)
    pipe.write_bytes(b"\xa0 there\npartial")
    thread.join(timeout=5)
    assert reader_got == ["hello ≠ there"]
    pipe.close_write()
    assert lines.next_line() == "partial", "the last line without a newline, at EOF"
    with pytest.raises(EOFError):
        lines.next_line()


def test_a_line_already_buffered_is_returned_before_the_stop_is_looked_at(pipe: _PipeStdin) -> None:
    stop = threading.Event()
    lines = OperatorLines(pipe, stop)
    pipe.write_bytes(b"one\ntwo\n")
    assert lines.next_line() == "one"
    stop.set()
    assert lines.next_line() == "two", "the reader's loop checks the stop between lines, not next_line"
    assert lines.next_line() is None


def test_a_stdin_with_no_fd_blocks_and_drops_a_line_that_arrives_after_the_stop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stdin = _BlockingStdin()
    stop = threading.Event()
    with caplog.at_level(logging.DEBUG, logger=SLN.logger.name):
        lines = OperatorLines(stdin, stop)
    assert not lines.interruptible
    assert any("no selectable fd" in r.getMessage() for r in caplog.records)
    seen: list[str] = []
    thread = _reader(lines, stop, seen)
    stdin.give("hello\n")
    _until(lambda: seen == ["hello"])
    _until(lambda: stdin.calls == 2)  # blocked on the next line
    stop.set()
    with caplog.at_level(logging.WARNING, logger=SLN.logger.name):
        assert stop_reader(thread, timeout_s=0.2) is False
    assert any("did not stop within 0.2s" in r.getMessage() for r in caplog.records)
    stdin.give("meant for a later run\n")
    thread.join(timeout=5)
    assert not thread.is_alive() and seen == ["hello"], "the late line is dropped, not processed"


def test_the_blocking_fallback_strips_only_the_newline_and_ends_at_eof() -> None:
    stop = threading.Event()
    lines = OperatorLines(io.StringIO("  spaced \r\nlast"), stop)
    assert lines.next_line() == "  spaced \r", "as input() returns it"
    assert lines.next_line() == "last"
    with pytest.raises(EOFError):
        lines.next_line()


def test_windows_takes_the_blocking_fallback(monkeypatch: pytest.MonkeyPatch, pipe: _PipeStdin) -> None:
    monkeypatch.setattr(SLN, "sys", SimpleNamespace(platform="win32"))  # only the module's own reference
    assert not OperatorLines(pipe, threading.Event()).interruptible
    assert not OperatorLines(None, threading.Event()).interruptible


def test_a_reader_with_no_time_left_is_not_waited_for_and_not_blamed_on_stdin(
    pipe: _PipeStdin, caplog: pytest.LogCaptureFixture
) -> None:
    stop = threading.Event()
    thread = _reader(OperatorLines(pipe, stop), stop, [])
    with caplog.at_level(logging.INFO, logger=SLN.logger.name):
        assert stop_reader(thread, timeout_s=0.0) is False
    assert [r.levelno for r in caplog.records] == [logging.INFO], "an expired bound is not a stuck reader"
    assert "no time left" in caplog.records[0].getMessage()
    stop.set()
    assert stop_reader(thread) is True


class _Prompt:
    def __init__(self) -> None:
        self.has_pending_prompt = True
        self.responses: list[str] = []

    def deliver_response(self, text: str) -> None:
        self.responses.append(text)
        self.has_pending_prompt = False


def test_a_pending_prompt_takes_the_line_then_the_parser_gets_the_next(pipe: _PipeStdin) -> None:
    stop = threading.Event()
    prompt = _Prompt()
    seen: list[str] = []
    pipe.write_bytes(b"  yes please \n/cancel\n")
    thread = _reader(OperatorLines(pipe, stop), stop, seen, sim_prompt_handler=prompt)
    thread.join(timeout=5)
    assert prompt.responses == ["yes please"] and seen == ["/cancel"]


@pytest.mark.parametrize("pending", [True, False])
def test_end_of_input_stops_the_run_only_under_a_pending_prompt(pipe: _PipeStdin, pending: bool) -> None:
    stop, operator_stop = threading.Event(), threading.Event()
    prompt = _Prompt()
    prompt.has_pending_prompt = pending
    pipe.close_write()
    thread = _reader(OperatorLines(pipe, stop), stop, [], operator_stop=operator_stop, sim_prompt_handler=prompt)
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert operator_stop.is_set() is pending and stop.is_set() is pending


def test_an_interrupt_stops_the_run() -> None:
    class _Interrupting:
        def next_line(self) -> str | None:
            raise KeyboardInterrupt

    stop, operator_stop = threading.Event(), threading.Event()
    read_operator_lines(
        lines=_Interrupting(),  # type: ignore[arg-type]
        stop_event=stop,
        operator_stop=operator_stop,
        sim_prompt_handler=None,
        process_line=lambda line: False,
        stop_by_operator=lambda s, o: (o.set(), s.set()),
    )
    assert stop.is_set() and operator_stop.is_set()
