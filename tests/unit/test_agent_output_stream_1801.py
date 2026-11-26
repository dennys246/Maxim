"""Two `AgentOutputManager` defects found while #1801 moved the `log_agentic` scrub into `AbstractionBuffer.append`.

- `write_to_stream` auto-opened an unopened stream through `open_stream` while holding the same non-reentrant
  `threading.Lock`, so the first write to a stream nobody opened deadlocked (reachable from
  `agents/output_mixin.py::_log_to_stream`).
- `__init__` took `abstraction_buffer or get_abstraction_buffer()`, so an empty buffer passed in (`len()` 0 is falsy)
  was silently swapped for the global one.
"""

from __future__ import annotations

import threading
from pathlib import Path


def _manager(tmp_path: Path, buffer: object | None = None):
    from maxim.utils.agent_output import AgentOutputManager

    return AgentOutputManager(
        instance_id="i1801",
        base_data_dir=str(tmp_path / "data"),
        sandbox_dir=str(tmp_path / "sandbox"),
        shared_outputs_dir=str(tmp_path / "shared"),
        abstraction_buffer=buffer,
    )


def test_the_first_write_to_an_unopened_stream_does_not_deadlock(tmp_path: Path) -> None:
    from maxim.utils.structured_logging import AbstractionBuffer

    manager = _manager(tmp_path, AbstractionBuffer(max_entries=10, verbosity=3, console_output=False))
    done = threading.Event()

    def _write() -> None:
        manager.write_to_stream("exec", "trace", {"k": "v"})
        done.set()

    worker = threading.Thread(target=_write, daemon=True)
    worker.start()
    worker.join(timeout=5.0)
    assert done.is_set(), "write_to_stream deadlocked auto-opening the stream"


def test_an_empty_buffer_passed_in_is_the_one_used(tmp_path: Path) -> None:
    from maxim.utils.structured_logging import AbstractionBuffer, get_abstraction_buffer

    mine = AbstractionBuffer(max_entries=10, verbosity=3, console_output=False)
    assert len(mine) == 0  # falsy: the defect swapped it for the global buffer
    manager = _manager(tmp_path, mine)
    assert manager._abstraction is mine
    assert manager._abstraction is not get_abstraction_buffer()


def test_no_buffer_passed_still_uses_the_global_one(tmp_path: Path) -> None:
    from maxim.utils.structured_logging import get_abstraction_buffer

    assert _manager(tmp_path, None)._abstraction is get_abstraction_buffer()
