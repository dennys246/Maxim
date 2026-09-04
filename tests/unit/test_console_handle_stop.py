"""``MaximHandle.stop()`` wait bounds — both waits are caller-chosen.

A machine's stop grace or an operator's "end session" has its own deadline;
``stop()`` must let such a caller shorten BOTH bounded waits (the campaign
lock and the talk-loop join) without changing the lifespan's long defaults.
Before this, the talk-loop join was a fixed 20 s inside ``_stop_talk_loop``,
so a hung talk loop cost every caller 20 s regardless of its own deadline.
No LLM, no bio-stack: the handle is built bare and the instance is a fake.
"""

from __future__ import annotations

import threading
import time

from maxim.console.handle import MaximHandle


class _FakeInstance:
    def __init__(self):
        self.shutdowns: list[str] = []

    def shutdown(self, *, consolidation: str) -> None:
        self.shutdowns.append(consolidation)


def _bare_handle(*, talk_thread: threading.Thread | None = None) -> MaximHandle:
    h = MaximHandle.__new__(MaximHandle)
    h.agent_id = "test_agent"
    h._stopped = False
    h._campaign_lock = threading.Lock()
    h._talk_lock = threading.Lock()
    h._talk_bridge = None
    h._talk_thread = talk_thread
    h._talk_stop = None
    h._talk_worker = None
    h.instance = _FakeInstance()
    return h


def _hung_thread(release: threading.Event) -> threading.Thread:
    t = threading.Thread(target=release.wait, name="hung-talk-loop", daemon=True)
    t.start()
    return t


class TestStopWaits:
    def test_talk_join_is_bounded_by_the_caller(self):
        release = threading.Event()
        h = _bare_handle(talk_thread=_hung_thread(release))
        try:
            t0 = time.monotonic()
            h.stop(talk_join_s=0.05)
            elapsed = time.monotonic() - t0
        finally:
            release.set()
        assert elapsed < 2.0, f"stop() waited {elapsed:.2f}s for a hung talk loop despite talk_join_s=0.05"
        assert h.instance.shutdowns == ["full"]  # proceeded LOUDLY, still consolidated
        assert h._stopped

    def test_campaign_wait_is_bounded_by_the_caller(self):
        h = _bare_handle()
        h._campaign_lock.acquire()  # a "live campaign" that never ends
        t0 = time.monotonic()
        h.stop(campaign_wait_s=0.05, talk_join_s=0.05)
        elapsed = time.monotonic() - t0
        assert elapsed < 2.0
        assert h.instance.shutdowns == ["full"]

    def test_defaults_are_forwarded_to_the_talk_loop_join(self, monkeypatch):
        seen: dict[str, float] = {}
        h = _bare_handle()

        def _spy(self, *, join_s: float = 20.0, required: bool = False) -> None:
            seen["join_s"] = join_s

        monkeypatch.setattr(MaximHandle, "_stop_talk_loop", _spy)
        h.stop()
        assert seen == {"join_s": 20.0}
        h2 = _bare_handle()
        h2.stop(talk_join_s=3.0)
        assert seen == {"join_s": 3.0}

    def test_second_stop_is_a_noop(self):
        h = _bare_handle()
        h.stop(talk_join_s=0.05)
        h.stop(talk_join_s=0.05)
        assert h.instance.shutdowns == ["full"]
