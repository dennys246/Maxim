"""Guards for utils/logging.py::log_swallowed_exception (fail-loud Stage 1).

Pins the contract measurement_path_fail_loud.md Stage 2 depends on: every
distinct swallow site logs a structured ``swallowed_exception`` event at
WARNING exactly once per process (DEBUG after), the event carries the
traceback, the helper can never raise, and the LEGACY explicit form
(exc + operation) keeps its pre-Stage-1 semantics byte-for-byte.
"""

from __future__ import annotations

import logging

import pytest

from maxim.utils.logging import _reset_swallow_seen_for_test, log_swallowed_exception


@pytest.fixture(autouse=True)
def _fresh_dedup():
    _reset_swallow_seen_for_test()
    yield
    _reset_swallow_seen_for_test()


def _swallow_once(log: logging.Logger) -> None:
    try:
        raise ValueError("boom")
    except Exception:
        log_swallowed_exception(logger=log)


class TestStage1ZeroArgForm:
    def test_first_fire_warns_then_debug(self, caplog):
        log = logging.getLogger("test.swallowed")
        with caplog.at_level(logging.DEBUG, logger="test.swallowed"):
            _swallow_once(log)
            _swallow_once(log)
        records = [r for r in caplog.records if "swallowed_exception" in r.message]
        assert len(records) == 2
        assert records[0].levelno == logging.WARNING
        assert records[1].levelno == logging.DEBUG

    def test_event_carries_site_and_traceback(self, caplog):
        log = logging.getLogger("test.swallowed.site")
        with caplog.at_level(logging.WARNING, logger="test.swallowed.site"):
            _swallow_once(log)
        rec = next(r for r in caplog.records if "swallowed_exception" in r.message)
        assert "test_swallowed_log.py:_swallow_once:" in rec.getMessage()
        assert rec.exc_info is not None and rec.exc_info[0] is ValueError

    def test_distinct_sites_each_warn(self, caplog):
        log = logging.getLogger("test.swallowed.two")

        def swallow_elsewhere():
            try:
                raise KeyError("x")
            except Exception:
                log_swallowed_exception(logger=log)

        with caplog.at_level(logging.DEBUG, logger="test.swallowed.two"):
            _swallow_once(log)
            swallow_elsewhere()
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warns) == 2  # two distinct sites -> two first-fires

    def test_never_raises_even_with_broken_logger(self):
        class Broken(logging.Logger):
            def log(self, *a, **k):  # noqa: A003
                raise RuntimeError("logger is broken")

        broken = Broken("broken")
        try:
            raise ValueError("boom")
        except Exception:
            log_swallowed_exception(logger=broken)  # must not propagate either exception

    def test_outside_except_block_is_harmless(self, caplog):
        log = logging.getLogger("test.swallowed.outside")
        with caplog.at_level(logging.WARNING, logger="test.swallowed.outside"):
            log_swallowed_exception(logger=log)  # no in-flight exception: logs, no crash
        assert any("swallowed_exception" in r.message for r in caplog.records)


class TestLegacyExplicitForm:
    """The pre-Stage-1 callers (memory_agent, media_loop, selfy) must be untouched."""

    def test_debug_default_no_dedup_escalation(self, caplog):
        log = logging.getLogger("test.swallowed.legacy")
        with caplog.at_level(logging.DEBUG, logger="test.swallowed.legacy"):
            for _ in range(2):
                try:
                    raise OSError("disk")
                except OSError as e:
                    log_swallowed_exception(e, operation="risky_op", context={"k": 1}, logger=log)
        records = [r for r in caplog.records if "Swallowed" in r.message]
        assert len(records) == 2
        assert all(r.levelno == logging.DEBUG for r in records)  # never escalates
        assert "Swallowed OSError in risky_op" in records[0].getMessage()
        assert "[k=1]" in records[0].getMessage()

    def test_explicit_level_respected(self, caplog):
        log = logging.getLogger("test.swallowed.legacy2")
        with caplog.at_level(logging.DEBUG, logger="test.swallowed.legacy2"):
            try:
                raise OSError("disk")
            except OSError as e:
                log_swallowed_exception(e, operation="op", logger=log, level=logging.INFO)
        rec = next(r for r in caplog.records if "Swallowed" in r.message)
        assert rec.levelno == logging.INFO
