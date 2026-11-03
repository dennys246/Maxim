"""Unit tests for logging helper utilities."""

from __future__ import annotations

import logging


class TestLogSwallowedException:
    """Test log_swallowed_exception function."""

    def test_logs_exception_type_and_message(self, caplog):
        """Logs exception type and message."""
        from maxim.utils.logging import log_swallowed_exception

        with caplog.at_level(logging.DEBUG):
            try:
                raise ValueError("test error")
            except ValueError as e:
                log_swallowed_exception(e, operation="test_op")

        assert "ValueError" in caplog.text
        assert "test error" in caplog.text
        assert "test_op" in caplog.text

    def test_includes_context(self, caplog):
        """Includes context dict in log message."""
        from maxim.utils.logging import log_swallowed_exception

        with caplog.at_level(logging.DEBUG):
            try:
                raise RuntimeError("oops")
            except RuntimeError as e:
                log_swallowed_exception(e, operation="risky_call", context={"attempt": 3, "timeout": 5.0})

        assert "attempt=3" in caplog.text
        assert "timeout=5.0" in caplog.text

    def test_respects_log_level(self, caplog):
        """Uses specified log level."""
        from maxim.utils.logging import log_swallowed_exception

        with caplog.at_level(logging.WARNING):
            try:
                raise KeyError("missing")
            except KeyError as e:
                # Default DEBUG level should not appear
                log_swallowed_exception(e, operation="lookup")

        assert "KeyError" not in caplog.text

        with caplog.at_level(logging.WARNING):
            try:
                raise KeyError("missing")
            except KeyError as e:
                log_swallowed_exception(e, operation="lookup", level=logging.WARNING)

        assert "KeyError" in caplog.text

    def test_uses_custom_logger(self, caplog):
        """Uses provided logger."""
        from maxim.utils.logging import log_swallowed_exception

        custom_logger = logging.getLogger("custom.test")

        with caplog.at_level(logging.DEBUG, logger="custom.test"):
            try:
                raise OSError("file error")
            except OSError as e:
                log_swallowed_exception(e, operation="file_op", logger=custom_logger)

        assert "OSError" in caplog.text


class TestLogRecoverableError:
    """Test log_recoverable_error function."""

    def test_logs_operation_and_recovery(self, caplog):
        """Logs operation, exception, and recovery action."""
        from maxim.utils.logging import log_recoverable_error

        with caplog.at_level(logging.WARNING):
            try:
                raise FileNotFoundError("config.yaml")
            except FileNotFoundError as e:
                log_recoverable_error(e, operation="load_config", recovery="using defaults")

        assert "load_config" in caplog.text
        assert "FileNotFoundError" in caplog.text
        assert "using defaults" in caplog.text

    def test_includes_context(self, caplog):
        """Includes context dict in log message."""
        from maxim.utils.logging import log_recoverable_error

        with caplog.at_level(logging.WARNING):
            try:
                raise ConnectionError("timeout")
            except ConnectionError as e:
                log_recoverable_error(
                    e,
                    operation="connect",
                    recovery="retrying",
                    context={"host": "localhost", "port": 8080},
                )

        assert "host='localhost'" in caplog.text
        assert "port=8080" in caplog.text

    def test_logs_at_warning_level(self, caplog):
        """Logs at WARNING level by default."""
        from maxim.utils.logging import log_recoverable_error

        with caplog.at_level(logging.WARNING):
            try:
                raise ValueError("bad input")
            except ValueError as e:
                log_recoverable_error(e, operation="parse", recovery="using default")

        assert any(record.levelno == logging.WARNING for record in caplog.records)


class TestThirdPartyHttpLoggersQuieted:
    """configure_logging demotes httpx/httpcore to WARNING (2026-08-01):
    httpx logs every request at INFO, and the live DoA feed polls the
    robot's REST endpoint at ~6.7 Hz — ~7 console lines/second and a JSONL
    flood. Maxim's own utils/http.py ``http_request`` events remain the
    sanctioned per-call trace."""

    def test_httpx_and_httpcore_are_warning_after_configure(self):
        import logging as _logging

        from maxim.utils.logging import configure_logging

        # Simulate the chatty state: library loggers left at NOTSET/INFO.
        for name in ("httpx", "httpcore"):
            _logging.getLogger(name).setLevel(_logging.NOTSET)
        configure_logging(force=True)
        for name in ("httpx", "httpcore"):
            assert _logging.getLogger(name).getEffectiveLevel() >= _logging.WARNING, (
                f"{name} logger passes INFO — the DoA feed's REST polling will flood the console/JSONL again"
            )


def test_quiet_console_handlers_leaves_the_jsonl_run_log_whole(tmp_path) -> None:
    """#1476: the sim's teardown quiets the terminal, not the JSONL run log (MAXIM_LOG_FILE keeps DEBUG and up), so a
    record saved after shutdown still logs its store digest. Non-terminal handlers (the leader's remote log buffer,
    a plain handler) keep their levels too: raising the buffer leaked WARNING into later tests."""
    import io
    import logging

    from maxim.utils.logging import quiet_console_handlers

    jsonl_path = tmp_path / "run.jsonl"
    console_stream = io.StringIO()
    jsonl = logging.FileHandler(jsonl_path, encoding="utf-8")
    jsonl.setLevel(logging.DEBUG)
    console = logging.StreamHandler(console_stream)
    console.setLevel(logging.INFO)
    buffered: list[str] = []

    class _BufferLike(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            buffered.append(record.getMessage())

    buffer_like = _BufferLike(logging.INFO)
    root, maxim_logger = logging.getLogger(), logging.getLogger("maxim")
    old_level = root.level
    old_handler_levels = {hd: hd.level for hd in root.handlers + maxim_logger.handlers}
    root.addHandler(jsonl)
    root.addHandler(console)
    maxim_logger.addHandler(buffer_like)
    root.setLevel(logging.DEBUG)
    try:
        quiet_console_handlers(logging.WARNING)
        logging.getLogger("maxim.memory.hippocampus_persistence").info("after shutdown")
        jsonl.flush()
        assert "after shutdown" in jsonl_path.read_text(encoding="utf-8")
        assert console_stream.getvalue() == "" and buffered == ["after shutdown"]
        assert (console.level, jsonl.level, buffer_like.level) == (logging.WARNING, logging.DEBUG, logging.INFO)
    finally:
        root.removeHandler(jsonl)
        root.removeHandler(console)
        maxim_logger.removeHandler(buffer_like)
        jsonl.close()
        root.setLevel(old_level)
        for hd, lvl in old_handler_levels.items():
            hd.setLevel(lvl)


def test_the_sim_teardown_quiets_handlers_never_a_logger() -> None:
    import ast
    import inspect

    from maxim.simulation import sim_teardown

    tree = ast.parse(inspect.getsource(sim_teardown._signal_shutdown).lstrip())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(getattr(c.func, "attr", None) == "setLevel" for c in calls)
    assert any(getattr(c.func, "id", None) == "quiet_console_handlers" for c in calls)
