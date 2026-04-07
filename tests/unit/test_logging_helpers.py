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
