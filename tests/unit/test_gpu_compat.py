"""Unit tests for GPU compatibility utilities."""

from __future__ import annotations


class TestEnvFlag:
    """Test environment flag parsing."""

    def test_env_flag_true_values(self):
        """True values are parsed correctly."""
        from maxim.utils.gpu_compat import env_flag
        import os

        true_values = ["1", "true", "t", "yes", "y", "on", "TRUE", "True", "YES"]
        for val in true_values:
            os.environ["TEST_FLAG"] = val
            assert env_flag("TEST_FLAG") is True, f"Failed for {val}"

    def test_env_flag_false_values(self):
        """False values are parsed correctly."""
        from maxim.utils.gpu_compat import env_flag
        import os

        false_values = ["0", "false", "f", "no", "n", "off", "FALSE", "False", "NO"]
        for val in false_values:
            os.environ["TEST_FLAG"] = val
            assert env_flag("TEST_FLAG") is False, f"Failed for {val}"

    def test_env_flag_default(self):
        """Default value is used when not set."""
        from maxim.utils.gpu_compat import env_flag
        import os

        os.environ.pop("NONEXISTENT_FLAG", None)
        assert env_flag("NONEXISTENT_FLAG", True) is True
        assert env_flag("NONEXISTENT_FLAG", False) is False


class TestIsConnectionError:
    """Test connection error detection."""

    def test_detects_lost_connection(self):
        """Detects 'lost connection' errors."""
        from maxim.utils.gpu_compat import is_connection_error

        assert is_connection_error("Lost connection to robot") is True
        assert is_connection_error("LOST CONNECTION") is True

    def test_detects_disconnected(self):
        """Detects 'disconnected' errors."""
        from maxim.utils.gpu_compat import is_connection_error

        assert is_connection_error("Robot disconnected") is True

    def test_detects_timeout(self):
        """Detects timeout errors."""
        from maxim.utils.gpu_compat import is_connection_error

        assert is_connection_error("Connection timeout") is True
        assert is_connection_error("Request timed out") is True

    def test_detects_connection_refused(self):
        """Detects connection refused/reset errors."""
        from maxim.utils.gpu_compat import is_connection_error

        assert is_connection_error("Connection refused") is True
        assert is_connection_error("Connection reset by peer") is True
        assert is_connection_error("Connection closed") is True

    def test_detects_channel_closed(self):
        """Detects channel closed (Dynamixel errors)."""
        from maxim.utils.gpu_compat import is_connection_error

        assert is_connection_error("channel closed unexpectedly") is True

    def test_ignores_non_connection_errors(self):
        """Non-connection errors return False."""
        from maxim.utils.gpu_compat import is_connection_error

        assert is_connection_error("File not found") is False
        assert is_connection_error("Invalid parameter") is False
        assert is_connection_error("") is False
        assert is_connection_error(None) is False


class TestGPUAvailable:
    """Test GPU availability detection."""

    def test_returns_bool(self):
        """is_gpu_available returns a boolean."""
        from maxim.utils.gpu_compat import is_gpu_available

        result = is_gpu_available()
        assert isinstance(result, bool)
