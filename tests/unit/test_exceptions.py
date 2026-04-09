"""Unit tests for custom exception hierarchy."""

from __future__ import annotations

import pytest


class TestMaximError:
    """Test base exception class."""

    def test_basic_creation(self):
        """Can create with just a message."""
        from maxim.exceptions import MaximError

        err = MaximError("Something went wrong")

        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.context == {}

    def test_with_context(self):
        """Can create with context dict."""
        from maxim.exceptions import MaximError

        err = MaximError("Failed", context={"operation": "connect", "attempt": 3})

        assert err.context == {"operation": "connect", "attempt": 3}

    def test_is_exception(self):
        """Inherits from Exception."""
        from maxim.exceptions import MaximError

        assert issubclass(MaximError, Exception)

        with pytest.raises(MaximError):
            raise MaximError("test")


class TestConnectionErrors:
    """Test connection-related exceptions."""

    def test_connection_error_hierarchy(self):
        """Connection errors inherit correctly."""
        from maxim.exceptions import (
            MaximConnectionError,
            ConnectionLostError,
            ConnectionTimeoutError,
            MaximError,
            ReconnectionFailedError,
        )

        assert issubclass(MaximConnectionError, MaximError)
        assert issubclass(ConnectionTimeoutError, MaximConnectionError)
        assert issubclass(ConnectionLostError, MaximConnectionError)
        assert issubclass(ReconnectionFailedError, MaximConnectionError)

    def test_catch_all_connection_errors(self):
        """Can catch all connection errors with base class."""
        from maxim.exceptions import MaximConnectionError, ConnectionLostError

        with pytest.raises(MaximConnectionError):
            raise ConnectionLostError("Connection dropped")


class TestToolExecutionErrors:
    """Test tool execution exceptions."""

    def test_tool_error_with_details(self):
        """Tool error captures tool name and params."""
        from maxim.exceptions import ToolExecutionError

        err = ToolExecutionError(
            "Tool failed",
            tool_name="move_head",
            tool_params={"x": 0.5, "y": 0.3},
        )

        assert err.tool_name == "move_head"
        assert err.tool_params == {"x": 0.5, "y": 0.3}
        assert err.context["tool_name"] == "move_head"

    def test_tool_error_hierarchy(self):
        """Tool errors inherit correctly."""
        from maxim.exceptions import (
            MaximError,
            ToolExecutionError,
            ToolNotFoundError,
            ToolTimeoutError,
            ToolValidationError,
        )

        assert issubclass(ToolExecutionError, MaximError)
        assert issubclass(ToolNotFoundError, ToolExecutionError)
        assert issubclass(ToolValidationError, ToolExecutionError)
        assert issubclass(ToolTimeoutError, ToolExecutionError)


class TestModelErrors:
    """Test model-related exceptions."""

    def test_model_error_with_name(self):
        """Model error captures model name."""
        from maxim.exceptions import ModelLoadError

        err = ModelLoadError("Failed to load", model_name="yolo-v8")

        assert err.model_name == "yolo-v8"
        assert err.context["model_name"] == "yolo-v8"

    def test_model_error_hierarchy(self):
        """Model errors inherit correctly."""
        from maxim.exceptions import (
            MaximError,
            ModelConfigError,
            ModelError,
            ModelInferenceError,
            ModelLoadError,
        )

        assert issubclass(ModelError, MaximError)
        assert issubclass(ModelLoadError, ModelError)
        assert issubclass(ModelInferenceError, ModelError)
        assert issubclass(ModelConfigError, ModelError)


class TestPlanningErrors:
    """Test planning-related exceptions."""

    def test_constraint_violation(self):
        """Constraint violation captures constraint name."""
        from maxim.exceptions import ConstraintViolation

        err = ConstraintViolation(
            "Movement exceeds limits",
            constraint_name="head_pitch_limit",
        )

        assert err.constraint_name == "head_pitch_limit"
        assert err.context["constraint_name"] == "head_pitch_limit"

    def test_planning_error_hierarchy(self):
        """Planning errors inherit correctly."""
        from maxim.exceptions import (
            ConstraintViolation,
            MaximError,
            NoValidPlanError,
            PlanningError,
        )

        assert issubclass(PlanningError, MaximError)
        assert issubclass(ConstraintViolation, PlanningError)
        assert issubclass(NoValidPlanError, PlanningError)


class TestHardwareErrors:
    """Test hardware-related exceptions."""

    def test_hardware_error_hierarchy(self):
        """Hardware errors inherit correctly."""
        from maxim.exceptions import (
            AudioError,
            CameraError,
            HardwareError,
            MaximError,
            MotorError,
            SensorError,
        )

        assert issubclass(HardwareError, MaximError)
        assert issubclass(MotorError, HardwareError)
        assert issubclass(SensorError, HardwareError)
        assert issubclass(CameraError, HardwareError)
        assert issubclass(AudioError, HardwareError)


class TestCatchAllMaximError:
    """Test catching all Maxim errors."""

    def test_catch_any_maxim_error(self):
        """Can catch any Maxim error with MaximError."""
        from maxim.exceptions import (
            ConnectionLostError,
            MaximError,
            ModelLoadError,
            ToolNotFoundError,
        )

        errors = [
            ConnectionLostError("lost"),
            ToolNotFoundError("not found"),
            ModelLoadError("failed"),
        ]

        for error in errors:
            with pytest.raises(MaximError):
                raise error
