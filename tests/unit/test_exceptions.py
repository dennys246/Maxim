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


class TestKeywordOnlyExceptionsRoundTrip:
    """Exceptions whose ``__init__`` takes keyword-only fields pickle and copy faithfully (#1621 delta review):
    the default ``BaseException.__reduce__`` calls ``cls(*args)``, which a keyword-only ``__init__`` refuses with a
    ``TypeError``, so such an error could not cross a process boundary or be copied."""

    @staticmethod
    def _cases() -> list[BaseException]:
        from pathlib import Path

        from maxim.exceptions import CampaignUnloadable, KeyFileUnusable
        from maxim.tunnel.keys import LeaderKeyError, LeaderKeyMissing, LocalServerKeyError

        key = Path("/tmp/k")
        return [
            CampaignUnloadable(campaign="c.yaml", reason="c.yaml: percept 1 has no cli_input line"),
            KeyFileUnusable(path=key, reason="mode 0644", fix_hint="chmod 600"),
            LeaderKeyError(path=key, reason="empty"),  # a subclass with its own keyword-only __init__
            LeaderKeyMissing(path=key),  # narrows the signature to path= only
            LocalServerKeyError(path=key, reason="empty"),  # rewrites message and args after __init__
        ]

    @pytest.mark.parametrize("how", ["pickle", "copy", "deepcopy"])
    def test_round_trip_keeps_type_message_and_fields(self, how: str) -> None:
        import copy
        import pickle

        rebuild = {
            "pickle": lambda e: pickle.loads(pickle.dumps(e)),
            "copy": copy.copy,
            "deepcopy": copy.deepcopy,
        }[how]
        for error in self._cases():
            again = rebuild(error)
            assert type(again) is type(error)
            assert str(again) == str(error) and again.args == error.args
            assert again.__dict__ == error.__dict__

    def test_campaign_unloadable_names_its_path_once(self) -> None:
        from maxim.exceptions import CampaignUnloadable

        named = CampaignUnloadable(campaign="c.yaml", reason="Scenario c.yaml: bad")
        assert str(named).count("c.yaml") == 1 and str(named).endswith(": bad")
        unnamed = CampaignUnloadable(campaign="c.yaml", reason="percept 0 is not a mapping")
        assert str(unnamed).count("c.yaml") == 1
        # load_scenario formats Path(campaign), which drops "./": still printed once.
        dotted = CampaignUnloadable(campaign="./x.yaml", reason="Scenario x.yaml: bad")
        assert str(dotted) == "research campaign ./x.yaml cannot be loaded: bad"
        # load_campaign_turns labels its strict errors "<path>: " with the path as given.
        for campaign in ("./notturn.yaml", "empty.yaml", "/abs/dir/c.yaml"):
            turns = CampaignUnloadable(campaign=campaign, reason=f"{campaign}: percept 0 has no cli_input line")
            assert str(turns) == f"research campaign {campaign} cannot be loaded: percept 0 has no cli_input line"
        # ...and through the real loader, for a file with no turn at all.
        import tempfile
        from pathlib import Path as _P

        from maxim.simulation.experiment import load_campaign_turns

        with tempfile.TemporaryDirectory() as tmp:
            bad = _P(tmp) / "pain.yaml"
            bad.write_text("percepts:\n  - type: pain_signal\n    intensity: 0.5\n")
            try:
                load_campaign_turns(str(bad), strict=True)
            except ValueError as exc:
                real = CampaignUnloadable(campaign=str(bad), reason=str(exc))
                assert str(real).count(str(bad)) == 1, str(real)
            else:  # pragma: no cover - the loader must refuse a pain-only file
                raise AssertionError("strict load accepted a scenario with no cli_input line")
        # A short name that happens to occur inside the reason is still named.
        short = CampaignUnloadable(campaign="a", reason="percept 0 has a bad value")
        assert str(short).startswith("research campaign a cannot be loaded:")
