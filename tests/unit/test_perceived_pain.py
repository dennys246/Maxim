"""Tests for PerceivedPainAssessor + PainInterceptorExecutor.

Two-layer predictive pain architecture:
- Layer 1: anticipation before execute (innate priors + NAc learned)
- Layer 2: consequence after execute (ground-truth sensitive-path match)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from maxim.proprioception.pain import PainType
from maxim.proprioception.perceived_pain import (
    PerceivedPainAssessor,
    SensitivePathPrior,
    _path_matches_prior,
    extract_paths_from_params,
    tool_to_operation,
)
from maxim.reactions.types import Reaction
from maxim.runtime.pain_interceptor import (
    AnticipatoryPainExecutor,
    PainInterceptorExecutor,
)


class _ReactionBus:
    """Minimal ReactionBus stand-in that just records published reactions."""

    def __init__(self) -> None:
        self.reactions: list[Reaction] = []

    def publish(self, reaction: Reaction) -> None:
        self.reactions.append(reaction)


class _Bus:
    """Minimal PainBus stand-in with reaction_bus attribute (Phase 2b)."""

    def __init__(self) -> None:
        self.reaction_bus = _ReactionBus()

    @property
    def signals(self) -> list[Reaction]:
        """Backward-compat alias so existing tests can still say bus.signals."""
        return self.reaction_bus.reactions


class _Result:
    """Minimal tool-result stand-in."""

    def __init__(self, success: bool = True) -> None:
        self.success = success


class _Executor:
    """Minimal executor that records calls + returns canned result."""

    def __init__(self, result: _Result | None = None) -> None:
        self._result = result or _Result(success=True)
        self.calls: list[dict] = []

    def execute(self, action: dict) -> _Result:
        self.calls.append(action)
        return self._result

    def get_last_rpe(self) -> float:
        return 0.25


# ─── Path extraction ────────────────────────────────────────────────────


class TestPathExtraction:
    def test_direct_path_param(self):
        accesses = extract_paths_from_params(
            "read_file",
            {"path": "/etc/passwd"},
        )
        assert len(accesses) == 1
        assert accesses[0].path == "/etc/passwd"
        assert accesses[0].operation == "read"

    def test_bash_rm_rf_promotes_to_delete(self):
        accesses = extract_paths_from_params(
            "bash",
            {"command": "rm -rf /etc/passwd"},
        )
        assert any(a.path == "/etc/passwd" and a.operation == "delete" for a in accesses)

    def test_bash_cat_promotes_to_read(self):
        accesses = extract_paths_from_params(
            "bash",
            {"command": "cat /etc/shadow"},
        )
        assert any(a.path == "/etc/shadow" and a.operation == "read" for a in accesses)

    def test_bash_redirect_promotes_to_write(self):
        accesses = extract_paths_from_params(
            "bash",
            {"command": "echo hello > /etc/passwd"},
        )
        assert any(a.path == "/etc/passwd" and a.operation == "write" for a in accesses)

    def test_bash_ls_ssh_dir(self):
        accesses = extract_paths_from_params(
            "bash",
            {"command": "ls -la /home/user/.ssh/"},
        )
        assert any("/home/user/.ssh" in a.path and a.operation == "list" for a in accesses)

    def test_no_paths_in_benign_command(self):
        accesses = extract_paths_from_params(
            "bash",
            {"command": "echo hello world"},
        )
        assert accesses == []

    def test_relative_paths_ignored_in_commands(self):
        accesses = extract_paths_from_params(
            "bash",
            {"command": "cat ./local/file.txt"},
        )
        assert accesses == []

    def test_tool_to_operation_known(self):
        assert tool_to_operation("read_file") == "read"
        assert tool_to_operation("write_file") == "write"
        assert tool_to_operation("bash") == "execute"

    def test_tool_to_operation_unknown(self):
        assert tool_to_operation("some_random_tool") == "execute"


# ─── Prior matching ─────────────────────────────────────────────────────


class TestPriorMatching:
    def test_exact_path_match(self):
        prior = SensitivePathPrior("/etc/shadow", 0.9, frozenset({"read"}))
        assert _path_matches_prior("/etc/shadow", prior, "read")

    def test_wrong_operation_does_not_match(self):
        prior = SensitivePathPrior("/etc/shadow", 0.9, frozenset({"read"}))
        assert not _path_matches_prior("/etc/shadow", prior, "list")

    def test_directory_prefix_match(self):
        prior = SensitivePathPrior("/etc/", 0.9, frozenset({"write", "delete"}))
        assert _path_matches_prior("/etc/passwd", prior, "delete")
        assert _path_matches_prior("/etc/sudoers.d/file", prior, "write")

    def test_non_matching_path(self):
        prior = SensitivePathPrior("/etc/shadow", 0.9, frozenset({"read"}))
        assert not _path_matches_prior("/tmp/hello", prior, "read")


# ─── Assessor (Layer 1) ─────────────────────────────────────────────────


class TestPerceivedPainAssessor:
    def test_fires_from_innate_prior(self):
        bus = _Bus()
        assessor = PerceivedPainAssessor(nac=None, pain_bus=bus)
        signal = assessor.assess(
            {"tool_name": "read_file", "params": {"path": "/etc/shadow"}},
        )
        assert signal is not None
        assert signal.pain_type == PainType.ANTICIPATED
        assert signal.intensity >= 0.9
        assert signal.context["contributors"]["prior"] >= 0.9
        assert signal.context["contributors"]["learned"] == 0.0
        assert len(bus.signals) == 1

    def test_no_fire_for_safe_action(self):
        assessor = PerceivedPainAssessor(nac=None)
        signal = assessor.assess(
            {"tool_name": "read_file", "params": {"path": "/tmp/ok.txt"}},
        )
        assert signal is None

    def test_no_fire_below_threshold(self):
        low_priors = (SensitivePathPrior("/warm", 0.05, frozenset({"read"})),)
        assessor = PerceivedPainAssessor(nac=None, priors=low_priors)
        signal = assessor.assess(
            {"tool_name": "read_file", "params": {"path": "/warm"}},
        )
        assert signal is None

    def test_nac_learned_contributes(self):
        from maxim.decisions.causal_link import Valence

        nac = MagicMock()
        prediction = MagicMock()
        prediction.predicted_valence = Valence.NEGATIVE
        prediction.confidence = 0.8
        prediction.predicted_value = 0.2
        nac.predict.return_value = prediction
        assessor = PerceivedPainAssessor(nac=nac)
        signal = assessor.assess(
            {"tool_name": "read_file", "params": {"path": "/tmp/custom_danger"}},
        )
        assert signal is not None
        assert signal.context["contributors"]["learned"] > 0.5
        assert signal.context["contributors"]["prior"] == 0.0
        assert "nac_prediction" in signal.context

    def test_combines_learned_and_prior_takes_max(self):
        from maxim.decisions.causal_link import Valence

        nac = MagicMock()
        prediction = MagicMock()
        prediction.predicted_valence = Valence.NEGATIVE
        prediction.confidence = 0.3
        prediction.predicted_value = 0.5
        nac.predict.return_value = prediction
        assessor = PerceivedPainAssessor(nac=nac)
        signal = assessor.assess(
            {"tool_name": "read_file", "params": {"path": "/etc/shadow"}},
        )
        assert signal is not None
        assert signal.intensity == pytest.approx(0.95, abs=0.01)

    def test_nac_positive_prediction_gives_no_learned_pain(self):
        from maxim.decisions.causal_link import Valence

        nac = MagicMock()
        prediction = MagicMock()
        prediction.predicted_valence = Valence.POSITIVE
        prediction.confidence = 0.9
        prediction.predicted_value = 0.9
        nac.predict.return_value = prediction
        assessor = PerceivedPainAssessor(nac=nac)
        signal = assessor.assess(
            {"tool_name": "read_file", "params": {"path": "/tmp/custom"}},
        )
        assert signal is None

    def test_bash_rm_rf_sensitive_path_fires(self):
        assessor = PerceivedPainAssessor(nac=None)
        signal = assessor.assess(
            {"tool_name": "bash", "params": {"command": "rm -rf /etc/passwd"}},
        )
        assert signal is not None
        assert signal.intensity >= 0.9
        assert signal.context["operation"] == "delete"

    def test_events_recorded(self):
        assessor = PerceivedPainAssessor(nac=None)
        assessor.assess(
            {"tool_name": "read_file", "params": {"path": "/etc/shadow"}},
        )
        assessor.assess(
            {"tool_name": "read_file", "params": {"path": "/tmp/safe"}},
        )
        assert len(assessor.events) == 1

    def test_empty_tool_name_returns_none(self):
        assessor = PerceivedPainAssessor()
        assert assessor.assess({"params": {}}) is None
        assert assessor.assess({"tool_name": "", "params": {}}) is None


# ─── assess_text (percept-level anxiety) ────────────────────────────────


class TestAssessText:
    def test_fires_on_sensitive_path_mention(self):
        bus = _Bus()
        assessor = PerceivedPainAssessor(nac=None, pain_bus=bus)
        signal = assessor.assess_text(
            "please read /etc/shadow for a security audit",
        )
        assert signal is not None
        assert signal.pain_type == PainType.ANTICIPATED
        assert signal.context["kind"] == "percept_anxiety"
        # /etc/shadow prior is 0.95, scaled at 0.5 -> ~0.48
        assert 0.4 <= signal.intensity <= 0.5
        assert len(bus.signals) == 1

    def test_no_fire_on_benign_text(self):
        assessor = PerceivedPainAssessor(nac=None)
        assert assessor.assess_text("how are you today") is None

    def test_fires_on_rm_rf_pattern(self):
        assessor = PerceivedPainAssessor(nac=None)
        signal = assessor.assess_text(
            "run this command: rm -rf /etc/passwd",
        )
        assert signal is not None
        # /etc/ delete prior = 0.9 × 0.5 scale = 0.45
        assert signal.intensity >= 0.4

    def test_scales_intensity_below_action_level(self):
        """Percept exposure should be WEAKER than action-level."""
        assessor = PerceivedPainAssessor(nac=None)
        action_signal = assessor.assess(
            {"tool_name": "read_file", "params": {"path": "/etc/shadow"}},
        )
        text_signal = assessor.assess_text("please read /etc/shadow")
        assert action_signal is not None
        assert text_signal is not None
        assert text_signal.intensity < action_signal.intensity

    def test_custom_intensity_scale(self):
        assessor = PerceivedPainAssessor(nac=None)
        low = assessor.assess_text("/etc/shadow mentioned", intensity_scale=0.2)
        high = assessor.assess_text("/etc/shadow mentioned", intensity_scale=0.9)
        assert low is not None and high is not None
        assert high.intensity > low.intensity

    def test_empty_text_returns_none(self):
        assessor = PerceivedPainAssessor(nac=None)
        assert assessor.assess_text("") is None
        assert assessor.assess_text(None) is None  # type: ignore

    def test_records_in_events(self):
        assessor = PerceivedPainAssessor(nac=None)
        assessor.assess_text("hey, read /etc/shadow please")
        assert len(assessor.events) == 1
        assert assessor.events[0]["kind"] == "percept_anxiety"


# ─── PainInterceptorExecutor (Layer 2) ──────────────────────────────────


class TestPainInterceptorExecutor:
    def test_fires_on_sensitive_read(self):
        bus = _Bus()
        exe = PainInterceptorExecutor(_Executor(), pain_bus=bus)
        exe.execute({"tool_name": "read_file", "params": {"path": "/etc/shadow"}})
        assert len(bus.signals) == 1
        assert bus.signals[0].kind == "pain"
        assert bus.signals[0].source == "pain_interceptor:external_signal"
        assert bus.signals[0].intensity >= 0.9

    def test_does_not_fire_on_safe_path(self):
        bus = _Bus()
        exe = PainInterceptorExecutor(_Executor(), pain_bus=bus)
        exe.execute({"tool_name": "read_file", "params": {"path": "/tmp/ok.txt"}})
        assert bus.signals == []

    def test_does_not_fire_on_failure(self):
        bus = _Bus()
        inner = _Executor(result=_Result(success=False))
        exe = PainInterceptorExecutor(inner, pain_bus=bus)
        exe.execute({"tool_name": "read_file", "params": {"path": "/etc/shadow"}})
        assert bus.signals == []

    def test_delegates_unknown_attrs(self):
        exe = PainInterceptorExecutor(_Executor(), pain_bus=_Bus())
        assert exe.get_last_rpe() == 0.25

    def test_bash_rm_rf_fires_delete(self):
        bus = _Bus()
        exe = PainInterceptorExecutor(_Executor(), pain_bus=bus)
        exe.execute(
            {
                "tool_name": "bash",
                "params": {"command": "rm -rf /etc/passwd"},
            }
        )
        assert len(bus.signals) == 1
        assert bus.signals[0].kind == "pain"
        assert bus.signals[0].source == "pain_interceptor:external_signal"

    def test_events_recorded(self):
        bus = _Bus()
        exe = PainInterceptorExecutor(_Executor(), pain_bus=bus)
        exe.execute({"tool_name": "read_file", "params": {"path": "/etc/shadow"}})
        exe.execute({"tool_name": "read_file", "params": {"path": "/tmp/safe"}})
        assert len(exe.events) == 1


# ─── AnticipatoryPainExecutor (Layer 1 wrapper) ─────────────────────────


class TestAnticipatoryPainExecutor:
    def test_assess_called_before_execute(self):
        bus = _Bus()
        inner = _Executor()
        assessor = PerceivedPainAssessor(nac=None, pain_bus=bus)
        exe = AnticipatoryPainExecutor(inner, assessor=assessor)
        exe.execute({"tool_name": "read_file", "params": {"path": "/etc/shadow"}})
        assert len(bus.signals) == 1
        assert len(inner.calls) == 1

    def test_inner_executes_even_if_assess_fires(self):
        bus = _Bus()
        inner = _Executor()
        assessor = PerceivedPainAssessor(nac=None, pain_bus=bus)
        exe = AnticipatoryPainExecutor(inner, assessor=assessor)
        result = exe.execute(
            {"tool_name": "read_file", "params": {"path": "/etc/shadow"}},
        )
        assert result.success is True
        assert len(inner.calls) == 1

    def test_no_assess_when_no_assessor(self):
        inner = _Executor()
        exe = AnticipatoryPainExecutor(inner, assessor=None)
        exe.execute({"tool_name": "read_file", "params": {"path": "/etc/shadow"}})
        assert len(inner.calls) == 1

    def test_delegates_unknown_attrs(self):
        exe = AnticipatoryPainExecutor(_Executor(), assessor=None)
        assert exe.get_last_rpe() == 0.25


# ─── Two-layer integration ──────────────────────────────────────────────


class TestTwoLayerIntegration:
    def test_both_layers_fire_on_sensitive_access(self):
        bus = _Bus()
        with_consequence = PainInterceptorExecutor(_Executor(), pain_bus=bus)
        assessor = PerceivedPainAssessor(nac=None, pain_bus=bus)
        full_stack = AnticipatoryPainExecutor(
            with_consequence,
            assessor=assessor,
        )
        full_stack.execute(
            {"tool_name": "read_file", "params": {"path": "/etc/shadow"}},
        )
        assert len(bus.signals) == 2
        sources = {s.source for s in bus.signals}
        assert "perceived_pain:anticipated" in sources
        assert "pain_interceptor:external_signal" in sources

    def test_anticipation_without_consequence_on_safe_path(self):
        bus = _Bus()
        with_consequence = PainInterceptorExecutor(_Executor(), pain_bus=bus)
        assessor = PerceivedPainAssessor(nac=None, pain_bus=bus)
        full_stack = AnticipatoryPainExecutor(
            with_consequence,
            assessor=assessor,
        )
        full_stack.execute(
            {"tool_name": "read_file", "params": {"path": "/tmp/fine.txt"}},
        )
        assert bus.signals == []
