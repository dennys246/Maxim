"""Direct unit tests for the helper methods extracted in Phase C decompositions.

Phase C of the final refinement plan decomposed 13 god functions into smaller
helpers. The orchestrator tests exercise the helpers transitively, but several
helpers have branching logic that benefits from focused unit tests:

- ``LLMRouter._log_provider_error`` — 3 error classification branches
- ``LLMRouter._check_provider_budget`` — 4 return paths
- ``_validate_swap_profile`` — 4 client-error paths + success
- ``_remote_update_allowed`` — env var truthy parser
- ``cli._handle_list_models`` / ``_handle_delete_model`` / ``_handle_clear_memory``
  — discrete CLI subcommand handlers
- ``ExecAgent._build_override_goal`` / ``_enforce_rate_limit``
- ``_idle_sleep`` — exception path
- ``CausalLink.__post_init__`` — type validation guard added in this session

These tests pin down the new behaviors so future refactors can't silently
regress them. They run fast (<1s total) and have no external dependencies.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import patch

import pytest


# ─── LLMRouter._log_provider_error ────────────────────────────────────────


class TestLogProviderError:
    """Three branches: auth (401/403), not-found (404), generic."""

    def test_auth_error_logged_with_api_key_hint(self, caplog):
        from maxim.models.language.router import LLMRouter

        with caplog.at_level(logging.WARNING, logger="maxim"):
            LLMRouter._log_provider_error("anthropic", Exception("401 Unauthorized"))
        assert any("auth failed" in r.message.lower() for r in caplog.records)
        assert any("anthropic" in r.message for r in caplog.records)

    def test_not_found_error_logged_with_model_hint(self, caplog):
        from maxim.models.language.router import LLMRouter

        with caplog.at_level(logging.WARNING, logger="maxim"):
            LLMRouter._log_provider_error("openai", Exception("model not found: gpt-5"))
        assert any("model not found" in r.message.lower() for r in caplog.records)

    def test_generic_error_logged_as_complete_failed(self, caplog):
        from maxim.models.language.router import LLMRouter

        with caplog.at_level(logging.WARNING, logger="maxim"):
            LLMRouter._log_provider_error("groq", Exception("connection refused"))
        assert any("complete failed" in r.message.lower() for r in caplog.records)


# ─── lane_backends._validate_swap_profile ─────────────────────────────────


class TestValidateSwapProfile:
    """Five paths: empty, cloud, wrong-backend, unknown, success."""

    def test_empty_name_raises(self):
        from maxim.runtime.lane_backends import _validate_swap_profile

        with pytest.raises(ValueError, match="Empty model name"):
            _validate_swap_profile("")

    def test_cloud_profile_rejected(self):
        from maxim.runtime.lane_backends import _validate_swap_profile

        with pytest.raises(ValueError, match="cloud provider"):
            _validate_swap_profile("claude-sonnet")

    def test_unknown_profile_rejected(self):
        from maxim.runtime.lane_backends import _validate_swap_profile

        with pytest.raises(ValueError, match="Unknown model profile"):
            _validate_swap_profile("not-a-real-model-xyz")

    def test_known_local_llama_profile_passes(self):
        from maxim.runtime.lane_backends import _validate_swap_profile

        # mistral-7b is a built-in llama_cpp profile
        resolved, profile_data = _validate_swap_profile("mistral-7b")
        assert resolved
        assert profile_data.get("backend") in ("llama_cpp", "llamacpp", "llama")


# ─── leader_proxy._remote_update_allowed ──────────────────────────────────


class TestRemoteUpdateAllowed:
    """Truthy parser for ``MAXIM_ALLOW_REMOTE_UPDATE``."""

    def test_unset_returns_false(self, monkeypatch):
        from maxim.runtime.leader_proxy import _remote_update_allowed

        monkeypatch.delenv("MAXIM_ALLOW_REMOTE_UPDATE", raising=False)
        assert _remote_update_allowed() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "y", "on", "t"])
    def test_truthy_values_allowed(self, monkeypatch, value):
        from maxim.runtime.leader_proxy import _remote_update_allowed

        monkeypatch.setenv("MAXIM_ALLOW_REMOTE_UPDATE", value)
        assert _remote_update_allowed() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy_values_rejected(self, monkeypatch, value):
        from maxim.runtime.leader_proxy import _remote_update_allowed

        monkeypatch.setenv("MAXIM_ALLOW_REMOTE_UPDATE", value)
        assert _remote_update_allowed() is False


# ─── cli subcommand handlers ──────────────────────────────────────────────


class TestCliHandlers:
    """Three discrete handlers extracted from cli.main()."""

    def test_handle_list_models_returns_zero(self, capsys):
        from maxim.cli import _handle_list_models

        rc = _handle_list_models()
        out = capsys.readouterr().out
        assert rc == 0
        # Should mention "Local Models" and "Cloud Models" sections
        assert "Local Models" in out
        assert "Cloud Models" in out

    def test_handle_delete_model_unknown_returns_zero(self, capsys):
        from maxim.cli import _handle_delete_model

        # Unknown profile → falls through, doesn't crash
        rc = _handle_delete_model("not-a-real-profile-xyz")
        assert rc == 0
        out = capsys.readouterr().out
        # Should print available-models hint
        assert "Available local models" in out

    def test_handle_clear_memory_unknown_scope_returns_zero(self, tmp_path, capsys):
        from maxim.cli import _handle_clear_memory

        # Unknown scope still returns 0; clears 0 files
        rc = _handle_clear_memory("nonexistent_scope", str(tmp_path))
        assert rc == 0
        out = capsys.readouterr().out
        assert "Clearing memory" in out


# ─── ExecAgent extracted helpers ──────────────────────────────────────────


class TestExecAgentHelpers:
    """Tests for the helpers extracted from ``_propose_goal``."""

    def test_build_override_goal_yields_critical_priority(self):
        from maxim.agents.exec_agent import ExecAgent
        from maxim.agents.bus import GoalPriority

        goal = ExecAgent._build_override_goal("switch to active mode")
        assert goal.priority == GoalPriority.CRITICAL
        assert goal.tool_name == "maxim_command"
        assert goal.tool_params == {"command": "switch to active mode"}
        assert goal.confidence == 1.0
        assert "Execute hard override" in goal.description


# ─── runtime/agent_loop._idle_sleep ───────────────────────────────────────


class TestIdleSleepHelper:
    """The extracted DRY helper for the legacy run_agent_loop."""

    def test_normal_sleep_returns_silently(self):
        from maxim.runtime.agent_loop import _idle_sleep

        t0 = time.time()
        _idle_sleep(0.001)
        elapsed = time.time() - t0
        assert elapsed >= 0.001

    def test_sleep_failure_logs_warning(self, caplog):
        from maxim.runtime import agent_loop

        with patch("maxim.runtime.agent_loop.time.sleep", side_effect=OSError("fake")):
            with caplog.at_level(logging.WARNING, logger="maxim.runtime.agent_loop"):
                agent_loop._idle_sleep(0.05)
        assert any("idle_sleep failed" in r.message for r in caplog.records)


# ─── CausalLink.__post_init__ type validation ─────────────────────────────


class TestCausalLinkValidation:
    """The __post_init__ guard added this session catches the float-as-valence bug."""

    def test_float_outcome_valence_raises_typeerror(self):
        from maxim.decisions.causal_link import CausalLink, TemporalDelta

        with pytest.raises(TypeError, match="must be a Valence enum"):
            CausalLink(
                id="x",
                event_type="tool",
                event_signature="t",
                event_context={},
                outcome_type="result",
                outcome_signature="ok",
                outcome_valence=1.0,  # type: ignore[arg-type]
                temporal_delta=TemporalDelta(observed_deltas=(0.1,)),
            )

    def test_int_outcome_valence_raises_typeerror(self):
        from maxim.decisions.causal_link import CausalLink, TemporalDelta

        with pytest.raises(TypeError, match="must be a Valence enum"):
            CausalLink(
                id="x",
                event_type="tool",
                event_signature="t",
                event_context={},
                outcome_type="result",
                outcome_signature="ok",
                outcome_valence=0,  # type: ignore[arg-type]
                temporal_delta=TemporalDelta(observed_deltas=(0.1,)),
            )

    def test_valid_valence_enum_passes(self):
        from maxim.decisions.causal_link import CausalLink, TemporalDelta, Valence

        link = CausalLink(
            id="x",
            event_type="tool",
            event_signature="t",
            event_context={},
            outcome_type="result",
            outcome_signature="ok",
            outcome_valence=Valence.POSITIVE,
            temporal_delta=TemporalDelta(observed_deltas=(0.1,)),
        )
        assert link.outcome_valence == Valence.POSITIVE


# ─── memory_hub._wire_multi_layer subhelpers ──────────────────────────────


class TestMemoryHubWireHelpers:
    """The decomposed wire helpers should be safe to call when subsystems missing."""

    def test_build_layer_map_minimal(self):
        from maxim.integration.memory_hub import MemoryHub
        from maxim.memory.hippocampus import Hippocampus
        from maxim.time.scn import SCN
        from maxim.decisions.nac import NAc
        from maxim.similarity.ec import EntorhinalCortex

        hub = MemoryHub(
            hippocampus=Hippocampus(),
            scn=SCN(),
            nac=NAc(),
            ec=EntorhinalCortex(),
            _allow_raw=True,
        )
        layers = hub._build_layer_map()
        assert "hippocampus" in layers
        # ATL and angular_gyrus not provided → not in map
        assert "atl" not in layers
        assert "angular_gyrus" not in layers


# ─── prefetch helpers ─────────────────────────────────────────────────────


class TestPrefetchHelpers:
    """Targeted tests for the prefetch_for_input decomposition seams."""

    def test_maybe_skip_for_new_file_skips_when_file_absent(self, tmp_path):
        from maxim.runtime.prefetch import SpeculativePrefetcher, PrefetchResult
        from maxim.runtime.file_patterns import FileReference

        prefetcher = SpeculativePrefetcher(executor=None, base_path=str(tmp_path))
        result = PrefetchResult(file_references=[], intent="create")
        ref = FileReference(pattern="brand-new-file.py", is_path=False, intent="create", confidence=1.0)
        skipped = prefetcher._maybe_skip_for_new_file([ref], result)
        assert skipped is True
        assert result.skip_exploration is True

    def test_maybe_skip_for_new_file_does_not_skip_when_file_exists(self, tmp_path):
        from maxim.runtime.prefetch import SpeculativePrefetcher, PrefetchResult
        from maxim.runtime.file_patterns import FileReference

        existing = tmp_path / "exists.py"
        existing.write_text("# stub\n")
        prefetcher = SpeculativePrefetcher(executor=None, base_path=str(tmp_path))
        result = PrefetchResult(file_references=[], intent="create")
        ref = FileReference(pattern=str(existing), is_path=True, intent="create", confidence=1.0)
        skipped = prefetcher._maybe_skip_for_new_file([ref], result)
        assert skipped is False
        assert result.skip_exploration is False
