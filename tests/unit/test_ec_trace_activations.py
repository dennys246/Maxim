"""Unit tests for the ``MAXIM_EC_TRACE_ACTIVATIONS`` instrumentation.

Added in 0.9.1 (release_0_9_1.md Stage 0d). The env var gates per-tick
``sim_ec_activation`` events emitted from
``EntorhinalCortex.pattern_complete_or_separate`` to drive the Roy-4
post-hoc co-activation analysis (cross_modal_substrate_binding.md Stage 1
validation).

These tests cover the env-var resolver, JSONL event shape, modality_tag
derivation, and multi-agent isolation via the ``_current_agent_id``
contextvar.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from maxim.similarity.ec import (
    EntorhinalCortex,
    _EC_TRACE_MODALITY_TAG_MAP,
    _ec_trace_enabled,
    _emit_ec_activation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Resolver — MAXIM_EC_TRACE_ACTIVATIONS interpretation
# ─────────────────────────────────────────────────────────────────────────────


class TestEcTraceEnabled:
    """Env-var truthiness resolver semantics."""

    def test_unset_returns_false(self):
        # Autouse fixture _isolate_maxim_ec_trace_env scrubs pre-test.
        assert _ec_trace_enabled() is False

    def test_one_returns_true(self, monkeypatch):
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        assert _ec_trace_enabled() is True

    def test_zero_returns_false(self, monkeypatch):
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "0")
        assert _ec_trace_enabled() is False

    def test_empty_string_returns_false(self, monkeypatch):
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "")
        assert _ec_trace_enabled() is False

    @pytest.mark.parametrize("falsy", ["false", "False", "FALSE", "no", "off", " 0 "])
    def test_falsy_strings(self, monkeypatch, falsy):
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", falsy)
        assert _ec_trace_enabled() is False

    @pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE"])
    def test_truthy_strings(self, monkeypatch, truthy):
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", truthy)
        assert _ec_trace_enabled() is True


# ─────────────────────────────────────────────────────────────────────────────
# Modality tag mapping — Roy-4 expects {linguistic, drive, sensor}
# ─────────────────────────────────────────────────────────────────────────────


class TestModalityTagDerivation:
    """The plan spec names three categories: sensor / linguistic / drive.

    The EC ``modality`` argument is more granular (``text``,
    ``interoception``, ``vision``, ...); ``_EC_TRACE_MODALITY_TAG_MAP``
    collapses it onto the three plan categories so the analyzer can
    pivot on ``modality_tag`` without re-mapping.
    """

    def test_text_maps_to_linguistic(self):
        assert _EC_TRACE_MODALITY_TAG_MAP["text"] == "linguistic"

    def test_interoception_maps_to_drive(self):
        assert _EC_TRACE_MODALITY_TAG_MAP["interoception"] == "drive"

    def test_vision_maps_to_sensor(self):
        assert _EC_TRACE_MODALITY_TAG_MAP["vision"] == "sensor"


# ─────────────────────────────────────────────────────────────────────────────
# Emission gating — sim_log must be called when enabled, NOT when disabled
# ─────────────────────────────────────────────────────────────────────────────


class TestEmissionGating:
    def test_disabled_emits_nothing(self):
        """Env var unset → no sim_log call regardless of sim state."""
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            _emit_ec_activation(
                node_id="n-1",
                similarity=0.5,
                is_new=False,
                modality="text",
            )
            mock_sim_log.assert_not_called()

    def test_enabled_but_sim_inactive_emits_nothing(self, monkeypatch):
        """Env var on but ``sim_logger._sim_active`` False → no emission.

        sim_log itself short-circuits on inactive sessions; the EC trace
        helper additionally guards on ``_sim_active`` so it doesn't even
        construct the event dict in raw-library usage.
        """
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", False, raising=False)
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            _emit_ec_activation(
                node_id="n-1",
                similarity=0.5,
                is_new=False,
                modality="text",
            )
            mock_sim_log.assert_not_called()

    def test_enabled_and_sim_active_emits_event(self, monkeypatch):
        """Env var on + sim active → exactly one sim_log call with EC_TRACE."""
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", True, raising=False)
        monkeypatch.setattr(sl, "_sim_start", 0.0, raising=False)
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            _emit_ec_activation(
                node_id="node-abc-123",
                similarity=0.73,
                is_new=False,
                modality="text",
            )
            mock_sim_log.assert_called_once()
            args, kwargs = mock_sim_log.call_args
            assert args[0] == "EC_TRACE"
            data = args[2]
            assert data["active_node_id"] == "node-abc-123"
            assert data["activation_strength"] == 0.73
            assert data["modality_tag"] == "linguistic"
            assert data["modality"] == "text"
            assert data["is_new"] is False
            assert isinstance(data["tick"], int)

    def test_new_node_activation_strength_is_one(self, monkeypatch):
        """Pattern separation (is_new=True) → activation_strength=1.0.

        Cold-start events MUST emit per the plan spec ("Event MUST emit
        even on cold-start when active_node_id is newly created").
        """
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", True, raising=False)
        monkeypatch.setattr(sl, "_sim_start", 0.0, raising=False)
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            _emit_ec_activation(
                node_id="fresh-node",
                similarity=0.0,
                is_new=True,
                modality="text",
            )
            mock_sim_log.assert_called_once()
            data = mock_sim_log.call_args.args[2]
            assert data["is_new"] is True
            assert data["activation_strength"] == 1.0

    def test_emission_exception_does_not_propagate(self, monkeypatch):
        """Instrumentation must never crash the substrate path.

        If sim_log raises, ``_emit_ec_activation`` swallows the exception
        (debug-log only). The EC.pattern_complete_or_separate return
        value is the contract; emission is best-effort.
        """
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", True, raising=False)
        monkeypatch.setattr(sl, "_sim_start", 0.0, raising=False)
        with patch("maxim.simulation.sim_logger.sim_log", side_effect=RuntimeError("boom")):
            # Must not raise
            _emit_ec_activation(
                node_id="n-1",
                similarity=0.5,
                is_new=False,
                modality="text",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-agent isolation — contextvar agent_id is propagated to the event
# ─────────────────────────────────────────────────────────────────────────────


class TestMultiAgentIsolation:
    """``_current_agent_id`` contextvar populates the event's ``agent_id``.

    The Roy-4 analyzer groups by agent_id when computing co-activation,
    so two agents firing the same node UUIDs in the same tick MUST emit
    distinguishable events. The contextvar is the canonical multi-agent
    marker (see CLAUDE.md "Per-agent stash dicts" rule).
    """

    def test_agent_id_from_contextvar_propagates(self, monkeypatch):
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", True, raising=False)
        monkeypatch.setattr(sl, "_sim_start", 0.0, raising=False)
        token = sl._current_agent_id.set("agent-alpha")
        try:
            with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
                _emit_ec_activation(
                    node_id="n-1",
                    similarity=0.5,
                    is_new=False,
                    modality="text",
                )
                kwargs = mock_sim_log.call_args.kwargs
                assert kwargs["agent_id"] == "agent-alpha"
        finally:
            sl._current_agent_id.reset(token)

    def test_two_agent_ids_emit_disjoint_events(self, monkeypatch):
        """Sequential emissions under different agent_id contexts produce events tagged to each agent."""
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", True, raising=False)
        monkeypatch.setattr(sl, "_sim_start", 0.0, raising=False)
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            token_a = sl._current_agent_id.set("agent-alpha")
            try:
                _emit_ec_activation(
                    node_id="n-1",
                    similarity=0.5,
                    is_new=False,
                    modality="text",
                )
            finally:
                sl._current_agent_id.reset(token_a)
            token_b = sl._current_agent_id.set("agent-bravo")
            try:
                _emit_ec_activation(
                    node_id="n-2",
                    similarity=0.6,
                    is_new=False,
                    modality="interoception",
                )
            finally:
                sl._current_agent_id.reset(token_b)

            assert mock_sim_log.call_count == 2
            kwargs_a = mock_sim_log.call_args_list[0].kwargs
            kwargs_b = mock_sim_log.call_args_list[1].kwargs
            assert kwargs_a["agent_id"] == "agent-alpha"
            assert kwargs_b["agent_id"] == "agent-bravo"

    def test_no_agent_context_emits_none(self, monkeypatch):
        """When the contextvar isn't set, agent_id is None (sim_log handles)."""
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", True, raising=False)
        monkeypatch.setattr(sl, "_sim_start", 0.0, raising=False)
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            _emit_ec_activation(
                node_id="n-1",
                similarity=0.5,
                is_new=False,
                modality="text",
            )
            kwargs = mock_sim_log.call_args.kwargs
            assert kwargs["agent_id"] is None


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end via EC.pattern_complete_or_separate — both branches emit
# ─────────────────────────────────────────────────────────────────────────────


class TestPatternCompleteEmits:
    """Drive ``pattern_complete_or_separate`` and assert both return paths emit."""

    def test_separation_path_emits_new_node_event(self, monkeypatch):
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", True, raising=False)
        monkeypatch.setattr(sl, "_sim_start", 0.0, raising=False)
        ec = EntorhinalCortex()
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            result = ec.pattern_complete_or_separate(
                embedding=[1.0, 0.0, 0.0],
                modality="text",
                geometry=None,
            )
            assert result.is_new is True
            mock_sim_log.assert_called_once()
            data = mock_sim_log.call_args.args[2]
            assert data["is_new"] is True
            assert data["modality_tag"] == "linguistic"
            assert data["active_node_id"] == result.node_id

    def test_completion_path_emits_existing_node_event(self, monkeypatch):
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", True, raising=False)
        monkeypatch.setattr(sl, "_sim_start", 0.0, raising=False)
        ec = EntorhinalCortex()
        # Pre-register a node so the next call pattern-completes onto it.
        ec.register_substrate_node("preexisting", [1.0, 0.0, 0.0], "text")
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            result = ec.pattern_complete_or_separate(
                embedding=[1.0, 0.0, 0.0],
                modality="text",
                geometry=None,
            )
            assert result.is_new is False
            assert result.node_id == "preexisting"
            mock_sim_log.assert_called_once()
            data = mock_sim_log.call_args.args[2]
            assert data["is_new"] is False
            assert data["active_node_id"] == "preexisting"
            assert data["activation_strength"] > 0.99  # cos sim of identical vectors

    def test_interoception_path_uses_drive_modality_tag(self, monkeypatch):
        """Frozen-prototype (interoception) modality branch also emits the event."""
        monkeypatch.setenv("MAXIM_EC_TRACE_ACTIVATIONS", "1")
        from maxim.simulation import sim_logger as sl

        monkeypatch.setattr(sl, "_sim_active", True, raising=False)
        monkeypatch.setattr(sl, "_sim_start", 0.0, raising=False)
        ec = EntorhinalCortex()
        ec.register_substrate_node("sensor-1", [1.0, 0.0, 0.0], "interoception")
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            ec.pattern_complete_or_separate(
                embedding=[1.0, 0.0, 0.0],
                modality="interoception",
                threshold=0.5,
                geometry=None,
            )
            mock_sim_log.assert_called_once()
            data = mock_sim_log.call_args.args[2]
            assert data["modality_tag"] == "drive"
            assert data["modality"] == "interoception"

    def test_disabled_pattern_complete_emits_nothing(self):
        """No env var → pattern_complete_or_separate emits no events."""
        ec = EntorhinalCortex()
        with patch("maxim.simulation.sim_logger.sim_log") as mock_sim_log:
            ec.pattern_complete_or_separate(
                embedding=[1.0, 0.0, 0.0],
                modality="text",
                geometry=None,
            )
            mock_sim_log.assert_not_called()
