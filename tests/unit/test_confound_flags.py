"""Per-flag pin tests for the V1 substrate-attribution scaffold-disable flags.

Each flag has two assertions:

1. **Default-on (env unset):** the gated injector fires exactly as today.
2. **Disabled (env=1):** the injector skips. This pins the conditional
   structurally — a refactor that drops ``if ..._enabled():`` fails the
   second assertion immediately.

Per ``feedback_opt_in_env_in_hot_paths.md``, the autouse scrub fixture
in ``tests/conftest.py::_isolate_maxim_confound_flags_env`` is non-
optional. These tests rely on it: each test only mutates env via
``monkeypatch.setenv`` so the scrub restores state on exit. If the
scrub is removed or omits a flag, a setenv leak will surface as the
default-on assertion failing in the next test in sequence.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest

from maxim.agents.autonomy import AutonomyLevel
from maxim.agents.prompt_builder import PromptBuilder, build_identity_section
from maxim.agents.prompt_budgeter import PromptBudgeter
from maxim.cli import _resolve_persona
from maxim.runtime.confound_flags import (
    ALL_FLAGS,
    acting_coach_enabled,
    default_persona_enabled,
    pfc_preamble_enabled,
    sim_sandbox_text_enabled,
)


# ── Module sanity ──────────────────────────────────────────────────────


class TestConfoundFlagsModule:
    def test_all_flags_lists_every_helper(self):
        # If a future PR adds a new helper here without registering it in
        # ALL_FLAGS, the autouse scrub silently misses it. Pin the
        # invariant.
        assert "MAXIM_DISABLE_PFC_PREAMBLE" in ALL_FLAGS
        assert "MAXIM_DISABLE_ACTING_COACH" in ALL_FLAGS
        assert "MAXIM_DISABLE_SIM_SANDBOX_TEXT" in ALL_FLAGS
        assert "MAXIM_NO_DEFAULT_PERSONA" in ALL_FLAGS
        assert len(ALL_FLAGS) == 4

    def test_every_enabled_helper_references_an_all_flags_entry(self):
        # Pre-merge review fold (F11/F2): structural backstop. A future
        # PR that adds a `def foo_enabled():` helper but forgets to
        # register the corresponding env var name in ALL_FLAGS will
        # silently leak across tests. Static-grep the source so the leak
        # surfaces here BEFORE it bites the autouse scrub.
        import inspect
        import re

        from maxim.runtime import confound_flags

        source = inspect.getsource(confound_flags)
        # Find every function named *_enabled defined in the module.
        helpers = re.findall(r"^def\s+(\w+_enabled)\s*\(", source, re.MULTILINE)
        assert helpers, "No *_enabled() helpers found — confound_flags.py shape changed unexpectedly"

        # Each helper's body should reference one of the env var names
        # in ALL_FLAGS.
        for helper_name in helpers:
            helper_fn = getattr(confound_flags, helper_name)
            helper_src = inspect.getsource(helper_fn)
            referenced = [name for name in ALL_FLAGS if name in helper_src]
            assert referenced, (
                f"Helper {helper_name!r} references no flag in ALL_FLAGS — "
                f"either add the flag to ALL_FLAGS so the autouse scrub picks it up, "
                f"or the helper belongs in a different module."
            )

    def test_helpers_default_to_enabled(self):
        # Defaults preserve current behavior — every helper returns True
        # when env is unset.
        assert pfc_preamble_enabled() is True
        assert acting_coach_enabled() is True
        assert sim_sandbox_text_enabled() is True
        assert default_persona_enabled() is True

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "Yes", " 1 "])
    def test_truthy_disables(self, monkeypatch, value):
        monkeypatch.setenv("MAXIM_DISABLE_PFC_PREAMBLE", value)
        assert pfc_preamble_enabled() is False

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "anything-else"])
    def test_non_truthy_keeps_enabled(self, monkeypatch, value):
        monkeypatch.setenv("MAXIM_DISABLE_PFC_PREAMBLE", value)
        assert pfc_preamble_enabled() is True


# ── PFC preamble gate ─────────────────────────────────────────────────


class _StubCounter:
    """Minimal token counter for PromptBudgeter — char/4 heuristic."""

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _make_request_with_bio_signal():
    """Construct an LLMRequest-like mock with at least one bio-signal."""
    req = MagicMock()
    req.acting_coach = None
    req.context = MagicMock()
    req.context.bio_enrichment_context = "some bio enrichment text"
    req.context.deliberation_transcript = None
    req.context.working_memory_thoughts = None
    req.context.causal_context = None
    req.context.motor_programs = None
    return req


class TestPfcPreambleGate:
    def test_default_on_emits_preamble(self):
        budgeter = PromptBudgeter(total_budget=8000, response_reserve=512, token_counter=_StubCounter())
        PromptBuilder._add_pfc_preamble_section(budgeter, _make_request_with_bio_signal())
        names = [s.name for s in budgeter._sections]
        assert "pfc_preamble" in names

    def test_disabled_skips_preamble(self, monkeypatch):
        monkeypatch.setenv("MAXIM_DISABLE_PFC_PREAMBLE", "1")
        budgeter = PromptBudgeter(total_budget=8000, response_reserve=512, token_counter=_StubCounter())
        PromptBuilder._add_pfc_preamble_section(budgeter, _make_request_with_bio_signal())
        names = [s.name for s in budgeter._sections]
        assert "pfc_preamble" not in names


# ── Acting Coach gate ─────────────────────────────────────────────────


def _make_request_with_coach():
    """Construct an LLMRequest-like mock with a real ActingCoachConfig attached."""
    from maxim.prompts.acting_coach import ActingCoachConfig

    req = MagicMock()
    req.acting_coach = ActingCoachConfig()
    req.available_tools = set()
    req.context = MagicMock()
    req.context.causal_context = None
    req.context.body_state = ""
    req.context.motor_programs = None
    return req


class TestActingCoachGate:
    def test_default_on_emits_coach_section(self):
        budgeter = PromptBudgeter(total_budget=8000, response_reserve=512, token_counter=_StubCounter())
        PromptBuilder._add_acting_coach_section(budgeter, _make_request_with_coach())
        names = [s.name for s in budgeter._sections]
        assert "acting_coach" in names

    def test_disabled_skips_coach_section(self, monkeypatch):
        monkeypatch.setenv("MAXIM_DISABLE_ACTING_COACH", "1")
        budgeter = PromptBudgeter(total_budget=8000, response_reserve=512, token_counter=_StubCounter())
        PromptBuilder._add_acting_coach_section(budgeter, _make_request_with_coach())
        names = [s.name for s in budgeter._sections]
        assert "acting_coach" not in names

    def test_disabled_keeps_generic_identity(self, monkeypatch):
        from maxim.prompts.acting_coach import ActingCoachConfig

        monkeypatch.setenv("MAXIM_DISABLE_ACTING_COACH", "1")
        mode = MagicMock()
        mode.name = "passive"
        mode.goal = "assist"
        req = MagicMock()
        req.autonomy_level = AutonomyLevel.SUPERVISED
        req.is_sleeping = False
        req.acting_coach = ActingCoachConfig()
        result = build_identity_section(mode, req, "Monday", "10:00 AM")
        # Generic identity, NOT the embodied "body in a world" rewrite.
        assert "Maxim, a robot assistant" in result
        assert "body in a world" not in result

    def test_default_on_uses_embodied_identity(self):
        from maxim.prompts.acting_coach import ActingCoachConfig

        mode = MagicMock()
        mode.name = "passive"
        mode.goal = "explore"
        req = MagicMock()
        req.autonomy_level = AutonomyLevel.SUPERVISED
        req.is_sleeping = False
        req.acting_coach = ActingCoachConfig()
        result = build_identity_section(mode, req, "Monday", "10:00 AM")
        assert "body in a world" in result


# ── Sim sandbox text gate ─────────────────────────────────────────────


class TestSimSandboxTextGate:
    def test_default_on_emits_sandbox_text(self, monkeypatch):
        # Force _sim_active=True so the conditional reaches our gate.
        import maxim.simulation.sim_logger as sim_logger

        monkeypatch.setattr(sim_logger, "_sim_active", True, raising=False)
        mode = MagicMock()
        mode.name = "passive"
        mode.goal = "test"
        req = MagicMock()
        req.autonomy_level = AutonomyLevel.SUPERVISED
        req.is_sleeping = False
        req.acting_coach = None
        result = build_identity_section(mode, req, "Monday", "10:00 AM")
        assert "SIMULATION ENVIRONMENT" in result

    def test_disabled_skips_sandbox_text(self, monkeypatch):
        import maxim.simulation.sim_logger as sim_logger

        monkeypatch.setattr(sim_logger, "_sim_active", True, raising=False)
        monkeypatch.setenv("MAXIM_DISABLE_SIM_SANDBOX_TEXT", "1")
        mode = MagicMock()
        mode.name = "passive"
        mode.goal = "test"
        req = MagicMock()
        req.autonomy_level = AutonomyLevel.SUPERVISED
        req.is_sleeping = False
        req.acting_coach = None
        result = build_identity_section(mode, req, "Monday", "10:00 AM")
        assert "SIMULATION ENVIRONMENT" not in result


# ── Default-persona resolution ────────────────────────────────────────


def _make_args(persona: str = "adversarial") -> argparse.Namespace:
    return argparse.Namespace(sim_persona=persona)


class TestDefaultPersonaResolution:
    def test_default_on_returns_persona_string(self):
        args = _make_args(persona="adversarial")
        assert _resolve_persona(args, default="adversarial") == "adversarial"

    def test_default_on_returns_explicit_persona(self):
        args = _make_args(persona="cooperative")
        assert _resolve_persona(args, default="adversarial") == "cooperative"

    def test_disabled_returns_none(self, monkeypatch):
        monkeypatch.setenv("MAXIM_NO_DEFAULT_PERSONA", "1")
        args = _make_args(persona="adversarial")
        # Plan asserts "cli.py's persona resolution returns None".
        assert _resolve_persona(args, default="adversarial") is None

    def test_disabled_returns_none_even_with_explicit_persona(self, monkeypatch):
        # The flag is unconditional — researchers running --no-persona
        # shouldn't also pass --persona; if they do, the flag wins.
        monkeypatch.setenv("MAXIM_NO_DEFAULT_PERSONA", "1")
        args = _make_args(persona="cooperative")
        assert _resolve_persona(args, default="adversarial") is None


# ── Neutral persona registry entry ────────────────────────────────────


class TestNeutralPersona:
    def test_neutral_persona_exists(self):
        from maxim.simulation.personas import SIMULATION_PERSONAS

        assert "neutral" in SIMULATION_PERSONAS
        assert SIMULATION_PERSONAS["neutral"].context_prompt == ""
        assert SIMULATION_PERSONAS["neutral"].max_initiative == 0.0

    def test_get_persona_neutral_skips_early_finish_guidance(self):
        from maxim.simulation.personas import get_persona

        p = get_persona("neutral", continuous=False)
        assert p is not None
        assert p.context_prompt == ""

    def test_get_persona_adversarial_still_appends_guidance(self):
        from maxim.simulation.personas import get_persona

        p = get_persona("adversarial", continuous=False)
        assert p is not None
        assert "WHEN TO ABORT EARLY" in p.context_prompt

    def test_get_persona_neutral_continuous_appends_suffix(self):
        # Pre-merge review fold (F4): continuous mode is a procedural
        # invariant ("never auto-finish"), not behavioural framing. Even
        # the neutral persona must carry CONTINUOUS_SUFFIX or a
        # continuous run with --no-persona breaks the contract.
        from maxim.simulation.personas import get_persona

        p = get_persona("neutral", continuous=True)
        assert p is not None
        assert "NEVER" in p.context_prompt
