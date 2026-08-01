"""Guards for thalamic-relay stage 4 — audio-orient consumption + sim wiring.

Pins that a DoA/audio percept becomes *recognized in the agentic loop*:
- ``format_audio_orientation`` renders a passive azimuth line (and stays empty
  for non-audio / None / azimuth-less percepts, so it is safe to fold
  unconditionally);
- the rendered line rides ``StructuredContext.auto_sense_context`` into the
  composed prompt (the same passive-perception channel the loop uses);
- ``build_audio_composite`` attaches an ambient AzimuthDoASource;
- ``default_sim_doa_reader`` emits speech-present readings deterministically;
- ``audio_orient_enabled`` is default-OFF and reads the env flag.
"""

from __future__ import annotations

import os

from maxim.agents.modality import SensoryModality, SensoryTag
from maxim.agents.percept_factory import make_audio_percept, make_text_percept
from maxim.embodiment.audio_localization import (
    audio_attention_profile,
    format_audio_orientation,
    should_emit_orientation,
)
from maxim.simulation.audio_orient_wiring import (
    audio_orient_enabled,
    build_audio_composite,
    default_sim_doa_reader,
    sim_audio_salience_novelty,
)
from maxim.simulation.composite_source import CompositePerceptSource


# ── format_audio_orientation ────────────────────────────────────────────────


def test_format_none_and_non_audio_are_empty():
    assert format_audio_orientation(None) == ""
    # A text percept has no azimuth → empty (safe to fold unconditionally).
    assert format_audio_orientation(make_text_percept("hello", source="cli")) == ""


def test_format_left_right_center():
    left = format_audio_orientation(make_audio_percept(-0.7))
    right = format_audio_orientation(make_audio_percept(0.3))
    center = format_audio_orientation(make_audio_percept(0.0))
    assert "left" in left and "-0.70" in left
    assert "right" in right and "+0.30" in right
    assert "centered" in center or "ahead" in center


def test_format_clamps_out_of_range():
    assert "left" in format_audio_orientation(make_audio_percept(-5.0))
    assert "right" in format_audio_orientation(make_audio_percept(5.0))


def test_format_ignores_azimuth_on_non_sound_modality():
    """An ``azimuth`` key on a non-SOUND percept must not render — guards
    against a stray metadata key being read as a sound direction."""
    p = make_text_percept("x", source="cli")
    p.metadata = {"azimuth": -0.5}
    p.sensory = SensoryTag(modality=SensoryModality.NARRATIVE)
    assert format_audio_orientation(p) == ""


# ── rides the prompt via auto_sense_context ─────────────────────────────────


class _RecordingBudgeter:
    """Captures the (name, content) pairs a section builder emits."""

    def __init__(self) -> None:
        self.sections: dict[str, str] = {}

    def add(self, name, content, *args, **kwargs):
        self.sections[name] = content


def test_audio_line_reaches_prompt_via_auto_sense_context():
    """The passive azimuth line, placed on ``auto_sense_context``, is emitted by
    the prompt builder's perception-section pass — the same channel the loop
    folds it into (section 1.16). Tests the render seam directly."""
    from types import SimpleNamespace

    from maxim.agents.bus import StructuredContext
    from maxim.agents.prompt_builder import PromptBuilder

    line = format_audio_orientation(make_audio_percept(-0.6))
    assert line

    ctx = StructuredContext(timestamp=0.0)
    ctx.auto_sense_context = line
    req = SimpleNamespace(context=ctx, agent_states=None, recent_outcomes=None)

    budgeter = _RecordingBudgeter()
    PromptBuilder._add_perception_sections(budgeter, req)

    assert "auto_sense" in budgeter.sections
    body = budgeter.sections["auto_sense"]
    assert "azimuth" in body and "left" in body


# ── build_audio_composite ───────────────────────────────────────────────────


class _Base:
    @property
    def name(self) -> str:
        return "base"

    @property
    def capabilities(self) -> set[str]:
        return {"cli"}

    def next_percept(self):
        return None

    def is_exhausted(self) -> bool:
        return False


def test_audio_attention_profile_default_passes_no_gates():
    """The default DoA weights (0.5/0.3) clear NONE of the attention gates —
    the mechanistic reason a default sound is perceived but not attended."""
    p = audio_attention_profile(0.5, 0.3)
    assert p["salience"] == 0.5 and p["novelty"] == 0.3
    assert p["passes_salience_gate"] is False  # 0.5 is not > 0.5
    assert p["passes_novelty_gate"] is False
    assert p["passes_proposal_gate"] is False
    assert p["passes_high_novelty_memory"] is False


def test_audio_attention_profile_hot_passes_gates():
    p = audio_attention_profile(0.9, 0.9)
    assert p["passes_salience_gate"] is True
    assert p["passes_novelty_gate"] is True
    assert p["passes_proposal_gate"] is True  # both > 0.5
    assert p["passes_high_novelty_memory"] is True  # 0.9 > 0.7


def test_azimuth_source_salience_novelty_are_tunable():
    """The DoA percept's attention weight is the scaling-experiment knob —
    defaults are sub-threshold; raising them pushes the sound above the
    `> 0.5` attention gates."""
    from maxim.embodiment.audio_localization import AzimuthDoASource

    reader = default_sim_doa_reader(period=1)
    # default (sub-threshold)
    p_default = AzimuthDoASource(reader).next_percept()
    assert p_default.salience == 0.5 and p_default.novelty == 0.3
    # scaled up (above the 0.5 gates)
    reader2 = default_sim_doa_reader(period=1)
    p_hot = AzimuthDoASource(reader2, salience=0.9, novelty=0.9).next_percept()
    assert p_hot.salience == 0.9 and p_hot.novelty == 0.9


def test_azimuth_source_has_pending_true():
    """Ambient live sensor must be sampled every tick (S2) — has_pending is
    explicitly True so the loop's idle-sleep can't starve it."""
    from maxim.embodiment.audio_localization import AzimuthDoASource

    src = AzimuthDoASource(default_sim_doa_reader())
    assert src.has_pending() is True


def test_hot_audio_escalates_baseline_does_not():
    """The escalation contract (B1 fix): passes_salience_gate is the thalamic
    escalation signal. Default sound (0.5) is sub-threshold → not escalated;
    scaled-up sound (0.9) escalates → reaches the LLM."""
    assert audio_attention_profile(0.5, 0.3)["passes_salience_gate"] is False
    assert audio_attention_profile(0.9, 0.9)["passes_salience_gate"] is True


def test_sim_audio_salience_novelty_env_override():
    assert sim_audio_salience_novelty() == (0.5, 0.3)  # defaults (conftest scrubs env)
    os.environ["MAXIM_SIM_AUDIO_SALIENCE"] = "0.9"
    os.environ["MAXIM_SIM_AUDIO_NOVELTY"] = "0.85"
    try:
        assert sim_audio_salience_novelty() == (0.9, 0.85)
        # malformed → default, clamped
        os.environ["MAXIM_SIM_AUDIO_SALIENCE"] = "nope"
        assert sim_audio_salience_novelty()[0] == 0.5
        os.environ["MAXIM_SIM_AUDIO_SALIENCE"] = "5"
        assert sim_audio_salience_novelty()[0] == 1.0  # clamped to [0,1]
    finally:
        os.environ.pop("MAXIM_SIM_AUDIO_SALIENCE", None)
        os.environ.pop("MAXIM_SIM_AUDIO_NOVELTY", None)


def test_build_audio_composite_attaches_ambient_audio():
    base = _Base()
    composite = build_audio_composite(base, default_sim_doa_reader(), agent_id="sim_aut")
    assert isinstance(composite, CompositePerceptSource)
    assert "audio" in composite.capabilities and "cli" in composite.capabilities
    # The base is scripted/exhaustible; audio is ambient — nothing exhausted yet.
    assert composite.is_exhausted() is False


def test_composite_surfaces_audio_percept_from_reader():
    """End-to-end offline: base idle + reader speech-present → the composite
    yields an audio percept whose azimuth renders a passive line."""
    base = _Base()
    # period=1 → every call is speech-present, so the first drain yields audio.
    reader = default_sim_doa_reader(period=1)
    composite = build_audio_composite(base, reader, agent_id="sim_aut")
    percept = composite.next_percept()
    assert percept is not None
    assert format_audio_orientation(percept) != ""


# ── default_sim_doa_reader ──────────────────────────────────────────────────


def test_default_reader_is_periodic_and_deterministic():
    r = default_sim_doa_reader(period=3)
    outs = [r() for _ in range(6)]
    speech_flags = [flag for (_, flag) in outs]
    assert speech_flags == [False, False, True, False, False, True]
    # Deterministic: a fresh reader repeats the pattern.
    r2 = default_sim_doa_reader(period=3)
    assert [flag for (_, flag) in (r2() for _ in range(6))] == speech_flags


def test_default_reader_throttled_by_default():
    """Default cadence is sparse (period=40) so sound is an event, not the
    every-2s hum the first live run produced."""
    r = default_sim_doa_reader()
    speech = [flag for (_, flag) in (r() for _ in range(40))]
    assert speech.count(True) == 1  # exactly one event in 40 calls


def test_default_reader_max_events_caps_emissions():
    r = default_sim_doa_reader(period=2, max_events=3)
    speech = [flag for (_, flag) in (r() for _ in range(20))]
    assert speech.count(True) == 3  # goes silent after the cap


def test_should_emit_orientation_change_gate():
    assert should_emit_orientation(None, -0.6) is True  # first sound always emits
    assert should_emit_orientation(-0.6, -0.6) is False  # unchanged → suppressed
    assert should_emit_orientation(-0.6, 0.6) is True  # big change → emit
    assert should_emit_orientation(-0.6, -0.62) is False  # sub-threshold jitter → suppressed
    assert should_emit_orientation(-0.6, -0.4) is True  # ≥0.15 change → emit


# ── audio_orient_enabled ────────────────────────────────────────────────────


def test_audio_orient_enabled_default_off():
    # The autouse conftest scrub guarantees the var is unset here.
    assert audio_orient_enabled() is False


def test_audio_orient_enabled_reads_flag():
    os.environ["MAXIM_SIM_AUDIO_ORIENT"] = "1"
    try:
        assert audio_orient_enabled() is True
    finally:
        os.environ.pop("MAXIM_SIM_AUDIO_ORIENT", None)


# ── Stage 3 (live_audio_orient_wiring.md): the percept lane on the LIVE path ─
#
# §1.16's gate was re-based from ``sim.is_sim_mode`` (a proxy) onto
# ``sim.current_percept is not None`` (the real condition), and
# NullSimulationAdapter learned to CARRY a percept from a live producer
# (the Stage-2 DoA feed) without flipping is_sim_mode.


class _ObsEnv:
    def observe(self):
        return {}


class TestNullAdapterPerceptCarry:
    def test_is_sim_mode_stays_false(self):
        """The load-bearing invariant: carrying a percept must NOT make the
        live path look like a sim (12 is_sim_mode consumer sites)."""
        from maxim.runtime.sim_adapter import NullSimulationAdapter

        adapter = NullSimulationAdapter()
        adapter.carry_percept(make_audio_percept(-0.6, source="test:doa"))
        adapter.next_observation(_ObsEnv())
        assert adapter.is_sim_mode is False
        assert adapter.current_percept is not None

    def test_carried_percept_surfaces_next_tick_then_clears(self):
        from maxim.runtime.sim_adapter import NullSimulationAdapter

        adapter = NullSimulationAdapter()
        assert adapter.current_percept is None
        p = make_audio_percept(-0.6, source="test:doa")
        adapter.carry_percept(p)
        # Not visible until the tick boundary (mirrors SimulationAdapter).
        assert adapter.current_percept is None
        adapter.next_observation(_ObsEnv())
        assert adapter.current_percept is p
        # Next tick with nothing carried → cleared, never stale.
        adapter.next_observation(_ObsEnv())
        assert adapter.current_percept is None

    def test_latest_wins_between_ticks(self):
        """Direction data ages badly — an undelivered bearing is overwritten
        by a fresher one, not queued behind it."""
        from maxim.runtime.sim_adapter import NullSimulationAdapter

        adapter = NullSimulationAdapter()
        adapter.carry_percept(make_audio_percept(-0.6, source="test:doa"))
        fresh = make_audio_percept(0.4, source="test:doa")
        adapter.carry_percept(fresh)
        adapter.next_observation(_ObsEnv())
        assert adapter.current_percept is fresh

    def test_carried_percept_satisfies_the_1_16_gate(self):
        """The exact §1.16 gate expression: fires with a carried percept on a
        non-sim adapter; skips without one."""
        from maxim.runtime.sim_adapter import NullSimulationAdapter

        aut_mode = "llm-primary"
        sim = NullSimulationAdapter()
        assert not (getattr(sim, "current_percept", None) is not None and aut_mode != "substrate-primary")
        sim.carry_percept(make_audio_percept(-0.6, source="test:doa"))
        sim.next_observation(_ObsEnv())
        assert getattr(sim, "current_percept", None) is not None and aut_mode != "substrate-primary"
        # substrate-primary stays excluded (the drive/EC path reads the
        # sensor directly; §1.16 would double-write).
        aut_mode = "substrate-primary"
        assert not (getattr(sim, "current_percept", None) is not None and aut_mode != "substrate-primary")


def test_agent_loop_1_16_gate_reads_current_percept_not_is_sim_mode():
    """Source pin: the §1.16 gate must key on the side-channel, not the
    is_sim_mode proxy — reverting it silently re-darkens the live path."""
    import inspect

    import maxim.runtime.agent_loop as agent_loop

    src = inspect.getsource(agent_loop)
    assert 'if getattr(sim, "current_percept", None) is not None and aut_mode != "substrate-primary":' in src, (
        "§1.16 gate no longer keys on sim.current_percept (Stage 3 re-gate reverted?)"
    )
    assert 'if sim.is_sim_mode and aut_mode != "substrate-primary":' not in src, (
        "the old is_sim_mode §1.16 gate is back — live audio percepts would be dropped"
    )
