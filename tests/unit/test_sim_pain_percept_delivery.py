"""Pain intensity is validated by the ``Reaction`` type (#863 review round).

``Reaction.intensity`` was documented as [0, 1] and never checked, and reward
distribution feeds it to NAc unclamped — so a scenario ``intensity: 5`` became
a reward of -5, and ``"high"`` was published as-is. The type now rejects both,
so every producer is covered; ``SimulationAdapter`` no longer wraps the publish
in ``except Exception: pass``, and the LLM scenario generator coerces its own
output at its boundary so bad model output does not end a run.
"""

from __future__ import annotations

import pytest

from maxim.agents.percept_factory import make_intero_percept
from maxim.reactions.bus import ReactionBus
from maxim.runtime.sim_adapter import SimulationAdapter
from maxim.simulation.simulation_generator import _clean_percepts, _pain_intensity


class _OneShotSource:
    def __init__(self, percept) -> None:
        self._percept = percept

    @property
    def name(self) -> str:
        return "oneshot"

    @property
    def capabilities(self) -> set[str]:
        return {"proprioception"}

    def next_percept(self):
        p, self._percept = self._percept, None
        return p

    def is_exhausted(self) -> bool:
        return False


def _pain(intensity) -> object:
    return make_intero_percept(
        "pain_signal",
        source="proprioception",
        metadata={"pain_type": "burn", "intensity": intensity},
    )


def _adapter(percept, bus: ReactionBus) -> SimulationAdapter:
    return SimulationAdapter(_OneShotSource(percept), pain_bus=bus)


def test_a_pain_percept_is_published_with_its_intensity():
    bus = ReactionBus(_allow_raw=True)
    _adapter(_pain(0.7), bus).next_observation(environment=None)
    [reaction] = bus.history("pain")
    assert reaction.intensity == pytest.approx(0.7)
    assert reaction.source == "sim_adapter:burn"


@pytest.mark.parametrize(("intensity", "error"), [("high", TypeError), (5, ValueError), (float("nan"), ValueError)])
def test_a_malformed_pain_percept_is_rejected_before_it_reaches_the_bus(intensity, error):
    bus = ReactionBus(_allow_raw=True)
    with pytest.raises(error):
        _adapter(_pain(intensity), bus).next_observation(environment=None)
    assert bus.history("pain") == []


def _reaction(intensity):
    from maxim.decisions.causal_link import Valence
    from maxim.reactions.types import WORLD_AGENT_ID, Reaction, ReactionContext

    return Reaction(
        kind="pain",
        intensity=intensity,
        valence=Valence.NEGATIVE,
        timestamp=0.0,
        context=ReactionContext(agent_id=WORLD_AGENT_ID),
        source="test",
    )


@pytest.mark.parametrize("ok", [0, 0.0, 0.5, 1, 1.0])
def test_reaction_accepts_the_unit_range(ok):
    assert _reaction(ok).intensity == ok


def test_reaction_accepts_numpy_reals():
    np = pytest.importorskip("numpy")
    assert _reaction(np.float32(0.25)).intensity == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        (-0.01, ValueError),
        (1.01, ValueError),
        (5, ValueError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        ("0.5", TypeError),
        (None, TypeError),
        (True, TypeError),
    ],
)
def test_reaction_rejects_intensity_outside_its_contract(bad, error):
    with pytest.raises(error):
        _reaction(bad)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(0.4, 0.4), ("0.4", 0.4), (5, 1.0), (-2, 0.0), ("high", 0.5), (None, 0.5), (float("nan"), 0.5)],
)
def test_generated_intensity_is_a_float_in_unit_range(raw, expected):
    assert _pain_intensity(raw, step=0) == pytest.approx(expected)


def test_clean_percepts_validates_generated_pain_intensity():
    cleaned = _clean_percepts([{"at": 0, "source": "proprioception", "metadata": {"intensity": "high"}}])
    [pain] = [p for p in cleaned if p["source"] == "proprioception"]
    assert pain["metadata"]["intensity"] == 0.5


# ── PainSignal: the OTHER carrier of pain intensity ─────────────────────────
# PainBus.publish hands a PainSignal to its direct subscribers (tool-pain
# bridge, hippocampus) BEFORE converting it to a Reaction, so validating only
# the Reaction left every PainSignal producer able to deliver a bad value.


@pytest.mark.parametrize(
    ("bad", "error"), [(5, ValueError), (-0.4, ValueError), (float("nan"), ValueError), ("x", TypeError)]
)
def test_pain_signal_rejects_intensity_outside_its_contract(bad, error):
    from maxim.proprioception.pain import PainSignal, PainType

    with pytest.raises(error):
        PainSignal(pain_type=PainType.EXTERNAL_SIGNAL, intensity=bad, timestamp=0.0)


def test_pain_signal_accepts_the_unit_range():
    from maxim.proprioception.pain import PainSignal, PainType

    assert PainSignal(pain_type=PainType.EXTERNAL_SIGNAL, intensity=1.0, timestamp=0.0).intensity == 1.0


# ── The LLM-facing pain tools reject out-of-range arguments ────────────────


class _RecordingBridge:
    turn_count = 0

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def inject_pain(self, **kw) -> None:
        self.calls.append(kw)


@pytest.mark.parametrize("bad", [7, -0.1, float("nan"), "high"])
def test_inject_pain_returns_a_retryable_error_instead_of_killing_the_aut(bad):
    from maxim.simulation.tools import InjectPainTool

    bridge = _RecordingBridge()
    out = InjectPainTool(bridge).execute(intensity=bad)
    assert out.success is False and "intensity" in out.error
    assert bridge.calls == []  # nothing reached the AUT


def test_inject_pain_passes_a_valid_intensity_through():
    from maxim.simulation.tools import InjectPainTool

    bridge = _RecordingBridge()
    assert InjectPainTool(bridge).execute(intensity=0.6).success is True
    assert bridge.calls == [{"pain_type": "external_signal", "intensity": 0.6}]


@pytest.mark.parametrize("bad", [-0.2, 1.5, float("inf"), "lots"])
def test_damage_component_rejects_an_out_of_range_amount(bad):
    """A negative amount used to HEAL the component (apply_damage(-0.2))."""
    from maxim.simulation.tools import DamageComponentTool

    out = DamageComponentTool(embodiment=object(), entity_map=None).execute(component="torso", amount=bad)
    assert out.success is False and "amount" in out.error


# ── A rejected intensity is REPORTED, not lost at DEBUG ────────────────────


def test_a_rejected_cerebellum_reaction_is_reported(caplog):
    from maxim.embodiment.backends.cerebellum_modulator import CerebellumModulator

    mod = CerebellumModulator.__new__(CerebellumModulator)
    mod._reaction_bus = ReactionBus(_allow_raw=True)

    class _E:
        name = "arm"

    mod._entity, mod._name = _E(), "grip"
    with caplog.at_level("WARNING"):
        mod._emit_failure_reaction("grasp", 5.0)
    assert mod._reaction_bus.history("pain") == []
    assert any(r.levelname == "WARNING" for r in caplog.records)


@pytest.mark.parametrize(("raw", "expected"), [("0.3", 0.3), (1, 1.0), (0, 0.0)])
def test_inject_pain_still_accepts_the_valid_forms(raw, expected):
    from maxim.simulation.tools import InjectPainTool

    bridge = _RecordingBridge()
    assert InjectPainTool(bridge).execute(intensity=raw).success is True
    assert bridge.calls[0]["intensity"] == pytest.approx(expected)


class _FakeRoot:
    full_path = "body"
    entity_type = "body"

    def __init__(self) -> None:
        self.vital_metrics = {"health": 1.0}

    def get_component(self, _name):
        return None


class _FailingPainBus:
    def publish(self, _signal) -> None:
        raise RuntimeError("pain bus down")


class _FakeEmbodiment:
    agent_id = "a"

    def __init__(self) -> None:
        self.root = _FakeRoot()
        self._pain_bus = _FailingPainBus()

    def evaluate_failures(self):
        return []


def test_a_failed_damage_pain_publish_is_reported_not_silently_passed(caplog):
    """The damage tool's pain publish was ``except Exception: pass``."""
    from maxim.simulation.tools import DamageComponentTool

    emb = _FakeEmbodiment()
    with caplog.at_level("WARNING"):
        out = DamageComponentTool(embodiment=emb, entity_map=None).execute(component="torso", amount="0.2")
    assert out.success is True
    assert emb.root.vital_metrics["health"] == pytest.approx(0.8)  # the damage itself still applied
    assert any(r.levelname == "WARNING" for r in caplog.records)
