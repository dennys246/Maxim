"""Guards for the thalamic-relay un-flatten (thalamus_relay_design_pass.md Decision 2).

The design review's Executor lens found the draft's "attach the Percept to the
observation dict" leaks the object into persisted state: ``state.update`` absorbs
every observation key into ``state.data``, which is deep-copied by ``snapshot()``
and serialized to ``state.json`` + episode snapshots (masked by ``default=str``).

The fix carries the Percept on a typed adapter side-channel (``sim.current_percept``)
and keeps the observation dict scalar-only. These tests pin:

1. the five text keys are byte-identical vs the pre-change shape;
2. feeding the observation into ``RuntimeState.update`` adds NO Percept-bearing
   key to ``state.data`` (the anti-leak guard) and survives ``snapshot()`` clean;
3. ``current_percept`` is the emitted object, and ``None`` on the idle path.
"""

from __future__ import annotations

from maxim.agents.bus import Percept
from maxim.runtime.sim_adapter import NullSimulationAdapter, SimulationAdapter
from maxim.runtime.state import RuntimeState


class _OneShotSource:
    """Emits a single Percept, then idles forever."""

    def __init__(self, percept: Percept | None) -> None:
        self._percept = percept

    @property
    def name(self) -> str:
        return "oneshot"

    @property
    def capabilities(self) -> set[str]:
        return {"cli"}

    def next_percept(self) -> Percept | None:
        p, self._percept = self._percept, None
        return p

    def is_exhausted(self) -> bool:
        return False


def _percept() -> Percept:
    return Percept(
        timestamp=1.0,
        source="cli",
        cli_input="hello world",
        transcript_chunk="hello world",
        hard_override=None,
        raw_transcript_text="hello world",
        salience=0.8,
        modality=None,  # non-None percept can still carry None modality
        sensory=None,
    )


def test_observation_dict_has_only_the_five_scalar_text_keys():
    p = _percept()
    adapter = SimulationAdapter(_OneShotSource(p))
    obs = adapter.next_observation(environment=None)
    assert set(obs.keys()) == {
        "source",
        "transcript",
        "cli_input",
        "hard_override",
        "raw_transcript_text",
    }
    # No Percept object smuggled onto the dict.
    assert all(not isinstance(v, Percept) for v in obs.values())
    assert obs["cli_input"] == "hello world"


def test_current_percept_is_the_emitted_object():
    p = _percept()
    adapter = SimulationAdapter(_OneShotSource(p))
    adapter.next_observation(environment=None)
    assert adapter.current_percept is p


def test_current_percept_cleared_to_none_on_idle_tick():
    p = _percept()
    adapter = SimulationAdapter(_OneShotSource(p))
    adapter.next_observation(environment=None)  # emits p
    assert adapter.current_percept is p
    adapter.next_observation(environment=None)  # idle → None, no stale percept
    assert adapter.current_percept is None


def test_no_percept_leaks_into_runtime_state_data():
    """The anti-leak guard: the persisted state surface stays Percept-free."""
    p = _percept()
    adapter = SimulationAdapter(_OneShotSource(p))
    state = RuntimeState()

    obs = adapter.next_observation(environment=None)
    state.update(obs)

    # No key holds a Percept, and the loaded/persisted snapshot is clean.
    assert "percept" not in state.data
    assert all(not isinstance(v, Percept) for v in state.data.values())
    snap = state.snapshot()
    assert all(not isinstance(v, Percept) for v in snap["data"].values())


def test_null_adapter_current_percept_is_none():
    adapter = NullSimulationAdapter()
    assert adapter.current_percept is None
