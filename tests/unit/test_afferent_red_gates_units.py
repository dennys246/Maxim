"""GL3.B0 strict red gates, the unit-level half (thalamic_relay.md §6 "GL3.B0").

GL3.B0 is the census-and-red-gates stage of the thalamic relay plan: tests only,
no ``src/`` change. Each gate here pins one census gap from §2 as a
``xfail(strict=True)`` test that drives the REAL producer path and fails today
for exactly the stated reason. ``raises=AssertionError`` on every gate means a
fixture error (an import, a constructor, a wiring break) is reported as a
FAILURE, not silently counted as the expected red; only the gate's own assert
can keep it red. When the stage named in ``reason=`` lands, the gate XPASSes and
strict mode turns that into a failure: remove the marker then, never before, and
never re-point a gate that fails to flip (CLAUDE.md, "a red gate that does not
flip is data").

Gates and the stage that flips each:

(a) An IMAGINED affordance encode creates no NAc eligibility on its node
    (§5.7 rule; TR12 strict default "imagined creates none"). Today
    ``LinguisticEncoder.encode_decomposed`` calls ``nac.update_eligibility``
    for every chunk, so credit can reach an imagined name. Flips with GL3.B1.
(b) A Minecraft game-event percept carries a non-empty ``agent_id`` equal to its
    AUT's (§2 gap 2, F0.5), driven through ``build_minecraft_aut`` (the only
    production construction of the source; the real client holds no agent_id).
    Today ``MinecraftPerceptSource.next_percept`` calls ``make_text_percept``
    without ``agent_id``. Flips with GL3.B1.
(c) Reachy ``DoAFeed``'s two lanes (the sensor lane ``world_set_azimuth`` and
    the percept lane ``make_audio_percept`` -> ``percept_sink``) carry ONE
    physical event id (§2 gap 1; §5.8 ``reachy.doa`` row). Today no pid exists
    on either lane. Flips with GL3.B8.
(g) The EC node a narrator-tool consequence (``SetEntitySensorTool``) reaches
    through the body channel encode carries per-node provenance ``narrated``,
    never ``experienced`` (§5.7 item 2, owner decision G6). Today EC keeps no
    per-node provenance set. Flips with GL3.B1.

Gates (d), (e) and (f) run through ``_loop_harness`` and live with it.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _substrate_stack():
    """A real EC + ATL + NAc trio (the shape test_imagination.py builds).

    Under the test isolation root the sentence model is not cached, so
    ``LinguisticEncoder`` uses its deterministic hash fallback. The gates here
    do not depend on embedding quality: eligibility and provenance are written
    per encoded node whatever the vector is.
    """
    from maxim.decisions.nac import NAc
    from maxim.memory.atl import ATL
    from maxim.similarity.ec import ECConfig, EntorhinalCortex

    return EntorhinalCortex(ECConfig(pattern_complete_threshold=0.50)), ATL(), NAc()


def _fixture(ok: object, msg: str) -> None:
    """A precondition of the gate, not the gate itself.

    ``pytest.fail`` raises ``Failed``, which is not an ``AssertionError``, so a
    broken fixture escapes ``xfail(raises=AssertionError)`` and fails loudly
    instead of being counted as the expected red.
    """
    if not ok:
        pytest.fail(f"fixture broken (not the gate): {msg}", pytrace=False)


def _pid_of(obj: Any) -> Any:
    """The physical event id a lane record carries, wherever it is stamped.

    No pid surface exists today, so this looks in every place a GL3.B8 stamp
    could reasonably land on a percept: a ``pid`` attribute, the context's
    ``pid``, or ``metadata["pid"]``.
    """
    pid = getattr(obj, "pid", None)
    if pid is None:
        pid = getattr(getattr(obj, "context", None), "pid", None)
    if pid is None:
        meta = getattr(obj, "metadata", None)
        if isinstance(meta, dict):
            pid = meta.get("pid")
    return pid


# ---------------------------------------------------------------------------
# (a) imagined affordance encode -> no eligibility
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flips with GL3.B1: provenance= on LinguisticEncoder.encode*; imagined creates no NAc eligibility",
)
def test_gl3b0_a_imagined_affordance_encode_creates_no_eligibility():
    from maxim.embodiment.entity_map import EntityMap
    from maxim.imagination.cache import ImaginationCache
    from maxim.imagination.designer import DesignResult
    from maxim.imagination.trigger import ImaginationTrigger
    from maxim.similarity.encoder import LinguisticEncoder

    ec, atl, nac = _substrate_stack()
    encoder = LinguisticEncoder(ec=ec, atl=atl, nac=nac)

    index = MagicMock()
    index.find.return_value = None  # novel phrase: no seed component matches
    registry = MagicMock()
    designer = MagicMock()
    designer.imagine.return_value = DesignResult(
        ref="creatures/giant_spider",
        spec={
            "entity": {
                "name": "giant_spider",
                "entity_type": "creature",
                "modulators": {
                    "combat": {
                        "name": "combat",
                        "abstract": True,
                        "affordances": {"web_spit": {"description": "spit a sticky web"}},
                    },
                },
            },
        },
        synonyms=[],
        validation_warnings=(),
    )
    trigger = ImaginationTrigger(
        component_index=index,
        component_registry=registry,
        designer=designer,
        cache=ImaginationCache(),
        encoder=encoder,
        agent_id="aut-1",
        imagination_threshold=1,
    )
    trigger._entity_map = EntityMap()  # the orchestrator's wiring seam

    # The production imagined path: process_percept -> _resolve_phrase -> designer
    # -> register_ephemeral(provenance="imagined") -> _encode_entity_affordances.
    results = trigger.process_percept("You see a giant spider.")
    _fixture([r.imagined for r in results] == [True], f"the designer path must run: {results}")
    _fixture(registry.register_ephemeral.call_count == 1, "register_ephemeral called once")
    _fixture(
        registry.register_ephemeral.call_args.kwargs.get("provenance") == "imagined",
        "the component is registered as imagined",
    )

    imagined_nodes = [nid for nid in ec._substrate_nodes]
    _fixture(imagined_nodes, "the imagined affordance must reach EC")

    # Only a POSITIVE trace is eligibility (credit is proportional to it): a fix that records an
    # imagined node at activation 0.0 carries no credit and must flip this gate.
    with_eligibility = {
        nid: nac._eligibility[("aut-1", nid)]
        for nid in imagined_nodes
        if nac._eligibility.get(("aut-1", nid), 0.0) > 0.0
    }
    assert with_eligibility == {}, f"imagined affordance nodes carry NAc eligibility: {with_eligibility}"


# ---------------------------------------------------------------------------
# (b) Minecraft event percept -> non-empty agent_id
# ---------------------------------------------------------------------------


class _OneEventWorld:
    """``_loop_harness._ScriptedWorld`` (the scripted ``MinecraftClient`` surface the backend and the
    percept source read) with ONE game event queued, in the shape ``MinecraftClient`` queues from the
    bridge (``{"kind", "text"}``). Built lazily so the harness import stays inside the test."""

    @staticmethod
    def build(event: dict[str, str]) -> Any:
        from tests.unit._loop_harness import _ScriptedWorld, _StepClock

        class _World(_ScriptedWorld):
            def __init__(self) -> None:
                super().__init__(_StepClock(), submerged=False)
                self._events = [dict(event)]

            def has_events(self) -> bool:
                return bool(self._events)

            def pop_event(self) -> dict[str, str] | None:  # type: ignore[override]
                return self._events.pop(0) if self._events else None

        return _World()


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flips with GL3.B1: Minecraft event percepts carry the agent_id (F0.5)",
)
def test_gl3b0_b_minecraft_event_percept_carries_agent_id(tmp_path):
    """Through the REAL composition: ``build_minecraft_aut`` (the only production construction of
    ``MinecraftPerceptSource``) owns the ``agent_id``; the real ``MinecraftClient`` carries none, so the
    stamp must come from the builder, and this gate reads the AUT's own ``percept_source``."""
    from maxim.simulation.minecraft_harness import build_minecraft_aut

    agent_id = "steve-aut"
    world = _OneEventWorld.build({"kind": "damage", "text": "you took 3.0 damage from zombie"})
    aut = build_minecraft_aut(agent_id=agent_id, bridge_port=0, persistence_dir=str(tmp_path / agent_id), client=world)
    try:
        _fixture(aut.agent_id == agent_id and aut.client is world, "the AUT wraps the scripted client")
        _fixture(aut.percept_source.has_pending(), "one damage event queued on the AUT's percept source")
        percept = aut.percept_source.next_percept()
        _fixture(percept is not None and "[minecraft:damage]" in percept.content, "the event became a percept")
        _fixture(not aut.percept_source.has_pending(), "exactly one event was queued")
    finally:
        aut.bio.on_session_end()

    stamped = percept.context.agent_id if percept.context is not None else None
    assert stamped and stamped == aut.agent_id, (
        f"the AUT {aut.agent_id!r}'s Minecraft damage percept carries agent_id {stamped!r}"
    )


# ---------------------------------------------------------------------------
# (c) DoAFeed's two lanes share one pid
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)  # DoAFeed.run runs in real time (its sample poll / timeout)
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flips with GL3.B8: reachy.doa receptor stamps one PhysicalEventId on both DoA lanes",
)
def test_gl3b0_c_doa_feed_lanes_share_a_pid(monkeypatch):
    import maxim.embodiment.audio_localization as audio_localization
    from maxim.embodiment.audio_localization import DoAFeed
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry

    body = ComponentRegistry().instantiate("bodies/reachy_mini")
    emb = Embodiment(body)
    stop = threading.Event()

    burst = [(0.0, True)] * 3  # one speech-gated reading (DoA 0 rad -> azimuth -1.0), then silence

    def reader():
        if burst:
            return burst.pop(0)
        stop.set()
        return None

    # Sensor lane: record each world_set_azimuth call (kwargs included) and
    # delegate to the real writer, so the body is still written.
    sensor_writes: list[dict[str, Any]] = []
    real_world_set = audio_localization.world_set_azimuth

    def recording_world_set(embodiment, azimuth, **kwargs):
        sensor_writes.append({"azimuth": azimuth, **kwargs})
        return real_world_set(embodiment, azimuth, **kwargs)

    monkeypatch.setattr(audio_localization, "world_set_azimuth", recording_world_set)

    # Percept lane: the sink the runtime wires to adapter.carry_percept.
    percepts: list[Any] = []
    feed = DoAFeed(
        reader,
        emb,
        stop_event=stop,
        percept_sink=percepts.append,
        agent_id="reachy-aut",
        sample_poll_s=0.0,
        sample_timeout_s=0.5,
    )
    feed.run()

    _fixture(len(sensor_writes) == 1 and body.vital_metrics["azimuth"] == -1.0, "sensor lane fired once")
    _fixture(len(percepts) == 1, "percept lane fired once")

    sensor_pid = sensor_writes[0].get("pid")
    percept_pid = _pid_of(percepts[0])
    assert sensor_pid is not None and percept_pid is not None and sensor_pid == percept_pid, (
        f"DoA lanes share no physical event id: sensor lane pid={sensor_pid!r}, percept lane pid={percept_pid!r}"
    )


# ---------------------------------------------------------------------------
# (g) narrator-tool consequence -> EC node provenance "narrated"
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="flips with GL3.B1: EC per-node provenance set; narrator-tool writes stamp 'narrated' (G6)",
)
def test_gl3b0_g_narrator_tool_consequence_reaches_ec_as_narrated():
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.embodiment.entity_map import EntityMap
    from maxim.embodiment.sensory_streams import INTEROCEPTION_TAG
    from maxim.runtime.substrate_proposal import _encode_current_clusters
    from maxim.similarity.encoder import SensorEncoder
    from maxim.simulation.tools import SetEntitySensorTool

    body = ComponentRegistry().instantiate("bodies/infant_humanoid")
    emb = Embodiment(body)
    ec, atl, nac = _substrate_stack()

    # The narrator's consequence write on the AUT body (orch_registry tool).
    out = SetEntitySensorTool(embodiment=emb, entity_map=EntityMap()).execute(
        sensor="hunger", value=0.9, source="narrator"
    )
    _fixture(out.success, f"narrator write failed: {out.error}")
    _fixture(body.vital_metrics["hunger"] == pytest.approx(0.9), "the narrator write landed on the body")

    # The body channels' pull read of the written sensors reaches EC.
    clusters = _encode_current_clusters(
        SensorEncoder(ec=ec, atl=atl, nac=nac), "aut-1", SimpleNamespace(embodiment=emb)
    )
    node_id = clusters.get(INTEROCEPTION_TAG)
    _fixture(node_id, f"the interoception channel must reach an EC node: {clusters}")

    meta = ec.substrate_node_metadata(node_id)
    _fixture(meta is not None, "EC knows the node")
    provenance = meta.get("provenance")
    assert provenance is not None and "narrated" in provenance and "experienced" not in provenance, (
        f"EC node reached by a narrator-tool write has provenance {provenance!r}; expected 'narrated', never 'experienced'"
    )
