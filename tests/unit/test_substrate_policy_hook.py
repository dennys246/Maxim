"""The substrate policy hook (baseline floors B1, docs/plans/baseline_floors_b1_build.md PR-A).

A ``SubstratePolicy`` picks in place of ``NAc.recommend_action`` inside ``propose_via_substrate``; everything
before the pick (filter, encode, cluster note, cue, pain tick) and after it is shared, and ``policy=None`` is
today's path. Real Minecraft agent on the fake bridge, as the substrate wake tests do.
"""

from __future__ import annotations

import math
import threading
import time
from types import MappingProxyType

import pytest

from maxim.runtime.substrate_proposal import NO_SITUATION_CUE, SubstratePolicyError, propose_via_substrate


@pytest.fixture
def aut(tmp_path):
    from maxim.simulation.minecraft_harness import FakeBridgeServer, build_minecraft_aut

    srv = FakeBridgeServer(state_interval_s=0.1)
    agent = build_minecraft_aut(
        agent_id="policy_probe",
        bridge_port=srv.port,
        persistence_dir=str(tmp_path / "aut"),
        entity_ref="bodies/minecraft_player",
    )
    time.sleep(0.3)
    agent.backend.sync_world_sensors()
    try:
        yield agent
    finally:
        try:
            agent.client.close()
        except Exception:  # noqa: BLE001 — teardown of a fake bridge
            pass
        srv.close()


def _encoder(aut):
    from maxim.similarity.encoder import SensorEncoder, SensorEncoderConfig

    return SensorEncoder(ec=aut.bio.ec, atl=aut.bio.atl, nac=aut.bio.nac, config=SensorEncoderConfig())


class _Policy:
    def __init__(self, pick=None, tool_suffix="escape_water"):
        self.pick, self.tool_suffix, self.calls = pick, tool_suffix, []

    def select(self, **kw):
        self.calls.append(kw)
        if self.pick is not None:
            return self.pick
        tool = next(t for t in kw["available_tools"] if t.endswith(self.tool_suffix))
        return {"tool_name": tool, "params": {}, "confidence": 0.9, "reasoning": "scripted"}


def _propose(aut, policy, **kw):
    return propose_via_substrate(
        nac=aut.bio.nac,
        agent_id=aut.agent_id,
        executor=aut.executor,
        situation_cue=NO_SITUATION_CUE,
        sensor_encoder=_encoder(aut),
        policy=policy,
        **kw,
    )


def test_a_policy_picks_in_place_of_recommend_action(aut, monkeypatch):
    called = []
    monkeypatch.setattr(aut.bio.nac, "recommend_action", lambda **kw: called.append(kw))
    policy = _Policy()
    proposal = _propose(aut, policy)
    assert called == [], "recommend_action must not run when a policy is given"
    assert proposal is not None and proposal.action["tool_name"].endswith("escape_water")
    assert proposal.action["params"] == {}
    assert proposal.strategy_used == "substrate-primary"
    assert proposal.reasoning == "[_Policy] scripted"
    assert proposal.clusters, "the shared encode must still attach this tick's clusters"


def test_the_policy_sees_no_nac_and_a_read_only_world_observation(aut):
    policy = _Policy()
    _propose(aut, policy)
    (kw,) = policy.calls
    assert set(kw) == {"agent_id", "available_tools", "clusters", "observation"}
    assert isinstance(kw["observation"], MappingProxyType) and "oxygen" in kw["observation"]
    assert isinstance(kw["available_tools"], tuple)
    from maxim.tools.introspection import INTROSPECTION_TOOL_NAMES

    assert not set(kw["available_tools"]) & set(INTROSPECTION_TOOL_NAMES)


def test_none_from_a_policy_is_idle_never_a_fallback(aut, monkeypatch):
    called = []
    monkeypatch.setattr(aut.bio.nac, "recommend_action", lambda **kw: called.append(kw))

    class Idle:
        def select(self, **kw):
            return None

    assert _propose(aut, Idle()) is None
    assert called == []


def test_a_policy_bypasses_min_confidence(aut):
    tool = next(t for t in aut.executor.registry.list() if t.endswith("escape_water"))
    proposal = _propose(aut, _Policy(pick={"tool_name": tool, "confidence": 0.01}), min_confidence=0.9)
    assert proposal is not None and math.isclose(proposal.confidence, 0.01)


@pytest.mark.parametrize(
    "pick",
    [
        {"tool_name": "memory_recall"},  # an introspection tool the filter removed
        {"tool_name": "no_such_tool"},
        {"tool_name": None},
        "minecraft_player_escape_water",  # not a mapping
    ],
)
def test_a_pick_outside_the_available_tools_raises(aut, pick):
    with pytest.raises(SubstratePolicyError):
        _propose(aut, _Policy(pick=pick))


@pytest.mark.parametrize("bad", [{"params": [1, 2]}, {"confidence": float("inf")}, {"confidence": True}])
def test_malformed_params_or_confidence_raise(aut, bad):
    tool = next(t for t in aut.executor.registry.list() if t.endswith("escape_water"))
    with pytest.raises(SubstratePolicyError):
        _propose(aut, _Policy(pick={"tool_name": tool, **bad}))


def test_without_a_policy_recommend_action_runs_with_its_usual_arguments(aut, monkeypatch):
    seen = []
    monkeypatch.setattr(aut.bio.nac, "recommend_action", lambda **kw: seen.append(kw))
    assert _propose(aut, None) is None
    (kw,) = seen
    assert set(kw) == {
        "agent_id",
        "available_tools",
        "current_drives",
        "current_cluster_id",
        "current_clusters",
        "min_confidence",
    }


@pytest.mark.timeout(120)
def test_a_policy_drives_the_harness_loop_to_the_executor(aut):
    """The composition: a policy passed to ``run_minecraft_aut`` reaches ``propose_via_substrate`` through
    ``run_agentic_loop`` → ``substrate_tick``, and its pick is EXECUTED. RED when the threading in any of
    the three is removed (the loop would then fall back to recommend_action, which has no reason to pick
    ``escape_water`` on the fake bridge with no fear booked)."""
    from maxim.simulation.minecraft_harness import run_minecraft_aut

    executed: list[str] = []
    orig = aut.executor.execute

    def spy(action):
        executed.append((action or {}).get("tool_name"))
        return orig(action)

    aut.executor.execute = spy
    policy = _Policy()
    stop = threading.Event()
    th = threading.Thread(
        target=run_minecraft_aut,
        args=(aut,),
        kwargs={"max_steps": 100_000, "target_hz": 4.0, "stop_event": stop, "substrate_policy": policy},
        daemon=True,
    )
    try:
        th.start()
        deadline = time.monotonic() + 10.0
        while not any(str(t).endswith("escape_water") for t in executed) and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        stop.set()
        th.join(timeout=30.0)
    assert policy.calls, "the loop never asked the policy"
    assert any(str(t).endswith("escape_water") for t in executed), executed


def test_harness_kwargs_carry_the_policy_only_when_given(aut):
    import threading as _t

    from maxim.simulation.minecraft_harness import _loop_kwargs

    base = _loop_kwargs(aut, max_steps=1, stop_event=_t.Event(), target_hz=4.0)
    assert "substrate_policy" not in base
    policy = _Policy()
    assert (
        _loop_kwargs(aut, max_steps=1, stop_event=_t.Event(), target_hz=4.0, substrate_policy=policy)[
            "substrate_policy"
        ]
        is policy
    )
