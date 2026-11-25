"""#1228: in a dual-LLM run (``aut_model``) every AUT-side LLM consumer holds the AUT's own router.

LIVE before the fix: the AUT's FearAgent gate (``llm_for_fear = llm_router``) and its imagination designer ran on the
narrator's router, so any "the AUT ran on model X" statement was partly false. Also handed the AUT's router now: a
spawned sub-simulation's agent (owner decision A) and the AUT's ExecAgent slot, whose goal-proposal path is dormant
(``_worker_loop`` starts only from ``MaximAgent.on_start``, which no production runtime calls), so that slot held no
router and built none; it now holds the AUT's. (The factory's ``fear_llm`` was a dead slot: the AUT config has
``with_fear_gate=False``; the orchestrator no longer passes it.)

Driven through the real ``start_simulation_mode`` by the harness's ``aut_llm_slots`` probe arm (the imagination arm's
run, an embodied AUT, with ``aut_model`` set). Each build is a distinct ``_RouterView`` whose ``cfg.profile`` names the
model it serves, so a consumer's router says whose it is. The constructors are spied to record what each consumer was
handed; ``served_by`` records which view answered each sorted call.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.unit._orch_harness import assert_this_checkout, run_arm

assert_this_checkout()

AUT = "orch-golden-aut"  # the arm's aut_model
AUT_ROLES = {"aut", "fear", "designer"}  # the FakeRouter roles that are the AUT's own cognition
NARRATOR_ROLES = {"narrator", "manifest", "roundup"}  # world generation and the post-run roundup


def _whose(router: Any) -> str:
    if router is None:
        return "none"
    profile = getattr(getattr(router, "cfg", None), "profile", None)
    return "aut" if profile == AUT else "shared"


@pytest.fixture(scope="module")
def slots(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from maxim.agents.exec_agent import ExecAgent
    from maxim.agents.fear_agent import FearAgent
    from maxim.simulation.entity_designer import EntityDesigner
    from maxim.simulation.tools import SpawnSubSimulationTool

    held: dict[str, list[str]] = {"ExecAgent": [], "FearAgent": [], "EntityDesigner": [], "SpawnSubSimulationTool": []}
    mp = pytest.MonkeyPatch()

    def _spy(cls: Any, label: str, arg: str) -> None:
        orig = cls.__init__

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            held[label].append(_whose(kwargs.get(arg)))
            orig(self, *args, **kwargs)

        mp.setattr(cls, "__init__", init)

    _spy(ExecAgent, "ExecAgent", "shared_router")
    _spy(FearAgent, "FearAgent", "llm")
    _spy(EntityDesigner, "EntityDesigner", "llm_router")
    _spy(SpawnSubSimulationTool, "SpawnSubSimulationTool", "llm_router")
    try:
        trace = run_arm("aut_llm_slots", tmp_path_factory.mktemp("aut_llm_slots"))
    finally:
        mp.undo()
    return {"held": held, "trace": trace}


def test_the_run_is_dual_llm(slots: dict[str, Any]) -> None:
    assert [b["profile"] for b in slots["trace"]["router_builds"]] == [None, AUT]


def test_the_aut_fear_agent_holds_the_aut_router(slots: dict[str, Any]) -> None:
    assert slots["held"]["FearAgent"] == ["aut"], slots["held"]


def test_the_aut_imagination_designer_holds_the_aut_router(slots: dict[str, Any]) -> None:
    assert slots["held"]["EntityDesigner"] == ["aut"], slots["held"]


def test_the_sub_simulation_tool_is_handed_the_aut_router(slots: dict[str, Any]) -> None:
    """Owner decision A: a spawned sub-AUT is an agent under test (its slots: ``test_agent_router_slots_1228.py``)."""
    assert slots["held"]["SpawnSubSimulationTool"] == ["aut"], slots["held"]


def test_each_exec_agent_is_handed_its_own_agents_router(slots: dict[str, Any]) -> None:
    """The AUT's MaximAgent is built first, the narrator's second: each holds its own role's router (a dormant slot
    in the sim, so this pins the hand-off, not a live call)."""
    assert slots["held"]["ExecAgent"] == ["aut", "shared"], slots["held"]


def test_every_aut_call_is_served_by_the_aut_router(slots: dict[str, Any]) -> None:
    served = slots["trace"]["served_by"]
    roles = {role for _, role in served}
    assert {"aut", "designer", "narrator"} <= roles, served  # not vacuous: the AUT, its designer and the narrator ran
    wrong = [
        (build, role)
        for build, role in served
        if (role in AUT_ROLES and build != "<build-2>") or (role in NARRATOR_ROLES and build != "<build-1>")
    ]
    assert wrong == [], served
    assert slots["trace"]["unsorted_prompts"] == []
