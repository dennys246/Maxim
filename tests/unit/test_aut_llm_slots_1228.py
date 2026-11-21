"""#1228: in a dual-LLM run (``aut_model``) every AUT-side LLM consumer holds the AUT's own router.

The AUT's FearAgent got the narrator's router (``create_full_agent(fear_llm=llm_router)`` and ``llm_for_fear =
llm_router``), its imagination designer did too, and its ExecAgent was handed no router at all, so its goal proposals
built a private one from ``load_llm_config()`` on the narrator's profile. Any "the AUT ran on model X" statement was
then partly false.

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

    held: dict[str, list[str]] = {"ExecAgent": [], "FearAgent": [], "EntityDesigner": []}
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


def test_each_exec_agent_is_handed_its_own_agents_router(slots: dict[str, Any]) -> None:
    """The AUT's MaximAgent is built first, the narrator's second: neither ExecAgent builds a private router."""
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
