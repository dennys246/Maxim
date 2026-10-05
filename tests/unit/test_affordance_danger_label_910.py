"""#910: the affordance annotators promise only what they can produce.

``tools/discovery.py::SensePresenceTool._annotate_aff`` and
``integration/bio_enrichment.py::BioEnrichmentPipeline._annotate_affordance_valence`` and the substrate
fallback of ``tools/discovery.py::SenseToolsTool._nac_annotation`` read ``NAc.reward_bias``, which every
writer clamps to ``[0, max]``. Their danger branches could never fire: an affordance that hurt the agent read
the same as one it never tried. Owner decisions 2026-10-04 (engram plan E3, option 2, extended to the third
site): delete the dead branches and pin the honest contract, "effective or unlabeled". Reading the store that
does hold danger (``percept_valences`` / ``cluster_fear``) is option 1, deferred; the strict xfail below is
its revive marker. The deletion itself is guarded by the injected-negative-bias tests in
``test_tool_discovery.py`` and ``test_bio_enrichment.py``, which fail if a danger branch returns.
"""

from __future__ import annotations

import pytest

AFFORDANCE = "breathe_fire"  # AFFORDANCE_STRATEGY chunks: "breathe fire", "breathe", "fire"


@pytest.fixture
def substrate(tmp_path):
    from maxim.decisions.nac import NAc, NACConfig
    from maxim.memory.atl import ATL, ATLConfig

    atl = ATL(ATLConfig(persistence_path=str(tmp_path / "atl.json")))
    node, _ = atl.find_or_create(name="fire", category="substrate", definition="fire")
    return NAc(NACConfig()), atl, node


def _sense_tool(nac, atl):
    from maxim.embodiment.entity_map import EntityMap
    from maxim.tools.discovery import SensePresenceTool

    return SensePresenceTool(entity_map=EntityMap(), nac=nac, atl=atl, agent_id="a")


def _pipeline(nac, atl):
    from maxim.integration.bio_enrichment import BioEnrichmentPipeline

    return BioEnrichmentPipeline(nac=nac, atl=atl, agent_id="a")


def _annotations(nac, atl) -> tuple[str, str]:
    return _sense_tool(nac, atl)._annotate_aff(AFFORDANCE), _pipeline(nac, atl)._annotate_affordance_valence(AFFORDANCE)


def _sense_tools(nac, atl):
    from unittest.mock import MagicMock

    from maxim.tools.discovery import SenseToolsTool

    registry = MagicMock()
    registry.get.return_value = MagicMock(_affordance_name=AFFORDANCE)
    return SenseToolsTool(entity_map=MagicMock(), tool_registry=registry, nac=nac, atl=atl, agent_id="a")


def test_only_negative_experience_annotates_as_unlabeled(substrate):
    """The honest contract: harm clamps out of ``reward_bias``, so the label says nothing, not "dangerous"."""
    nac, atl, node = substrate
    nac.credit_node("a", node, -0.8)
    assert nac.reward_bias("a", node) == 0.0
    assert _annotations(nac, atl) == (AFFORDANCE, AFFORDANCE)


def test_harm_after_reward_removes_the_effective_label(substrate):
    """The one way harm shows through ``reward_bias``: it clamps the bias back to 0, which drops the label."""
    nac, atl, node = substrate
    nac.credit_node("a", node, 0.5)
    assert _annotations(nac, atl)[0] == f"{AFFORDANCE} [effective]"
    nac.credit_node("a", node, -0.8)
    assert _annotations(nac, atl) == (AFFORDANCE, AFFORDANCE)
    assert _sense_tools(nac, atl)._nac_annotation(f"dragon_{AFFORDANCE}") == ""


def test_a_rewarded_affordance_is_still_labeled_effective(substrate):
    nac, atl, node = substrate
    nac.credit_node("a", node, 0.5)
    sense, pipeline = _annotations(nac, atl)
    assert sense == f"{AFFORDANCE} [effective]"
    assert pipeline.startswith(f"{AFFORDANCE} [effective")
    assert _sense_tools(nac, atl)._nac_annotation(f"dragon_{AFFORDANCE}") == "similar affordance worked well"


@pytest.mark.xfail(strict=True, reason="#910: the annotation swallows every fault with except Exception: pass")
def test_a_fault_in_the_sense_annotation_is_reported_not_swallowed(substrate, monkeypatch):
    """``_annotate_aff`` wrapped everything in ``except Exception: pass``."""
    import maxim.tools.discovery as discovery

    nac, atl, _ = substrate
    reported: list[bool] = []
    monkeypatch.setattr(discovery, "log_swallowed_exception", lambda *a, **k: reported.append(True), raising=False)

    def _boom(*_a, **_k):
        raise RuntimeError("atl down")

    monkeypatch.setattr(atl, "recall", _boom)
    assert _sense_tool(nac, atl)._annotate_aff(AFFORDANCE) == AFFORDANCE  # the tool still answers...
    assert reported == [True]  # ...and the fault is reported


@pytest.mark.xfail(strict=True, reason="#910: the annotation swallows every fault with except Exception: pass")
def test_a_fault_in_the_sense_tools_annotation_is_reported_not_swallowed(substrate, monkeypatch):
    """``_nac_annotation``'s substrate fallback wrapped everything in ``except Exception: pass``."""
    import maxim.tools.discovery as discovery

    nac, atl, _ = substrate
    reported: list[bool] = []
    monkeypatch.setattr(discovery, "log_swallowed_exception", lambda *a, **k: reported.append(True), raising=False)

    def _boom(*_a, **_k):
        raise RuntimeError("atl down")

    monkeypatch.setattr(atl, "recall", _boom)
    assert _sense_tools(nac, atl)._nac_annotation(f"dragon_{AFFORDANCE}") == ""
    assert reported == [True]


@pytest.mark.xfail(
    strict=True,
    reason="#910 option 1 (deferred): the annotators do not read the stores that hold learned danger",
)
def test_learned_harm_in_the_percept_store_reaches_a_danger_label(substrate):
    """Revive marker for option 1. Harm is seeded where option 1 would read it -- the Pavlovian percept store,
    keyed as its only production writer keys it (``proprioception/pain_bus.py``: the OWNING entity's class, the
    YAML noun, never an affordance word) -- and never through ``reward_bias`` (un-clamping that is the wrong
    revival, and would flip this for the wrong reason). Option 1 must reach the owning entity ("dragon") from
    the affordance; that PR may change the call shape below to pass it, and removes the xfail when it flips."""
    nac, atl, _ = substrate
    nac.record_percept_valence("dragon", "burn", -0.8, agent_id="a")
    assert nac.get_percept_valence("dragon", "burn", agent_id="a") < 0  # the harm is really stored
    sense, pipeline = _annotations(nac, atl)
    assert "DANGER" in sense.upper() and "DANGER" in pipeline.upper()
