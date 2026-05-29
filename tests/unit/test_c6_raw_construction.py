"""C6: hard TypeError on raw construction of PainBus / ReactionBus / MemoryHub.

In 0.9.x, raw construction (e.g. ``PainBus()``) emitted a
``DeprecationWarning`` that named the canonical builder.

In 1.0 this is a hard ``TypeError`` (v1_refinement.md §C6 + §1.1-T4).
Production code routes through the builder (``build_pain_bus`` /
``build_reaction_bus`` / ``build_memory_hub``), which constructs the
underlying object with ``_allow_raw=True`` internally so no error
fires. Tests that need a bare object pass ``_allow_raw=True`` explicitly.

Path (b) was chosen for the C6 flip: ``DefaultNetwork._init_pain_circuit``
keeps its explicit ``PainBus(_allow_raw=True)`` opt-out (deferred to
the Wave-2 ``memory_hub_unification.md`` split-subscriber-ownership
fix). The CI grep guard allow-lists that single production opt-out
site.

Each subject has three test shapes:
    (a) raw construction without ``_allow_raw`` → raises TypeError
    (b) raw construction with ``_allow_raw=True``  → silent
    (c) construction via builder                   → silent
"""

from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# PainBus
# ─────────────────────────────────────────────────────────────────────────────


class TestPainBusRawConstructionRaises:
    def test_raw_construction_without_allow_raw_raises(self) -> None:
        from maxim.proprioception.pain_bus import PainBus

        with pytest.raises(TypeError, match=r"build_pain_bus"):
            PainBus()

    def test_raw_construction_error_mentions_c6_and_canonical_door(self) -> None:
        from maxim.proprioception.pain_bus import PainBus

        with pytest.raises(TypeError) as exc:
            PainBus()

        msg = str(exc.value)
        assert "PainBus" in msg
        assert "build_pain_bus" in msg
        assert "(C6)" in msg

    def test_allow_raw_true_silent(self) -> None:
        from maxim.proprioception.pain_bus import PainBus

        PainBus(_allow_raw=True)  # must not raise

    def test_builder_no_error(self) -> None:
        from maxim.proprioception.pain_bus import build_pain_bus

        bus = build_pain_bus(hippocampus=None, nac=None)
        assert bus is not None


# ─────────────────────────────────────────────────────────────────────────────
# ReactionBus
# ─────────────────────────────────────────────────────────────────────────────


class TestReactionBusRawConstructionRaises:
    def test_raw_construction_without_allow_raw_raises(self) -> None:
        from maxim.reactions.bus import ReactionBus

        with pytest.raises(TypeError, match=r"build_reaction_bus"):
            ReactionBus()

    def test_raw_construction_error_mentions_c6(self) -> None:
        from maxim.reactions.bus import ReactionBus

        with pytest.raises(TypeError) as exc:
            ReactionBus()

        msg = str(exc.value)
        assert "ReactionBus" in msg
        assert "build_reaction_bus" in msg
        assert "(C6)" in msg

    def test_allow_raw_true_silent(self) -> None:
        from maxim.reactions.bus import ReactionBus

        ReactionBus(_allow_raw=True)

    def test_builder_no_error(self) -> None:
        from maxim.reactions.bus import build_reaction_bus

        bus = build_reaction_bus()
        assert bus is not None

    def test_pain_bus_internal_reaction_bus_silent(self) -> None:
        """PainBus constructs ReactionBus internally with _allow_raw=True;
        the outer PainBus(_allow_raw=True) call must propagate so the
        internal ReactionBus does not itself raise."""
        from maxim.proprioception.pain_bus import PainBus

        PainBus(_allow_raw=True)  # must not raise (no inner ReactionBus TypeError)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryHub
# ─────────────────────────────────────────────────────────────────────────────


def _make_memory_hub_core_systems():
    """Return the four required core bio-systems for MemoryHub."""
    from maxim.decisions.nac import NAc
    from maxim.memory.hippocampus import Hippocampus
    from maxim.similarity.ec import EntorhinalCortex
    from maxim.time.scn import SCN

    return {
        "hippocampus": Hippocampus(),
        "scn": SCN(),
        "nac": NAc(),
        "ec": EntorhinalCortex(),
    }


class TestMemoryHubRawConstructionRaises:
    def test_raw_construction_without_allow_raw_raises(self) -> None:
        from maxim.integration.memory_hub import MemoryHub

        core = _make_memory_hub_core_systems()
        with pytest.raises(TypeError, match=r"build_memory_hub"):
            MemoryHub(**core)

    def test_raw_construction_error_mentions_c6(self) -> None:
        from maxim.integration.memory_hub import MemoryHub

        core = _make_memory_hub_core_systems()
        with pytest.raises(TypeError) as exc:
            MemoryHub(**core)

        msg = str(exc.value)
        assert "MemoryHub" in msg
        assert "build_memory_hub" in msg
        assert "(C6)" in msg

    def test_allow_raw_true_silent(self) -> None:
        from maxim.integration.memory_hub import MemoryHub

        core = _make_memory_hub_core_systems()
        hub = MemoryHub(**core, _allow_raw=True)
        assert hub is not None

    def test_builder_no_error(self) -> None:
        from maxim.integration.memory_hub import build_memory_hub

        core = _make_memory_hub_core_systems()
        hub = build_memory_hub(agent_id="default_agent", **core)
        assert hub is not None

    def test_allow_raw_kw_only(self) -> None:
        """``_allow_raw`` must be keyword-only (per the established pattern).

        Passing it positionally would silently bind to whatever positional
        slot exists, which could be a sibling field. Verify it's
        keyword-only by checking the dataclass field metadata.
        """
        from dataclasses import fields

        from maxim.integration.memory_hub import MemoryHub

        for f in fields(MemoryHub):
            if f.name == "_allow_raw":
                assert f.kw_only is True, "_allow_raw must be keyword-only"
                return
        pytest.fail("_allow_raw field not found on MemoryHub")


# ─────────────────────────────────────────────────────────────────────────────
# Cross-cutting: builders propagate _allow_raw=True so production paths
# stay silent. This was the load-bearing rule the 0.9 deprecation cycle
# pinned; the 1.0 flip turns it from "no warning" into "no exception."
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildersDoNotRaise:
    def test_build_pain_bus_constructs_pain_bus_and_reaction_bus(self) -> None:
        from maxim.proprioception.pain_bus import build_pain_bus

        bus = build_pain_bus(hippocampus=None, nac=None)
        assert bus is not None
        assert bus.reaction_bus is not None

    def test_build_reaction_bus_constructs_reaction_bus(self) -> None:
        from maxim.reactions.bus import build_reaction_bus

        bus = build_reaction_bus()
        assert bus is not None

    def test_build_memory_hub_constructs_memory_hub(self) -> None:
        from maxim.integration.memory_hub import build_memory_hub

        core = _make_memory_hub_core_systems()
        hub = build_memory_hub(agent_id="default_agent", **core)
        assert hub is not None


# ─────────────────────────────────────────────────────────────────────────────
# DefaultNetwork opt-out (path (b)): the one allowed production
# _allow_raw=True site stays alive after the flip.
# ─────────────────────────────────────────────────────────────────────────────


class TestDefaultNetworkOptOutStillWorks:
    def test_default_network_pain_bus_opt_out_construct_silent(self) -> None:
        """Path (b) of the C6 flip keeps DefaultNetwork's explicit
        ``PainBus(_allow_raw=True)`` opt-out alive — Wave-2's
        split-subscriber-ownership fix is the proper resolution but
        ships separately. This test pins the contract so the next
        contributor doesn't accidentally tighten the door."""
        from maxim.proprioception.pain_bus import PainBus

        # Mirror the DefaultNetwork._init_pain_circuit call shape.
        bus = PainBus(_allow_raw=True)
        assert bus is not None
