"""C6 deprecation: raw construction warnings for PainBus / ReactionBus / MemoryHub.

In 0.9, raw construction (e.g. ``PainBus()``) emits a ``DeprecationWarning``
that names the canonical builder. Production code routes through the builder
(``build_pain_bus`` / ``build_reaction_bus`` / ``build_memory_hub``), which
constructs the underlying object with ``_allow_raw=True`` internally so no
warning fires. Tests that need a bare object pass ``_allow_raw=True``
explicitly.

In 1.0 this becomes a hard ``TypeError`` (1.1-T4 follow-up). See
``docs/plans/v1_refinement.md`` §C6 + §1.1-T4.

Each subject has three test shapes:
    (a) raw construction without ``_allow_raw`` → emits DeprecationWarning
    (b) raw construction with ``_allow_raw=True``  → silent
    (c) construction via builder                   → silent
"""

from __future__ import annotations

import warnings

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# PainBus
# ─────────────────────────────────────────────────────────────────────────────


class TestPainBusDeprecationWarning:
    def test_raw_construction_without_allow_raw_warns(self) -> None:
        from maxim.proprioception.pain_bus import PainBus

        with pytest.warns(DeprecationWarning, match=r"build_pain_bus"):
            PainBus()

    def test_raw_construction_warning_mentions_c6_and_canonical_door(self) -> None:
        from maxim.proprioception.pain_bus import PainBus

        with pytest.warns(DeprecationWarning) as record:
            PainBus()

        msg = str(record[0].message)
        assert "PainBus" in msg
        assert "build_pain_bus" in msg
        assert "C6" in msg
        assert "1.0" in msg

    def test_allow_raw_true_silent(self) -> None:
        from maxim.proprioception.pain_bus import PainBus

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            PainBus(_allow_raw=True)  # must not raise

    def test_builder_no_warning(self) -> None:
        from maxim.proprioception.pain_bus import build_pain_bus

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            bus = build_pain_bus(hippocampus=None, nac=None)
        assert bus is not None


# ─────────────────────────────────────────────────────────────────────────────
# ReactionBus
# ─────────────────────────────────────────────────────────────────────────────


class TestReactionBusDeprecationWarning:
    def test_raw_construction_without_allow_raw_warns(self) -> None:
        from maxim.reactions.bus import ReactionBus

        with pytest.warns(DeprecationWarning, match=r"build_reaction_bus"):
            ReactionBus()

    def test_raw_construction_warning_mentions_c6(self) -> None:
        from maxim.reactions.bus import ReactionBus

        with pytest.warns(DeprecationWarning) as record:
            ReactionBus()

        msg = str(record[0].message)
        assert "ReactionBus" in msg
        assert "build_reaction_bus" in msg
        assert "C6" in msg
        assert "1.0" in msg

    def test_allow_raw_true_silent(self) -> None:
        from maxim.reactions.bus import ReactionBus

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ReactionBus(_allow_raw=True)

    def test_builder_no_warning(self) -> None:
        from maxim.reactions.bus import build_reaction_bus

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            bus = build_reaction_bus()
        assert bus is not None

    def test_pain_bus_internal_reaction_bus_silent(self) -> None:
        """PainBus constructs ReactionBus internally with _allow_raw=True;
        the only warning that fires is PainBus's own (when raw)."""
        from maxim.proprioception.pain_bus import PainBus

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            PainBus(_allow_raw=True)

        # PainBus itself opts out via _allow_raw=True, AND its internal
        # ReactionBus call also passes _allow_raw=True. So zero warnings.
        c6_warnings = [w for w in caught if "C6 deprecation" in str(w.message)]
        assert c6_warnings == []


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


class TestMemoryHubDeprecationWarning:
    def test_raw_construction_without_allow_raw_warns(self) -> None:
        from maxim.integration.memory_hub import MemoryHub

        core = _make_memory_hub_core_systems()
        with pytest.warns(DeprecationWarning, match=r"build_memory_hub"):
            MemoryHub(**core)

    def test_raw_construction_warning_mentions_c6(self) -> None:
        from maxim.integration.memory_hub import MemoryHub

        core = _make_memory_hub_core_systems()
        with pytest.warns(DeprecationWarning) as record:
            MemoryHub(**core)

        c6 = [r for r in record if "C6 deprecation" in str(r.message)]
        assert len(c6) == 1
        msg = str(c6[0].message)
        assert "MemoryHub" in msg
        assert "build_memory_hub" in msg
        assert "1.0" in msg

    def test_allow_raw_true_silent(self) -> None:
        from maxim.integration.memory_hub import MemoryHub

        core = _make_memory_hub_core_systems()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            MemoryHub(**core, _allow_raw=True)

        c6 = [w for w in caught if "C6 deprecation" in str(w.message)]
        assert c6 == []

    def test_builder_no_warning(self) -> None:
        from maxim.integration.memory_hub import build_memory_hub

        core = _make_memory_hub_core_systems()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            hub = build_memory_hub(**core)

        c6 = [w for w in caught if "C6 deprecation" in str(w.message)]
        assert c6 == []
        assert hub is not None

    def test_allow_raw_kw_only(self) -> None:
        """_allow_raw must be keyword-only (per the established pattern).

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
# Cross-cutting: builders pass _allow_raw=True internally
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildersSilenceWarnings:
    """The whole point of C6 is that production code routes through builders
    and never sees the warning. Each builder must propagate _allow_raw=True
    to the underlying constructor."""

    def test_build_pain_bus_silences_pain_bus_and_reaction_bus(self) -> None:
        from maxim.proprioception.pain_bus import build_pain_bus

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            build_pain_bus(hippocampus=None, nac=None)

        c6 = [w for w in caught if "C6 deprecation" in str(w.message)]
        assert c6 == []

    def test_build_reaction_bus_silences_reaction_bus(self) -> None:
        from maxim.reactions.bus import build_reaction_bus

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            build_reaction_bus()

        c6 = [w for w in caught if "C6 deprecation" in str(w.message)]
        assert c6 == []

    def test_build_memory_hub_silences_memory_hub(self) -> None:
        from maxim.integration.memory_hub import build_memory_hub

        core = _make_memory_hub_core_systems()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            build_memory_hub(**core)

        c6 = [w for w in caught if "C6 deprecation" in str(w.message)]
        assert c6 == []
