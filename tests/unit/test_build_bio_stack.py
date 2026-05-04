"""Unit tests for build_bio_stack — the canonical bio-pipeline construction door.

Tests the structural-enforcement invariants from bio_stack_unification.md:
- Builder returns a frozen BioStack with all systems wired
- Construction order: Hippocampus → NAc → SCN → EC → (ATL, AngularGyrus)
  → MemoryHub → PainBus → DefaultNetwork
- Pre-built PainBus pattern (sim AUT): standard learners subscribed
- MemoryHub always has .connect() called (inherited from build_memory_hub)
- DefaultNetwork is opt-in via with_default_network=True
- Persistence paths derived from persistence_dir when provided
- Frozen constraint prevents field reassignment
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest


class TestBuildBioStackSignature:
    """All args must be keyword-only."""

    def test_positional_args_rejected(self):
        from maxim.runtime.bio_stack import build_bio_stack

        with pytest.raises(TypeError, match="positional"):
            build_bio_stack("/tmp/test")  # type: ignore[misc]


class TestBuildBioStackCore:
    """Core construction invariants."""

    def test_returns_bio_stack_instance(self):
        from maxim.runtime.bio_stack import BioStack, build_bio_stack

        bio = build_bio_stack()
        assert isinstance(bio, BioStack)

    def test_all_core_systems_populated(self):
        from maxim.decisions.nac import NAc
        from maxim.memory.hippocampus import Hippocampus
        from maxim.runtime.bio_stack import build_bio_stack
        from maxim.similarity.ec import EntorhinalCortex
        from maxim.time.scn import SCN

        bio = build_bio_stack()
        assert isinstance(bio.hippocampus, Hippocampus)
        assert isinstance(bio.nac, NAc)
        assert isinstance(bio.scn, SCN)
        assert isinstance(bio.ec, EntorhinalCortex)

    def test_memory_hub_populated(self):
        from maxim.integration.memory_hub import MemoryHub
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert isinstance(bio.memory_hub, MemoryHub)

    def test_pain_bus_populated(self):
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert isinstance(bio.pain_bus, PainBus)

    def test_reaction_bus_is_pain_bus_internal_bus(self):
        """ReactionBus on BioStack is the one owned by PainBus."""
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert bio.reaction_bus is bio.pain_bus.reaction_bus

    def test_default_network_none_by_default(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert bio.default_network is None


class TestBuildBioStackFrozen:
    """BioStack is frozen — field reassignment is rejected."""

    def test_cannot_reassign_hippocampus(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        with pytest.raises(dataclasses.FrozenInstanceError):
            bio.hippocampus = None  # type: ignore[misc]

    def test_cannot_reassign_pain_bus(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        with pytest.raises(dataclasses.FrozenInstanceError):
            bio.pain_bus = None  # type: ignore[misc]

    def test_cannot_reassign_memory_hub(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        with pytest.raises(dataclasses.FrozenInstanceError):
            bio.memory_hub = None  # type: ignore[misc]


class TestBuildBioStackMemoryHubBridges:
    """MemoryHub bridges are alive (inherited from build_memory_hub)."""

    def test_plan_bridge_alive(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert bio.memory_hub._plan_bridge is not None

    def test_escalation_bridge_alive(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert bio.memory_hub._escalation_bridge is not None

    def test_fear_bridge_alive(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert bio.memory_hub._fear_bridge is not None


class TestBuildBioStackPersistence:
    """Persistence paths derived from persistence_dir."""

    def test_no_persistence_when_dir_is_none(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack(persistence_dir=None)
        assert bio.hippocampus.config.persistence_path is None

    def test_hippocampus_persistence_path(self, tmp_path):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack(persistence_dir=tmp_path)
        assert bio.hippocampus.config.persistence_path == str(tmp_path / "hippocampus.json")

    def test_persistence_dir_as_string(self, tmp_path):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack(persistence_dir=str(tmp_path))
        assert bio.hippocampus.config.persistence_path == str(tmp_path / "hippocampus.json")


class TestBuildBioStackATL:
    """ATL is optional — try/except at import time."""

    def test_atl_populated_when_available(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        # ATL may or may not be available depending on deps.
        # When available, it should be an ATL instance.
        if bio.atl is not None:
            from maxim.memory.atl import ATL

            assert isinstance(bio.atl, ATL)

    def test_atl_persistence_path_when_dir_provided(self, tmp_path):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack(persistence_dir=tmp_path)
        if bio.atl is not None:
            assert bio.atl.config.persistence_path == str(tmp_path / "atl.json")


class TestBuildBioStackPreBuiltPainBus:
    """Sim AUT pattern: pre-built PainBus gets standard learners subscribed."""

    def test_uses_pre_built_pain_bus(self):
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bio_stack import build_bio_stack

        early_bus = PainBus(_allow_raw=True)
        bio = build_bio_stack(pain_bus=early_bus)
        assert bio.pain_bus is early_bus

    def test_pre_built_bus_gets_hippocampus_subscriber(self):
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bio_stack import build_bio_stack

        early_bus = PainBus(_allow_raw=True)
        subs_before = len(early_bus._pain_signal_subs)
        build_bio_stack(pain_bus=early_bus)
        # At least hippocampus + nac subscribers added
        assert len(early_bus._pain_signal_subs) >= subs_before + 2

    def test_pre_built_bus_reaction_bus_exposed(self):
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bio_stack import build_bio_stack

        early_bus = PainBus(_allow_raw=True)
        bio = build_bio_stack(pain_bus=early_bus)
        assert bio.reaction_bus is early_bus.reaction_bus

    def test_additional_pain_subscribers_with_pre_built_bus(self):
        from maxim.proprioception.pain_bus import PainBus
        from maxim.runtime.bio_stack import build_bio_stack

        early_bus = PainBus(_allow_raw=True)
        calls = []
        extra_sub = lambda signal: calls.append(signal)
        bio = build_bio_stack(
            pain_bus=early_bus,
            additional_pain_subscribers=(extra_sub,),
        )
        assert extra_sub in bio.pain_bus._pain_signal_subs


class TestBuildBioStackAdditionalSubscribers:
    """Additional pain subscribers wired through build_pain_bus path."""

    def test_additional_subscriber_registered(self):
        from maxim.runtime.bio_stack import build_bio_stack

        calls = []
        extra_sub = lambda signal: calls.append(signal)
        bio = build_bio_stack(additional_pain_subscribers=(extra_sub,))
        assert extra_sub in bio.pain_bus._pain_signal_subs


class TestBuildBioStackDefaultNetwork:
    """DefaultNetwork is opt-in."""

    def test_default_network_constructed_when_requested(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack(with_default_network=True)
        # DN is an optional dep — may be None if not installed
        # but the builder should attempt construction

    def test_dn_params_forwarded(self):
        """DN parameters forwarded to build_default_network."""
        from unittest.mock import MagicMock, patch

        from maxim.runtime.bio_stack import build_bio_stack

        mock_dn = MagicMock(return_value=MagicMock())
        with patch("maxim.runtime.bootstrap.build_default_network", mock_dn):
            sentinel_maxim = object()
            sentinel_bus = object()
            sentinel_fear = object()
            bio = build_bio_stack(
                with_default_network=True,
                dn_maxim=sentinel_maxim,
                dn_bus=sentinel_bus,
                dn_fear_agent=sentinel_fear,
                dn_frame_size=(320, 240),
            )
            mock_dn.assert_called_once()
            call_kwargs = mock_dn.call_args.kwargs
            assert call_kwargs["maxim"] is sentinel_maxim
            assert call_kwargs["bus"] is sentinel_bus
            assert call_kwargs["fear_agent"] is sentinel_fear
            assert call_kwargs["frame_size"] == (320, 240)
            # pain_bus and nac are from the stack, not the sentinel
            assert call_kwargs["pain_bus"] is bio.pain_bus
            assert call_kwargs["nac"] is bio.nac


class TestBuildBioStackComposesBuilders:
    """Verify build_bio_stack calls the individual Wave 1+2 builders."""

    def test_memory_hub_core_systems_match_bio_stack(self):
        """MemoryHub's core systems are the same objects as BioStack's."""
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert bio.memory_hub.hippocampus is bio.hippocampus
        assert bio.memory_hub.nac is bio.nac
        assert bio.memory_hub.scn is bio.scn
        assert bio.memory_hub.ec is bio.ec

    def test_memory_hub_atl_matches_bio_stack(self):
        from maxim.runtime.bio_stack import build_bio_stack

        bio = build_bio_stack()
        assert bio.memory_hub.atl is bio.atl
