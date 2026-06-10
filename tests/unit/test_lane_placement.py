"""Phase 1 of lane_capability_placement_split.md — the placement axis.

These tests pin the *additive* placement type (``Origin`` / ``ProviderPlacement``
/ ``LaneConfig.placement``) and the one-way back-compat derivation shim. The
load-bearing assertion is the **equivalence matrix**: for every existing config
shape, ``placement_kind(derive_placement(cfg, ...))`` must equal what the legacy
``LaneBackendManager._classify`` returns today — so Phase 2 can switch dispatch
to read ``origin`` with a proven before/after equivalence and zero behavior drift.

Offline/deterministic by construction: cloud cases use a public IP literal
(``8.8.8.8``) so no DNS lookup is needed.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from maxim.runtime.lane_backends import (
    BackendGateError,
    LaneBackendManager,
    _is_cloud_url,
    derive_placement,
    placement_kind,
    primary_placement,
)
from maxim.runtime.worker_pool import LaneConfig, Origin, ProviderPlacement

# A public IP literal — classified as cloud by _is_cloud_url with no DNS lookup.
_CLOUD_URL = "http://8.8.8.8:8000/v1"
_PRIVATE_URL = "http://192.168.1.50:8100/v1"
_LOOPBACK_URL = "http://127.0.0.1:8100/v1"


def _lane(name: str = "large", **kw) -> LaneConfig:
    return LaneConfig(name=name, max_workers=1, **kw)


# ─── type shape ─────────────────────────────────────────────────────────────


class TestOriginEnum:
    def test_values(self):
        assert Origin.LOCAL.value == "local"
        assert Origin.CLOUD.value == "cloud"
        assert Origin.PEER.value == "peer"

    def test_str_subclass_compares_to_legacy_strings(self):
        # str-Enum so it serializes cleanly and equals its kind string.
        assert Origin.LOCAL == "local"
        assert Origin.CLOUD == "cloud"

    def test_json_serializes_to_plain_string(self):
        assert json.dumps({"o": Origin.PEER}) == '{"o": "peer"}'

    def test_str_returns_bare_value_not_enum_repr(self):
        # __str__ override closes the f"{origin}" -> "Origin.PEER" trap.
        assert str(Origin.PEER) == "peer"
        assert f"{Origin.LOCAL}" == "local"


class TestProviderPlacementShape:
    def test_frozen(self):
        p = ProviderPlacement(origin=Origin.LOCAL, model="mistral-7b")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.model = "other"  # type: ignore[misc]

    def test_defaults(self):
        p = ProviderPlacement(origin=Origin.LOCAL)
        assert p.model is None
        assert p.url is None
        assert p.api_key is None
        assert p.timeout_s is None
        assert p.extra == {}

    def test_hashable_despite_extra_dict(self):
        # extra is hash=False, compare=False per CC3 escape-hatch rules, so the
        # frozen dataclass stays hashable (usable inside the placement tuple).
        p = ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet", extra={"k": "v"})
        assert isinstance(hash(p), int)
        assert {p}  # set construction exercises __hash__

    def test_extra_excluded_from_equality(self):
        a = ProviderPlacement(origin=Origin.LOCAL, model="m", extra={"x": 1})
        b = ProviderPlacement(origin=Origin.LOCAL, model="m", extra={"y": 2})
        assert a == b  # extra has compare=False

    def test_extra_key_colliding_with_declared_field_rejected(self):
        # CC3 path-(a) guard mirrors LaneTierConfig: a colliding extra key must
        # not silently shadow a declared field on a future round-trip.
        with pytest.raises(ValueError, match="collide with declared fields"):
            ProviderPlacement(origin=Origin.LOCAL, extra={"model": "shadow"})

    def test_extra_noncolliding_key_accepted(self):
        p = ProviderPlacement(origin=Origin.LOCAL, extra={"routing_weight": 0.5})
        assert p.extra == {"routing_weight": 0.5}


class TestLaneConfigPlacementField:
    def test_defaults_empty(self):
        assert _lane().placement == ()

    def test_existing_configs_unaffected(self):
        # An ordinary local lane gains the field but nothing else changes.
        cfg = _lane(model_profile="smollm-1.7b-instruct")
        assert cfg.placement == ()
        assert cfg.model_profile == "smollm-1.7b-instruct"


# ─── derivation ─────────────────────────────────────────────────────────────


class TestDerivePlacement:
    def test_local_profile(self):
        cfg = _lane(model_profile="mistral-7b-instruct-v0.2")
        (p,) = derive_placement(cfg, peer_owned=False)
        assert p.origin is Origin.LOCAL
        assert p.model == "mistral-7b-instruct-v0.2"
        assert p.url is None

    def test_cloud_profile_with_no_url_is_local_origin(self):
        # Load-bearing: --cloud-lane sets a cloud *profile* with no remote_url;
        # _classify returns "local" so it does NOT consume a MAX_CLOUD_LANES
        # slot. Derivation must reproduce that (LOCAL origin), not CLOUD.
        cfg = _lane(model_profile="claude-sonnet")
        (p,) = derive_placement(cfg, peer_owned=False)
        assert p.origin is Origin.LOCAL

    def test_public_remote_is_cloud(self):
        cfg = _lane(model_profile="x", remote_url=_CLOUD_URL, remote_model="srv-model")
        (p,) = derive_placement(cfg, peer_owned=False)
        assert p.origin is Origin.CLOUD
        assert p.model == "srv-model"  # remote_model wins over model_profile
        assert p.url == _CLOUD_URL

    def test_private_remote_is_peer(self):
        cfg = _lane(remote_url=_PRIVATE_URL)
        (p,) = derive_placement(cfg, peer_owned=False)
        assert p.origin is Origin.PEER

    def test_peer_owned_public_url_is_peer_not_cloud(self):
        # A peer-owned tunnel URL resolves to Cloudflare (public) IPs but is
        # your own infra — must be PEER, mirroring _classify's _peer_owned short
        # circuit.
        cfg = _lane(remote_url=_CLOUD_URL)
        (p,) = derive_placement(cfg, peer_owned=True)
        assert p.origin is Origin.PEER

    def test_remote_carries_key_and_timeout(self):
        cfg = _lane(remote_url=_PRIVATE_URL, remote_api_key="sk-abc", remote_timeout_s=300.0)
        (p,) = derive_placement(cfg, peer_owned=False)
        assert p.api_key == "sk-abc"
        assert p.timeout_s == 300.0

    def test_empty_lane_derives_empty(self):
        assert derive_placement(_lane(), peer_owned=False) == ()

    def test_explicit_placement_wins_over_derivation(self):
        explicit = (ProviderPlacement(origin=Origin.CLOUD, model="gpt-4o"),)
        cfg = _lane(model_profile="mistral-7b", placement=explicit)
        assert derive_placement(cfg, peer_owned=False) == explicit


class TestPrimaryPlacement:
    def test_returns_primary(self):
        cfg = _lane(model_profile="mistral-7b")
        ap = primary_placement(cfg, peer_owned=False)
        assert ap is not None and ap.origin is Origin.LOCAL

    def test_none_for_empty_lane(self):
        assert primary_placement(_lane(), peer_owned=False) is None


class TestPlacementKind:
    def test_empty_is_local(self):
        assert placement_kind(()) == "local"

    @pytest.mark.parametrize(
        "origin,expected",
        [(Origin.LOCAL, "local"), (Origin.CLOUD, "cloud"), (Origin.PEER, "self-hosted")],
    )
    def test_primary_origin_maps_to_legacy_kind(self, origin, expected):
        placement = (ProviderPlacement(origin=origin, model="m"),)
        assert placement_kind(placement) == expected


# ─── equivalence: Phase-2 classification must match the FROZEN legacy logic ──


def _legacy_classify_oracle(cfg: LaneConfig, *, peer_owned: bool) -> str:
    """Frozen copy of the PRE-Phase-2 ``_classify`` URL-sniff logic.

    Phase 2 reimplemented ``LaneBackendManager._classify`` on top of the
    placement layer, so comparing the new ``_classify`` to itself would be
    tautological. This oracle pins the *original* behavior so the matrix proves
    Phase 2 changed nothing for any existing (derived) config.
    """
    if cfg.remote_url:
        if peer_owned:
            return "self-hosted"
        return "cloud" if _is_cloud_url(cfg.remote_url) else "self-hosted"
    return "local"


# (cfg-kwargs, peer_owned) covering every classification branch.
_CLASSIFY_CASES = [
    ({}, False),  # empty lane → "local"
    ({"model_profile": "smollm-1.7b-instruct"}, False),  # local
    ({"model_profile": "claude-sonnet"}, False),  # cloud profile → still local
    ({"remote_url": _CLOUD_URL}, False),  # public → cloud
    ({"remote_url": _CLOUD_URL, "remote_model": "m"}, False),  # public → cloud
    ({"remote_url": _PRIVATE_URL}, False),  # private → self-hosted
    ({"remote_url": _LOOPBACK_URL}, False),  # loopback → self-hosted
    ({"remote_url": _CLOUD_URL}, True),  # peer-owned public → self-hosted
    ({"remote_url": _PRIVATE_URL}, True),  # peer-owned private → self-hosted
    ({"model_profile": "x", "remote_url": _CLOUD_URL}, False),  # remote wins
]


class TestClassifyEquivalence:
    """Phase-2 placement-based ``_classify`` == the frozen legacy URL-sniff.

    The proof obligation that Phase 2 (routing classification/dispatch through
    the placement layer) preserved behavior byte-for-byte for existing configs.
    """

    @pytest.mark.parametrize("kwargs,peer_owned", _CLASSIFY_CASES)
    def test_classify_matches_frozen_legacy_oracle(self, kwargs, peer_owned):
        cfg = _lane(**kwargs)
        mgr = LaneBackendManager({"large": cfg}, peer_owned=peer_owned)
        legacy = _legacy_classify_oracle(cfg, peer_owned=peer_owned)
        assert mgr._classify(cfg) == legacy, f"{kwargs} peer_owned={peer_owned}"

    @pytest.mark.parametrize("kwargs,peer_owned", _CLASSIFY_CASES)
    def test_classify_and_placement_helpers_agree(self, kwargs, peer_owned):
        # _classify is now the placement path; document they are one and the same.
        cfg = _lane(**kwargs)
        mgr = LaneBackendManager({"large": cfg}, peer_owned=peer_owned)
        derived = placement_kind(derive_placement(cfg, peer_owned=peer_owned))
        assert mgr._classify(cfg) == derived


class TestExplicitPlacementDrivesClassification:
    """Phase 2 acceptance: classification is driven by an EXPLICIT cfg.placement
    with NO URL-sniffing — the capability and placement axes are independent.

    Note: Phase 2 wires explicit placement into *classification* only; building
    a backend from an explicit placement is fenced off until Phase 3 (see
    test_get_backend_on_explicit_placement_raises_phase2_fence).
    """

    def test_explicit_placement_classifies_without_remote_url(self):
        # CLOUD origin, but NO remote_url — impossible to reach "cloud" under the
        # old URL-sniff path. Proves _classify reads origin, not the URL.
        cfg = _lane(placement=(ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet"),))
        assert cfg.remote_url is None
        mgr = LaneBackendManager({"large": cfg})
        assert mgr.get_lane_kind("large") == "cloud"

    def test_independent_capability_x_placement(self):
        # large capability placed LOCAL, small capability placed CLOUD — resolved
        # purely from explicit placement, no URL on either lane.
        large = _lane(name="large", placement=(ProviderPlacement(origin=Origin.LOCAL, model="mistral-7b"),))
        small = _lane(name="small", placement=(ProviderPlacement(origin=Origin.CLOUD, model="claude-haiku"),))
        mgr = LaneBackendManager({"large": large, "small": small})
        assert mgr.get_lane_kind("large") == "local"
        assert mgr.get_lane_kind("small") == "cloud"

    def test_explicit_peer_placement_is_self_hosted(self):
        cfg = _lane(placement=(ProviderPlacement(origin=Origin.PEER, model="m", url="http://leader:8100/v1"),))
        mgr = LaneBackendManager({"large": cfg})
        assert mgr.get_lane_kind("large") == "self-hosted"

    def test_get_backend_on_explicit_placement_raises_phase2_fence(self):
        # Phase-2 structural fence: building a backend from an explicit
        # placement is not wired until Phase 3, so it fails LOUD rather than
        # silently compiling from mismatched/absent legacy fields. Classifying
        # the same lane (above) still works.
        cfg = _lane(placement=(ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet"),))
        mgr = LaneBackendManager({"large": cfg}, max_cloud_lanes=1)
        with pytest.raises(BackendGateError, match="Phase 3"):
            mgr.get_backend("large")
