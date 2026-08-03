"""CostTracker local-lane guards (2026-08-03 owner-requested fix).

Local/self-hosted lanes have no pricing entry BY DESIGN, but record()
conflated that with pricing-data corruption: every local LLM call raised
→ the router logged 'CostTracker.record failed ... Missing pricing' as a
WARNING → and (the hidden behavioral bug) token counters never ran, so
local sessions reported ZERO tokens. Pins:

- pricing_required=False + missing pricing → zero cost, NO raise, tokens
  still counted (the sim-report accounting the raise was silently
  zeroing);
- pricing_required=True (metered lane) + missing pricing → still raises
  loudly (genuine corruption must not be hidden);
- priced models unaffected either way;
- has_cloud_placement(): local-only mesh → False; a CLOUD anywhere —
  even a fallback tail — → True (primary-only classification misses it);
- lane provider configs already declare pricing_required=False, and the
  router passes it through (source pin).
"""

from __future__ import annotations

import pytest

from maxim.models.language.cost_tracker import CostTracker, CostTrackerConfig, ModelPricing


def _tracker(pricing=None):
    return CostTracker(pricing=pricing or {}, config=CostTrackerConfig())


class TestRecordLocalLanes:
    def test_unpriced_local_returns_zero_and_counts_tokens(self):
        t = _tracker()
        cost = t.record(
            provider="lane-large",
            model="/Users/x/.maxim/models/LLM/Qwen2.5-32B-Instruct-Q4_K_M.gguf",
            input_tokens=1200,
            output_tokens=340,
            pricing_required=False,
        )
        assert cost == 0.0
        tokens = t.get_session_tokens()
        assert tokens["input_tokens"] == 1200
        assert tokens["output_tokens"] == 340

    def test_unpriced_metered_still_raises(self):
        t = _tracker()
        with pytest.raises(ValueError, match="Missing pricing"):
            t.record(
                provider="anthropic",
                model="claude-unknown-model",
                input_tokens=10,
                output_tokens=5,
                pricing_required=True,
            )

    def test_tokens_counted_even_when_metered_raise_fires(self):
        """The hidden behavioral bug: pre-fix the raise ran BEFORE the
        counters, so failed-pricing calls recorded zero tokens."""
        t = _tracker()
        with pytest.raises(ValueError):
            t.record(provider="anthropic", model="nope", input_tokens=50, output_tokens=20)
        assert t.get_session_tokens()["input_tokens"] == 50

    def test_priced_model_unaffected(self):
        t = _tracker({"claude-sonnet": ModelPricing(3.0, 15.0, 0.3)})
        cost = t.record(
            provider="anthropic",
            model="claude-sonnet",
            input_tokens=1_000_000,
            output_tokens=0,
            pricing_required=True,
        )
        assert cost == pytest.approx(3.0)


class TestHasCloudPlacement:
    def _manager(self, lane_configs, *, peer_owned=False):
        from maxim.runtime.lane_backends import LaneBackendManager

        return LaneBackendManager(lane_configs, peer_owned=peer_owned)

    def _lane(self, **kw):
        from maxim.runtime.worker_pool import LaneConfig

        return LaneConfig(name=kw.pop("name", "large"), max_workers=1, **kw)

    def test_local_only_mesh_is_false(self):
        mgr = self._manager({"large": self._lane(model_profile="qwen2.5-14b-instruct")})
        assert mgr.has_cloud_placement() is False

    def test_self_hosted_remote_is_still_not_cloud(self):
        # peer_owned=True is the production condition for a peer.yml-derived
        # tunnel lane (the big-mac-mini topology): PEER origin, not CLOUD.
        # (Without peer_owned, a public-host URL legitimately derives CLOUD.)
        mgr = self._manager(
            {"large": self._lane(remote_url="https://maxim.big-mac-mini.org/v1", remote_model="qwen")},
            peer_owned=True,
        )
        assert mgr.get_lane_kind("large") == "self-hosted"
        assert mgr.has_cloud_placement() is False

    def test_cloud_fallback_tail_counts(self):
        """Primary-only classification (get_lane_kind) misses a
        [LOCAL, CLOUD-fallback] lane — presence anywhere must count."""
        from maxim.runtime.worker_pool import Origin, ProviderPlacement

        mgr = self._manager(
            {
                "large": self._lane(
                    model_profile="qwen2.5-14b-instruct",
                    placement=(
                        ProviderPlacement(origin=Origin.LOCAL, model="qwen2.5-14b-instruct"),
                        ProviderPlacement(origin=Origin.CLOUD, model="claude-sonnet"),
                    ),
                )
            }
        )
        assert mgr.get_lane_kind("large") == "local"  # primary-only view
        assert mgr.has_cloud_placement() is True  # presence view


def test_router_passes_pricing_required_through():
    """Source pin: the record() call must consult the provider's declared
    pricing_required — reverting to the bare call resurrects the per-call
    WARNING on every local LLM call."""
    import inspect

    import maxim.models.language.router as router_mod

    src = inspect.getsource(router_mod)
    assert "pricing_required=self._provider_pricing_required(" in src
