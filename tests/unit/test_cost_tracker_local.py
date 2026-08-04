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
- has_cloud_billing_surface(): closes the cloud-profile-no-URL blind spot
  (--llm claude-sonnet derives a LOCAL placement but bills) — the banner
  keys off this, fail-closed;
- record-time metering keys on treat_as_cloud, NOT the provider config's
  pricing_required flag: lane_backends hardcodes pricing_required=False on
  every remote lane entry INCLUDING kind=="cloud" URL lanes, so reusing
  that flag at record time silently unmeters real cloud spend (the
  two-lens review's CRITICAL cross-confirmed finding).
"""

from __future__ import annotations

from unittest.mock import MagicMock

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


class TestHasCloudBillingSurface:
    """The startup banner's predicate. Must fail CLOSED: wrongly printing
    'Cost tracking disabled' on a billing config is the expensive
    direction."""

    def _manager(self, lane_configs, *, peer_owned=False):
        from maxim.runtime.lane_backends import LaneBackendManager

        return LaneBackendManager(lane_configs, peer_owned=peer_owned)

    def _lane(self, **kw):
        from maxim.runtime.worker_pool import LaneConfig

        return LaneConfig(name=kw.pop("name", "large"), max_workers=1, **kw)

    @pytest.fixture(autouse=True)
    def _isolated_llm_config(self, tmp_path, monkeypatch):
        """Pin load_llm_config to a minimal local-only file so the
        cloud_enabled leg reads test state, not the dev machine's."""
        cfg = tmp_path / "llm.json"
        cfg.write_text('{"enabled": false}')
        monkeypatch.setenv("MAXIM_LLM_CONFIG", str(cfg))
        monkeypatch.delenv("MAXIM_LLM_CLOUD_ENABLED", raising=False)
        yield cfg

    def test_local_only_mesh_is_false(self):
        mgr = self._manager({"large": self._lane(model_profile="qwen2.5-14b-instruct")})
        assert mgr.has_cloud_billing_surface() is False

    def test_peer_tunnel_is_false(self):
        mgr = self._manager(
            {"large": self._lane(remote_url="https://maxim.big-mac-mini.org/v1", remote_model="qwen")},
            peer_owned=True,
        )
        assert mgr.has_cloud_billing_surface() is False

    def test_cloud_profile_without_url_is_true(self):
        """THE headline lie the two-lens review caught: --llm claude-sonnet
        derives a LOCAL placement (by design, the MAX_CLOUD_LANES cap
        exemption) but dispatches to a real metered Anthropic backend.
        has_cloud_placement() says False here; the banner must not."""
        mgr = self._manager({"large": self._lane(model_profile="claude-sonnet")})
        assert mgr.has_cloud_placement() is False  # placement view (correct)
        assert mgr.has_cloud_billing_surface() is True  # billing view

    def test_cloud_profile_canonical_key_is_true(self):
        mgr = self._manager({"large": self._lane(model_profile="claude-sonnet-4-6")})
        assert mgr.has_cloud_billing_surface() is True

    def test_cloud_fallback_tail_is_true(self):
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
        assert mgr.has_cloud_billing_surface() is True

    def test_cloud_enabled_env_is_true_even_on_local_mesh(self, monkeypatch):
        """User llm.json providers can carry cloud entries the lane view
        can't see; cloud.enabled is the belt-and-suspenders leg."""
        monkeypatch.setenv("MAXIM_LLM_CLOUD_ENABLED", "1")
        mgr = self._manager({"large": self._lane(model_profile="qwen2.5-14b-instruct")})
        assert mgr.has_cloud_billing_surface() is True

    def _banner_text(self, mgr, caplog):
        import logging

        from maxim.runtime.lane_backends import _print_lane_banner

        with caplog.at_level(logging.INFO, logger="maxim.runtime.lane_backends"):
            _print_lane_banner(mgr)
        return "\n".join(r.message for r in caplog.records)

    def test_banner_omits_disabled_line_on_cloud_profile(self, caplog):
        mgr = self._manager({"large": self._lane(model_profile="claude-sonnet")})
        assert "Cost tracking disabled" not in self._banner_text(mgr, caplog), (
            "--llm claude-sonnet is the highest-spend path; the banner must not claim cost tracking is off there"
        )

    def test_banner_prints_disabled_line_on_local_only(self, caplog):
        mgr = self._manager({"large": self._lane(model_profile="qwen2.5-14b-instruct")})
        assert "Cost tracking disabled" in self._banner_text(mgr, caplog)


class TestRecordMeteringDecision:
    """Behavioral guard for the CRITICAL cross-confirmed finding: the
    record-time metering decision is treat_as_cloud (is_cloud and not
    allow_local_endpoints), NOT the provider config's pricing_required
    flag. lane_backends._build_remote_backend and the CLOUD-with-url
    placement tail both hardcode pricing_required=False (their opt-out of
    the pre-dispatch estimate gate) — keying record() on that flag makes a
    per-token-billed OpenAI-compatible endpoint (Together/Fireworks via
    remote_url) silently $0.00 with every budget ceiling inert."""

    def _router(self, provider_cfg):
        import dataclasses

        from maxim.models.language.config import LLMConfig
        from maxim.models.language.router import LLMRouter

        cfg = dataclasses.replace(LLMConfig(), enabled=True, providers={"p": provider_cfg})
        router = LLMRouter(cfg)
        router._audit_logger = MagicMock()  # no CWD-relative JSONL writes
        return router

    def _invoke(self, router):
        from maxim.models.language.types import LLMResponse

        backend = MagicMock()
        backend.requires_prompt_formatting = False
        backend.complete_with_usage = MagicMock(
            return_value=LLMResponse(content="hi", input_tokens=100, output_tokens=10, model="m", provider="p")
        )
        router._backends["p"] = backend
        provider_cfg = router._providers["p"]
        treat_as_cloud = router._provider_is_cloud(provider_cfg) and not bool(
            provider_cfg.get("allow_local_endpoints", False)
        )
        return router._invoke_backend(
            backend=backend,
            provider_key="p",
            redacted_system="",
            redacted_user="u",
            model="m",
            model_override=None,
            temperature=0.0,
            max_tokens=8,
            tools=None,
            thinking=None,
            stream=False,
            redaction_result=None,
            request_context=None,
            now=0.0,
            treat_as_cloud=treat_as_cloud,
        )

    _CLOUD_URL_LANE = {
        # The exact shape _build_remote_backend produces for kind=="cloud":
        # a metered OpenAI-compatible endpoint with the estimate-gate
        # opt-out flag. Record must stay LOUD despite pricing_required=False.
        "type": "openai",
        "base_url": "https://api.together.xyz/v1",
        "model": "m",
        "pricing_required": False,
    }

    def test_cloud_url_lane_with_missing_pricing_stays_loud(self, caplog):
        import logging

        router = self._router(dict(self._CLOUD_URL_LANE))
        with caplog.at_level(logging.WARNING, logger="maxim.models.language.router"):
            text, usage = self._invoke(router)
        assert text == "hi"
        assert usage["cost_usd"] == 0.0
        assert any("CostTracker.record failed" in r.message for r in caplog.records), (
            "a metered cloud call with no pricing entry must WARN — silence "
            "here is the silent-unmetered-spend inversion"
        )
        # Cloud calls DO emit the cloud audit entry.
        assert router._audit_logger.write.call_count == 1

    def test_cloud_record_warning_dedups_per_provider_model(self, caplog):
        import logging

        router = self._router(dict(self._CLOUD_URL_LANE))
        with caplog.at_level(logging.WARNING, logger="maxim.models.language.router"):
            self._invoke(router)
            self._invoke(router)
        warns = [r for r in caplog.records if r.levelno >= logging.WARNING and "CostTracker.record failed" in r.message]
        assert len(warns) == 1, "repeat failures for the same (provider, model) drop to DEBUG"

    def test_self_hosted_lane_records_quietly(self, caplog):
        import logging

        router = self._router(
            {
                "type": "openai",
                "base_url": "http://127.0.0.1:8100/v1",
                "model": "m",
                "allow_local_endpoints": True,
                "pricing_required": False,
            }
        )
        with caplog.at_level(logging.DEBUG, logger="maxim.models.language.router"):
            text, usage = self._invoke(router)
        assert text == "hi"
        assert usage["cost_usd"] == 0.0
        assert not any("CostTracker.record failed" in r.message for r in caplog.records), (
            "self-hosted lanes have no pricing BY DESIGN — the per-call "
            "WARNING spam is the bug the original patch fixed"
        )
        # And self-hosted calls do NOT emit cloud audit entries — the
        # peer tunnel was filling cloud_audit.jsonl (13 MB in a day) plus
        # a console "cloud_audit" INFO per call before the gate.
        assert router._audit_logger.write.call_count == 0

    def test_record_site_keys_on_treat_as_cloud_not_provider_flag(self):
        """Source pin: the record() call must pass treat_as_cloud. Keying
        on _provider_pricing_required at the record site is the reviewed
        and rejected design — it silently unmeters cloud URL lanes."""
        import inspect

        import maxim.models.language.router as router_mod

        src = inspect.getsource(router_mod.LLMRouter._invoke_backend)
        assert "pricing_required=treat_as_cloud" in src
        assert "pricing_required=self._provider_pricing_required(" not in src
