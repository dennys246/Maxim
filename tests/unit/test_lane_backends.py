"""Tests for LaneBackendManager (multi-LLM Phase 3 + 4)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from maxim.runtime.lane_backends import (
    BackendGateError,
    LaneBackendManager,
    _is_cloud_url,
    _llm_server_responding_at,
    _validate_remote_urls,
    build_primary_router,
)
from maxim.runtime.worker_pool import LaneConfig


def _large_lane(profile: str | None = "smollm-1.7b-instruct", **kw) -> LaneConfig:
    return LaneConfig(
        name="large",
        max_workers=1,
        model_profile=profile,
        device=kw.pop("device", "cpu"),
        n_gpu_layers=kw.pop("n_gpu_layers", 0),
        **kw,
    )


class TestConstruction:
    def test_empty_manager(self):
        mgr = LaneBackendManager({})
        assert mgr.lanes == ()
        assert mgr.describe() == {}

    def test_lanes_exposed(self):
        mgr = LaneBackendManager(
            {
                "large": _large_lane(),
                "review": LaneConfig(name="review", max_workers=1, model_profile="smollm-1.7b-instruct"),
            }
        )
        assert set(mgr.lanes) == {"large", "review"}

    def test_describe_before_load(self):
        mgr = LaneBackendManager(
            {"large": _large_lane(profile="mistral-7b-instruct-v0.2", device="gpu", n_gpu_layers=-1)}
        )
        info = mgr.describe()
        assert info["large"]["profile"] == "mistral-7b-instruct-v0.2"
        assert info["large"]["device"] == "gpu"
        assert info["large"]["n_gpu_layers"] == -1
        assert info["large"]["loaded"] is False


class TestLaneResolution:
    def test_unknown_lane_returns_none(self):
        mgr = LaneBackendManager({"large": _large_lane()})
        assert mgr.get_backend("bogus") is None

    def test_lane_without_profile_or_remote_returns_none(self):
        # The record lane (or any lane with no LLM assignment) should return None
        mgr = LaneBackendManager(
            {
                "record": LaneConfig(name="record", max_workers=2),
            }
        )
        assert mgr.get_backend("record") is None


class TestLocalBackendBuild:
    def test_build_invokes_load_llm_config_with_profile(self, monkeypatch):
        """The lane's model_profile should be passed as profile_override."""
        # Earlier tests may set MAXIM_LLM_PROFILE via os.environ.setdefault(),
        # which changes the code path in _build_local_backend.
        monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
        monkeypatch.delenv("MAXIM_LLM_ENABLED", raising=False)
        mgr = LaneBackendManager({"large": _large_lane(profile="mistral-7b-instruct-v0.2")})
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = object()  # placeholder LLMConfig
            mock_router.return_value = "backend-instance"
            backend = mgr.get_backend("large")
            assert backend == "backend-instance"
            mock_load.assert_called_once_with(profile_override="mistral-7b-instruct-v0.2")

    def test_env_profile_override_wins(self, monkeypatch):
        """MAXIM_LLM_PROFILE set by the user should override the lane's choice."""
        monkeypatch.setenv("MAXIM_LLM_PROFILE", "llama-3-8b-instruct")
        mgr = LaneBackendManager({"large": _large_lane(profile="smollm-1.7b-instruct")})
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter"),
        ):
            mock_load.return_value = object()
            mgr.get_backend("large")
            mock_load.assert_called_once_with(profile_override="llama-3-8b-instruct")

    def test_env_n_gpu_layers_override_wins(self, monkeypatch):
        """MAXIM_LLM_N_GPU_LAYERS set by the user should suppress lane override."""
        monkeypatch.setenv("MAXIM_LLM_N_GPU_LAYERS", "0")
        mgr = LaneBackendManager({"large": _large_lane(n_gpu_layers=-1)})
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
            patch("dataclasses.replace") as mock_replace,
        ):
            mock_load.return_value = object()
            mgr.get_backend("large")
            # When the env var is set we should NOT call dataclasses.replace
            mock_replace.assert_not_called()
            mock_router.assert_called_once()


class TestLazyLoading:
    def test_backend_cached_after_first_call(self):
        mgr = LaneBackendManager({"large": _large_lane()})
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = object()
            mock_router.return_value = "backend-instance"
            a = mgr.get_backend("large")
            b = mgr.get_backend("large")
            assert a is b
            # load_llm_config and LLMRouter both invoked exactly once
            assert mock_load.call_count == 1
            assert mock_router.call_count == 1

    def test_describe_reflects_loaded_state(self):
        mgr = LaneBackendManager({"large": _large_lane()})
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter"),
        ):
            mock_load.return_value = object()
            assert mgr.describe()["large"]["loaded"] is False
            mgr.get_backend("large")
            assert mgr.describe()["large"]["loaded"] is True


class TestUnload:
    def test_unload_calls_backend_unload(self):
        mgr = LaneBackendManager({"large": _large_lane()})
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = object()

            # Give the fake backend an unload() so the manager calls it
            class FakeBackend:
                def __init__(self):
                    self.unloaded = False

                def unload(self):
                    self.unloaded = True

            fake = FakeBackend()
            mock_router.return_value = fake
            mgr.get_backend("large")
            mgr.unload_all()
            assert fake.unloaded is True
            # After unload, lane should re-create on next access
            assert mgr.describe()["large"]["loaded"] is False

    def test_unload_swallows_exceptions(self):
        mgr = LaneBackendManager({"large": _large_lane()})
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = object()

            class BrokenBackend:
                def unload(self):
                    raise RuntimeError("oops")

            mock_router.return_value = BrokenBackend()
            mgr.get_backend("large")
            # Should not raise despite the broken unload()
            mgr.unload_all()
            assert mgr.describe()["large"]["loaded"] is False


class TestUrlClassification:
    def test_localhost_is_self_hosted(self):
        assert _is_cloud_url("http://localhost:8000/v1") is False
        assert _is_cloud_url("http://127.0.0.1:8000/v1") is False
        assert _is_cloud_url("http://[::1]:8000/v1") is False

    def test_private_subnet_is_self_hosted(self):
        assert _is_cloud_url("http://192.168.1.100:8000/v1") is False
        assert _is_cloud_url("http://10.0.0.5:8000/v1") is False
        assert _is_cloud_url("http://172.16.0.1:8000/v1") is False

    def test_public_host_is_cloud(self):
        assert _is_cloud_url("https://api.anthropic.com/v1") is True
        assert _is_cloud_url("https://api.openai.com/v1") is True

    def test_empty_url_is_not_cloud(self):
        assert _is_cloud_url(None) is False
        assert _is_cloud_url("") is False

    def test_unparseable_fails_safe_to_cloud(self):
        # We can't tell — err on the side of gating
        assert _is_cloud_url("not-a-url://garbage") is True


class TestBackendCap:
    def test_cap_refuses_when_exceeded(self):
        """MAXIM_MAX_CONCURRENT_BACKENDS caps how many backends can exist."""
        mgr = LaneBackendManager(
            {
                "large": _large_lane(),
                "review": LaneConfig(name="review", max_workers=1, model_profile="smollm-1.7b-instruct"),
            },
            max_backends=1,
        )
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = object()
            mock_router.return_value = "backend-a"
            assert mgr.get_backend("large") == "backend-a"
            # Second lane should refuse
            with pytest.raises(BackendGateError, match="MAXIM_MAX_CONCURRENT_BACKENDS"):
                mgr.get_backend("review")

    def test_cap_respects_cached_lanes(self):
        """A cached lane doesn't re-count against the cap on subsequent calls."""
        mgr = LaneBackendManager({"large": _large_lane()}, max_backends=1)
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_load.return_value = object()
            mock_router.return_value = "backend"
            mgr.get_backend("large")
            # Same lane, returns cached — no new backend created
            assert mgr.get_backend("large") == "backend"

    def test_cap_from_env(self, monkeypatch):
        monkeypatch.setenv("MAXIM_MAX_CONCURRENT_BACKENDS", "3")
        mgr = LaneBackendManager({"large": _large_lane()})
        assert mgr.max_backends == 3

    def test_default_cap(self, monkeypatch):
        monkeypatch.delenv("MAXIM_MAX_CONCURRENT_BACKENDS", raising=False)
        mgr = LaneBackendManager({})
        assert mgr.max_backends == 2


class TestCloudLaneCap:
    def test_cloud_refused_by_default(self):
        """MAXIM_MAX_CLOUD_LANES=0 means no cloud access allowed."""
        mgr = LaneBackendManager(
            {
                "large": LaneConfig(
                    name="large",
                    max_workers=1,
                    remote_url="https://api.anthropic.com/v1",
                    remote_model="claude-3-opus",
                ),
            },
            max_cloud_lanes=0,
        )
        with pytest.raises(BackendGateError, match="MAXIM_MAX_CLOUD_LANES"):
            mgr.get_backend("large")

    def test_cloud_allowed_under_cap(self):
        mgr = LaneBackendManager(
            {
                "large": LaneConfig(
                    name="large",
                    max_workers=1,
                    remote_url="https://api.anthropic.com/v1",
                    remote_model="claude-3-opus",
                ),
            },
            max_cloud_lanes=1,
            max_backends=5,
        )
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            from maxim.models.language.config import LLMConfig

            mock_load.return_value = LLMConfig()
            mock_router.return_value = "cloud-backend"
            assert mgr.get_backend("large") == "cloud-backend"

    def test_self_hosted_not_counted_as_cloud(self):
        """Local llama-cpp-server on private IP should bypass cloud gate."""
        mgr = LaneBackendManager(
            {
                "large": LaneConfig(
                    name="large",
                    max_workers=1,
                    remote_url="http://192.168.1.10:8000/v1",
                    remote_model="mistral-7b",
                ),
            },
            max_cloud_lanes=0,  # even with zero cloud lanes, self-hosted OK
            max_backends=5,
        )
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            from maxim.models.language.config import LLMConfig

            mock_load.return_value = LLMConfig()
            mock_router.return_value = "self-hosted"
            assert mgr.get_backend("large") == "self-hosted"

    def test_default_cloud_cap_is_zero(self, monkeypatch):
        monkeypatch.delenv("MAXIM_MAX_CLOUD_LANES", raising=False)
        mgr = LaneBackendManager({})
        assert mgr.max_cloud_lanes == 0


class TestRemoteBackendBuild:
    def _remote_lane(self, url: str, model: str = "m", key: str | None = None) -> LaneConfig:
        return LaneConfig(
            name="large",
            max_workers=1,
            remote_url=url,
            remote_model=model,
            remote_api_key=key,
        )

    def test_self_hosted_sets_allow_local(self):
        mgr = LaneBackendManager(
            {"large": self._remote_lane("http://192.168.1.10:8000/v1", "mistral-7b")},
            max_backends=5,
        )
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            from maxim.models.language.config import LLMConfig

            mock_load.return_value = LLMConfig()
            mgr.get_backend("large")
            # The LLMConfig passed to LLMRouter should have providers with the
            # self-hosted entry set to allow local endpoints.
            (remote_cfg,), _ = mock_router.call_args
            provider = remote_cfg.providers["lane-large"]
            assert provider["base_url"] == "http://192.168.1.10:8000/v1"
            assert provider["model"] == "mistral-7b"
            assert provider["allow_local_endpoints"] is True
            assert remote_cfg.backend == "openai"
            assert remote_cfg.enabled is True
            # Self-hosted shouldn't force cloud_enabled on
            assert remote_cfg.cloud_enabled is False

    def test_cloud_forces_cloud_enabled(self):
        mgr = LaneBackendManager(
            {"large": self._remote_lane("https://api.anthropic.com/v1", "claude-3")},
            max_cloud_lanes=1,
            max_backends=5,
        )
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            from maxim.models.language.config import LLMConfig

            mock_load.return_value = LLMConfig(cloud_enabled=False)
            mgr.get_backend("large")
            (remote_cfg,), _ = mock_router.call_args
            # cloud lane should escalate cloud_enabled to True so LLMRouter's
            # internal gate doesn't reject the request
            assert remote_cfg.cloud_enabled is True
            assert remote_cfg.providers["lane-large"]["allow_local_endpoints"] is False

    def test_api_key_propagated_to_env(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_LARGE_API_KEY", raising=False)
        mgr = LaneBackendManager(
            {"large": self._remote_lane("https://api.anthropic.com/v1", "claude-3", key="sk-test")},
            max_cloud_lanes=1,
            max_backends=5,
        )
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter"),
        ):
            from maxim.models.language.config import LLMConfig

            mock_load.return_value = LLMConfig()
            mgr.get_backend("large")
            import os

            assert os.environ.get("MAXIM_LANE_LARGE_API_KEY") == "sk-test"


class TestBuildPrimaryRouter:
    def test_returns_manager_even_when_router_none(self, monkeypatch):
        """Factory always returns a manager; router may be None if build fails."""
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        monkeypatch.delenv("MAXIM_LANE_REVIEW_REMOTE_URL", raising=False)
        with (
            patch("maxim.runtime.capabilities.detect_compute_resources") as mock_detect,
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            mock_detect.return_value = (False, None, 0.0, 8.0)  # no GPU, 8GB RAM
            from maxim.models.language.config import LLMConfig

            mock_load.return_value = LLMConfig()
            mock_router.return_value = "router-instance"
            router, manager = build_primary_router()
            # No GPU + 8GB RAM → medium + small tiers, no large.
            # Large tier backend is None; router may be None or medium backend.
            assert manager is not None
            # Tier-based lanes: should have at least "small"
            assert "small" in manager.lanes

    def test_uses_provided_capabilities(self, monkeypatch):
        """When capabilities passed, factory does not call detect_compute_resources."""
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA RTX 5080", vram_gb=15.9, ram_gb=32.0)
        with (
            patch("maxim.runtime.capabilities.detect_compute_resources") as mock_detect,
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter") as mock_router,
        ):
            from maxim.models.language.config import LLMConfig

            mock_load.return_value = LLMConfig()
            mock_router.return_value = "router"
            build_primary_router(capabilities=caps)
            mock_detect.assert_not_called()

    def test_env_override_applied(self, monkeypatch):
        """MAXIM_LANE_LARGE_REMOTE_URL must flow through to the LaneBackendManager."""
        # Pre-clear any inherited env (test runner may have it set from .env)
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_MODEL", raising=False)
        monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:8000/v1")
        monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_MODEL", "mistral-7b")
        from maxim.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA RTX 5080", vram_gb=15.9, ram_gb=32.0)
        from maxim.runtime.llm_server import ProbeResult

        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter"),
            # P6: _validate_remote_urls now uses the structured probe.
            # Force "ok" so the env-supplied URL flows through.
            patch(
                "maxim.runtime.llm_server.probe_llm_server",
                return_value=ProbeResult("http://127.0.0.1:8000/v1", "ok", "HTTP 200", 10.0),
            ),
            # Mock the persisted-model read so the developer's real
            # ~/.maxim/util/active_llm_model.txt doesn't pollute the
            # test. Without this, the persisted profile is injected
            # into MAXIM_LLM_PROFILE which then triggers
            # _apply_local_llm_override to clear the remote_url.
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            # Also skip the pre-upgrade pin migration — it fires as a
            # side effect and can also set MAXIM_LLM_PROFILE on the
            # current run via its consistency guarantee.
            patch("maxim.runtime.lane_backends._maybe_pin_pre_upgrade_profile"),
            # P5: skip ensure_available so we don't try to download
            # smollm-1.7b in the test environment. The lane URL flow
            # is what we're testing here, not the download path.
            patch(
                "maxim.runtime.lane_backends._ensure_lane_profiles_available",
                side_effect=lambda lc, _caps, _log: lc,
            ),
        ):
            from maxim.models.language.config import LLMConfig

            mock_load.return_value = LLMConfig()
            _, manager = build_primary_router(capabilities=caps)
            info = manager.describe()["large"]
            assert info["remote_url"] == "http://127.0.0.1:8000/v1"
            assert info["kind"] == "self-hosted"

    def test_logger_receives_describe(self, monkeypatch):
        monkeypatch.delenv("MAXIM_LANE_LARGE_REMOTE_URL", raising=False)
        from maxim.runtime.capabilities import RuntimeCapabilities

        mock_logger = MagicMock()
        caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=8.0)
        with (
            patch("maxim.models.language.config.load_llm_config") as mock_load,
            patch("maxim.models.language.router.LLMRouter"),
        ):
            from maxim.models.language.config import LLMConfig

            mock_load.return_value = LLMConfig()
            build_primary_router(capabilities=caps, logger=mock_logger)
            # Lane assignments should be emitted at INFO level
            assert mock_logger.info.called
            # Find the Lane assignments call (may not be the only info call)
            lane_call = [c for c in mock_logger.info.call_args_list if "Lane assignments" in str(c)]
            assert lane_call, "Expected a 'Lane assignments' info log"


class TestLLMServerHealthCheck:
    def test_responding_url_returns_true(self):
        # Plan 1 R1: _llm_server_responding_at now goes through
        # maxim.utils.http.fetch_url, which uses httpx under the hood. Mock
        # at the fetch_url layer and assert it was called with the expected
        # /v1/models path. User-Agent is now set once at _external endpoint
        # registration (http.DEFAULT_USER_AGENT = "maxim-peer/1.0") — that's
        # verified structurally in tests/unit/test_http_client.py.
        from maxim.utils import http as _http

        fake_resp = _http.Response(
            status=200,
            headers={},
            content=b"{}",
            elapsed_ms=1.0,
            endpoint=_http._EXTERNAL_ENDPOINT,
            request_id="r",
        )
        with patch("maxim.utils.http.fetch_url", return_value=fake_resp) as mock_fetch:
            assert _llm_server_responding_at("http://127.0.0.1:8100/v1") is True
            called_url = mock_fetch.call_args[0][0]
            assert called_url.endswith("/v1/models")

    def test_dead_url_returns_false(self):
        from maxim.utils import http as _http

        with patch(
            "maxim.utils.http.fetch_url",
            side_effect=_http.HTTPConnectionError("_external", fix_hint="refused"),
        ):
            assert _llm_server_responding_at("http://127.0.0.1:8100/v1") is False

    def test_empty_url_returns_false(self):
        assert _llm_server_responding_at("") is False
        assert _llm_server_responding_at(None) is False


class TestValidateRemoteUrls:
    """P6 rewrote `_validate_remote_urls` to use the structured probe
    instead of the old loopback-only `_llm_server_responding_at`. The
    tests below now mock `probe_llm_server` and assert per-outcome
    behavior. Public/cloud URLs ARE probed (the old "skip public"
    heuristic was the bug — dead Cloudflare tunnels were silently
    wedging peers).
    """

    @pytest.fixture(autouse=True)
    def _isolate_probe_cache(self, tmp_path, monkeypatch):
        """Point MAXIM_DATA_HOME at a tmp dir so the on-disk probe cache
        doesn't leak between tests (or from the dev machine's real
        ~/.maxim/util/last_probe_status.json)."""
        home = tmp_path / "maxim_home"
        (home / "util").mkdir(parents=True)
        monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
        from maxim.utils import paths

        paths._reset_caches()

    @staticmethod
    def _ok(url):
        from maxim.runtime.llm_server import ProbeResult

        return ProbeResult(url, "ok", "HTTP 200", 10.0)

    @staticmethod
    def _down(url, outcome="connection_refused"):
        from maxim.runtime.llm_server import ProbeResult

        return ProbeResult(url, outcome, "boom", None)

    def test_unreachable_loopback_dropped(self):
        from unittest.mock import MagicMock as MM

        logger = MM()
        url = "http://127.0.0.1:8000/v1"
        cfgs = {
            "large": LaneConfig(
                name="large",
                max_workers=1,
                remote_url=url,
                remote_model="mistral-7b",
                remote_api_key="test-key",
            )
        }
        with patch("maxim.runtime.llm_server.probe_llm_server", return_value=self._down(url)):
            out = _validate_remote_urls(cfgs, logger)
        assert out["large"].remote_url is None
        assert out["large"].remote_model is None
        assert out["large"].remote_api_key is None
        logger.warning.assert_called_once()

    def test_reachable_loopback_kept(self):
        url = "http://127.0.0.1:8100/v1"
        cfgs = {
            "large": LaneConfig(
                name="large",
                max_workers=1,
                remote_url=url,
                remote_model="mistral-7b",
            )
        }
        with patch("maxim.runtime.llm_server.probe_llm_server", return_value=self._ok(url)):
            out = _validate_remote_urls(cfgs, None)
        assert out["large"].remote_url == url

    def test_cloud_url_now_probed(self):
        """P6: dead public URLs (e.g. expired Cloudflare tunnels) MUST be
        probed and dropped — the old skip-public-urls heuristic was the
        bug we're fixing. A reachable cloud URL stays wired."""
        url = "https://api.anthropic.com/v1"
        cfgs = {
            "large": LaneConfig(
                name="large",
                max_workers=1,
                remote_url=url,
                remote_model="claude-3",
            )
        }
        with patch("maxim.runtime.llm_server.probe_llm_server", return_value=self._ok(url)) as mock_probe:
            out = _validate_remote_urls(cfgs, None)
        mock_probe.assert_called_once()
        assert out["large"].remote_url == url

    def test_no_remote_url_untouched(self):
        cfgs = {"large": LaneConfig(name="large", max_workers=1, model_profile="smollm-1.7b-instruct")}
        out = _validate_remote_urls(cfgs, None)
        assert out["large"].remote_url is None
        assert out["large"].model_profile == "smollm-1.7b-instruct"


class TestDescribeIncludesKind:
    def test_kind_local(self):
        mgr = LaneBackendManager({"large": _large_lane()})
        assert mgr.describe()["large"]["kind"] == "local"

    def test_kind_self_hosted(self):
        mgr = LaneBackendManager(
            {
                "large": LaneConfig(name="large", max_workers=1, remote_url="http://127.0.0.1:8000/v1"),
            }
        )
        assert mgr.describe()["large"]["kind"] == "self-hosted"

    def test_kind_cloud(self):
        mgr = LaneBackendManager(
            {
                "large": LaneConfig(name="large", max_workers=1, remote_url="https://api.anthropic.com/v1"),
            }
        )
        assert mgr.describe()["large"]["kind"] == "cloud"
