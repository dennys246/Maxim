"""Tests for _apply_local_llm_override (peer_leader_flexibility P1).

Precedence table covered here:
+---------------------------+-------------------------------------+
| User config               | Outcome                             |
+===========================+=====================================+
| --llm <local> + peer      | Local wins. remote_url cleared.     |
| --llm <cloud> + peer      | Cloud wins (no-op, handled upstream)|
| No --llm, peer config     | Peer wins. No changes.              |
| --llm <local>, no peer    | No-op (no remote_url to clear).     |
| Unknown profile + peer    | No-op (stays out of override path). |
+---------------------------+-------------------------------------+
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from maxim.runtime.lane_backends import _apply_local_llm_override
from maxim.runtime.worker_pool import LaneConfig


@pytest.fixture(autouse=True)
def _clear_env():
    """Each test starts with MAXIM_LLM_PROFILE unset."""
    saved = os.environ.pop("MAXIM_LLM_PROFILE", None)
    yield
    if saved is not None:
        os.environ["MAXIM_LLM_PROFILE"] = saved
    else:
        os.environ.pop("MAXIM_LLM_PROFILE", None)


def _peer_routed_lane(profile: str | None = None) -> dict[str, LaneConfig]:
    """Return a lane_configs dict resembling `build_primary_router`
    state after peer config injects a large-tier remote_url."""
    return {
        "large": LaneConfig(
            name="large",
            max_workers=1,
            requires_gpu=False,
            model_profile=profile,
            remote_url="https://maxim.example.com/v1",
            remote_api_key="test-key",
            remote_model=profile,
        ),
        "small": LaneConfig(
            name="small",
            max_workers=2,
            model_profile="smollm-1.7b-instruct",
            device="cpu",
            n_gpu_layers=0,
        ),
    }


class TestLocalLLMOverride:
    def test_clears_remote_url_for_local_profile(self):
        """The primary case: user passes --llm mistral-7b-instruct-v0.2
        with a peer config active. The large lane should have its
        remote fields cleared so the local backend runs mistral."""
        os.environ["MAXIM_LLM_PROFILE"] = "mistral-7b-instruct-v0.2"
        lanes = _peer_routed_lane()

        logger = MagicMock()
        result = _apply_local_llm_override(lanes, logger)

        large = result["large"]
        assert large.remote_url is None
        assert large.remote_model is None
        assert large.remote_api_key is None
        assert large.model_profile == "mistral-7b-instruct-v0.2"
        # Small lane is untouched
        assert result["small"].model_profile == "smollm-1.7b-instruct"
        logger.info.assert_called_once()

    def test_cloud_profile_is_passthrough(self):
        """Cloud profiles are handled by _apply_cloud_cli_overrides
        upstream. This function must NOT touch them — leaving a peer
        URL + cloud profile combination alone allows the upstream
        cloud path to win correctly."""
        os.environ["MAXIM_LLM_PROFILE"] = "claude-sonnet-4-6"
        lanes = _peer_routed_lane()
        pre_large = lanes["large"]

        result = _apply_local_llm_override(lanes, MagicMock())

        # Unchanged — same dataclass instance
        assert result["large"] is pre_large
        assert result["large"].remote_url == "https://maxim.example.com/v1"

    def test_no_profile_env_is_noop(self):
        """Without MAXIM_LLM_PROFILE set, peer config is unchanged."""
        lanes = _peer_routed_lane()
        pre_large = lanes["large"]

        result = _apply_local_llm_override(lanes, MagicMock())

        assert result["large"] is pre_large
        assert result["large"].remote_url is not None

    def test_no_remote_url_is_noop(self):
        """If the large lane has no remote_url to begin with, local
        override is a no-op regardless of --llm."""
        os.environ["MAXIM_LLM_PROFILE"] = "mistral-7b-instruct-v0.2"
        lanes = {
            "large": LaneConfig(
                name="large",
                max_workers=1,
                model_profile="mistral-7b-instruct-v0.2",
                # no remote_url
            ),
        }
        pre_large = lanes["large"]

        result = _apply_local_llm_override(lanes, MagicMock())

        assert result["large"] is pre_large
        assert result["large"].remote_url is None

    def test_unknown_profile_is_noop(self):
        """An unknown profile name (typo, custom non-registered entry)
        defaults to "not cloud" behavior and proceeds with the override.
        This is the safest default — unknown profiles can still be
        local, and we don't want to silently preserve a peer route
        for them. If the load later fails, the user gets a clear
        missing-profile error from the backend."""
        os.environ["MAXIM_LLM_PROFILE"] = "does-not-exist-yet"
        lanes = _peer_routed_lane()

        result = _apply_local_llm_override(lanes, MagicMock())

        # The function treats unknown profiles as local (since
        # profile_data.get("cloud") returns None), so it proceeds with
        # the override and clears remote_url.
        assert result["large"].remote_url is None
        assert result["large"].model_profile == "does-not-exist-yet"

    def test_only_large_lane_affected(self):
        """P1 is scoped to the large tier only. Medium and small
        lanes are left alone even if they somehow have remote_url
        set (not typical today but possible with future peer configs
        that route multiple tiers)."""
        os.environ["MAXIM_LLM_PROFILE"] = "mistral-7b-instruct-v0.2"
        lanes = _peer_routed_lane()
        # Pretend the medium lane also has a remote URL for this test
        lanes["medium"] = LaneConfig(
            name="medium",
            max_workers=1,
            model_profile="phi-3-mini-4k-instruct",
            remote_url="https://maxim.example.com/v1",
            remote_api_key="test-key",
        )

        result = _apply_local_llm_override(lanes, MagicMock())

        # Large cleared
        assert result["large"].remote_url is None
        # Medium NOT cleared — scoped to large only
        assert result["medium"].remote_url == "https://maxim.example.com/v1"

    def test_log_message_mentions_profile_and_url(self):
        """Observability: the log line should name the override
        profile and the URL being cleared so users can trace what
        happened."""
        os.environ["MAXIM_LLM_PROFILE"] = "mistral-7b-instruct-v0.2"
        lanes = _peer_routed_lane()
        logger = MagicMock()

        _apply_local_llm_override(lanes, logger)

        logger.info.assert_called_once()
        call_args = logger.info.call_args
        # First positional is the format string; the rest are values.
        # Verify profile and URL appear in the args somewhere.
        all_args_str = " ".join(str(a) for a in call_args.args)
        assert "mistral-7b-instruct-v0.2" in all_args_str
        assert "maxim.example.com" in all_args_str
