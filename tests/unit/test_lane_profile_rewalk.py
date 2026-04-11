"""Direct tests for `_ensure_lane_profiles_available` (P5 plumbing).

The function is exercised end-to-end in `test_ensure_available.py`, but
those tests treat it as a black box. The behaviors below test the
re-walk pathway directly so a refactor that breaks the fall-through
gets caught by a focused test rather than a slow integration test.

Key invariants:
* Lane with `remote_url` set is left alone (no auto-download for
  remote-served lanes).
* Lane with cloud profile is left alone (no GGUF to download).
* Lane with `model_profile=None` is left alone.
* Lane with a profile that's NOT in the LLM_MODELS registry is left
  alone (custom profiles handled out-of-band).
* Lane whose profile IS in LLM_MODELS but not on disk:
    - calls `ensure_available` for that profile
    - on success: lane stays as-is
    - on failure: re-walks `detect_tiers` with the failed profile
      filtered out, lands on the next-smaller profile that IS available
* Re-walk happens at most once per build call.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from maxim.runtime.capabilities import RuntimeCapabilities
from maxim.runtime.lane_backends import _ensure_lane_profiles_available
from maxim.runtime.worker_pool import LaneConfig


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch):
    monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
    monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)


@pytest.fixture
def caps():
    return RuntimeCapabilities(has_gpu=True, gpu_type="cuda", vram_gb=16.0, ram_gb=64.0)


def _lane(name, profile, **kw):
    return LaneConfig(name=name, max_workers=1, model_profile=profile, **kw)


class TestSkipPaths:
    def test_remote_lane_skipped(self, caps):
        cfgs = {
            "large": _lane(
                "large",
                "qwen2.5-14b-instruct",
                remote_url="https://leader.example.com/v1",
            )
        }
        with patch("maxim.models.download.ensure_available") as mock_ea:
            out = _ensure_lane_profiles_available(cfgs, caps, logger=None)
        mock_ea.assert_not_called()
        assert out["large"].remote_url == "https://leader.example.com/v1"

    def test_cloud_profile_skipped(self, caps):
        cfgs = {"large": _lane("large", "claude-sonnet")}
        with (
            patch.dict(
                "maxim.models.language.config._BUILTIN_PROFILES",
                {"claude-sonnet": {"cloud": True, "backend": "anthropic"}},
                clear=False,
            ),
            patch("maxim.models.download.ensure_available") as mock_ea,
        ):
            _ensure_lane_profiles_available(cfgs, caps, logger=None)
        mock_ea.assert_not_called()

    def test_no_profile_skipped(self, caps):
        cfgs = {"large": _lane("large", None)}
        with patch("maxim.models.download.ensure_available") as mock_ea:
            _ensure_lane_profiles_available(cfgs, caps, logger=None)
        mock_ea.assert_not_called()

    def test_profile_not_in_registry_skipped(self, caps):
        """Custom profiles outside LLM_MODELS are handled out-of-band
        (the user pointed llm.json at a local GGUF). We should NOT try
        to auto-download them or re-walk."""
        cfgs = {"large": _lane("large", "user-custom-llama-99b")}
        with patch("maxim.models.download.ensure_available") as mock_ea:
            _ensure_lane_profiles_available(cfgs, caps, logger=None)
        mock_ea.assert_not_called()

    def test_profile_already_on_disk_skipped(self, caps):
        cfgs = {"large": _lane("large", "qwen2.5-14b-instruct")}
        with (
            patch(
                "maxim.runtime.lane_backends._profile_has_local_file",
                return_value=True,
            ),
            patch("maxim.models.download.ensure_available") as mock_ea,
        ):
            _ensure_lane_profiles_available(cfgs, caps, logger=None)
        mock_ea.assert_not_called()


class TestEnsureAvailableInvocation:
    def test_calls_ensure_available_when_profile_missing(self, caps):
        cfgs = {"large": _lane("large", "qwen2.5-14b-instruct")}
        with (
            patch(
                "maxim.runtime.lane_backends._profile_has_local_file",
                return_value=False,
            ),
            patch(
                "maxim.models.download.ensure_available",
                return_value=True,
            ) as mock_ea,
        ):
            _ensure_lane_profiles_available(cfgs, caps, logger=None)
        mock_ea.assert_called_once()
        assert mock_ea.call_args[0][0] == "qwen2.5-14b-instruct"

    def test_success_leaves_profile_intact(self, caps):
        cfgs = {"large": _lane("large", "qwen2.5-14b-instruct")}
        with (
            patch(
                "maxim.runtime.lane_backends._profile_has_local_file",
                return_value=False,
            ),
            patch("maxim.models.download.ensure_available", return_value=True),
        ):
            out = _ensure_lane_profiles_available(cfgs, caps, logger=None)
        assert out["large"].model_profile == "qwen2.5-14b-instruct"


class TestRewalkOnFailure:
    def test_failure_triggers_rewalk_to_smaller_profile(self, caps):
        """When the chosen profile can't be downloaded, the function
        re-runs detect_tiers with that profile filtered out and lands
        on the next-best lane the user already has on disk."""
        cfgs = {"large": _lane("large", "qwen2.5-14b-instruct")}
        rewalked = {
            "large": _lane(
                "large",
                "mistral-7b-instruct-v0.2",
                requires_gpu=True,
                device="gpu",
            )
        }
        with (
            patch(
                "maxim.runtime.lane_backends._profile_has_local_file",
                return_value=False,
            ),
            patch("maxim.models.download.ensure_available", return_value=False),
            patch(
                "maxim.runtime.lane_models.detect_tiers",
                return_value=rewalked,
            ) as mock_detect,
        ):
            out = _ensure_lane_profiles_available(cfgs, caps, logger=None)
        mock_detect.assert_called_once()
        # detect_tiers got passed a profile_available callback that
        # excludes the failed profile.
        kwargs = mock_detect.call_args.kwargs
        callback = kwargs.get("profile_available")
        assert callback is not None
        # The callback should reject qwen (just-failed) and accept others.
        assert callback("qwen2.5-14b-instruct") is False
        assert out["large"].model_profile == "mistral-7b-instruct-v0.2"

    def test_rewalk_runs_at_most_once_per_call(self, caps):
        """Even when MULTIPLE lanes have missing profiles, the re-walk
        only fires once. Subsequent failed lanes have model_profile
        cleared instead of triggering a second tier walk (which would
        be expensive and possibly contradict the first re-walk's result)."""
        cfgs = {
            "large": _lane("large", "qwen2.5-14b-instruct"),
            "medium": _lane("medium", "mistral-7b-instruct-v0.2"),
        }
        with (
            patch(
                "maxim.runtime.lane_backends._profile_has_local_file",
                return_value=False,
            ),
            patch("maxim.models.download.ensure_available", return_value=False),
            patch(
                "maxim.runtime.lane_models.detect_tiers",
                return_value={"large": _lane("large", "smollm-1.7b-instruct")},
            ) as mock_detect,
        ):
            _ensure_lane_profiles_available(cfgs, caps, logger=None)
        # Exactly one re-walk
        assert mock_detect.call_count == 1

    def test_rewalk_failure_clears_profile(self, caps):
        """If the re-walk itself raises, the lane's model_profile is
        cleared so downstream code falls back to its own defaults."""
        cfgs = {"large": _lane("large", "qwen2.5-14b-instruct")}
        with (
            patch(
                "maxim.runtime.lane_backends._profile_has_local_file",
                return_value=False,
            ),
            patch("maxim.models.download.ensure_available", return_value=False),
            patch(
                "maxim.runtime.lane_models.detect_tiers",
                side_effect=RuntimeError("walk exploded"),
            ),
        ):
            out = _ensure_lane_profiles_available(cfgs, caps, logger=None)
        assert out["large"].model_profile is None
