"""Tests for _maybe_pin_pre_upgrade_profile — migration safety.

The pinning fires exactly once per machine, on first post-upgrade
startup, to freeze the leader's profile to what the old tier table
would have picked. This prevents silent model switches when the tier
table expands in P4a.

Covered scenarios:
- Fires on existing install (util/ exists, no pin marker, no persisted)
- Skips fresh install (util/ absent)
- Skips when pin marker already exists (idempotent)
- Skips when persisted model already exists (nothing to do)
- Skips when MAXIM_LLM_PROFILE env is explicitly set
- Skips MPS systems (pre-P4a never pinned MPS anyway)
- Selects correct profile by VRAM via the frozen table
- Writes both the file AND the env var so current run is pinned
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from maxim.runtime.capabilities import RuntimeCapabilities
from maxim.runtime.lane_backends import (
    _maybe_pin_pre_upgrade_profile,
    _pick_pre_upgrade_infer_profile,
    _PRE_P4A_INFER_VRAM_TIERS,
)


def _fake_caps(has_gpu=True, gpu_type="RTX 5080", vram_gb=16.0, ram_gb=64.0):
    return RuntimeCapabilities(
        has_gpu=has_gpu,
        gpu_type=gpu_type,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
    )


def _prepare_existing_install(tmp_path: Path) -> Path:
    """Fake a machine that has run Maxim before: util/ exists with at
    least one file (e.g., a log or config)."""
    home = tmp_path / "dot_maxim"
    util = home / "util"
    util.mkdir(parents=True)
    (util / "some_prior_state.txt").write_text("anything")
    return home


class TestPreUpgradeProfilePick:
    def test_14gb_picks_llama_13b(self):
        assert _pick_pre_upgrade_infer_profile(14.0) == "llama-2-13b-chat"

    def test_8gb_picks_llama_8b(self):
        assert _pick_pre_upgrade_infer_profile(8.0) == "llama-3-8b-instruct"

    def test_4gb_picks_mistral(self):
        assert _pick_pre_upgrade_infer_profile(4.0) == "mistral-7b-instruct-v0.2"

    def test_below_4gb_falls_to_smollm(self):
        assert _pick_pre_upgrade_infer_profile(2.0) == "smollm-1.7b-instruct"

    def test_frozen_table_shape(self):
        """Drift guard: the frozen table MUST NOT be modified when
        _INFER_VRAM_TIERS in lane_models.py changes. This test asserts
        the exact values so any accidental drift produces a loud failure
        with a clear error message."""
        assert _PRE_P4A_INFER_VRAM_TIERS == (
            (14.0, "llama-2-13b-chat"),
            (8.0, "llama-3-8b-instruct"),
            (4.0, "mistral-7b-instruct-v0.2"),
            (0.0, "smollm-1.7b-instruct"),
        )


class TestPinningConditions:
    def setup_method(self):
        # Ensure no stale env from other tests
        os.environ.pop("MAXIM_LLM_PROFILE", None)

    def teardown_method(self):
        os.environ.pop("MAXIM_LLM_PROFILE", None)

    def test_fires_on_existing_install_with_gpu(self, tmp_path):
        home = _prepare_existing_install(tmp_path)
        caps = _fake_caps(vram_gb=16.0)
        logger = MagicMock()

        written_profile = {}

        def _fake_write(profile):
            written_profile["value"] = profile

        with (
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=_fake_write),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            _maybe_pin_pre_upgrade_profile(caps, logger)

        # Should have pinned to llama-2-13b-chat (the 14GB+ tier)
        assert written_profile.get("value") == "llama-2-13b-chat"
        # And set MAXIM_LLM_PROFILE for the current run
        assert os.environ.get("MAXIM_LLM_PROFILE") == "llama-2-13b-chat"
        # Marker file should exist
        assert (home / "util" / "tier_pinned_at_upgrade").exists()
        logger.info.assert_called_once()

    def test_skips_fresh_install_no_util_dir(self, tmp_path):
        # Fresh install: data_home exists but util/ does not
        home = tmp_path / "dot_maxim"
        home.mkdir()
        caps = _fake_caps()
        logger = MagicMock()

        write_called = MagicMock()
        with (
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=write_called),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            _maybe_pin_pre_upgrade_profile(caps, logger)

        write_called.assert_not_called()
        assert "MAXIM_LLM_PROFILE" not in os.environ

    def test_skips_when_util_dir_empty(self, tmp_path):
        # Edge case: util/ exists but has no content (newly-created
        # empty directory, perhaps from a prior crash before first use)
        home = tmp_path / "dot_maxim"
        util = home / "util"
        util.mkdir(parents=True)
        caps = _fake_caps()

        write_called = MagicMock()
        with (
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=write_called),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            _maybe_pin_pre_upgrade_profile(caps, None)

        write_called.assert_not_called()

    def test_skips_when_marker_exists(self, tmp_path):
        home = _prepare_existing_install(tmp_path)
        marker = home / "util" / "tier_pinned_at_upgrade"
        marker.touch()
        caps = _fake_caps(vram_gb=16.0)

        write_called = MagicMock()
        with (
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=write_called),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            _maybe_pin_pre_upgrade_profile(caps, None)

        # Pinning skipped because marker already present
        write_called.assert_not_called()
        assert "MAXIM_LLM_PROFILE" not in os.environ

    def test_skips_when_persisted_model_exists(self, tmp_path):
        home = _prepare_existing_install(tmp_path)
        caps = _fake_caps(vram_gb=16.0)

        write_called = MagicMock()
        with (
            patch(
                "maxim.runtime.lane_backends._read_persisted_model",
                return_value="mistral-7b-instruct-v0.2",
            ),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=write_called),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            _maybe_pin_pre_upgrade_profile(caps, None)

        # Persisted model exists → nothing to pin, but marker is still
        # written so we skip future runs
        write_called.assert_not_called()
        assert (home / "util" / "tier_pinned_at_upgrade").exists()

    def test_skips_when_env_profile_set(self, tmp_path):
        home = _prepare_existing_install(tmp_path)
        caps = _fake_caps(vram_gb=16.0)
        os.environ["MAXIM_LLM_PROFILE"] = "qwen2.5-14b-instruct"

        write_called = MagicMock()
        with (
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=write_called),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            _maybe_pin_pre_upgrade_profile(caps, None)

        # User explicitly set a profile → don't override
        write_called.assert_not_called()
        # And don't touch their env setting
        assert os.environ.get("MAXIM_LLM_PROFILE") == "qwen2.5-14b-instruct"

    def test_skips_mps_systems(self, tmp_path):
        """Pre-P4a, MPS was hard-excluded from the large-tier branch,
        so MPS systems would not have had a large-tier pin at all.
        Mirror that exactly — don't pin MPS systems retroactively."""
        home = _prepare_existing_install(tmp_path)
        caps = _fake_caps(gpu_type="mps", vram_gb=18.0)

        write_called = MagicMock()
        with (
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=write_called),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            _maybe_pin_pre_upgrade_profile(caps, None)

        write_called.assert_not_called()
        assert "MAXIM_LLM_PROFILE" not in os.environ
        # But the marker is still written so we don't re-check every run
        assert (home / "util" / "tier_pinned_at_upgrade").exists()

    def test_skips_cpu_only_systems_below_4gb(self, tmp_path):
        """No GPU or VRAM < 4GB → pre-P4a would not have had a large
        tier → no pin needed."""
        home = _prepare_existing_install(tmp_path)
        caps = _fake_caps(has_gpu=False, gpu_type=None, vram_gb=0.0)

        write_called = MagicMock()
        with (
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=write_called),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            _maybe_pin_pre_upgrade_profile(caps, None)

        write_called.assert_not_called()
        assert (home / "util" / "tier_pinned_at_upgrade").exists()

    def test_pin_exceptions_do_not_propagate(self, tmp_path):
        """Any unexpected error inside the pinning path must be
        swallowed so startup is never blocked by a failed migration."""
        home = _prepare_existing_install(tmp_path)
        caps = _fake_caps(vram_gb=16.0)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated failure in persisted_model write")

        logger = MagicMock()
        with (
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=_boom),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            # Must not raise
            _maybe_pin_pre_upgrade_profile(caps, logger)

        logger.debug.assert_called()


class TestPinningIntegration:
    """Integration-style tests that exercise the pinning logic's
    interaction with build_primary_router's persisted-model restore."""

    def test_env_var_set_means_current_run_respects_pin(self, tmp_path):
        """The critical correctness property: when pinning fires, the
        current run's MAXIM_LLM_PROFILE env var must be set so tier
        detection picks up the pinned profile immediately, not on the
        NEXT startup. Without this, the first post-upgrade run would
        use the new table and the second run would use the pin — a
        confusing two-state transition."""
        home = _prepare_existing_install(tmp_path)
        caps = _fake_caps(vram_gb=8.0)
        os.environ.pop("MAXIM_LLM_PROFILE", None)

        written = {}

        def _fake_write(profile):
            written["value"] = profile

        with (
            patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None),
            patch("maxim.runtime.lane_backends._write_persisted_model", side_effect=_fake_write),
            patch("maxim.utils.paths.data_home", return_value=home),
        ):
            _maybe_pin_pre_upgrade_profile(caps, None)

        # Both paths must be consistent: the persisted file got the
        # same profile that the env var now names.
        assert written["value"] == "llama-3-8b-instruct"
        assert os.environ.get("MAXIM_LLM_PROFILE") == "llama-3-8b-instruct"

        # Cleanup for other tests
        os.environ.pop("MAXIM_LLM_PROFILE", None)
