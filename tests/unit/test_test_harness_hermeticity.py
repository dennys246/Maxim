"""Executable contract for the default pytest process boundary."""

from __future__ import annotations

import os
from pathlib import Path


def test_user_state_and_cache_roots_are_session_isolated() -> None:
    """Default tests must not read or write the operator's home/cache state."""
    isolation_root = Path.home().parent

    assert isolation_root.name.startswith("maxim-tests-")
    assert "MAXIM_DATA_HOME" not in os.environ
    assert Path(os.environ["XDG_CONFIG_HOME"]).is_relative_to(isolation_root)
    assert Path(os.environ["XDG_CACHE_HOME"]).is_relative_to(isolation_root)
    if not os.environ.get("MAXIM_RUN_MODEL_TESTS"):
        assert Path(os.environ["HF_HOME"]).is_relative_to(isolation_root)
        assert Path(os.environ["TORCH_HOME"]).is_relative_to(isolation_root)


def test_model_hubs_default_to_offline_mode() -> None:
    """An installed optional ML package must not authorize a network lookup."""
    if not os.environ.get("MAXIM_RUN_MODEL_TESTS"):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_cost_tracker_default_state_uses_isolated_data_home() -> None:
    """Atexit cost persistence must stay inside the test-owned root."""
    from maxim.models.language.cost_tracker import CostTrackerConfig

    state_path = Path(CostTrackerConfig().state_path)
    assert state_path.is_relative_to(Path.home() / ".maxim")
