"""#1411: a config write in a long-lived process is seen by the next router build.

``lane_backends._apply_lane_config_to_env`` projects ``config.json`` lane settings into ``MAXIM_LANE_*`` env vars,
guarded to run once per process so a second run does not re-attribute its own env vars as ``source="env"``. Under
``maxim serve`` (found by maxim-pulse on the 1.3.3 candidate) a failed talk BEFORE setup ran that projection with no
lane configured, and the ``/api/setup/mesh`` write that followed was never projected: talk kept failing "No LLM
backend available" until the server restarted. A config write now undoes exactly what the projection wrote and the
next build projects again; an operator's own env value is never touched.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

RED = pytest.mark.xfail(strict=True, reason="#1411 red gate: the lane env projection runs once per process")
LEADER_A = "http://10.0.0.5:8099/v1"
LEADER_B = "http://10.0.0.6:8099/v1"
_LANE_ENV = (
    "MAXIM_LANE_LARGE_REMOTE_URL",
    "MAXIM_LANE_LARGE_REMOTE_API_KEY",
    "MAXIM_LANE_LARGE_REMOTE_MODEL",
    "MAXIM_MAX_CLOUD_LANES",
    "MAXIM_DISABLE_PEER_CONFIG",
)


@pytest.fixture(autouse=True)
def _clean_lane_env(monkeypatch):
    for name in _LANE_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAXIM_SKIP_REMOTE_PROBE", "1")


def _apply():
    from maxim.runtime import lane_backends

    return lane_backends._apply_lane_config_to_env(None)


def _setup(url: str, key: str) -> None:
    from maxim.runtime.config_writer import apply_mesh_setup

    apply_mesh_setup(url, key)


@RED
def test_a_mesh_setup_after_a_first_projection_is_seen_by_the_next() -> None:
    """Pulse's sequence at the seam: nothing configured, project; set up; project again."""
    assert _apply() is False
    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    assert _apply() is True
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] == LEADER_A
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_API_KEY"] == "mesh-key-aaaaaaaaaaaaaaaa"


@RED
def test_switching_leaders_routes_to_the_new_one_not_the_first() -> None:
    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    assert _apply() is True and os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] == LEADER_A
    _setup(LEADER_B, "mesh-key-bbbbbbbbbbbbbbbb")
    assert _apply() is True
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] == LEADER_B
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_API_KEY"] == "mesh-key-bbbbbbbbbbbbbbbb"


def test_an_operator_env_value_is_never_touched(monkeypatch) -> None:
    """The operator's export wins over config.json, and a re-projection neither overwrites nor removes it."""
    monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://operator.example:8099/v1")
    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    _apply()
    _setup(LEADER_B, "mesh-key-bbbbbbbbbbbbbbbb")
    _apply()
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] == "http://operator.example:8099/v1"


def test_a_reprojection_keeps_the_doctor_attribution_on_config() -> None:
    """The once-per-process guard existed so a re-run would not read its own env vars back as source="env"."""
    from maxim.runtime.config_loader import resolve_setting

    _apply()
    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    _apply()
    _setup(LEADER_B, "mesh-key-bbbbbbbbbbbbbbbb")
    _apply()
    value, source = resolve_setting("lanes.large.remote_url")
    assert (value, source) == (LEADER_B, "config")


@RED
def test_the_router_build_after_setup_points_at_the_leader_in_the_same_process() -> None:
    """The composed path the Console's talk takes: build_primary_router before setup, then after it."""
    from maxim.models.language.config import LLMConfig
    from maxim.runtime.lane_backends import build_primary_router

    def build():
        with (
            patch("maxim.runtime.capabilities.detect_compute_resources", return_value=(False, None, 0.0, 8.0)),
            patch("maxim.models.language.config.load_llm_config", return_value=LLMConfig()),
            patch("maxim.models.language.router.LLMRouter"),
        ):
            return build_primary_router()

    def large_url(manager):
        cfg = manager._configs.get("large")
        return cfg.remote_url if cfg is not None else None

    _router, manager = build()
    assert large_url(manager) is None
    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    _router, manager = build()
    assert large_url(manager) == LEADER_A
