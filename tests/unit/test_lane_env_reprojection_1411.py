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


def test_a_mesh_setup_after_a_first_projection_is_seen_by_the_next() -> None:
    """Pulse's sequence at the seam: nothing configured, project; set up; project again."""
    assert _apply() is False
    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    assert _apply() is True
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] == LEADER_A
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_API_KEY"] == "mesh-key-aaaaaaaaaaaaaaaa"


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
    monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_API_KEY", "operator-key-0123456789")  # config's key is a file ref
    monkeypatch.setenv("MAXIM_MAX_CLOUD_LANES", "0")  # resolve_setting never reads it: only the env-wins check keeps it
    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    _apply()
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_API_KEY"] == "operator-key-0123456789"
    _setup(LEADER_B, "mesh-key-bbbbbbbbbbbbbbbb")
    _apply()
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] == "http://operator.example:8099/v1"
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_API_KEY"] == "operator-key-0123456789"
    assert os.environ["MAXIM_MAX_CLOUD_LANES"] == "0"


def test_a_value_changed_after_the_projection_wrote_it_is_left_alone() -> None:
    """The undo restores only what the projection wrote and still holds: a value someone set since (an in-process
    override, a later export) survives the next config write and is no longer the projection's to manage."""
    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    _apply()
    os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] = "http://override.example:8099/v1"
    _setup(LEADER_B, "mesh-key-bbbbbbbbbbbbbbbb")
    _apply()
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_URL"] == "http://override.example:8099/v1"
    assert os.environ["MAXIM_LANE_LARGE_REMOTE_API_KEY"] == "mesh-key-bbbbbbbbbbbbbbbb"  # the untouched var moved on


def test_an_unchanged_config_is_projected_once() -> None:
    """No config write, no re-projection: the once-per-process contract holds between writes."""
    from maxim.runtime import lane_backends

    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    _apply()
    with patch.object(lane_backends, "_project_lane_tier", side_effect=AssertionError("re-projected")):
        assert _apply() is True


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


# ── review-round folds ─────────────────────────────────────────────────────────────────────────────────


def _build_with_persisted(persisted: str | None):
    from maxim.models.language.config import LLMConfig
    from maxim.runtime import lane_backends

    with (
        patch("maxim.runtime.capabilities.detect_compute_resources", return_value=(False, None, 0.0, 8.0)),
        patch("maxim.models.language.config.load_llm_config", return_value=LLMConfig()),
        patch("maxim.models.language.router.LLMRouter"),
        patch.object(lane_backends, "_read_persisted_model", return_value=persisted),
    ):
        _router, manager = lane_backends.build_primary_router()
    cfg = manager._configs.get("large")
    return cfg.remote_url if cfg is not None else None


def test_a_box_with_a_persisted_local_model_routes_to_the_leader_after_setup(monkeypatch) -> None:
    """Executor lens BLOCKER: the first build restores the persisted local profile into MAXIM_LLM_PROFILE; after the
    setup, that profile must not survive to make _apply_local_llm_override clear the new leader URL."""
    monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
    _build_with_persisted("llama-2-13b-chat")
    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    assert _build_with_persisted("llama-2-13b-chat") == LEADER_A
    assert "MAXIM_LLM_PROFILE" not in os.environ


def test_the_projection_runs_under_its_lock() -> None:
    """No other thread can observe an undo-then-reproject half done."""
    import threading

    from maxim.runtime import lane_backends

    seen: list[bool] = []
    real = lane_backends._project_lane_tier

    def probe(tier, cfg, logger_obj):
        acquired: list[bool] = []
        t = threading.Thread(target=lambda: acquired.append(lane_backends._lane_env_lock.acquire(blocking=False)))
        t.start()
        t.join()
        if acquired and acquired[0]:
            lane_backends._lane_env_lock.release()
        seen.append(bool(acquired and acquired[0]))
        return real(tier, cfg, logger_obj)

    with patch.object(lane_backends, "_project_lane_tier", side_effect=probe):
        _apply()
    assert seen and not any(seen)


def test_an_overwritten_value_is_restored_by_the_undo(monkeypatch) -> None:
    """The cloud-lane cap raised to admit config.json's CLOUD placements comes back down when the config changes."""
    from maxim.runtime import lane_backends

    monkeypatch.setenv("MAXIM_MAX_CLOUD_LANES", "0")
    lane_backends._project_env("MAXIM_MAX_CLOUD_LANES", "2", overwrite=True)
    assert os.environ["MAXIM_MAX_CLOUD_LANES"] == "2"
    lane_backends._undo_lane_env_projection()
    assert os.environ["MAXIM_MAX_CLOUD_LANES"] == "0"


def test_a_live_talk_loop_is_rebuilt_after_a_config_write(monkeypatch, tmp_path) -> None:
    """Architecture lens: a talk loop built under leader A keeps A's router; a later setup must rebuild it."""
    import threading
    from types import SimpleNamespace

    from maxim.console import handle as H
    from maxim.runtime.config_loader import config_generation, invalidate_config_cache

    h = object.__new__(H.MaximHandle)
    h._talk_lock = threading.Lock()
    alive = SimpleNamespace(is_alive=lambda: True)
    h._talk_bridge, h._talk_thread, h._talk_stop, h._talk_worker = object(), alive, None, None
    h._talk_bound, h._ending_talk_loops = None, []  # #1536: set in __init__, so a __new__ handle sets them too
    h._talk_generation = config_generation()
    h._workspace = tmp_path / "ws"
    stopped: list[bool] = []
    monkeypatch.setattr(h, "_stop_talk_loop", lambda **_k: stopped.append(True), raising=False)
    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", lambda: (None, None))
    assert h._ensure_talk_loop() is h._talk_bridge  # unchanged config: the live loop is reused
    assert stopped == []
    invalidate_config_cache()
    with pytest.raises(Exception, match="No LLM backend"):
        h._ensure_talk_loop()  # rebuilt against the new config (no backend in this stub)
    assert stopped == [True]


# ── delta-round folds ──────────────────────────────────────────────────────────────────────────────────


def test_the_persisted_restore_runs_inside_the_projections_lock_on_its_result() -> None:
    """Restored after the lock was released, a concurrent setup could let it act on a stale "no peer config"."""
    import threading

    from maxim.runtime import lane_backends

    held: list[bool] = []

    def probe(has_peer_config, logger):
        got: list[bool] = []
        t = threading.Thread(target=lambda: got.append(lane_backends._lane_env_lock.acquire(blocking=False)))
        t.start()
        t.join()
        if got and got[0]:
            lane_backends._lane_env_lock.release()
        held.append(not (got and got[0]))
        held.append(has_peer_config)

    _setup(LEADER_A, "mesh-key-aaaaaaaaaaaaaaaa")
    with patch.object(lane_backends, "_restore_persisted_profile", side_effect=probe):
        assert lane_backends._apply_lane_config_to_env(None, restore_persisted=True) is True
    assert held == [True, True]


def test_a_writer_outside_the_projection_skips_a_stale_generation(monkeypatch) -> None:
    from maxim.runtime import lane_backends
    from maxim.runtime.config_loader import invalidate_config_cache

    monkeypatch.delenv("MAXIM_MAX_CLOUD_LANES", raising=False)
    _has, gen_a = lane_backends._project_lane_env_for_build(None)  # build A projects
    assert lane_backends._project_env_if_current("MAXIM_MAX_CLOUD_LANES", "2", derived_at=gen_a, overwrite=True)
    lane_backends._undo_lane_env_projection()
    invalidate_config_cache()  # a write lands while build A is still running
    lane_backends._project_lane_env_for_build(None)  # build B re-projects at the new generation
    # A's late writer must not pass on the GLOBAL generation B just made current.
    assert (
        lane_backends._project_env_if_current("MAXIM_MAX_CLOUD_LANES", "3", derived_at=gen_a, overwrite=True) is False
    )
    assert "MAXIM_MAX_CLOUD_LANES" not in os.environ


def test_a_talk_loop_that_will_not_stop_stays_visible(monkeypatch) -> None:
    """A straggler whose handles were cleared would let the next talk() build a second loop beside it."""
    import threading
    from types import SimpleNamespace

    from maxim.console import handle as H

    h = object.__new__(H.MaximHandle)
    h._talk_lock = threading.Lock()
    straggler = SimpleNamespace(is_alive=lambda: True, join=lambda timeout=None: None)
    bridge = SimpleNamespace(finish=lambda: None)
    h._talk_bridge, h._talk_thread, h._talk_stop, h._talk_worker = bridge, straggler, threading.Event(), None
    h._talk_bound, h._ending_talk_loops = None, []  # #1536: set in __init__, so a __new__ handle sets them too
    with pytest.raises(RuntimeError, match="did not exit"):
        h._stop_talk_loop(join_s=0.01, required=True)
    assert h._talk_thread is straggler and h._talk_bridge is bridge
    # The next talk must not reuse the finished straggler's bridge (a 180 s hang): it refuses again, loudly.
    with pytest.raises(RuntimeError, match="did not exit"):
        h._ensure_talk_loop()
