"""#1538: the dual-LLM AUT router build takes its profile as a parameter, never through the process env.

The AUT's build used to set ``MAXIM_LLM_PROFILE`` to ``aut_model``, call ``build_primary_router`` and restore the
variable on the success path only, so a build that raised left the AUT's model as the process's profile for the rest
of the run (every later router build and ``load_llm_config()`` read it, the ``/new`` re-entry's among them). And while
the build ran, every other thread that resolved a profile saw the AUT's. Owner decision 2026-11-02: pass the profile
into the build (``build_primary_router(profile=)``), which hands it to the lane manager for the lanes it builds later.

The orchestrator half is driven through the real ``start_simulation_mode`` by the harness's probe arms
``aut_router`` and ``aut_build_fails`` (``PROBE_ARMS`` in ``_orch_harness.py``); the lane half calls
``lane_backends`` directly.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from tests.unit._orch_harness import AUT_BUILD_FAILS, OPERATOR_PROFILE_ENV, assert_this_checkout, run_arm

assert_this_checkout()

OPERATOR = OPERATOR_PROFILE_ENV["MAXIM_LLM_PROFILE"]


@pytest.fixture(scope="module")
def traces(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm in ("aut_router", "aut_build_fails"):
        workdir = tmp_path_factory.mktemp(arm)
        out[arm] = run_arm(arm, workdir)
    return out


def test_a_failed_aut_build_leaves_the_operator_profile(traces: dict[str, dict[str, Any]]) -> None:
    trace = traces["aut_build_fails"]
    # The AUT's build was attempted and refused (the run fell back to the shared router).
    assert [b["profile"] or b["env"] for b in trace["router_builds"]] == [OPERATOR, AUT_BUILD_FAILS]
    assert trace["maxim_env_after"].get("MAXIM_LLM_PROFILE") == OPERATOR


def test_the_aut_build_is_handed_its_profile_and_the_env_keeps_the_operators(
    traces: dict[str, dict[str, Any]],
) -> None:
    assert traces["aut_router"]["router_builds"] == [
        {"profile": None, "env": OPERATOR},  # the shared build: the operator's profile, from the env
        {"profile": "orch-golden-aut", "env": OPERATOR},  # the AUT's: handed its own, and the env unchanged
    ]
    assert traces["aut_router"]["maxim_env_after"].get("MAXIM_LLM_PROFILE") == OPERATOR


def test_the_aut_build_falls_back_to_the_shared_router(traces: dict[str, dict[str, Any]]) -> None:
    """The fallback itself is unchanged: the AUT runs on the shared router (one build registered)."""
    prov = traces["aut_build_fails"]["report"]["provenance"]
    assert prov["aut_router_serial"] == prov["language_router_serial"] == "<build-1>", prov
    assert prov["aut_profile"] == OPERATOR, prov


def test_build_primary_router_takes_a_profile_and_leaves_the_env_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    import maxim.runtime.lane_backends as LB

    monkeypatch.setenv("MAXIM_LLM_PROFILE", OPERATOR)
    seen: dict[str, Any] = {}

    def _local_override(lane_configs: dict[str, Any], logger: Any, **kw: Any) -> dict[str, Any]:
        seen["local_override"] = kw.get("profile")
        seen["env_during_build"] = os.environ.get("MAXIM_LLM_PROFILE")
        return lane_configs

    def _spawn(caps: Any, lane_configs: dict[str, Any], logger: Any, **kw: Any) -> dict[str, Any]:
        seen["auto_spawn"] = kw.get("profile")
        return lane_configs

    monkeypatch.setattr(LB, "_apply_local_llm_override", _local_override)
    monkeypatch.setattr(LB, "_maybe_auto_spawn_server", _spawn)
    monkeypatch.setattr(LB.LaneBackendManager, "get_backend", lambda self, name: None)
    monkeypatch.setattr(LB, "_print_lane_banner", lambda manager: None)
    from maxim.runtime.capabilities import RuntimeCapabilities

    _, manager = LB.build_primary_router(RuntimeCapabilities(has_gpu=False), profile="orch-aut", logger=None)
    assert seen == {"local_override": "orch-aut", "env_during_build": OPERATOR, "auto_spawn": "orch-aut"}
    assert manager.profile_override == "orch-aut"
    assert os.environ["MAXIM_LLM_PROFILE"] == OPERATOR


def test_the_tier_table_takes_the_builds_profile_over_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.runtime.capabilities import RuntimeCapabilities
    from maxim.runtime.lane_models import detect_tiers

    monkeypatch.setenv("MAXIM_LLM_PROFILE", OPERATOR)
    caps = RuntimeCapabilities(has_gpu=True, gpu_type="cuda", vram_gb=24.0)
    assert detect_tiers(caps, profile="orch-aut")["large"].model_profile == "orch-aut"
    assert detect_tiers(caps)["large"].model_profile == OPERATOR  # no profile: the env's, as before


def test_the_manager_builds_its_later_lanes_on_the_builds_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lane the manager builds after the build returned (``get_backend`` on demand) serves the build's profile:
    the env var the old callers wrote was restored by then, so a later lane got the operator's."""
    import maxim.models.language.config as config_mod
    import maxim.runtime.lane_backends as LB
    from maxim.runtime.worker_pool import DEFAULT_TIERS

    monkeypatch.setenv("MAXIM_LLM_PROFILE", OPERATOR)
    asked: list[str | None] = []

    def _refuse(profile_override: str | None = None) -> Any:
        asked.append(profile_override)
        raise RuntimeError("stop after the profile is chosen")

    monkeypatch.setattr(config_mod, "load_llm_config", _refuse)
    cfg = DEFAULT_TIERS["small"]
    assert LB.LaneBackendManager({"small": cfg}, profile_override="orch-aut")._build_local_backend(cfg) is None
    assert LB.LaneBackendManager({"small": cfg})._build_local_backend(cfg) is None
    assert asked == ["orch-aut", OPERATOR]


def test_a_build_handed_a_profile_restores_no_persisted_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no session profile and no peer config, a build restores the persisted model into MAXIM_LLM_PROFILE.
    A build handed its own skips that, as the env write around it used to make it skip."""
    import maxim.runtime.lane_backends as LB

    monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
    monkeypatch.setattr(LB, "_read_persisted_model", lambda: "persisted-model")
    monkeypatch.setattr(LB, "_apply_lane_config_to_env_locked", lambda logger_obj: False)
    LB._project_lane_env_for_build(None, restore_persisted=False)
    assert "MAXIM_LLM_PROFILE" not in os.environ
    LB._project_lane_env_for_build(None)  # the default still restores it (the guard is not vacuous)
    assert os.environ.get("MAXIM_LLM_PROFILE") == "persisted-model"
    LB._undo_lane_env_projection()  # the recorded write, undone as a config write would (#1411)
    os.environ.pop("MAXIM_LLM_PROFILE", None)


def test_a_build_handed_a_profile_runs_no_pre_upgrade_pin(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    import maxim.runtime.lane_backends as LB
    import maxim.utils.paths as paths

    monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
    (tmp_path / "util").mkdir()
    (tmp_path / "util" / "something").write_text("x")
    monkeypatch.setattr(paths, "data_home", lambda: tmp_path)
    monkeypatch.setattr(LB, "_read_persisted_model", lambda: "persisted-model")  # stops at (2): marker written
    marker = tmp_path / "util" / "tier_pinned_at_upgrade"
    LB._maybe_pin_pre_upgrade_profile(None, None, profile="orch-aut")
    assert not marker.exists()
    LB._maybe_pin_pre_upgrade_profile(None, None)  # without one, it reaches the persisted check and marks
    assert marker.exists()


def test_a_build_handed_a_profile_leaves_an_unset_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No session profile, a persisted model, no peer config: the build that is handed a profile writes no
    MAXIM_LLM_PROFILE (neither its own nor the persisted one) and serves its own."""
    import maxim.runtime.lane_backends as LB
    from maxim.runtime.capabilities import RuntimeCapabilities

    monkeypatch.delenv("MAXIM_LLM_PROFILE", raising=False)
    monkeypatch.setattr(LB, "_read_persisted_model", lambda: "persisted-model")
    monkeypatch.setattr(LB, "_apply_lane_config_to_env_locked", lambda logger_obj: False)
    monkeypatch.setattr(LB.LaneBackendManager, "get_backend", lambda self, name: None)
    monkeypatch.setattr(LB, "_print_lane_banner", lambda manager: None)
    seen: list[str | None] = []
    real_override = LB._apply_local_llm_override

    def _local_override(lane_configs: dict[str, Any], logger: Any, **kw: Any) -> dict[str, Any]:
        seen.append(os.environ.get("MAXIM_LLM_PROFILE"))
        return real_override(lane_configs, logger, **kw)

    monkeypatch.setattr(LB, "_apply_local_llm_override", _local_override)
    _, manager = LB.build_primary_router(RuntimeCapabilities(has_gpu=False), profile="orch-aut", logger=None)
    assert seen == [None] and "MAXIM_LLM_PROFILE" not in os.environ
    assert manager.profile_override == "orch-aut"


def test_auto_spawn_serves_the_builds_profile_over_the_env_and_the_persisted_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import maxim.runtime.lane_backends as LB
    from maxim.runtime.worker_pool import DEFAULT_TIERS

    monkeypatch.setenv("MAXIM_LLM_PROFILE", OPERATOR)
    monkeypatch.setattr(LB, "_read_persisted_model", lambda: "persisted-model")
    cfg = DEFAULT_TIERS["small"]
    assert LB._auto_spawn_profile(cfg, "orch-aut", None) == "orch-aut"
    assert LB._auto_spawn_profile(cfg, None, None) == OPERATOR
    monkeypatch.delenv("MAXIM_LLM_PROFILE")
    assert LB._auto_spawn_profile(cfg, None, None) == "persisted-model"


def test_the_local_override_runs_the_builds_profile_here(monkeypatch: pytest.MonkeyPatch) -> None:
    """A large lane routed to a leader: a build handed a local profile serves it on this machine, as ``--llm``
    does, and the env's profile is not the one it runs."""
    import dataclasses

    import maxim.runtime.lane_backends as LB
    from maxim.runtime.worker_pool import DEFAULT_TIERS

    monkeypatch.setenv("MAXIM_LLM_PROFILE", OPERATOR)
    lane = dataclasses.replace(DEFAULT_TIERS["small"], name="large", remote_url="http://leader:8100")
    out = LB._apply_local_llm_override({"large": lane}, None, profile="orch-aut")
    assert (out["large"].model_profile, out["large"].remote_url) == ("orch-aut", None)
