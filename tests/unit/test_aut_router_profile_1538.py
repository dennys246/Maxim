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


@pytest.mark.xfail(strict=True, reason="#1538: a failed AUT build leaves MAXIM_LLM_PROFILE set to the AUT's model")
def test_a_failed_aut_build_leaves_the_operator_profile(traces: dict[str, dict[str, Any]]) -> None:
    trace = traces["aut_build_fails"]
    # The AUT's build was attempted and refused (the run fell back to the shared router).
    assert [b["profile"] or b["env"] for b in trace["router_builds"]] == [OPERATOR, AUT_BUILD_FAILS]
    assert trace["maxim_env_after"].get("MAXIM_LLM_PROFILE") == OPERATOR


@pytest.mark.xfail(strict=True, reason="#1538: the AUT's build reads its profile from the process env")
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


@pytest.mark.xfail(strict=True, raises=TypeError, reason="#1538: build_primary_router has no profile= parameter")
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
    _, manager = LB.build_primary_router(profile="orch-aut", logger=None)
    assert seen == {"local_override": "orch-aut", "env_during_build": OPERATOR, "auto_spawn": "orch-aut"}
    assert manager.profile_override == "orch-aut"
    assert os.environ["MAXIM_LLM_PROFILE"] == OPERATOR
