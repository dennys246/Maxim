"""#1601: llm.json ``tiers`` reach the router build, and a malformed llm.json (or any ``functions`` entry) refuses it.

``build_primary_router`` and ``_profile_is_pinned`` read both sections from ``LLMConfig._raw``, an attribute the frozen,
slotted config never had, so since feb65551 neither section changed anything. Owner decision (2026-11-24): apply them
and fail loud. These tests drive the shipped build with a real ``llm.json`` file found through ``MAXIM_LLM_CONFIG``.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

from maxim.exceptions import LLMRoutingConfigInvalid


@pytest.fixture
def llm_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in ("MAXIM_LLM_PROFILE", "MAXIM_LANE_LARGE_REMOTE_URL", "MAXIM_LANE_LARGE_REMOTE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MAXIM_SKIP_REMOTE_PROBE", "1")
    path = tmp_path / "llm.json"
    monkeypatch.setenv("MAXIM_LLM_CONFIG", str(path))

    def write(payload: object) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    return write


def _build():
    """The shipped build on a fixed no-GPU box, with only the machine-specific side effects stubbed."""
    from maxim.runtime.capabilities import RuntimeCapabilities
    from maxim.runtime.lane_backends import build_primary_router

    caps = RuntimeCapabilities(has_gpu=False, vram_gb=0.0, ram_gb=32.0)
    with ExitStack() as stack:
        stack.enter_context(patch("maxim.models.language.router.LLMRouter"))
        stack.enter_context(patch("maxim.runtime.lane_backends._read_persisted_model", return_value=None))
        stack.enter_context(patch("maxim.runtime.lane_backends._maybe_pin_pre_upgrade_profile"))
        stack.enter_context(
            patch(
                "maxim.runtime.lane_backends._ensure_lane_profiles_available",
                side_effect=lambda lc, _caps, _log, **_kw: lc,
            )
        )
        return build_primary_router(capabilities=caps)


def test_a_tiers_override_changes_the_lane(llm_json) -> None:
    llm_json({"tiers": {"small": {"model_profile": "qwen2.5-0.5b-instruct", "n_gpu_layers": 0}}})
    _router, manager = _build()
    assert manager.describe()["small"]["profile"] == "qwen2.5-0.5b-instruct"
    assert manager.describe()["small"]["n_gpu_layers"] == 0


def test_no_llm_json_sections_changes_nothing(llm_json) -> None:
    from maxim.runtime.function_router import DEFAULT_FUNCTIONS

    llm_json({"profile": "smollm-1.7b-instruct"})
    _router, manager = _build()
    assert manager.function_router.get_function("agent_inference") == DEFAULT_FUNCTIONS["agent_inference"]


def test_no_llm_json_at_all_builds(tmp_path, monkeypatch, llm_json) -> None:
    monkeypatch.setenv("MAXIM_LLM_CONFIG", str(tmp_path / "absent.json"))
    _router, manager = _build()
    assert manager is not None


@pytest.mark.parametrize(
    "payload, names",
    [
        ({"tiers": ["small"]}, "'tiers' must be an object"),
        ({"functions": "small"}, "'functions' must be an object"),
        ({"tiers": {"small": {"max_worker": 3}}}, "tiers.small: unknown field"),
        ({"tiers": {"large": {"device": "cpu"}}}, "no lane reads it yet"),
        # owner decision 2026-11-27: nothing consults the function router yet (#1833), so any entry refuses
        ({"functions": {"agent_inference": {"tier": "small"}}}, "cannot route anything yet"),
        ([1, 2], "must hold a JSON object"),
    ],
)
def test_a_malformed_section_refuses_the_build(llm_json, payload, names) -> None:
    """Fail loud: the build raises naming the entry, instead of starting with the entry silently ignored."""
    llm_json(payload)
    with pytest.raises(LLMRoutingConfigInvalid, match=names):
        _build()


def test_an_llm_json_that_does_not_parse_refuses_the_build(tmp_path, monkeypatch, llm_json) -> None:
    """A trailing comma used to read as an empty file: both sections silently applied nothing (#1601 review)."""
    broken = tmp_path / "llm.json"
    broken.write_text('{"tiers": {"large": {"model_profile": "x"},}', encoding="utf-8")
    with pytest.raises(LLMRoutingConfigInvalid, match="cannot be read as JSON"):
        _build()


def test_the_refusal_comes_before_any_download_or_spawn(llm_json) -> None:
    """The preflight refuses before tier detection, so a bad functions entry never downloads or spawns a model."""
    llm_json({"functions": {"agent_inference": {"tier": "small"}}})
    with patch("maxim.runtime.lane_models.detect_tiers") as detect, pytest.raises(LLMRoutingConfigInvalid):
        _build()
    detect.assert_not_called()


def test_a_tiers_profile_that_differs_from_the_session_model_refuses(llm_json, monkeypatch) -> None:
    """Every local lane serves the session's model, so a different tiers model_profile would be reported but never
    served; it refuses instead (owner decision 2026-11-27)."""
    monkeypatch.setenv("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")
    llm_json({"tiers": {"small": {"model_profile": "qwen2.5-0.5b-instruct"}}})
    with pytest.raises(LLMRoutingConfigInvalid, match="this session serves 'smollm-1.7b-instruct'"):
        _build()


def test_a_tiers_profile_equal_to_the_session_model_builds(llm_json, monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_LLM_PROFILE", "smollm-1.7b-instruct")
    llm_json({"tiers": {"small": {"model_profile": "smollm-1.7b-instruct", "n_gpu_layers": 0}}})
    _router, manager = _build()
    assert manager.describe()["small"]["profile"] == "smollm-1.7b-instruct"


def test_another_lanes_miss_keeps_this_lanes_override(monkeypatch) -> None:
    """The preflight re-walk re-picks only the lane whose model is missing; it used to overwrite every lane, undoing an
    llm.json override on another lane (#1601 review, executor S1)."""
    from maxim.runtime import lane_backends, lane_models
    from maxim.runtime.capabilities import RuntimeCapabilities
    from maxim.runtime.worker_pool import LaneConfig

    lanes = {
        "large": LaneConfig(name="large", max_workers=1, model_profile="missing-large", n_gpu_layers=-1),
        "small": LaneConfig(name="small", max_workers=2, model_profile="user-small", n_gpu_layers=0),
    }
    rewalk = {
        "large": LaneConfig(name="large", max_workers=1, model_profile="mid-on-disk", n_gpu_layers=-1),
        "small": LaneConfig(name="small", max_workers=2, model_profile="smollm-auto", n_gpu_layers=0),
    }
    monkeypatch.setattr(lane_models, "detect_tiers", lambda *a, **k: dict(rewalk))
    monkeypatch.setattr(lane_backends, "_profile_has_local_file", lambda p: p != "missing-large")
    monkeypatch.setattr("maxim.models.download.ensure_available", lambda *a, **k: False)
    monkeypatch.setattr("maxim.models.download.llm_models", lambda: {"missing-large", "user-small", "mid-on-disk"})
    caps = RuntimeCapabilities(has_gpu=True, gpu_type="NVIDIA", vram_gb=16.0, ram_gb=32.0)
    out = lane_backends._ensure_lane_profiles_available(lanes, caps, None)
    assert out["large"].model_profile == "mid-on-disk"
    assert out["small"].model_profile == "user-small"


@pytest.mark.parametrize("site", ["warn_unless_refusal", "research"])
def test_a_config_refusal_is_not_turned_into_a_run_without_the_llm(site, monkeypatch) -> None:
    """The broad LLM-start handlers re-raise CONFIG_REFUSALS beside SECURITY_REFUSALS (#1601 review)."""
    err = LLMRoutingConfigInvalid("llm.json tiers.large: bad")
    if site == "warn_unless_refusal":
        from maxim.utils.logging import warn_unless_refusal

        with pytest.raises(LLMRoutingConfigInvalid):
            warn_unless_refusal("Failed to create LLM worker: %s", err)
    else:
        from maxim.simulation import research_orchestrator

        def _raise(*a, **k):
            raise err

        monkeypatch.setattr("maxim.runtime.lane_backends.build_run_router", _raise)
        with pytest.raises(LLMRoutingConfigInvalid):
            research_orchestrator._build_research_llm()


def test_doctor_reports_the_refusal_instead_of_raising(llm_json) -> None:
    from maxim.doctor.checks import check_llm_routing

    llm_json({"tiers": {"large": {"device": "cpu"}}})
    result = check_llm_routing()
    assert result.status == "fail" and "no lane reads it yet" in result.message
    llm_json({"tiers": {"large": {"n_gpu_layers": 20}}})
    assert check_llm_routing().status == "ok"
