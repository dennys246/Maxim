"""#1601: llm.json ``tiers`` and ``functions`` reach the router build, and a malformed one refuses it.

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

from maxim.exceptions import InvalidConfigError


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
    llm_json({"tiers": {"small": {"model_profile": "qwen2.5-0.5b-instruct", "max_workers": 3}}})
    _router, manager = _build()
    assert manager.describe()["small"]["profile"] == "qwen2.5-0.5b-instruct"
    assert manager._configs["small"].max_workers == 3  # describe() does not carry max_workers


def test_a_functions_override_reaches_the_function_router(llm_json) -> None:
    llm_json({"functions": {"agent_inference": {"tier": "small", "fallback": []}}})
    _router, manager = _build()
    spec = manager.function_router.get_function("agent_inference")
    assert (spec.tier, spec.fallback) == ("small", ())


def test_no_llm_json_sections_changes_nothing(llm_json) -> None:
    from maxim.runtime.function_router import DEFAULT_FUNCTIONS

    llm_json({"profile": "smollm-1.7b-instruct"})
    _router, manager = _build()
    assert manager.function_router.get_function("agent_inference") == DEFAULT_FUNCTIONS["agent_inference"]


@pytest.mark.parametrize(
    "payload, names",
    [
        ({"tiers": ["small"]}, "'tiers' must be an object"),
        ({"functions": "small"}, "'functions' must be an object"),
        ({"tiers": {"small": {"max_worker": 3}}}, "tiers.small: unknown field"),
        ({"functions": {"agent_inference": {"fallback": ["large"]}}}, "functions.agent_inference: needs a 'tier'"),
    ],
)
def test_a_malformed_section_refuses_the_build(llm_json, payload, names) -> None:
    """Fail loud: the build raises naming the entry, instead of starting with the entry silently ignored."""
    llm_json(payload)
    with pytest.raises(InvalidConfigError, match=names):
        _build()
