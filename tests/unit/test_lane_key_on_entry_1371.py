"""A lane's API key rides on its provider entry, never through the process env (#1371; owner decision 2026-11-10: A).

``runtime/lane_backends.py::_build_remote_backend`` (and the placement compiler for PEER and url-CLOUD entries) used
``os.environ.setdefault(MAXIM_LANE_<TIER>_API_KEY | MAXIM_PLACEMENT_<KEY>_API_KEY, key)`` and pointed the backend at
that name. A value already in the env -- Maxim's own write from an earlier build in this process, or one a child
inherited from its parent -- therefore beat a key rotated in config since, and the key leaked to every child. The
documented operator override is ``MAXIM_LANE_<TIER>_REMOTE_API_KEY``, which already beats config when the lane's key
resolves (``resolve_setting``), so the resolved key is the one the backend must send. The internal names are no longer
read; a value in them that differs from the key in use is ignored with one WARNING naming the right variable.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from unittest.mock import patch

import pytest

from maxim.models.language.config import LLMConfig


def _lane(url: str, key: str | None):
    from maxim.runtime.worker_pool import LaneConfig

    return LaneConfig(name="large", max_workers=1, remote_url=url, remote_api_key=key, remote_model="m")


def _entry_for(url: str, key: str | None) -> dict:
    from maxim.runtime.lane_backends import LaneBackendManager

    mgr = LaneBackendManager({"large": _lane(url, key)}, max_cloud_lanes=1, max_backends=5)
    with (
        patch("maxim.models.language.config.load_llm_config", return_value=LLMConfig()),
        patch("maxim.models.language.router.LLMRouter") as router,
    ):
        mgr.get_backend("large")
    return router.call_args[0][0].providers["lane-large"]


def _key_the_backend_sends(entry: dict) -> str:
    if entry["type"] == "maxim_peer":
        from maxim.models.language.maxim_peer_backend import _MaximPeerBackend

        backend = _MaximPeerBackend(dataclasses.replace(LLMConfig(), providers={"p": entry}), provider_key="p")
    else:
        from maxim.models.language.openai_backend import _OpenAIBackend

        backend = _OpenAIBackend(dataclasses.replace(LLMConfig(), providers={"p": entry}), provider_key="p")
    return backend._get_api_key()


SELF_HOSTED = "http://192.168.1.50:8100/v1"
CLOUD = "https://api.example-llm.com/v1"


@pytest.mark.parametrize("url", [SELF_HOSTED, CLOUD], ids=["self-hosted", "cloud"])
def test_a_rotated_config_key_beats_a_stale_internal_env_value(monkeypatch, url: str) -> None:
    monkeypatch.setenv("MAXIM_LANE_LARGE_API_KEY", "stale-key-from-an-earlier-build")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")  # the cloud backend's profile fallback
    entry = _entry_for(url, "rotated-key-0123456789")
    assert _key_the_backend_sends(entry) == "rotated-key-0123456789"
    assert entry.get("api_key_resolved") is True and "api_key_env" not in entry


@pytest.mark.parametrize("url", [SELF_HOSTED, CLOUD], ids=["self-hosted", "cloud"])
def test_the_build_never_writes_the_key_to_the_process_env(monkeypatch, url: str) -> None:
    monkeypatch.delenv("MAXIM_LANE_LARGE_API_KEY", raising=False)
    entry = _entry_for(url, "rotated-key-0123456789")
    assert "MAXIM_LANE_LARGE_API_KEY" not in os.environ
    assert entry.get("api_key") == "rotated-key-0123456789"


def test_a_stale_internal_value_is_warned_about_once_without_its_value(monkeypatch, caplog) -> None:
    from maxim.runtime import lane_backends

    lane_backends._reset_ignored_key_env_warnings()
    monkeypatch.setenv("MAXIM_LANE_LARGE_API_KEY", "stale-key-from-an-earlier-build")
    with caplog.at_level(logging.WARNING):
        _entry_for(SELF_HOSTED, "rotated-key-0123456789")
        _entry_for(SELF_HOSTED, "rotated-key-0123456789")
    warned = [r for r in caplog.records if "MAXIM_LANE_LARGE_API_KEY" in r.getMessage()]
    assert len(warned) == 1
    message = warned[0].getMessage()
    assert "MAXIM_LANE_LARGE_REMOTE_API_KEY" in message
    assert "stale-key" not in message and "rotated-key" not in message


@pytest.mark.parametrize("origin", ["PEER", "CLOUD"])
def test_a_placement_entry_carries_its_key_and_writes_no_env(monkeypatch, origin: str) -> None:
    from maxim.runtime.lane_backends import _placement_entry_to_provider, _placement_provider_api_key_env
    from maxim.runtime.worker_pool import Origin, ProviderPlacement

    name = _placement_provider_api_key_env("fallback-1")
    monkeypatch.setenv(name, "stale-placement-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    placement = ProviderPlacement(
        origin=Origin[origin], url=SELF_HOSTED if origin == "PEER" else CLOUD, model="m", api_key="placement-key-0123"
    )
    prov, _is_cloud = _placement_entry_to_provider(placement, "fallback-1")
    assert prov is not None
    assert prov.get("api_key") == "placement-key-0123"
    assert prov.get("api_key_resolved") is True and "api_key_env" not in prov
    assert _key_the_backend_sends(prov) == "placement-key-0123"
    assert os.environ[name] == "stale-placement-key"  # untouched: never written, never read


def test_control_a_keyless_self_hosted_lane_still_sends_not_needed(monkeypatch) -> None:
    monkeypatch.delenv("MAXIM_LANE_LARGE_API_KEY", raising=False)
    entry = _entry_for(SELF_HOSTED, None)
    assert _key_the_backend_sends(entry) == "not-needed"


def test_control_a_cloud_profile_entry_still_reads_its_env_first(monkeypatch) -> None:
    """A provider entry that NAMES an env var (a cloud profile: ANTHROPIC_API_KEY, OPENAI_API_KEY) keeps CLI > env >
    config: the env value wins over a config-carried key."""
    from maxim.models.language.openai_backend import _OpenAIBackend

    monkeypatch.setenv("TEST_PROFILE_KEY_1371", "from-env")
    entry = {"type": "openai", "base_url": CLOUD, "api_key_env": "TEST_PROFILE_KEY_1371", "api_key": "from-config"}
    backend = _OpenAIBackend(dataclasses.replace(LLMConfig(), providers={"p": entry}), provider_key="p")
    assert backend._get_api_key() == "from-env"


@pytest.mark.parametrize("backend", ["openai", "anthropic"])
def test_control_a_hand_written_provider_with_a_key_stays_env_first(monkeypatch, backend: str) -> None:
    """An llm.json provider carrying a literal key (no marker) keeps CLI > env > config: the exported env wins."""
    env = "OPENAI_API_KEY" if backend == "openai" else "ANTHROPIC_API_KEY"
    monkeypatch.setenv(env, "from-env")
    entry = {"type": backend, "base_url": CLOUD, "api_key": "from-llm-json"}
    if backend == "openai":
        from maxim.models.language.openai_backend import _OpenAIBackend as cls
    else:
        from maxim.models.language.anthropic_backend import _AnthropicBackend as cls
    backend_obj = cls(dataclasses.replace(LLMConfig(), providers={"p": entry}), provider_key="p")
    assert backend_obj._get_api_key() == "from-env"


def test_a_marked_entry_wins_even_if_it_names_an_env_var(monkeypatch) -> None:
    """The marker, not the absence of api_key_env, says "already resolved": a later producer adding an env name
    cannot silently flip a lane back to env-first."""
    from maxim.models.language.openai_backend import _OpenAIBackend

    monkeypatch.setenv("OPENAI_API_KEY", "stale")
    entry = {
        "type": "openai",
        "base_url": CLOUD,
        "api_key": "resolved",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_resolved": True,
    }
    backend = _OpenAIBackend(dataclasses.replace(LLMConfig(), providers={"p": entry}), provider_key="p")
    assert backend._get_api_key() == "resolved"
