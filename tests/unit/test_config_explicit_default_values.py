"""A config.json value that equals the schema default is still the operator's: it resolves with source "config".

``_read_from_config`` treated any value equal to the schema default as unset, so ``maxim config set llm.n_ctx 8192``
(8192 IS the default) resolved as ``("default")``. The report stamps that source, so the O19 re-runs (whose C4 and
preflight require ``configured_n_ctx_source == "config"``) could never pass, and C7a's documented off-switch
``cloud.enabled false`` (the default) was invisible. Owner decision 2026-10-01: fix the loader, not the readers.
"""

from __future__ import annotations

import json

import pytest

from maxim.runtime.config_loader import reset_config_cache, resolve_setting


@pytest.fixture
def config_file(tmp_path, monkeypatch):
    def write(payload: dict) -> None:
        (tmp_path / "maxim").mkdir(exist_ok=True)
        (tmp_path / "maxim" / "config.json").write_text(json.dumps(payload))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        reset_config_cache()

    return write


@pytest.mark.xfail(strict=True, reason="a config value equal to the schema default reads as 'default'")
def test_an_explicit_default_equal_value_resolves_from_config(config_file) -> None:
    config_file({"llm": {"profile": "mistral-7b", "n_ctx": 8192}})  # the O19 rig's config
    assert resolve_setting("llm.n_ctx") == (8192, "config")


@pytest.mark.xfail(strict=True, reason="a config value equal to the schema default reads as 'default'")
def test_an_explicit_false_resolves_from_config(config_file) -> None:
    config_file({"cloud": {"enabled": False}})  # C7a's documented off-switch
    assert resolve_setting("cloud.enabled") == (False, "config")


def test_an_absent_field_is_still_the_default(config_file) -> None:
    config_file({"llm": {"profile": "mistral-7b"}})
    assert resolve_setting("llm.n_ctx")[1] == "default"
    assert resolve_setting("llm.profile") == ("mistral-7b", "config")


def test_the_environment_still_wins(config_file, monkeypatch) -> None:
    config_file({"llm": {"n_ctx": 8192}})
    monkeypatch.setenv("MAXIM_LLM_N_CTX", "4096")
    assert resolve_setting("llm.n_ctx") == (4096, "env")


def test_a_json_null_is_unset(config_file) -> None:
    config_file({"llm": {"profile": None}})
    assert resolve_setting("llm.profile")[1] == "default"
