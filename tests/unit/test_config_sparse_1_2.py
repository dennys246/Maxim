"""config.json format 1.2: the file holds exactly the operator's choices (owner decision 2026-10-01).

Before 1.2 the writer dumped every field (``asdict``), defaults included, so no reader could tell a choice from a
default and ``_read_from_config`` guessed "value != default". ``maxim config set llm.n_ctx 8192`` (8192 IS the
default) therefore resolved as ``default`` — the O19 re-runs (C4 requires ``configured_n_ctx_source == "config"``)
could never pass — and ``cloud.enabled false`` (the default) could not switch C7a off.

From 1.2: the writer persists exactly the explicit set E (the paths the operator set), each write path states what it
assigns, and the loader reads a 1.2 (or unversioned, i.e. hand-written) file by key presence. A file stamped < 1.2 is
a full dump whose intent is unknowable, so it keeps today's reading until it is rewritten. Every assertion is on
``resolve_setting`` sources: config equality ignores E.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maxim.runtime.config_loader import load_config, resolve_setting

XFAIL = pytest.mark.xfail(strict=True, reason="config.json 1.2 (sparse, keys = operator choices) not built yet")


def _file(tmp_path: Path) -> Path:
    return tmp_path / "config.json"


def _src(path: Path, field: str) -> tuple:
    return resolve_setting(field, config=load_config(path))


def _write_raw(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload))


# ── the writer persists exactly the operator's choices ───────────────────────────────────────────────────


@XFAIL
def test_setting_the_default_value_pins_it(tmp_path) -> None:
    from maxim.runtime.config_writer import set_field

    path = _file(tmp_path)
    set_field("llm.n_ctx", "8192", path=path)  # the O19 rig's command; 8192 is the schema default
    assert json.loads(path.read_text()) == {"_format_version": "1.2", "llm": {"n_ctx": 8192}}
    assert _src(path, "llm.n_ctx") == (8192, "config")
    set_field("llm.profile", "mistral-7b", path=path)
    assert _src(path, "llm.n_ctx") == (8192, "config")  # a later set keeps the earlier pin
    assert _src(path, "llm.enabled")[1] == "default"  # never written, never a choice


@XFAIL
def test_unset_and_null_return_a_field_to_its_default(tmp_path) -> None:
    from maxim.runtime.config_writer import set_field, unset_field

    path = _file(tmp_path)
    set_field("llm.n_ctx", "8192", path=path)
    set_field("cloud.session_budget_usd", "2.5", path=path)
    unset_field("llm.n_ctx", path=path)
    assert _src(path, "llm.n_ctx")[1] == "default"
    set_field("cloud.session_budget_usd", None, path=path)  # null = unset (peer forget's form)
    assert _src(path, "cloud.session_budget_usd")[1] == "default"
    assert json.loads(path.read_text()) == {"_format_version": "1.2"}


@XFAIL
def test_cloud_enabled_false_is_an_explicit_choice(tmp_path) -> None:
    from maxim.runtime.config_writer import set_field

    path = _file(tmp_path)
    set_field("cloud.enabled", "false", path=path)
    assert _src(path, "cloud.enabled") == (False, "config")


@XFAIL
def test_a_mutator_changing_an_undeclared_field_is_refused(tmp_path) -> None:
    from maxim.exceptions import ConfigurationError
    from maxim.runtime.config_writer import _apply_field_to_config, mutate_config

    with pytest.raises(ConfigurationError, match="did not declare"):
        mutate_config(
            lambda c: _apply_field_to_config(c, "llm.n_ctx", 4096), path=_file(tmp_path), assigned=frozenset()
        )


@XFAIL
def test_the_setup_verbs_pin_what_they_assign(tmp_path) -> None:
    from maxim.runtime.config_writer import apply_cloud_setup

    path = _file(tmp_path)
    apply_cloud_setup("anthropic", "claude-sonnet", "sk-ant-test", monthly_budget_usd=5.0, path=path)
    assert _src(path, "cloud.session_budget_usd") == (5.0, "config")  # equal to the default, still the operator's
    assert _src(path, "cloud.enabled") == (True, "config")
    assert _src(path, "llm.n_ctx")[1] == "default"


# ── reading: by version ──────────────────────────────────────────────────────────────────────────────────


def _full_dump_1_1(tmp_path: Path, **llm) -> Path:
    """A file as every pre-1.2 build wrote it: the whole tree, defaults included."""
    from dataclasses import asdict

    from maxim.runtime.config_loader import MaximConfig

    payload = asdict(MaximConfig())
    for lane in payload["lanes"].values():
        lane.update(lane.pop("extra", {}) or {})
    payload["llm"].update(llm)
    payload["_format_version"] = "1.1"
    path = _file(tmp_path)
    _write_raw(path, payload)
    return path


def test_a_1_1_full_dump_reads_exactly_as_before(tmp_path) -> None:
    path = _full_dump_1_1(tmp_path, profile="mistral-7b", n_ctx=8192)
    assert _src(path, "llm.profile") == ("mistral-7b", "config")
    assert _src(path, "llm.n_ctx")[1] == "default"  # its intent is unknowable: today's reading
    assert _src(path, "llm.enabled")[1] == "default"


@XFAIL
def test_the_first_write_converts_a_1_1_dump_keeping_only_its_choices(tmp_path) -> None:
    from maxim.runtime.config_writer import set_field

    path = _full_dump_1_1(tmp_path, profile="mistral-7b", n_ctx=8192)
    set_field("llm.n_ctx", "8192", path=path)  # the rig's conversion step
    assert json.loads(path.read_text()) == {
        "_format_version": "1.2",
        "llm": {"profile": "mistral-7b", "n_ctx": 8192},
    }
    assert _src(path, "llm.n_ctx") == (8192, "config")


@XFAIL
def test_a_hand_written_unversioned_file_is_read_by_presence(tmp_path) -> None:
    path = _file(tmp_path)
    _write_raw(path, {"llm": {"n_ctx": 8192}, "cloud": {"enabled": False}})
    assert _src(path, "llm.n_ctx") == (8192, "config")
    assert _src(path, "cloud.enabled") == (False, "config")


@XFAIL
def test_a_hand_written_key_survives_the_first_set(tmp_path) -> None:
    from maxim.runtime.config_writer import set_field

    path = _file(tmp_path)
    _write_raw(path, {"llm": {"n_ctx": 8192}})
    set_field("llm.profile", "mistral-7b", path=path)
    assert _src(path, "llm.n_ctx") == (8192, "config")


def test_a_null_key_is_not_a_choice(tmp_path) -> None:
    path = _file(tmp_path)
    _write_raw(path, {"_format_version": "1.2", "llm": {"profile": None}})
    assert _src(path, "llm.profile")[1] == "default"


# ── restore pins a value equal to its default (round 3, A) ──────────────────────────────────────────────


@XFAIL
def test_restore_pins_a_preserved_default_equal_value(tmp_path) -> None:
    from maxim.runtime.config_writer import preserved_path, restore_preserved, set_field

    path = _file(tmp_path)
    set_field("llm.profile", "mistral-7b", path=path)
    preserved_path(path).write_text(
        json.dumps(
            {
                "_format_version": "1.0",
                "entries": {
                    "llm.n_ctx": {"value": 8192, "source_format_version": "1.3", "preserved_at": "2026-10-01T00:00:00Z"}
                },
            }
        )
    )
    preserved_path(path).chmod(0o600)
    assert restore_preserved(path, confirm=lambda rows: True) == ["llm.n_ctx"]
    assert _src(path, "llm.n_ctx") == (8192, "config")


# ── doctor reads through the loader's rule ───────────────────────────────────────────────────────────────


@XFAIL
def test_doctor_sees_a_pinned_default_equal_value(tmp_path) -> None:
    from maxim.doctor.checks import _read_config_for_doctor

    path = _file(tmp_path)
    _write_raw(path, {"_format_version": "1.2", "llm": {"n_ctx": 8192}})
    assert _read_config_for_doctor(load_config(path), "llm.n_ctx") == 8192


# ── the sim's report stamps the source before the router copies config into env ──────────────────────────


def test_the_report_stamps_provenance_before_the_router_is_built() -> None:
    """``_apply_lane_config_to_env`` (inside ``build_primary_router``) copies config values into env, after which
    ``resolve_setting`` reads ``env``: the start stamp must come first or ``configured_n_ctx_source`` is wrong."""
    import inspect

    from maxim.simulation import orchestrator

    source = inspect.getsource(orchestrator)
    assert source.index("capture_start_provenance()") < source.index("build_primary_router(")
