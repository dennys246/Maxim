"""#974 -- an older build never silently drops a newer config.json's settings.

After a downgrade, the older build reads a newer-minor config.json fine (#856: unknown keys tolerated),
but `maxim config set` rewrote the file with only the settings it knew, stamped at its own version, so the
newer build's settings vanished. Owner decision 2026-09-28:

- `maxim config set` (mutate_config / set_field) REFUSES a newer file, naming the way out;
- `maxim config downgrade` keeps the known settings and moves the unknown ones to a 0600 sidecar,
  `config.preserved.json`, that nothing reads for configuration;
- `maxim config restore-preserved` (after upgrading) re-validates them against the schema, shows a diff,
  and applies only on an interactive confirmation -- never automatically.

Injection review (same day): names and values are escaped wherever they are printed; the restore needs a
real terminal (no --yes); security-relevant settings are flagged; restored values pass the same
validation as `config set`; the sidecar read is size-capped.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from maxim.exceptions import ConfigurationError
from maxim.runtime import config_loader as cl
from maxim.runtime import config_writer as cw


def _newer_version() -> str:
    major, minor = (int(p) for p in cl.CONFIG_FORMAT_VERSION.split(".")[:2])
    return f"{major}.{minor + 1}"


def _write_newer_config(path: Path, **extra) -> dict:
    data = {
        "_format_version": _newer_version(),
        "role": "solo",
        "llm": {"n_ctx": 4096, "future_knob": 7},  # a field this build does not know, in a known section
        "future_section": {"enabled": True},  # a section this build does not know
        **extra,
    }
    path.write_text(json.dumps(data))
    return data


# ── refuse: `config set` never drops a newer file's settings ───────────────────────────────────


def test_config_set_on_a_newer_file_refuses_and_names_the_way_out(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    original = _write_newer_config(path)
    with pytest.raises(ConfigurationError) as err:
        cw.set_field("llm.n_ctx", "8192", path=path)
    message = str(err.value)
    assert "maxim config downgrade" in message
    assert "llm.future_knob" in message and "future_section" in message  # names, not values
    assert json.loads(path.read_text()) == original  # untouched


def test_config_set_on_a_current_file_still_works(tmp_path: Path) -> None:
    """Anti-vacuity arm: the refusal is about the NEWER version, not about writing."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION, "role": "solo"}))
    cw.set_field("llm.n_ctx", "8192", path=path)
    assert json.loads(path.read_text())["llm"]["n_ctx"] == 8192


# ── downgrade: keep what this build knows, set the rest aside ─────────────────────────────────


def test_downgrade_keeps_known_settings_and_sets_the_rest_aside(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_newer_config(path)

    result = cw.downgrade_config(path=path)

    data = json.loads(path.read_text())
    assert data["_format_version"] == cl.CONFIG_FORMAT_VERSION
    assert data["role"] == "solo" and data["llm"]["n_ctx"] == 4096  # known settings kept
    assert "future_section" not in data and "future_knob" not in data["llm"]
    side = json.loads(result.sidecar.read_text())
    assert {k: e["value"] for k, e in side["entries"].items()} == {
        "llm.future_knob": 7,
        "future_section": {"enabled": True},
    }
    assert {e["source_format_version"] for e in side["entries"].values()} == {_newer_version()}
    assert side["_format_version"]
    assert sorted(result.preserved) == ["future_section", "llm.future_knob"]
    cw.set_field("llm.n_ctx", "8192", path=path)  # the CLI works again


def test_the_sidecar_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_newer_config(path)
    result = cw.downgrade_config(path=path)
    assert stat.S_IMODE(os.stat(result.sidecar).st_mode) == 0o600


def test_the_sidecar_is_written_before_the_config_so_a_failure_loses_nothing(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "config.json"
    original = _write_newer_config(path)

    def _fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(cw, "atomic_write_json", _fail)  # the config write fails
    with pytest.raises(OSError):
        cw.downgrade_config(path=path)
    assert json.loads(path.read_text()) == original  # config untouched
    assert json.loads(cw.preserved_path(path).read_text())["entries"]  # and the settings are already safe


def test_downgrade_of_a_current_file_refuses(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION, "role": "solo"}))
    with pytest.raises(ConfigurationError, match="nothing to set aside"):
        cw.downgrade_config(path=path)


def test_a_second_downgrade_merges_into_the_existing_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_newer_config(path)
    cw.downgrade_config(path=path)
    path.write_text(json.dumps({"_format_version": _newer_version(), "other_future": 1}))
    result = cw.downgrade_config(path=path)
    entries = json.loads(result.sidecar.read_text())["entries"]
    assert {k: e["value"] for k, e in entries.items()} == {
        "llm.future_knob": 7,
        "future_section": {"enabled": True},
        "other_future": 1,
    }


# ── restore: re-validated, shown, confirmed -- never automatic ───────────────────────────────


def _preserve(tmp_path: Path, entries: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION, "role": "solo"}))
    sidecar = cw.preserved_path(path)
    wrapped = {
        k: {"value": v, "source_format_version": "9.9", "preserved_at": "2026-09-28T00:00:00Z"}
        for k, v in entries.items()
    }
    sidecar.write_text(json.dumps({"_format_version": "1.0", "entries": wrapped}))
    return path


def test_restore_applies_known_settings_only_after_confirmation(tmp_path: Path) -> None:
    path = _preserve(tmp_path, {"llm.n_ctx": 4096, "future_section": {"enabled": True}})
    shown: list = []

    declined = cw.restore_preserved(path=path, confirm=lambda plan: shown.append(plan) or False)
    assert declined == [] and "n_ctx" not in json.loads(path.read_text()).get("llm", {})

    applied = cw.restore_preserved(path=path, confirm=lambda plan: True)
    assert applied == ["llm.n_ctx"]
    assert json.loads(path.read_text())["llm"]["n_ctx"] == 4096
    # still unknown to this build: stays set aside
    left = json.loads(cw.preserved_path(path).read_text())["entries"]
    assert {k: e["value"] for k, e in left.items()} == {"future_section": {"enabled": True}}
    assert [row.path for row in shown[0]] == ["llm.n_ctx"]
    assert shown[0][0].source_format_version == "9.9" and shown[0][0].preserved_at.startswith("2026")


def test_restore_validates_like_config_set_and_is_all_or_nothing(tmp_path: Path) -> None:
    path = _preserve(tmp_path, {"llm.n_ctx": -5, "role": "leader"})
    before = path.read_text()
    with pytest.raises(ConfigurationError):
        cw.restore_preserved(path=path, confirm=lambda plan: True)
    assert path.read_text() == before


def test_security_relevant_settings_are_flagged_in_the_plan(tmp_path: Path) -> None:
    path = _preserve(tmp_path, {"role": "leader", "llm.n_ctx": 4096})
    plans: list = []
    cw.restore_preserved(path=path, confirm=lambda plan: plans.append(plan) or False)
    flagged = {row.path: row.security_relevant for row in plans[0]}
    assert flagged == {"role": True, "llm.n_ctx": False}


def test_an_oversized_sidecar_is_refused(tmp_path: Path) -> None:
    path = _preserve(tmp_path, {})
    cw.preserved_path(path).write_text(
        json.dumps({"_format_version": "1.0", "entries": {"x": {"value": "a" * 2_000_000}}})
    )
    with pytest.raises(ConfigurationError, match="too large"):
        cw.restore_preserved(path=path, confirm=lambda plan: True)


# ── injection: nothing crafted reaches the terminal raw; no restore without a person ──────────


def test_names_and_values_are_escaped_before_they_are_printed() -> None:
    from maxim.runtime.config_cli import _safe

    rendered = _safe("role\x1b[2K\rharmless\nnext" + "x" * 500)
    assert "\x1b" not in rendered and "\n" not in rendered and "\r" not in rendered
    assert "\\x1b" in rendered and len(rendered) <= 130


def test_the_refusal_escapes_crafted_key_names(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_newer_config(path, **{"evil\x1b[2Kkey": 1})
    with pytest.raises(ConfigurationError) as err:
        cw.set_field("llm.n_ctx", "8192", path=path)
    assert "\x1b" not in str(err.value) and "\\x1b" in str(err.value)


def test_restore_refuses_without_an_interactive_terminal(tmp_path: Path, monkeypatch, capsys) -> None:
    from maxim.runtime import config_cli

    path = _preserve(tmp_path, {"llm.n_ctx": 4096})
    monkeypatch.setattr(config_cli, "config_path", lambda: path)
    monkeypatch.setattr(config_cli, "_interactive_terminal", lambda: False)
    assert config_cli.run_config_subcommand(["restore-preserved"]) == 2
    assert "terminal" in capsys.readouterr().err
    assert "n_ctx" not in json.loads(path.read_text()).get("llm", {})
    assert config_cli.run_config_subcommand(["restore-preserved", "--yes"]) == 2  # no bypass flag exists


# ── review folds: sections are restored field by field; the classifier fails safe ───────────


def test_a_preserved_section_is_restored_field_by_field_and_every_field_is_shown(tmp_path: Path) -> None:
    """A planted sidecar holding whole sections used to replace the operator's sections wholesale:
    it emptied tools.deny, turned the console sandbox off, and silently reset console.port, unflagged."""
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "_format_version": cl.CONFIG_FORMAT_VERSION,
                "tools": {"deny": ["bash", "write_file"]},
                "console": {"sandbox": True, "port": 9000},
            }
        )
    )
    entries = {"tools": {"deny": []}, "console": {"sandbox": False}}
    wrapped = {k: {"value": v, "source_format_version": "9.9", "preserved_at": "t"} for k, v in entries.items()}
    cw.preserved_path(path).write_text(json.dumps({"_format_version": "1.0", "entries": wrapped}))
    plans: list = []
    cw.restore_preserved(path=path, confirm=lambda plan: plans.append(plan) or True)

    rows = {row.path: row for row in plans[0]}
    assert set(rows) == {"tools.deny", "console.sandbox"}  # one row per FIELD, not per section
    assert all(row.security_relevant for row in rows.values())
    assert rows["tools.deny"].current == ["bash", "write_file"]
    data = json.loads(path.read_text())
    assert data["console"]["port"] == 9000  # a field the preserved section lacked is untouched


def test_every_config_field_is_classified_and_only_tuning_knobs_are_unflagged() -> None:
    """Fail safe: anything not on the explicit tuning-knob list -- including a field a later build adds -- is
    flagged. Pinning the full split makes a new field's classification a visible decision."""
    unflagged = {p for p in cl.config_field_paths() if not cw.is_security_relevant(p)}
    assert unflagged == set(cw._NON_SECURITY_FIELDS)
    for sensitive in (
        "role",
        "llm.profile",
        "llm.auto_download",
        "tools.allow",
        "tools.deny",
        "console.sandbox",
        "lanes.large.remote_url",
        "lanes.large.remote_api_key_ref",
        "data.home",
        "cloud.enabled",
    ):
        assert cw.is_security_relevant(sensitive), sensitive
    assert cw.is_security_relevant("some_future_section.knob")


# ── verification-pass folds ────────────────────────────────────────────────────────────────


def _sidecar_with(path: Path, entries: dict[str, tuple]) -> None:
    wrapped = {k: {"value": v, "source_format_version": ver, "preserved_at": at} for k, (v, ver, at) in entries.items()}
    cw.preserved_path(path).write_text(json.dumps({"_format_version": "1.0", "entries": wrapped}))


def test_a_lane_tier_extra_is_never_restored_as_a_literal_key(tmp_path: Path) -> None:
    """`extra` is inlined into the tier on disk; restoring it as a field double-nested it."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION}))
    _sidecar_with(path, {"lanes.large.extra": ({"foo": 1}, "9.9", "t")})
    assert cw.restore_preserved(path=path, confirm=lambda plan: True) == []
    assert "extra" not in json.dumps(json.loads(path.read_text()))


def test_the_newest_of_two_colliding_values_wins_and_says_so(tmp_path: Path) -> None:
    """A section from one downgrade and a field from a later one land on the same field."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION}))
    _sidecar_with(
        path,
        {
            "tools.deny": (["old"], "1.2", "2026-01-01T00:00:00Z"),
            "tools": ({"deny": ["new"]}, "1.3", "2026-06-01T00:00:00Z"),
        },
    )
    plans: list = []
    cw.restore_preserved(path=path, confirm=lambda plan: plans.append(plan) or True)
    (row,) = plans[0]
    assert row.preserved == ["new"] and row.superseded == 1
    assert json.loads(path.read_text())["tools"]["deny"] == ["new"]


def test_the_current_value_shown_is_the_one_in_effect(tmp_path: Path) -> None:
    """An absent console.sandbox is in effect `false`; the diff says so rather than `null`."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION}))
    default_sandbox = cl.MaximConfig().console.sandbox
    _sidecar_with(path, {"console.sandbox": (not default_sandbox, "9.9", "t")})
    plans: list = []
    cw.restore_preserved(path=path, confirm=lambda plan: plans.append(plan) or False)
    assert plans[0][0].current == default_sandbox


def test_a_value_already_in_effect_is_not_asked_about_and_nothing_is_discarded_unseen(tmp_path: Path) -> None:
    """With nothing to restore, the sidecar stays exactly as it was: the operator saw none of it (an
    older superseded value in it must not vanish without a word)."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION, "llm": {"n_ctx": 4096}}))
    _sidecar_with(
        path,
        {
            "llm": ({"n_ctx": 8192}, "1.2", "2026-01-01T00:00:00Z"),
            "llm.n_ctx": (4096, "1.3", "2026-06-01T00:00:00Z"),
        },
    )
    before = cw.preserved_path(path).read_text()
    asked: list = []
    assert cw.restore_preserved(path=path, confirm=lambda plan: asked.append(plan) or True) == []
    assert asked == [] and cw.preserved_path(path).read_text() == before


def test_an_unparseable_timestamp_never_wins_a_collision(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION}))
    _sidecar_with(
        path,
        {
            "tools.deny": (["planted"], "9.9", "zzzz-not-a-date"),
            "tools": ({"deny": ["real"]}, "1.3", "2026-06-01T00:00:00Z"),
        },
    )
    plans: list = []
    cw.restore_preserved(path=path, confirm=lambda plan: plans.append(plan) or False)
    assert plans[0][0].preserved == ["real"]


@pytest.mark.parametrize("bad", ["dir", "plain_value"])
def test_a_malformed_sidecar_is_refused_loudly(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION}))
    sidecar = cw.preserved_path(path)
    if bad == "dir":
        sidecar.mkdir()
    else:
        sidecar.write_text(json.dumps({"_format_version": "1.0", "entries": {"tools.deny": ["x"]}}))
    with pytest.raises(ConfigurationError):
        cw.restore_preserved(path=path, confirm=lambda plan: True)


def test_restoring_other_fields_never_discards_an_unseen_value(tmp_path: Path) -> None:
    """Two set-aside values of a field this build does not know: restoring an unrelated field keeps both."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION}))
    # A known section carrying an unknown field, plus the same unknown field set aside separately: both
    # flatten onto `llm.future_knob`, so keeping only the newest per field would drop the older one.
    _sidecar_with(
        path,
        {
            "llm": ({"future_knob": 1, "n_ctx": 4096}, "1.2", "2026-01-01T00:00:00Z"),
            "llm.future_knob": (2, "1.3", "2026-06-01T00:00:00Z"),
        },
    )
    assert cw.restore_preserved(path=path, confirm=lambda plan: True) == ["llm.n_ctx"]
    left = json.loads(cw.preserved_path(path).read_text())["entries"]
    assert left["llm"]["value"]["future_knob"] == 1  # the older value, never shown, is kept
    assert left["llm.future_knob"]["value"] == 2


def test_a_restored_or_superseded_value_is_never_offered_again(tmp_path: Path) -> None:
    """A kept entry (for its unknown key) must not still carry the fields that were settled: otherwise the
    next restore re-offers an old value over the operator's later change, or brings a superseded one back."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"_format_version": cl.CONFIG_FORMAT_VERSION}))
    _sidecar_with(
        path,
        {
            "llm": ({"n_ctx": 2048, "future_knob": 1}, "1.2", "2026-01-01T00:00:00Z"),  # older, with an unknown key
            "llm.n_ctx": (4096, "1.3", "2026-06-01T00:00:00Z"),  # newer: supersedes 2048
        },
    )
    assert cw.restore_preserved(path=path, confirm=lambda plan: True) == ["llm.n_ctx"]
    cw.set_field("llm.n_ctx", "8192", path=path)  # the operator's later choice

    asked: list = []
    assert cw.restore_preserved(path=path, confirm=lambda plan: asked.append(plan) or True) == []
    assert asked == []  # neither 4096 (restored) nor 2048 (superseded) comes back
    left = json.loads(cw.preserved_path(path).read_text())["entries"]
    assert left == {"llm": {**left["llm"], "value": {"future_knob": 1}}}  # the unknown key is still kept
    assert json.loads(path.read_text())["llm"]["n_ctx"] == 8192
