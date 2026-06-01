"""Regression guards for ``maxim.runtime.config_cli`` (C2 of
config_unification.md).

Surface covered:

* ``config get`` (all fields + single field + unknown field)
* ``config set`` (round-trip, unknown field refused, set coercion,
  inline API-key rejection)
* ``config path``
* ``config list`` (text + --json)
* ``config edit`` (skeleton seeded + post-edit validation)
* Verb dispatch + usage banner
* Exit codes match the documented contract (0/1/2)

Tests run the CLI verbs in-process by calling
``run_config_subcommand`` directly. ``capsys`` captures stdout/stderr;
``tmp_path`` + ``XDG_CONFIG_HOME`` redirect the config file location
so tests don't touch the developer's real config.
"""

from __future__ import annotations

import json

import pytest

from maxim.runtime.config_cli import run_config_subcommand
from maxim.runtime.config_loader import (
    LLMConfigSection,
    MaximConfig,
    config_path,
    load_config,
    reset_config_cache,
)


@pytest.fixture
def isolated_config_dir(tmp_path, monkeypatch):
    """Redirect ``~/.config/maxim`` to a tmp dir for the duration of the test."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    reset_config_cache()
    yield tmp_path
    reset_config_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch + usage
# ─────────────────────────────────────────────────────────────────────────────


class TestDispatch:
    def test_no_argv_prints_usage_and_exits_2(self, capsys, isolated_config_dir):
        rc = run_config_subcommand([])
        out = capsys.readouterr().out
        assert rc == 2
        assert "Usage: maxim config" in out

    def test_help_flag_prints_usage_and_exits_0(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["--help"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Usage: maxim config" in out

    def test_unknown_verb_exits_2(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["nonsense"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown config verb" in err


# ─────────────────────────────────────────────────────────────────────────────
# path
# ─────────────────────────────────────────────────────────────────────────────


class TestPath:
    def test_prints_resolved_config_path(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["path"])
        out = capsys.readouterr().out.strip()
        assert rc == 0
        assert str(config_path()) == out
        assert str(isolated_config_dir) in out


# ─────────────────────────────────────────────────────────────────────────────
# get
# ─────────────────────────────────────────────────────────────────────────────


class TestGet:
    def test_get_no_args_prints_all_fields_with_defaults(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["get"])
        out = capsys.readouterr().out
        assert rc == 0
        # Every absorbed field path should appear
        assert "llm.profile" in out
        assert "llm.n_ctx" in out
        assert "lanes.large.remote_url" in out
        assert "[source=default]" in out

    def test_get_single_field_default(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["get", "llm.n_ctx"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "llm.n_ctx:" in out
        assert "8192" in out
        assert "[source=default]" in out

    def test_get_unknown_field_exits_2(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["get", "llm.does_not_exist"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown field path" in err

    def test_get_single_field_with_env_set(self, capsys, isolated_config_dir, monkeypatch):
        monkeypatch.setenv("MAXIM_LLM_N_CTX", "16384")
        rc = run_config_subcommand(["get", "llm.n_ctx"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "16384" in out
        assert "[source=env]" in out

    def test_get_single_field_with_config_set(self, capsys, isolated_config_dir):
        # Set up a config.json file
        from maxim.runtime.config_writer import write_config

        write_config(
            MaximConfig(llm=LLMConfigSection(profile="qwen-32b")),
        )
        reset_config_cache()
        rc = run_config_subcommand(["get", "llm.profile"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "qwen-32b" in out
        assert "[source=config]" in out


# ─────────────────────────────────────────────────────────────────────────────
# list
# ─────────────────────────────────────────────────────────────────────────────


class TestList:
    def test_list_human_readable(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["list"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Resolved config:" in out
        assert "llm.profile" in out
        assert "Sources:" in out

    def test_list_json_output(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["list", "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        data = json.loads(out)
        assert "config_path" in data
        assert "fields" in data
        assert isinstance(data["fields"], list)
        paths = {f["path"] for f in data["fields"]}
        assert "llm.profile" in paths
        assert "lanes.large.remote_url" in paths


# ─────────────────────────────────────────────────────────────────────────────
# set
# ─────────────────────────────────────────────────────────────────────────────


class TestSet:
    def test_set_round_trip(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["set", "llm.profile", "qwen-32b"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "set llm.profile" in out

        # Verify persistence by re-reading
        reset_config_cache()
        rc = run_config_subcommand(["get", "llm.profile"])
        out = capsys.readouterr().out
        assert "qwen-32b" in out
        assert "[source=config]" in out

    def test_set_int_field_coerces_string(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["set", "proxy.max_concurrent", "8"])
        assert rc == 0
        loaded = load_config()
        assert loaded.proxy.max_concurrent == 8
        assert isinstance(loaded.proxy.max_concurrent, int)

    def test_set_bool_field_coerces_string(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["set", "llm.auto_download", "true"])
        assert rc == 0
        loaded = load_config()
        assert loaded.llm.auto_download is True

    def test_set_unknown_field_exits_2(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["set", "llm.nonsense", "value"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Unknown field path" in err

    def test_set_invalid_value_exits_2(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["set", "llm.n_ctx", "100"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "below minimum" in err

    def test_set_invalid_enum_exits_2(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["set", "role", "boss"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "expected one of" in err

    def test_set_missing_value_exits_2(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(["set", "llm.profile"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Usage:" in err

    def test_set_inline_api_key_rejected(self, capsys, isolated_config_dir):
        """Cross-confirmed I-3/IM3 fold reaches the CLI surface."""
        rc = run_config_subcommand(["set", "lanes.large.remote_api_key_ref", "sk-abc123def"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "Inline plaintext" in err

    def test_set_file_path_api_key_ref_accepted(self, capsys, isolated_config_dir):
        rc = run_config_subcommand(
            [
                "set",
                "lanes.large.remote_api_key_ref",
                "~/.config/maxim/api_key",
            ]
        )
        assert rc == 0
        loaded = load_config()
        assert loaded.lanes.large.remote_api_key_ref == "~/.config/maxim/api_key"

    def test_set_clear_via_null(self, capsys, isolated_config_dir):
        # Seed
        run_config_subcommand(["set", "llm.profile", "qwen-32b"])
        capsys.readouterr()
        # Clear
        rc = run_config_subcommand(["set", "llm.profile", "null"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "cleared" in out
        loaded = load_config()
        assert loaded.llm.profile is None

    def test_set_clear_via_dash(self, capsys, isolated_config_dir):
        run_config_subcommand(["set", "llm.profile", "qwen-32b"])
        capsys.readouterr()
        rc = run_config_subcommand(["set", "llm.profile", "-"])
        assert rc == 0
        loaded = load_config()
        assert loaded.llm.profile is None


# ─────────────────────────────────────────────────────────────────────────────
# edit
# ─────────────────────────────────────────────────────────────────────────────


class TestEdit:
    def test_edit_seeds_file_when_missing(self, capsys, isolated_config_dir, monkeypatch):
        # Editor is "true" — exits 0 without modifying anything
        monkeypatch.setenv("EDITOR", "true")
        monkeypatch.delenv("VISUAL", raising=False)
        target = config_path()
        assert not target.is_file()

        rc = run_config_subcommand(["edit"])
        assert rc == 0
        assert target.is_file()
        # Verify the seeded file parses cleanly
        loaded = load_config(target)
        assert loaded == MaximConfig()

    def test_edit_validates_after_save(self, capsys, isolated_config_dir, monkeypatch, tmp_path):
        # Create an editor stub that injects invalid JSON
        editor_script = tmp_path / "bad_editor.sh"
        editor_script.write_text('#!/bin/bash\necho "{not valid json" > "$1"\n')
        editor_script.chmod(0o755)
        monkeypatch.setenv("EDITOR", str(editor_script))
        monkeypatch.delenv("VISUAL", raising=False)

        rc = run_config_subcommand(["edit"])
        err = capsys.readouterr().err
        assert rc == 2
        assert "validation error" in err

    def test_edit_visual_takes_precedence_over_editor(self, capsys, isolated_config_dir, monkeypatch):
        monkeypatch.setenv("VISUAL", "true")
        monkeypatch.setenv("EDITOR", "false")  # would exit non-zero
        rc = run_config_subcommand(["edit"])
        assert rc == 0  # VISUAL=true wins


# ─────────────────────────────────────────────────────────────────────────────
# CI grep allow-list (IM2 fold)
# ─────────────────────────────────────────────────────────────────────────────


class TestWriterAllowList:
    """Sanity check that the canonical writer is the only call site
    of atomic_write_json against config.json. The CI grep in test.yml
    enforces this across the codebase; the test here is a tripwire
    that fires if the writer module itself is accidentally renamed or
    the entry-point pattern drifts."""

    def test_config_writer_module_exists_and_exports_write_config(self):
        from maxim.runtime import config_writer

        assert hasattr(config_writer, "write_config")
        assert hasattr(config_writer, "mutate_config")
        assert hasattr(config_writer, "set_field")

    def test_config_writer_uses_atomic_write_json(self):
        """The writer module must import from maxim.utils.atomic_io —
        bypassing this contract (e.g., raw open() + json.dump()) is a
        regression."""
        import inspect

        from maxim.runtime import config_writer

        source = inspect.getsource(config_writer)
        assert "atomic_write_json" in source
        # No raw write() of config.json
        assert ".write_text(json.dumps" not in source
        assert "open(" not in source or "open(tmp" not in source
