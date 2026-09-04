"""Tests for Maxim API key management (src/maxim/tunnel/keys.py)."""

from __future__ import annotations

import os

import pytest

from maxim.tunnel import keys
from maxim.tunnel.cli import run_tunnel_subcommand


@pytest.fixture
def isolated_key_file(tmp_path, monkeypatch):
    """Redirect the key files to temp paths for the duration of a test.

    Mirrors the real layout since keys became NAMED (hardening PR 2): the
    mesh key stays the default, other names land beside it.
    """
    target = tmp_path / "api_key"
    monkeypatch.setattr(keys, "key_file_path", lambda name=keys.MESH_KEY_NAME: tmp_path / name)
    yield target


# ─── generation ────────────────────────────────────────────────────────────


class TestGeneration:
    def test_generate_returns_256_bit_string(self):
        key = keys.generate_key()
        assert isinstance(key, str)
        # 32 random bytes → 43-char base64url
        assert len(key) >= 40

    def test_generate_is_random(self):
        assert keys.generate_key() != keys.generate_key()


# ─── file I/O ──────────────────────────────────────────────────────────────


class TestFileIO:
    def test_key_exists_false_when_missing(self, isolated_key_file):
        assert keys.key_exists() is False

    def test_read_returns_none_when_missing(self, isolated_key_file):
        assert keys.read_key() is None

    def test_write_creates_file_and_parent(self, isolated_key_file):
        path = keys.write_key("test-key")
        assert path.is_file()
        assert path.read_text().strip() == "test-key"

    def test_read_after_write_round_trip(self, isolated_key_file):
        keys.write_key("round-trip-key")
        assert keys.read_key() == "round-trip-key"

    def test_write_overwrites_existing(self, isolated_key_file):
        keys.write_key("first")
        keys.write_key("second")
        assert keys.read_key() == "second"

    def test_write_sets_restrictive_perms_on_posix(self, isolated_key_file):
        if os.name == "nt":
            pytest.skip("POSIX-only permission check")
        keys.write_key("perm-test")
        mode = isolated_key_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_read_returns_none_on_empty_file(self, isolated_key_file):
        isolated_key_file.parent.mkdir(parents=True, exist_ok=True)
        isolated_key_file.write_text("")
        assert keys.read_key() is None

    def test_read_strips_trailing_whitespace(self, isolated_key_file):
        isolated_key_file.parent.mkdir(parents=True, exist_ok=True)
        isolated_key_file.write_text("abc-key\n\n")
        assert keys.read_key() == "abc-key"


# ─── ensure / rotate ───────────────────────────────────────────────────────


class TestEnsureAndRotate:
    def test_ensure_generates_new_when_missing(self, isolated_key_file):
        key = keys.ensure_key()
        assert key
        assert keys.read_key() == key

    def test_ensure_returns_existing(self, isolated_key_file):
        keys.write_key("existing-key")
        assert keys.ensure_key() == "existing-key"

    def test_rotate_always_replaces(self, isolated_key_file):
        original = keys.ensure_key()
        rotated = keys.rotate_key()
        assert original != rotated
        assert keys.read_key() == rotated


# ─── truncation for display ───────────────────────────────────────────────


class TestTruncation:
    def test_truncate_long_key(self):
        out = keys.truncate_for_display("abcdefghijklmnopqrstuvwxyz", keep=4)
        assert out.startswith("abcd")
        assert out.endswith("wxyz")
        assert "…" in out

    def test_truncate_short_key_unchanged(self):
        assert keys.truncate_for_display("short") == "short"


# ─── snippet rendering ────────────────────────────────────────────────────


class TestSnippets:
    def test_render_all_shells(self):
        out = keys.render_snippets("KEY123")
        assert set(out.keys()) == {"bash_zsh", "fish", "powershell", "cmd"}
        for snippet in out.values():
            assert "KEY123" in snippet
            assert keys.ENV_VAR in snippet

    def test_bash_snippet_has_both_session_and_persist(self):
        out = keys.render_snippets("K")
        # Session-only export
        assert "export" in out["bash_zsh"]
        # Persistence hint — the snippet was simplified to recommend
        # `maxim peer connect` for persistence instead of editing
        # ~/.bashrc or ~/.zshrc directly. The snippet still contains
        # both a session-only path and a persistence path; the
        # persistence path is now operator-friendly rather than
        # shell-rc-file-bound.
        assert "maxim peer connect" in out["bash_zsh"]

    def test_fish_snippet_uses_universal_var(self):
        out = keys.render_snippets("K")
        assert "set -Ux" in out["fish"]

    def test_powershell_snippet_has_env_and_profile(self):
        out = keys.render_snippets("K")
        assert "$env:" in out["powershell"]
        assert "$PROFILE" in out["powershell"]

    def test_cmd_snippet_has_setx_and_set(self):
        out = keys.render_snippets("K")
        assert "setx" in out["cmd"]
        assert "set " in out["cmd"]

    def test_format_all_contains_every_shell(self):
        text = keys.format_all_snippets("TESTKEY")
        for label in ("bash", "fish", "PowerShell", "cmd"):
            assert label in text


# ─── CLI: maxim tunnel key ─────────────────────────────────────────────────


class TestKeyCLI:
    def test_key_without_subaction(self, capsys):
        code = run_tunnel_subcommand(["key"])
        err = capsys.readouterr().err
        assert "show|rotate|export" in err
        assert code == 2

    def test_key_unknown_action(self, capsys):
        code = run_tunnel_subcommand(["key", "bogus"])
        assert code == 2

    def test_key_show_without_key(self, capsys, isolated_key_file):
        code = run_tunnel_subcommand(["key", "show"])
        out = capsys.readouterr().out
        assert "No API key" in out
        assert code == 1

    def test_key_show_prints_key(self, capsys, isolated_key_file):
        keys.write_key("display-me")
        code = run_tunnel_subcommand(["key", "show"])
        out = capsys.readouterr().out
        assert "display-me" in out
        assert code == 0

    def test_key_rotate_generates(self, capsys, isolated_key_file):
        code = run_tunnel_subcommand(["key", "rotate"])
        assert code == 0
        assert keys.read_key() is not None
        out = capsys.readouterr().out
        assert "Generated new API key" in out

    def test_key_rotate_warns_invalidation(self, capsys, isolated_key_file):
        keys.write_key("old-key")
        run_tunnel_subcommand(["key", "rotate"])
        out = capsys.readouterr().out
        assert "now invalid" in out
        assert "401" in out

    def test_key_export_without_key(self, capsys, isolated_key_file):
        code = run_tunnel_subcommand(["key", "export"])
        out = capsys.readouterr().out
        assert "No API key" in out
        assert code == 1

    def test_key_export_prints_snippets(self, capsys, isolated_key_file):
        keys.write_key("exportkey123")
        code = run_tunnel_subcommand(["key", "export"])
        out = capsys.readouterr().out
        assert "exportkey123" in out
        # All four shells present
        for label in ("bash", "fish", "PowerShell", "cmd"):
            assert label in out
        assert code == 0
