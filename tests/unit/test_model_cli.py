"""Tests for `maxim model add/remove/list` (L3 of leader_ux_profile_management.md).

Drives ``run_model_subcommand`` with ``path=tmp_path / 'profiles.yml'``
so no test touches the operator's real ``~/.config/maxim/profiles.yml``.
The L2 loader is exercised indirectly via the round-trip — every add
followed by a load proves both halves stay in sync.
"""

from __future__ import annotations

import pytest
import yaml

from maxim.models.language.profile_loader import (
    apply_user_profiles,
    load_user_profiles,
)
from maxim.models.model_cli import (
    _parse_hf_arg,
    infer_chat_format,
    run_model_subcommand,
)


# ─────────────────────────────────────────────────────────────────────────────
# Chat-format auto-inference
# ─────────────────────────────────────────────────────────────────────────────


class TestChatFormatInference:
    @pytest.mark.parametrize(
        "name,repo,expected",
        [
            # Qwen
            ("my-qwen-32b", None, "chatml"),
            ("my-model", "bartowski/Qwen2.5-32B-Instruct-GGUF", "chatml"),
            # Llama 3
            ("llama-3.1-70b-mine", None, "llama3_instruct"),
            ("llama3-custom", None, "llama3_instruct"),
            # Llama 2
            ("llama-2-quirky", None, "llama2_chat"),
            ("llama2-q5", None, "llama2_chat"),
            # Mixtral (must match BEFORE mistral falls through)
            ("my-mixtral", None, "mistral_instruct"),
            ("mistral-7b-mine", None, "mistral_instruct"),
            # Phi
            ("phi-3-custom", None, "phi3"),
            ("phi3-mini-cpu", None, "phi3"),
            ("phi-2-mine", None, "phi"),
            ("phi2-extra", None, "phi"),
            # Gemma
            ("gemma-7b-extra", None, "gemma"),
            # Repo basename used when name doesn't match
            ("random-name", "google/gemma-7b-it-GGUF", "gemma"),
            # No match -> None
            ("totally-novel-model", None, None),
            ("smollm-experimental", None, None),  # smollm not in the table
        ],
    )
    def test_infer_chat_format(self, name, repo, expected):
        assert infer_chat_format(name, repo) == expected

    def test_inference_is_case_insensitive(self):
        assert infer_chat_format("MY-QWEN-32B", None) == "chatml"
        assert infer_chat_format("my-model", "Bartowski/QWEN2.5-32B-Instruct-GGUF") == "chatml"


# ─────────────────────────────────────────────────────────────────────────────
# --hf arg parsing
# ─────────────────────────────────────────────────────────────────────────────


class TestHfArgParsing:
    def test_simple_repo_file(self):
        repo, filename = _parse_hf_arg("bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q4_K_M.gguf")
        assert repo == "bartowski/Qwen2.5-32B-Instruct-GGUF"
        assert filename == "Qwen2.5-32B-Instruct-Q4_K_M.gguf"

    def test_multi_colon_filename_preserved(self):
        # The first ":" separates repo from filename. Any subsequent
        # ":" stays inside the filename. Rare in practice but worth
        # pinning so a future "fix" doesn't accidentally split twice.
        repo, filename = _parse_hf_arg("foo/bar:multi:colon:file.gguf")
        assert repo == "foo/bar"
        assert filename == "multi:colon:file.gguf"

    def test_missing_colon_exits_with_code_2(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _parse_hf_arg("just-a-repo-no-file")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "REPO:FILE format" in err

    def test_empty_repo_exits(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _parse_hf_arg(":just-a-file.gguf")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "must be non-empty" in err

    def test_empty_file_exits(self, capsys):
        with pytest.raises(SystemExit) as exc:
            _parse_hf_arg("foo/bar:")
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert "must be non-empty" in err


# ─────────────────────────────────────────────────────────────────────────────
# Full add → list → remove cycle
# ─────────────────────────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_add_writes_yaml_then_list_finds_it(self, tmp_path, capsys):
        target = tmp_path / "profiles.yml"
        rc = run_model_subcommand(
            [
                "add",
                "my-qwen-32b-q5",
                "--hf",
                "bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q5_K_M.gguf",
                "--n-ctx",
                "16384",
                "--alias",
                "qwen32q5",
            ],
            path=target,
        )
        assert rc == 0
        # File was created
        assert target.is_file()
        # YAML round-trips through the loader cleanly
        raw = load_user_profiles(path=target)
        assert "my-qwen-32b-q5" in raw
        entry = raw["my-qwen-32b-q5"]
        assert entry["backend"] == "llama_cpp"
        assert entry["prompt_style"] == "chatml"  # inferred from "qwen"
        assert entry["n_ctx"] == 16384
        assert entry["download"]["hf_repo"] == "bartowski/Qwen2.5-32B-Instruct-GGUF"
        assert entry["aliases"] == ["qwen32q5"]

        # `list` surfaces it
        rc = run_model_subcommand(["list"], path=target)
        assert rc == 0
        out = capsys.readouterr().out
        assert "my-qwen-32b-q5" in out
        assert "chatml" in out
        assert "qwen32q5" in out

        # And the L2 loader successfully consumes it (proves CLI→loader
        # round-trip integrity end-to-end)
        builtins: dict = {}
        aliases: dict = {}
        llm_models: dict = {}
        apply_user_profiles(builtins, aliases, llm_models, path=target)
        assert "my-qwen-32b-q5" in builtins
        assert aliases["qwen32q5"] == "my-qwen-32b-q5"

        # Remove cleans it up
        rc = run_model_subcommand(["remove", "my-qwen-32b-q5"], path=target)
        assert rc == 0
        raw = load_user_profiles(path=target)
        assert "my-qwen-32b-q5" not in raw

    def test_local_path_add(self, tmp_path):
        target = tmp_path / "profiles.yml"
        gguf = tmp_path / "fake.gguf"
        gguf.write_bytes(b"")  # presence not validated at add time
        rc = run_model_subcommand(
            [
                "add",
                "my-local",
                "--local",
                str(gguf),
                "--chat-format",
                "llama3_instruct",
            ],
            path=target,
        )
        assert rc == 0
        raw = load_user_profiles(path=target)
        assert raw["my-local"]["local_path"] == str(gguf)
        assert "download" not in raw["my-local"]


# ─────────────────────────────────────────────────────────────────────────────
# Refuse to overwrite without --force
# ─────────────────────────────────────────────────────────────────────────────


class TestForceOverwrite:
    def test_add_refuses_existing_profile_without_force(self, tmp_path, capsys):
        target = tmp_path / "profiles.yml"
        argv = [
            "add",
            "my-qwen-32b",
            "--hf",
            "bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q4_K_M.gguf",
        ]
        assert run_model_subcommand(argv, path=target) == 0
        rc = run_model_subcommand(argv, path=target)
        assert rc == 2
        err = capsys.readouterr().err
        assert "already exists" in err
        assert "--force" in err

    def test_add_with_force_overwrites(self, tmp_path):
        target = tmp_path / "profiles.yml"
        # First add at Q4
        rc = run_model_subcommand(
            [
                "add",
                "my-qwen-32b",
                "--hf",
                "bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q4_K_M.gguf",
            ],
            path=target,
        )
        assert rc == 0
        # Overwrite at Q5 with --force
        rc = run_model_subcommand(
            [
                "add",
                "my-qwen-32b",
                "--hf",
                "bartowski/Qwen2.5-32B-Instruct-GGUF:Qwen2.5-32B-Instruct-Q5_K_M.gguf",
                "--force",
            ],
            path=target,
        )
        assert rc == 0
        raw = load_user_profiles(path=target)
        assert raw["my-qwen-32b"]["download"]["hf_file"] == "Qwen2.5-32B-Instruct-Q5_K_M.gguf"


# ─────────────────────────────────────────────────────────────────────────────
# Chat-format must be supplied when inference fails
# ─────────────────────────────────────────────────────────────────────────────


class TestChatFormatRequired:
    def test_chat_format_required_when_inference_fails(self, tmp_path, capsys):
        target = tmp_path / "profiles.yml"
        rc = run_model_subcommand(
            [
                "add",
                "totally-novel-model",
                "--hf",
                "user/unknown-repo:model.gguf",
            ],
            path=target,
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "could not infer --chat-format" in err
        # File should NOT have been written
        assert not target.is_file()

    def test_explicit_chat_format_overrides_inference(self, tmp_path):
        """User can override even when inference would produce a different
        answer — e.g., adding a Qwen-named model that uses llama3 template."""
        target = tmp_path / "profiles.yml"
        rc = run_model_subcommand(
            [
                "add",
                "my-qwen-weird",
                "--hf",
                "user/repo:model.gguf",
                "--chat-format",
                "llama3_instruct",
            ],
            path=target,
        )
        assert rc == 0
        raw = load_user_profiles(path=target)
        assert raw["my-qwen-weird"]["prompt_style"] == "llama3_instruct"


# ─────────────────────────────────────────────────────────────────────────────
# Remove error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestRemoveErrors:
    def test_remove_when_no_file_exists(self, tmp_path, capsys):
        target = tmp_path / "profiles.yml"
        rc = run_model_subcommand(["remove", "anything"], path=target)
        assert rc == 1
        err = capsys.readouterr().err
        assert "no user profiles file" in err

    def test_remove_unknown_profile(self, tmp_path, capsys):
        target = tmp_path / "profiles.yml"
        # Seed the file with a different profile
        target.write_text("profiles:\n  other: {backend: llama_cpp, prompt_style: chatml, local_path: /x}\n")
        rc = run_model_subcommand(["remove", "not-here"], path=target)
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err


# ─────────────────────────────────────────────────────────────────────────────
# List on edge cases
# ─────────────────────────────────────────────────────────────────────────────


class TestListEdgeCases:
    def test_list_when_no_file(self, tmp_path, capsys):
        target = tmp_path / "profiles.yml"
        rc = run_model_subcommand(["list"], path=target)
        assert rc == 0
        out = capsys.readouterr().out
        assert "no user profiles file" in out

    def test_list_when_file_has_empty_profiles_key(self, tmp_path, capsys):
        target = tmp_path / "profiles.yml"
        target.write_text("profiles: {}\n")
        rc = run_model_subcommand(["list"], path=target)
        assert rc == 0
        out = capsys.readouterr().out
        assert "no user profiles defined" in out


# ─────────────────────────────────────────────────────────────────────────────
# Refuses bad profile names (slashes / colons)
# ─────────────────────────────────────────────────────────────────────────────


class TestProfileNameValidation:
    @pytest.mark.parametrize("bad_name", ["bad/name", "bad:name", "weird/with:both"])
    def test_refuses_slash_or_colon_in_name(self, tmp_path, capsys, bad_name):
        target = tmp_path / "profiles.yml"
        rc = run_model_subcommand(
            [
                "add",
                bad_name,
                "--hf",
                "foo/bar:m.gguf",
                "--chat-format",
                "chatml",
            ],
            path=target,
        )
        assert rc == 2
        err = capsys.readouterr().err
        assert "must not contain '/' or ':'" in err


# ─────────────────────────────────────────────────────────────────────────────
# YAML output is loader-compatible (round-trip integrity)
# ─────────────────────────────────────────────────────────────────────────────


def test_written_yaml_is_loader_compatible(tmp_path):
    """Stronger than the round-trip test above: directly load the
    written file with the loader and assert every field survived."""
    target = tmp_path / "profiles.yml"
    rc = run_model_subcommand(
        [
            "add",
            "round-trip-test",
            "--hf",
            "user/repo:model-Q4_K_M.gguf",
            "--chat-format",
            "chatml",
            "--n-ctx",
            "4096",
            "--alias",
            "rtt",
            "--alias",
            "rt-test",
        ],
        path=target,
    )
    assert rc == 0
    # Read back with PyYAML directly
    data = yaml.safe_load(target.read_text())
    entry = data["profiles"]["round-trip-test"]
    assert entry["backend"] == "llama_cpp"
    assert entry["prompt_style"] == "chatml"
    assert entry["n_ctx"] == 4096
    assert entry["download"]["hf_repo"] == "user/repo"
    assert entry["download"]["hf_file"] == "model-Q4_K_M.gguf"
    assert entry["aliases"] == ["rtt", "rt-test"]
    # Apply with the loader — no errors, fields propagate
    builtins: dict = {}
    aliases_dict: dict = {}
    llm_models: dict = {}
    apply_user_profiles(builtins, aliases_dict, llm_models, path=target)
    assert builtins["round-trip-test"]["model"] == "round-trip-test"
    assert builtins["round-trip-test"]["n_ctx"] == 4096
    assert aliases_dict["rtt"] == "round-trip-test"
    assert aliases_dict["rt-test"] == "round-trip-test"
