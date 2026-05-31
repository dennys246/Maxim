"""Tests for the L2 user profile YAML loader.

Covers the schema-shape / collision-precedence / bad-schema / missing-file
matrix from [docs/plans/leader_ux_profile_management.md]. The loader is
tested via its public ``apply_user_profiles(builtins, aliases, models,
*, path)`` entry point with fresh dicts per test — no module-level state
is mutated, so tests stay isolated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from maxim.exceptions import ConfigurationError
from maxim.models.language.profile_loader import (
    _infer_quantization_from_filename,
    apply_user_profiles,
    load_user_profiles,
    profiles_config_path,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fresh_dicts() -> tuple[dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]]:
    """Return fresh (builtins, aliases, llm_models) dicts for one test.

    Each test gets its own dicts so collisions / mutations don't leak.
    Pre-populated with one canonical bundled entry so user-vs-builtin
    collision tests have something to collide with.
    """
    builtins: dict[str, dict[str, Any]] = {
        "qwen2.5-14b-instruct": {
            "backend": "llama_cpp",
            "model": "qwen2.5-14b-instruct",
            "model_base": "Qwen2.5-14B-Instruct",
            "prompt_style": "chatml",
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "n_ctx": 32768,
            "arch": {
                "n_layers": 48,
                "n_kv_heads": 8,
                "head_dim": 128,
                "kv_type_bytes": 2,
                "weights_gb": 8.4,
            },
        },
    }
    aliases: dict[str, str] = {
        "qwen2.5-14b": "qwen2.5-14b-instruct",
    }
    llm_models: dict[str, dict[str, Any]] = {}
    return builtins, aliases, llm_models


def _write_profiles(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "profiles.yml"
    target.write_text(body)
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────────


class TestPathResolution:
    def test_path_respects_xdg_config_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        # platform.system() may return Windows on Windows CI; the path
        # helper handles both branches. The XDG branch only fires on
        # non-Windows, so skip the assertion otherwise.
        import platform

        if platform.system() == "Windows":
            pytest.skip("XDG_CONFIG_HOME branch is POSIX-only")
        assert profiles_config_path() == tmp_path / "maxim" / "profiles.yml"

    def test_path_falls_back_to_home_dot_config(self, monkeypatch):
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        import platform

        if platform.system() == "Windows":
            pytest.skip("default XDG branch is POSIX-only")
        path = profiles_config_path()
        assert str(path).endswith(".config/maxim/profiles.yml")


# ─────────────────────────────────────────────────────────────────────────────
# Empty / missing file = silent no-op
# ─────────────────────────────────────────────────────────────────────────────


class TestEmptyAndMissing:
    def test_missing_file_returns_empty(self, tmp_path):
        result = load_user_profiles(path=tmp_path / "nonexistent.yml")
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        target = _write_profiles(tmp_path, "")
        assert load_user_profiles(path=target) == {}

    def test_whitespace_only_file_returns_empty(self, tmp_path):
        target = _write_profiles(tmp_path, "\n\n   \n")
        assert load_user_profiles(path=target) == {}

    def test_null_yaml_returns_empty(self, tmp_path):
        target = _write_profiles(tmp_path, "~\n")  # YAML null
        assert load_user_profiles(path=target) == {}

    def test_profiles_key_absent_returns_empty(self, tmp_path):
        target = _write_profiles(tmp_path, "version: 1\nother: stuff\n")
        assert load_user_profiles(path=target) == {}

    def test_profiles_key_null_returns_empty(self, tmp_path):
        target = _write_profiles(tmp_path, "profiles: ~\n")
        assert load_user_profiles(path=target) == {}

    def test_missing_file_apply_is_noop(self, tmp_path):
        builtins, aliases, llm_models = _fresh_dicts()
        snapshot_builtins = dict(builtins)
        snapshot_aliases = dict(aliases)
        apply_user_profiles(builtins, aliases, llm_models, path=tmp_path / "nonexistent.yml")
        assert builtins == snapshot_builtins
        assert aliases == snapshot_aliases
        assert llm_models == {}


# ─────────────────────────────────────────────────────────────────────────────
# Minimum valid profile (schema shape)
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaShape:
    def test_minimum_valid_hf_download_profile_merges(self, tmp_path):
        body = """
profiles:
  my-qwen-32b-q5:
    backend: llama_cpp
    prompt_style: chatml
    download:
      hf_repo: bartowski/Qwen2.5-32B-Instruct-GGUF
      hf_file: Qwen2.5-32B-Instruct-Q5_K_M.gguf
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        apply_user_profiles(builtins, aliases, llm_models, path=target)

        assert "my-qwen-32b-q5" in builtins
        merged = builtins["my-qwen-32b-q5"]
        # Required fields preserved
        assert merged["backend"] == "llama_cpp"
        assert merged["prompt_style"] == "chatml"
        # Auto-derived fields
        assert merged["model"] == "my-qwen-32b-q5"
        assert merged["model_base"] == "Qwen2.5-32B-Instruct-GGUF"
        # Defaults applied
        assert merged["n_ctx"] == 8192
        assert merged["stop"] == ["<|im_end|>", "<|endoftext|>"]
        # No arch supplied, so 'arch' is absent (callers handle with
        # coarse VRAM estimation per the plan).
        assert "arch" not in merged

        # LLM_MODELS entry created with the correct URL
        assert "my-qwen-32b-q5" in llm_models
        assert llm_models["my-qwen-32b-q5"]["url"] == (
            "https://huggingface.co/bartowski/Qwen2.5-32B-Instruct-GGUF/resolve/main/Qwen2.5-32B-Instruct-Q5_K_M.gguf"
        )
        assert llm_models["my-qwen-32b-q5"]["filename"] == "Qwen2.5-32B-Instruct-Q5_K_M.gguf"
        assert llm_models["my-qwen-32b-q5"]["quantization"] == "Q5_K_M"

    def test_local_path_profile_merges_with_expanduser(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        body = """
profiles:
  my-local-model:
    backend: llama_cpp
    prompt_style: llama3_instruct
    local_path: ~/models/custom.gguf
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        apply_user_profiles(builtins, aliases, llm_models, path=target)

        assert "my-local-model" in builtins
        assert builtins["my-local-model"]["model_base"] == "my-local-model"
        assert llm_models["my-local-model"]["url"] is None  # No download URL
        assert llm_models["my-local-model"]["local_path"] == (str(tmp_path / "models" / "custom.gguf"))

    def test_optional_fields_pass_through(self, tmp_path):
        body = """
profiles:
  my-qwen-32b-q5:
    backend: llama_cpp
    prompt_style: chatml
    n_ctx: 16384
    stop: ["<|custom|>"]
    aliases: [qwen32q5, q32]
    arch:
      n_layers: 64
      n_kv_heads: 8
      head_dim: 128
      kv_type_bytes: 2
      weights_gb: 23.0
    download:
      hf_repo: bartowski/Qwen2.5-32B-Instruct-GGUF
      hf_file: Qwen2.5-32B-Instruct-Q5_K_M.gguf
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        apply_user_profiles(builtins, aliases, llm_models, path=target)

        merged = builtins["my-qwen-32b-q5"]
        assert merged["n_ctx"] == 16384
        assert merged["stop"] == ["<|custom|>"]
        assert merged["arch"]["n_layers"] == 64
        assert aliases["qwen32q5"] == "my-qwen-32b-q5"
        assert aliases["q32"] == "my-qwen-32b-q5"


# ─────────────────────────────────────────────────────────────────────────────
# Collision precedence (user-wins with WARNING)
# ─────────────────────────────────────────────────────────────────────────────


class TestCollisionPrecedence:
    def test_user_profile_overrides_builtin_with_warning(self, tmp_path, caplog):
        body = """
profiles:
  qwen2.5-14b-instruct:
    backend: llama_cpp
    prompt_style: chatml
    download:
      hf_repo: bartowski/Qwen2.5-14B-Instruct-GGUF
      hf_file: Qwen2.5-14B-Instruct-Q5_K_M.gguf
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()

        with caplog.at_level(logging.WARNING, logger="maxim.models.language.profile_loader"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

        # User wins: the n_ctx default (8192) replaces the built-in (32768)
        # — the merged dict reflects the user entry's auto-derived defaults,
        # NOT a merge of fields. This is intentional per the plan's
        # "operator opted in" rationale.
        assert builtins["qwen2.5-14b-instruct"]["n_ctx"] == 8192
        # Warning emitted naming the profile + file path
        assert any(
            "qwen2.5-14b-instruct" in rec.message and "overrides bundled profile" in rec.message
            for rec in caplog.records
        )

    def test_user_alias_overrides_builtin_alias_with_warning(self, tmp_path, caplog):
        body = """
profiles:
  my-qwen-32b:
    backend: llama_cpp
    prompt_style: chatml
    aliases: [qwen2.5-14b]   # shadow the built-in alias
    download:
      hf_repo: bartowski/Qwen2.5-32B-Instruct-GGUF
      hf_file: Qwen2.5-32B-Instruct-Q4_K_M.gguf
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()

        with caplog.at_level(logging.WARNING, logger="maxim.models.language.profile_loader"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

        # User wins — alias now resolves to the user profile
        assert aliases["qwen2.5-14b"] == "my-qwen-32b"
        assert any("qwen2.5-14b" in rec.message and "overrides bundled alias" in rec.message for rec in caplog.records)

    def test_intra_user_alias_collision_is_hard_error(self, tmp_path):
        body = """
profiles:
  profile-a:
    backend: llama_cpp
    prompt_style: chatml
    aliases: [shared-alias]
    download:
      hf_repo: foo/bar
      hf_file: model-a.gguf
  profile-b:
    backend: llama_cpp
    prompt_style: chatml
    aliases: [shared-alias]
    download:
      hf_repo: foo/baz
      hf_file: model-b.gguf
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"shared-alias.*claimed by both"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)


# ─────────────────────────────────────────────────────────────────────────────
# Bad-schema error paths
# ─────────────────────────────────────────────────────────────────────────────


class TestSchemaErrors:
    def test_yaml_syntax_error_raises(self, tmp_path):
        target = _write_profiles(tmp_path, "profiles:\n  bad: [unclosed\n")
        with pytest.raises(ConfigurationError, match="not valid YAML"):
            load_user_profiles(path=target)

    def test_top_level_not_a_mapping_raises(self, tmp_path):
        target = _write_profiles(tmp_path, "- item1\n- item2\n")
        with pytest.raises(ConfigurationError, match="top-level must be a mapping"):
            load_user_profiles(path=target)

    def test_profiles_key_not_a_mapping_raises(self, tmp_path):
        target = _write_profiles(tmp_path, "profiles:\n  - just-a-list\n")
        with pytest.raises(ConfigurationError, match="'profiles:' must be a mapping"):
            load_user_profiles(path=target)

    def test_missing_backend_raises(self, tmp_path):
        body = """
profiles:
  m:
    prompt_style: chatml
    download: {hf_repo: foo/bar, hf_file: m.gguf}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"'backend' is required"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_bad_backend_enum_raises(self, tmp_path):
        body = """
profiles:
  m:
    backend: not-a-real-backend
    prompt_style: chatml
    download: {hf_repo: foo/bar, hf_file: m.gguf}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"'backend'.*one of"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_missing_prompt_style_raises(self, tmp_path):
        body = """
profiles:
  m:
    backend: llama_cpp
    download: {hf_repo: foo/bar, hf_file: m.gguf}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"'prompt_style' is required"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_bad_prompt_style_enum_raises(self, tmp_path):
        body = """
profiles:
  m:
    backend: llama_cpp
    prompt_style: not-a-template
    download: {hf_repo: foo/bar, hf_file: m.gguf}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"'prompt_style'.*one of"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_both_download_and_local_path_raises(self, tmp_path):
        body = """
profiles:
  m:
    backend: llama_cpp
    prompt_style: chatml
    download: {hf_repo: foo/bar, hf_file: m.gguf}
    local_path: ~/foo.gguf
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"exactly one of 'download:' or 'local_path:'"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_neither_download_nor_local_path_raises(self, tmp_path):
        body = """
profiles:
  m:
    backend: llama_cpp
    prompt_style: chatml
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"must specify either 'download:'.*'local_path:'"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_download_missing_hf_repo_raises(self, tmp_path):
        body = """
profiles:
  m:
    backend: llama_cpp
    prompt_style: chatml
    download: {hf_file: m.gguf}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"'download.hf_repo' is required"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_download_missing_hf_file_raises(self, tmp_path):
        body = """
profiles:
  m:
    backend: llama_cpp
    prompt_style: chatml
    download: {hf_repo: foo/bar}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"'download.hf_file' is required"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_negative_n_ctx_raises(self, tmp_path):
        body = """
profiles:
  m:
    backend: llama_cpp
    prompt_style: chatml
    n_ctx: -1
    download: {hf_repo: foo/bar, hf_file: m.gguf}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"'n_ctx' must be a positive integer"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_arch_missing_required_field_raises(self, tmp_path):
        body = """
profiles:
  m:
    backend: llama_cpp
    prompt_style: chatml
    arch:
      n_layers: 32
      n_kv_heads: 8
      # head_dim, kv_type_bytes, weights_gb missing
    download: {hf_repo: foo/bar, hf_file: m.gguf}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"'arch' is missing required fields"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)

    def test_profile_name_with_slash_raises(self, tmp_path):
        body = """
profiles:
  bad/name:
    backend: llama_cpp
    prompt_style: chatml
    download: {hf_repo: foo/bar, hf_file: m.gguf}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        with pytest.raises(ConfigurationError, match=r"must not contain '/' or ':'"):
            apply_user_profiles(builtins, aliases, llm_models, path=target)


# ─────────────────────────────────────────────────────────────────────────────
# Validation is atomic — partial-application doesn't happen
# ─────────────────────────────────────────────────────────────────────────────


class TestAtomicValidation:
    def test_one_bad_profile_blocks_all_user_profiles(self, tmp_path):
        """If any profile in the file fails validation, no user
        profile is merged. Avoids the "half the profiles loaded, half
        didn't, and the operator can't tell which is which" failure
        mode."""
        body = """
profiles:
  good-profile:
    backend: llama_cpp
    prompt_style: chatml
    download: {hf_repo: foo/bar, hf_file: good.gguf}
  bad-profile:
    backend: llama_cpp
    # missing prompt_style
    download: {hf_repo: foo/bar, hf_file: bad.gguf}
"""
        target = _write_profiles(tmp_path, body)
        builtins, aliases, llm_models = _fresh_dicts()
        snapshot_builtins = dict(builtins)
        with pytest.raises(ConfigurationError):
            apply_user_profiles(builtins, aliases, llm_models, path=target)
        # Neither profile was added — atomic semantics
        assert "good-profile" not in builtins
        assert builtins == snapshot_builtins


# ─────────────────────────────────────────────────────────────────────────────
# Quantization inference (best-effort)
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantizationInference:
    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("Qwen2.5-32B-Instruct-Q4_K_M.gguf", "Q4_K_M"),
            ("model-q5_k_m.gguf", "Q5_K_M"),
            ("Model.F16.gguf", "F16"),
            ("nothing-matchable.gguf", None),
        ],
    )
    def test_quantization_inference(self, filename, expected):
        assert _infer_quantization_from_filename(filename) == expected


# ─────────────────────────────────────────────────────────────────────────────
# CI grep proxy: yaml.safe_load is the only entry point in the loader
# ─────────────────────────────────────────────────────────────────────────────


def test_loader_module_uses_only_safe_load():
    """``yaml.load`` allows arbitrary Python object construction via
    untrusted YAML — this is a hard security boundary. The loader must
    use ``yaml.safe_load`` exclusively. The .github/workflows/test.yml
    grep enforces this; this test is the local-dev fast-feedback
    counterpart for the same invariant."""
    src = Path(__file__).resolve().parent.parent.parent / "src" / "maxim" / "models" / "language" / "profile_loader.py"
    text = src.read_text()
    # Must contain at least one safe_load call
    assert "yaml.safe_load" in text
    # Must NOT contain any yaml.load(...) call. Allow the substring
    # "yaml.safe_load" but reject bare "yaml.load(" or "yaml.load ".
    for line in text.splitlines():
        stripped = line.strip()
        # The comment that explains the rule mentions yaml.load — those
        # are inside string literals or comments. Skip docstring/comment
        # lines for the strict regex.
        if stripped.startswith("#"):
            continue
        # Bare "yaml.load(" or "yaml.load " in code is disallowed.
        # The regex below ignores "yaml.safe_load" because of the
        # explicit "safe_" prefix check.
        if "yaml.load(" in stripped and "yaml.safe_load" not in stripped:
            pytest.fail(
                f"profile_loader.py uses bare yaml.load() — must use "
                f"yaml.safe_load() exclusively. Offending line: {stripped!r}"
            )
