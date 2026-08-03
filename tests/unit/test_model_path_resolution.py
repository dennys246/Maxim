"""Model-path resolution guards (2026-08-03 root-cause fix).

The runtime prompted to download models ALREADY on disk at every launch:
install.sh wrote profiles with relative legacy paths
("data/models/LLM/...") and load_llm_config passed explicit model_path
through VERBATIM — no expanduser, no anchoring to the canonical
~/.maxim/models dir, no existence fallback to build_model_path. All
seven existence-check sites (lane ensure_available, warm-up, auto-spawn,
hot-swap, --list-models, doctor tiers) read that one poisoned field;
only the downloader resolved correctly ("Already exists" right after
the prompt).

Pins: explicit relative legacy paths re-anchor at model_dir(); ~ paths
expanduser; a nonexistent explicit path falls back to build_model_path
(canonical + case-insensitive); the DOCUMENTED user config location
(~/.maxim/config/llm.json) is consulted; profile_has_local_file — the
prompt guard itself — is True for a downloaded model declared with a
legacy relative path.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture()
def fake_maxim_home(tmp_path, monkeypatch):
    """Isolated MAXIM_DATA_HOME with one downloaded GGUF, and a CWD llm.json
    declaring it via the LEGACY relative path — the exact broken install.sh
    layout."""
    from maxim.utils import paths as paths_mod

    home = tmp_path / "maxim_home"
    llm_dir = home / "models" / "LLM"
    llm_dir.mkdir(parents=True)
    gguf = llm_dir / "SmolLM-1.7B-Instruct.Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF fake")
    monkeypatch.setenv("MAXIM_DATA_HOME", str(home))
    # data_home() caches after first call — reset so THIS test's home wins,
    # and again on teardown so we don't pollute later tests.
    paths_mod._reset_caches()

    workdir = tmp_path / "checkout"
    (workdir / "data" / "util").mkdir(parents=True)
    (workdir / "data" / "util" / "llm.json").write_text(
        json.dumps(
            {
                "enabled": True,
                "profile": "smollm-1.7b-instruct",
                "profiles": {
                    "smollm-1.7b-instruct": {
                        "backend": "llama_cpp",
                        # The legacy relative path install.sh used to write:
                        "model_path": "data/models/LLM/SmolLM-1.7B-Instruct.Q4_K_M.gguf",
                    }
                },
            }
        )
    )
    monkeypatch.chdir(workdir)
    monkeypatch.delenv("MAXIM_LLM_MODEL_PATH", raising=False)
    monkeypatch.delenv("MAXIM_LLM_CONFIG", raising=False)
    yield home, gguf
    paths_mod._reset_caches()


class TestExplicitPathResolution:
    def test_legacy_relative_path_reanchors_at_model_dir(self, fake_maxim_home):
        home, gguf = fake_maxim_home
        from maxim.models.language.config import load_llm_config

        cfg = load_llm_config(profile_override="smollm-1.7b-instruct")
        assert cfg.model_path == str(gguf), (
            "legacy 'data/models/LLM/...' must re-anchor at ~/.maxim/models — "
            "the verbatim passthrough is the every-launch download prompt bug"
        )
        assert os.path.exists(cfg.model_path)

    def test_tilde_path_is_expanded(self, fake_maxim_home, tmp_path, monkeypatch):
        home, gguf = fake_maxim_home
        # Point HOME at tmp and place the model where ~ resolves.
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "mymodels"
        target.mkdir()
        f = target / "custom.gguf"
        f.write_bytes(b"GGUF fake")
        monkeypatch.setenv("MAXIM_LLM_MODEL_PATH", "~/mymodels/custom.gguf")
        from maxim.models.language.config import load_llm_config

        cfg = load_llm_config(profile_override="smollm-1.7b-instruct")
        assert cfg.model_path == str(f)

    def test_nonexistent_explicit_path_falls_back_to_canonical(self, fake_maxim_home, monkeypatch):
        home, gguf = fake_maxim_home
        monkeypatch.setenv("MAXIM_LLM_MODEL_PATH", "/nowhere/else/model.gguf")
        from maxim.models.language.config import load_llm_config

        cfg = load_llm_config(profile_override="smollm-1.7b-instruct")
        # build_model_path finds the real file (samefile: macOS's
        # case-insensitive filesystem may return a different spelling of
        # the same path; Linux CI exercises the case-insensitive scan).
        assert os.path.exists(cfg.model_path)
        assert os.path.samefile(cfg.model_path, str(gguf))

    def test_prompt_guard_sees_the_downloaded_model(self, fake_maxim_home):
        """profile_has_local_file is the pre-prompt guard — pre-fix it was
        False for every legacy-declared profile, which is why the download
        prompt fired at every single launch."""
        home, gguf = fake_maxim_home
        from maxim.runtime.llm_server import profile_has_local_file

        assert profile_has_local_file("smollm-1.7b-instruct") is True


class TestConfigDiscovery:
    def test_user_config_location_is_consulted_first(self, fake_maxim_home, tmp_path, monkeypatch):
        from maxim.models.language import config as config_mod

        user_cfg_dir = tmp_path / "user_config"
        monkeypatch.setattr("maxim.utils.paths.user_config", lambda: user_cfg_dir, raising=True)
        candidates = config_mod._llm_config_candidates()
        cwd_entry = os.path.join(os.getcwd(), "data", "util", "llm.json")
        user_entry = str(user_cfg_dir / "llm.json")
        assert user_entry in candidates
        assert candidates.index(user_entry) < candidates.index(cwd_entry), (
            "documented user config must outrank the legacy checkout path"
        )
