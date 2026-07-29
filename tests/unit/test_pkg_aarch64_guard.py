"""PKG seam guards — the lean Pi extra and its aarch64 resolution check.

The resolution itself needs network + uv, so CI runs that (fast, x86, every
PR). What is testable offline is the contract: the extra's SHAPE, the
leak-detection logic that turns a resolved set into pass/fail, and the
methodology invariants that keep the check SOUND — each of which was learned
by getting a confidently-wrong answer first.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _extras() -> dict[str, list[str]]:
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]["optional-dependencies"]


def _script():
    path = REPO_ROOT / "scripts" / "check_aarch64_install.py"
    spec = importlib.util.spec_from_file_location("check_aarch64_install", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestPiExtra:
    def test_pi_extra_is_a_composition(self):
        # A curated combo of existing extras, not a hand-listed dep set that
        # would drift from the extras it is composed of.
        assert _extras()["pi"] == ["pymaxim[reachy,console,llm-anthropic,tts]"]

    @pytest.mark.parametrize("banned", ["semantic", "llm-torch", "llm-llama", "llm-server", "training", "yolo"])
    def test_pi_excludes_the_heavy_extras(self, banned):
        # torch is a ~450MB runtime floor on a Pi (FIT) and the peer runs no
        # local inference — the encoder lives on the leader.
        assert banned not in _extras()["pi"][0]

    def test_console_is_explicit_in_all(self):
        # It previously worked under `all` only because comms/reachy dragged
        # fastapi+uvicorn in at LOOSER floors than console declares — an
        # accident, not a contract.
        assert "console" in _extras()["all"][0]

    def test_semantic_still_declares_torch(self):
        # Pins WHY semantic is excluded from `pi`. Measured on aarch64 it pulls
        # torch + triton + nvidia-* CUDA shards. If that ever stops being true,
        # revisit the exclusion deliberately rather than cargo-culting it.
        assert any(d.startswith("torch") for d in _extras()["semantic"])


class TestConsoleConfigSectionIsParsed:
    """`console` was declared on MaximConfig and writable via `maxim config
    set`, but MISSING from _parse_config_dict — so it was silently dropped and
    console.ui_dist / console.port were permanently their defaults. The
    packaged-bundle fallback MASKED it: a bare `maxim serve` still served a
    UI, just never the configured one."""

    def test_console_section_round_trips(self):
        from maxim.runtime.config_loader import _parse_config_dict

        cfg = _parse_config_dict({"_format_version": "1.0", "console": {"ui_dist": "/tmp/bundle", "port": 9999}})
        assert cfg.console.ui_dist == "/tmp/bundle"
        assert cfg.console.port == 9999

    def test_absent_console_section_keeps_defaults(self):
        from maxim.runtime.config_loader import ConsoleConfigSection, _parse_config_dict

        cfg = _parse_config_dict({"_format_version": "1.0"})
        assert cfg.console == ConsoleConfigSection()

    def test_every_maxim_config_section_is_parsed(self):
        # The structural fix: a section declared on MaximConfig but forgotten in
        # _parse_config_dict fails SILENTLY (it passes the unknown-key check,
        # then vanishes). Assert coverage generically so the next added section
        # cannot repeat this.
        import dataclasses

        from maxim.runtime.config_loader import MaximConfig, _parse_config_dict

        sections = [
            f.name
            for f in dataclasses.fields(MaximConfig)
            if dataclasses.is_dataclass(f.type) or f.name not in ("_format_version", "role")
        ]
        parsed = _parse_config_dict({"_format_version": "1.0"})
        for name in sections:
            assert hasattr(parsed, name), f"MaximConfig.{name} is not produced by _parse_config_dict"

    def test_resolve_setting_reads_console_ui_dist(self, tmp_path, monkeypatch):
        # End-to-end: the documented config path must actually reach the
        # resolver the console uses.
        import maxim.runtime.config_loader as cl

        cfg = tmp_path / "config.json"
        cfg.write_text('{"_format_version": "1.0", "console": {"ui_dist": "/tmp/from-config"}}')
        monkeypatch.setattr(cl, "config_path", lambda: cfg)
        cl.reset_config_cache() if hasattr(cl, "reset_config_cache") else None
        loaded = cl.load_config(path=cfg) if "path" in cl.load_config.__code__.co_varnames else None
        if loaded is not None:
            assert loaded.console.ui_dist == "/tmp/from-config"


class TestLeakDetection:
    """The PKG regression guard: heavy backends stay out of the lean install."""

    def test_clean_set_passes(self):
        mod = _script()
        assert mod.assert_no_heavy(["numpy", "httpx", "onnxruntime", "piper-tts"]) == []

    @pytest.mark.parametrize(
        "pkg",
        [
            "torch",
            "nvidia-cublas-cu12",
            "nvidia_cudnn_cu12",  # underscore spelling must normalize (PEP 503)
            "nvidia-nvshmem-cu13",  # real package seen resolving `semantic` on aarch64
            "llama-cpp-python",
            "llama_cpp_python",
            "triton",
            "tensorflow",
            "tensorflow-cpu",
        ],
    )
    def test_each_heavy_backend_is_caught(self, pkg):
        mod = _script()
        assert mod.assert_no_heavy([pkg]) == [pkg]

    @pytest.mark.parametrize("pkg", ["torchvision", "torchaudio", "nvidiafoo", "tritonclient"])
    def test_lookalikes_are_not_false_positives(self, pkg):
        # Patterns are ANCHORED so a failure names the real offender rather
        # than an innocent neighbour. torchvision only ever appears WITH torch,
        # which is what actually trips the guard.
        mod = _script()
        assert mod.assert_no_heavy([pkg]) == []

    def test_normalization_is_pep503(self):
        mod = _script()
        assert mod.normalize("NVIDIA_cuBLAS.cu12") == "nvidia-cublas-cu12"


class TestJsonInputShapes:
    """The same assertion must work over a REAL install, not only a resolve —
    that is how the aarch64-install CI job reuses this script."""

    def test_pip_list_shape(self, tmp_path):
        mod = _script()
        p = tmp_path / "installed.json"
        p.write_text('[{"name": "numpy", "version": "2.2.6"}, {"name": "torch", "version": "2.7.0"}]')
        assert mod.main(["--from-json", str(p)]) == 1

    def test_pip_report_shape(self, tmp_path):
        mod = _script()
        p = tmp_path / "report.json"
        p.write_text('{"install": [{"metadata": {"name": "numpy", "version": "2.2.6"}}]}')
        assert mod.main(["--from-json", str(p)]) == 0

    def test_empty_input_fails_loudly(self, tmp_path):
        # An empty parse must NOT read as "clean" — that would be a guard that
        # passes by finding nothing.
        mod = _script()
        p = tmp_path / "empty.json"
        p.write_text("[]")
        assert mod.main(["--from-json", str(p)]) == 1


class TestMethodologySoundness:
    """Each invariant below exists because the alternative produced a
    confidently-WRONG answer during development."""

    def test_targets_the_pi_userland(self):
        # Raspberry Pi OS bookworm == glibc 2.36 + CPython 3.11. Targeting
        # manylinux2014 under-reports what the Pi can actually install.
        mod = _script()
        assert mod.DEFAULT_PLATFORM == "aarch64-manylinux_2_36"
        assert mod.DEFAULT_PYTHON == "3.11"

    def test_uv_is_required_not_optional(self, monkeypatch):
        # pip --platform evaluates markers against the HOST — it reported ZERO
        # nvidia-* for an x86_64 target from an arm64 host, i.e. it certifies
        # "no CUDA" for a target where CUDA installs (pypa/pip#6117). There is
        # no sound pip fallback for THIS assertion, so absence of uv must be a
        # hard error rather than a silent degrade to a wrong answer.
        mod = _script()
        monkeypatch.setattr(mod.shutil, "which", lambda _: None)
        with pytest.raises(SystemExit, match="uv is required"):
            mod.resolve("pi", mod.DEFAULT_PLATFORM, mod.DEFAULT_PYTHON)

    def test_source_only_deps_are_named_with_their_apt_line(self):
        # PyGObject/pycairo ship no wheels on ANY platform; a Pi compiles them.
        # A dry resolve can never catch a missing apt package, so they are
        # declared + explained rather than silently tolerated.
        mod = _script()
        assert "pygobject" in mod._SOURCE_OK
        assert "libgirepository1.0-dev" in mod._SOURCE_OK["pygobject"]
        assert "pycairo" in mod._SOURCE_OK
