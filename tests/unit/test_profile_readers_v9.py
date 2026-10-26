"""V9 / step 0a guard (#1315 §1A.1): every public profile reader resolves a user ``profiles.yml`` profile and alias.

WHAT THIS FILE IS. NOT a red gate: a guard that PASSES today, because ``models/language/config.py`` merges
``$XDG_CONFIG_HOME/maxim/profiles.yml`` into ``_BUILTIN_PROFILES``, ``_PROFILE_ALIASES`` and
``models/download.py::LLM_MODELS`` at import (``_apply_user_profiles_at_import``), and that must KEEP passing through
step 0a, which moves that load to a first-use ``_ensure_user_profiles()`` (``docs/plans/atonia_gate_approach_notes.md``
§1A.1: "Red gate, by outcome rather than by enumeration"). A reader that still reads the raw dicts after 0a would see
no user profile and fail here. It is parametrised over ``READERS``: a new public reader joins that tuple and gets a
branch in ``_READER_BODY``.

HOW. Each reader runs in a fresh ``sys.executable -I`` child whose ``XDG_CONFIG_HOME`` holds one user profile
(``PROFILE``) with one alias (``ALIAS``). A malformed ``profiles.yml`` must raise ``ConfigurationError`` on the first
access AND on the second (§1A.1: "Failure is never cached").

A reader that fails today is a real finding and is a strict xfail with its reason (``KNOWN_FAILING``).
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
MARKER = "@@PROFILE_READER@@"
CHILD_TIMEOUT_S = 180

PROFILE = "probe-v9-model"
ALIAS = "probev9alias"
HF_FILE = "ProbeV9-Model.Q4_K_M.gguf"

PROFILES_YML = textwrap.dedent(
    f"""\
    profiles:
      {PROFILE}:
        backend: llama_cpp
        prompt_style: chatml
        download:
          hf_repo: example/ProbeV9-GGUF
          hf_file: {HF_FILE}
        n_ctx: 4096
        aliases: [{ALIAS}]
        arch:
          n_layers: 8
          n_kv_heads: 2
          head_dim: 64
          kv_type_bytes: 2
          weights_gb: 0.5
    """
)

MALFORMED_YML = "profiles: [unclosed\n  - : :\n"

READERS: tuple[str, ...] = (
    "cli --list-models",
    "maxim.list_models",
    "normalize_llm_profile",
    "list_llm_profiles",
    "build_model_path (load_llm_config)",
    "models.download LLM_MODELS",
    "ensure_available lookup",
    "doctor VRAM projection (name)",
    "doctor VRAM projection (alias)",
)

# Readers that do not resolve the user profile TODAY (real findings, each a strict xfail).
KNOWN_FAILING: dict[str, str] = {
    "models.download LLM_MODELS": (
        "the user profile is merged into models.download.LLM_MODELS only as a side effect of importing "
        "models.language.config, so a process that reads the table first (import maxim.models.download) misses it; "
        "flips with step 0a's ensuring download.llm_models() accessor"
    ),
    "doctor VRAM projection (alias)": (
        "doctor/checks.py::_check_vram_context_pressure looks llm.profile up in _BUILTIN_PROFILES without "
        "normalising it, so an alias set as llm.profile gets 'VRAM projection unavailable'; flips when the doctor "
        "lookup normalises (step 0a's accessor migration, or its own fix)"
    ),
}


class ScenarioBroken(Exception):
    """The scenario did not happen as designed: NOT what the guard checks."""


_READER_BODY = textwrap.dedent(
    r"""
    import sys, json, io, os, contextlib
    CFG = json.load(open(sys.argv[1]))
    sys.path.insert(0, CFG["src"])
    sys.dont_write_bytecode = True
    MARKER, NAME, ALIAS, HF_FILE = CFG["marker"], CFG["profile"], CFG["alias"], CFG["hf_file"]
    OUT = {"ok": False, "detail": None}

    def emit():
        print(MARKER + json.dumps(OUT, default=repr), flush=True)

    import maxim
    OUT["maxim_file"] = maxim.__file__
    reader = CFG["reader"]
    try:
        if reader == "cli --list-models":
            from maxim.cli import _handle_list_models
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _handle_list_models()
            text = buf.getvalue()
            OUT["ok"] = NAME in text and ALIAS in text
            OUT["detail"] = [ln for ln in text.splitlines() if NAME in ln or ALIAS in ln]
        elif reader == "maxim.list_models":
            models = maxim.list_models()
            names = [m.name for group in models.values() for m in group]
            OUT["ok"] = NAME in names
            OUT["detail"] = len(names)
        elif reader == "normalize_llm_profile":
            from maxim.models.language.config import normalize_llm_profile
            got = (normalize_llm_profile(ALIAS), normalize_llm_profile(NAME))
            OUT["ok"] = got == (NAME, NAME)
            OUT["detail"] = got
        elif reader == "list_llm_profiles":
            from maxim.models.language.config import list_llm_profiles
            OUT["ok"] = NAME in list_llm_profiles()
        elif reader == "build_model_path (load_llm_config)":
            from maxim.models.language.config import load_llm_config
            cfg = load_llm_config(profile_override=ALIAS)
            OUT["detail"] = (cfg.profile, cfg.model_path)
            OUT["ok"] = cfg.profile == NAME and os.path.basename(cfg.model_path) == HF_FILE
        elif reader == "models.download LLM_MODELS":
            from maxim.models import download
            table = download.llm_models() if hasattr(download, "llm_models") else download.LLM_MODELS
            OUT["ok"] = NAME in table
        elif reader == "ensure_available lookup":
            from unittest.mock import MagicMock
            from maxim.models import download
            log = MagicMock()
            download.ensure_available(NAME, auto=False, interactive=False, logger=log)
            said = [str(c) for c in log.method_calls]
            OUT["detail"] = said
            OUT["ok"] = not any("is not in the LLM_MODELS" in c for c in said)
        elif reader.startswith("doctor VRAM projection"):
            from unittest.mock import patch
            os.environ["MAXIM_LLM_PROFILE"] = ALIAS if "(alias)" in reader else NAME
            os.environ["MAXIM_LLM_N_CTX"] = "4096"
            from maxim.doctor import checks
            with patch("maxim.runtime.capabilities.detect_compute_resources", return_value=(True, "cuda", 24.0, 64.0)):
                rows = checks._check_vram_context_pressure()
            msgs = [r.message for r in rows]
            OUT["detail"] = msgs
            OUT["ok"] = bool(msgs) and msgs[0].startswith("projected")
        else:
            OUT["detail"] = "no branch for reader " + reader
            OUT["broken"] = True
    except Exception as e:  # noqa: BLE001 -- reported to the parent
        OUT["detail"] = type(e).__name__ + ": " + str(e)[:300]
    emit()
    """
)

_MALFORMED_BODY = textwrap.dedent(
    r"""
    import sys, json, importlib
    CFG = json.load(open(sys.argv[1]))
    sys.path.insert(0, CFG["src"])
    sys.dont_write_bytecode = True
    OUT = {"accesses": []}
    import maxim
    OUT["maxim_file"] = maxim.__file__
    for _ in range(2):
        try:
            config = importlib.import_module("maxim.models.language.config")
            config.normalize_llm_profile("anything")
            config.list_llm_profiles()
            OUT["accesses"].append(None)
        except Exception as e:  # noqa: BLE001
            OUT["accesses"].append([c.__name__ for c in type(e).__mro__])
    print(CFG["marker"] + json.dumps(OUT), flush=True)
    """
)


def _run(tmp_path: Path, body: str, yml: str, reader: str = "") -> dict[str, Any]:
    xdg = tmp_path / "xdg"
    (xdg / "maxim").mkdir(parents=True)
    (xdg / "maxim" / "profiles.yml").write_text(yml)
    home = tmp_path / "home"
    home.mkdir()
    script = tmp_path / "child.py"
    script.write_text(body)
    cfg = {"src": str(SRC), "marker": MARKER, "profile": PROFILE, "alias": ALIAS, "hf_file": HF_FILE, "reader": reader}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    env = {
        "HOME": str(home),
        "TMPDIR": str(tmp_path),
        "XDG_CONFIG_HOME": str(xdg),
        "XDG_CACHE_HOME": str(tmp_path / "xdg-cache"),
        "MPLCONFIGDIR": str(tmp_path / "mpl"),
        "PYTHONPATH": str(SRC),  # ignored under -I; the child inserts SRC itself
        "PATH": "/usr/bin:/bin",
        "MAXIM_SKIP_REMOTE_PROBE": "1",
    }
    proc = subprocess.run(  # noqa: S603 -- our own interpreter, our own script
        [sys.executable, "-I", "-B", str(script), str(cfg_path)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_S,
    )
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(MARKER)), None)
    if line is None:
        raise ScenarioBroken(f"child did not report (rc={proc.returncode}): {proc.stderr[-3000:]}")
    out = json.loads(line[len(MARKER) :])
    if not str(out.get("maxim_file", "")).startswith(str(SRC)):
        raise ScenarioBroken(f"the child imported a maxim outside this checkout: {out.get('maxim_file')!r}")
    if out.get("broken"):
        raise ScenarioBroken(str(out.get("detail")))
    return out


def _reader_param(reader: str) -> Any:
    if reader in KNOWN_FAILING:
        mark = pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"V9 finding: {KNOWN_FAILING[reader]}")
        return pytest.param(reader, marks=mark, id=reader)
    return pytest.param(reader, id=reader)


@pytest.mark.parametrize("reader", [_reader_param(r) for r in READERS])
def test_every_public_reader_resolves_a_user_profile(tmp_path: Path, reader: str) -> None:
    """In a fresh process with one user profile and one alias in ``profiles.yml``, ``reader`` resolves both."""
    out = _run(tmp_path, _READER_BODY, PROFILES_YML, reader)
    assert out["ok"], f"{reader} did not resolve user profile {PROFILE!r} / alias {ALIAS!r}: {out['detail']}"


def test_a_malformed_profiles_file_raises_on_every_access(tmp_path: Path) -> None:
    """A malformed ``profiles.yml`` raises ``ConfigurationError`` on the first access and again on the second, so a
    swallowing caller cannot turn the first failure into a permanently silent "no user profiles" (§1A.1)."""
    out = _run(tmp_path, _MALFORMED_BODY, MALFORMED_YML)
    accesses = out["accesses"]
    assert len(accesses) == 2, accesses
    for i, mro in enumerate(accesses, start=1):
        assert mro is not None and "ConfigurationError" in mro, f"access {i} did not raise ConfigurationError: {mro}"
