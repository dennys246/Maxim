"""V9 / step 0a guard (#1315 §1A.1): every public profile reader resolves a user ``profiles.yml`` profile and alias.

WHAT THIS FILE IS. A guard: ``models/language/config.py`` merges ``$XDG_CONFIG_HOME/maxim/profiles.yml`` into
its profile and alias tables and ``models/download.py``'s download table on FIRST USE, through
``_ensure_user_profiles()``, never at import (step 0a, ``docs/plans/atonia_gate_approach_notes.md`` §1A.1: "Red gate,
by outcome rather than by enumeration"). A reader that read the raw tables instead of an ensuring accessor would see
no user profile and fail here. It is parametrised over ``READERS``: a new public reader joins that tuple and gets a
branch in ``_READER_BODY``.

HOW. Each reader runs in a fresh ``sys.executable -I`` child whose ``XDG_CONFIG_HOME`` holds one user profile
(``PROFILE``) with one alias (``ALIAS``). A malformed ``profiles.yml`` must raise ``ConfigurationError`` on the first
access AND on the second (§1A.1: "Failure is never cached").

A reader that fails is a real finding, and is its own def with a literal strict-xfail decorator (lint rule 3b):
``test_doctor_vram_projection_resolves_an_alias`` (#1479, not 0a's; driven with a built-in alias).
``test_download_llm_models_holds_a_user_profile`` and the lazy-init pins were red gates that flipped with step 0a.
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
    "ensure_available lookup",
    "model swap validator (lane_backends._validate_swap_profile)",
    "router-build placement tail (lane_backends._placement_entry_to_provider, LOCAL)",
    "doctor VRAM projection (name)",
)

# Built-in alias for the #1479 gate: it maps to a built-in profile that carries arch metadata.
BUILTIN_ALIAS = "mistral"


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
            table = download.llm_models()
            OUT["ok"] = NAME in table
        elif reader == "ensure_available lookup":
            from unittest.mock import MagicMock
            from maxim.models import download
            log = MagicMock()
            download.ensure_available(NAME, auto=False, interactive=False, logger=log)
            said = [str(c) for c in log.method_calls]
            OUT["detail"] = said
            # Positive: the profile was FOUND in the registry and reported as not downloaded (no consent here).
            OUT["ok"] = any(f"Profile '{NAME}' is not downloaded" in c for c in said)
        elif reader == "router-build placement tail (lane_backends._placement_entry_to_provider, LOCAL)":
            # build_primary_router compiles the placement TAIL (fallback entries) through this mapper; the primary LOCAL
            # lane goes through load_llm_config(profile_override=...), the build_model_path reader above.
            from maxim.runtime.lane_backends import _placement_entry_to_provider
            from maxim.runtime.worker_pool import Origin, ProviderPlacement
            prov, is_cloud = _placement_entry_to_provider(ProviderPlacement(origin=Origin.LOCAL, model=ALIAS), "local")
            OUT["detail"] = (prov, is_cloud)
            OUT["ok"] = (prov is not None and not is_cloud and prov.get("model") == NAME
                         and os.path.basename(prov.get("model_path") or "") == HF_FILE)
        elif reader == "model swap validator (lane_backends._validate_swap_profile)":
            from maxim.runtime.lane_backends import _validate_swap_profile
            resolved, data = _validate_swap_profile(ALIAS)
            OUT["detail"] = (resolved, data.get("backend"))
            OUT["ok"] = resolved == NAME and data.get("backend") == "llama_cpp"
        elif reader.startswith("doctor VRAM projection"):
            from unittest.mock import patch
            os.environ["MAXIM_LLM_PROFILE"] = CFG.get("doctor_profile") or NAME
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


def _run(tmp_path: Path, body: str, yml: str, reader: str = "", **cfg_extra: Any) -> dict[str, Any]:
    xdg = tmp_path / "xdg"
    (xdg / "maxim").mkdir(parents=True)
    (xdg / "maxim" / "profiles.yml").write_text(yml)
    home = tmp_path / "home"
    home.mkdir()
    script = tmp_path / "child.py"
    script.write_text(body)
    cfg = {
        "src": str(SRC),
        "marker": MARKER,
        "profile": PROFILE,
        "alias": ALIAS,
        "hf_file": HF_FILE,
        "reader": reader,
        **cfg_extra,
    }
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


@pytest.mark.parametrize("reader", READERS)
def test_every_public_reader_resolves_a_user_profile(tmp_path: Path, reader: str) -> None:
    """In a fresh process with one user profile and one alias in ``profiles.yml``, ``reader`` resolves both."""
    out = _run(tmp_path, _READER_BODY, PROFILES_YML, reader)
    assert out["ok"], f"{reader} did not resolve user profile {PROFILE!r} / alias {ALIAS!r}: {out['detail']}"


def test_download_llm_models_holds_a_user_profile(tmp_path: Path) -> None:
    """``models.download``'s profile table (``llm_models()``, step 0a's ensuring accessor), read by a fresh
    process that imported nothing else, holds the user profile."""
    out = _run(tmp_path, _READER_BODY, PROFILES_YML, "models.download LLM_MODELS")
    assert out["ok"], f"models.download's table does not hold user profile {PROFILE!r}"


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "#1479: doctor/checks.py::_check_vram_context_pressure looks llm.profile up without normalising it, so an "
        "alias (built-in or user) gets 'VRAM projection unavailable'; step 0a does not fix it"
    ),
)
def test_doctor_vram_projection_resolves_an_alias(tmp_path: Path) -> None:
    """With ``llm.profile`` set to the built-in alias ``mistral``, the doctor's VRAM row projects (#1479)."""
    out = _run(tmp_path, _READER_BODY, PROFILES_YML, "doctor VRAM projection (alias)", doctor_profile=BUILTIN_ALIAS)
    assert out["ok"], f"the doctor's VRAM projection did not resolve alias {BUILTIN_ALIAS!r}: {out['detail']}"


def test_a_malformed_profiles_file_raises_on_every_access(tmp_path: Path) -> None:
    """A malformed ``profiles.yml`` raises ``ConfigurationError`` on the first access and again on the second, so a
    swallowing caller cannot turn the first failure into a permanently silent "no user profiles" (§1A.1)."""
    out = _run(tmp_path, _MALFORMED_BODY, MALFORMED_YML)
    accesses = out["accesses"]
    assert len(accesses) == 2, accesses
    for i, mro in enumerate(accesses, start=1):
        assert mro is not None and "ConfigurationError" in mro, f"access {i} did not raise ConfigurationError: {mro}"


# ── step 0a: the lazy package inits (V9) ──────────────────────────────────

LAZY_PACKAGES: tuple[str, ...] = ("maxim.runtime", "maxim.agents", "maxim.tools", "maxim.utils")


@pytest.mark.parametrize("package", LAZY_PACKAGES)
def test_step0a_package_init_is_lazy(package: str) -> None:
    """``package`` defines a module-level PEP 562 ``__getattr__`` and a lazy table ``_LAZY`` (name -> defining
    module, the seam chosen here), and ``__all__`` is exactly the lazy table's names plus the names the init still
    binds eagerly (§1A.1: "A test pins ``__all__`` against the lazy table")."""
    import importlib

    mod = importlib.import_module(package)
    assert callable(vars(mod).get("__getattr__")), f"{package} defines no module-level __getattr__ (PEP 562)"
    lazy = vars(mod).get("_LAZY")
    assert isinstance(lazy, dict) and lazy, f"{package} has no lazy table _LAZY"
    exported = set(getattr(mod, "__all__", ()))
    eager = {n for n in exported if n in vars(mod) and n not in lazy}
    assert exported == set(lazy) | eager, (
        f"{package}.__all__ does not match its lazy table: only in __all__ "
        f"{sorted(exported - set(lazy) - eager)}, only in _LAZY {sorted(set(lazy) - exported)}"
    )


# ── step 0a: no reader bypasses the ensuring accessors ─────────────────────

# Each private table -> the one module that may name it. Anywhere else, a read would see the tables before the
# user's profiles.yml is merged (the merge runs on first use of an accessor, never at import), so a user profile
# would silently disappear from that reader (§1A.1: "the grep, not the plan's count, forces every reader to migrate").
_CONFIG = "src/maxim/models/language/config.py"
PRIVATE_TABLE_OWNERS: dict[str, frozenset[str]] = {
    "_BUILTIN_PROFILES": frozenset({_CONFIG}),
    "_PROFILE_ALIASES": frozenset({_CONFIG}),
    "_normalize_profile": frozenset({_CONFIG}),
    # config.py's ``_ensure_user_profiles`` IS the merge into the raw table (``llm_models()`` there would recurse).
    "LLM_MODELS": frozenset({"src/maxim/models/download.py", _CONFIG}),
}


def _private_table_uses(path: Path) -> list[tuple[int, str]]:
    """Every import, name or attribute reference to a private table in ``path`` (an AST scan: docstrings, comments
    and string literals that mention a name are not reads)."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            hits += [(node.lineno, a.name) for a in node.names if a.name in PRIVATE_TABLE_OWNERS]
        elif isinstance(node, ast.Name) and node.id in PRIVATE_TABLE_OWNERS:
            hits.append((node.lineno, node.id))
        elif isinstance(node, ast.Attribute) and node.attr in PRIVATE_TABLE_OWNERS:
            hits.append((node.lineno, node.attr))
    return hits


def test_no_reader_outside_its_owner_names_a_private_profile_table() -> None:
    """``_BUILTIN_PROFILES``, ``_PROFILE_ALIASES`` and ``_normalize_profile`` are named only in ``config.py``, and
    ``LLM_MODELS`` only in ``download.py`` (and in ``config.py``'s merge), across ``src/`` and ``scripts/``: every other reader uses ``get_profile``,
    ``iter_profiles``, ``profile_aliases``, ``normalize_llm_profile`` or ``download.llm_models()``. Tests may patch
    the tables directly."""
    offenders: list[str] = []
    for root in (REPO / "src", REPO / "scripts"):
        for path in sorted(root.rglob("*.py")):
            rel = path.relative_to(REPO).as_posix()
            for lineno, name in _private_table_uses(path):
                if rel not in PRIVATE_TABLE_OWNERS[name]:
                    offenders.append(f"{rel}:{lineno}: {name}")
    assert not offenders, "read through an ensuring accessor instead:\n  " + "\n  ".join(offenders)


def test_the_private_table_scan_sees_each_kind_of_reference(tmp_path: Path) -> None:
    """The scan above is not vacuous: it reports an import, a bare name and an attribute read, and ignores a
    docstring that only mentions the name."""
    probe = tmp_path / "probe.py"
    probe.write_text(
        '"""Mentions _BUILTIN_PROFILES in prose only."""\n'
        "from maxim.models.language.config import _PROFILE_ALIASES\n"
        "import maxim.models.download as d\n"
        "x = _PROFILE_ALIASES\n"
        "y = d.LLM_MODELS\n"
    )
    assert sorted(_private_table_uses(probe)) == [(2, "_PROFILE_ALIASES"), (4, "_PROFILE_ALIASES"), (5, "LLM_MODELS")]
