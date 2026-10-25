"""#1343: a model download needs consent, on every path.

``models/download.py::ensure_available`` is the consent gate (``--auto-download`` / ``llm.auto_download`` /
``MAXIM_AUTO_DOWNLOAD_MODELS``, else a y/N prompt at a terminal, else a refusal). Two internal paths called
``download_llm`` directly and skipped it: the CLI agentic path, and ``_LlamaCppBackend._ensure``, the lazy load
EVERY router reaches when its local file is missing. So a multi-GB GGUF downloaded with no consent.

``download_llm`` stays public: calling it is the consent on the explicit-request paths (``maxim.download_model``,
``python -m maxim.models.download --llm``). The guard is an allow-list of its callers in ``src/``.
Design: docs/plans/archive/download_consent_gate.md.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "maxim"

# The only places that may call ``download_llm``: the gate itself, and the two paths where the call IS the request.
ALLOWED_CALLERS = {
    ("models/download.py", "ensure_available"),
    ("models/download.py", "main"),
    ("api.py", "download_model"),
}


def _download_llm_callers(target: str = "download_llm") -> set[tuple[str, str]]:
    """(file, enclosing function) of every call to ``target`` in src/, through any import alias.

    The CLI agentic site is checked at its helper, ``cli.py::_ensure_agentic_model``; if the ``_main_impl`` call to
    it were deleted, the path still fails safe (the backend behind it refuses without a grant)."""
    found: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = {target}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                aliases |= {a.asname for a in node.names if a.name == target and a.asname}
        rel = str(path.relative_to(SRC))

        def visit(node: ast.AST, func: str) -> None:
            for child in ast.iter_child_nodes(node):
                name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
                if isinstance(child, ast.Call):
                    target = child.func
                    called = target.id if isinstance(target, ast.Name) else getattr(target, "attr", None)
                    if called in aliases:
                        found.add((rel, func))
                visit(child, name)

        visit(tree, "<module>")
    return found


def _download_main_callers() -> set[str]:
    """Files outside download.py that call ``maxim.models.download.main``, through any import form (AST)."""
    found: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "download.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: set[str] = set()  # names bound to the download module
        mains: set[str] = set()  # names bound to its main
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in ("maxim.models", "models"):
                modules |= {a.asname or a.name for a in node.names if a.name == "download"}
            if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("models.download"):
                mains |= {a.asname or a.name for a in node.names if a.name == "main"}
            if isinstance(node, ast.Import):
                modules |= {a.asname for a in node.names if a.name.endswith("models.download") and a.asname}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Name) and f.id in mains:
                found.add(str(path.relative_to(SRC)))
            if (
                isinstance(f, ast.Attribute)
                and f.attr == "main"
                and isinstance(f.value, ast.Name)
                and f.value.id in modules
            ):
                found.add(str(path.relative_to(SRC)))
    return found


def test_only_the_gate_and_the_explicit_requests_call_download_llm() -> None:
    callers = _download_llm_callers()
    assert ("models/download.py", "ensure_available") in callers  # the scan still sees the gate's own call
    assert callers <= ALLOWED_CALLERS, sorted(callers - ALLOWED_CALLERS)


@pytest.fixture
def no_consent(monkeypatch, tmp_path):
    """A missing GGUF for a registered profile, no auto-download flag, stdin not a terminal."""
    from maxim.utils import storage

    monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
    (tmp_path / "home" / "util").mkdir(parents=True)
    storage._reset_cache_for_tests()
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    calls: list[str] = []
    monkeypatch.setattr("maxim.models.download.download_llm", lambda name, *a, **k: calls.append(name) or True)
    return calls


def test_the_backends_lazy_load_never_downloads_without_consent(no_consent, tmp_path) -> None:
    from maxim.models.download import LLM_MODELS
    from maxim.models.language.llama_backend import _LlamaCppBackend

    profile = next(iter(LLM_MODELS))
    cfg = SimpleNamespace(
        model_path=str(tmp_path / "missing.gguf"), profile=profile, model_base=profile, quantization="Q4_K_M"
    )
    assert _LlamaCppBackend(cfg)._ensure() is False
    assert no_consent == []


def test_no_internal_path_reaches_the_same_download_another_way() -> None:
    """The design pass's widening: ``download_file`` is the same download one layer down, and calling the explicit
    request paths from inside src/ would be an auto-download wearing the user's consent."""
    assert {f for f, _ in _download_llm_callers("download_file")} <= {"models/download.py"}
    assert _download_llm_callers("download_model") == set()
    # ``python -m maxim.models.download`` is the CLI's own entry; no other module may run its ``main`` in-process
    assert _download_main_callers() == set()


def test_the_scanner_resolves_an_import_alias(tmp_path, monkeypatch) -> None:
    import tests.unit.test_download_consent_1343 as me

    (tmp_path / "x.py").write_text("from maxim.models.download import download_llm as dl\ndef f():\n    dl('m')\n")
    monkeypatch.setattr(me, "SRC", tmp_path)
    assert me._download_llm_callers() == {("x.py", "f")}


def _backend(tmp_path, profile=None):
    from maxim.models.download import LLM_MODELS
    from maxim.models.language.llama_backend import _LlamaCppBackend

    profile = profile or next(iter(LLM_MODELS))
    cfg = SimpleNamespace(
        model_path=str(tmp_path / "missing.gguf"), profile=profile, model_base=profile, quantization="Q4_K_M"
    )
    return _LlamaCppBackend(cfg)


def test_the_backend_never_prompts_even_at_a_terminal(no_consent, monkeypatch, tmp_path) -> None:
    """The lazy load runs on the warmup thread and in every complete(): a prompt there could take the user's reply to
    the agent as consent (design pass, SHOULD-FIX 1)."""
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    asked: list[str] = []
    monkeypatch.setattr("maxim.models.download._prompt_yes_no_with_timeout", lambda q, **k: asked.append(q) or True)
    assert _backend(tmp_path)._ensure() is False
    assert asked == [] and no_consent == []


def test_a_refusal_is_remembered_so_later_calls_fail_fast(no_consent, monkeypatch, tmp_path) -> None:
    calls: list[str] = []
    monkeypatch.setattr("maxim.models.download.ensure_available", lambda p, **k: calls.append(p) or False)
    backend = _backend(tmp_path)
    assert backend._ensure() is False and backend._ensure() is False
    assert len(calls) == 1


def test_the_backend_asks_the_gate_non_interactively(no_consent, monkeypatch, tmp_path) -> None:
    seen: dict = {}

    def _gate(profile, **kwargs):
        seen.update(kwargs, profile=profile)
        return False  # stop before the real load

    monkeypatch.setattr("maxim.models.download.ensure_available", _gate)
    _backend(tmp_path)._ensure()
    assert seen["interactive"] is False and seen["profile"]


def test_the_backend_downloads_with_a_standing_grant(no_consent, monkeypatch, tmp_path) -> None:
    """With the operator's grant the backend's own lazy load still downloads, through the gate (no regression)."""
    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")
    monkeypatch.setattr("maxim.runtime.llm_server.profile_has_local_file", lambda p: False)
    backend = _backend(tmp_path)
    backend._ensure()  # the fake download writes nothing, so the load itself fails afterwards
    assert no_consent == [backend.cfg.profile]


def test_the_cli_agentic_path_asks_the_gate_and_never_downloads_without_consent(no_consent, tmp_path) -> None:
    from maxim.cli import _ensure_agentic_model
    from maxim.models.download import LLM_MODELS

    profile = next(iter(LLM_MODELS))
    router = SimpleNamespace(cfg=SimpleNamespace(model_path=str(tmp_path / "missing.gguf"), profile=profile))
    _ensure_agentic_model(router)
    assert no_consent == []


def test_the_cli_agentic_path_downloads_with_the_operators_grant(no_consent, monkeypatch, tmp_path) -> None:
    from maxim.cli import _ensure_agentic_model
    from maxim.models.download import LLM_MODELS

    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")
    monkeypatch.setattr("maxim.runtime.llm_server.profile_has_local_file", lambda p: False)
    profile = next(iter(LLM_MODELS))
    _ensure_agentic_model(SimpleNamespace(cfg=SimpleNamespace(model_path=str(tmp_path / "x.gguf"), profile=profile)))
    assert no_consent == [profile]


def test_the_cli_agentic_path_leaves_a_present_model_alone(no_consent, monkeypatch, tmp_path) -> None:
    """Even with a standing grant and a registry file missing, a model at the router's own path is used as is."""
    from maxim.cli import _ensure_agentic_model
    from maxim.models.download import LLM_MODELS

    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")
    monkeypatch.setattr("maxim.runtime.llm_server.profile_has_local_file", lambda p: False)
    present = tmp_path / "m.gguf"
    present.write_text("")
    profile = next(iter(LLM_MODELS))
    _ensure_agentic_model(SimpleNamespace(cfg=SimpleNamespace(model_path=str(present), profile=profile)))
    assert no_consent == []


def test_the_main_scanner_sees_each_import_form(tmp_path, monkeypatch) -> None:
    import tests.unit.test_download_consent_1343 as me

    (tmp_path / "a.py").write_text("from maxim.models import download as d\nd.main()\n")
    (tmp_path / "b.py").write_text("from maxim.models.download import main as m\nm()\n")
    (tmp_path / "c.py").write_text("def main():\n    pass\nmain()\n")  # an unrelated main
    monkeypatch.setattr(me, "SRC", tmp_path)
    assert me._download_main_callers() == {"a.py", "b.py"}


# ── Review-round folds ─────────────────────────────────────────────────────────────────────────


def test_the_gate_honours_a_config_only_grant(monkeypatch) -> None:
    """``llm.auto_download`` reached the gate only through build_primary_router's env copy (Executor + Wire)."""
    from maxim.models import download

    monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
    monkeypatch.setattr("maxim.runtime.config_loader.resolve_setting", lambda field, *a, **k: (True, "config"))
    assert download._auto_download_enabled() is True


def test_the_env_wins_over_config(monkeypatch) -> None:
    from maxim.models import download

    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "0")
    monkeypatch.setattr("maxim.runtime.config_loader.resolve_setting", lambda field, *a, **k: (True, "config"))
    assert download._auto_download_enabled() is False


def test_an_unreadable_config_grants_nothing(monkeypatch) -> None:
    """A real broken config.json (not a mocked raise): resolve_setting falls back to the default, False."""
    from maxim.models import download
    from maxim.runtime import config_loader

    monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
    path = config_loader.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    config_loader.reset_config_cache()
    try:
        assert download._auto_download_enabled() is False
    finally:
        path.unlink()
        config_loader.reset_config_cache()


def test_a_real_config_grant_is_honoured(monkeypatch) -> None:
    """An operator's config.json grant, read from disk (a hand-written, unversioned file is read by presence; the
    config-writer API is CI-allow-listed to its own tests)."""
    from maxim.models import download
    from maxim.runtime import config_loader

    monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
    path = config_loader.config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"llm": {"auto_download": true}}')
    config_loader.reset_config_cache()
    try:
        assert download._auto_download_enabled() is True
    finally:
        config_loader.config_path().unlink()
        config_loader.reset_config_cache()


def test_a_caller_that_never_asks_is_told_so_on_its_logger(no_consent, monkeypatch, capsys) -> None:
    """Not "stdin is not a tty" (it may be one), and not on stdout from a worker thread."""
    import logging

    from maxim.models import download

    monkeypatch.setattr("maxim.runtime.llm_server.profile_has_local_file", lambda p: False)
    records: list[str] = []
    log = logging.getLogger("test.1343")
    monkeypatch.setattr(log, "warning", lambda msg, *a: records.append(msg))
    profile = next(iter(download.LLM_MODELS))
    assert download.ensure_available(profile, interactive=False, logger=log) is False
    assert records and "does not ask" in records[0] and "llm.auto_download" in records[0]
    assert "not a tty" not in capsys.readouterr().out


def test_an_exception_is_reported_and_not_remembered(no_consent, monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    def _raises(profile, **kwargs):
        calls.append(profile)
        raise OSError("disk")

    monkeypatch.setattr("maxim.models.download.ensure_available", _raises)
    backend = _backend(tmp_path)
    assert backend._ensure() is False and backend._ensure() is False
    assert len(calls) == 2


def test_a_backend_without_a_profile_refuses_without_asking(no_consent, monkeypatch, tmp_path) -> None:
    from maxim.models.language.llama_backend import _LlamaCppBackend

    asked: list[str] = []
    monkeypatch.setattr("maxim.models.download.ensure_available", lambda p, **k: asked.append(p) or True)
    cfg = SimpleNamespace(model_path=str(tmp_path / "x.gguf"), profile="", model_base="", quantization="Q4_K_M")
    assert _LlamaCppBackend(cfg)._ensure() is False
    assert asked == []


def test_the_cli_path_without_a_profile_says_where_to_look(no_consent, tmp_path, capsys) -> None:
    from maxim.cli import _ensure_agentic_model

    _ensure_agentic_model(SimpleNamespace(cfg=SimpleNamespace(model_path=str(tmp_path / "x.gguf"), profile="")))
    assert "--list-models" in capsys.readouterr().out and no_consent == []
