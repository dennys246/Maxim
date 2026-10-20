"""#1343: a model download needs consent, on every path.

``models/download.py::ensure_available`` is the consent gate (``--auto-download`` / ``llm.auto_download`` /
``MAXIM_AUTO_DOWNLOAD_MODELS``, else a y/N prompt at a terminal, else a refusal). Two internal paths called
``download_llm`` directly and skipped it: the CLI agentic path, and ``_LlamaCppBackend._ensure``, the lazy load
EVERY router reaches when its local file is missing. So a multi-GB GGUF downloaded with no consent.

``download_llm`` stays public: calling it is the consent on the explicit-request paths (``maxim.download_model``,
``python -m maxim.models.download --llm``). The guard is an allow-list of its callers in ``src/``.
Design: docs/plans/download_consent_gate.md.
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


def _download_llm_callers() -> set[tuple[str, str]]:
    """(file, enclosing function) of every call to ``download_llm`` in src/, through any import alias."""
    found: set[tuple[str, str]] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = {"download_llm"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                aliases |= {a.asname for a in node.names if a.name == "download_llm" and a.asname}
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


@pytest.mark.xfail(strict=True, reason="#1343: cli.py and llama_backend.py call download_llm without the gate")
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


@pytest.mark.xfail(strict=True, reason="#1343: the backend's lazy load downloads with no consent")
def test_the_backends_lazy_load_never_downloads_without_consent(no_consent, tmp_path) -> None:
    from maxim.models.download import LLM_MODELS
    from maxim.models.language.llama_backend import _LlamaCppBackend

    profile = next(iter(LLM_MODELS))
    cfg = SimpleNamespace(
        model_path=str(tmp_path / "missing.gguf"), profile=profile, model_base=profile, quantization="Q4_K_M"
    )
    assert _LlamaCppBackend(cfg)._ensure() is False
    assert no_consent == []
