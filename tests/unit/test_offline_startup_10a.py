"""#1315 item 10a: what engages the offline latch from config, and the router's in-process-only rule (notes §6).

The child-process red gates in ``tests/integration/test_offline_latch.py`` prove the router predicate end to end.
These pin the wiring around it in-process: ``llm.offline`` engages the latch first in ``cli.py::main`` and in every
``maxim.api`` verb that builds a run, auto-spawn is ruled out offline, and a backend that is not in-process is never
constructed after the latch. The conftest autouse ``_reset_offline_latch`` fixture disengages the latch after each
test.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
import types
from pathlib import Path

import pytest

from maxim import _offline
from maxim.runtime import config_loader as cl

REPO = Path(__file__).resolve().parents[2]
# The api verbs that build a run, connect a robot or diagnose: each engages first (§6 "Startup order").
API_VERBS = ("run", "imagine", "connect", "diagnose", "campaign", "benchmark", "research")


class _Sentinel(Exception):
    pass


def _write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "maxim" / "config.json"
    path.parent.mkdir(parents=True)
    path.write_text(text)
    return path


# ── engage_offline_if_configured ──────────────────────────────────────────


def test_llm_offline_true_engages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_config(tmp_path, monkeypatch, json.dumps({"_format_version": "1.5", "llm": {"offline": True}}))
    assert cl.engage_offline_if_configured() is True
    assert _offline.is_offline()


def test_a_file_that_fails_validation_still_engages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The raw key is read, not the validated config: an unknown key elsewhere does not unlatch it (fail-closed)."""
    _write_config(tmp_path, monkeypatch, json.dumps({"_format_version": "1.5", "llm": {"offline": True, "nope": 1}}))
    with pytest.raises(Exception):  # the validated load refuses this file
        cl.load_config()
    assert cl.engage_offline_if_configured() is True


@pytest.mark.parametrize(
    "llm",
    [{"offline": False}, {"offline": "true"}, {"offline": 1}, {}, None],
    ids=["false", "string", "int", "absent", "no-llm-section"],
)
def test_only_a_json_true_engages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, llm: object) -> None:
    payload: dict[str, object] = {"_format_version": "1.5"}
    if llm is not None:
        payload["llm"] = llm
    _write_config(tmp_path, monkeypatch, json.dumps(payload))
    assert cl.engage_offline_if_configured() is False
    assert not _offline.is_offline()


def test_no_config_file_engages_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert cl.engage_offline_if_configured() is False


def test_invalid_json_is_reported_not_engaged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write_config(tmp_path, monkeypatch, "{not json")
    with caplog.at_level(logging.WARNING, logger=cl.logger.name):
        assert cl.engage_offline_if_configured() is False
    assert any("llm.offline" in r.getMessage() for r in caplog.records)


def test_the_env_latch_still_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv(_offline.OFFLINE_ENV, "1")
    assert cl.engage_offline_if_configured() is True


# ── the callers engage first ──────────────────────────────────────────────


def test_cli_main_engages_before_anything_else(monkeypatch: pytest.MonkeyPatch) -> None:
    """``cli.py::main`` calls it before ``cli_signals.installed()`` and the guarded body (deletion probe: remove the
    call and the sentinel never raises)."""
    from maxim import cli, cli_signals

    monkeypatch.setattr(cl, "engage_offline_if_configured", lambda: (_ for _ in ()).throw(_Sentinel()))

    def too_early() -> None:
        raise AssertionError("cli_signals.installed() ran before the offline engage")

    monkeypatch.setattr(cli_signals, "installed", too_early)
    with pytest.raises(_Sentinel):
        cli.main(["--help"])


def _first_statement_after_docstring(fn: ast.FunctionDef) -> ast.stmt:
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]
    return body[0]


def _calls(stmt: ast.stmt) -> str | None:
    value = stmt.value if isinstance(stmt, ast.Expr | ast.Assign) else None
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        return value.func.id
    return None


@pytest.mark.parametrize("verb", API_VERBS)
def test_each_api_verb_engages_first(verb: str) -> None:
    tree = ast.parse((REPO / "src" / "maxim" / "api.py").read_text())
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    first = _calls(_first_statement_after_docstring(defs[verb]))
    if first == "_run_preflight":
        first = _calls(_first_statement_after_docstring(defs["_run_preflight"]))
    assert first == "_engage_offline", f"maxim.api.{verb} does not engage the offline latch first (got {first})"


def test_api_run_engages_before_its_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import api

    monkeypatch.setattr(api, "_engage_offline", lambda: (_ for _ in ()).throw(_Sentinel()))
    with pytest.raises(_Sentinel):
        api.run(goal="   ")  # an invalid goal would raise ConfigurationError if the checks ran first


# ── auto-spawn and the router ─────────────────────────────────────────────


def test_auto_spawn_is_ruled_out_offline() -> None:
    from maxim.runtime.lane_backends import _auto_spawn_ruled_out

    gpu = types.SimpleNamespace(has_gpu=True)
    assert _auto_spawn_ruled_out(gpu) is False
    _offline.engage("test")
    assert _auto_spawn_ruled_out(gpu) is True


def _router(providers: dict[str, dict[str, object]]):
    from maxim.models.language.config import LLMConfig
    from maxim.models.language.router import LLMRouter

    return LLMRouter(dataclasses.replace(LLMConfig(), enabled=True, providers=providers))


def test_in_process_lanes_build_offline() -> None:
    _offline.engage("test")
    _router({"local": {"type": "llama_cpp", "model": "m"}, "torch": {"type": "transformers", "model": "m"}})
    assert not _offline.ledger()


def test_no_backend_that_is_not_in_process_is_constructed_after_engage() -> None:
    """A router built before ``engage()`` still has its cloud lane configured: building that backend afterwards is
    refused, not handed an SDK client."""
    router = _router({"claude": {"type": "anthropic", "model": "claude-sonnet-4-6", "pricing_required": False}})
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        router._get_backend_for_provider("claude")
    assert router._backends.get("claude") is None
    assert [r.event for r in _offline.ledger()] == ["router[backend]"]
