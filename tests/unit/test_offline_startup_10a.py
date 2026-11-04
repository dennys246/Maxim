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
from maxim.exceptions import ConfigurationError
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
    with pytest.raises(ConfigurationError):  # the validated load refuses this file
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


@pytest.mark.parametrize("text", ["{not json", '{"llm": {"offline": true},}', "[1, 2]", '"offline"'])
def test_a_file_that_cannot_be_parsed_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> None:
    """Whether it means offline is unknowable, so it refuses (owner decision V20), and never engages or runs online."""
    _write_config(tmp_path, monkeypatch, text)
    with pytest.raises(ConfigurationError):
        cl.engage_offline_if_configured()
    assert not _offline.is_offline()


def test_an_empty_file_is_absent_and_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _write_config(tmp_path, monkeypatch, "  \n")
    with caplog.at_level(logging.WARNING):
        assert cl.engage_offline_if_configured() is False
    assert not caplog.records


def test_cli_main_refuses_an_unparseable_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from maxim import cli

    _write_config(tmp_path, monkeypatch, "{not json")
    assert cli.main(["--help"]) == 2
    assert "invalid JSON" in capsys.readouterr().err


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


def test_api_engage_reads_the_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``api._engage_offline`` itself engages (deletion probe: an empty body fails here)."""
    from maxim import api

    _write_config(tmp_path, monkeypatch, json.dumps({"llm": {"offline": True}}))
    api._engage_offline()
    assert _offline.is_offline()


def test_build_primary_router_engages_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A library caller (``maxim.create.router()``) that never ran ``cli.main`` is latched at the router build."""
    from maxim.runtime import lane_backends

    _write_config(tmp_path, monkeypatch, json.dumps({"llm": {"offline": True}}))
    monkeypatch.setattr(lane_backends, "_require_valid_user_profiles", lambda: None)
    lane_backends._router_build_preflight()
    assert _offline.is_offline()


def test_a_peer_lane_is_refused_by_name_not_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shipped builder raises the refusal (it used to log a WARNING and return None, and callers fell back to
    another router); the URL probe neither runs nor drops the lane."""
    from maxim.runtime import lane_backends

    monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("MAXIM_LANE_LARGE_REMOTE_MODEL", "m")
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal, match="lane-large"):
        lane_backends.build_primary_router(capabilities=types.SimpleNamespace(has_gpu=False))
    assert [r.event for r in _offline.ledger()] == ["router[build]"]


def test_validate_remote_urls_keeps_the_lane_offline() -> None:
    from maxim.runtime import lane_backends

    lanes = {"large": types.SimpleNamespace(remote_url="http://127.0.0.1:9/v1", remote_api_key=None)}
    _offline.engage("test")
    assert lane_backends._validate_remote_urls(lanes, None)["large"].remote_url == "http://127.0.0.1:9/v1"


def test_router_build_and_dispatch_refuse_in_process() -> None:
    """In-process twins of the child-process red gates, so coverage sees the refusal lines."""
    remote = {"lane": {"type": "maxim_peer", "base_url": "http://127.0.0.1:9/v1", "model": "m"}}
    router = _router(remote)
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        _router(remote)
    with pytest.raises(_offline.OfflineRefusal):  # generate_json builds its backend first: that refuses
        router.generate_json("hello")
    with pytest.raises(_offline.OfflineRefusal):  # generate() no longer rewraps it as ModelInferenceError
        router.generate("hello")
    with pytest.raises(_offline.OfflineRefusal):  # in the caller's thread, not the warmup thread
        router.warmup()
    assert [r.event for r in _offline.ledger()] == [
        "router[build]",
        "router[backend]",
        "router[dispatch]",
        "router[warmup]",
    ]


def test_a_doctor_check_refused_offline_is_its_own_row() -> None:
    from maxim.doctor import checks

    def check_needs_network() -> checks.CheckResult:
        _offline.refuse("socket.getaddrinfo", "name lookup")  # what the hook raises for a check's lookup

    def check_many() -> list[checks.CheckResult]:
        _offline.refuse("test", "spawn")
        return []

    _offline.engage("test")
    row = checks._refused_offline_row(check_needs_network)()
    assert (row.status, row.name) == ("info", "needs network") and "refused (offline)" in row.message
    assert [r.status for r in checks._refused_offline_row(check_many)()] == ["info"]


def test_every_doctor_check_is_wrapped() -> None:
    import inspect

    from maxim.doctor import checks

    for name, fn in vars(checks).items():
        if name.startswith("check_") and inspect.isfunction(fn):
            assert hasattr(fn, "__wrapped__"), f"{name} is not wrapped for offline"


def test_the_local_lane_builder_raises_the_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local lane whose router refuses offline (a cloud fallback or placement tail rode in) is refused by name:
    ``_build_local_backend`` re-raises instead of logging and returning None."""
    from maxim.models.language import router as router_mod
    from maxim.runtime.lane_backends import LaneBackendManager
    from maxim.runtime.worker_pool import LaneConfig

    def refusing_router(*_a: object, **_k: object) -> None:
        _offline.refuse("router[build]", "lanes that are not in-process: claude")

    monkeypatch.setattr(router_mod, "LLMRouter", refusing_router)
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal, match="claude"):
        LaneBackendManager({})._build_local_backend(LaneConfig(name="large", max_workers=1))


def test_a_check_that_catches_the_refusal_still_reads_refused() -> None:
    """A check that absorbs the latch's refusal into its own verdict ("nothing responding") is reported as refused:
    the ledger shows the latch answered."""
    from maxim.doctor import checks

    def check_swallows() -> checks.CheckResult:
        try:
            _offline.refuse("socket.connect", "127.0.0.1:8100")
        except _offline.OfflineRefusal:
            return checks.CheckResult(name="server", status="fail", message="nothing responding")
        raise AssertionError("unreachable")

    _offline.engage("test")
    row = checks._refused_offline_row(check_swallows)()
    assert row.status == "info" and "socket.connect" in row.message


def test_doctor_resolved_config_has_an_llm_offline_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.doctor import checks

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert any(r.name == "llm.offline" for r in checks.check_resolved_config())


def test_maxim_config_set_accepts_llm_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim.runtime import config_cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    assert config_cli.run_config_subcommand(["set", "llm.offline", "true"]) == 0
    assert json.loads((tmp_path / "maxim" / "config.json").read_text())["llm"]["offline"] is True


def test_download_model_engages_first(monkeypatch: pytest.MonkeyPatch) -> None:
    from maxim import api

    monkeypatch.setattr(api, "_engage_offline", lambda: (_ for _ in ()).throw(_Sentinel()))
    with pytest.raises(_Sentinel):
        api.download_model("mistral-7b")


def test_maxim_diagnostics_engages_first(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from maxim.utils import reachy_diagnostics

    monkeypatch.setattr(cl, "engage_offline_if_configured", lambda: (_ for _ in ()).throw(ConfigurationError("bad")))
    assert reachy_diagnostics.main() == 2
    assert "bad" in capsys.readouterr().err


def test_a_config_that_is_not_utf8_refuses_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_config(tmp_path, monkeypatch, "")
    path.write_bytes(b'{"llm": {"offline": true}, "x": "\xff"}')
    with pytest.raises(ConfigurationError, match="failed to read"):
        cl.engage_offline_if_configured()


def test_the_research_llm_build_does_not_degrade_an_offline_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """It used to fall back to the heuristic Writer/Reviewer with a WARNING (delta round SF1)."""
    from maxim.runtime import lane_backends
    from maxim.simulation import research_orchestrator as ro

    def refuse(**_k: object) -> None:
        _offline.refuse("router[build]", "lanes that are not in-process: lane-large")

    monkeypatch.setattr(lane_backends, "build_primary_router", refuse)
    _offline.engage("test")
    with pytest.raises(_offline.OfflineRefusal):
        ro._build_research_llm()


def test_cli_main_reports_an_offline_refusal_and_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from maxim import cli

    monkeypatch.setattr(cli, "_main_impl", lambda _argv: _offline.refuse("router[build]", "lane-large"))
    _offline.engage("test")
    assert cli.main([]) == 2
    assert "Refusing to run" in capsys.readouterr().err


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#1315 item 10a delta SF1: flips when the orchestrator's AUT router build re-raises SECURITY_REFUSALS "
    "(after #1538 extracts simulation/orchestrator.py::_build_aut_router)",
)
def test_the_aut_router_build_does_not_fall_back_on_an_offline_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """An offline sim whose AUT model is not in-process must refuse, not fall back to the shared router: that would
    silently swap the model under test. Fails today: no ``_build_aut_router``, and the inline block falls back."""
    from maxim.runtime import lane_backends
    from maxim.simulation import orchestrator

    build = getattr(orchestrator, "_build_aut_router", None)
    assert build is not None, "simulation/orchestrator.py has no _build_aut_router yet (#1538)"

    def refuse(**_k: object) -> None:
        _offline.refuse("router[build]", "lanes that are not in-process: lane-large")

    monkeypatch.setattr(lane_backends, "build_primary_router", refuse)
    monkeypatch.setattr(orchestrator, "build_primary_router", refuse, raising=False)
    _offline.engage("test")
    shared = object()
    try:
        result = build("claude-sonnet-4-6", shared)
    except _offline.OfflineRefusal:
        return
    raise AssertionError(f"_build_aut_router fell back on an offline refusal (returned {result!r})")
