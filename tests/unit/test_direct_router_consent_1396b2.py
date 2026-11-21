"""#1396 bullet 2: routers built outside ``build_primary_router``'s preflight are asked for, refused, or never asked.

Before this, the sim's shared and AUT routers, research, the robot runtime, the benchmark's paper writer and
``ExecAgent`` built ``LLMRouter(load_llm_config(...))`` with no consent check: a missing model left a router that
refuses every call (the llama.cpp backend never prompts since #1343), and a model the operator named and declined
let the run go on regardless (the dual-LLM AUT quietly took the shared model). The preflight also handed its re-walk
the very lane profile just declined (a loop variable shadowed the build's ``profile``).

Owner decisions 2026-11-13: OD1 fix the shadowing here; OD2 a named model that stays missing refuses the run, typed
(``LocalModelUnavailable``, exit 2); OD3 ExecAgent's CLI and lazy builds are in scope. Design: the scratchpad notes
approach_v2/v3 (adversarial design pass + two delta rounds).
"""

from __future__ import annotations

import ast
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "maxim"
CLI = SRC / "cli.py"
P = "b2-model"  # not an alias, so normalize_llm_profile leaves it as is


# ── fakes ─────────────────────────────────────────────────────────────────────


class _Router:
    built: list[_Router] = []

    def __init__(self, cfg: Any = None, **kw: Any) -> None:  # kw: the lane manager's build keywords
        self.cfg = cfg
        self.warmed = False
        _Router.built.append(self)

    def warmup(self) -> bool:
        self.warmed = True
        return True

    def wait_ready(self, timeout: float = 0.0) -> bool:
        return True

    def enabled(self) -> bool:
        return True


def _cfg(tmp_path: Path, profile: str = P, *, backend: str = "llama_cpp", present: bool = False) -> SimpleNamespace:
    path = tmp_path / f"{profile}.gguf"
    if present:
        path.write_text("")
    return SimpleNamespace(enabled=True, backend=backend, model_path=str(path), profile=profile, model_base="")


class _Manager:
    def __init__(self, kind: str = "local", declined: frozenset[str] = frozenset()) -> None:
        self.kind, self.declined_profiles = kind, declined
        self.unusable_models: dict[str, str] = {}  # the seam records its refusal here (#1658)

    def get_lane_kind(self, lane: str) -> str:
        return self.kind


@pytest.fixture
def gate(monkeypatch):
    """The consent gate, recorded: each call's (profile, interactive); answers ``gate.answer`` (default: declined)."""
    rec = SimpleNamespace(calls=[], answer=False)

    def ensure_available(profile, *, auto=None, interactive=None, logger=None):
        rec.calls.append((profile, interactive))
        return rec.answer

    monkeypatch.setattr("maxim.models.download.ensure_available", ensure_available)
    return rec


@pytest.fixture
def fallback(monkeypatch, tmp_path):
    """``build_primary_router`` gives no router; the fallback config is a local llama.cpp model not on disk."""
    _Router.built = []
    seen: dict[str, Any] = {"primary_calls": 0}

    def primary(**kw):
        seen["primary_calls"] += 1
        seen["primary_kw"] = kw
        return None, seen.get("manager", _Manager())

    def load(profile_override=None):
        seen["profile_override"] = profile_override
        return _cfg(tmp_path, profile_override or P)

    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", primary)
    monkeypatch.setattr("maxim.models.language.router.load_llm_config", load)
    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)
    return seen


# ── OD1: the preflight's re-walk is handed the build's profile ───────────────


def _preflight_env(monkeypatch, *, registry: set[str]):
    from maxim.runtime import lane_backends as LB

    walks: list[Any] = []

    def detect_tiers(caps, *, profile_available=None, profile=None):
        walks.append(profile)
        from maxim.runtime.worker_pool import LaneConfig

        return {"large": LaneConfig(name="large", max_workers=1, model_profile="present-model")}

    monkeypatch.setattr("maxim.runtime.lane_models.detect_tiers", detect_tiers)
    monkeypatch.setattr("maxim.models.download.llm_models", lambda: {k: {} for k in registry})
    monkeypatch.setattr("maxim.models.language.config.get_profile", lambda p: {})
    monkeypatch.setattr(LB, "_profile_has_local_file", lambda p: False)
    return LB, walks


def _lanes(profile: str):
    from maxim.runtime.worker_pool import LaneConfig

    return {"large": LaneConfig(name="large", max_workers=1, model_profile=profile)}


def test_the_rewalk_is_handed_the_builds_profile_not_the_declined_lane_profile(gate, monkeypatch) -> None:
    LB, walks = _preflight_env(monkeypatch, registry={"declined-lane", "build-model"})
    out = LB._ensure_lane_profiles_available(_lanes("declined-lane"), object(), None, profile="build-model")
    assert walks == ["build-model"]
    assert out["large"].model_profile == "present-model"


def test_an_implicit_build_whose_profile_is_declined_still_rewalks(gate, monkeypatch) -> None:
    """v3.1 G3: a build handed no profile keeps today's re-walk (no refusal)."""
    LB, walks = _preflight_env(monkeypatch, registry={"env-model"})
    out = LB._ensure_lane_profiles_available(_lanes("env-model"), object(), None)
    assert walks == [None] and out["large"].model_profile == "present-model"


def test_the_preflight_refuses_a_named_model_it_could_not_download(gate, monkeypatch) -> None:
    from maxim.models.language.types import LocalModelUnavailable

    LB, walks = _preflight_env(monkeypatch, registry={P})
    with pytest.raises(LocalModelUnavailable) as err:
        LB._ensure_lane_profiles_available(_lanes(P), object(), None, profile=P)
    assert err.value.profile == P and walks == []  # no re-walk to some other model


def test_the_preflight_reports_the_profiles_it_left_missing(gate, monkeypatch) -> None:
    LB, _ = _preflight_env(monkeypatch, registry={"env-model"})
    declined: set[str] = set()
    LB._ensure_lane_profiles_available(_lanes("env-model"), object(), None, declined_out=declined)
    assert declined == {"env-model"}


def test_the_build_normalizes_an_alias_before_the_preflight_reads_it(monkeypatch) -> None:
    """v3.1 G1: ``mistral-7b`` (the API's default) reached the preflight raw, missed the registry, and was never
    asked for or refused; the router was then built on the missing file."""
    from maxim.models.language.config import normalize_llm_profile
    from maxim.runtime import lane_backends as LB
    from maxim.runtime.capabilities import RuntimeCapabilities

    seen: dict[str, Any] = {}

    def preflight(lane_configs, caps, logger, **kw):
        seen.update(kw)
        return lane_configs

    monkeypatch.setattr(LB, "_ensure_lane_profiles_available", preflight)
    monkeypatch.setattr(LB, "_maybe_auto_spawn_server", lambda caps, lane_configs, logger, **kw: lane_configs)
    monkeypatch.setattr(LB, "_apply_local_llm_override", lambda lane_configs, logger, **kw: lane_configs)
    monkeypatch.setattr(LB.LaneBackendManager, "get_backend", lambda self, name: None)
    monkeypatch.setattr(LB, "_print_lane_banner", lambda manager: None)
    _, manager = LB.build_primary_router(RuntimeCapabilities(has_gpu=False), profile="mistral-7b", logger=None)
    assert seen["profile"] == normalize_llm_profile("mistral-7b") != "mistral-7b"
    assert manager.profile_override == seen["profile"]


def test_an_implicit_build_on_a_declined_env_profile_runs_with_no_llm(gate, monkeypatch, tmp_path) -> None:
    """Review round W3, the COMPOSED outcome: ``MAXIM_LLM_PROFILE`` set, no profile handed to the build, and its model
    declined at the preflight. The re-walk moves the lane, but ``_build_local_backend`` serves the env profile, so the
    router would load the declined file; the seam does not ask again and the run has no LLM. Not "a smaller model"
    (the residual is filed); the operator is asked once."""
    from maxim.models.download import LLM_MODELS
    from maxim.runtime import lane_backends as LB
    from maxim.runtime.capabilities import RuntimeCapabilities
    from maxim.runtime.worker_pool import LaneConfig

    env_profile = next(iter(LLM_MODELS))
    monkeypatch.setenv("MAXIM_LLM_PROFILE", env_profile)
    walks: list[str] = []

    def detect_tiers(caps, **kw):
        # The build's own walk picks the env profile; the preflight's re-walk, after the decline, a present model.
        walks.append(env_profile if not walks else "present-model")
        return {"large": LaneConfig(name="large", max_workers=1, model_profile=walks[-1])}

    monkeypatch.setattr("maxim.runtime.lane_models.detect_tiers", detect_tiers)
    monkeypatch.setattr(LB, "_profile_has_local_file", lambda p: p == "present-model")
    monkeypatch.setattr("maxim.runtime.llm_server.profile_has_local_file", lambda p: p == "present-model")
    monkeypatch.setattr(LB, "_maybe_auto_spawn_server", lambda caps, lane_configs, logger, **kw: lane_configs)
    monkeypatch.setattr(LB, "_apply_local_llm_override", lambda lane_configs, logger, **kw: lane_configs)
    monkeypatch.setattr(LB, "_print_lane_banner", lambda manager: None)
    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)
    router, manager = LB.build_run_router(RuntimeCapabilities(has_gpu=False), logger=None)
    assert walks == [env_profile, "present-model"]  # the preflight declined, then re-walked
    assert manager is not None and env_profile in manager.declined_profiles
    assert router is None  # the manager served the env profile; the seam refused it without asking again
    assert [p for p, _ in gate.calls] == [env_profile]  # asked once, by the preflight


# ── router_model_problem (moved from cli.py) ─────────────────────────────────


def test_a_profile_this_build_already_declined_is_not_asked_again(gate, tmp_path) -> None:
    from maxim.models.download import router_model_problem
    from maxim.models.language.types import LocalModelUnavailable

    assert router_model_problem(_Router(_cfg(tmp_path)), declined={P}) == LocalModelUnavailable.NOT_DOWNLOADED
    assert gate.calls == []


def test_a_custom_model_path_still_missing_after_the_registry_file_is_present_is_refused(
    gate, tmp_path, capsys, caplog
) -> None:
    """The cause is its own (review round W8/E4): the download hint would send the operator to a command that does
    not help, and the message no longer claims llm.json set a path the quantization may have built."""
    from maxim.models.download import router_model_problem
    from maxim.models.language.types import LocalModelUnavailable

    gate.answer = True  # the registry file is on disk; the router's own model_path is not
    with caplog.at_level(logging.WARNING):
        assert router_model_problem(_Router(_cfg(tmp_path))) == LocalModelUnavailable.MODEL_PATH_MISSING
    assert "model_path" in caplog.text and "--auto-download" not in caplog.text and "llm.json" not in caplog.text
    assert capsys.readouterr().out == ""  # the library check never prints (review round A4/E5)


def test_the_library_check_never_prints_a_refusal(gate, tmp_path, capsys, monkeypatch) -> None:
    """Review round A4/E5: it runs mid-run for ExecAgent's lazy build and under a live display; only a CLI caller
    prints its own line."""
    from maxim.models.download import consented_router

    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)
    consented_router(_cfg(tmp_path))
    consented_router(SimpleNamespace(enabled=True, backend="llama_cpp", model_path=str(tmp_path / "x.gguf")))
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("provider_type", ["anthropic", "maxim_peer"])
def test_a_router_with_a_non_local_provider_tail_keeps_serving_while_its_local_file_is_missing(
    gate, tmp_path, monkeypatch, provider_type
) -> None:
    """Review round E1: ``--cloud-fallback`` gives the router ``['local', 'cloud-fallback']``. The router skips a
    local provider that fails to load and serves from the tail, so the check neither asks nor drops the router (the
    run used to keep its cloud LLM; the check had made a declined local model mean no LLM at all)."""
    from maxim.models.download import consented_router

    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)
    cfg = _cfg(tmp_path)
    cfg.providers = {"local": {"type": "llama_cpp"}, "cloud-fallback": {"type": provider_type, "model": "m"}}
    router, cause = consented_router(cfg, interactive=None)
    assert isinstance(router, _Router) and cause is None
    assert gate.calls == []


def test_a_router_whose_only_provider_is_local_is_still_checked(gate, tmp_path, monkeypatch) -> None:
    from maxim.models.download import consented_router

    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)
    cfg = _cfg(tmp_path)
    cfg.providers = {"local": {"type": "llama_cpp"}, "other-local": {"backend": "llamacpp"}}
    assert consented_router(cfg, interactive=None)[0] is None
    assert gate.calls == [(P, None)]


# ── the seam: build_run_router ───────────────────────────────────────────────


def test_a_fallback_router_on_a_missing_model_is_asked_for_on_the_main_thread(gate, fallback) -> None:
    from maxim.runtime.lane_backends import build_run_router

    router, _ = build_run_router(logger=None)
    assert router is None and gate.calls == [(P, None)]
    assert not any(r.warmed for r in _Router.built)  # nothing started a load


def test_a_declined_named_model_refuses_the_run(gate, fallback) -> None:
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.runtime.lane_backends import build_run_router

    with pytest.raises(LocalModelUnavailable):
        build_run_router(logger=None, profile=P)
    assert gate.calls == [(P, None)]


def test_off_the_main_thread_the_seam_never_asks(gate, fallback) -> None:
    from maxim.runtime.lane_backends import build_run_router

    result: list[Any] = []
    t = threading.Thread(target=lambda: result.append(build_run_router(logger=None)))
    t.start()
    t.join(10)
    assert result and result[0][0] is None
    assert gate.calls == [(P, False)]


def test_a_granted_download_hands_the_router_back(gate, fallback, monkeypatch, tmp_path) -> None:
    from maxim.runtime.lane_backends import build_run_router

    def granted(profile, *, auto=None, interactive=None, logger=None):
        (tmp_path / f"{profile}.gguf").write_text("")  # the gate put the file where the router loads it
        gate.calls.append((profile, interactive))
        return True

    monkeypatch.setattr("maxim.models.download.ensure_available", granted)
    router, _ = build_run_router(logger=None)
    assert isinstance(router, _Router) and gate.calls == [(P, None)]


def test_a_fallback_profile_the_preflight_already_declined_is_not_asked_again(gate, fallback) -> None:
    """A box with no large tier (CPU-only, low VRAM) takes the fallback on every start: one prompt per profile."""
    from maxim.runtime.lane_backends import build_run_router

    fallback["manager"] = _Manager(declined=frozenset({P}))
    router, _ = build_run_router(logger=None)
    assert router is None and gate.calls == []


@pytest.mark.parametrize(
    "case",
    ["cloud-backend", "present-file"],
)
def test_a_router_with_nothing_to_download_is_handed_back_untouched(gate, monkeypatch, tmp_path, case) -> None:
    from maxim.runtime.lane_backends import build_run_router

    cfg = _cfg(tmp_path, backend="anthropic") if case == "cloud-backend" else _cfg(tmp_path, present=True)
    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", lambda **kw: (None, _Manager()))
    monkeypatch.setattr("maxim.models.language.router.load_llm_config", lambda profile_override=None: cfg)
    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)
    router, _ = build_run_router(logger=None, profile=P)
    assert isinstance(router, _Router) and gate.calls == []


def test_a_primary_router_on_a_declined_model_is_not_passed_through(gate, monkeypatch, tmp_path) -> None:
    """First design pass, fold 1: the primary router itself can carry the missing model (the manager serves the
    build's profile ahead of the lane's); it gets the same check as the fallback."""
    from maxim.runtime.lane_backends import build_run_router

    primary = _Router(_cfg(tmp_path))
    monkeypatch.setattr(
        "maxim.runtime.lane_backends.build_primary_router", lambda **kw: (primary, _Manager(declined=frozenset({P})))
    )
    router, _ = build_run_router(logger=None)
    assert router is None and gate.calls == []


def test_a_primary_router_on_a_remote_lane_is_not_checked(gate, monkeypatch, tmp_path) -> None:
    """A spawned server, a peer or a cloud lane has no local file here."""
    from maxim.runtime.lane_backends import build_run_router

    primary = _Router(_cfg(tmp_path))
    monkeypatch.setattr(
        "maxim.runtime.lane_backends.build_primary_router", lambda **kw: (primary, _Manager(kind="self-hosted"))
    )
    router, _ = build_run_router(logger=None, profile=P)
    assert router is primary and gate.calls == []


# ── the run sites ────────────────────────────────────────────────────────────


def test_the_shared_router_refuses_a_named_model_and_is_none_without_one(gate, fallback) -> None:
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.simulation.orchestrator import _build_shared_router

    with pytest.raises(LocalModelUnavailable):
        _build_shared_router(P)
    assert _build_shared_router(None) is None
    assert not any(r.warmed for r in _Router.built)


def test_a_declined_aut_model_refuses_the_run_instead_of_measuring_the_shared_model(gate, fallback) -> None:
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.simulation.orchestrator import _build_aut_router

    shared = object()
    with pytest.raises(LocalModelUnavailable):
        _build_aut_router(P, shared)
    assert not any(r.warmed for r in _Router.built)


def test_a_declined_aut_model_refuses_before_anything_warms_and_releases_the_runs_registration(monkeypatch) -> None:
    """Review round E3/W1: the AUT's refusal landed after the shared model had started loading and the sandbox was
    up (a library caller kept both). Now both routers are BUILT first; the refusal comes before any warmup, and the
    run's serving registration is released, as the stop refusal does."""
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.simulation import orchestrator

    order: list[str] = []
    shared = SimpleNamespace(warmup=lambda: order.append("shared warmup"))

    def build_run_router(*, logger=None, profile=None):
        order.append(f"build {profile}")
        if profile == "aut-model":
            raise LocalModelUnavailable("aut-model")
        return shared, None

    monkeypatch.setattr("maxim.runtime.lane_backends.build_run_router", build_run_router)
    monkeypatch.setattr(
        "maxim.simulation.report.release_start_provenance", lambda start: order.append(f"release {start['k']}")
    )
    with pytest.raises(LocalModelUnavailable):
        orchestrator._build_run_llms("shared-model", aut_model="aut-model", start_provenance={"k": "start"})
    assert order == ["build shared-model", "build aut-model", "release start"]


def test_the_run_builds_both_routers_before_the_sandbox_and_warms_the_aut_one_later(monkeypatch) -> None:
    """The order inside ``start_simulation_mode``: ``_build_run_llms`` (both builds, then the shared warmup) before
    the sandbox; the AUT's own warmup where its worker is built, so the load order is unchanged."""
    from tests.unit._orchestrator_source import start_simulation_mode_def

    fn = start_simulation_mode_def()

    def line(name: str) -> int:
        (call,) = [n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name]
        return call.lineno

    assert line("_build_run_llms") < line("_setup_sim_sandbox") < line("_warm_aut_router")
    assert not [
        n for n in ast.walk(fn) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_build_aut_router"
    ]

    from maxim.simulation import orchestrator

    order: list[str] = []
    routers = {
        p: SimpleNamespace(warmup=lambda p=p: order.append(f"warm {p}"), wait_ready=lambda timeout=0.0: True)
        for p in ("shared", "aut")
    }
    monkeypatch.setattr(
        "maxim.runtime.lane_backends.build_run_router",
        lambda *, logger=None, profile=None: (order.append(f"build {profile}") or routers[profile], None),
    )
    shared, aut = orchestrator._build_run_llms("shared", aut_model="aut", start_provenance={})
    assert order == ["build shared", "build aut", "warm shared"]
    assert orchestrator._warm_aut_router(aut, shared, "aut") is aut and order[-1] == "warm aut"


def test_research_refuses_a_named_model_past_its_broad_except(gate, fallback) -> None:
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.simulation.research_orchestrator import _build_research_llm

    with pytest.raises(LocalModelUnavailable):
        _build_research_llm(P)


def test_the_paper_writer_takes_one_plain_router_and_never_asks(gate, fallback, tmp_path, monkeypatch) -> None:
    """Second design pass, fold 4: no full build (process services, a spawned server, the banner, a decision
    record) for a post-benchmark step; a missing model is no paper, not a router that refuses every call."""
    from maxim.simulation.benchmark import BenchmarkRunner

    # ExperimentLog does not take write_paper's ``nickname=`` (a separate, pre-existing defect: the paper step fails
    # before it reaches the router); stubbed so this test reaches the router build.
    monkeypatch.setattr("maxim.simulation.research_tools.ExperimentLog", lambda **kw: SimpleNamespace(record=None))
    report = SimpleNamespace(results={}, suite="s", runs_per_model=1, models=[])
    assert BenchmarkRunner.write_paper(object.__new__(BenchmarkRunner), report, tmp_path) is None
    assert gate.calls == [(P, False)]
    assert fallback["primary_calls"] == 0


def test_consented_router_never_asks_by_default(gate, monkeypatch, tmp_path) -> None:
    from maxim.models.download import consented_router
    from maxim.models.language.types import LocalModelUnavailable

    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)
    assert consented_router(_cfg(tmp_path)) == (None, LocalModelUnavailable.NOT_IN_REGISTRY)  # P is not a registry key
    assert gate.calls == [(P, False)]
    router, cause = consented_router(_cfg(tmp_path, present=True))
    assert isinstance(router, _Router) and cause is None


def test_a_registry_model_the_gate_did_not_download_has_its_own_cause(gate, monkeypatch, tmp_path) -> None:
    from maxim.models.download import consented_router
    from maxim.models.language.types import LocalModelUnavailable

    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)
    monkeypatch.setattr("maxim.models.download.llm_models", lambda: {P: {}})
    assert consented_router(_cfg(tmp_path)) == (None, LocalModelUnavailable.NOT_DOWNLOADED)


# ── LocalModelUnavailable ────────────────────────────────────────────────────


def test_the_refusal_is_also_a_configuration_error() -> None:
    """Owner decision 2026-11-14: ``maxim.run()`` / ``create.router()`` document ``ConfigurationError``, so an
    existing ``except ConfigurationError`` still catches the refusal; the CLI still reads it as a BackendError."""
    from maxim.exceptions import ConfigurationError
    from maxim.models.language.types import BackendError, LocalModelUnavailable

    err = LocalModelUnavailable(P)
    assert isinstance(err, ConfigurationError) and isinstance(err, BackendError)
    assert err.profile == P and err.provider_key == P and err.status is None and err.response is None
    with pytest.raises(ConfigurationError):
        raise err


def test_each_cause_has_its_own_static_hint_without_the_profile() -> None:
    """Review round E4/W8: a hint per cause, static text chosen by the cause, never the user's profile in it."""
    from maxim.models.language.types import LocalModelUnavailable

    hints = {
        cause: LocalModelUnavailable("user-typed-name", cause=cause).fix_hint for cause in LocalModelUnavailable.CAUSES
    }
    assert len(set(hints.values())) == len(hints) == 3
    assert all("user-typed-name" not in h for h in hints.values())
    assert "auto_download" in hints[LocalModelUnavailable.NOT_DOWNLOADED]
    assert "maxim model add" in hints[LocalModelUnavailable.NOT_IN_REGISTRY]
    assert "llm.json" in hints[LocalModelUnavailable.MODEL_PATH_MISSING]
    with pytest.raises(ValueError):
        LocalModelUnavailable(P, cause="made-up")


def test_a_named_refusal_carries_the_checks_cause(gate, fallback, monkeypatch, tmp_path) -> None:
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.runtime.lane_backends import build_run_router

    gate.answer = True  # the registry file is on disk, the router's own model_path is not
    monkeypatch.setattr("maxim.models.download.llm_models", lambda: {P: {}})
    with pytest.raises(LocalModelUnavailable) as err:
        build_run_router(logger=None, profile=P)
    assert err.value.cause == LocalModelUnavailable.MODEL_PATH_MISSING


def _call_sites(path: Path, target: str, *, attr_ok: bool = True) -> list[tuple[str, ast.Call]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[str, ast.Call]] = []

    def visit(node: ast.AST, func: str, cls: str) -> None:
        for child in ast.iter_child_nodes(node):
            f2 = (
                (cls + "." if cls else "") + child.name
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else func
            )
            c2 = child.name if isinstance(child, ast.ClassDef) else cls
            if isinstance(child, ast.Call):
                fn = child.func
                name = fn.id if isinstance(fn, ast.Name) else (getattr(fn, "attr", None) if attr_ok else None)
                if name == target:
                    found.append((func, child))
            visit(child, f2, c2)

    visit(tree, "<module>", "")
    return found


def test_no_run_site_hand_builds_a_router_outside_the_consent_checks() -> None:
    """Caller grep (CLAUDE.md "a fix ships with a caller"): every ``LLMRouter(...)`` / ``_LlamaCppBackend(...)`` in
    src/ outside models/language/, and in scripts/, is the ONE consented builder (``models/download.py::
    consented_router``; review round A1, D43), the lane manager's own lane builds (the primary build, behind its
    preflight and the seam's check), or the mid-sim LLMAgent, which never asks by design."""
    allowed = {
        ("src/maxim/models/download.py", "consented_router", "LLMRouter"),
        ("src/maxim/runtime/lane_backends.py", "LaneBackendManager._build_local_backend", "LLMRouter"),
        ("src/maxim/runtime/lane_backends.py", "LaneBackendManager._build_remote_backend", "LLMRouter"),
        ("src/maxim/agents/llm_agent.py", "LLMAgent._get_backend", "_LlamaCppBackend"),  # S6: never asks (mid-sim)
    }
    root = SRC.parents[1]
    found = set()
    for path in sorted([*SRC.rglob("*.py"), *(root / "scripts").rglob("*.py")]):
        rel = path.relative_to(root).as_posix()
        if rel.startswith("src/maxim/models/language/"):
            continue
        for target in ("LLMRouter", "_LlamaCppBackend"):
            found |= {(rel, func, target) for func, _ in _call_sites(path, target)}
    assert found == allowed, sorted(found ^ allowed)


@pytest.mark.parametrize(
    "rel, func",
    [
        ("src/maxim/cli.py", "_cli_local_router"),
        ("src/maxim/agents/exec_agent.py", "ExecAgent._ensure_router"),
        ("src/maxim/runtime/lane_backends.py", "build_run_router"),
        ("src/maxim/simulation/benchmark.py", "BenchmarkRunner.write_paper"),
        ("scripts/exp44/rerun_ablated_offline.py", "_build_router"),
    ],
)
def test_every_plain_router_build_goes_through_the_one_consented_builder(rel, func) -> None:
    path = SRC.parents[1] / rel
    assert func in {f for f, _ in _call_sites(path, "consented_router")}, f"{rel}::{func} no longer calls it"


def test_the_robot_runtime_builds_through_the_seam() -> None:
    path = SRC / "embodied_runtime" / "agentic_runtime.py"
    assert {func for func, _ in _call_sites(path, "build_run_router")} == {"AgenticRuntimeMixin._start_agentic_runtime"}
    assert _call_sites(path, "build_primary_router") == []


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_every_cli_sim_with_an_llm_flag_hands_the_named_model_to_the_build() -> None:
    """v3 F1: ``--llm X`` reached the build only through ``MAXIM_LLM_PROFILE``, so a declined X read as an implicit
    choice. The bare-``maxim`` menu has no ``--llm``, so its sims pass none."""
    missing = [
        func
        for func, call in _call_sites(CLI, "start_simulation_mode")
        if func != "_run_menu_sim" and "llm_profile" not in {k.arg for k in call.keywords}
    ]
    assert missing == []


def test_the_goal_sim_passes_the_llm_flag_as_its_profile(monkeypatch) -> None:
    from maxim import cli

    seen: dict[str, Any] = {}
    monkeypatch.setattr("maxim.simulation.orchestrator.start_simulation_mode", lambda **kw: seen.update(kw))
    monkeypatch.setattr("maxim.simulation.arcs.select_arc_for_goal", lambda goal: None)
    monkeypatch.setattr(cli, "_simulation_result_exit_code", lambda result: 0)
    with pytest.raises(SystemExit):
        cli._run_goal_sim(
            SimpleNamespace(language_model=P),
            "g",
            debug_raw=False,
            aut_mode="llm-primary",
            research_telemetry=False,
            entity_ref=None,
            pre_campaign_turns=None,
        )
    assert seen["llm_profile"] == P


def _agentic(refused: str | None):
    exec_agent = SimpleNamespace(
        _refused_profile=refused,
        _ensure_llm=lambda: None,
        _ensure_router=lambda consent="never": None,
    )
    return SimpleNamespace(exec_agent=exec_agent)


def test_the_cli_agentic_path_refuses_when_the_llm_flags_model_is_missing() -> None:
    from maxim import cli
    from maxim.models.language.types import LocalModelUnavailable

    with pytest.raises(LocalModelUnavailable):
        cli._cli_agentic_llm_worker(
            _agentic(P),
            SimpleNamespace(language_model=P, sim=None),
            logging.getLogger("t"),
            embodiment_ref=None,
            component_registry=None,
        )


@pytest.mark.parametrize("named", [None, "another-model"], ids=["no-llm-flag", "an-exec-agent-override"])
def test_the_cli_agentic_path_runs_without_the_llm_for_a_model_nobody_named(named) -> None:
    """Second delta round, G5: an llm.json ``agent_profiles.exec_agent`` override is not the ``--llm`` model."""
    from maxim import cli

    worker = cli._cli_agentic_llm_worker(
        _agentic(P),
        SimpleNamespace(language_model=named, sim=None),
        logging.getLogger("t"),
        embodiment_ref=None,
        component_registry=None,
    )
    assert worker is None


def test_a_named_model_refusal_exits_2_with_the_fix_hint(monkeypatch) -> None:
    from maxim import cli
    from maxim.models.language.types import LocalModelUnavailable

    warned: list[tuple[str, Any]] = []

    def _raise(argv=None):
        raise LocalModelUnavailable(P)

    monkeypatch.setattr(cli, "_main_impl", _raise)
    monkeypatch.setattr("maxim.utils.logging.user_warn", lambda msg, **kw: warned.append((msg, kw.get("fix"))))
    assert cli._main_guarded([]) == 2
    assert warned and P in warned[0][0] and "auto_download" in warned[0][1]


# ── ExecAgent ────────────────────────────────────────────────────────────────


@pytest.fixture
def exec_agent(monkeypatch, tmp_path):
    from maxim.agents import exec_agent as mod

    _Router.built = []
    monkeypatch.setattr(mod, "load_llm_config", lambda profile_override=None: _cfg(tmp_path))
    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)  # built by models/download.consented_router
    agent = object.__new__(mod.ExecAgent)
    agent._router = None
    agent._no_router = False  # shared_router=None: the lazy build (#1228)
    agent._refused_profile = None
    agent._refused_cause = None
    return agent


def test_the_lazy_library_router_never_asks_and_remembers_the_refusal(gate, exec_agent) -> None:
    assert exec_agent._ensure_router() is None
    assert exec_agent._ensure_router() is None  # the lazy goal path: no rebuild, no second warning
    assert gate.calls == [(P, False)] and len(_Router.built) == 1
    assert exec_agent._refused_profile == P


def test_the_cli_ask_is_not_stopped_by_an_earlier_lazy_refusal(gate, exec_agent) -> None:
    exec_agent._ensure_router()
    assert exec_agent._ensure_router(consent="ask") is None
    assert gate.calls == [(P, False), (P, None)]


def test_an_ask_off_the_main_thread_never_prompts(gate, exec_agent) -> None:
    """Review round E2a: ``consent="ask"`` asks only on the main thread; a worker thread's build never prompts."""
    t = threading.Thread(target=lambda: exec_agent._ensure_router(consent="ask"))
    t.start()
    t.join(10)
    assert gate.calls == [(P, False)]


def test_the_cli_agentic_path_builds_exec_agents_router_with_the_ask() -> None:
    """Review round E2b: the CLI agentic path is the one ExecAgent build that may ask (OD3, the #1343 fold)."""
    from maxim import cli

    asked: list[str] = []
    exec_agent = SimpleNamespace(
        _refused_profile=None,
        _ensure_llm=lambda: None,
        _ensure_router=lambda consent="never": asked.append(consent),
    )
    cli._cli_agentic_llm_worker(
        SimpleNamespace(exec_agent=exec_agent),
        SimpleNamespace(language_model=None, sim=None),
        logging.getLogger("t"),
        embodiment_ref=None,
        component_registry=None,
    )
    assert asked == ["ask"]


def test_the_cli_refusal_names_the_exec_agents_cause() -> None:
    from maxim import cli
    from maxim.models.language.types import LocalModelUnavailable

    agentic = _agentic(P)
    agentic.exec_agent._refused_cause = LocalModelUnavailable.NOT_IN_REGISTRY
    with pytest.raises(LocalModelUnavailable) as err:
        cli._cli_agentic_llm_worker(
            agentic,
            SimpleNamespace(language_model=P, sim=None),
            logging.getLogger("t"),
            embodiment_ref=None,
            component_registry=None,
        )
    assert err.value.cause == LocalModelUnavailable.NOT_IN_REGISTRY


# ── S6: LLMAgent never asks; its refusal says how to grant ───────────────────


def test_the_mid_sim_agent_refusal_names_the_grant(monkeypatch, tmp_path, caplog) -> None:
    from maxim.models.download import LLM_MODELS
    from maxim.models.language.llama_backend import _LlamaCppBackend

    monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
    monkeypatch.setattr("maxim.runtime.llm_server.profile_has_local_file", lambda p: False)
    monkeypatch.setattr("maxim.models.download.download_llm", lambda *a, **k: pytest.fail("downloaded"))
    profile = next(iter(LLM_MODELS))
    cfg = SimpleNamespace(model_path=str(tmp_path / "x.gguf"), profile=profile, model_base="", quantization="Q4_K_M")
    with caplog.at_level(logging.WARNING):
        assert _LlamaCppBackend(cfg)._ensure() is False
    assert "--auto-download" in caplog.text and "llm.auto_download" in caplog.text
