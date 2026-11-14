"""#1657: ``--auto-curate``'s router is waited for and unloaded; #1659: the swap check reads the router's spellings.

Before #1657 the curation router was warmed but never waited for (early generations could fall to templates while the
model loaded), and it was dropped without being unloaded, so an in-process llama.cpp model stayed resident while the
sim loaded its own: the model was held twice. ``LLMRouter`` had no release method at all, so foundry's
``hasattr(router, "shutdown")`` timeout path was a silent no-op. ``LLMRouter.unload()`` now releases every backend the
router built, and ``cli.py::_auto_curate_before_sim`` owns the curation router's lifetime.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from tests.unit import test_foundry_curate_consent_1396 as consent

_run_foundry = consent._run_foundry
_run_real_curate = consent._run_real_curate
fake_router = consent.fake_router  # the shared fixture, re-exported so pytest finds it here


class _Backend:
    def __init__(self, log: list, name: str, *, fail: bool = False) -> None:
        self.log, self.name, self.fail = log, name, fail

    def complete(self, prompt, **kw):  # the router's interface check
        return ""

    def unload(self) -> None:
        self.log.append(self.name)
        if self.fail:
            raise RuntimeError(f"{self.name} would not unload")


def _router():
    from maxim.models.language.config import LLMConfig
    from maxim.models.language.router import LLMRouter

    cfg = dataclasses.replace(
        LLMConfig(),
        enabled=True,
        providers={"local": {"type": "llama_cpp", "model": "missing.gguf", "pricing_required": False}},
    )
    return LLMRouter(cfg)


# ── LLMRouter.unload ──────────────────────────────────────────────────────────


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_unload_releases_every_built_backend_once() -> None:
    """The default backend is also in the per-provider cache: it is released once, and every other backend too."""
    log: list[str] = []
    router = _router()
    default, cloud = _Backend(log, "default"), _Backend(log, "cloud")
    router._backend = default
    router._backends.update({"local": default, "cloud": cloud})
    router.unload()
    assert sorted(log) == ["cloud", "default"]
    router.unload()
    assert sorted(log) == ["cloud", "default"]  # a second call releases nothing again


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_a_backend_that_fails_to_unload_is_logged_and_the_others_are_still_released(caplog) -> None:
    log: list[str] = []
    router = _router()
    router._backends.update({"a": _Backend(log, "a", fail=True), "b": _Backend(log, "b")})
    with caplog.at_level(logging.WARNING):
        router.unload()
    assert sorted(log) == ["a", "b"]
    assert "would not unload" in caplog.text


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_an_unloaded_router_builds_no_backend_again(monkeypatch) -> None:
    """After unload, a late warmup (or any call) builds nothing, so a model is not loaded behind its owner's back."""
    from maxim.models.language import router as router_mod

    constructed: list[object] = []
    monkeypatch.setattr(router_mod, "_LlamaCppBackend", lambda cfg: constructed.append(cfg))
    router = _router()
    router.unload()
    assert router._get_backend() is None
    assert router._get_backend_for_provider("local") is None
    assert router.warmup() is False
    assert router.wait_ready(timeout=5.0) is False
    assert router._backends == {}
    assert constructed == []  # not even a lazy backend is constructed (review delta NIT)


class _SlowBackend(_Backend):
    """A backend whose warmup blocks until the test lets the load finish (or makes it raise)."""

    def __init__(self, log: list, *, raises: bool = False) -> None:
        super().__init__(log, "slow")
        self.started, self.finish, self.raises = threading.Event(), threading.Event(), raises

    def warmup(self) -> bool:
        self.started.set()
        assert self.finish.wait(10.0)
        if self.raises:
            raise RuntimeError("load blew up")
        return True


def _warming(router, backend):
    """Start a warmup on ``router`` whose default backend is ``backend``; return once its load is in flight."""
    router._get_backend_for_provider = lambda key: backend
    assert router.warmup() is True
    assert backend.started.wait(5.0)


@pytest.mark.parametrize("raises", [False, True], ids=["load-ends", "load-raises"])
@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_unload_never_waits_on_a_load_in_flight_and_the_warmup_releases_it(raises) -> None:
    """Review round (Executor, Architecture, Wire integrity): a timed-out load made unload() block for the rest of
    the load. Now it returns at once, and the warmup releases the backend when its load ends or fails."""
    log: list[str] = []
    router = _router()
    slow = _SlowBackend(log, raises=raises)
    _warming(router, slow)
    started = time.monotonic()
    router.unload()
    assert time.monotonic() - started < 1.0
    assert log == []  # not released under the load
    slow.finish.set()
    assert router._ready_event.wait(5.0)
    assert log == ["slow"]
    assert router.wait_ready(timeout=0.1) is False and router.is_ready is False


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_a_warmup_that_raises_is_reported_failed_at_once() -> None:
    """A raising warmup used to kill its thread before ``_ready_event`` was set, so ``wait_ready`` sat out its whole
    timeout; now the failure is reported as soon as the load fails."""
    router = _router()
    slow = _SlowBackend([], raises=True)
    _warming(router, slow)
    slow.finish.set()
    started = time.monotonic()
    assert router.wait_ready(timeout=5.0) is False
    assert time.monotonic() - started < 2.0


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_an_unloaded_router_is_not_ready_even_after_a_good_warmup() -> None:
    router = _router()
    slow = _SlowBackend([])
    _warming(router, slow)
    slow.finish.set()
    assert router.wait_ready(timeout=5.0) is True
    router.unload()
    assert router.is_ready is False and router.wait_ready(timeout=0.1) is False


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_a_backend_built_while_unload_runs_is_released_not_cached(monkeypatch) -> None:
    """A call outside ``_backend_lock`` can build a backend while ``unload()`` clears the cache: it is released once
    and never cached."""
    from maxim.models.language import router as router_mod

    log: list[str] = []
    router = _router()

    def build_during_unload(cfg):
        router.unload()  # unload lands while this backend is being built
        return _Backend(log, "late")

    monkeypatch.setattr(router_mod, "_LlamaCppBackend", build_during_unload)
    assert router._get_backend_for_provider("local") is None
    assert log == ["late"] and router._backends == {}


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_a_backend_without_unload_is_skipped() -> None:
    router = _router()
    router._backends["bare"] = SimpleNamespace(complete=lambda *a, **k: "")
    router.unload()  # nothing to call, nothing raised
    assert router._backends == {}


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_runtime_shutdown_now_releases_the_lane_managers_routers() -> None:
    """``LaneBackendManager.unload_all`` (agentic runtime shutdown, "releases VRAM immediately") calls ``unload`` on
    the routers it caches; before ``LLMRouter.unload`` existed it skipped every real router."""
    from maxim.runtime.lane_backends import LaneBackendManager

    log: list[str] = []
    router = _router()
    router._backends["local"] = _Backend(log, "lane-model")
    mgr = object.__new__(LaneBackendManager)
    mgr._lock, mgr._backends, mgr._cloud_lanes = threading.RLock(), {"medium": router}, set()
    mgr.unload_all()
    assert log == ["lane-model"] and mgr._backends == {}


# ── --auto-curate owns its router ─────────────────────────────────────────────


@pytest.fixture
def unloads(fake_router, monkeypatch):
    """The fake router counts unloads; a gap's download is granted, so the router is built and warmed."""
    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")
    calls: list[object] = []
    from maxim.models.language import router as router_mod

    monkeypatch.setattr(router_mod.LLMRouter, "unload", lambda self: calls.append(self), raising=False)
    return calls


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_a_gap_run_unloads_the_curation_router_once_it_ends(fake_router, unloads, monkeypatch) -> None:
    runners = _run_real_curate(monkeypatch, fake_router.profile, gaps={"weapons": 2, "armor": 1})
    assert runners == [fake_router.built[0]] * 2
    assert unloads == [fake_router.built[0]]


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_the_curation_router_is_unloaded_when_curation_raises(fake_router, unloads, monkeypatch, caplog) -> None:
    from maxim.simulation import foundry

    def boom(**kw):
        raise RuntimeError("runner exploded")

    class Runner:
        def __init__(self, **kw):
            pass

        run = staticmethod(lambda count: boom())

    monkeypatch.setattr(foundry, "FoundryRunner", Runner)
    monkeypatch.setattr(foundry, "analyze_coverage", lambda genre, **k: {"weapons": 2})
    monkeypatch.setattr("maxim.embodiment.component_registry.ComponentRegistry", lambda: object())
    monkeypatch.setattr("maxim.embodiment.component_index.ComponentIndex", lambda registry: None)
    from maxim import cli

    with caplog.at_level(logging.WARNING):
        cli._auto_curate_before_sim(
            SimpleNamespace(auto_curate=True, no_curate=False, language_model=fake_router.profile), "bodies/x"
        )
    assert unloads == [fake_router.built[0]]
    assert "Auto-curation failed" in caplog.text


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_a_router_not_ready_in_time_is_not_handed_to_the_runners_and_is_unloaded(
    fake_router,
    unloads,
    monkeypatch,
    capsys,
) -> None:
    from maxim.models.language import router as router_mod

    waits: list[float] = []

    def not_ready(self, timeout=0.0):
        waits.append(timeout)
        return False

    monkeypatch.setattr(router_mod.LLMRouter, "wait_ready", not_ready)
    runners = _run_real_curate(monkeypatch, fake_router.profile, gaps={"weapons": 2})
    assert waits, "the curation router is waited for, as foundry's is"
    assert runners == [None]
    assert unloads == [fake_router.built[0]]
    assert "curation will use template fallback" in capsys.readouterr().out


def test_full_coverage_and_a_declined_gap_unload_nothing(fake_router, monkeypatch) -> None:
    """Nothing warmed, nothing to release: full coverage builds no router, and a declined model hands back None."""
    calls: list[object] = []
    from maxim.models.language import router as router_mod

    monkeypatch.setattr(router_mod.LLMRouter, "unload", lambda self: calls.append(self), raising=False)
    _run_real_curate(monkeypatch, fake_router.profile, gaps={})
    _run_real_curate(monkeypatch, fake_router.profile, gaps={"weapons": 2})  # no grant: declined
    assert calls == []


@pytest.mark.xfail(strict=True, reason="#1657: flips with LLMRouter.unload and the CLI owning the curation router")
def test_foundry_unloads_a_router_that_did_not_load_in_time(fake_router, unloads, monkeypatch) -> None:
    """Foundry's timeout path called ``shutdown()`` behind a hasattr check LLMRouter never answered."""
    from maxim.models.language import router as router_mod

    monkeypatch.setattr(router_mod.LLMRouter, "wait_ready", lambda self, timeout=0.0: False)
    handed = _run_foundry(monkeypatch, fake_router.profile)
    assert handed == [None]
    assert unloads == [fake_router.built[0]]


# ── #1659: one spelling table ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "spelling",
    [
        "llama_cpp",
        pytest.param(
            "Llama-CPP", marks=pytest.mark.xfail(strict=True, reason="#1659: flips with the router's spelling table")
        ),
        pytest.param(
            " LLAMACPP ", marks=pytest.mark.xfail(strict=True, reason="#1659: flips with the router's spelling table")
        ),
        "llama",
    ],
)
def test_the_swap_check_accepts_every_spelling_the_router_builds_as_llama_cpp(monkeypatch, spelling) -> None:
    from maxim.models.language import config
    from maxim.runtime import lane_backends

    monkeypatch.setattr(config, "normalize_llm_profile", lambda p: p)
    monkeypatch.setattr(config, "get_profile", lambda p: {"backend": spelling})
    monkeypatch.setattr(config, "list_llm_profiles", lambda: ["p"])
    assert lane_backends._validate_swap_profile("p") == ("p", {"backend": spelling})


def test_the_swap_check_still_refuses_a_non_llama_backend(monkeypatch) -> None:
    from maxim.models.language import config
    from maxim.runtime import lane_backends

    monkeypatch.setattr(config, "normalize_llm_profile", lambda p: p)
    monkeypatch.setattr(config, "get_profile", lambda p: {"backend": "transformers"})
    monkeypatch.setattr(config, "list_llm_profiles", lambda: ["p"])
    with pytest.raises(ValueError, match="not compatible with llama-cpp-server"):
        lane_backends._validate_swap_profile("p")
