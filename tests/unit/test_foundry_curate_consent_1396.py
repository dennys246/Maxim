"""#1396: foundry and auto-curate ask before downloading a missing model (owner decision 2026-11-10).

Both build their ``--llm`` router outside ``build_primary_router``, so its preflight never asked, and the llama.cpp
backend's lazy load never prompts (#1343). At a terminal the run fell back to templates without the operator ever
being asked. ``cli.py::_cli_local_router`` now asks the consent gate (``models/download.py::ensure_available``) on the
main thread before warmup, and returns None (the template fallback, no warmup) when the model stays missing.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

CLI = Path(__file__).resolve().parents[2] / "src" / "maxim" / "cli.py"


@pytest.fixture
def fake_router(monkeypatch, tmp_path):
    """A registered profile whose GGUF is missing; the router records warmups, download_llm records downloads."""
    from maxim.models.download import LLM_MODELS
    from maxim.utils import storage

    profile = next(iter(LLM_MODELS))
    monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
    (tmp_path / "home" / "util").mkdir(parents=True)
    storage._reset_cache_for_tests()
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr("maxim.runtime.llm_server.profile_has_local_file", lambda p: False)
    record = SimpleNamespace(profile=profile, downloads=[], warmups=0, built=[], unloaded=[])

    class FakeRouter:
        def __init__(self, cfg):
            self.cfg = cfg
            record.built.append(self)

        def warmup(self):
            record.warmups += 1

        def wait_ready(self, timeout=0.0):
            return True

        def unload(self):
            record.unloaded.append(self)

    monkeypatch.setattr(
        "maxim.models.language.config.load_llm_config",
        lambda profile_override=None: SimpleNamespace(
            model_path=str(tmp_path / "missing.gguf"), profile=profile_override
        ),
    )
    monkeypatch.setattr("maxim.models.language.router.LLMRouter", FakeRouter)
    monkeypatch.setattr(
        "maxim.models.download.download_llm", lambda name, *a, **k: record.downloads.append(name) or True
    )
    return record


def _run_foundry(monkeypatch, profile: str) -> list:
    """Run ``_handle_foundry`` with a fake FoundryRunner; returns the routers the runner was handed."""
    from maxim import cli

    handed: list = []

    class FakeRunner:
        def __init__(self, **kw):
            handed.append(kw["llm_router"])

        def run(self, count):
            return SimpleNamespace(
                output_dir="out", generated=0, validated=0, tested=0, promoted=[], review=[], rejected=[]
            )

    monkeypatch.setattr("maxim.simulation.foundry.FoundryRunner", FakeRunner)
    assert cli._handle_foundry(SimpleNamespace(language_model=profile), "theme") == 0
    return handed


def _run_curate(monkeypatch, profile: str) -> list:
    """Run ``_auto_curate_before_sim`` with fake registry, index and curator; returns the routers it was handed.

    The fake curator stands for a coverage gap: it calls the router factory, as the real one does once it finds a
    gap (the no-gap case runs the real ``auto_curate``: ``test_full_coverage_never_builds_the_router_or_asks``)."""
    from maxim import cli

    handed: list = []

    def fake_auto_curate(**kw):
        assert kw.get("llm_router") is None  # the CLI hands over a factory, never a router built before the check
        handed.append(kw["llm_router_factory"]())
        return SimpleNamespace(total_promoted=0, total_skipped_dedup=0, categories_below_threshold=1, total_generated=0)

    monkeypatch.setattr("maxim.simulation.foundry.auto_curate", fake_auto_curate)
    monkeypatch.setattr("maxim.embodiment.component_registry.ComponentRegistry", lambda: object())
    monkeypatch.setattr("maxim.embodiment.component_index.ComponentIndex", lambda registry: None)
    args = SimpleNamespace(auto_curate=True, no_curate=False, language_model=profile)
    cli._auto_curate_before_sim(args, "bodies/base_humanoid")
    return handed


def test_foundry_asks_and_without_consent_falls_back_with_no_download_or_warmup(
    fake_router, monkeypatch, capsys
) -> None:
    handed = _run_foundry(monkeypatch, fake_router.profile)
    assert fake_router.downloads == []
    assert fake_router.warmups == 0  # a refused model is never warmed: the backend would only refuse again
    assert handed == [None]  # the template fallback
    assert "model not available" in capsys.readouterr().out  # the operator is told why


def test_foundry_downloads_with_the_operators_grant_then_warms(fake_router, monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")
    handed = _run_foundry(monkeypatch, fake_router.profile)
    assert fake_router.downloads == [fake_router.profile]
    assert fake_router.warmups == 1
    assert handed == fake_router.built


def test_curate_asks_and_without_consent_falls_back_with_no_download_or_warmup(
    fake_router, monkeypatch, capsys
) -> None:
    handed = _run_curate(monkeypatch, fake_router.profile)
    assert fake_router.downloads == []
    assert fake_router.warmups == 0
    assert handed == [None]
    assert "model not available" in capsys.readouterr().out  # the operator is told why


def test_curate_downloads_with_the_operators_grant_then_warms(fake_router, monkeypatch) -> None:
    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")
    handed = _run_curate(monkeypatch, fake_router.profile)
    assert fake_router.downloads == [fake_router.profile]
    assert fake_router.warmups == 1
    assert handed == fake_router.built


def test_a_present_model_is_used_as_is_without_asking(fake_router, monkeypatch, tmp_path) -> None:
    """A model already at the router's own path needs no gate: no download, and the router is warmed."""
    present = tmp_path / "missing.gguf"  # the fixture's model_path
    present.write_bytes(b"gguf")
    handed = _run_foundry(monkeypatch, fake_router.profile)
    assert fake_router.downloads == []
    assert fake_router.warmups == 1
    assert handed == fake_router.built


def _profiles_by_backend() -> dict[str, str]:
    """One cloud profile, one local non-llama.cpp (transformers) profile, one llama.cpp profile outside the registry."""
    from maxim.models.download import llm_models
    from maxim.models.language.config import iter_profiles

    rows = iter_profiles()
    return {
        "cloud": next(n for n, p in rows if p.get("cloud")),
        "torch": next(n for n, p in rows if not p.get("cloud") and p.get("backend") not in (None, "llama_cpp")),
        "unregistered_gguf": next(n for n, p in rows if p.get("backend") == "llama_cpp" and n not in llm_models()),
    }


def _with_backend(monkeypatch, tmp_path, profile: str) -> None:
    """The fake load_llm_config, carrying the profile's real backend (what ``_ensure_router_model`` reads)."""
    from maxim.models.language.config import get_profile

    backend = (get_profile(profile) or {}).get("backend", "llama_cpp")
    monkeypatch.setattr(
        "maxim.models.language.config.load_llm_config",
        lambda profile_override=None: SimpleNamespace(
            model_path=str(tmp_path / "missing.gguf"), profile=profile_override, backend=backend
        ),
    )


@pytest.mark.parametrize("kind", ["cloud", "torch"])
@pytest.mark.parametrize("run", [_run_foundry, _run_curate], ids=["foundry", "curate"])
def test_a_router_with_no_local_gguf_keeps_its_router_and_asks_nothing(
    fake_router, monkeypatch, tmp_path, kind, run
) -> None:
    """Review round D1 (all three lenses): ``load_llm_config`` fills a ``.gguf`` ``model_path`` even for a cloud
    backend, so a missing file alone must not send ``--llm claude-sonnet`` to the template fallback."""
    profile = _profiles_by_backend()[kind]
    _with_backend(monkeypatch, tmp_path, profile)
    asked: list[str] = []
    monkeypatch.setattr("maxim.models.download.ensure_available", lambda name, **k: asked.append(name) or False)
    handed = run(monkeypatch, profile)
    assert asked == []
    assert fake_router.downloads == []
    assert handed == fake_router.built and handed[0] is not None
    assert fake_router.warmups == 1


@pytest.mark.parametrize("run", [_run_foundry, _run_curate], ids=["foundry", "curate"])
def test_a_llama_cpp_profile_outside_the_registry_still_reaches_the_gate(
    fake_router, monkeypatch, tmp_path, capsys, run
) -> None:
    """Delta round: a built-in llama.cpp profile with no registry entry needs a GGUF the gate cannot fetch. It still
    reaches the gate, which says how to place the file by hand, and the run takes the template fallback unwarmed."""
    profile = _profiles_by_backend()["unregistered_gguf"]
    _with_backend(monkeypatch, tmp_path, profile)
    handed = run(monkeypatch, profile)
    assert fake_router.downloads == []
    assert fake_router.warmups == 0
    assert handed == [None]
    assert "model not available" in capsys.readouterr().out


@pytest.mark.parametrize("run", [_run_foundry, _run_curate], ids=["foundry", "curate"])
def test_a_decline_at_a_terminal_falls_back_with_no_download(fake_router, monkeypatch, run) -> None:
    """At a terminal the gate really asks; a "no" leaves the model missing, so no warmup and the template fallback."""
    questions: list[str] = []
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        "maxim.models.download._prompt_yes_no_with_timeout", lambda q, timeout_s=30.0: questions.append(q) or False
    )
    handed = run(monkeypatch, fake_router.profile)
    assert len(questions) == 1  # asked once, on this path
    assert fake_router.downloads == []
    assert fake_router.warmups == 0
    assert handed == [None]


def test_in_cli_only_the_consent_builder_constructs_an_llm_router() -> None:
    """A new ``LLMRouter(...)`` in cli.py would be a router built with no one asking about a missing model: route it
    through ``_cli_local_router`` (or ``_ensure_router_model`` before its warmup) instead. Import aliases count."""
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    names = {"LLMRouter"} | {
        a.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for a in node.names
        if a.name == "LLMRouter" and a.asname
    }
    builders: set[str] = set()

    def visit(node: ast.AST, func: str) -> None:
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
            if isinstance(child, ast.Call):
                f = child.func
                called = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                if called in names:
                    builders.add(func)
            visit(child, name)

    visit(tree, "<module>")
    assert builders == {"_cli_local_router"}, sorted(builders)


@pytest.mark.parametrize("spelling", ["llamacpp", "Llama-CPP", "llama"])
def test_every_spelling_the_router_reads_as_llama_cpp_still_asks(fake_router, monkeypatch, tmp_path, spelling) -> None:
    """Delta round 3 NIT: llm.json or MAXIM_LLM_BACKEND may spell the backend any way the router accepts; each one
    builds a ``_LlamaCppBackend``, so each still needs the gate (no download without consent, no warmup)."""
    monkeypatch.setattr(
        "maxim.models.language.config.load_llm_config",
        lambda profile_override=None: SimpleNamespace(
            model_path=str(tmp_path / "missing.gguf"), profile=profile_override, backend=spelling
        ),
    )
    handed = _run_foundry(monkeypatch, fake_router.profile)
    assert fake_router.downloads == []
    assert fake_router.warmups == 0
    assert handed == [None]


def _run_real_curate(monkeypatch, profile: str, gaps: dict[str, int]) -> list:
    """Run ``_auto_curate_before_sim`` through the REAL ``auto_curate``, with ``analyze_coverage`` reporting ``gaps``
    and a fake FoundryRunner; returns the router each category's runner was handed."""
    from maxim import cli

    runners: list = []

    class FakeRunner:
        def __init__(self, **kw):
            runners.append(kw["llm_router"])

        def run(self, count):
            return SimpleNamespace(promoted=[], dedup_skipped=[], generated=0, promoted_paths=[])

    monkeypatch.setattr("maxim.simulation.foundry.analyze_coverage", lambda genre, **k: dict(gaps))
    monkeypatch.setattr("maxim.simulation.foundry.FoundryRunner", FakeRunner)
    monkeypatch.setattr("maxim.embodiment.component_registry.ComponentRegistry", lambda: object())
    monkeypatch.setattr("maxim.embodiment.component_index.ComponentIndex", lambda registry: None)
    args = SimpleNamespace(auto_curate=True, no_curate=False, language_model=profile)
    cli._auto_curate_before_sim(args, "bodies/base_humanoid")
    return runners


def test_full_coverage_never_builds_the_router_or_asks(fake_router, monkeypatch, capsys) -> None:
    """Owner decision 2026-11-12 (Q1 A): with every category covered, curation never uses the model, so nothing is
    built and nothing is asked: no "download ~N GB?" for a model the run would not touch."""
    asked: list[str] = []
    monkeypatch.setattr("maxim.models.download.ensure_available", lambda name, **k: asked.append(name) or False)
    runners = _run_real_curate(monkeypatch, fake_router.profile, gaps={})
    assert asked == []
    assert fake_router.built == []
    assert fake_router.downloads == [] and fake_router.warmups == 0
    assert runners == []
    assert "Coverage OK" in capsys.readouterr().out


def test_a_gap_builds_the_router_once_for_every_category(fake_router, monkeypatch) -> None:
    """With gaps, the router is built (and asked about) once, after the check, and every category's runner gets it."""
    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")
    runners = _run_real_curate(monkeypatch, fake_router.profile, gaps={"weapons": 2, "armor": 1})
    assert len(fake_router.built) == 1
    assert runners == [fake_router.built[0]] * 2
    assert fake_router.downloads == [fake_router.profile]
    assert fake_router.warmups == 1


def test_a_declined_gap_hands_every_category_the_template_fallback(fake_router, monkeypatch) -> None:
    runners = _run_real_curate(monkeypatch, fake_router.profile, gaps={"weapons": 2, "armor": 1})
    assert fake_router.downloads == [] and fake_router.warmups == 0
    assert runners == [None, None]


def test_auto_curate_calls_no_factory_without_a_gap_and_refuses_both_kinds_of_router(monkeypatch) -> None:
    from maxim.simulation import foundry

    calls: list[int] = []
    monkeypatch.setattr(foundry, "analyze_coverage", lambda genre, **k: {})
    foundry.auto_curate("fantasy", registry=object(), llm_router_factory=lambda: calls.append(1))
    assert calls == []
    with pytest.raises(TypeError, match="not both"):
        foundry.auto_curate("fantasy", registry=object(), llm_router=object(), llm_router_factory=lambda: None)


def test_a_warmup_error_keeps_the_built_router_and_curation_goes_on(fake_router, monkeypatch, capsys) -> None:
    """Review round NIT: the factory's handler, moved from the old curate block, keeps what was built when warmup
    raises (debug-logged, as before the move) and never reaches the outer "Auto-curation failed" handler."""
    monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")
    from maxim.models.language import router as router_mod

    def boom(self):
        raise RuntimeError("warmup failed")

    monkeypatch.setattr(router_mod.LLMRouter, "warmup", boom)
    runners = _run_real_curate(monkeypatch, fake_router.profile, gaps={"weapons": 2})
    assert runners == fake_router.built and runners[0] is not None
    assert "Auto-curation failed" not in capsys.readouterr().err
