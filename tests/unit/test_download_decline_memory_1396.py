"""#1396 Q2: the download gate remembers a "no" for the rest of the run (owner decision 2026-11-12).

``models/download.py::ensure_available`` kept no state, so one run could ask the same question twice: ``--sim
interactive`` asks in ``cli.py::_ensure_router_model`` and again in the sim's preflight
(``lane_backends._ensure_lane_profiles_available``); ``--auto-curate`` asks, then the preflight does. A "no" (or a
30 s timeout) is now remembered inside a run: ``download_decline_scope`` (stacked: each scope starts empty and restores
the outer state on exit), entered unarmed by ``cli.py::_main_guarded``, armed by one ``arm_download_decline_memory()``
in ``_main_impl`` after the subcommand dispatch, and entered armed for each bare-``maxim`` menu choice
(``cli.py::_run_menu_choice``). Outside an armed scope (library, console, ``maxim serve``, every subcommand) the gate
asks every time, as before. Approach note v3 + adversarial pass: in the PR body.
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src" / "maxim"
_RED = pytest.mark.xfail(
    strict=True, reason="#1396 Q2 red gate: the download gate has no decline memory yet; the fix commit removes this"
)


@pytest.fixture
def gate(monkeypatch, tmp_path):
    """A registered profile whose GGUF is missing, at a fake terminal: records each question and each download."""
    from maxim.models import download
    from maxim.utils import storage

    profile = next(iter(download.LLM_MODELS))
    monkeypatch.delenv("MAXIM_AUTO_DOWNLOAD_MODELS", raising=False)
    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
    (tmp_path / "home" / "util").mkdir(parents=True)
    storage._reset_cache_for_tests()
    on_disk = {"present": False}
    monkeypatch.setattr("maxim.runtime.llm_server.profile_has_local_file", lambda p: on_disk["present"])
    monkeypatch.setattr("maxim.runtime.lane_backends._profile_has_local_file", lambda p: on_disk["present"])
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    rec = SimpleNamespace(profile=profile, questions=[], downloads=[], answer=False, on_disk=on_disk)

    def ask(question, timeout_s=30.0):
        rec.questions.append(question)
        return rec.answer

    monkeypatch.setattr(download, "_prompt_yes_no_with_timeout", ask)
    monkeypatch.setattr(download, "download_llm", lambda name, *a, **k: rec.downloads.append(name) or True)
    return rec


def _ask(rec) -> bool:
    from maxim.models.download import ensure_available

    return ensure_available(rec.profile)


@_RED
@pytest.mark.parametrize("answer", [False, None], ids=["no", "timeout"])
def test_an_answer_at_the_prompt_is_asked_once_per_armed_scope(gate, capsys, answer) -> None:
    from maxim.models.download import download_decline_scope

    gate.answer = answer
    with download_decline_scope(armed=True):
        assert _ask(gate) is False
        assert _ask(gate) is False
    assert len(gate.questions) == 1
    out = capsys.readouterr().out
    assert "declined earlier in this run" in out
    assert f"python -m maxim.models.download --llm {gate.profile}" in out  # the remedy that works mid-run
    assert "From the next run: --auto-download, or `maxim config set llm.auto_download true`" in out
    assert "read on every call" not in out  # config is cached per process: a mid-run config set is not seen
    assert gate.downloads == []


def test_outside_any_scope_the_gate_asks_every_time(gate) -> None:
    """The library, the console and ``maxim serve`` never open a scope: today's behaviour, pinned."""
    _ask(gate)
    _ask(gate)
    assert len(gate.questions) == 2


@_RED
def test_an_unarmed_scope_asks_every_time(gate) -> None:
    from maxim.models.download import download_decline_scope

    with download_decline_scope(armed=False):
        _ask(gate)
        _ask(gate)
    assert len(gate.questions) == 2


@_RED
def test_arming_turns_on_the_innermost_scope_and_without_a_scope_does_nothing(gate, caplog) -> None:
    from maxim.models.download import arm_download_decline_memory, download_decline_scope

    with caplog.at_level("WARNING"):
        arm_download_decline_memory()  # no scope open: warned, no effect
    assert any("no download decline scope" in r.getMessage() for r in caplog.records)
    _ask(gate)
    _ask(gate)
    assert len(gate.questions) == 2
    with download_decline_scope(armed=False):
        _ask(gate)  # unarmed: asked (3), and not remembered
        arm_download_decline_memory()
        _ask(gate)  # armed now: the unarmed "no" was never recorded, so asked (4)
        _ask(gate)  # ...and this one is remembered
    assert len(gate.questions) == 4


@_RED
def test_nested_scopes_neither_inherit_nor_wipe_the_outer_memory(gate) -> None:
    from maxim.models.download import download_decline_scope

    with download_decline_scope(armed=True):
        _ask(gate)  # asked (1), declined
        with download_decline_scope(armed=True):
            _ask(gate)  # a fresh scope asks again (2)
            _ask(gate)  # ...once
        _ask(gate)  # the outer "no" is still remembered
    assert len(gate.questions) == 2


@_RED
def test_a_scope_restores_the_outer_state_when_its_body_raises(gate) -> None:
    from maxim.models.download import download_decline_scope

    with pytest.raises(RuntimeError), download_decline_scope(armed=True):
        _ask(gate)
        raise RuntimeError("boom")
    _ask(gate)  # outside again: asks
    assert len(gate.questions) == 2


@_RED
def test_an_in_process_grant_after_a_no_still_downloads(gate, monkeypatch) -> None:
    """Grants are checked before the memory. (An operator cannot grant mid-run from outside: config is cached per
    process, so the printed hint offers the manual download now and the grants from the next run.)"""
    from maxim.models.download import download_decline_scope

    with download_decline_scope(armed=True):
        _ask(gate)
        monkeypatch.setenv("MAXIM_AUTO_DOWNLOAD_MODELS", "1")  # the env var is re-read on every call
        assert _ask(gate) is True
    assert gate.downloads == [gate.profile]
    assert len(gate.questions) == 1


@_RED
def test_a_file_that_appears_after_a_no_still_loads(gate) -> None:
    from maxim.models.download import download_decline_scope

    with download_decline_scope(armed=True):
        _ask(gate)
        gate.on_disk["present"] = True
        assert _ask(gate) is True


@_RED
def test_a_refusal_that_asked_nothing_is_not_remembered(gate, monkeypatch) -> None:
    """No terminal: refused without a question. A later call at a terminal still asks."""
    from maxim.models.download import download_decline_scope

    with download_decline_scope(armed=True):
        monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: False))
        assert _ask(gate) is False
        monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
        _ask(gate)
    assert len(gate.questions) == 1


@_RED
def test_a_yes_whose_download_fails_asks_again(gate, monkeypatch) -> None:
    """Only a "no" or a timeout is remembered: a "yes" whose download failed may be retried with a fresh ask."""
    from maxim.models import download
    from maxim.models.download import download_decline_scope

    gate.answer = True
    monkeypatch.setattr(download, "download_llm", lambda name, *a, **k: (gate.downloads.append(name), False)[1])
    with download_decline_scope(armed=True):
        assert _ask(gate) is False
        assert _ask(gate) is False
    assert len(gate.questions) == 2
    assert gate.downloads == [gate.profile, gate.profile]


@_RED
def test_a_call_on_another_thread_inside_the_scope_sees_the_memory(gate) -> None:
    """Module state, not a contextvar: Python threads do not inherit the caller's context, and preflights run off
    the main thread (the reason the note rejects a contextvar)."""
    from maxim.models.download import download_decline_scope

    with download_decline_scope(armed=True):
        _ask(gate)
        t = threading.Thread(target=_ask, args=(gate,))
        t.start()
        t.join(5)
    assert len(gate.questions) == 1


@_RED
def test_the_composed_sim_path_asks_once_after_a_no(gate) -> None:
    """The seam #1396 names: ``--sim interactive`` asks in ``_ensure_router_model``, then the sim's own preflight."""
    from maxim import cli
    from maxim.models.download import download_decline_scope
    from maxim.runtime.lane_backends import _ensure_lane_profiles_available
    from maxim.runtime.worker_pool import LaneConfig

    router = SimpleNamespace(
        cfg=SimpleNamespace(backend="llama_cpp", model_path="/nonexistent.gguf", profile=gate.profile)
    )
    lanes = {"large": LaneConfig(name="large", max_workers=1, model_profile=gate.profile)}
    with download_decline_scope(armed=True):
        assert cli._ensure_router_model(router) is False
        out = _ensure_lane_profiles_available(lanes, None, None)
    assert len(gate.questions) == 1
    assert out["large"].model_profile is None  # the preflight's own fallback, unchanged


@_RED
def test_main_guarded_opens_an_unarmed_scope_that_main_impl_arms(gate, monkeypatch) -> None:
    from maxim import cli
    from maxim.models.download import arm_download_decline_memory

    def run_mode(argv):
        arm_download_decline_memory()
        _ask(gate)
        _ask(gate)
        return 0

    def subcommand(argv):  # never arms
        _ask(gate)
        _ask(gate)
        return 0

    monkeypatch.setattr(cli, "_main_impl", run_mode)
    assert cli._main_guarded([]) == 0
    assert len(gate.questions) == 1
    monkeypatch.setattr(cli, "_main_impl", subcommand)
    assert cli._main_guarded([]) == 0
    assert len(gate.questions) == 3
    _ask(gate)  # after the run: no scope, asks
    assert len(gate.questions) == 4


@_RED
def test_each_menu_choice_is_its_own_run(gate, monkeypatch) -> None:
    """A "no" in one bare-``maxim`` menu choice never refuses the next choice (each is a separate intent)."""
    from maxim import cli

    monkeypatch.setattr(cli, "_run_menu_sim", lambda action: (_ask(gate), _ask(gate)))
    cli._run_menu_choice("interactive")
    cli._run_menu_choice("interactive")
    assert len(gate.questions) == 2


def _calls_in(callee: str) -> set[str]:
    """``{file::function}`` (file relative to src/maxim, or ``scripts/...``) of every call to ``callee`` (attribute or
    name) under src/maxim and scripts/: a harness that opened a scope in-process would widen it too."""
    found: set[str] = set()
    for path in [*SRC.rglob("*.py"), *(REPO / "scripts").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def visit(node, func):
            for child in ast.iter_child_nodes(node):
                name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
                if isinstance(child, ast.Call):
                    f = child.func
                    called = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                    if called == callee:
                        rel = path.relative_to(SRC) if SRC in path.parents else path.relative_to(REPO)
                        found.add(f"{rel.as_posix()}::{func}")
                visit(child, name)

        visit(tree, "<module>")
    return found


@_RED
def test_only_the_cli_run_sites_open_or_arm_the_memory() -> None:
    """A new scope or arm site widens what a "no" refuses: it must be a deliberate, reviewed change."""
    assert _calls_in("download_decline_scope") == {"cli.py::_main_guarded", "cli.py::_run_menu_choice"}
    assert _calls_in("arm_download_decline_memory") == {"cli.py::_main_impl"}
    assert "cli.py::_bare_maxim_menu" in _calls_in("_run_menu_choice")
    assert "cli.py::_bare_maxim_menu" not in _calls_in("_run_menu_sim")


@_RED
def test_an_inner_scopes_no_does_not_leak_into_the_outer_scope(gate) -> None:
    """Exit restores the outer SET, not only the armed flag: the CLI's run must not inherit a menu choice's "no"."""
    from maxim.models.download import download_decline_scope

    with download_decline_scope(armed=True):
        with download_decline_scope(armed=True):
            _ask(gate)  # (1) declined in the inner scope
        _ask(gate)  # (2) the outer scope never asked
    assert len(gate.questions) == 2


@_RED
def test_a_path_that_never_asks_is_unchanged_after_a_no(gate, capsys, caplog) -> None:
    """The memory sits inside the prompting branch only: the llama backend's lazy load (``interactive=False``, on the
    warmup thread) keeps its own logged message and prints nothing (#1343), even after a "no" at the terminal."""
    import logging

    from maxim.models.download import download_decline_scope, ensure_available

    log = logging.getLogger("test.decline")
    with download_decline_scope(armed=True):
        _ask(gate)
        capsys.readouterr()
        with caplog.at_level("WARNING"):
            assert ensure_available(gate.profile, interactive=False, logger=log) is False
    assert "declined earlier" not in capsys.readouterr().out
    assert any("this path does not ask" in r.getMessage() for r in caplog.records)
    assert len(gate.questions) == 1


@_RED
def test_a_no_answered_after_its_scope_exited_is_dropped(gate, monkeypatch) -> None:
    """A thread still at the prompt when its scope exits: its answer belongs to no current run, so it is dropped
    rather than landing in the outer scope."""
    from maxim.models import download
    from maxim.models.download import download_decline_scope

    asked, release = threading.Event(), threading.Event()

    def slow(question, timeout_s=30.0):
        gate.questions.append(question)
        asked.set()
        release.wait(5)
        return False

    monkeypatch.setattr(download, "_prompt_yes_no_with_timeout", slow)
    with download_decline_scope(armed=True):
        with download_decline_scope(armed=True):
            t = threading.Thread(target=_ask, args=(gate,))
            t.start()
            assert asked.wait(5)
        release.set()
        t.join(5)
        monkeypatch.setattr(download, "_prompt_yes_no_with_timeout", lambda q, timeout_s=30.0: gate.questions.append(q))
        _ask(gate)  # the outer scope never asked: it asks
    assert len(gate.questions) == 2


def test_a_subcommand_runs_unarmed_through_the_real_main_impl(gate, monkeypatch) -> None:
    """The arm comes AFTER the subcommand dispatch: ``maxim serve`` (the console, a long-lived server) and every other
    subcommand ask every time. Through the real ``_main_impl``, with only the tool-subcommand dispatcher stubbed."""
    from maxim import cli

    def serve(raw_argv):
        _ask(gate)
        _ask(gate)
        return 0

    monkeypatch.setattr(cli, "_dispatch_tool_cli", serve)
    assert cli._main_guarded(["serve"]) == 0
    assert len(gate.questions) == 2


def test_the_arm_comes_after_the_last_pre_run_dispatch_too(gate, monkeypatch) -> None:
    """The far end of the arm's placement (delta round): foundry, the last dispatch before the arm, runs unarmed.
    Through the real ``_main_impl``, with only ``_handle_foundry`` stubbed (and profiles.yml validation, which runs
    just before it)."""
    from maxim import cli

    def foundry(args, theme):
        _ask(gate)
        _ask(gate)
        return 0

    monkeypatch.setattr(cli, "_handle_foundry", foundry)
    monkeypatch.setattr("maxim.models.language.config.require_valid_user_profiles", lambda: None)
    assert cli._main_guarded(["--foundry", "fantasy weapons"]) == 0
    assert len(gate.questions) == 2
