"""#1658 (+ #1677's worker-thread cases): the router build's download preflight never prompts off the operator's startup.

Before this, ``lane_backends._ensure_lane_profiles_available`` (the preflight inside ``build_primary_router``) called
``ensure_available`` with the default ``interactive=None``, so it asked "Proceed? [y/N]" whenever stdin was a terminal,
on any thread. The console's talk build runs on a FastAPI worker thread (``console/server.py::post_run`` is a sync
route): a talk request against ``maxim serve`` started at a terminal sat 30 s on the server's stdin, where a line typed
there for anything else would read as consent. The robot's wake-word rebuild (a worker thread) was in the same class.
#1674 had already put a main-thread rule on its own router-level check; the preflight one layer down did not share it,
and ``ExecAgent`` kept a third copy of the rule.

Owner decisions 2026-11-17 (each the recommended option): OD-A the thread rule governs the preflight for every caller;
OD-B an explicit ``interactive=False`` keyword, which the console's talk build passes; OD-C the console's error names the
missing models and the grant; OD-D ``maxim serve`` is grant-only; OD-E the robot's first build in ``live()`` (main
thread) is left as is; OD-F talk builds through the ``build_run_router`` seam; OD-G the one rule lives in
``models/download.py::consent_interactive``. Design: scratchpad approach v2 + two design passes.
"""

from __future__ import annotations

import ast
import logging
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "maxim"
SCRIPTS = ROOT / "scripts"


def _on_worker(fn, *args, **kwargs) -> Any:
    """Run ``fn`` on a fresh non-main thread and return its result (re-raising its exception)."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread below
            box["error"] = e

    t = threading.Thread(target=run, name="test-worker")
    t.start()
    t.join(10.0)
    assert not t.is_alive(), "worker did not finish"
    if "error" in box:
        raise box["error"]
    return box.get("value")


# ── the one rule (OD-G) ───────────────────────────────────────────────────────


def test_consent_interactive_asks_only_on_the_main_thread() -> None:
    from maxim.models.download import consent_interactive

    assert consent_interactive() is None
    assert consent_interactive(None) is None
    assert consent_interactive(False) is False
    assert _on_worker(consent_interactive) is False
    assert _on_worker(consent_interactive, None) is False
    assert _on_worker(consent_interactive, False) is False


@pytest.mark.parametrize("value", [True, 1, "yes"])
def test_consent_interactive_refuses_to_force_a_prompt(value: Any) -> None:
    from maxim.models.download import consent_interactive

    with pytest.raises(TypeError):
        consent_interactive(value)


# ── the preflight (OD-A, OD-B) ────────────────────────────────────────────────


@pytest.fixture
def preflight(monkeypatch):
    """The preflight on one local lane whose registry model is missing; the gate records each call's ``interactive``."""
    from maxim.runtime import lane_backends as LB
    from maxim.runtime.worker_pool import LaneConfig

    calls: list[Any] = []

    def ensure_available(profile, *, auto=None, interactive=None, logger=None):
        calls.append(interactive)
        return True  # granted: no re-walk, so the test sees only the ask

    monkeypatch.setattr("maxim.models.download.ensure_available", ensure_available)
    monkeypatch.setattr("maxim.models.download.llm_models", lambda: {"m-1658": {}})
    monkeypatch.setattr("maxim.models.language.config.get_profile", lambda p: {})
    monkeypatch.setattr(LB, "_profile_has_local_file", lambda p: False)
    lanes = {"large": LaneConfig(name="large", max_workers=1, model_profile="m-1658")}
    return SimpleNamespace(LB=LB, lanes=lanes, calls=calls)


def test_the_preflight_on_the_main_thread_still_asks(preflight) -> None:
    """Control: the operator's own startup (main thread) keeps the terminal ask."""
    preflight.LB._ensure_lane_profiles_available(preflight.lanes, None, None)
    assert preflight.calls == [None]


def test_the_preflight_on_a_worker_thread_never_asks(preflight) -> None:
    _on_worker(preflight.LB._ensure_lane_profiles_available, preflight.lanes, None, None)
    assert preflight.calls == [False]


def test_the_preflight_honours_an_explicit_never_on_the_main_thread(preflight) -> None:
    preflight.LB._ensure_lane_profiles_available(preflight.lanes, None, None, interactive=False)
    assert preflight.calls == [False]


def test_build_primary_router_hands_its_interactive_to_the_preflight() -> None:
    tree = ast.parse((SRC / "runtime" / "lane_backends.py").read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_primary_router")
    assert "interactive" in [a.arg for a in fn.args.kwonlyargs]
    calls = [
        n
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "_ensure_lane_profiles_available"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw.get("interactive"), ast.Name) and kw["interactive"].id == "interactive"


# ── the seam (D1: the resolved value reaches both checks) ─────────────────────


class _Manager:
    def __init__(
        self, declined: frozenset[str] = frozenset(), kind: str = "local", unusable: dict[str, str] | None = None
    ) -> None:
        self.declined_profiles, self.kind = declined, kind
        self.unusable_models: dict[str, str] = dict(unusable or {})

    def get_lane_kind(self, lane: str) -> str:
        return self.kind


@pytest.fixture
def seam(monkeypatch, tmp_path):
    """``build_primary_router`` records its keywords and gives no router; the fallback is a local model not on disk."""
    seen: dict[str, Any] = {"gate": []}

    def primary(**kw):
        seen["primary_kw"] = kw
        return None, _Manager()

    def load(profile_override=None):
        return SimpleNamespace(
            enabled=True, backend="llama_cpp", model_path=str(tmp_path / "x.gguf"), profile="x-1658", model_base=""
        )

    def ensure_available(profile, *, auto=None, interactive=None, logger=None):
        seen["gate"].append(interactive)
        return False

    class _Router:
        def __init__(self, cfg: Any = None, **kw: Any) -> None:
            self.cfg = cfg

    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", primary)
    monkeypatch.setattr("maxim.models.language.router.load_llm_config", load)
    monkeypatch.setattr("maxim.models.language.router.LLMRouter", _Router)
    monkeypatch.setattr("maxim.models.download.ensure_available", ensure_available)
    monkeypatch.setattr("maxim.models.download.llm_models", lambda: {"x-1658": {}})
    return seen


def test_the_seam_on_a_worker_thread_never_asks_in_either_check(seam) -> None:
    from maxim.runtime.lane_backends import build_run_router

    router, _ = _on_worker(build_run_router)
    assert router is None
    assert seam["primary_kw"].get("interactive") is False, "the preflight would ask on this thread"
    assert seam["gate"] and set(seam["gate"]) == {False}


def test_the_seam_honours_an_explicit_never_on_the_main_thread(seam) -> None:
    from maxim.runtime.lane_backends import build_run_router

    router, _ = build_run_router(interactive=False)
    assert router is None
    assert seam["primary_kw"].get("interactive") is False
    assert seam["gate"] and set(seam["gate"]) == {False}


def test_the_seam_records_the_fallback_model_it_refused(seam) -> None:
    """Review round (all three lenses): with no lane router, the ``llm.json`` fallback's missing model is the one the
    run would have served, so the seam records it (and its cause) for a caller that names it."""
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.runtime.lane_backends import build_run_router

    router, manager = _on_worker(build_run_router)
    assert router is None
    assert manager.unusable_models == {"x-1658": LocalModelUnavailable.NOT_DOWNLOADED}


def test_the_seam_records_a_primary_router_whose_model_it_refused(monkeypatch, tmp_path) -> None:
    """The in-process large lane's router fails the seam's check (here its model_path is gone, the registry file is
    present): no router, and the cause is the one the check found, not "not downloaded"."""
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.runtime.lane_backends import build_run_router

    cfg = SimpleNamespace(
        enabled=True, backend="llama_cpp", model_path=str(tmp_path / "gone.gguf"), profile="p-1658", model_base=""
    )
    monkeypatch.setattr(
        "maxim.runtime.lane_backends.build_primary_router", lambda **kw: (SimpleNamespace(cfg=cfg), _Manager())
    )
    monkeypatch.setattr("maxim.models.download.ensure_available", lambda *a, **k: True)
    router, manager = _on_worker(build_run_router)
    assert router is None
    assert manager.unusable_models == {"p-1658": LocalModelUnavailable.MODEL_PATH_MISSING}


def test_the_manager_carries_only_the_large_lanes_declines_as_its_routers_unusable_models(monkeypatch) -> None:
    """The hand-off at the end of the build: every decline still rides on ``declined_profiles`` (not asked twice), and
    only the large lane's become ``unusable_models``, the router's own (deletion probe: the hand-off went untested)."""
    from maxim.models.language.types import LocalModelUnavailable
    from maxim.runtime import lane_backends as LB
    from maxim.runtime.worker_pool import LaneConfig

    monkeypatch.setattr(LB, "_apply_cloud_fallback_override", lambda lanes, logger, peer: lanes)
    monkeypatch.setattr(LB, "_function_router_for", lambda lanes, cfg: None)
    monkeypatch.setattr(LB, "_print_lane_banner", lambda manager: None)
    monkeypatch.setattr(LB, "_append_decision_record", lambda *a, **k: None)
    monkeypatch.setattr(LB.LaneBackendManager, "get_backend", lambda self, lane: None)
    lanes = {"large": LaneConfig(name="large", max_workers=1, model_profile=None)}
    router, manager = LB._primary_router_and_manager(
        None,
        lanes,
        None,
        profile=None,
        _has_peer_config=False,
        _auto_spawn_sources={},
        _llm_json_func_config={},
        declined={"big-1658", "small-1658"},
        large_declined={"big-1658"},
    )
    assert router is None
    assert manager.declined_profiles == frozenset({"big-1658", "small-1658"})
    assert manager.unusable_models == {"big-1658": LocalModelUnavailable.NOT_DOWNLOADED}


def test_only_the_large_lanes_preflight_declines_count_as_the_routers_model(monkeypatch) -> None:
    """Executor S1: on a no-GPU box only a small tier resolves, and the large lane has no model; the small lane's
    missing model is not talk's, so naming it would send the operator after a download that does not fix talk."""
    from maxim.runtime import lane_backends as LB
    from maxim.runtime.worker_pool import LaneConfig

    monkeypatch.setattr("maxim.models.download.ensure_available", lambda *a, **k: False)
    monkeypatch.setattr("maxim.models.download.llm_models", lambda: {"big-1658": {}, "small-1658": {}})
    monkeypatch.setattr("maxim.models.language.config.get_profile", lambda p: {})
    monkeypatch.setattr(LB, "_profile_has_local_file", lambda p: False)
    lanes = {
        "large": LaneConfig(name="large", max_workers=1, model_profile="big-1658"),
        "small": LaneConfig(name="small", max_workers=1, model_profile="small-1658"),
    }
    declined: set[str] = set()
    large: set[str] = set()
    LB._ensure_lane_profiles_available(lanes, None, None, declined_out=declined, large_declined_out=large)
    assert declined == {"big-1658", "small-1658"}
    assert large == {"big-1658"}


# ── the console's talk build (OD-B, OD-C, OD-F, D4, D5) ───────────────────────


def _handle(tmp_path: Path) -> Any:
    from maxim.console.handle import MaximHandle

    h = MaximHandle.__new__(MaximHandle)
    h.agent_id = "t"
    h._stopped = False
    h._talk_lock = threading.Lock()
    h._talk_bridge = h._talk_thread = h._talk_stop = h._talk_worker = None
    h._talk_bound = None
    h._talk_generation = None
    h._ending_talk_loops = []
    h._workspace = tmp_path / "ws"
    return h


@pytest.fixture
def talk_build(monkeypatch):
    """The talk build's router comes from ``build_run_router``, recorded; ``build_primary_router`` must not be called."""
    seen: dict[str, Any] = {"calls": [], "result": (None, _Manager())}

    def run_router(*args, **kw):
        seen["calls"].append(kw)
        return seen["result"]

    def primary(*args, **kw):
        raise AssertionError("the talk build went around the build_run_router seam")

    monkeypatch.setattr("maxim.runtime.lane_backends.build_run_router", run_router)
    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", primary)
    return seen


def test_talk_builds_through_the_seam_and_never_asks(tmp_path, talk_build) -> None:
    from maxim.console import handle as H
    from maxim.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        _handle(tmp_path)._ensure_talk_loop()
    assert len(talk_build["calls"]) == 1
    kw = talk_build["calls"][0]
    assert kw.get("interactive") is False
    assert kw.get("logger") is H.logger, "the 'does not ask' line belongs in the server log, not on its stdout (D5)"


def test_talk_with_no_model_names_it_and_the_grant_and_never_says_declined(tmp_path, talk_build) -> None:
    from maxim.exceptions import ConfigurationError

    talk_build["result"] = (None, _Manager(unusable={"qwen-1658": "not_downloaded"}))
    with pytest.raises(ConfigurationError) as info:
        _handle(tmp_path)._ensure_talk_loop()
    msg = str(info.value)
    assert "qwen-1658" in msg
    assert "maxim config set llm.auto_download true" in msg
    assert "python -m maxim.models.download --llm qwen-1658" in msg
    assert "declin" not in msg.lower(), "talk never asked, so nobody declined"


def test_talk_names_the_fallback_model_the_seam_refused(tmp_path, seam) -> None:
    """Review round (all three lenses): no lane router and the ``llm.json`` fallback's model missing used to end in
    'configure one first'; the real seam now hands talk that model, so the error names it and the grant."""
    from maxim.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError) as info:
        _on_worker(_handle(tmp_path)._ensure_talk_loop)
    msg = str(info.value)
    assert "x-1658" in msg and "python -m maxim.models.download --llm x-1658" in msg
    assert "configure one first" not in msg
    assert set(seam["gate"]) == {False}, "talk asked"


def test_talk_gives_each_model_its_own_fix() -> None:
    """Executor N1: several models, each with its own command; a cause other than 'not downloaded' gets its own
    static fix, not the download command."""
    from maxim.console.handle import _unusable_models
    from maxim.models.language.types import LocalModelUnavailable as U

    msg = _unusable_models({"b-1658": U.NOT_DOWNLOADED, "a-1658": U.MODEL_PATH_MISSING})
    assert "python -m maxim.models.download --llm b-1658" in msg
    assert "--llm a-1658" not in msg
    assert U("a-1658", cause=U.MODEL_PATH_MISSING).fix_hint in msg
    assert "declin" not in msg.lower()


def test_talk_with_no_router_and_nothing_missing_keeps_the_configure_message(tmp_path, monkeypatch) -> None:
    """Control: no lane and no fallback, with no model left missing, is the existing 'configure one first'."""
    from maxim.exceptions import ConfigurationError

    monkeypatch.setattr("maxim.runtime.lane_backends.build_run_router", lambda *a, **k: (None, _Manager()))
    monkeypatch.setattr("maxim.runtime.lane_backends.build_primary_router", lambda *a, **k: (None, _Manager()))
    with pytest.raises(ConfigurationError, match="configure one first"):
        _handle(tmp_path)._ensure_talk_loop()


def test_talk_on_a_smaller_model_logs_which_one_is_missing_and_the_grant(tmp_path, talk_build, monkeypatch, caplog):
    from maxim.console import handle as H

    router = SimpleNamespace(n_ctx=2048, get_token_counter=lambda: None)
    talk_build["result"] = (router, _Manager(unusable={"big-1658": "not_downloaded"}))

    class _Worker:
        def __init__(self, **kw: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr("maxim.agents.llm_worker.LLMWorker", _Worker)
    monkeypatch.setattr(H.MaximHandle, "_launch_talk_loop", lambda self, worker, ws: "bridge")
    with caplog.at_level(logging.INFO, logger=H.logger.name):
        assert _handle(tmp_path)._ensure_talk_loop() == "bridge"
    text = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.INFO)
    assert "big-1658" in text and "llm.auto_download" in text


# ── pins ──────────────────────────────────────────────────────────────────────

_GATES = {"ensure_available", "consented_router", "router_model_problem"}
_BUILDS = {"build_primary_router", "build_run_router"}


def _py_files() -> list[Path]:
    return sorted(p for base in (SRC, SCRIPTS) for p in base.rglob("*.py"))


def _qualified(tree: ast.Module) -> list[tuple[ast.AST, str]]:
    """Every node with the qualname of its innermost enclosing def/class ('<module>' at top level)."""
    out: list[tuple[ast.AST, str]] = []

    def walk(node: ast.AST, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            out.append((child, ".".join(scope) or "<module>"))
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                walk(child, [*scope, child.name])
            else:
                walk(child, scope)

    walk(tree, [])
    return out


def _callee(call: ast.Call) -> str | None:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _imported_as(tree: ast.Module, names: set[str]) -> set[str]:
    """``names`` plus every local name an ``import ... as`` gives them in ``tree``."""
    out = set(names)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            out |= {a.asname for a in node.names if a.name in names and a.asname}
    return out


def _thread_checks(source: str) -> list[str]:
    """Qualnames of the ``main_thread()`` calls (aliased imports included) in a file that calls a consent gate or
    builds a router; empty for any other file."""
    tree = ast.parse(source)
    triggers = _imported_as(tree, _GATES | _BUILDS)
    nodes = _qualified(tree)
    if not any(isinstance(n, ast.Call) and _callee(n) in triggers for n, _ in nodes):
        return []
    thread_rule = _imported_as(tree, {"main_thread"})
    return [scope for node, scope in nodes if isinstance(node, ast.Call) and _callee(node) in thread_rule]


def test_only_consent_interactive_decides_by_thread_at_a_consent_site() -> None:
    """A file that calls a consent gate or builds a router never asks ``threading.main_thread()`` itself (an aliased
    import included): the one rule is ``models/download.py::consent_interactive`` (OD-G). Other uses of
    ``main_thread()`` (signals, the camera display) are not consent sites and are out of view."""
    offenders, seen_the_rule = [], False
    for path in _py_files():
        for scope in _thread_checks(path.read_text(encoding="utf-8")):
            if path == SRC / "models" / "download.py" and scope == "consent_interactive":
                seen_the_rule = True
            else:
                offenders.append(f"{path.relative_to(ROOT)}::{scope}")
    assert seen_the_rule, "positive control: the scan no longer sees the one rule itself"
    assert offenders == []
    planted = "from threading import main_thread as mt\nfrom x import build_run_router as b\ndef f():\n    b(); mt()\n"
    assert _thread_checks(planted) == ["f"], "positive control: an aliased check beside an aliased build"


def _forced_prompts(source: str) -> list[str]:
    """Qualnames of gate calls (aliased imports included) whose ``interactive=`` is anything but omitted, a ``None`` or
    ``False`` constant, a bare name or a ``consent_interactive(...)`` call, or that pass ``**kwargs``."""
    tree = ast.parse(source)
    gates = _imported_as(tree, _GATES)
    out = []
    for node, scope in _qualified(tree):
        if not (isinstance(node, ast.Call) and _callee(node) in gates):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        value = kw.get("interactive")
        allowed = (
            value is None
            or (isinstance(value, ast.Constant) and (value.value is None or value.value is False))
            or isinstance(value, ast.Name)
            or (isinstance(value, ast.Call) and _callee(value) == "consent_interactive")
        )
        if not allowed or None in kw:
            out.append(scope)
    return out


def test_no_gate_call_outside_its_module_forces_a_prompt() -> None:
    """Outside ``models/download.py`` no gate call (an aliased import included) passes ``interactive=True``, any other
    literal, an expression such as ``sys.stdin.isatty()``, or a ``**kwargs`` that could carry one: each skips the rule
    and can ask off the operator's startup. Allowed: omitted or ``None`` (``ensure_available`` applies the rule to its
    default, review round S2), ``False``, a name (a resolved value handed on) or ``consent_interactive(...)``."""
    offenders = []
    for path in _py_files():
        if path != SRC / "models" / "download.py":
            offenders += [f"{path.relative_to(ROOT)}::{s}" for s in _forced_prompts(path.read_text(encoding="utf-8"))]
    assert offenders == []
    planted = (
        "import sys\nfrom maxim.models.download import ensure_available as ea, consent_interactive\n"
        "def a():\n    ea('x', interactive=True)\n"
        "def b():\n    ea('x', interactive=sys.stdin.isatty())\n"
        "def c(**kw):\n    ea('x', **kw)\n"
        "def ok():\n    ea('x'); ea('x', interactive=False); ea('x', interactive=consent_interactive())\n"
    )
    assert _forced_prompts(planted) == ["a", "b", "c"], "positive control"


@pytest.fixture
def gate_at_a_terminal(monkeypatch):
    """The real ``ensure_available`` on a registry model that is missing, stdin a terminal, no grant: the prompt is a
    recorder answering "no"."""
    from maxim.models import download as D

    asked: list[str] = []
    monkeypatch.setattr("maxim.runtime.llm_server.profile_has_local_file", lambda p: False)
    monkeypatch.setattr(D, "llm_models", lambda: {"g-1658": {"size_gb": 0.0}})
    monkeypatch.setattr("maxim.utils.storage.can_download", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(D, "_auto_download_enabled", lambda: False)
    monkeypatch.setattr(D, "_prompt_yes_no_with_timeout", lambda q, timeout_s=30.0: asked.append(q) or False)
    monkeypatch.setattr("sys.stdin", SimpleNamespace(isatty=lambda: True))
    return SimpleNamespace(D=D, asked=asked)


def test_the_gates_own_default_asks_on_the_main_thread(gate_at_a_terminal) -> None:
    """Control: the operator's startup at a terminal is still asked."""
    assert gate_at_a_terminal.D.ensure_available("g-1658") is False
    assert len(gate_at_a_terminal.asked) == 1


def test_the_gates_own_default_never_asks_off_the_main_thread(gate_at_a_terminal) -> None:
    """Review round S2: the invariant lives in the gate, not only in its callers' pins."""
    assert _on_worker(gate_at_a_terminal.D.ensure_available, "g-1658") is False
    assert gate_at_a_terminal.asked == []


def test_a_coroutine_on_the_main_threads_event_loop_is_never_asked() -> None:
    """Executor N2: under ``uvicorn.run`` an ``async def`` route runs on the main thread's loop; it is a request, not
    the startup."""
    import asyncio

    from maxim.models.download import consent_interactive

    async def inside() -> Any:
        return consent_interactive()

    assert asyncio.run(inside()) is False
    assert consent_interactive() is None, "control: the main thread with no loop running"


# Router builds that may ask at a terminal: the operator's own startup, or governed by the thread rule. Keyed by
# (path relative to the repo, qualname of the enclosing def). Everything else passes a literal ``interactive=False``.
_MAY_ASK = {
    ("src/maxim/runtime/lane_backends.py", "build_run_router"),
    ("src/maxim/cli.py", "_start_server_llm"),
    ("src/maxim/api.py", "_build_run_router"),
    ("src/maxim/create.py", "router"),
    ("src/maxim/simulation/orchestrator.py", "_build_shared_router"),
    ("src/maxim/simulation/orchestrator.py", "_build_aut_router"),
    ("src/maxim/simulation/research_orchestrator.py", "_build_research_llm"),
    ("src/maxim/embodied_runtime/agentic_runtime.py", "AgenticRuntimeMixin._start_agentic_runtime"),
    ("scripts/smoke_test_remote_lane.py", "<module>"),  # an operator's script, run from a terminal
}


def _build_references() -> list[tuple[str, str, ast.AST, ast.Call | None]]:
    """Every reference to a router build in ``src/`` and ``scripts/``: (path, qualname, node, the call it heads)."""
    refs = []
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = set(_BUILDS)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                aliases |= {a.asname for a in node.names if a.name in _BUILDS and a.asname}
        heads = {id(n.func): n for n in ast.walk(tree) if isinstance(n, ast.Call)}
        rel = path.relative_to(ROOT).as_posix()
        for node, scope in _qualified(tree):
            name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
            if name in aliases:
                refs.append((rel, scope, node, heads.get(id(node))))
    return refs


def test_every_router_build_is_classified_as_may_ask_or_never_asks() -> None:
    """Lexical, not a reachability proof: a caller of an allow-listed wrapper (``create.router``, ``api.run``) is not
    seen here, and a wrapper run on a worker thread (the console's campaign → the orchestrator) is governed by the
    thread rule, not by this pin. A new build site, a ``**kwargs`` passthrough or the function passed as a value fails
    until it is classified."""
    unclassified = []
    for rel, scope, node, call in _build_references():
        if rel == "src/maxim/runtime/lane_backends.py" and scope in {"<module>", "build_primary_router"}:
            continue  # the definitions and __all__
        if call is not None and not any(k.arg is None for k in call.keywords):
            kw = {k.arg: k.value for k in call.keywords}
            value = kw.get("interactive")
            if isinstance(value, ast.Constant) and value.value is False:
                continue
            if (rel, scope) in _MAY_ASK:
                continue
        unclassified.append(f"{rel}::{scope}")
    assert unclassified == []
