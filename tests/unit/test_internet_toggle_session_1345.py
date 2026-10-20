"""#1345: the agent's internet toggle lowers its own session only, and never writes the operator's switch.

``util/internet_access.json`` is the operator's persisted on/off, read on every request. Its only writer in ``src/`` was
the agent's ``internet_access_toggle``, so an agent could turn an operator's "off" back on, or persist "off" for
every later session. Owner decision 2026-10-19 (strict): the agent may LOWER internet for its own session, in
memory, never persisted, and may not raise it back. The session's lower is the process environment
(``MAXIM_INTERNET_LOWERED``): it survives a registry rebuild and a re-exec, and dies with the process tree.
Design: docs/plans/internet_toggle_session_gate.md.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "maxim"


@pytest.fixture
def operator_home(tmp_path, monkeypatch):
    """A data home whose operator toggle file says ``off``; returns the file and its bytes."""
    from maxim.utils import paths
    from maxim.utils.internet_access import _reset_policy_cache, set_internet_access

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MAXIM_INTERNET_LOWERED", raising=False)
    paths._reset_caches()
    _reset_policy_cache()
    toggle = tmp_path / "home" / "util" / "internet_access.json"
    set_internet_access(False, source="operator", path=toggle)
    yield toggle, toggle.read_bytes()
    paths._reset_caches()
    _reset_policy_cache()


def _toggle(**kwargs):
    from maxim.tools.internet_search import InternetAccessTool

    return InternetAccessTool().execute(**kwargs)


def _writers() -> set[str]:
    """Files in src/ that call set_internet_access / save_internet_access, through any import alias."""
    names = {"set_internet_access", "save_internet_access"}
    found: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = set(names)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                aliases |= {a.asname for a in node.names if a.name in names and a.asname}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                called = f.id if isinstance(f, ast.Name) else getattr(f, "attr", None)
                if called in aliases:
                    found.add(str(path.relative_to(SRC)))
    return found


@pytest.mark.xfail(strict=True, reason="#1345: the agent's toggle writes the operator's persisted switch")
def test_only_the_internet_access_module_writes_the_operator_switch() -> None:
    assert _writers() <= {"utils/internet_access.py"}, sorted(_writers())


@pytest.mark.xfail(strict=True, reason="#1345: the agent can turn the operator's off back on")
def test_the_agent_cannot_turn_the_operators_off_back_on(operator_home) -> None:
    from maxim.utils.internet_access import load_internet_policy

    toggle, before = operator_home
    result = _toggle(enabled=True)
    assert result.success is False
    assert toggle.read_bytes() == before
    assert load_internet_policy().enabled is False


@pytest.mark.xfail(strict=True, reason="#1345: a lower is persisted for every later session")
def test_a_lower_lasts_this_session_and_never_touches_the_file(tmp_path, monkeypatch) -> None:
    from maxim.utils import paths
    from maxim.utils.internet_access import _reset_policy_cache, live_internet_policy_getter

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MAXIM_INTERNET_LOWERED", raising=False)
    paths._reset_caches()
    _reset_policy_cache()
    toggle = tmp_path / "home" / "util" / "internet_access.json"

    assert _toggle(enabled=False).success is True
    assert not toggle.exists()  # nothing persisted
    fresh = live_internet_policy_getter(True)()  # a rebuilt registry's getter sees the same session
    assert (fresh.enabled, fresh.source) == (False, "session-lowered")
