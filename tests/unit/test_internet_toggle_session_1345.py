"""#1345: the agent's internet toggle lowers its own session only, and never writes the operator's switch.

``util/internet_access.json`` is the operator's persisted on/off, read on every request. Its only writer in ``src/`` was
the agent's ``internet_access_toggle``, so an agent could turn an operator's "off" back on, or persist "off" for
every later session. Owner decision 2026-10-19 (strict): the agent may LOWER internet for its own session, in
memory, never persisted, and may not raise it back. The session's lower is the process environment
(``MAXIM_INTERNET_LOWERED``): it survives a registry rebuild and a re-exec, and dies with the process tree.
Design: docs/plans/archive/internet_toggle_session_gate.md.
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


def test_only_the_internet_access_module_writes_the_operator_switch() -> None:
    assert _writers() <= {"utils/internet_access.py"}, sorted(_writers())


def test_the_agent_cannot_turn_the_operators_off_back_on(operator_home) -> None:
    from maxim.utils.internet_access import load_internet_policy

    toggle, before = operator_home
    result = _toggle(enabled=True)
    assert result.success is False
    assert toggle.read_bytes() == before
    assert load_internet_policy().enabled is False


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


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """A data home with no operator toggle (internet on by default) and no session lower."""
    from maxim.utils import paths
    from maxim.utils.internet_access import _reset_policy_cache

    monkeypatch.setenv("MAXIM_DATA_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("MAXIM_INTERNET_LOWERED", raising=False)
    paths._reset_caches()
    _reset_policy_cache()
    yield tmp_path / "home" / "util" / "internet_access.json"
    paths._reset_caches()
    _reset_policy_cache()


def test_the_agent_cannot_raise_back_after_its_own_lower(fresh_home) -> None:
    from maxim.utils.internet_access import load_internet_policy

    assert _toggle(enabled=False).success is True
    raised = _toggle(enabled=True)
    assert raised.success is False and "only the operator" in raised.error
    assert load_internet_policy().enabled is False


def test_turning_on_an_already_on_session_is_a_truthful_no_op(fresh_home) -> None:
    result = _toggle(enabled=True)
    assert result.success is True and "already on" in result.output
    assert not fresh_home.exists()


def test_a_lower_never_rewrites_an_existing_operator_on(fresh_home) -> None:
    from maxim.utils.internet_access import set_internet_access

    set_internet_access(True, path=fresh_home)
    before = fresh_home.read_bytes()
    _toggle(enabled=False)
    assert fresh_home.read_bytes() == before


def test_the_internet_tools_refuse_the_next_request_after_a_lower(fresh_home) -> None:
    from maxim.tools.http_fetch import HttpFetchTool
    from maxim.tools.internet_search import InternetSearchTool
    from maxim.utils.internet_access import live_internet_policy_getter

    getter = live_internet_policy_getter(True)
    fetch = HttpFetchTool(get_internet_policy=getter, content_safety_checker=lambda *a, **k: None)
    search = InternetSearchTool(get_internet_policy=getter)
    _toggle(enabled=False)
    for result in (fetch.execute(url="https://example.com"), search.execute(query="anything")):
        assert result.success is False and "this session" in (result.error or "")


def test_a_child_process_inherits_the_lower(fresh_home) -> None:
    import os
    import subprocess
    import sys

    _toggle(enabled=False)
    out = subprocess.run(
        [sys.executable, "-c", "from maxim.utils.internet_access import session_internet_lowered as s; print(s())"],
        env={**os.environ, "PYTHONPATH": str(SRC.parent)},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert out.stdout.strip() == "True", out.stderr


def test_an_explicit_falsy_value_is_not_lowered(fresh_home, monkeypatch) -> None:
    from maxim.utils.internet_access import load_internet_policy, session_internet_lowered

    monkeypatch.setenv("MAXIM_INTERNET_LOWERED", "0")
    assert session_internet_lowered() is False and load_internet_policy().enabled is True


def test_the_lower_is_written_once(fresh_home, monkeypatch) -> None:
    import os

    from maxim.utils import internet_access

    writes: list[str] = []
    real = os.environ.__class__.__setitem__

    def _spy(self, key, value):
        if key == internet_access.SESSION_LOWERED_ENV:
            writes.append(value)
        real(self, key, value)

    monkeypatch.setattr(os.environ.__class__, "__setitem__", _spy)
    internet_access.lower_internet_for_session()
    internet_access.lower_internet_for_session()
    assert writes == ["1"]


def test_the_startup_warning_names_the_session_lower_not_the_file(fresh_home, caplog) -> None:
    from maxim.utils.internet_access import live_internet_policy_getter, lower_internet_for_session

    lower_internet_for_session()
    caplog.clear()
    live_internet_policy_getter(True)
    assert "MAXIM_INTERNET_LOWERED" in caplog.text and "delete the toggle file" not in caplog.text


def test_a_legacy_agent_written_off_is_named_at_startup(fresh_home, caplog) -> None:
    from maxim.utils.internet_access import live_internet_policy_getter, set_internet_access

    set_internet_access(False, source="tool", path=fresh_home)  # what the tool wrote before #1345
    caplog.clear()
    live_internet_policy_getter(True)
    assert "written by an agent" in caplog.text


def test_the_refusal_names_the_remedy_for_why_access_is_off(operator_home) -> None:
    """Review fold: one period, and the remedy that matches the source (the operator's file, then a session lower)."""
    operator_off = _toggle(enabled=True).error
    assert ".." not in operator_off and "util/internet_access.json" in operator_off

    _toggle(enabled=False)
    session_off = _toggle(enabled=True).error
    assert ".." not in session_off and "MAXIM_INTERNET_LOWERED" in session_off


def test_a_string_enabled_says_it_is_a_string() -> None:
    error = _toggle(enabled="false").error
    assert "JSON boolean" in error and "str" in error


def test_an_unreadable_policy_file_names_the_policy_file(fresh_home, caplog) -> None:
    """Delta-round fold: not the toggle file, which is fine; and the legacy-writer note still fires."""
    from maxim.utils.internet_access import _reset_policy_cache, live_internet_policy_getter, set_internet_access

    policy = fresh_home.parent / "internet_policy.json"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text("{not json")
    set_internet_access(False, source="tool", path=fresh_home)
    _reset_policy_cache()
    caplog.clear()
    live_internet_policy_getter(True)
    assert "written by an agent" in caplog.text
    error = _toggle(enabled=True).error
    assert "util/internet_policy.json" in error and ".." not in error
