"""#963 Q3 -- a search's follow-up keeps ``engage`` only in an active- or singularity-class mode.

``get_tool_followup_type`` downgraded ``engage`` (the "offer follow-ups" template) to ``respond`` only for the
literal mode name ``"passive"``, which no shipped code writes into the loop state: the run modes are ``observe``,
``live``, ``exploration``, ... So the downgrade never fired on a shipped path, and a passive-class run mode
(``observe``, ``sleep``, ``train``, ``reflection``) got ``engage``. Owner decision Q3 (#963, 2026-10-09): judge the
mode by CLASS. ``engage`` stays only when ``get_mode(name)`` is a definition of class ``active`` or ``singularity``;
an unknown name, ``None`` or a passive-class mode gives ``respond`` (fail closed). A template-only change: both types
queue the same follow-up bookkeeping, and both templates tell the model to answer with ``respond``.

Its own red gates (and its own commit). Each ``xfail(strict=True)`` gate fails on ``main`` for the reason it states;
the plain tests are guards, green on ``main``.
"""

from __future__ import annotations

import pytest

from maxim.modes.definitions import get_tool_followup_type
from tests.unit._execute_learn_driver import run_once

pytestmark = pytest.mark.timeout(60)

_LITERAL_ONLY = "#963 Q3: only the literal 'passive' downgrades engage"


@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("observe", marks=pytest.mark.xfail(strict=True, reason=_LITERAL_ONLY)),
        pytest.param("sleep", marks=pytest.mark.xfail(strict=True, reason=_LITERAL_ONLY)),
        pytest.param("train", marks=pytest.mark.xfail(strict=True, reason=_LITERAL_ONLY)),
        pytest.param("reflection", marks=pytest.mark.xfail(strict=True, reason=_LITERAL_ONLY)),
        "passive",  # a guard: the literal already downgrades
    ],
)
def test_a_passive_class_mode_downgrades_engage_to_respond(mode):
    assert get_tool_followup_type("internet_search", mode) == "respond"
    assert get_tool_followup_type("web_search", mode) == "respond"


@pytest.mark.xfail(strict=True, reason="#963 Q3: an unknown mode name keeps engage (fails open)")
def test_an_unknown_mode_name_gives_respond():
    assert get_tool_followup_type("internet_search", "no-such-mode") == "respond"


@pytest.mark.xfail(strict=True, reason="#963 Q3: no mode keeps engage (fails open)")
@pytest.mark.parametrize("mode", [None, ""])
def test_no_mode_gives_respond(mode):
    assert get_tool_followup_type("internet_search", mode) == "respond"


@pytest.mark.parametrize(
    "mode", ["active", "live", "exploration", "research", "active-assistance", "agentic", "singularity"]
)
def test_an_active_or_singularity_class_mode_keeps_engage(mode):
    """A guard: the active- and singularity-class names keep the engage template."""
    assert get_tool_followup_type("internet_search", mode) == "engage"


@pytest.mark.parametrize("mode", ["observe", "active", "no-such-mode", None])
def test_a_tool_that_does_not_engage_is_unchanged_by_the_mode(mode):
    """A guard: only ``engage`` is mode-dependent; every other follow-up type is the tool's own."""
    from maxim.modes.definitions import TOOL_DESCRIPTIONS

    for name, info in TOOL_DESCRIPTIONS.items():
        if info.get("followup_type") != "engage":
            assert get_tool_followup_type(name, mode) == info.get("followup_type"), name


@pytest.mark.xfail(strict=True, reason="#963 Q3: an observe run mode (passive-class) still queues an engage follow-up")
@pytest.mark.parametrize("level", ["autonomous", "supervised", "planning"])
def test_an_observe_run_mode_queues_a_respond_follow_up(monkeypatch, tmp_path, level):
    """Through the real loop, on every dispatch path: the default CLI loop's run mode is ``observe``."""
    obs = run_once(monkeypatch, tmp_path, tool="internet_search", level=level, state_mode="observe")
    [fu] = obs.followups
    assert (fu.followup_type, fu.mode) == ("respond", "observe")
