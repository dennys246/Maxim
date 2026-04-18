"""Tests for RequestInteractionTool registration + handler wiring (C1)."""

from __future__ import annotations

from typing import Any

import pytest

from maxim.interactive.prompts import (
    PromptHandler,
    PromptRequest,
    PromptResponse,
)
from maxim.runtime.bootstrap import build_tool_registry
from maxim.simulation.sim_logger import (
    InteractiveMode,
    set_interactive_mode,
)
from maxim.tools.display import DisplayModeTool, RequestInteractionTool


class _RecordingHandler(PromptHandler):
    """Captures prompts and returns a canned response."""

    def __init__(self, answer: str = "yes") -> None:
        self.requests: list[PromptRequest] = []
        self._answer = answer

    def prompt(self, request: PromptRequest) -> PromptResponse:
        self.requests.append(request)
        return PromptResponse(value=self._answer)


@pytest.fixture(autouse=True)
def _restore_interactive_mode():
    prior = None
    try:
        from maxim.simulation.sim_logger import get_interactive_mode

        prior = get_interactive_mode()
    except Exception:
        prior = InteractiveMode.AUTO
    yield
    set_interactive_mode(prior)


def test_build_tool_registry_registers_display_tools():
    registry = build_tool_registry()
    assert registry.get("display_mode") is not None
    assert registry.get("request_interaction") is not None
    assert isinstance(registry.get("display_mode"), DisplayModeTool)
    assert isinstance(registry.get("request_interaction"), RequestInteractionTool)


def test_request_interaction_uses_handler_when_interactive_on():
    handler = _RecordingHandler(answer="proceed")
    registry = build_tool_registry(prompt_handler=handler)

    set_interactive_mode(InteractiveMode.ON)
    tool = registry.get("request_interaction")
    out = tool.execute(question="Continue?", options=["proceed", "abort"])

    assert out.success is True
    assert "proceed" in (out.output or "")
    assert out.output.startswith("User responded:")
    assert out.metadata["was_prompted"] is True
    assert len(handler.requests) == 1
    assert handler.requests[0].question == "Continue?"


def test_request_interaction_disabled_when_interactive_off():
    handler = _RecordingHandler()
    registry = build_tool_registry(prompt_handler=handler)

    set_interactive_mode(InteractiveMode.OFF)
    tool = registry.get("request_interaction")
    out = tool.execute(question="Anything?")

    assert out.success is True
    assert "autonomously" in (out.output or "").lower()
    assert out.metadata["was_prompted"] is False
    assert out.metadata["reason"] == "interactive_mode_off"
    assert handler.requests == []


def test_request_interaction_critical_overrides_off():
    handler = _RecordingHandler(answer="approved")
    registry = build_tool_registry(prompt_handler=handler)

    set_interactive_mode(InteractiveMode.OFF)
    tool = registry.get("request_interaction")
    out = tool.execute(question="Approve plan?", critical=True)

    assert out.success is True
    assert "approved" in (out.output or "")
    assert len(handler.requests) == 1


def test_request_interaction_falls_back_without_handler(capsys: Any):
    registry = build_tool_registry(prompt_handler=None)

    set_interactive_mode(InteractiveMode.ON)
    tool = registry.get("request_interaction")
    out = tool.execute(question="Pick one", options=["a", "b"])

    assert out.success is True
    # Falls back to display_scene printing the question but cannot collect response
    assert "could not be collected" in (out.output or "").lower()
    assert out.metadata["was_prompted"] is False
    assert out.metadata["reason"] == "no_handler"
