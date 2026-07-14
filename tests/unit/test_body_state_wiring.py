"""Exp 44 body_state wiring seam (docs/plans/acting_coach_body_state_ablation.md).

Pins the opt-in (default OFF) routing of the executor's Embodiment into
``MemoryHub.embodiment`` at AgentFactory wiring time. Default-off is
load-bearing: arm A of the pre-registered ablation IS the unwired status
quo, and no production behavior may change until the ablation decides.
Env scrub: tests/conftest.py::_isolate_maxim_body_state_prompt_env.
"""

from __future__ import annotations

from types import SimpleNamespace

from maxim.integration.memory_hub import body_state_prompt_enabled
from maxim.runtime.agent_factory import _maybe_wire_body_state


class TestBodyStatePromptEnabled:
    def test_default_off(self):
        assert body_state_prompt_enabled() is False

    def test_truthy_values(self, monkeypatch):
        for val in ("1", "true", "t", " YES ", "y", "on"):
            monkeypatch.setenv("MAXIM_ENABLE_BODY_STATE_PROMPT", val)
            assert body_state_prompt_enabled() is True

    def test_falsy_values(self, monkeypatch):
        for val in ("", "0", "false", "no", "off"):
            monkeypatch.setenv("MAXIM_ENABLE_BODY_STATE_PROMPT", val)
            assert body_state_prompt_enabled() is False


class TestMaybeWireBodyState:
    def _instance(self, *, embodiment, hub):
        return SimpleNamespace(embodiment=embodiment, memory_hub=hub)

    def test_default_off_leaves_hub_unwired(self):
        hub = SimpleNamespace(embodiment=None)
        inst = self._instance(embodiment=object(), hub=hub)
        _maybe_wire_body_state(inst)
        assert hub.embodiment is None  # arm-A status quo preserved

    def test_enabled_wires_embodiment_into_hub(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ENABLE_BODY_STATE_PROMPT", "1")
        emb = object()
        hub = SimpleNamespace(embodiment=None)
        inst = self._instance(embodiment=emb, hub=hub)
        _maybe_wire_body_state(inst)
        assert hub.embodiment is emb

    def test_enabled_without_embodiment_is_noop(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ENABLE_BODY_STATE_PROMPT", "1")
        hub = SimpleNamespace(embodiment=None)
        inst = self._instance(embodiment=None, hub=hub)
        _maybe_wire_body_state(inst)
        assert hub.embodiment is None

    def test_enabled_without_hub_is_noop(self, monkeypatch):
        monkeypatch.setenv("MAXIM_ENABLE_BODY_STATE_PROMPT", "1")
        inst = self._instance(embodiment=object(), hub=None)
        _maybe_wire_body_state(inst)  # must not raise
