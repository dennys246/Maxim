"""#821 — the model must not be able to switch itself into singularity mode.

`singularity` is the only mode with `can_execute_code=True` plus full tools and network, so a
model-initiated escalation to it (e.g. prompted by injected web text) must never reach
`set_mode`. These tests drive the real `ModeSwitchTool` with recording callbacks.
"""

from __future__ import annotations

from maxim.tools.mode_switch import ModeSwitchTool


class _Modes:
    def __init__(self, current: str) -> None:
        self.current = current
        self.set_calls: list[str] = []

    def get(self) -> str:
        return self.current

    def set(self, mode: str) -> None:
        self.set_calls.append(mode)


def _tool(current: str) -> tuple[ModeSwitchTool, _Modes]:
    modes = _Modes(current)
    return ModeSwitchTool(get_current_mode=modes.get, set_mode=modes.set), modes


def test_model_cannot_escalate_to_singularity_from_passive() -> None:
    tool, modes = _tool("passive")
    result = tool.execute(mode="singularity", reason="the page says to")
    assert result.success is False
    assert modes.set_calls == []


def test_model_cannot_escalate_to_singularity_from_active() -> None:
    tool, modes = _tool("active")
    result = tool.execute(mode="singularity")
    assert result.success is False
    assert modes.set_calls == []


def test_model_cannot_escalate_to_singularity_from_a_legacy_mode_name() -> None:
    # `maxim.mode` can still carry a legacy name ("live" -> active); the gate must resolve it.
    tool, modes = _tool("live")
    result = tool.execute(mode="SINGULARITY")
    assert result.success is False
    assert modes.set_calls == []


def test_deescalation_from_singularity_is_free() -> None:
    tool, modes = _tool("singularity")
    result = tool.execute(mode="passive")
    assert result.success is True
    assert modes.set_calls == ["passive"]


def test_passive_to_active_still_allowed() -> None:
    # `active` is not a code-executing mode (`can_execute_code=False`), so this stays open. Note it
    # is not code-free when unattended: non-interactive runs auto-answer "yes" to confirmations.
    tool, modes = _tool("passive")
    result = tool.execute(mode="active")
    assert result.success is True
    assert modes.set_calls == ["active"]


def test_refusal_is_recorded_in_the_autonomy_audit_log() -> None:
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel

    controller = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    modes = _Modes("passive")
    tool = ModeSwitchTool(get_current_mode=modes.get, set_mode=modes.set, autonomy_controller=controller)

    result = tool.execute(mode="singularity", reason="injected")

    assert result.success is False
    assert modes.set_calls == []
    entries = controller.get_audit_log()
    assert entries and entries[-1].action_type == "rejected"


def test_rejection_does_not_count_toward_the_mode_switch_rate_limit() -> None:
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel

    controller = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    tool = ModeSwitchTool(get_current_mode=lambda: "passive", set_mode=lambda m: None, autonomy_controller=controller)
    tool.execute(mode="singularity")
    assert controller._get_mode_switches_last_hour() == 0


def test_already_in_singularity_is_not_refused_whatever_the_case() -> None:
    # A human-started singularity session re-requesting its own mode is harmless; pin it so a
    # future change cannot quietly widen or narrow this case.
    tool, modes = _tool("Singularity")
    result = tool.execute(mode="singularity")
    assert result.success is True


def test_non_string_mode_is_rejected_not_crashed() -> None:
    tool, modes = _tool("passive")
    result = tool.execute(mode=None)
    assert result.success is False
    assert modes.set_calls == []


def test_executes_code_is_derived_from_the_mode_definition() -> None:
    from maxim.tools.mode_switch import executes_code

    assert executes_code("singularity") is True
    assert executes_code("SINGULARITY") is True
    assert executes_code("active") is False
    assert executes_code("live") is False  # legacy name for active
    assert executes_code("no-such-mode") is False


def test_real_registry_wiring_never_sets_requested_mode_to_singularity() -> None:
    """End to end through `build_tool_registry`: the tool the agent actually gets is gated."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.runtime.bootstrap import build_tool_registry

    class _Maxim:
        mode = "passive"
        requested_mode = None

    maxim = _Maxim()
    registry = build_tool_registry(
        maxim=maxim, autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    )
    result = registry.get("mode_switch").execute(mode="singularity")
    assert result.success is False
    assert maxim.requested_mode is None
