"""#821 + #924 — the model must not be able to raise its own capability.

#821: `singularity` is the only mode with `can_execute_code=True` plus full tools and network, so a
model-initiated escalation to it (e.g. prompted by injected web text) must never reach `set_mode`.
#924 (owner decision 2026-09-27, STRICT): no capability raise at all -- passive -> active gains acting
on the host -- until the in-session approval surface (#922) exists. Lowering stays free. These tests
drive the real `ModeSwitchTool` with recording callbacks.
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


def test_passive_to_active_is_refused() -> None:
    # #924 STRICT (inverts #821's pinned `test_passive_to_active_still_allowed`): active gains acting on
    # the host, which passive refuses at dispatch (#826), so the model may not grant it to itself.
    tool, modes = _tool("passive")
    result = tool.execute(mode="active", reason="the task needs it")
    assert result.success is False
    assert modes.set_calls == []
    assert "raises capability" in (result.error or "")


def test_active_to_passive_is_free() -> None:
    tool, modes = _tool("active")
    result = tool.execute(mode="passive")
    assert result.success is True
    assert modes.set_calls == ["passive"]


def test_raises_capability_is_derived_from_dispatch_enforcement() -> None:
    from maxim.modes.definitions import raises_capability

    assert raises_capability("passive", "active") is True  # gains HOST_ACTING_TOOLS
    assert raises_capability("active", "singularity") is True  # gains code execution
    assert raises_capability("passive", "singularity") is True
    assert raises_capability("live", "singularity") is True  # legacy name for active
    assert raises_capability("active", "passive") is False
    assert raises_capability("singularity", "passive") is False
    assert raises_capability("singularity", "active") is False
    assert raises_capability("passive", "passive") is False
    assert raises_capability("passive", "no-such-mode") is True  # unknown TARGET: fail closed
    # "agentic" (the robot runtime's run mode) is active-class since #829; an UNKNOWN current mode is
    # judged as passive, which is what dispatch enforces for it
    assert raises_capability("agentic", "passive") is False
    assert raises_capability("agentic", "active") is False
    assert raises_capability("no-such-mode", "active") is True
    assert raises_capability("agentic", "singularity") is True


def test_an_agentic_run_may_lower_itself() -> None:
    tool, modes = _tool("agentic")
    assert tool.execute(mode="passive").success is True
    assert modes.set_calls == ["passive"]
    tool, modes = _tool("agentic")
    assert tool.execute(mode="singularity").success is False and modes.set_calls == []


def test_a_legacy_alias_of_the_current_mode_is_already_there() -> None:
    tool, modes = _tool("live")  # legacy name for active
    result = tool.execute(mode="active")
    assert result.success is True and result.metadata["was_change"] is False
    assert modes.set_calls == []


def _synthetic_modes(monkeypatch, **variants):
    """Register synthetic modes derived from passive, so each arm of `raises_capability` is pinned alone."""
    import dataclasses

    from maxim.modes import definitions

    base = dataclasses.replace(definitions.get_mode("passive"), forbidden_tools=set(), can_act_on_host=True)
    for name, changes in variants.items():
        monkeypatch.setitem(definitions.OPERATIONAL_MODES, name, dataclasses.replace(base, name=name, **changes))


def test_gaining_only_network_access_is_a_raise(monkeypatch) -> None:
    from maxim.modes.definitions import raises_capability

    _synthetic_modes(monkeypatch, zz_offline={"can_access_network": False}, zz_online={"can_access_network": True})
    assert raises_capability("zz-offline", "zz-online") is True
    assert raises_capability("zz-online", "zz-offline") is False


def test_dropping_only_the_confirmation_requirement_is_a_raise(monkeypatch) -> None:
    from maxim.modes.definitions import raises_capability

    _synthetic_modes(
        monkeypatch, zz_confirm={"confirmations_required": True}, zz_unconfirmed={"confirmations_required": False}
    )
    assert raises_capability("zz-confirm", "zz-unconfirmed") is True
    assert raises_capability("zz-unconfirmed", "zz-confirm") is False


def test_lifting_only_a_forbidden_tool_is_a_raise(monkeypatch) -> None:
    from maxim.modes.definitions import raises_capability

    # a tool in NO capability set, so only the forbidden-tools arm can see it
    _synthetic_modes(monkeypatch, zz_strict={"forbidden_tools": {"zz_uncategorised_tool"}}, zz_lax={})
    assert raises_capability("zz-strict", "zz-lax") is True
    assert raises_capability("zz-lax", "zz-strict") is False


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
    from maxim.modes.definitions import executes_code

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


def test_cli_seam_refuses_a_code_executing_mode() -> None:
    from maxim.cli import _runtime_mode_switch_allowed

    assert _runtime_mode_switch_allowed("singularity", current_operational="singularity", granted=None) is False
    assert _runtime_mode_switch_allowed("active", current_operational="active", granted=None) is True
    assert _runtime_mode_switch_allowed("live", current_operational="active", granted=None) is True
    assert _runtime_mode_switch_allowed("no-such-mode", current_operational="active", granted=None) is False


# ── #827: no approval surface means no silent "pending"; a mode with nowhere to apply is a failure ──


def test_an_autonomy_request_with_no_approver_is_unavailable_not_pending() -> None:
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel

    controller = AutonomyController(initial_level=AutonomyLevel.PLANNING)
    request = controller.request_autonomy(AutonomyLevel.AUTONOMOUS, justification="let me")
    assert request.status == "unavailable"
    assert controller.get_pending_requests() == []
    assert controller.current_level == AutonomyLevel.PLANNING


def test_an_attached_surface_sees_the_request_and_can_grant_it() -> None:
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel

    controller = AutonomyController(initial_level=AutonomyLevel.PLANNING)
    shown = []
    controller.set_approval_surface(shown.append)
    request = controller.request_autonomy(AutonomyLevel.SUPERVISED, justification="need to write")
    assert shown == [request] and request.status == "pending"
    assert controller.approve_autonomy_request(request) is True
    assert controller.current_level == AutonomyLevel.SUPERVISED


def test_the_level_tool_says_no_approver_instead_of_awaiting_approval(monkeypatch) -> None:
    import maxim.simulation.sim_logger as sim_logger
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.tools.mode_switch import AutonomyLevelTool

    monkeypatch.setattr(sim_logger, "should_prompt", lambda *a, **k: True)  # the interactive branch
    controller = AutonomyController(initial_level=AutonomyLevel.PLANNING)
    result = AutonomyLevelTool(controller).execute(level="autonomous", reason="trust me")
    assert result.success is False
    assert "no human approver" in (result.error or "")
    assert "awaiting approval" not in (result.output or "")
    assert controller.get_pending_requests() == []


def test_a_mode_switch_with_no_runtime_to_apply_it_fails() -> None:
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.runtime.bootstrap import build_tool_registry

    class _NoRuntime:  # a mode to read, but no `requested_mode` to apply a change to
        mode = "active"

    controller = AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    registry = build_tool_registry(maxim=_NoRuntime(), autonomy_controller=controller)
    # a LOWERING switch, so the #924 gate lets it through and the #827 path is what refuses it
    result = registry.get("mode_switch").execute(mode="passive")
    assert result.success is False
    assert "no runtime" in (result.error or "")
    assert not [e for e in controller.get_audit_log() if e.action_type == "executed"]


def test_a_mode_switch_with_a_runtime_still_applies() -> None:
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.runtime.bootstrap import build_tool_registry

    class _Maxim:
        mode = "passive"
        requested_mode = None

    maxim = _Maxim()
    maxim.mode = "active"  # a LOWERING switch: under #924 only those are self-grantable
    registry = build_tool_registry(
        maxim=maxim, autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    )
    result = registry.get("mode_switch").execute(mode="passive")
    assert result.success is True and maxim.requested_mode == "passive"


def test_real_registry_wiring_refuses_passive_to_active() -> None:
    """#924 end to end through `build_tool_registry`: the tool the agent actually gets refuses the raise."""
    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.runtime.bootstrap import build_tool_registry

    class _Maxim:
        mode = "passive"
        requested_mode = None

    maxim = _Maxim()
    registry = build_tool_registry(
        maxim=maxim, autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS)
    )
    result = registry.get("mode_switch").execute(mode="active")
    assert result.success is False
    assert maxim.requested_mode is None


def test_the_unread_mode_transition_policy_is_gone() -> None:
    """``allowed_mode_transitions`` was declared and never read -- a policy that looked enforced."""
    from maxim.agents.autonomy import SupervisionPolicy

    assert not hasattr(SupervisionPolicy(), "allowed_mode_transitions")
