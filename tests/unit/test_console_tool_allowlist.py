"""The console's hard tool allowlist — ``tools.allow`` / ``tools.deny`` →
``AgentPermissions`` → ``Executor`` (1.1.3).

Before this landed, ``AgentPermissions`` and the Executor's gate both
existed, but no console path built one: ``create_full_agent`` called
``build_executor`` without ``permissions=``, so every ``MaximHandle``
agent ran with ``Executor._permissions is None`` and an operator had no
way to keep ``read_file`` / ``bash`` away from a chat agent. These pin
the seam end-to-end with the heavy parts (bio-stack, LLM) mocked: the
handle resolves the setting, puts it on ``AgentConfig``, the factory
passes it to ``build_executor``, and the executor refuses the call.
"""

from __future__ import annotations

import pytest

from maxim.tools.base import Tool


class _Tool(Tool):
    """Minimal registrable tool for the roster tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.description = f"{name} (test tool)"
        super().__init__()

    def execute(self, **kwargs):  # noqa: ANN003
        return "ok"


def _build_handle(monkeypatch, tmp_path):
    """A MaximHandle whose factory step is faked down to build_executor.

    The fake keeps the real seam under test — it builds the executor the
    way the real factory does (``permissions=config.permissions``) over
    the handle's REAL tool registry — and skips the bio-stack.
    """
    from maxim.runtime import agent_factory
    from maxim.runtime.bootstrap import build_executor

    captured: dict = {}

    def fake_create_full_agent(self, config, *, tool_registry=None, pain_bus=None, fear_llm=None, auto_load=False):
        captured["config"] = config
        executor = build_executor(tool_registry, pain_bus=None, permissions=config.permissions)
        return agent_factory.AgentInstance(
            agent_id=config.agent_id,
            role=config.role,
            config=config,
            tool_registry=tool_registry,
            executor=executor,
        )

    monkeypatch.setattr(agent_factory.AgentFactory, "create_full_agent", fake_create_full_agent)
    monkeypatch.chdir(tmp_path)  # build_tool_registry(active) ensures a workspace under CWD

    from maxim.console.handle import MaximHandle

    handle = MaximHandle(agent_id="console_agent", home=tmp_path / "home")
    return handle, captured


class TestConsoleToolAllowlist:
    def test_unconfigured_leaves_the_gate_off(self, monkeypatch, tmp_path):
        handle, captured = _build_handle(monkeypatch, tmp_path)
        assert captured["config"].permissions is None
        assert handle.instance.executor._permissions is None
        assert "read_file" in handle.instance.tool_registry.list()

    def test_env_allow_list_refuses_read_file_at_the_executor(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAXIM_TOOLS_ALLOW", "respond,speak")
        handle, captured = _build_handle(monkeypatch, tmp_path)

        perms = captured["config"].permissions
        assert perms is not None
        assert perms.tool_allow == frozenset({"respond", "speak"})
        assert handle.instance.executor._permissions is perms

        result = handle.instance.executor.execute({"tool_name": "read_file", "params": {"path": "x"}})
        assert result.success is False
        assert "allow-list" in (result.error or "")
        # The tools the agent needs to reply are still admitted.
        assert perms.can_invoke_tool("respond")[0] is True
        assert perms.can_invoke_tool("speak")[0] is True

    def test_env_deny_list_refuses_named_tool_only(self, monkeypatch, tmp_path):
        monkeypatch.setenv("MAXIM_TOOLS_DENY", "bash")
        handle, _ = _build_handle(monkeypatch, tmp_path)
        executor = handle.instance.executor
        denied = executor.execute({"tool_name": "bash", "params": {"command": "true"}})
        assert denied.success is False
        assert "denied" in (denied.error or "")
        assert executor._permissions.can_invoke_tool("read_file")[0] is True

    def test_config_json_allow_list_reaches_the_executor(self, monkeypatch, tmp_path):
        """The config.json path (``maxim config set tools.allow ...``), not
        just the env override."""
        import json

        from maxim.runtime.config_loader import config_path, reset_config_cache

        # Written as a file, not through `config_writer.set_field`: that
        # writer has a CI-enforced caller allow-list (config_unification IM2),
        # and the round-trip of `maxim config set` itself is pinned in
        # tests/unit/test_config_writer.py. This test is about the LOADER
        # → handle → Executor path.
        cp = config_path()
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps({"tools": {"allow": ["respond"]}}))
        reset_config_cache()
        handle, captured = _build_handle(monkeypatch, tmp_path)
        assert captured["config"].permissions.tool_allow == frozenset({"respond"})
        result = handle.instance.executor.execute({"tool_name": "read_file", "params": {"path": "x"}})
        assert result.success is False
        assert "allow-list" in (result.error or "")

    def test_kind_selector_admits_generated_affordances(self, monkeypatch, tmp_path):
        """``kind:sem-modulator-derived`` is the only way to allow the
        per-entity affordance tools an embodiment generates at build time."""
        from maxim.tools.base import Tool, ToolOutput

        class _Affordance(Tool):
            name = "reachy_mini_head_yaw_turn_left"
            description = "generated affordance"
            input_schema: dict = {}
            kind = "sem-modulator-derived"

            def execute(self, **kwargs):
                return ToolOutput(success=True, output="turned")

        monkeypatch.setenv("MAXIM_TOOLS_ALLOW", "respond,kind:sem-modulator-derived")
        handle, _ = _build_handle(monkeypatch, tmp_path)
        handle.instance.tool_registry.register(_Affordance())
        executor = handle.instance.executor
        assert executor.execute({"tool_name": "reachy_mini_head_yaw_turn_left", "params": {}}).success is True
        assert executor.execute({"tool_name": "read_file", "params": {"path": "x"}}).success is False


class TestRosterAdvertisesOnlyPermittedTools:
    """Bugs ledger D82: the gate used to act at dispatch only, so the prompt
    kept advertising tools the executor would refuse."""

    def _executor(self, permissions):
        from maxim.agents.permissions import AgentPermissions
        from maxim.runtime.bootstrap import build_executor
        from maxim.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(_Tool("respond"))
        registry.register(_Tool("read_file"))
        return build_executor(
            registry,
            pain_bus=None,
            permissions=AgentPermissions(**permissions) if permissions is not None else None,
        )

    def test_permits_mirrors_the_dispatch_gate(self):
        from maxim.agents.permissions import AgentPermissions

        ex = self._executor({"tool_allow": frozenset({"respond"})})
        assert isinstance(ex._permissions, AgentPermissions)
        assert ex.permits("respond") is True
        assert ex.permits("read_file") is False
        denied = ex.execute({"tool_name": "read_file", "params": {}})
        assert not denied.success

    def test_no_permissions_permits_everything(self):
        ex = self._executor(None)
        assert ex.permits("respond") and ex.permits("read_file") and ex.permits("never_registered")

    def test_agent_loop_filters_the_advertised_roster_through_permits(self):
        import inspect

        from maxim.runtime import agent_loop

        src = inspect.getsource(agent_loop.run_agentic_loop)
        assert "available_tools = [t for t in available_tools if executor.permits(t)]" in src
        # The filter runs AFTER the roster is final (SEM union + Wire 3) and
        # BEFORE the descriptions are collected for the prompt.
        assert src.index("executor.permits(t)") < src.index("last_surfaced_tools = list(available_tools)")


class TestSimAgentsUntouched:
    def test_orchestrator_agent_configs_carry_no_permissions(self):
        """The sim AUT / orchestrator build their own ``AgentConfig`` without
        ``permissions=`` — the console allowlist must not leak into them."""
        import inspect

        from maxim.simulation import orchestrator

        src = inspect.getsource(orchestrator)
        assert "permissions=tool_permissions_from_settings" not in src
        assert "MAXIM_TOOLS_ALLOW" not in src


@pytest.mark.parametrize("env_name", ["MAXIM_TOOLS_ALLOW", "MAXIM_TOOLS_DENY"])
def test_env_vars_are_scrubbed_between_tests(env_name):
    import os

    assert env_name not in os.environ
