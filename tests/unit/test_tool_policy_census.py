"""#1315 item 3 census (approach notes §3 "The census tests"): every place ``src/maxim`` builds or bypasses a dispatch.

An AST scan over ``src/maxim`` (``scripts/`` is out of the census; ``scripts/orient_substrate``'s direct
``tools[action].execute()`` calls are named out of scope here). It pins:

1. every ``Executor(`` / ``build_executor(`` / ``AgentConfig(`` call passes ``policy=``;
2. every caller of the permissive factories ``ToolPolicy.unrestricted(`` and ``ToolPolicy.deny(``, with its reason;
3. every ``ToolRegistry(`` / ``SimToolRegistry(`` built outside ``build_tool_registry``;
4. every direct ``.execute(`` / ``.run(`` on a tool outside ``Executor`` (receivers named ``*tool*``, or the result
   of a registry ``get``/``find``).

A new site fails the matching test until it is added to the pinned set with its reason. The callers of the helpers that
return a permissive policy are pinned too, and ``ToolPolicy`` is never aliased. Stated blind spots: ``getattr``,
``functools.partial``, ``executor.submit``, ``**kwargs`` splats, subscript receivers (``tools[name].execute``) and
receivers named neither ``*tool*`` nor in ``KNOWN_TOOL_RECEIVERS``; the dispatch-time check is the backstop.

Regression guard for the item 3 invariants in ``docs/agents/runtime-tools.md``.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "maxim"

PERMISSIVE_CALLERS: dict[tuple[str, str, str], str] = {
    ("api.py", "_build_run_executor", "unrestricted"): "maxim.run: the stable verb keeps ceiling ALL (V10)",
    (
        "embodied_runtime/agentic_runtime.py",
        "AgenticRuntimeMixin._build_runtime_executor",
        "unrestricted",
    ): "the robot's own agent: gated by the run mode, the launch grant and the autonomy forbids",
    ("runtime/tool_policy.py", "policy_from_tools_settings", "deny"): "the console's tools.allow / tools.deny",
    ("runtime/tool_policy.py", "cli_tool_policy", "unrestricted"): "the CLI agent: the operator's own",
    ("runtime/tool_policy.py", "aut_tool_policy", "unrestricted"): "the sim AUT: measured on its full roster",
    ("runtime/tool_policy.py", "orchestrator_tool_policy", "unrestricted"): "the narrator drives the AUT (V12)",
    ("simulation/foundry.py", "run_gauntlet", "unrestricted"): "the gauntlet probes every affordance of its entity",
    ("simulation/minecraft_harness.py", "build_minecraft_aut", "unrestricted"): "the AUT, inside its own sandbox",
}

REGISTRIES_OUTSIDE_THE_BUILDER: dict[tuple[str, str, str], str] = {
    ("embodiment/tool_bridge.py", "derive_capability_map", "ToolRegistry"): "a throwaway map; nothing dispatches it",
    ("runtime/agent_factory.py", "AgentFactory._create_tool_registry", "ToolRegistry"): "empty until the runtime "
    "registers tools; dispatched only through create_full_agent's executor",
    ("simulation/foundry.py", "run_sem_protocol_tests", "ToolRegistry"): "protocol checks of one entity's tools",
    ("simulation/foundry.py", "run_gauntlet", "ToolRegistry"): "the gauntlet's entity tools, under its executor",
    ("simulation/minecraft_harness.py", "build_minecraft_aut", "ToolRegistry"): "the AUT's, under its executor",
    ("simulation/orchestrator.py", "start_simulation_mode", "SimToolRegistry"): "the narrator's, under its "
    "executor (orchestrator_tool_policy)",
}

# The census-pinned helpers that RETURN a permissive policy (each one is in PERMISSIVE_CALLERS above): a call to one
# is a permissive construction too, so its callers are pinned as well (architecture review A2).
PERMISSIVE_HELPERS = frozenset(
    {"cli_tool_policy", "aut_tool_policy", "orchestrator_tool_policy", "policy_from_tools_settings"}
)

PERMISSIVE_HELPER_CALLERS: dict[tuple[str, str, str], str] = {
    ("cli.py", "_cli_agent_config", "cli_tool_policy"): "the CLI agent's AgentConfig",
    ("simulation/orchestrator.py", "_aut_agent_config", "aut_tool_policy"): "the sim AUT's AgentConfig",
    ("simulation/orchestrator.py", "_orchestrator_agent_config", "orchestrator_tool_policy"): "the narrator's",
    ("console/handle.py", "MaximHandle.__init__", "policy_from_tools_settings"): "the console's tools settings",
}

DIRECT_TOOL_CALLS: dict[tuple[str, str, str], str] = {
    ("runtime/executor.py", "Executor._run_started", "tool.run"): "THE dispatch: runs the object the gates judged",
    ("runtime/loop_perception.py", "_checked_call", "tool.execute"): "auto_sense, after executor.check (V6)",
    ("integration/bio_enrichment.py", "BioEnrichmentPipeline._dispatch_reflex_tool", "tool.execute"): "the sim "
    "world's body reflexes (damage_component, set_entity_sensor): the world acting on its own body, no agent "
    "executor in the pipeline",
    ("simulation/tools.py", "OrchestratorActorTool.execute", "eph_tool.execute"): "the narrator's actor composite: "
    "runs inside the narrator's own dispatch (its tools are ALL, V12)",
    ("runtime/loop_perception.py", "auto_sense", "_sense.execute"): "auto_sense's interoception call, after "
    "executor.check (V6)",
}

#: Receiver names a direct call is recognised by beyond ``*tool*`` (wire review F12, executor review E9).
KNOWN_TOOL_RECEIVERS = frozenset({"_sense"})


def _calls() -> list[tuple[str, str, ast.Call]]:
    out: list[tuple[str, str, ast.Call]] = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))

        def walk(node: ast.AST, stack: list[str], rel: str = rel) -> None:
            for child in ast.iter_child_nodes(node):
                inner = stack
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    inner = [*stack, child.name]
                if isinstance(child, ast.Call):
                    out.append((rel, ".".join(inner) or "<module>", child))
                walk(child, inner)

        walk(tree, [])
    return out


def _name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return None


def test_every_executor_and_agent_config_passes_policy() -> None:
    missing = [
        f"{rel}::{where} {_name(call.func)}(...) line {call.lineno}"
        for rel, where, call in _calls()
        if _name(call.func) in ("Executor", "build_executor", "AgentConfig")
        and not any(k.arg == "policy" for k in call.keywords)
    ]
    assert not missing, f"construction sites without policy= (V3): {missing}"


def test_permissive_policy_callers_are_pinned() -> None:
    found = {
        (rel, where, call.func.attr)
        for rel, where, call in _calls()
        if isinstance(call.func, ast.Attribute)
        and call.func.attr in ("unrestricted", "deny")
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "ToolPolicy"
    }
    assert found == set(PERMISSIVE_CALLERS), (
        f"unpinned: {sorted(found - set(PERMISSIVE_CALLERS))}; stale pins: {sorted(set(PERMISSIVE_CALLERS) - found)}"
    )


def test_permissive_helper_callers_are_pinned() -> None:
    found = {
        (rel, where, _name(call.func) or "")
        for rel, where, call in _calls()
        if _name(call.func) in PERMISSIVE_HELPERS and not rel.startswith("runtime/tool_policy.py")
    }
    pinned = set(PERMISSIVE_HELPER_CALLERS)
    assert found == pinned, f"unpinned: {sorted(found - pinned)}; stale pins: {sorted(pinned - found)}"


def test_tool_policy_is_never_aliased() -> None:
    """The factory scan matches ``ToolPolicy.<factory>`` and the permissive helpers by name, so neither is imported
    under another name, ``ToolPolicy`` or one of its factories is never assigned to another name
    (``U = ToolPolicy.unrestricted``), and ``ToolPolicy`` is never reached as a module attribute. Stated blind spots
    (micro-round R1): an annotated or tuple assignment of a factory, and a helper bound to a plain name; none occurs
    in ``src/`` today, and the dispatch-time check is the backstop."""
    sites = []
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                sites += [
                    f"{rel}: import {a.name} as {a.asname}"
                    for a in node.names
                    if a.asname and a.name in {"ToolPolicy", *PERMISSIVE_HELPERS}
                ]
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id == "ToolPolicy":
                sites.append(f"{rel}: assigned at line {node.lineno}")
            elif (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "ToolPolicy"
            ):
                sites.append(f"{rel}: ToolPolicy.{node.value.attr} assigned at line {node.lineno}")
            elif isinstance(node, ast.Attribute) and node.attr == "ToolPolicy":
                sites.append(f"{rel}: attribute access at line {node.lineno}")
    assert not sites, f"ToolPolicy reached under another name: {sites}"


def test_registries_outside_the_builder_are_pinned() -> None:
    found = {
        (rel, where, _name(call.func) or "")
        for rel, where, call in _calls()
        if _name(call.func) in ("ToolRegistry", "SimToolRegistry") and where != "build_tool_registry"
    }
    pinned = set(REGISTRIES_OUTSIDE_THE_BUILDER)
    assert found == pinned, f"unpinned: {sorted(found - pinned)}; stale pins: {sorted(pinned - found)}"


def _tool_receiver(call: ast.Call) -> bool:
    receiver = call.func.value if isinstance(call.func, ast.Attribute) else None
    if isinstance(receiver, ast.Call):
        return _name(receiver.func) in ("get", "find")
    name = _name(receiver) if receiver is not None else None
    return name is not None and ("tool" in name.lower() or name in KNOWN_TOOL_RECEIVERS)


def test_direct_tool_calls_outside_the_executor_are_pinned() -> None:
    found = {
        (rel, where, ast.unparse(call.func))
        for rel, where, call in _calls()
        if isinstance(call.func, ast.Attribute) and call.func.attr in ("execute", "run") and _tool_receiver(call)
    }
    pinned = set(DIRECT_TOOL_CALLS)
    assert found == pinned, f"unpinned: {sorted(found - pinned)}; stale pins: {sorted(pinned - found)}"


def test_every_pin_has_a_reason() -> None:
    for table in (PERMISSIVE_CALLERS, PERMISSIVE_HELPER_CALLERS, REGISTRIES_OUTSIDE_THE_BUILDER, DIRECT_TOOL_CALLS):
        assert all(isinstance(r, str) and len(r) > 10 for r in table.values())
