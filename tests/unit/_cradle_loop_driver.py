"""Scripted cradle sequence through the REAL loop and capture (grounding GL2a, autonomic_layer.md §5.2).

No harness drove a FIXED affordance sequence on a cradle body through ``run_agentic_loop`` before GL2a:
``_loop_harness.py`` drives the scripted Minecraft world, whose actions the substrate chooses. This
driver runs the real loop substrate-primary on the canonical builders (``build_bio_stack`` +
``build_executor`` with the AUT's ``agent_id``), with the substrate proposer replaced by one that
returns the script, one proposal at a time (the ``_execute_learn_driver.py`` seam). Every action goes
``run_agentic_loop`` -> ``tool_dispatch.execute_and_learn`` -> ``capture_loop_action``, and the run
returns what the REAL Hippocampus captured, in capture order: never a hand-composed capture.

Time is ``_loop_harness._StepClock`` (the GLOBAL ``time`` module, so ``embodiment/body.py``'s drift
reads it). The loop paces at ``TARGET_HZ``, so a tick advances the clock by ``1 / TARGET_HZ`` (100 us:
the smallest step that still moves ``time.time()`` at the clock's epoch, which the substrate cadence
needs), so drift between scripted actions stays below every tolerance the tests use. A step may ask
for ``advance_s`` seconds, spent on the loop thread when its proposal is made, i.e. between the loop
tick and the tool call.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.unit._loop_harness import _StepClock, assert_this_checkout, preimport_loop_graph

AGENT_ID = "cradle_gl2a"
TARGET_HZ = 1e4
CLUSTERS = {"interoception": "c_intero_gl2a"}

# The modules a cradle run imports lazily, beyond the Minecraft graph ``preimport_loop_graph`` loads.
_PREIMPORT_CRADLE: tuple[str, ...] = (
    "maxim.agents.llm_types",
    "maxim.memory.encoding",
    "maxim.memory.hippocampus",
    "maxim.runtime.bio_integration",
    "maxim.runtime.tool_dispatch",
    "maxim.runtime.loop_setup",
    "maxim.runtime.substrate_proposal",
    "maxim.proprioception.pain",
)


@dataclass(frozen=True)
class Step:
    tool: str
    advance_s: float = 0.0


@dataclass
class CradleRun:
    proposed: list[str] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)
    outputs: list[Any] = field(default_factory=list)  # the executor's ToolOutput per executed step
    traces: list[Any] = field(default_factory=list)  # EpisodicMemory, capture order
    tool_names: list[str] = field(default_factory=list)  # every tool the run registered
    embodiment: Any = None
    hippocampus: Any = None  # the run's real Hippocampus (its traces are ``traces``)


def run_cradle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    body_ref: str,
    entity_refs: tuple[str, ...],
    script: tuple[Step, ...],
    initial: dict[str, float] | None = None,
) -> CradleRun:
    """Run ``script`` on ``body_ref`` with ``entity_refs`` in the scene; ``initial`` sets drive values
    (bare or qualified names) on the body before the first tick."""
    import importlib

    assert_this_checkout()
    preimport_loop_graph()
    for name in _PREIMPORT_CRADLE:
        importlib.import_module(name)

    from maxim.agents.autonomy import AutonomyController, AutonomyLevel
    from maxim.agents.llm_types import LLMProposal
    from maxim.agents.maxim_agent import MaximAgent
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.embodiment.sem import _resolve_sensor_slot
    from maxim.environment.filesystem_env import FileSystemEnv
    from maxim.runtime import agent_loop as AL
    from maxim.runtime import substrate_proposal
    from maxim.runtime.bio_stack import build_bio_stack
    from maxim.runtime.bootstrap import build_decision_engine, build_executor, build_memory
    from maxim.runtime.loop_controller import LoopController
    from maxim.runtime.state import RuntimeState
    from maxim.similarity import ec as ec_mod
    from maxim.simulation import sim_logger
    from maxim.tools.registry import ToolRegistry

    run = CradleRun()
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    counter = iter(range(1, 1_000_000))
    monkeypatch.setattr(ec_mod, "uuid4", lambda: uuid.UUID(int=next(counter)))
    monkeypatch.setattr(sim_logger, "_interactive_mode", sim_logger.InteractiveMode.OFF)

    clock = _StepClock()
    components = ComponentRegistry()
    bio = build_bio_stack(agent_id=AGENT_ID, persistence_dir=str(tmp_path / AGENT_ID))
    registry = ToolRegistry()
    executor = build_executor(
        tool_registry=registry,
        permissions=None,
        agent_id=AGENT_ID,
        pain_bus=bio.pain_bus,
        nac=bio.nac,
        hippocampus=bio.hippocampus,
        scn=bio.scn,
        cerebellum=bio.cerebellum,
        distributor=bio.distributor,
        entity_ref=body_ref,
        component_registry=components,
    )
    run.embodiment = executor.embodiment
    assert run.embodiment is not None and run.embodiment.agent_id == AGENT_ID
    for ref in entity_refs:
        executor.generate_entity_tools(components.instantiate(ref))
    run.tool_names = sorted(registry.list_all())
    for step in script:
        assert step.tool in run.tool_names, f"{step.tool!r} not registered; have {run.tool_names}"
    root = run.embodiment.root
    for name, value in (initial or {}).items():
        slot = _resolve_sensor_slot(root, name)
        assert slot is not None, f"the body has no sensor {name!r}"
        slot[0][slot[1]] = float(value)

    proposed: list[int] = []

    def _propose(**_kw: Any) -> Any:
        if len(proposed) >= len(script):
            return None
        step = script[len(proposed)]
        proposed.append(1)
        run.proposed.append(step.tool)
        if step.advance_s:
            clock.sleep(step.advance_s)  # the loop thread: between the loop tick and the tool call
        return LLMProposal(
            request_id=f"gl2a-{len(proposed)}",
            action={"tool_name": step.tool, "params": {}},
            reasoning="cradle script",
            strategy_used="substrate-primary",
            confidence=0.9,
            mode_goal_achieved=False,
            triggering_input="cradle script",
            cluster_id=CLUSTERS["interoception"],
            clusters=dict(CLUSTERS),
        )

    monkeypatch.setattr(substrate_proposal, "propose_via_substrate", _propose)
    real_init = LoopController.__init__

    def _init(self: Any, *a: Any, **k: Any) -> None:
        real_init(self, *a, **k)
        self.llm_submit_interval = 0.0  # propose on every tick the clock has moved

    monkeypatch.setattr(LoopController, "__init__", _init)

    real_execute = executor.execute

    def _execute(action: Any) -> Any:
        result = real_execute(action)
        run.executed.append((action or {}).get("tool_name"))
        run.outputs.append(result)
        if len(run.executed) >= len(script):
            stop.set()
        return result

    monkeypatch.setattr(executor, "execute", _execute)
    stop = threading.Event()
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    state = RuntimeState()
    state.data["mode"] = "active"
    hub, hippo = bio.memory_hub, bio.hippocampus
    agent = MaximAgent()
    agent.wire_memory_hub(hub)
    clock.install()
    try:
        AL.run_agentic_loop(
            agent,
            FileSystemEnv(str(workspace)),
            state,
            build_memory(),
            build_decision_engine(),
            executor,
            autonomy_controller=AutonomyController(initial_level=AutonomyLevel.AUTONOMOUS),
            pain_bus=bio.pain_bus,
            memory_hub=hub,
            hippocampus=hippo,
            aut_mode="substrate-primary",
            max_steps=4 * len(script) + 8,
            target_hz=TARGET_HZ,
            idle_sleep_s=0.0,
            stop_event=stop,
        )
    finally:
        clock.uninstall()
    assert hippo.flush(timeout=30.0), "the Hippocampus capture worker did not drain"
    traces = [m for m in hippo._memories.values() if getattr(m, "capture_seq", None) is not None]
    run.traces = sorted(traces, key=lambda m: m.capture_seq)
    run.hippocampus = hippo
    try:
        bio.on_session_end()
    except Exception as exc:  # teardown only; the traces are already read
        print(f"bio-stack session end raised: {exc!r}")
    return run
