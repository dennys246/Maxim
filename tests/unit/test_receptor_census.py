"""The receptor census: every live percept, sensor, pain and transduction site (grounding GL3.B0).

``docs/plans/thalamic_relay.md`` §2 (every percept source), §3.3 (every pain ingress) and §3.2 (who runs
``Embodiment.evaluate_failures``, and on which thread), as a CHECKED-IN TABLE. The scan below walks
``src/maxim`` with ``ast`` and finds every site in eight categories; the test fails when a site appears
that the table does not name (a new, unregistered producer) or the table names a site that is gone (a
stale census). The table is authoritative; §2 and §3.3 summarize it.

Each row is ``site -> (what it is, liveness, threads)``. A site is ``<path under src/>::<qualified
function>``, the function that HOLDS the call. Liveness is ``live``, ``dormant`` (its module or builder
says so), ``no live caller``, ``internal`` (a conversion inside the bus) or ``not the EC``. Threads name
who runs the site in a live run: ``loop`` (the agent loop's thread; ``sim.aut`` under ``--sim``),
``orchestrator`` (the ``start_simulation_mode`` caller running the orchestrator agent's loop, ``--sim``),
``sim.dm`` (interactive DM campaigns), ``sim.stdin`` / ``sim.stall`` (the human edge and the stall
watchdog), ``main`` (the generative / cradle-mother and fixture runners after ``sim.aut`` starts),
``doa-feed``, ``mc-sync`` (the Minecraft sync pump), ``console`` (a console request), ``pool-worker``
(``AgentPool``'s concurrent round), ``comms`` (a messaging webhook) and ``caller`` (a library entry point).

STATED LIMITS (GL3.B0 review). The scan checks that each SITE exists, nothing more: the liveness and
thread columns are code-read and hand-asserted, and a new caller ONE HOP UP (a new caller of
``inject_cli``, a hook assigned to an attribute, a new thread calling ``evaluate_failures`` through an
existing site) is invisible to it. Matching is by name: an aliased import, ``getattr`` or ``partial``
would be missed (none exists in ``src/`` today), and so would a direct ``vital_metrics[...] =`` write
outside the named writers (the Dormant ``CerebellumModulator`` has one). ``scripts/`` is out of scope: its in-process producers
drive their own offline stacks and none injects into a running loop (checked 2026-10-09).
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
FACTORY_MODULE = "maxim/agents/percept_factory.py"
FACTORIES = frozenset({"make_text_percept", "make_scene_percept", "make_intero_percept", "make_audio_percept"})
BODY_WRITERS = frozenset(
    {"_apply_sensor_deltas", "world_set_axis", "world_set_azimuth", "_write_sensor", "apply_damage"}
)

LIVENESS = frozenset({"live", "dormant", "no live caller", "internal", "not the EC"})
THREADS = frozenset(
    {
        "loop",
        "orchestrator",
        "sim.dm",
        "sim.stdin",
        "sim.stall",
        "main",
        "doa-feed",
        "mc-sync",
        "console",
        "pool-worker",
        "comms",
        "caller",
    }
)

_T = frozenset
_NARRATOR = _T({"orchestrator", "loop"})  # the narrator's instance, and the reflex dispatch's on the loop
_EVALUATORS = _T({"loop", "orchestrator", "caller"})  # whoever calls evaluate_failures

# category -> site -> (what it is, liveness, threads)
CENSUS: dict[str, dict[str, tuple[str, str, frozenset[str]]]] = {
    "percept_factory_call": {
        "maxim/comms/conversation.py::ConversationManager._archive": ("Messaging channels", "live", _T({"comms"})),
        "maxim/comms/conversation.py::ConversationManager.process_inbound": (
            "Messaging channels",
            "live",
            _T({"comms"}),
        ),
        "maxim/comms/gateway.py::CommunicationGateway.receive_inbound": ("Messaging channels", "live", _T({"comms"})),
        "maxim/embodiment/audio_localization.py::AzimuthDoASource.next_percept": ("Sim DoA", "live", _T({"loop"})),
        "maxim/embodiment/audio_localization.py::DoAFeed.run": (
            "Reachy DoA (live), percept lane",
            "live",
            _T({"doa-feed"}),
        ),
        "maxim/embodiment/percepts.py::EmbodimentPerceptSource.next_percept": (
            "Body percept source (protocol template)",
            "dormant",
            _T({"loop"}),
        ),
        "maxim/runtime/agent_loop.py::run_agentic_loop": (
            "Observation text (CLI / voice / environment) as an ABSTRACT percept",
            "live",
            _T({"loop"}),
        ),
        "maxim/runtime/agent_pool.py::AgentPool.run_turn": (
            "AgentPool turn text",
            "live",
            _T({"caller", "pool-worker"}),
        ),
        "maxim/simulation/conversational_source.py::ConversationalSource.inject_cli": (
            "Text percepts: narrator / DM / human / runners into the AUT's source; the stall nudge into the "
            "orchestrator's own",
            "live",
            _T({"orchestrator", "sim.dm", "sim.stdin", "sim.stall", "main", "console"}),
        ),
        "maxim/simulation/conversational_source.py::ConversationalSource.inject_pain": (
            "Sim pain injection (an INTEROCEPTION percept; the Reaction is made on the loop by sim_adapter)",
            "live",
            _T({"orchestrator", "main"}),
        ),
        "maxim/simulation/conversational_source.py::ConversationalSource.inject_sensor": (
            "Sim sensor injection",
            "no live caller",
            _T(),
        ),
        "maxim/simulation/minecraft.py::MinecraftPerceptSource.next_percept": (
            "Minecraft game events",
            "live",
            _T({"loop"}),
        ),
    },
    "raw_percept": {
        "maxim/agents/perception_agent.py::PerceptionAgent._on_captured_frame": (
            "Vision (DN / robot)",
            "live",
            _T({"caller"}),
        ),
        "maxim/agents/perception_agent.py::PerceptionAgent.process_captured_frame": (
            "Vision (DN / robot)",
            "live",
            _T({"caller"}),
        ),
        "maxim/agents/perception_agent.py::PerceptionAgent.process_observation": (
            "The per-pass percept of every MaximAgent loop (CLI / transcript text, vision when detections "
            "exist), published synchronously to the memory agent",
            "live",
            _T({"loop", "orchestrator"}),
        ),
        "maxim/simulation/scenario_source.py::_percept_from_dict": ("Scenario fixture percepts", "live", _T({"loop"})),
    },
    "sensor_encode": {
        "maxim/runtime/substrate_proposal.py::_encode_current_clusters": (
            "Body channels (interoception / audio / world), outcome-time encode",
            "live",
            _T({"loop"}),
        ),
        "maxim/runtime/substrate_proposal.py::propose_via_substrate": (
            "Body channels (interoception / audio / world), the substrate tick",
            "live",
            _T({"loop"}),
        ),
    },
    "linguistic_encode": {
        "maxim/embodiment/component_index.py::ComponentIndex._embed": (
            "ComponentIndex's own sentence model (component retrieval)",
            "not the EC",
            _T({"caller"}),
        ),
        "maxim/imagination/trigger.py::ImaginationTrigger._encode_entity_affordances": (
            "Imagined entity affordances (EC + NAc eligibility); the fixture runner's manifest reaches it on "
            "main while the AUT loop runs",
            "live",
            _T({"loop", "main"}),
        ),
        "maxim/imagination/trigger.py::encode_entity_affordances": (
            "Own-body affordance names (orchestrator setup, before sim.aut starts)",
            "live",
            _T({"caller"}),
        ),
        "maxim/integration/memory_hub.py::MemoryHub.on_percept_received": (
            "Text / vision percepts, EC route (MAXIM_SUBSTRATE_PATH=1); the agent bus is synchronous",
            "live",
            _T({"loop", "orchestrator", "comms"}),
        ),
        "maxim/similarity/encoder.py::LinguisticEncoder.encode": (
            "encode routes to encode_decomposed",
            "internal",
            _T(),
        ),
    },
    "pain_signal": {
        "maxim/embodiment/body.py::Embodiment._publish_drive_pain": (
            "Body transduction (drive breach), inside evaluate_failures",
            "live",
            _EVALUATORS,
        ),
        "maxim/embodiment/body.py::Embodiment._publish_pain": (
            "Body transduction (failure mode), inside evaluate_failures",
            "live",
            _EVALUATORS,
        ),
        "maxim/proprioception/pain.py::PainDetector._check_for_pain": (
            "PainDetector: motion pain (robot)",
            "live",
            _T({"caller"}),
        ),
        "maxim/proprioception/pain.py::PainDetector._check_movement_failure": (
            "PainDetector: motion pain (robot; armed by PainCircuitBridge.record_action_start)",
            "live",
            _T({"caller"}),
        ),
        "maxim/proprioception/pain.py::PainDetector.record_tool_error": (
            "PainDetector: tool-failure pain; de-wired, #1200 (no build_executor caller passes pain_detector=)",
            "no live caller",
            _T(),
        ),
        "maxim/proprioception/pain.py::PainDetector.record_tool_running": (
            "PainDetector: sustained-tool pain",
            "no live caller",
            _T(),
        ),
        "maxim/proprioception/pain_bus.py::_reaction_to_pain_signal": ("bus conversion", "internal", _T()),
        "maxim/proprioception/perceived_pain.py::PerceivedPainAssessor.assess": (
            "PerceivedPainAssessor, inside AnticipatoryPainExecutor (before execute)",
            "live",
            _T({"loop"}),
        ),
        "maxim/proprioception/perceived_pain.py::PerceivedPainAssessor.assess_text": (
            "PerceivedPainAssessor via bridge.percept_anxiety_hook on every non-substrate-primary send: "
            "pain into the AUT's buses from whichever thread sends, sim.dm included",
            "live",
            _T({"orchestrator", "sim.dm", "main"}),
        ),
        "maxim/simulation/sandbox.py::PainTriggerLayer._fire_pain": (
            "Sandbox PainTriggerLayer (a PainSignal when no ReactionBus)",
            "live",
            _T({"loop"}),
        ),
        "maxim/simulation/tools.py::DamageComponentTool.execute": (
            "Narrator / reflex consequence (direct PainSignal)",
            "live",
            _NARRATOR,
        ),
    },
    "reaction": {
        "maxim/embodiment/backends/cerebellum_modulator.py::CerebellumModulator._emit_failure_reaction": (
            "CerebellumModulator prediction pain (cerebellum_modulator_factory has no caller)",
            "dormant",
            _T(),
        ),
        "maxim/embodiment/backends/cerebellum_modulator.py::CerebellumModulator._emit_success_reaction": (
            "CerebellumModulator prediction reward (cerebellum_modulator_factory has no caller)",
            "dormant",
            _T(),
        ),
        "maxim/proprioception/perceived_pain.py::PerceivedPainAssessor.assess": (
            "PerceivedPainAssessor, inside AnticipatoryPainExecutor (before execute)",
            "live",
            _T({"loop"}),
        ),
        "maxim/proprioception/perceived_pain.py::PerceivedPainAssessor.assess_text": (
            "PerceivedPainAssessor via bridge.percept_anxiety_hook (see pain_signal)",
            "live",
            _T({"orchestrator", "sim.dm", "main"}),
        ),
        "maxim/reactions/compat.py::pain_signal_to_reaction": ("bus conversion", "internal", _T()),
        "maxim/runtime/pain_interceptor.py::PainInterceptorExecutor.execute": (
            "PainInterceptorExecutor",
            "live",
            _T({"loop"}),
        ),
        "maxim/runtime/sim_adapter.py::SimulationAdapter.next_observation": (
            "Sim pain percept -> Reaction (the live inject_pain path)",
            "live",
            _T({"loop"}),
        ),
        "maxim/simulation/conversational_source.py::ConversationalSource.inject_pain": (
            "The direct-Reaction branch needs pain_bus=, which no caller passes",
            "no live caller",
            _T(),
        ),
        "maxim/simulation/sandbox.py::PainTriggerLayer._fire_pain": (
            "Sandbox PainTriggerLayer (a Reaction when a ReactionBus exists)",
            "live",
            _T({"loop"}),
        ),
    },
    "evaluate_failures": {
        "maxim/embodiment/percepts.py::EmbodimentPerceptSource.next_percept": (
            "Body percept source",
            "dormant",
            _T({"loop"}),
        ),
        "maxim/embodiment/tool_bridge.py::ModulatorAffordanceTool.execute": (
            "Affordance self_effect / target_effect (the AUT's tools; OrchestratorActorTool's ephemeral ones)",
            "live",
            _T({"loop", "orchestrator"}),
        ),
        "maxim/runtime/loop_gates.py::tick_embodiment_drift": ("LLM-primary live tick", "live", _T({"loop"})),
        "maxim/runtime/substrate_proposal.py::propose_via_substrate": ("Substrate-primary tick", "live", _T({"loop"})),
        "maxim/simulation/foundry.py::run_gauntlet": ("Foundry gauntlet (no agent)", "live", _T({"caller"})),
        "maxim/simulation/tools.py::DamageComponentTool.execute": ("Narrator / reflex consequence", "live", _NARRATOR),
        "maxim/simulation/tools.py::OrchestratorActorTool.execute": (
            "Narrator actor affordance",
            "live",
            _T({"orchestrator"}),
        ),
        "maxim/simulation/tools.py::SetEntitySensorTool._adjust": ("Narrator / reflex consequence", "live", _NARRATOR),
        "maxim/simulation/tools.py::SetEntitySensorTool.execute": ("Narrator / reflex consequence", "live", _NARRATOR),
    },
    "body_write": {
        "maxim/embodiment/audio_localization.py::DoAFeed.run": (
            "Reachy DoA (live), sensor lane",
            "live",
            _T({"doa-feed"}),
        ),
        "maxim/embodiment/audio_localization.py::world_set_azimuth": ("helper", "internal", _T()),
        "maxim/embodiment/backends/minecraft.py::MinecraftWorldBackend.sync_world_sensors": (
            "Minecraft world state",
            "live",
            _T({"mc-sync", "loop"}),
        ),
        "maxim/embodiment/tool_bridge.py::_apply_sensor_deltas": ("helper", "internal", _T()),
        "maxim/embodiment/tool_bridge.py::ModulatorAffordanceTool.execute": (
            "Affordance self_effect / target_effect",
            "live",
            _T({"loop", "orchestrator"}),
        ),
        "maxim/hardware/reachy/motor_backend.py::ReachyOrientMotorBackend._world_set_measured": (
            "Reachy motor readback",
            "live",
            _T({"loop"}),
        ),
        "maxim/runtime/agent_loop.py::run_agentic_loop": (
            "Loop §1.16 azimuth echo (sim orienting)",
            "live",
            _T({"loop"}),
        ),
        "maxim/simulation/cradle_mother.py::reactive_mother_tick": (
            "Cradle reactive mother: hunger and azimuth into the AUT's body while sim.aut runs",
            "live",
            _T({"main"}),
        ),
        "maxim/simulation/dm_runtime.py::CascadeResolver.resolve": (
            "DM cascade (campaign entities; the AUT's body when a role resolves to it); on main when the "
            "campaign runs non-interactively",
            "live",
            _T({"sim.dm", "main"}),
        ),
        "maxim/simulation/tools.py::DamageComponentTool.execute": (
            "Narrator / reflex consequence (component damage)",
            "live",
            _NARRATOR,
        ),
        "maxim/simulation/tools.py::SetEntitySensorTool._adjust": ("Narrator / reflex consequence", "live", _NARRATOR),
        "maxim/simulation/tools.py::SetEntitySensorTool.execute": ("Narrator / reflex consequence", "live", _NARRATOR),
    },
}


def _callee(call: ast.Call) -> tuple[str | None, str | None]:
    func = call.func
    if isinstance(func, ast.Name):
        return func.id, None
    if isinstance(func, ast.Attribute):
        recv = func.value
        if isinstance(recv, ast.Name):
            return func.attr, recv.id
        if isinstance(recv, ast.Attribute):
            return func.attr, recv.attr
        return func.attr, None
    return None, None


def _categories(name: str | None, receiver: str | None, rel: str) -> list[str]:
    out = []
    if name in FACTORIES and rel != FACTORY_MODULE:
        out.append("percept_factory_call")
    if name == "Percept" and rel != FACTORY_MODULE:
        out.append("raw_percept")
    if name == "encode_sensors":
        out.append("sensor_encode")
    if name == "encode_decomposed" or (name == "encode" and receiver is not None and receiver.endswith("encoder")):
        out.append("linguistic_encode")
    if name == "PainSignal":
        out.append("pain_signal")
    if name == "Reaction":
        out.append("reaction")
    if name == "evaluate_failures":
        out.append("evaluate_failures")
    if name in BODY_WRITERS:
        out.append("body_write")
    return out


def scan(src: Path = SRC) -> dict[str, set[str]]:
    """Every site in ``src/maxim``, by category."""
    found: dict[str, set[str]] = defaultdict(set)
    for path in sorted((src / "maxim").rglob("*.py")):
        rel = path.relative_to(src).as_posix()

        def walk(node: ast.AST, stack: list[str]) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    walk(child, [*stack, child.name])
                    continue
                if isinstance(child, ast.Call):
                    name, receiver = _callee(child)
                    for category in _categories(name, receiver, rel):
                        found[category].add(f"{rel}::{'.'.join(stack) or '<module>'}")
                walk(child, stack)

        walk(ast.parse(path.read_text()), [])
    return found


def _diff(found: dict[str, set[str]], census: dict[str, dict]) -> list[str]:
    problems = []
    for category in sorted(set(found) | set(census)):
        actual, table = found.get(category, set()), set(census.get(category, {}))
        for site in sorted(actual - table):
            problems.append(f"NEW {category}: {site} -- name it, its liveness and threads in CENSUS")
        for site in sorted(table - actual):
            problems.append(f"STALE {category}: {site} -- no longer in src/; drop it from CENSUS")
    return problems


def test_the_census_names_every_site_and_only_existing_ones() -> None:
    problems = _diff(scan(), CENSUS)
    assert not problems, "\n".join(problems)


def test_every_row_has_a_known_liveness_and_threads_only_when_live() -> None:
    """Table consistency (the columns are hand-asserted; see the module docstring's limits)."""
    bad = []
    for category, sites in CENSUS.items():
        for site, (_, liveness, threads) in sites.items():
            if liveness not in LIVENESS or not threads <= THREADS:
                bad.append((category, site, liveness, sorted(threads)))
            if (liveness == "live") != bool(threads) and liveness != "dormant" and liveness != "not the EC":
                bad.append((category, site, "a live row names its threads; a dead one names none"))
    assert bad == []


def test_the_table_declares_the_narrator_on_the_orchestrator_thread() -> None:
    """Table consistency only (§3.2's declared edge): every narrator-tool body evaluation lists the
    orchestrator thread. The scan cannot see threads, so this pins what the table SAYS."""
    narrator = {s: t for s, (_, _, t) in CENSUS["evaluate_failures"].items() if "simulation/tools.py" in s}
    assert narrator and all("orchestrator" in t for t in narrator.values())


def test_the_substrate_channels_are_the_three_body_channels() -> None:
    from maxim.runtime.substrate_proposal import _SUBSTRATE_CHANNELS

    assert [ch.tag for ch in _SUBSTRATE_CHANNELS] == ["interoception", "audio", "world"]


def test_the_scan_is_not_vacuous(tmp_path: Path) -> None:
    """Known answer, both directions: a planted producer in every category is NEW, and a table site that
    is not there is STALE."""
    pkg = tmp_path / "maxim" / "probe"
    pkg.mkdir(parents=True)
    (pkg / "planted.py").write_text(
        "def produce(body, enc, bus, sensor_encoder, linguistic_encoder, p):\n"
        "    make_text_percept('x')\n"
        "    Percept()\n"
        "    sensor_encoder.encode_sensors({})\n"
        "    enc.encode_decomposed('x', 'text', 'a')\n"
        "    linguistic_encoder.encode(p)\n"
        "    PainSignal()\n"
        "    Reaction()\n"
        "    body.evaluate_failures()\n"
        "    world_set_axis(body, 'azimuth', 0.0)\n"
    )
    found = scan(tmp_path)
    site = "maxim/probe/planted.py::produce"
    assert {c for c, sites in found.items() if site in sites} == set(CENSUS)
    problems = _diff(found, CENSUS)
    assert all(f"NEW {c}: {site}" in "\n".join(problems) for c in CENSUS)
    assert any(p.startswith("STALE") for p in problems)
