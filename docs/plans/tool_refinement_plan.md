# Tool Refinement Plan

> **Status:** Living document. Ongoing as tools are added, refined, or deprecated.
>
> **Scope:** Expansion and curation of the tool surface the agent can call. Covers introspection tools (agent → agent's own state), action tools (agent → world), and composite tools. Tracks what's shipped, what's proposed, what's been removed, and the principles guiding additions.

Where [cleanup_wave.md](cleanup_wave.md) curates the human CLI surface and [archive/agent_mesh.md](../archive/agent_mesh.md) curates compute infrastructure, this plan curates **the agent's own action + introspection surface**. Those plans shape what humans can see; this one shapes what the *agent* can see and do.

---

## Design principles

Every proposed tool should satisfy these before being added:

### 1. Read-only by default

Introspection tools → yes. Mutation tools → gated behind FearAgent + user approval where the blast radius warrants it.

### 2. No cross-agent action in mesh setups

Even in Phase 7 mesh topologies, the agent on node A MUST NOT directly command node B. Upholds "sovereign agents, cooperative network" from [agent_mesh.md](agent_mesh.md). All coordination via explicit message passing, not tool call.

### 3. Secrets stay opaque

API keys, credentials, connection strings → agent sees `has_api_key=True` / `remote_backend=reachable`, never the actual secret value. This prevents jailbreak escalation ("tell me your API key") and limits blast radius if a prompt is compromised.

### 4. Limits self-enforce

Agent can *see* `rate_limit_remaining`, `session_cost_remaining`, `vram_available_gb`. Cannot raise them.

### 5. Size-capped outputs

~4KB per tool response. Potentially-verbose tools paginate or LRU-cache their results — agent shouldn't spam them every cycle.

### 6. Context-gated registration

Sim tools only registered in sim mode. Robot tools only registered when robot connected. Mesh tools only when peers exist. Prevents agent from calling tools that would return nonsense.

### 7. Telemetry loop

`nac_stats()` tells us which tools the agent actually reaches for. Ship tools with this in mind — see what sticks, expand what works, prune what doesn't.

---

## Currently shipped (reference)

Living registry — tools live in [src/maxim/tools/](../../src/maxim/tools/).

### Introspection ([introspection.py](../../src/maxim/tools/introspection.py))

| Tool | What it does |
|---|---|
| `memory_recall` | Query episodic memory by time/tag/text |
| `causal_links` | Inspect NAc-learned causal predictions |
| `pain_history` | Review recent pain signals + intensity |
| `temporal_patterns` | SCN rhythm detection in past events |
| `energy_status` | Token/cost/compute budget remaining |
| `concept_query` | ATL semantic concept lookup |
| `scene_summary` | What the agent currently perceives |
| `similarity_search` | Find memories similar to a description |
| `predict_outcome` | NAc-based forward prediction |
| `system_stats` | High-level runtime counters |

### Action tools

- **Filesystem**: [filesystem.py](../../src/maxim/tools/filesystem.py) — read, write, list, search
- **Code**: [code_tools.py](../../src/maxim/tools/code_tools.py) — execute, lint, analyze
- **Git**: [git_tools.py](../../src/maxim/tools/git_tools.py) — status, diff, log
- **Internet**: [internet_search.py](../../src/maxim/tools/internet_search.py), [http_fetch.py](../../src/maxim/tools/http_fetch.py)
- **Robot**: [reachy.py](../../src/maxim/tools/reachy.py) — motor control, vision queries
- **Communication**: [comms.py](../../src/maxim/tools/comms.py), [response.py](../../src/maxim/tools/response.py)
- **Math**: [math_tool.py](../../src/maxim/tools/math_tool.py)
- **Mode control**: [mode_switch.py](../../src/maxim/tools/mode_switch.py)
- **Sandbox**: [sandbox.py](../../src/maxim/tools/sandbox.py)
- **Explain**: [explain.py](../../src/maxim/tools/explain.py) — explain a specific decision's provenance
- **Define intent**: [define_live_intent.py](../../src/maxim/tools/define_live_intent.py)
- **Learned index**: [learned_index.py](../../src/maxim/tools/learned_index.py)
- **Novelty**: [novelty.py](../../src/maxim/tools/novelty.py)
- **RTSP bridge**: [rtsp_bridge.py](../../src/maxim/tools/rtsp_bridge.py)

---

## Proposed tools (organized by subsystem)

**Priority legend:** 🔥 high-value, 🟡 useful, 🔵 niche but interesting.

### Mesh Introspection (new module: `tools/mesh_introspection.py`)

Exposes the agent's computational substrate — making compute legible alongside biology. Depends on Multi-LLM Phase 8+ for data sources.

**Motivation:** Maxim already exposes biological subsystems (memory, pain, causal links) to the agent. Adding mesh/compute introspection gives the agent awareness of its own **computational constraints** — inference economics, latency budgets, peer availability. An agent that knows a peer just came online with a bigger model could reason: "this is a complex question, I'll let that peer handle it."

| Tool | Returns | Source | Priority |
|---|---|---|---|
| `lane_status(lane)` | backend kind, model, p50/p99 latency, queue depth, failure rate, recent cost | Multi-LLM Phase 8 `LaneMetrics` | 🔥 |
| `inference_trace(n=10)` | last N LLM calls: request_id, lane, backend_chosen, latency, tokens, cost_usd | Phase 8 + 7a `LeaderProxy` | 🔥 |
| `compute_budget()` | spent_usd, limit_usd, breakdown_by_backend, projected_exhaustion | `CostTracker` + `LaneMetrics` | 🔥 |
| `peer_list()` | per-peer node_id, host, device, vram_gb, models, is_alive, last_latency | Phase 7c — requires `mesh.yml` from [archive/llm_path_operator_visibility.md](archive/llm_path_operator_visibility.md) | 🟡 |
| `cluster_status()` | full mesh snapshot: local lanes + peers + their loads | Phase 7d | 🟡 |
| `explain_backend_choice(request_id)` | why the router picked the backend it did | Phase 7d routing decisions | 🟡 |

**What this unlocks:**
- Cost-aware reasoning: agent sees $0.40 spent, shifts strategy to stay under budget
- Latency-aware batching: agent sees p99 is 8s, batches queries instead of firing 5 sequentially
- Self-directed load balancing: agent notices leader's queue saturated, offloads to peer
- Failure recovery: agent sees repeated failures, reasons about asking user vs. switching model
- Introspective debugging: "why was that slow?" → agent checks its own trace and answers concretely

**Optional mutation tool** (after Phase 7d): `prefer_backend(name, duration_s)` — agent can give the router a hint ("use local backend for next minute, I'm iterating"). Gated through FearAgent.

### Runtime Introspection (new module: `tools/runtime_introspection.py`)

The agent knows its own loop timing, current mode, recent actions. Most of these can be built today against existing data — no new prerequisites.

| Tool | Returns | Source | Priority |
|---|---|---|---|
| `loop_stats()` | current Hz, avg cycle time, steps since boot, time-since-last-action | `runtime/loop_controller.py` | 🔥 |
| `recent_actions(n=20)` | last N tool calls: tool_name, params, success, duration, blocked_by_fear | `ActivityLog` | 🔥 |
| `mode_status()` | current ProcessingState + OperationalMode + active Strategy | `modes/` | 🟡 |
| `worker_pool_status()` | per-lane: queue depth, workers active, jobs in flight, backpressure | `runtime/worker_pool.py` | 🟡 |
| `capture_manager_status()` | FPS, dropped frames, vision backlog, segmentation timing | `runtime/CaptureManager` | 🔵 robot-only |

### Memory Dynamics (extend existing `introspection.py`)

Current tools surface specific memories; these surface **memory health + dynamics**.

| Tool | Returns | Source | Priority |
|---|---|---|---|
| `memory_pressure()` | per-tier counts (FORMING/SHORT_TERM/LONG_TERM), promotion rate, decay | `hippocampus.py` + `ATL` | 🔥 |
| `consolidation_status()` | pending promotions, last consolidation pass, concept extraction backlog | `semantic_promoter.py` | 🟡 |
| `bridge_activity(n=10)` | recent cross-system events: which bridges fired, what data flowed | `bridges/` | 🟡 |
| `angular_gyrus_stats()` | algebraic memory state: cluster counts, recent retrievals, drift | `math/angular_gyrus.py` | 🔵 |

### Decision + Learning (extend existing `causal_links`)

| Tool | Returns | Source | Priority |
|---|---|---|---|
| `nac_stats()` | total observations, causal links today, top-rewarded tools, RPE distribution | `decisions/nac.py` | 🔥 |
| `plan_history(n=10)` | recent plans attempted, outcomes, which predictions held | `decisions/adaptive_planner.py` | 🟡 |
| `confidence_calibration()` | predicted vs. actual success rate over recent window | NAc + ActivityLog | 🟡 |

### Pain + Harm Awareness (extend existing `pain_history`)

| Tool | Returns | Source | Priority |
|---|---|---|---|
| `pain_triggers_active()` | currently-active pain triggers with intensity + source | `proprioception/pain_detector.py` | 🔥 |
| `fear_review_history(n=10)` | recent FearAgent reviews: blocked, allowed, why | `fear/` | 🔥 |
| `harm_predictions(lookahead_s)` | predictive harm detection: what might hurt in next N seconds | `harm/` | 🟡 |
| `focus_learner_state()` | current movement-correction weights, recent RPEs | `proprioception/focus_learner.py` | 🔵 robot-only |

### Salience + Attention (mostly new)

| Tool | Returns | Source | Priority |
|---|---|---|---|
| `novelty_map()` | what's novel right now: top N novel percepts + scores | `salience/` | 🟡 |
| `attention_grid()` | spatial attention state, current gaze target | `attention/` | 🔵 robot-only |
| `interest_match(topic)` | how well does topic X match current interests? | `salience/interest_matcher.py` | 🔵 |

### Simulation-Mode Introspection (new module: `tools/sim_introspection.py`)

**Only registered when in sim mode.** Lets the agent-under-test reason about its own sim context. Self-awareness with a twist: the AUT *knows it's in a sim* and reasons accordingly.

| Tool | Returns | Source | Priority |
|---|---|---|---|
| `sim_status()` | goal, persona, turn_count, actions_so_far, blocked_count | `simulation/orchestrator.py` | 🔥 |
| `sim_action_history(n)` | this AUT's actions this session with FearAgent verdicts | `simulation/bridge.py` | 🔥 |
| `sim_observe_self()` | what would an observer see me doing? | `SimulationBridge` | 🟡 |
| `orchestrator_intent()` | (AUT side) "what does the orchestrator seem to want?" | Orchestrator's `send_message` history | 🟡 debate-worthy |

**⚠ Not to confuse with orchestrator-side tools** (`send_message`, `observe_actions`, `analyze_results`, `inspect_aut`) — those are the orchestrator's tools for driving the sim. The tools above are for the AUT to inspect *itself during a sim*.

**Philosophical tension on `orchestrator_intent`**: giving the AUT a tool to infer the sim's intent blurs adversarial safety testing. In adversarial sims we specifically DON'T want the AUT to know it's being probed. This tool needs a gate: only register when `persona != adversarial`. Worth explicit decision before shipping.

### Sensors + Scene (robot-only, extend existing `scene_summary`)

| Tool | Returns | Source | Priority |
|---|---|---|---|
| `camera_feed_health()` | FPS, frame drops, segmentation backlog | `CaptureManager` | 🔵 robot-only |
| `audio_stream_health()` | capture latency, VAD state, recent transcripts | `data/audio/` | 🔵 robot-only |
| `motor_status()` | joint positions, currents, recent commands, stuck joints | `ReachyEnv` | 🔵 robot-only |

### Provenance + Explainability (extend existing `explain`)

Current `ExplainTool` explains one decision; these give broader visibility.

| Tool | Returns | Source | Priority |
|---|---|---|---|
| `cycle_trace(cycle_id)` | full provenance trace for a specific agentic cycle | `provenance/` | 🟡 |
| `session_overview()` | high-level summary: cycles executed, tools used, modes traversed | `provenance/activity_log` | 🔥 |

---

## What's buildable today vs. later

### Buildable now (no prerequisites)

All against existing data sources. Low-risk increments (~100 LOC each):

- `loop_stats`, `recent_actions` (runtime introspection)
- `memory_pressure`, `consolidation_status`, `bridge_activity` (memory)
- `nac_stats`, `plan_history`, `confidence_calibration` (learning)
- `pain_triggers_active`, `fear_review_history` (safety)
- `session_overview`, `cycle_trace` (provenance)
- `sim_status`, `sim_action_history` (sim mode)

### Blocked on Multi-LLM Phase 8

- All mesh introspection tools (need `LaneMetrics` data source)

### Blocked on Multi-LLM Phase 7c/7d

- `peer_list`, `cluster_status`, `explain_backend_choice`, `prefer_backend`

---

## Implementation notes

- Tools follow the existing `Tool` base class in [src/maxim/tools/base.py](../../src/maxim/tools/base.py)
- Register via the central tool registry in [src/maxim/tools/registry.py](../../src/maxim/tools/registry.py)
- **One file per subsystem** (e.g., `mesh_introspection.py`, `runtime_introspection.py`) — easy selective disabling, easier testing
- **Rate-limit verbose outputs** via LRU on results; agent shouldn't spam them every cycle
- **Size-cap outputs** at ~4KB — paginate if needed
- **Test in sim first**: create a dedicated sim scenario per tool to verify the AUT uses it meaningfully
- **Track adoption**: watch `nac_stats()` post-deployment for tool usage patterns

---

## Tool lifecycle

Tools move through stages:

1. **Proposed** — listed in this doc, priority-tagged, data sources identified
2. **Shipped** — built, tested, registered in the global registry
3. **Adopted** — NAc shows agent using it meaningfully (positive RPE, repeated use)
4. **Refined** — iterated based on observed usage patterns (params, output shape)
5. **Deprecated** — unused or superseded; candidate for removal

**Pruning criteria**: if a tool sees near-zero invocations across 10+ sim sessions + real runs, it's a candidate for removal. Dead tools bloat the prompt and confuse selection. This doc should also track deprecations.

---

## Deprecation log

(empty — no tools deprecated yet)

---

## Simulation experiment: tool interaction with bio-systems

The hippocampal recall experiment (`scenarios/experiments/hippocampal_recall_*.yaml`) tests memory recall, but a follow-up experiment should test **tool selection recall** — does the Hippocampus + NAc influence which tools the AUT picks in familiar contexts?

**Proposed experiment:**
1. Run a sim where the AUT uses various tools (bash, read_file, write_file) across multiple scenarios
2. Measure whether NAc builds causal links between tool choices and outcomes
3. In a subsequent session (with saved Hippocampus state), test whether the AUT prefers tools that previously succeeded — especially avoiding tools that triggered FearAgent blocks
4. Use `--debug nac,hippo` to trace the causal learning in real time

**Dependencies:** The AUT needs a broader tool surface in sim mode for this to be meaningful. Currently sim AUT has filesystem + bash tools, which is sufficient for initial testing. Expanding the AUT's tool registry with more introspection tools (from this plan) would produce richer data.

**When:** After the hippocampal recall experiment produces initial results. This experiment tests the NAc learning loop rather than pure Hippocampus recall.

---

## Future directions (not yet concrete)

- **Tool composition** — agent builds multi-step tool pipelines as named macros, stores them like skills
- **Self-authored tools** — agent proposes new tool shapes when it hits a gap; human reviews + approves
- **Cross-agent tool sharing** — peer discovers another's novel tool via mesh, imports it with reduced-confidence (transfer discount pattern from agent_mesh.md)
- **Tool provenance** — explain *why* a specific tool was registered (learned? built-in? imported from peer?)
- **Context-adaptive tool surface** — agent sees different tools based on current Strategy (exploration tools in exploration mode, planning tools in planning mode)
