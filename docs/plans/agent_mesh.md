# Agent Mesh Plan

> **Status:** Phase 1a-1b foundations implemented via Research Protocol Phase 0. AgentProfile, UMR, MeshMessage, and LocalMessageBus live in `src/maxim/mesh/`. All multi-LLM infrastructure prerequisites are complete (Phases 1-8, 7a, 7a-ext, 7b — LeaderProxy, LaneMetrics, admission control, remote update). RuntimeCapabilities and CommunicationGateway channel system also implemented.
>
> **Prerequisites completed (from multi-LLM scaling, now archived):**
> - Phases 1-3: LaneBackendManager, lane configs, safety gates
> - Phases 4-6: Remote LLM, Cloudflare tunnel, auto-spawn, leader mode
> - Phase 7a: LeaderProxy (auth, logging, GPU metrics, debug endpoints)
> - Phase 7a-ext: Remote self-update (`maxim peer update`)
> - Phase 7b: Admission control (concurrency cap, per-peer rate limiting)
> - Phase 8: LaneMetrics (per-lane p50/p99, failure rate, token throughput)
> - System heartbeat (GPU/CPU/RAM/disk/WiFi sampling, stall detection)
>
> **Measured baselines**: 44ms short-completion on RTX 5080, ~5-20ms LAN hops, ~20-50ms Cloudflare tunnel hops. Mistral tokens are ~1.7x English word count. Use these when sizing mesh routing decisions, context budgets, and retry timeouts.

Cooperative peer-to-peer network of Maxim agent instances. Each agent owns its memories, causal models, and learned behaviors — but can share them cooperatively. Agents discover each other, advertise capabilities, delegate tasks, and learn from each other's experiences.

---

## Design Principle: Sovereign Agents, Cooperative Network

**Each Maxim instance is a sovereign agent.** It owns:
- Its hippocampus (episodic memories)
- Its NAc (causal links and RPE-learned predictions)
- Its ATL (semantic concepts)
- Its learned significance weights
- Its tool registry and skills

**No agent can modify another's state directly.** Cooperation happens through:
- **Sharing**: "Here's what I learned" — serialized CausalLinks/reflections sent as gifts
- **Requesting**: "Can you do this?" — delegated goals with results returned
- **Querying**: "What do you know about X?" — read-only queries against peer memory
- **Advertising**: "Here's what I can do" — capability broadcasts

The receiving agent always decides what to do with shared data. A CausalLink from a peer gets imported into the local NAc with reduced confidence (transfer discount). A delegated goal gets evaluated by the local AdaptivePlanner before accepting. Nothing is forced.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent Mesh Protocol                        │
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │  Maxim-A    │    │  Maxim-B    │    │  Maxim-C    │      │
│  │  (Reachy)   │◄──►│  (Home PC)  │◄──►│  (Laptop)   │      │
│  │             │    │             │    │             │      │
│  │  Sovereign: │    │  Sovereign: │    │  Sovereign: │      │
│  │  ├ Hippo    │    │  ├ Hippo    │    │  ├ Hippo    │      │
│  │  ├ NAc      │    │  ├ NAc      │    │  ├ NAc      │      │
│  │  ├ ATL      │    │  ├ ATL      │    │  ├ ATL      │      │
│  │  ├ EC       │    │  ├ EC       │    │  ├ EC       │      │
│  │  └ Tools    │    │  └ Tools    │    │  └ Tools    │      │
│  │             │    │             │    │             │      │
│  │  Shares:    │    │  Shares:    │    │  Shares:    │      │
│  │  ├ Identity │    │  ├ Identity │    │  ├ Identity │      │
│  │  ├ Lessons  │    │  ├ Lessons  │    │  ├ Lessons  │      │
│  │  └ Results  │    │  └ Results  │    │  └ Results  │      │
│  └─────────────┘    └─────────────┘    └─────────────┘      │
│                                                              │
│  Discovery: mDNS (LAN) + Cloudflare tunnel (WAN)            │
│  Transport: OpenAI-compat API + mesh protocol extension      │
│  Identity:  AgentIdentity broadcast via heartbeat            │
│  Trust:     Transfer discount on imported knowledge          │
└──────────────────────────────────────────────────────────────┘
```

---

## What's Already Serializable (Verified)

These types already have `to_dict()` / `from_dict()` and can go over the wire today:

| Type | Module | What it carries |
|------|--------|----------------|
| `EpisodicMemory` | memory/types.py | Full agentic loop: perceive → decide → act → outcome |
| `CompressedMemory` | memory/types.py | Lightweight: goal, tool, success, novelty |
| `CausalLink` | decisions/causal_link.py | event → outcome with R-W value, confidence, RPE, temporal delta |
| `OutcomePrediction` | decisions/causal_link.py | Predicted outcome with value, delay, confidence |
| `PredictedOutcome` | memory/types.py | Pattern completion: tool, success, goal, confidence |
| `Percept` | agents/bus.py | Sensor input with source, content, salience |
| `WorkingMemoryEntry` | agents/bus.py | Memory wrapper with tier, salience, decay |

**Not serializable yet (need to_dict/from_dict):**
- `ProposedGoal`, `SubGoal` — goal delegation
- `RuntimeCapabilities` — capability advertisement
- `StreamEvent` — telemetry
- `ToolResult` — task delegation results

---

## Implementation

### Phase 0a: PeerRegistry + mDNS Discovery (~200 LOC)

> Absorbed from multi-LLM Phase 7c. Infrastructure for auto-discovery — used by both inference routing and agent mesh communication.

Remove the need for manually shared URLs on LAN. Each Maxim instance advertises itself via mDNS (same mechanism the robot stack uses for Reachy discovery).

- Service type: `_maxim-llm._tcp.local.`
- TXT records: `node_id`, `models`, `vram_gb`, `device` (gpu|cpu), `proxy_port`
- `PeerInfo` dataclass: `{node_id, host, port, models, device, vram_gb, last_seen}`, 30s heartbeat timeout
- `PeerRegistry.peers() -> list[PeerInfo]`, `get_peer_for_model(model) -> PeerInfo | None`

**Opt-in gates:** `MAXIM_PEER_ENABLED=1` env var AND `zeroconf` importable (optional `[mesh]` extra: `pip install -e '.[mesh]'`). Solo/tunnel users are unaffected.

| File | Change | LOC |
|---|---|---|
| `mesh/peer_registry.py` | **New.** `PeerRegistry` class with mDNS advertise/browse | ~120 |
| `mesh/peer_info.py` | **New.** `PeerInfo`, `PeerAdvertisement` dataclasses | ~30 |
| `pyproject.toml` | Add `[mesh]` optional extra: `zeroconf>=0.80` | ~3 |
| `runtime/lane_backends.py` | `build_primary_router()` starts registry when enabled | ~20 |
| `doctor/checks.py` | New "mDNS broadcast reachable" check | ~20 |

### Phase 0b: InferenceRouter — Per-Request Backend Selection (~250 LOC)

> Absorbed from multi-LLM Phase 7d. The compute-routing layer that the agent mesh's task delegation builds on.

Per-request routing chain: local → LAN peer → remote tunnel → graceful failure. Augments `LaneBackendManager.get_backend()`.

```
Routing chain (first healthy backend wins):
  1. Local lane backend        — 0ms overhead
  2. Best LAN peer (from 0a)   — 5-20ms hop, selected by VRAM/GPU
  3. Remote tunnel backend      — 20-50ms hop
  4. None (caller degrades gracefully)
```

**Routing inputs** (from LaneMetrics + PeerRegistry):
- Per-backend p50/p99 latency and failure rate
- Peer VRAM + advertised device (GPU over CPU, higher VRAM wins ties)
- Context window fit (skip backends whose `n_ctx` can't hold the request)
- Exponential backoff on failing backends (30s/60s/120s, cap 10min)

| File | Change | LOC |
|---|---|---|
| `mesh/inference_router.py` | **New.** `InferenceRouter` class with routing chain + backoff | ~150 |
| `runtime/lane_backends.py` | `LaneBackendManager.attach_router()`, delegate from `get_backend()` | ~40 |
| `models/language/openai_backend.py` | Add `health_check() -> bool` (HEAD `/v1/models`, 1s timeout) | ~15 |
| `mesh/peer_info.py` | Add `estimated_latency_ms` field | ~10 |

### Phase 1: AgentIdentity — Who Am I?

Each Maxim instance builds an identity that describes what it is, what it can do, and what it has learned. This is the foundation for peer discovery and task routing.

```python
@dataclass
class AgentIdentity:
    """What this agent is and what it can do.

    Broadcast to peers via mDNS heartbeat. Read-only externally —
    only the owning agent updates its own identity.
    """
    # Core identity
    agent_id: str                          # Unique node ID (UUID, persisted across restarts)
    agent_name: str                        # Human-readable name ("reachy-kitchen", "desktop-coder")
    started_at: float                      # Startup timestamp

    # Hardware capabilities (from RuntimeCapabilities)
    capabilities: dict[str, Any]           # Serialized RuntimeCapabilities
    available_tools: list[str]             # Tool names from ToolRegistry
    available_skills: list[str]            # Skill names from ProtocolRegistry
    embodiment_summary: dict[str, Any] | None = None   # Optional: EmbodimentCapability
                                                       # {name, modalities, affordances,
                                                       #  hardware_backed: bool}
                                                       # Populated when Embodiment Core ships;
                                                       # peers can query "who has body with grasp?"

    # Learned knowledge summary (NOT the full data — just statistics)
    episodic_memory_count: int             # How many episodes in hippocampus
    causal_link_count: int                 # How many links in NAc
    concept_count: int                     # How many concepts in ATL
    top_tools: list[dict[str, Any]]        # Top 5 tools by success rate: [{name, uses, success_rate}]
    top_causal_patterns: list[dict]        # Top 5 high-confidence causal links (serialized summaries)
    recent_domains: list[str]              # What goal domains this agent has been working in

    # Inference capabilities (from multi-LLM plan)
    inference_models: list[dict[str, Any]] # [{profile, device, vram_gb}] — what LLMs are loaded
    inference_available: bool              # Can this node accept inference requests?

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict) -> AgentIdentity: ...

    @classmethod
    def build_from_agent(cls, agent, memory_hub, tool_registry, capabilities) -> AgentIdentity:
        """Construct identity from live agent systems."""
        ...
```

**Key design decision:** The identity shares *statistics* about knowledge, not the knowledge itself. A peer sees "Agent-B has 500 episodic memories and high-confidence patterns for grab-tool failures" — not the actual memories. The peer must explicitly request knowledge sharing (Phase 4).

### Phase 2: Mesh Protocol Messages

Define the wire protocol for inter-agent communication. Built on JSON over HTTP (reusing the existing Gateway/API server architecture).

```python
class MeshMessageType(Enum):
    """Inter-agent message types."""
    # Discovery
    HEARTBEAT = "heartbeat"                  # Periodic identity broadcast
    IDENTITY_REQUEST = "identity_request"    # "Tell me about yourself"
    IDENTITY_RESPONSE = "identity_response"

    # Task delegation
    GOAL_PROPOSAL = "goal_proposal"          # "Can you do this?"
    GOAL_ACCEPTED = "goal_accepted"          # "Yes, I'll do it"
    GOAL_REJECTED = "goal_rejected"          # "No, I can't / won't"
    GOAL_RESULT = "goal_result"              # "Here's what happened"

    # Knowledge sharing
    EXPERIENCE_OFFER = "experience_offer"    # "I learned something relevant to you"
    EXPERIENCE_REQUEST = "experience_request" # "What do you know about X?"
    EXPERIENCE_RESPONSE = "experience_response"
    CAUSAL_LINK_SHARE = "causal_link_share"  # "This tool fails in this context"
    REFLECTION_SHARE = "reflection_share"    # "Here's why it failed"

    # Prediction queries
    PREDICT_REQUEST = "predict_request"      # "What would happen if I did X?"
    PREDICT_RESPONSE = "predict_response"

    # Inference routing (delegates to multi-LLM plan)
    INFERENCE_REQUEST = "inference_request"
    INFERENCE_RESPONSE = "inference_response"


@dataclass
class MeshMessage:
    """Wire format for inter-agent communication."""
    msg_type: str                    # MeshMessageType value
    sender_id: str                   # AgentIdentity.agent_id
    recipient_id: str | None         # None = broadcast
    payload: dict[str, Any]          # Type-specific content
    timestamp: float
    correlation_id: str              # For request/response matching

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict) -> MeshMessage: ...
```

### Phase 3: MeshChannel — Transport Layer

Implement a `PeerChannel` that plugs into the existing `CommunicationGateway` channel system. This means peer messages flow through the same architecture as SMS/voice — the agent sees them as percepts on the bus.

```python
class PeerChannel(Channel):
    """Communication channel for peer-to-peer agent mesh.

    Plugs into CommunicationGateway using the existing Channel interface.
    Each peer is an HTTP endpoint (reuses the FastAPI server from comms/api.py).
    """

    def __init__(self, peer_registry: PeerRegistry, local_port: int) -> None:
        self._peer_registry = peer_registry
        self._local_port = local_port

    def send(self, recipient: str, body: str, metadata: dict | None = None) -> bool:
        """Send a mesh message to a peer agent."""
        # recipient is the agent_id
        peer = self._peer_registry.get_peer(recipient)
        if peer is None:
            return False
        msg = MeshMessage(
            msg_type=metadata.get("msg_type", MeshMessageType.GOAL_PROPOSAL.value),
            sender_id=self._local_agent_id,
            recipient_id=recipient,
            payload=json.loads(body),
            timestamp=time.time(),
            correlation_id=metadata.get("correlation_id", str(uuid4())),
        )
        # POST to peer's mesh endpoint
        resp = httpx.post(f"{peer.base_url}/mesh", json=msg.to_dict(), timeout=10.0)
        return resp.status_code == 200

    def receive(self, msg: MeshMessage) -> None:
        """Handle an incoming mesh message (called by API endpoint)."""
        # Convert to Percept and publish on local bus
        percept = Percept(
            timestamp=msg.timestamp,
            source=f"mesh:{msg.sender_id}",
            content=json.dumps(msg.payload),
            salience=0.7,  # Peer messages are important but not critical
            metadata={
                "msg_type": msg.msg_type,
                "sender_id": msg.sender_id,
                "correlation_id": msg.correlation_id,
                "external": True,
            },
        )
        self._bus.publish(percept)
```

**API endpoint** (add to existing `comms/api.py`):

```python
@app.post("/mesh")
async def mesh_receive(request: Request):
    data = await request.json()
    msg = MeshMessage.from_dict(data)
    peer_channel.receive(msg)
    return {"status": "ok"}
```

### Phase 4: Knowledge Sharing — Sovereign Exchange

The core cooperation mechanism. Agents share learned CausalLinks and reflections, but the receiving agent applies a **transfer discount** — peer knowledge starts at reduced confidence because it was learned in a different context.

#### 4a. Experience offer/request

```python
class ExperienceSharer:
    """Manages knowledge exchange between agents.

    Respects sovereignty: the local agent decides what to share
    and what to accept. Imported knowledge gets reduced confidence.
    """

    # Peer knowledge starts at this fraction of its original confidence
    TRANSFER_DISCOUNT = 0.5

    # Only share links above this confidence threshold
    SHARE_MIN_CONFIDENCE = 0.6

    def __init__(self, nac: NAc, hippocampus: Hippocampus) -> None:
        self._nac = nac
        self._hippocampus = hippocampus

    def get_shareable_links(self, tool_name: str | None = None) -> list[dict]:
        """Get high-confidence causal links suitable for sharing.

        Only shares links with confidence > SHARE_MIN_CONFIDENCE
        and observation_count >= 3 (not one-off events).
        """
        if tool_name:
            links = self._nac.get_links_for_event(f"tool:{tool_name}")
        else:
            links = self._nac.get_promotion_candidates(
                min_confidence=self.SHARE_MIN_CONFIDENCE,
                min_observations=3,
            )
        return [
            link.to_dict() if hasattr(link, 'to_dict') else link.metadata
            for link in links
        ]

    def import_causal_link(self, link_data: dict, source_agent: str) -> bool:
        """Import a CausalLink from a peer agent.

        Applies transfer discount and tags with source.
        Returns True if imported (novel), False if already known.
        """
        from maxim.decisions.causal_link import CausalLink

        link = CausalLink.from_dict(link_data)

        # Check if we already know this pattern
        existing = self._nac.get_links_for_event(link.event_signature)
        for ex in existing:
            if ex.outcome_signature == link.outcome_signature:
                # Already known — skip import
                return False

        # Apply transfer discount: peer's confidence is discounted
        link.confidence *= self.TRANSFER_DISCOUNT
        link.predicted_value = 0.5  # Reset R-W to neutral — let local experience refine
        link.observation_count = 1  # Treat as single observation locally

        # Tag provenance
        link.event_context["_imported_from"] = source_agent
        link.event_context["_import_timestamp"] = time.time()

        # Register in local NAc
        self._nac._register_imported_link(link)
        return True

    def get_shareable_reflections(self, tool_name: str | None = None) -> list[dict]:
        """Get reflections suitable for sharing."""
        memories = self._hippocampus.recall(
            limit=10,
            tool=tool_name,
        )
        reflections = []
        for m in memories:
            outcome = getattr(m, 'outcome', None)
            ref_text = None
            if hasattr(outcome, 'result') and isinstance(outcome.result, dict):
                ref_text = outcome.result.get("reflection")
            if ref_text:
                reflections.append(m.to_dict())
        return reflections

    def import_reflection(self, memory_data: dict, source_agent: str) -> str | None:
        """Import a reflection from a peer as a low-salience episodic memory.

        Imported reflections get reduced salience (0.5 vs 0.9 for local)
        so they don't dominate the agent's own experience.
        """
        from maxim.memory.types import EpisodicMemory

        memory = EpisodicMemory.from_dict(memory_data)

        # Reduce salience for imported knowledge
        memory.perception.salience = min(memory.perception.salience, 0.5)
        memory.perception.observations["_imported_from"] = source_agent

        # Capture into local hippocampus
        return self._hippocampus.capture(record=memory)
```

#### 4b. Transfer discount rationale

| Source | Confidence multiplier | Why |
|--------|----------------------|-----|
| Local experience | 1.0 | First-hand observation in this context |
| Peer on same hardware | 0.5 | Same tools, but different context/state |
| Peer on different hardware | 0.3 | Different capabilities may invalidate patterns |
| Remote (untrusted) | 0.1 | Unknown provenance |

The AdaptivePlanner already uses NAc confidence in its scoring. Imported links with low confidence will naturally rank below locally-learned links — the agent prefers its own experience but considers peer knowledge as a prior.

### Phase 5: Task Delegation

One agent can propose a goal to another agent. The receiving agent evaluates it with its own AdaptivePlanner and either accepts or rejects.

```python
class TaskDelegator:
    """Delegate goals to peer agents based on capabilities.

    The sender picks the best peer for a goal based on:
    1. Does the peer have the required tool?
    2. Does the peer's NAc predict success for this tool?
    3. How loaded is the peer? (inference queue depth)
    """

    def __init__(self, peer_registry: PeerRegistry, mesh_channel: PeerChannel) -> None:
        self._peers = peer_registry
        self._channel = mesh_channel

    def find_best_peer(self, goal: dict) -> PeerInfo | None:
        """Find the peer best suited for a goal."""
        required_tool = goal.get("tool_name")
        if not required_tool:
            return None

        candidates = []
        for peer in self._peers.all_peers():
            if not peer.is_alive:
                continue
            identity = peer.identity
            if identity is None:
                continue
            if required_tool in identity.available_tools:
                # Check if peer has success history with this tool
                success_rate = 0.5  # default
                for t in identity.top_tools:
                    if t.get("name") == required_tool:
                        success_rate = t.get("success_rate", 0.5)
                        break
                candidates.append((peer, success_rate))

        if not candidates:
            return None
        # Best success rate first
        candidates.sort(key=lambda x: -x[1])
        return candidates[0][0]

    async def delegate(self, goal: dict, peer: PeerInfo) -> dict | None:
        """Send a goal to a peer and await the result.

        Returns the result dict or None on timeout/rejection.
        """
        correlation_id = str(uuid4())
        msg = MeshMessage(
            msg_type=MeshMessageType.GOAL_PROPOSAL.value,
            sender_id=self._local_agent_id,
            recipient_id=peer.node_id,
            payload=goal,
            timestamp=time.time(),
            correlation_id=correlation_id,
        )
        self._channel.send(peer.node_id, json.dumps(msg.payload), {
            "msg_type": msg.msg_type,
            "correlation_id": correlation_id,
        })
        # Wait for response (with timeout)
        return await self._wait_for_response(correlation_id, timeout=60.0)
```

**On the receiving end:**

```python
class TaskReceiver:
    """Evaluates and optionally executes delegated goals from peers.

    Uses the local AdaptivePlanner to assess whether the goal is
    feasible, then executes it through the normal agent loop.
    """

    def __init__(self, planner, executor, nac) -> None:
        self._planner = planner
        self._executor = executor
        self._nac = nac

    def evaluate_proposal(self, goal: dict, sender_id: str) -> tuple[bool, str]:
        """Decide whether to accept a delegated goal.

        Criteria:
        1. Do we have the required tool? (hard requirement)
        2. Does NAc predict success? (soft preference)
        3. Are we overloaded? (queue depth check)
        """
        tool_name = goal.get("tool_name")
        if not tool_name:
            return False, "no tool specified"

        # Check tool availability
        if not self._executor.has_tool(tool_name):
            return False, f"tool {tool_name} not available"

        # Check NAc prediction
        prediction = self._nac.predict("tool", f"tool:{tool_name}")
        if prediction and prediction.predicted_valence.value == "negative" and prediction.confidence > 0.7:
            return False, f"NAc predicts failure (confidence={prediction.confidence:.2f})"

        return True, "accepted"
```

### Phase 6: Distributed Planning

Extend the AdaptivePlanner to consider peer capabilities during decomposition. When a sub-goal requires a capability the local agent doesn't have, it can tag it for delegation.

```python
# In AdaptivePlanner._decompose(), after LLM generates sub-actions:

def _tag_delegatable_subgoals(
    self,
    sub_actions: list[dict],
    peer_registry: PeerRegistry | None,
) -> list[dict]:
    """Tag sub-goals with preferred_node if a peer is better suited."""
    if peer_registry is None:
        return sub_actions

    for action in sub_actions:
        tool_name = action.get("tool_name")
        if not tool_name:
            continue

        # Check if we can do it locally
        local_prediction = None
        if self._nac:
            local_prediction = self._nac.predict("tool", f"tool:{tool_name}")

        # Check if a peer is better suited
        for peer in peer_registry.all_peers():
            if not peer.is_alive or peer.identity is None:
                continue
            peer_identity = peer.identity

            # Peer has the tool and we don't
            if tool_name not in (self._local_tools or []) and tool_name in peer_identity.available_tools:
                action["_preferred_node"] = peer.node_id
                break

            # Peer has better success rate
            if tool_name in peer_identity.available_tools:
                for t in peer_identity.top_tools:
                    if t.get("name") == tool_name and t.get("success_rate", 0) > 0.8:
                        if local_prediction and local_prediction.predicted_value < 0.5:
                            action["_preferred_node"] = peer.node_id
                            break

    return sub_actions
```

### Phase 7: Mesh API Endpoints

Extend the existing FastAPI server from `comms/api.py` with mesh-specific endpoints:

```python
# GET /mesh/identity — Return this agent's identity
# POST /mesh/message — Receive a mesh message
# GET /mesh/peers — List known peers
# POST /mesh/query/predict — "What would happen if I did X?"
# POST /mesh/query/experience — "What do you know about tool X?"
# GET /mesh/status — Cluster status view
```

These reuse the existing API server infrastructure — no new HTTP server needed.

### Phase 8: Serialization for Non-Serializable Types

Add `to_dict()` / `from_dict()` to the types that need network transport:

| Type | What to serialize |
|------|------------------|
| `ProposedGoal` | id, description, priority (str), tool_name, tool_params, reasoning, sub_goals |
| `SubGoal` | id, description, tool_name, tool_params, status (str), depends_on |
| `RuntimeCapabilities` | All fields (all primitives, already trivial) |
| `ToolResult` | success, output, error, error_kind (str) |

These are straightforward — all fields are primitives, enums (serialize as strings), or dicts.

---

## Implementation Sequencing

| Phase | What | Effort | Dependencies |
|-------|------|--------|-------------|
| **1** | `AgentIdentity` dataclass with `build_from_agent()` | Small | None |
| **2** | `MeshMessage` protocol + `MeshMessageType` enum | Small | None |
| **3** | `PeerChannel` + mesh API endpoints on existing FastAPI server | Medium | Multi-LLM Phase 7 (PeerRegistry) |
| **4** | `ExperienceSharer` — causal link + reflection import/export with transfer discount | Medium | Phase 2 |
| **5** | `TaskDelegator` + `TaskReceiver` — goal delegation | Medium | Phases 2, 3 |
| **6** | Distributed planning — `_tag_delegatable_subgoals()` in AdaptivePlanner | Small | Phase 5 |
| **7** | Mesh API endpoints | Small | Phase 3 |
| **8** | Serialization for ProposedGoal, SubGoal, RuntimeCapabilities, ToolResult | Small | Phase 5 |

**Recommended order:**
1. Phase 1 + 2 (identity + protocol) — foundations, no network yet
2. Phase 8 (serialization) — unblocks everything else
3. Phase 3 + 7 (transport + endpoints) — agents can talk
4. Phase 4 (knowledge sharing) — agents learn from each other
5. Phase 5 + 6 (task delegation + distributed planning) — agents cooperate

---

## Trust Model

### What agents share freely (broadcast):
- Identity (capabilities, tool list, knowledge statistics)
- Heartbeat (alive/dead status)

### What agents share on request:
- CausalLinks (with transfer discount applied by receiver)
- Reflections (with reduced salience applied by receiver)
- Predictions ("what would happen if...")

### What agents NEVER share:
- Full hippocampus dumps (too large, too context-specific)
- Raw StructuredContext (contains everything the agent is thinking)
- State snapshots (includes file paths, env vars, etc.)
- API keys or credentials (obviously)

### Transfer discount by trust level:

| Relationship | Discount | How established |
|-------------|---------|-----------------|
| **Self** | 1.0 | — |
| **Verified peer** (same owner, shared secret) | 0.5 | Pre-shared key in config |
| **Discovered peer** (LAN mDNS) | 0.3 | Automatic, no verification |
| **Remote peer** (Cloudflare tunnel) | 0.3 | Tunnel auth serves as verification |
| **Unknown** | 0.1 | Unsolicited contact |

---

## Interaction with Other Plans

| This plan | Other plan | Interaction |
|-----------|-----------|-------------|
| Phase 3 (PeerChannel) | Multi-LLM Phase 7 (PeerRegistry) | Reuses PeerRegistry for agent discovery |
| Phase 5 (task delegation) | Decision Engine (AdaptivePlanner) | Delegated goals evaluated by same planner |
| Phase 4 (knowledge sharing) | Causal Memory (NAc links) | CausalLinks are the primary shared knowledge type |
| Phase 4 (reflection sharing) | Decision Engine Phase 4 (reflections) | Reflections from peers imported as episodic memories |
| Phase 1 (AgentIdentity) | Adaptive Runtime (RuntimeCapabilities) | Identity includes serialized capabilities |
| Phase 1 (AgentIdentity) | Embodiment Core (EmbodimentCapability) | `AgentIdentity.embodiment_summary` advertises body: modalities, affordances, hardware-backed vs. imagined. Populated when Embodiment Core ships. |
| Future (not scheduled) | Embodiment — federation, affordance delegation, NAc transfer | Cross-agent affordance invocation, federated bodies (components from multiple peers), CausalLink transfer gated by spec similarity. Tracked in `future_plans.md`. |

---

## Risks

1. **Knowledge poisoning.** A malicious peer could send fabricated CausalLinks that cause bad decisions. **Mitigation:** Transfer discount reduces imported confidence. Locally-learned links always dominate. Links tagged with `_imported_from` can be audited and purged.

2. **Goal delegation loops.** Agent A delegates to B, B delegates back to A. **Mitigation:** Goals carry a `delegation_depth` counter. Reject goals with depth > 2. Include `delegation_chain: list[str]` to detect cycles.

3. **Network partition.** Peer goes offline mid-task. **Mitigation:** Task delegation has a timeout. AdaptivePlanner's replan loop handles delegation failure like any tool failure — decomposes and retries locally or on another peer.

4. **Memory bloat from imports.** Accepting too many peer reflections floods the hippocampus. **Mitigation:** Rate-limit imports (max 10 per peer per hour). Imported memories have reduced salience so they're evicted first during consolidation.

5. **Privacy leakage.** AgentIdentity broadcasts tool lists and domain summaries. On a shared LAN this reveals what the agent is doing. **Mitigation:** Identity broadcast is opt-in (`MAXIM_PEER_ADVERTISE=1`). Sensitive fields can be omitted. Cloudflare tunnel peers use zero-trust auth.

6. **Clock skew.** Agents on different machines may have different clocks, affecting temporal reasoning. **Mitigation:** MeshMessage includes sender timestamp. Receiver adjusts for skew using round-trip time estimation (same pattern as NTP lite).
