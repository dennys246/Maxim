# Agent Mesh Plan

> **Status:** Phases Pre through 7 COMPLETE (2026-04-07). All core mesh infrastructure is implemented and tested. Remaining: Phase 0a (mDNS) and 0b (InferenceRouter) are deferred until multiple LAN machines exist.
>
> **What shipped:**
> - **Pre-work:** Serialization (ProposedGoal, SubGoal, ToolResult, RuntimeCapabilities) + NAc._register_imported_link()
> - **Phase 1:** AgentIdentity (extends AgentProfile, persistent node_id, build_from_subsystems)
> - **Phase 2:** MeshMessage expansion (16 new types, protocol_version, correlation_id, GOAL_PROPOSAL payload schema)
> - **Phase 3:** PeerRegistry (from peer config), PeerChannel (async send queue, retry, Channel ABC), mesh API endpoint design
> - **Phase 3b:** MeshAdmissionControl (trust-level rate limits, burst detection, escalating gates)
> - **Phase 4:** ExperienceBroker + KnowledgeProvider/KnowledgeReceiver protocol (CausalLink, Reflection, MotorProgram adapters)
> - **Phase 5:** TaskDelegator (find_best_peer, sync delegation, loop detection) + TaskReceiver (depth/cycle/queue/tool/NAc checks)
> - **Phase 6:** AdaptivePlanner._tag_delegatable_subgoals (peer capability routing, gated peer skip)
> - **Phase 7:** PeerClockEstimator (NTP-lite EMA offset learning) + SCN.register_external (clock-corrected temporal bins)
>
> **What the Research Protocol proved (Phase 0):**
> - LocalMessageBus delivery works (synchronous, in-process, thread-safe)
> - MeshMessage serialization round-trips correctly (to_dict/from_dict)
> - AgentProfile identity is sufficient for local multi-agent coordination
> - UMR references enable cross-agent resource naming (`researcher.hippo.exp_001`)
> - Typed message enums (PAPER_DRAFT, REVIEW_RESULT) prevent protocol confusion
> - **Limitation exposed and resolved:** synchronous bus delivery blocks sender thread — PeerChannel uses async send queue (Phase 3)
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

**Not serializable yet (need to_dict/from_dict — see Pre-work):**
- `ProposedGoal`, `SubGoal` — goal delegation (Pre-work)
- `RuntimeCapabilities` — capability advertisement (Pre-work)
- `ToolResult` — task delegation results (Pre-work)
- `StreamEvent` — telemetry (deferred — no phase currently needs it)

---

## Implementation

### Pre-work: Serialization + NAc Import

These are blockers for Phases 4 and 5. Do them first — no network code, just adding methods to existing classes.

#### Serialization for non-serializable types

Add `to_dict()` / `from_dict()` to the types that need network transport:

| Type | Location | What to serialize |
|------|----------|------------------|
| `ProposedGoal` | `agents/bus.py` | id, description, priority (str), tool_name, tool_params, reasoning, confidence, sub_goals (recursive) |
| `SubGoal` | `agents/bus.py` | id, description, tool_name, tool_params, status (str), depends_on, on_failure (str), attempts, max_retries |
| `RuntimeCapabilities` | `runtime/capabilities.py` | All fields (all primitives, already trivial) |
| `ToolResult` | `agents/bus.py` | tool_call_id, tool_name, success, result, error, params |

These are straightforward — all fields are primitives, enums (serialize as strings), or dicts.

#### NAc._register_imported_link()

`ExperienceSharer.import_causal_link()` calls `nac._register_imported_link(link)`, which does not exist yet. Implement in `decisions/nac.py`:

```python
def _register_imported_link(self, link: CausalLink) -> None:
    """Register an externally-imported CausalLink.

    The link should already have transfer discount applied to confidence,
    predicted_value reset, and provenance tagged in event_context.
    """
    sig = link.event_signature
    if sig not in self._links:
        self._links[sig] = []
    self._links[sig].append(link)
    self._register_causal_in_ec(link)
```

~10 LOC in NAc. Add a unit test that imports a link and verifies it appears in `get_links_for_event()`.

---

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
| `mesh/peer_registry.py` | **Extend.** Add mDNS advertise/browse as discovery source to existing `PeerRegistry` (created in Phase 3 from peer config) | ~120 |
| `mesh/peer_info.py` | Add `PeerAdvertisement` dataclass for mDNS TXT records | ~30 |
| `pyproject.toml` | Add `[mesh]` optional extra: `zeroconf>=0.80` | ~3 |
| `runtime/lane_backends.py` | `build_primary_router()` starts mDNS when `MAXIM_PEER_ENABLED=1` | ~20 |
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

**Relationship to AgentProfile:** `AgentProfile` (in `mesh/identity.py`) is the lightweight identity used by local agents (research protocol Writer/Reviewer). `AgentIdentity` **extends** `AgentProfile` with hardware capabilities, knowledge statistics, and inference info needed for network-level coordination. Local agents keep using `AgentProfile`; `AgentIdentity` wraps one and adds the networked fields. The `build_from_agent()` factory constructs a full identity from an `AgentProfile` + live subsystems.

```python
@dataclass
class AgentIdentity:
    """What this agent is and what it can do.

    Extends AgentProfile with network-level metadata. Broadcast to peers
    via heartbeat. Read-only externally — only the owning agent updates
    its own identity.
    """
    # Core identity
    profile: AgentProfile                  # Nickname, role, capabilities, personality
    # NOTE: profile.agent_id is session-scoped (random on each start).
    # AgentIdentity.node_id is the persistent node identity, saved to disk
    # and stable across restarts. Peers use node_id to track each other.
    node_id: str                           # Persistent node ID (UUID, saved to data/util/node_id.txt)
    agent_name: str                        # Human-readable name ("reachy-kitchen", "desktop-coder")
    started_at: float                      # This instance's startup timestamp (not from profile)

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

Expand the existing `MeshMessage` and `MeshMessageType` in `mesh/message.py` — don't create a parallel wire format.

**Migration from existing implementation:** The current `MeshMessage` uses `sender`/`recipient`/`msg_id`/`in_reply_to` as field names, and `MeshMessageType` uses `auto()` (integer enum, serialized via `.name`). Phase 2 keeps these conventions and adds new fields:
- Add `protocol_version: int = 1` field
- Add `correlation_id: str` field (for request/response matching across network; `in_reply_to` stays for reply chains, `correlation_id` groups a full exchange)
- Keep `sender`/`recipient` names (not `sender_id`/`recipient_id`) for consistency with existing code
- Keep `auto()` for enum values; wire format uses `.name` (e.g. `"HEARTBEAT"`, `"GOAL_PROPOSAL"`)

```python
class MeshMessageType(Enum):
    """Inter-agent message types.

    Uses auto() consistently. Wire format serializes via .name.
    Existing research protocol types kept as-is.
    """
    # Research protocol types (existing, unchanged)
    EXPERIMENT_DATA = auto()
    PAPER_DRAFT = auto()
    REVIEW_RESULT = auto()
    REVISION_REQUEST = auto()
    TASK_COMPLETE = auto()
    REQUEST = auto()
    RESPONSE = auto()
    ERROR = auto()

    # Discovery (Phase 2)
    HEARTBEAT = auto()                       # Periodic identity broadcast
    IDENTITY_REQUEST = auto()                # "Tell me about yourself"
    IDENTITY_RESPONSE = auto()

    # Task delegation (Phase 2)
    GOAL_PROPOSAL = auto()                   # "Can you do this?"
    GOAL_ACCEPTED = auto()                   # "Yes, I'll do it"
    GOAL_REJECTED = auto()                   # "No, I can't / won't"
    GOAL_RESULT = auto()                     # "Here's what happened"

    # Knowledge sharing (Phase 2)
    EXPERIENCE_OFFER = auto()                # "I learned something relevant to you"
    EXPERIENCE_REQUEST = auto()              # "What do you know about X?"
    EXPERIENCE_RESPONSE = auto()
    CAUSAL_LINK_SHARE = auto()               # "This tool fails in this context"
    REFLECTION_SHARE = auto()                # "Here's why it failed"

    # Prediction queries (Phase 2)
    PREDICT_REQUEST = auto()                 # "What would happen if I did X?"
    PREDICT_RESPONSE = auto()

    # Inference routing (Phase 2, delegates to multi-LLM plan)
    INFERENCE_REQUEST = auto()
    INFERENCE_RESPONSE = auto()


@dataclass
class MeshMessage:
    """Typed message envelope for inter-agent communication.

    Extends the existing MeshMessage with network-level fields.
    Field names match the current implementation (sender, recipient, msg_id, in_reply_to).
    """
    sender: str                      # Sender nickname or agent_id
    recipient: str                   # Recipient nickname/agent_id ("*" for broadcast)
    msg_type: MeshMessageType
    payload: dict[str, Any] = field(default_factory=dict)
    msg_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    in_reply_to: str | None = None   # msg_id of the message this replies to

    # New fields (Phase 2) — optional with defaults for backward compat
    protocol_version: int = 1        # Bump on breaking changes
    correlation_id: str = ""         # Groups a full request/response exchange

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict) -> MeshMessage:
        """Deserialize. Raises ValueError if protocol_version > supported.
        Missing new fields default gracefully (protocol_version=1, correlation_id="").
        """
        ...
```

**Version rules:** Receivers reject messages with `protocol_version` higher than their own. Messages with lower versions are accepted (backward compat). Bump version when adding required fields or changing payload semantics — new optional fields don't require a bump.

**GOAL_PROPOSAL payload schema** (used by Phase 5 TaskDelegator):

```python
# Required fields in GOAL_PROPOSAL payload:
{
    "tool_name": str,              # Required tool
    "tool_params": dict,           # Tool parameters
    "description": str,            # Human-readable goal description
    "delegation_depth": int,       # Starts at 0, incremented on each delegation
    "delegation_chain": list[str], # Agent IDs that have touched this goal (cycle detection)
    "timeout_s": float,            # Max time peer should spend on this goal
}
# TaskReceiver rejects if delegation_depth > 2 or local agent_id in delegation_chain.
```

### Phase 3: MeshChannel — Transport Layer

Implement a `PeerChannel` that plugs into the existing `CommunicationGateway` channel system. This means peer messages flow through the same architecture as SMS/voice — the agent sees them as percepts on the bus.

**PeerRegistry dependency:** Phase 3 does NOT require mDNS (Phase 0a). Instead, it uses a lightweight `PeerRegistry` backed by the existing `peer/config.py` (tunnel URLs) and optional manual entries. mDNS auto-discovery is added later in Phase 0a as an additional peer source — not a prerequisite.

```python
class PeerRegistry:
    """Registry of known peers. Initially populated from peer config
    (tunnel URLs) and manual registration. Phase 0a adds mDNS as an
    additional discovery source.
    """
    def register(self, peer_id: str, base_url: str, trust_level: str) -> None: ...
    def get_peer(self, peer_id: str) -> PeerInfo | None: ...
    def all_peers(self) -> list[PeerInfo]: ...

    @classmethod
    def from_peer_config(cls) -> PeerRegistry:
        """Bootstrap from existing peer/config.py (tunnel URL + API key)."""
        ...
```

**Important: async send.** The research protocol exposed that `LocalMessageBus` runs handlers in the sender's thread. `PeerChannel.send()` involves HTTP I/O — if called from a bus handler, it blocks the sender. Solution: `PeerChannel` uses a background send queue (`threading.Thread` + `queue.Queue`). The `send()` method enqueues and returns immediately; the background thread drains the queue and POSTs to peers. Failures go to a retry queue with exponential backoff (1s/2s/4s, cap 30s, max 3 retries).

```python
class PeerChannel(Channel):
    """Communication channel for peer-to-peer agent mesh.

    Plugs into CommunicationGateway using the existing Channel interface.
    Each peer is an HTTP endpoint. Mount under /v1/mesh/* to inherit
    LeaderProxy auth, rate limiting, and GPU metrics.

    Sends are queued — the caller never blocks on network I/O.
    """

    def __init__(self, peer_registry: PeerRegistry, local_port: int) -> None:
        self._peer_registry = peer_registry
        self._local_port = local_port
        self._send_queue: queue.Queue[MeshMessage] = queue.Queue(maxsize=1000)
        self._sender_thread = threading.Thread(target=self._drain_queue, daemon=True)
        self._sender_thread.start()

    def send(self, recipient: str, body: str, metadata: dict | None = None) -> bool:
        """Enqueue a mesh message for async delivery to a peer."""
        peer = self._peer_registry.get_peer(recipient)
        if peer is None:
            return False
        msg_type_name = (metadata or {}).get("msg_type", "GOAL_PROPOSAL")
        msg = MeshMessage(
            sender=self._local_agent_id,
            recipient=recipient,
            msg_type=MeshMessageType[msg_type_name],
            payload=json.loads(body),
            correlation_id=(metadata or {}).get("correlation_id", uuid4().hex[:12]),
            protocol_version=1,
        )
        try:
            self._send_queue.put_nowait(msg)
            return True
        except queue.Full:
            logger.warning("mesh send queue full, dropping message to %s", recipient)
            return False

    def _drain_queue(self) -> None:
        """Background thread: drain send queue, POST to peers."""
        while True:
            msg = self._send_queue.get()
            peer = self._peer_registry.get_peer(msg.recipient)
            if peer is None:
                continue
            for attempt in range(3):
                try:
                    resp = httpx.post(
                        f"{peer.base_url}/v1/mesh/message",
                        json=msg.to_dict(),
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        break
                except httpx.RequestError:
                    time.sleep(min(1 * 2**attempt, 30))

    def receive(self, msg: MeshMessage) -> None:
        """Handle an incoming mesh message (called by API endpoint)."""
        # Reject unknown protocol versions
        if msg.protocol_version > 1:
            logger.warning("rejecting mesh message with protocol_version=%d", msg.protocol_version)
            return
        # Convert to Percept and publish on local bus
        percept = Percept(
            timestamp=msg.timestamp,
            source=f"mesh:{msg.sender}",
            content=json.dumps(msg.payload),
            salience=0.7,  # Peer messages are important but not critical
            metadata={
                "msg_type": msg.msg_type.name,
                "sender": msg.sender,
                "correlation_id": msg.correlation_id,
                "external": True,
            },
        )
        self._bus.publish(percept)
```

**API endpoints** (mount under `/v1/mesh/` on existing LeaderProxy to inherit auth + rate limiting):

```python
@app.post("/v1/mesh/message")
async def mesh_receive(request: Request):
    data = await request.json()
    msg = MeshMessage.from_dict(data)  # Raises ValueError on bad protocol_version
    peer_channel.receive(msg)
    return {"status": "ok"}
```

### Phase 4: Knowledge Sharing — Sovereign Exchange

The core cooperation mechanism. Any subsystem can share knowledge with peers through a generic **provider/receiver protocol**. The broker handles routing; each subsystem owns its own filtering, discounting, and import logic.

#### 4a. Design: KnowledgeProvider / KnowledgeReceiver protocol

The old design hardcoded CausalLinks and Reflections as the only shareable types, requiring a new method pair per type. The new design uses a registry of providers and receivers:

```python
class KnowledgeProvider(Protocol):
    """Any subsystem that can export shareable knowledge."""
    def knowledge_type(self) -> str: ...
    def get_shareable(self, limit: int = 10, **filters) -> list[dict]: ...

class KnowledgeReceiver(Protocol):
    """Any subsystem that can import knowledge from peers."""
    def knowledge_type(self) -> str: ...
    def import_knowledge(self, data: dict, source: str, trust: str) -> bool: ...
```

**Each subsystem owns its own logic:**
- NAc knows how to filter by confidence and apply transfer discounts
- Hippocampus knows how to cap salience on imported reflections
- DN could blend imported thresholds with local ones (future)
- Motor programs would check embodiment-spec similarity (future)

#### 4b. ExperienceBroker — generic registry

```python
class ExperienceBroker:
    """Routes knowledge between subsystems and peers.

    Subsystems register as providers and/or receivers. The broker
    dispatches EXPERIENCE_* messages to the right subsystem without
    knowing the specifics of each knowledge type.
    """

    def __init__(self) -> None:
        self._providers: dict[str, KnowledgeProvider] = {}
        self._receivers: dict[str, KnowledgeReceiver] = {}

    def register_provider(self, provider: KnowledgeProvider) -> None:
        self._providers[provider.knowledge_type()] = provider

    def register_receiver(self, receiver: KnowledgeReceiver) -> None:
        self._receivers[receiver.knowledge_type()] = receiver

    def get_shareable(
        self,
        knowledge_type: str | None = None,
        limit: int = 10,
        **filters,
    ) -> list[dict]:
        """Get shareable knowledge, optionally filtered by type."""
        if knowledge_type:
            provider = self._providers.get(knowledge_type)
            if provider is None:
                return []
            items = provider.get_shareable(limit=limit, **filters)
            return [{"knowledge_type": knowledge_type, **item} for item in items]
        # All types
        result = []
        for kt, provider in self._providers.items():
            items = provider.get_shareable(limit=limit, **filters)
            result.extend({"knowledge_type": kt, **item} for item in items)
        return result[:limit]

    def import_knowledge(
        self,
        knowledge_type: str,
        data: dict,
        source: str,
        trust: str,
    ) -> bool:
        """Route imported knowledge to the right receiver."""
        receiver = self._receivers.get(knowledge_type)
        if receiver is None:
            return False
        return receiver.import_knowledge(data, source, trust)

    @property
    def registered_types(self) -> list[str]:
        """Knowledge types that can be shared and/or received."""
        return sorted(set(self._providers) | set(self._receivers))
```

#### 4c. Built-in adapters (ship with Phase 4)

**CausalLinkProvider / CausalLinkReceiver** — wraps NAc:

```python
class CausalLinkProvider:
    SHARE_MIN_CONFIDENCE = 0.6
    SHARE_MIN_OBSERVATIONS = 3

    def __init__(self, nac: NAc) -> None:
        self._nac = nac

    def knowledge_type(self) -> str:
        return "causal_link"

    def get_shareable(self, limit: int = 10, **filters) -> list[dict]:
        tool_name = filters.get("tool_name")
        if tool_name:
            links = self._nac.get_links_for_event(f"tool:{tool_name}")
            return [
                link.to_dict() for link in links
                if link.confidence >= self.SHARE_MIN_CONFIDENCE
                and link.observation_count >= self.SHARE_MIN_OBSERVATIONS
            ][:limit]
        candidates = self._nac.get_promotion_candidates(
            min_confidence=self.SHARE_MIN_CONFIDENCE,
            min_observations=self.SHARE_MIN_OBSERVATIONS,
        )
        return [c.link.to_dict() for c in candidates[:limit]]


class CausalLinkReceiver:
    TRANSFER_DISCOUNTS: dict[str, float] = {
        "verified": 0.5,
        "discovered": 0.3,
        "remote": 0.3,
        "unknown": 0.1,
    }

    def __init__(self, nac: NAc) -> None:
        self._nac = nac

    def knowledge_type(self) -> str:
        return "causal_link"

    def import_knowledge(self, data: dict, source: str, trust: str) -> bool:
        from maxim.decisions.causal_link import CausalLink
        link = CausalLink.from_dict(data)

        # Deduplicate: skip if we already know this event→outcome
        existing = self._nac.get_links_for_event(link.event_signature)
        for ex in existing:
            if ex.outcome_signature == link.outcome_signature:
                return False

        # Apply trust-level transfer discount
        discount = self.TRANSFER_DISCOUNTS.get(trust, 0.1)
        link.confidence *= discount
        link.predicted_value = 0.5  # Reset R-W — local experience refines
        link.observation_count = 1

        # Tag provenance
        link.event_context["_imported_from"] = source
        link.event_context["_import_timestamp"] = time.time()
        link.event_context["_import_trust"] = trust

        self._nac._register_imported_link(link)
        return True
```

**ReflectionProvider / ReflectionReceiver** — wraps Hippocampus:

```python
class ReflectionProvider:
    def __init__(self, hippocampus: Hippocampus) -> None:
        self._hippocampus = hippocampus

    def knowledge_type(self) -> str:
        return "reflection"

    def get_shareable(self, limit: int = 10, **filters) -> list[dict]:
        memories = self._hippocampus.recall(limit=limit, tool=filters.get("tool_name"))
        result = []
        for m in memories:
            outcome = getattr(m, 'outcome', None)
            if hasattr(outcome, 'result') and isinstance(outcome.result, dict):
                if outcome.result.get("reflection"):
                    result.append(m.to_dict())
        return result


class ReflectionReceiver:
    SALIENCE_CAPS: dict[str, float] = {
        "verified": 0.5,
        "discovered": 0.35,
        "remote": 0.35,
        "unknown": 0.15,
    }

    def __init__(self, hippocampus: Hippocampus) -> None:
        self._hippocampus = hippocampus

    def knowledge_type(self) -> str:
        return "reflection"

    def import_knowledge(self, data: dict, source: str, trust: str) -> bool:
        from maxim.memory.types import EpisodicMemory
        memory = EpisodicMemory.from_dict(data)

        cap = self.SALIENCE_CAPS.get(trust, 0.15)
        memory.perception.salience = min(memory.perception.salience, cap)
        memory.perception.observations["_imported_from"] = source
        memory.perception.observations["_import_trust"] = trust

        return self._hippocampus.capture(record=memory) is not None
```

#### 4d. Wire protocol: EXPERIENCE_* payload schema

```python
# EXPERIENCE_OFFER payload:
{
    "knowledge_type": "causal_link",       # Required: which subsystem
    "items": [{ ... }, { ... }],           # Serialized knowledge items
    "count": 5,                            # How many items offered
}

# EXPERIENCE_REQUEST payload:
{
    "knowledge_type": "causal_link",       # What you want
    "filters": {"tool_name": "grasp"},     # Optional filters
    "limit": 10,                           # Max items requested
}

# EXPERIENCE_RESPONSE payload:
{
    "knowledge_type": "causal_link",
    "items": [{ ... }],
    "count": 3,
}
```

#### 4e. Future adapters (register when their subsystems ship)

| Knowledge type | Provider source | Receiver import logic | Trust floor |
|---|---|---|---|
| `causal_link` | NAc high-confidence links | Transfer discount + R-W reset | 0.1 |
| `reflection` | Hippocampus reflections | Salience cap by trust level | 0.1 |
| `adaptive_threshold` | DN threshold state | Blend with local (EMA, not replace) | 0.3 |
| `motor_program` | Embodiment ProgramRegistry | Spec-similarity gate before accept | 0.5 |
| `forward_model` | Cerebellum model params | Only accept if same entity type | 0.5 |
| `cascade_dynamics` | DM campaign cascade stats | Accept if entity types match; merge via EMA with local stats | 0.3 |
| `component_tuning` | SEM component DB tuned params | Accept if `verified_in` campaign list non-empty; merge with local component | 0.3 |
| `capability_snapshot` | CapabilityAgent | Read-only (no import — just routing info) | — |
| `contact` | CommunicationGateway | Merge into local contact registry | 0.5 |

Each future type is ~30-50 LOC (a provider class + a receiver class). The broker and wire protocol stay unchanged.

#### 4f. Transfer discount rationale

| Source | Default discount | Why |
|--------|-----------------|-----|
| Local experience | 1.0 | First-hand observation in this context |
| Verified peer (same owner) | 0.5 | Same tools, but different context/state |
| Discovered/remote peer | 0.3 | Different capabilities may invalidate patterns |
| Unknown | 0.1 | Unknown provenance |

Per-type overrides are possible (e.g., motor programs require 0.5 minimum trust). The AdaptivePlanner already uses NAc confidence in its scoring — imported links with low confidence naturally rank below locally-learned links.

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

    def __init__(self, peer_registry: PeerRegistry, mesh_channel: PeerChannel, local_agent_id: str) -> None:
        self._peers = peer_registry
        self._channel = mesh_channel
        self._local_agent_id = local_agent_id
        self._pending: dict[str, dict] = {}  # correlation_id -> {"event": Event, "result": dict|None}

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

    def delegate(self, goal: dict, peer: PeerInfo) -> dict | None:
        """Send a goal to a peer and block until result or timeout.

        Synchronous — the agent loop is sync, so this uses a threading.Event
        to wait for the correlated response. Returns the result dict or None.
        """
        correlation_id = uuid4().hex[:12]

        # Inject delegation tracking into payload
        goal.setdefault("delegation_depth", 0)
        goal["delegation_depth"] += 1
        goal.setdefault("delegation_chain", [])
        goal["delegation_chain"].append(self._local_agent_id)

        self._pending[correlation_id] = {"event": threading.Event(), "result": None}

        self._channel.send(peer.node_id, json.dumps(goal), {
            "msg_type": MeshMessageType.GOAL_PROPOSAL.name,
            "correlation_id": correlation_id,
        })

        # Block until response arrives or timeout
        timeout = goal.get("timeout_s", 60.0)
        self._pending[correlation_id]["event"].wait(timeout=timeout)
        result = self._pending.pop(correlation_id, {}).get("result")
        return result

    def handle_response(self, correlation_id: str, payload: dict) -> None:
        """Called by the bus handler when a GOAL_RESULT/GOAL_ACCEPTED/GOAL_REJECTED
        arrives with a matching correlation_id."""
        pending = self._pending.get(correlation_id)
        if pending:
            pending["result"] = payload
            pending["event"].set()
```

**On the receiving end:**

```python
class TaskReceiver:
    """Evaluates and optionally executes delegated goals from peers.

    Uses the local AdaptivePlanner to assess whether the goal is
    feasible, then executes it through the normal agent loop.
    """

    # Reject delegated goals when local queue exceeds this depth
    MAX_DELEGATION_QUEUE = 5

    def __init__(self, planner, executor, nac, local_agent_id: str, worker_pool=None) -> None:
        self._planner = planner
        self._executor = executor
        self._nac = nac
        self._local_agent_id = local_agent_id
        self._worker_pool = worker_pool
        self._active_delegations: int = 0

    def evaluate_proposal(self, goal: dict, sender_id: str) -> tuple[bool, str]:
        """Decide whether to accept a delegated goal.

        Criteria (checked in order):
        1. Delegation loop detection (depth > 2 or cycle in chain)
        2. Are we overloaded? (hard cap on concurrent delegations)
        3. Do we have the required tool? (hard requirement)
        4. Does NAc predict success? (soft preference)
        """
        # Delegation loop detection
        depth = goal.get("delegation_depth", 0)
        chain = goal.get("delegation_chain", [])
        if depth > 2:
            return False, f"delegation too deep (depth={depth})"
        if self._local_agent_id in chain:
            return False, f"delegation cycle detected (already in chain)"

        # Queue depth check — reject if already saturated
        if self._active_delegations >= self.MAX_DELEGATION_QUEUE:
            return False, f"overloaded ({self._active_delegations} active delegations)"

        tool_name = goal.get("tool_name")
        if not tool_name:
            return False, "no tool specified"

        # Check tool availability
        if not self._executor.has_tool(tool_name):
            return False, f"tool {tool_name} not available"

        # Check NAc prediction — reject if high-confidence negative prediction
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
    admission: MeshAdmissionControl | None = None,
) -> list[dict]:
    """Tag sub-goals with preferred_node if a peer is better suited.

    Skips peers that are dead, identity-less, or currently gated.
    """
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
            # Skip gated peers — they're misbehaving or overloaded
            if admission and admission.is_peer_gated(peer.node_id):
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

### Mesh API Endpoints

Mount under `/v1/mesh/` on the existing LeaderProxy to inherit auth, rate limiting, and GPU metrics. No new HTTP server needed.

```python
# GET  /v1/mesh/identity    — Return this agent's AgentIdentity
# POST /v1/mesh/message     — Receive a MeshMessage (protocol_version checked)
# GET  /v1/mesh/peers       — List known peers (id, name, trust, alive)
# POST /v1/mesh/query/predict    — "What would happen if I did X?"
# POST /v1/mesh/query/experience — "What do you know about tool X?"
# GET  /v1/mesh/status      — Cluster status (peer count, message rates, gated peers)
```

### Phase 3b: Mesh Admission Control

Per-peer rate limiting, penalty tracking, and gating for misbehaving peers. Protects the mesh from flooding, whether accidental (buggy peer in a tight loop) or malicious.

```python
@dataclass
class PeerAdmissionState:
    """Tracks a single peer's behavior for admission decisions."""
    peer_id: str
    trust_level: str                       # "verified" | "discovered" | "remote" | "unknown"
    messages_received: int = 0             # Total messages received from this peer
    messages_this_window: int = 0          # Messages in current rate-limit window
    window_start: float = 0.0             # Start of current window (monotonic)
    burst_timestamps: list[float] = field(default_factory=list)  # Recent message times for burst detection
    violations: int = 0                    # Cumulative rate-limit violations
    gated_until: float = 0.0              # Monotonic time when gate lifts (0 = not gated)
    gate_reason: str = ""                 # Why this peer was gated

    @property
    def is_gated(self) -> bool:
        return time.monotonic() < self.gated_until


class MeshAdmissionControl:
    """Rate limiter and circuit breaker for incoming mesh messages.

    Sits between the /v1/mesh/message endpoint and PeerChannel.receive().
    Peers that exceed rate limits accumulate violations; repeated
    violations trigger escalating gate durations.
    """

    # Rate limit: messages per peer per window
    DEFAULT_RATE_LIMIT = 60               # messages per window
    WINDOW_SECONDS = 60.0                 # 1-minute sliding window

    # Escalating gate durations (indexed by violation count, capped at last)
    GATE_DURATIONS = [30, 120, 600, 3600]  # 30s, 2min, 10min, 1hr

    # Auto-gate triggers (any single condition triggers a gate)
    BURST_THRESHOLD = 20                  # Messages in 5 seconds = burst
    BURST_WINDOW = 5.0

    def __init__(self, rate_limit: int = DEFAULT_RATE_LIMIT) -> None:
        self._rate_limit = rate_limit
        self._peers: dict[str, PeerAdmissionState] = {}
        self._lock = threading.Lock()

    def check(self, peer_id: str, trust_level: str = "unknown") -> tuple[bool, str]:
        """Check if a message from this peer should be admitted.

        Returns (admitted: bool, reason: str).
        Called on every incoming mesh message before dispatching.
        """
        with self._lock:
            state = self._peers.get(peer_id)
            if state is None:
                state = PeerAdmissionState(peer_id=peer_id, trust_level=trust_level)
                self._peers[peer_id] = state

            now = time.monotonic()

            # Check if peer is currently gated
            if state.is_gated:
                return False, f"gated until {state.gated_until - now:.0f}s ({state.gate_reason})"

            # Reset window if expired
            if now - state.window_start > self.WINDOW_SECONDS:
                state.messages_this_window = 0
                state.window_start = now

            state.messages_received += 1
            state.messages_this_window += 1

            # Burst detection — sliding window of recent timestamps
            state.burst_timestamps.append(now)
            burst_cutoff = now - self.BURST_WINDOW
            state.burst_timestamps = [t for t in state.burst_timestamps if t > burst_cutoff]
            if len(state.burst_timestamps) > self.BURST_THRESHOLD:
                state.violations += 1
                duration = self.GATE_DURATIONS[min(state.violations - 1, len(self.GATE_DURATIONS) - 1)]
                state.gated_until = now + duration
                state.gate_reason = f"burst detected ({len(state.burst_timestamps)} msgs in {self.BURST_WINDOW}s)"
                logger.warning("mesh: gating peer %s for %ds — %s", peer_id, duration, state.gate_reason)
                return False, state.gate_reason

            # Check per-window rate limit
            if state.messages_this_window > self._rate_limit:
                state.violations += 1
                duration = self.GATE_DURATIONS[min(state.violations - 1, len(self.GATE_DURATIONS) - 1)]
                state.gated_until = now + duration
                state.gate_reason = f"rate limit exceeded ({state.messages_this_window}/{self._rate_limit} in {self.WINDOW_SECONDS}s)"
                logger.warning("mesh: gating peer %s for %ds — %s", peer_id, duration, state.gate_reason)
                return False, state.gate_reason

            return True, "ok"

    def gate_peer(self, peer_id: str, duration_s: float, reason: str) -> None:
        """Manually gate a peer (e.g., from an admin command or anomaly detector)."""
        with self._lock:
            state = self._peers.get(peer_id)
            if state is None:
                state = PeerAdmissionState(peer_id=peer_id, trust_level="unknown")
                self._peers[peer_id] = state
            state.gated_until = time.monotonic() + duration_s
            state.gate_reason = reason
            state.violations += 1
            logger.warning("mesh: manually gating peer %s for %ds — %s", peer_id, duration_s, reason)

    def is_peer_gated(self, peer_id: str) -> bool:
        """Quick check used by distributed planning to skip gated peers."""
        with self._lock:
            state = self._peers.get(peer_id)
            return state.is_gated if state else False

    def ungate_peer(self, peer_id: str) -> None:
        """Manually lift a gate (e.g., after the peer is fixed)."""
        with self._lock:
            state = self._peers.get(peer_id)
            if state:
                state.gated_until = 0.0
                state.gate_reason = ""

    def get_status(self) -> list[dict[str, Any]]:
        """Return admission state for all known peers (for /v1/mesh/status)."""
        with self._lock:
            return [
                {
                    "peer_id": s.peer_id,
                    "trust_level": s.trust_level,
                    "messages_received": s.messages_received,
                    "violations": s.violations,
                    "is_gated": s.is_gated,
                    "gate_reason": s.gate_reason,
                }
                for s in self._peers.values()
            ]
```

**Integration point:** The `/v1/mesh/message` endpoint calls `admission.check(sender_id, trust_level)` before passing the message to `PeerChannel.receive()`. Rejected messages get a 429 response with the gate reason.

**Trust level assignment:** Determined from auth context on the incoming request:
- Pre-shared key match → `"verified"`
- mDNS-discovered peer → `"discovered"`
- Cloudflare tunnel auth → `"remote"` (upgrade to `"verified"` if key matches)
- No auth → `"unknown"`

| File | Change | LOC |
|---|---|---|
| `mesh/admission.py` | **New.** `MeshAdmissionControl`, `PeerAdmissionState` | ~120 |
| Phase 3 endpoint | Add `admission.check()` before `receive()` | ~10 |
| `doctor/checks.py` | New "mesh admission status" check (gated peer count) | ~15 |

### Phase 7: SCN Temporal Coordination

Use the existing SCN (Suprachiasmatic Nucleus) to handle clock skew between peers. The SCN already works in **phase-normalized time** (0.0–1.0 for each rhythm: circadian, weekly, monthly, annual) rather than absolute timestamps — this makes it naturally resistant to clock offsets. The Kuramoto oscillator network can learn inter-agent phase coupling.

#### Why SCN fits

1. **Phase-normalized indexing.** SCN bins memories by circadian/weekly/monthly/annual phases, not raw timestamps. Two agents with clocks offset by N seconds will compute different phases for the same real-world event, but the offset is stable and learnable.
2. **Kuramoto coupling.** The oscillator network already learns how phases co-activate. Adding a peer's clock as another coupled oscillator lets the network learn inter-agent drift over time.
3. **Temporal anomaly detection.** `temporal_anomaly_score()` already flags memories that occur at unusual times. Cross-agent events with anomalous timing relative to the local clock suggest skew.

#### Design

```python
@dataclass
class PeerClockEstimate:
    """Estimated clock offset for a peer agent."""
    peer_id: str
    offset_s: float = 0.0          # Estimated: peer_time - local_time
    drift_rate: float = 0.0        # Seconds per hour of drift
    confidence: float = 0.0        # 0.0 = no data, 1.0 = many sync points
    sync_points: int = 0           # Number of round-trips used to estimate
    last_sync: float = 0.0         # Local monotonic time of last sync


class PeerClockEstimator:
    """Learns clock offsets between local SCN and peer agents.

    Uses heartbeat round-trip times to estimate offset, then
    corrects incoming TemporalSignatures before SCN registration.
    """

    def __init__(self, scn: SCN) -> None:
        self._scn = scn
        self._estimates: dict[str, PeerClockEstimate] = {}

    def record_sync_point(self, peer_id: str, peer_timestamp: float, rtt_s: float) -> None:
        """Update clock estimate from a heartbeat round-trip.

        peer_timestamp: the peer's reported wall-clock time
        rtt_s: measured round-trip time for the heartbeat exchange
        """
        local_time = time.time()
        # Estimate peer's current time (NTP-lite: subtract half RTT)
        estimated_peer_now = peer_timestamp + rtt_s / 2
        offset = estimated_peer_now - local_time

        est = self._estimates.get(peer_id)
        if est is None:
            est = PeerClockEstimate(peer_id=peer_id)
            self._estimates[peer_id] = est

        # EMA of offset (α = 0.3 gives ~3 sync points to stabilize)
        alpha = 0.3
        if est.sync_points == 0:
            est.offset_s = offset
        else:
            est.offset_s = alpha * offset + (1 - alpha) * est.offset_s

        est.sync_points += 1
        est.confidence = min(1.0, est.sync_points / 10)  # Full confidence after 10 syncs
        est.last_sync = time.monotonic()

    def correct_timestamp(self, peer_id: str, peer_timestamp: float) -> float:
        """Convert a peer's timestamp to local-clock equivalent."""
        est = self._estimates.get(peer_id)
        if est is None or est.confidence < 0.1:
            return peer_timestamp  # No correction possible yet
        return peer_timestamp - est.offset_s

    def correct_signature(self, peer_id: str, sig: TemporalSignature) -> TemporalSignature:
        """Correct a peer's TemporalSignature for clock skew.

        Re-derives phases from the corrected timestamp so SCN bins
        align with local time.
        """
        corrected_ts = self.correct_timestamp(peer_id, sig.timestamp)
        return TemporalSignature.from_timestamp(corrected_ts)
```

**Integration with SCN:**

```python
# In SCN, add a method for registering peer memories:
def register_external(
    self,
    memory_id: str,
    signature: TemporalSignature,
    peer_id: str,
    clock_estimator: PeerClockEstimator,
    significance: float = 0.5,
) -> None:
    """Register a memory from a peer with clock skew correction."""
    corrected = clock_estimator.correct_signature(peer_id, signature)
    self.register(memory_id, corrected, significance)
```

**Sync point source:** HEARTBEAT messages include the sender's wall-clock time. The receiver measures RTT from the heartbeat request/response cycle. Each heartbeat refines the clock estimate.

| File | Change | LOC |
|---|---|---|
| `mesh/clock.py` | **New.** `PeerClockEstimator`, `PeerClockEstimate` | ~80 |
| `time/scn.py` | Add `register_external()` method | ~10 |
| `time/temporal_signature.py` | Add optional `source_agent_id` field | ~5 |
| Phase 3 heartbeat handler | Call `estimator.record_sync_point()` on each heartbeat | ~10 |

---

## Implementation Sequencing

| Phase | What | Effort | Dependencies |
|-------|------|--------|-------------|
| **Pre** | NAc `_register_imported_link()` + serialization for ProposedGoal, SubGoal, RuntimeCapabilities, ToolResult | Small | None |
| **1** | `AgentIdentity` dataclass (extends `AgentProfile`) with `build_from_agent()` | Small | None |
| **2** | `MeshMessage` protocol expansion + `protocol_version` + `correlation_id` + GOAL_PROPOSAL payload schema | Small | None |
| **3** | `PeerChannel` (async send queue) + `PeerRegistry` (from peer config) + mesh endpoints on `/v1/mesh/*` | Medium | Phase 2 |
| **3b** | `MeshAdmissionControl` — per-peer rate limiting, burst detection, gating | Small | Phase 3 |
| **4** | `ExperienceBroker` + `KnowledgeProvider`/`KnowledgeReceiver` protocol + CausalLink + Reflection adapters | Medium | Pre, Phase 2 |
| **5** | `TaskDelegator` + `TaskReceiver` — goal delegation with queue depth check | Medium | Pre, Phases 2, 3 |
| **6** | Distributed planning — `_tag_delegatable_subgoals()` in AdaptivePlanner (skips gated peers) | Small | Phases 3b, 5 |
| **7** | SCN temporal coordination — `PeerClockEstimator`, `register_external()` | Medium | Phase 3 |
| **0a** | mDNS discovery — additional peer source for PeerRegistry (optional, LAN-only) | Medium | Phase 3 |
| **0b** | InferenceRouter — per-request backend selection | Medium | Phase 0a |

**Recommended order:**
1. **Pre-work** (serialization + `_register_imported_link`) — unblocks everything, no network yet
2. **Phase 1 + 2** (identity + protocol) — foundations
3. **Phase 3 + 3b** (transport + admission control) — agents can talk, safely
4. **Phase 4** (knowledge sharing) — agents learn from each other
5. **Phase 5 + 6** (task delegation + distributed planning) — agents cooperate
6. **Phase 7** (SCN temporal coordination) — cross-agent temporal reasoning
7. **Phase 0a + 0b** (mDNS + InferenceRouter) — LAN auto-discovery, defer until multiple LAN machines exist

**Why this order:** The old sequencing had serialization (Phase 8) last, but Phase 5 (task delegation) needs `ProposedGoal.to_dict()` and `ToolResult.to_dict()` to work. Serialization is now pre-work. mDNS (0a/0b) is deferred — the existing Cloudflare tunnel setup already handles peer routing; mDNS only matters when multiple LAN machines join.

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

These discounts are defined in `ExperienceSharer.TRANSFER_DISCOUNTS` and used by `MeshAdmissionControl` for trust assignment. A peer's trust level is determined once at first contact and stored in `PeerAdmissionState.trust_level`.

| Relationship | Discount | How established | Admission rate limit |
|-------------|---------|-----------------|---------------------|
| **Self** | 1.0 | — | — |
| **Verified peer** (same owner, shared secret) | 0.5 | Pre-shared key in config | 120 msg/min |
| **Discovered peer** (LAN mDNS) | 0.3 | Automatic, no verification | 60 msg/min |
| **Remote peer** (Cloudflare tunnel) | 0.3 | Tunnel auth serves as verification | 60 msg/min |
| **Unknown** | 0.1 | Unsolicited contact | 20 msg/min |

---

## Interaction with Other Plans

| This plan | Other plan | Interaction |
|-----------|-----------|-------------|
| Phase 3 (PeerChannel) | `peer/config.py` (existing tunnel config) | PeerRegistry bootstraps from existing peer config; Phase 0a adds mDNS |
| Phase 3b (admission control) | Multi-LLM Phase 7b (admission control) | Same pattern as LeaderProxy rate limiting, extended to mesh messages |
| Phase 5 (task delegation) | Decision Engine (AdaptivePlanner) | Delegated goals evaluated by same planner |
| Phase 4 (knowledge sharing) | Causal Memory (NAc links) | CausalLinkProvider/Receiver wraps NAc; uses `_register_imported_link()` |
| Phase 4 (knowledge sharing) | Hippocampus (reflections) | ReflectionProvider/Receiver wraps recall/capture |
| Phase 4 (knowledge sharing) | Default Network (future) | AdaptiveThresholdProvider — registers when DN MVP ships |
| Phase 4 (knowledge sharing) | Embodiment (complete) | MotorProgramProvider — Cerebellum forward models, gated by entity-type similarity |
| Phase 4 (knowledge sharing) | DM MVP (ready) | CascadeDynamicsProvider — shares observed cascade resolution stats across campaign runs. ComponentTuningProvider — propagates simulation-tuned SEM component parameters. See [DM plan](dungeon_master_persona.md) cross-plan section. |
| Phase 5-6 (delegation + planning) | DM Extensions — multi-AUT party mode | Multiple AUTs control party characters; DM delegates encounter choices via TaskDelegator; cascade resolves across all AUT actions |
| Phase 7 (temporal coordination) | DM campaign timelines | Campaign events have natural temporal structure; PeerClockEstimator keeps multi-AUT SCN bins aligned for shared timeline queries |
| Phase 4 (knowledge sharing) | Doctor/CapabilityAgent (future) | CapabilityProvider — read-only broadcast, no import |
| Phase 7 (temporal coordination) | SCN (`time/scn.py`) | `PeerClockEstimator` corrects incoming TemporalSignatures; SCN gains `register_external()` |
| Phase 7 (temporal coordination) | Kuramoto oscillator (`time/oscillator.py`) | Future: treat peer clocks as coupled oscillators for drift learning |
| Phase 1 (AgentIdentity) | Adaptive Runtime (RuntimeCapabilities) | Identity includes serialized capabilities |
| Phase 1 (AgentIdentity) | Embodiment Core (complete) | `AgentIdentity.embodiment_summary` advertises body: modalities, affordances, hardware-backed vs. imagined. SEM entities (characters, objects) share the same protocol. |
| Pre-work (serialization) | `agents/bus.py` (ProposedGoal, SubGoal, ToolResult) | Add to_dict/from_dict to existing dataclasses — no behavioral changes |
| Future (not scheduled) | Embodiment — federation, affordance delegation, NAc transfer | Cross-agent affordance invocation, federated bodies (components from multiple peers), CausalLink transfer gated by spec similarity. Tracked in `future_plans.md`. |

---

## Risks

1. **Knowledge poisoning.** A malicious peer could send fabricated CausalLinks that cause bad decisions. **Mitigation:** Transfer discount reduces imported confidence (trust-level aware — verified=0.5, unknown=0.1). Locally-learned links always dominate. Links tagged with `_imported_from` + `_import_trust` can be audited and purged.

2. **Goal delegation loops.** Agent A delegates to B, B delegates back to A. **Mitigation:** Goals carry a `delegation_depth` counter. Reject goals with depth > 2. Include `delegation_chain: list[str]` to detect cycles.

3. **Network partition.** Peer goes offline mid-task. **Mitigation:** Task delegation has a timeout. AdaptivePlanner's replan loop handles delegation failure like any tool failure — decomposes and retries locally or on another peer. PeerChannel's send queue retries with exponential backoff (3 attempts max).

4. **Memory bloat from imports.** Accepting too many peer reflections floods the hippocampus. **Mitigation:** Rate-limit imports (max 10 per peer per hour). Imported memories have reduced salience so they're evicted first during consolidation.

5. **Privacy leakage.** AgentIdentity broadcasts tool lists and domain summaries. On a shared LAN this reveals what the agent is doing. **Mitigation:** Identity broadcast is opt-in (`MAXIM_PEER_ADVERTISE=1`). Sensitive fields can be omitted. Cloudflare tunnel peers use zero-trust auth.

6. **Clock skew.** Agents on different machines may have different clocks, affecting temporal reasoning in SCN and memory ordering. **Mitigation:** Phase 7 introduces `PeerClockEstimator` which uses heartbeat round-trip times (NTP-lite) to estimate per-peer offsets. The SCN's phase-normalized indexing is naturally resistant to small offsets — skew only matters when it shifts a memory into the wrong hour bin (>30 minutes). Incoming `TemporalSignature`s from peers are corrected before SCN registration via `register_external()`.

7. **Peer flooding / misbehavior.** A buggy or malicious peer sends a burst of messages that overwhelms the local agent. **Mitigation:** `MeshAdmissionControl` (Phase 3b) rate-limits per peer (60 msg/min default), with escalating gate durations on violations (30s → 2min → 10min → 1hr). Gated peers receive 429 responses. Manual gate/ungate available via `gate_peer()` / `ungate_peer()`. Burst detection triggers immediate gating (20 messages in 5 seconds). Peer admission status visible via `/v1/mesh/status` and `maxim doctor`.

8. **Protocol version mismatch.** Peers running different Maxim versions may send incompatible messages. **Mitigation:** `MeshMessage.protocol_version` field (added in Phase 2). Receivers reject messages with versions higher than their own. Lower versions accepted for backward compat. Version bumped only on breaking payload changes.
