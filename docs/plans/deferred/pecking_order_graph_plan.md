# Pecking Order Graph Plan

> **Status:** DEFERRED (post-1.0). Not started. Design phase complete.
>
> **Revive when:** (1) [substrate_plan.md](../archive/substrate_plan.md) A6 convergence harnesses pass, AND (2) a second real node exists in the mesh that isn't the Mac peer (i.e., there's an actual multi-node topology to unify). Until then, the single-leader + peer model is sufficient and POG's 1,200 LOC would be abstracting over a structure that doesn't exist yet.
>
> **Note on overlap with [cleanup_wave.md](../archive/cleanup_wave.md) (archived) C4:** Agent Permissions ships standalone in the cleanup wave. When POG is revived, it consumes the permissions layer via its AUTHORITY domain rather than replacing it.
> **Goal:** Unify leader/peer roles, mesh topology, compute routing, and Mother Maxim hierarchy into a single directed graph with domain-scoped pecking relationships on each edge.
> **Depends on:** PyPI publication (v1.0.0). Subsumes Mesh Phase 0a/0b, Capability Agent, Multi-Node Admin, and reshapes Mother Maxim's federation model.
> **Estimated scope:** ~1,200 LOC across 5 phases + prep items woven into publication.

---

## Vision

Today, Maxim has three separate hierarchy-like systems that don't talk to each other:

1. **Leader/Peer** — static, config-file-based. One GPU leader, N peers. No election, no failover.
2. **Mesh** — sovereign agents with trust levels and admission control, but flat topology. No hierarchy between peers.
3. **Mother Maxim** — planned as hub-and-spoke memory aggregator. Contributions flow up, queries flow down. But she's not in the mesh topology.

Plus **three separate gating mechanisms** that are all independent: LeaderProxy admission (rate limits + concurrency semaphore), Mesh admission (trust-level rate limits + burst detection), and LaneBackend gates (cloud caps + concurrent backend limits).

The Pecking Order Graph unifies all of this into a single data structure: a **rooted directed graph** where Mother Maxim is the root, every Maxim instance is a node, and each edge carries **domain-scoped pecking relationships** that determine who defers to whom for what.

### Why "pecking order" + graph?

A pure graph gives you topology (who connects to whom) but no decision rules. A pure pecking order gives you a linear ranking but can't express "Peer-2 has 192GB RAM and pecks its leader for memory workloads." Combining them:

- The **graph** is the topology — parent/child/sibling relationships between nodes.
- The **pecking order** is the directionality on each edge — who defers to whom, scoped by domain.
- Each node has a `pecked_by` (who has authority over me) and `pecks` (who I have authority over) for each domain.

This means a node can be pecked for compute (GPU leader dominates) but BE the pecker for memory (it has the richest episodic store or the most RAM). Authority flows one way (strict DAG, Mother at root), but capability-based pecking can be flexible.

---

## Core Data Model

### Node: `PeckingNode`

Every running Maxim instance registers as a node in the graph.

```python
@dataclass
class PeckingNode:
    """A Maxim instance in the pecking order graph."""

    node_id: str                        # Persistent (from AgentIdentity.node_id)
    agent_name: str                     # Human-readable ("reachy-kitchen", "desktop-mac")
    capabilities: RuntimeCapabilities   # Hardware: GPU, VRAM, RAM, tiers
    available_tiers: set[str]           # {"large", "medium", "small"}
    trust_level: str                    # "verified" | "discovered" | "remote" | "unknown"
    identity: AgentIdentity             # Full mesh identity (profile, knowledge stats, models)

    # Graph position
    parent_id: str | None = None        # Who registered me (None = root = Mother)
    children: set[str] = field(default_factory=set)  # Nodes that registered under me
    siblings: set[str] = field(default_factory=set)   # Peers at same level under same parent

    # Current state (updated via heartbeat)
    load: NodeLoad | None = None        # GPU util %, queue depth, thermal, RAM pressure
    version: str = ""                   # Maxim version (for update cascades)
    last_heartbeat: float = 0.0         # Monotonic timestamp
    is_alive: bool = True
```

### Edge: `PeckingRelation`

Each edge in the graph carries domain-scoped pecking relationships.

```python
class PeckingDomain(str, Enum):
    """Domains over which pecking order applies."""
    AUTHORITY = "authority"    # Who can push updates, gate actions, override decisions
    COMPUTE = "compute"       # Who handles inference requests (GPU routing)
    MEMORY = "memory"         # Who has richer memory / more RAM for context
    KNOWLEDGE = "knowledge"   # Who has more causal links / concepts (wisdom)
    EMBODIMENT = "embodiment" # Who has physical sensors / actuators

class PeckingDirection(str, Enum):
    PECKS = "pecks"           # I have authority over peer in this domain
    PECKED_BY = "pecked_by"   # Peer has authority over me in this domain
    MUTUAL = "mutual"         # Equals — route by load/latency (siblings)

@dataclass
class PeckingRelation:
    """Directional relationship between two nodes, scoped by domain."""
    source_id: str
    target_id: str
    domains: dict[PeckingDomain, PeckingDirection]
    latency_ms: float = 0.0     # Measured RTT (from PeerClockEstimator)
    bandwidth: str = "lan"      # "local" | "lan" | "tunnel" | "cloud"
    capacity: EdgeCapacity | None = None  # Current throughput limits

@dataclass
class EdgeCapacity:
    """Dynamic capacity constraints on an edge."""
    max_rps: float = 0.0            # Requests per second this edge can handle
    current_queue_depth: int = 0    # How backed up is the target?
    permission_flags: set[str] = field(default_factory=set)  # e.g. {"contribute", "query", "delegate"}
    gated_until: float = 0.0        # Monotonic time (0 = not gated)
    gate_reason: str = ""
```

### Graph: `PeckingGraph`

The graph itself, with routing and cascade operations.

```python
class PeckingGraph:
    """The pecking order topology.

    Invariants:
    - AUTHORITY domain is always a strict DAG (no cycles). Mother is root.
    - COMPUTE/MEMORY/KNOWLEDGE/EMBODIMENT can have flexible pecking
      between siblings (bidirectional based on capability).
    - Every node has exactly one parent in the AUTHORITY domain
      (except Mother, who has none).
    """

    nodes: dict[str, PeckingNode]
    edges: dict[tuple[str, str], PeckingRelation]  # (source, target)

    # --- Routing ---
    def find_pecker(self, node_id: str, domain: PeckingDomain) -> str | None:
        """Who does this node defer to for a given domain?
        Walks edges to find the nearest node that pecks this one."""

    def find_pecked(self, node_id: str, domain: PeckingDomain) -> list[str]:
        """Who does this node have authority over in a given domain?"""

    def route_request(self, from_node: str, domain: PeckingDomain,
                      requirements: dict) -> str | None:
        """Find the best node to handle a request.
        Walks toward root in the domain until finding a node with
        sufficient capability AND capacity. Falls back to siblings."""

    # --- Cascades ---
    def cascade_down(self, from_node: str, domain: PeckingDomain,
                     payload: Any) -> CascadeResult:
        """Push something down the graph (updates, wisdom, commands).
        Flows from pecker to pecked. Used for code updates, Mother wisdom."""

    def cascade_up(self, from_node: str, domain: PeckingDomain,
                   payload: Any) -> CascadeResult:
        """Push something up the graph (contributions, reports, metrics).
        Flows from pecked to pecker. Used for memory contributions."""

    # --- Gating ---
    def check_gate(self, from_node: str, to_node: str,
                   action: str) -> GateResult:
        """Can from_node perform action on/through to_node?
        Checks: edge capacity, permission flags, gate status, trust level."""

    # --- Topology ---
    def register(self, node: PeckingNode, parent_id: str) -> None:
        """Register a new node under a parent. Auto-computes pecking
        relations from capabilities comparison."""

    def deregister(self, node_id: str) -> None:
        """Remove a node. Orphaned children re-parent to grandparent."""

    def recompute_pecking(self, node_id: str) -> None:
        """Recalculate domain-specific pecking for a node based on
        current capabilities. Called on heartbeat capability changes."""
```

---

## Pecking Computation Rules

When a node registers (or capabilities change), pecking relations are computed per-domain:

### Authority (strict DAG)
- Parent always pecks child. No exceptions.
- Mother pecks all. She is the root.
- Authority cannot be earned by capability — it's structural (graph position).
- This ensures update cascades, memory governance, and abuse enforcement have a clear chain of command.

### Compute
- Compare `available_tiers` and `load.gpu_util`:
  - Node with `large` tier pecks nodes with only `medium`/`small`
  - Between nodes with same tiers, the one with lower `gpu_util` pecks (it has spare capacity)
  - Node with no GPU is always pecked for compute
- Tie-breaker: lower `latency_ms` on the edge wins
- Re-evaluated on every heartbeat (capabilities change dynamically)

### Memory
- Compare `identity.knowledge_stats.episodic_count` + `capabilities.ram_gb`:
  - More episodes + more RAM = pecks for memory
  - Mother always pecks for memory (she has the collective)
- This determines who gets queried for recall, who contributes up

### Knowledge
- Compare `identity.knowledge_stats.causal_link_count` + `concept_count`:
  - More links + concepts = pecks for knowledge
  - Mother always pecks for knowledge (consensus model)
- Determines who gets asked for predictions, who receives wisdom

### Embodiment
- Node with physical sensors/actuators pecks nodes without
- Among embodied nodes, the one with richer SEM entity tree pecks
- Disembodied nodes (pure compute) are always pecked for embodiment
- Determines who gets delegated physical tasks

---

## What This Replaces/Subsumes

| Currently planned | Becomes | Savings |
|---|---|---|
| **Static leader/peer role detection** (`detect_role()` in `leader_mode.py`) | Node registers in graph; "leader" = node that others register under. No config file needed for role — role emerges from graph position. `detect_role()` becomes `graph.get_role(node_id)`. | Simpler role model, no cloudflared config sniffing |
| **Capability Agent** (future_plans.md, ~500 LOC) | The graph IS the capability map. `can_run_model()` = walk graph for a node with that tier. `gate_action()` = `graph.check_gate()`. `recommended_tier()` = `graph.route_request(domain=COMPUTE)`. | ~300 LOC saved; CA was reimplementing graph queries |
| **InferenceRouter** (mesh Phase 0b, ~250 LOC deferred) | `graph.route_request(domain=COMPUTE, requirements={"tier": "large"})` walks toward root, checks capacity. Local -> LAN peer -> tunnel -> cloud is now just graph distance. | Unified with routing, not a separate layer |
| **Multi-Node Admin** (future_plans.md, ~200 LOC) | `graph.cascade_down(domain=AUTHORITY, payload=UpdateCommand)`. Each edge checks permission_flags for "update". Mother can cascade to all, leaders cascade to their children. | Fan-out is built into the graph, not a separate registry |
| **Three separate gating systems** | `graph.check_gate()` unifies: edge capacity (replaces LeaderProxy semaphore), permission flags (replaces mesh admission trust checks), and gate status (replaces `MeshAdmissionControl` escalation). Mesh admission state migrates into `EdgeCapacity`. | One gating model instead of three |
| **Mother's contribution/query topology** | Contributions cascade up to Mother (through enrichment at each level). Queries cascade down from Mother. The graph IS the federation topology. | Mother plan M-5 API routing becomes graph-native |
| **mDNS discovery** (mesh Phase 0a, ~200 LOC deferred) | Discovery adds nodes to the graph. The graph handles the "now what?" — pecking computation assigns roles automatically. | Discovery is just `graph.register()`, not a separate system |

---

## Cascade Protocols

### Update Cascade (Authority domain, downward)

```
Mother pushes v0.2.1
  → cascade_down(domain=AUTHORITY, payload=UpdateCommand("v0.2.1"))
    → Leader-A receives, validates, applies
      → cascade_down to Peer-1, Peer-2
        → Each validates, applies, reports status up
    → Leader-B receives, validates, applies
      → cascade_down to Peer-3
```

Each node:
1. Checks `edge.permission_flags` includes `"update"` — skip if not authorized
2. Checks `edge.capacity.gated_until` — block if gated
3. Applies update locally (git pull + pip install, reusing existing admin logic)
4. Reports result back up the graph (success/failure/version mismatch)
5. Only cascades to children after local success (prevents propagating broken updates)

Failure at any node does NOT block siblings. The graph tracks which nodes succeeded (version field on `PeckingNode`). Operator can see: "Mother: v0.2.1, Leader-A: v0.2.1, Peer-1: v0.2.0 (failed: disk full)".

### Memory Contribution Cascade (Knowledge domain, upward)

```
Peer-1 finishes a campaign, has new causal links
  → cascade_up(domain=KNOWLEDGE, payload=ContributionBundle)
    → Leader-A receives
      → Leader-A's NAc evaluates links against its own observations
      → Links that align with Leader-A's experience get confidence boost
      → Links that contradict get flagged
      → Enriched bundle cascades up to Mother
        → Mother applies coalescence (M-4 rules)
        → Consensus links propagate back down via wisdom cascade
```

This is the key insight: **knowledge doesn't jump directly to Mother**. It percolates through intermediate nodes, getting enriched at each level. A causal link observed by Peer-1, confirmed by Leader-A's independent experience, arrives at Mother with stronger evidence than a raw single-agent contribution.

### Wisdom Cascade (Knowledge domain, downward)

```
Mother has a high-confidence consensus link
  → cascade_down(domain=KNOWLEDGE, payload=WisdomBundle)
    → Leader-A receives, applies trust discount (verified=0.5)
      → Leader-A cascades to Peer-1, Peer-2 with additional discount
        → Each applies their own trust discount
```

Transfer discounts compound through the graph. A link from Mother (confidence 0.9) arrives at a grandchild peer as `0.9 * 0.5 * 0.5 = 0.225`. This naturally attenuates the further you are from the source — local experience always dominates distant wisdom.

### Compute Routing (Compute domain, upward)

```
Peer-2 needs large-tier inference, has only small tier
  → route_request(domain=COMPUTE, requirements={"tier": "large"})
    → Check parent (Leader-A): has large tier, gpu_util=45% → route here
    → If Leader-A overloaded (gpu_util=95%):
      → Check siblings: Peer-1 has no large tier → skip
      → Check grandparent (Mother): has large tier → route here
      → If Mother also overloaded: check Mother's other children (Leader-B)
        → Leader-B has large tier, gpu_util=20% → route here (cross-branch)
```

The graph naturally implements the deferred InferenceRouter's priority chain (local -> parent -> sibling -> grandparent -> cross-branch) without a separate routing layer.

---

## Gating Model

### Unified Gate Check

Every action between nodes passes through `graph.check_gate()`:

```python
@dataclass
class GateResult:
    allowed: bool
    reason: str = ""
    alternative: str | None = None    # Suggested alternative node
    retry_after: float = 0.0         # Seconds until gate lifts
    backpressure: float = 0.0        # 0.0-1.0, how much to slow down

class PeckingGraph:
    def check_gate(self, from_node: str, to_node: str,
                   action: str) -> GateResult:
        edge = self.edges.get((from_node, to_node))
        if not edge:
            return GateResult(False, "no edge between nodes")

        # 1. Permission check
        if action not in edge.capacity.permission_flags:
            return GateResult(False, f"no {action} permission on this edge")

        # 2. Gate check (escalating, like current MeshAdmissionControl)
        if edge.capacity.gated_until > time.monotonic():
            return GateResult(False, edge.capacity.gate_reason,
                              retry_after=edge.capacity.gated_until - time.monotonic())

        # 3. Capacity check
        if action == "inference" and edge.capacity.current_queue_depth > threshold:
            # Find alternative via sibling or cross-branch
            alt = self._find_alternative(from_node, PeckingDomain.COMPUTE)
            return GateResult(False, "target overloaded",
                              alternative=alt, backpressure=0.8)

        # 4. Trust check (pecking relations gate what you can do)
        relation = edge.domains.get(self._action_to_domain(action))
        if relation == PeckingDirection.PECKED_BY and action in PECKER_ONLY_ACTIONS:
            return GateResult(False, "only peckers can perform this action")

        return GateResult(True)
```

### What gets gated

| Action | Domain | Who can do it | Gate conditions |
|--------|--------|---------------|-----------------|
| `inference` | Compute | Any node (routed by capability) | Target queue full, GPU thermal, rate limit |
| `update` | Authority | Only peckers (parent pushes to child) | Permission flag, target gated, version mismatch |
| `contribute` | Knowledge | Any node (flows upward) | Trust level too low, rate limit, deidentification not run |
| `query` | Knowledge | Any node (flows downward) | Rate limit, target overloaded |
| `delegate` | Compute | Nodes that peck for compute | Target queue full, loop detection, NAc rejection |
| `restart` | Authority | Only peckers | Permission flag, active requests draining |
| `llm-swap` | Compute | Only peckers for compute | Target has active inference, model not available |

---

## Integration with Existing Systems

### What stays as-is
- **AgentIdentity** — still the node's self-description. Becomes `PeckingNode.identity`.
- **ExperienceBroker** — still handles knowledge type routing (CausalLink/Reflection/MotorProgram adapters). The graph determines *which node* to share with; the broker determines *how* to share.
- **PeerChannel** — still the network transport. Graph routing decides the destination; PeerChannel sends the message.
- **FunctionRouter** — still maps functions to tiers. The graph provides the `available_tiers` callable (now graph-aware, including remote tiers).
- **Transfer discount table** — still applies. But now discounts compound through the graph (see Wisdom Cascade above).
- **PeerClockEstimator** — still provides clock sync. Feeds `PeckingRelation.latency_ms`.

### What gets absorbed
- **`detect_role()`** — replaced by graph position. A "leader" is any node with children.
- **`MeshAdmissionControl`** — state migrates into `EdgeCapacity` on each edge. Escalation logic moves into `PeckingGraph.check_gate()`.
- **`PeerRateLimiter`** (LeaderProxy) — becomes `EdgeCapacity.max_rps` on the proxy edge.
- **LeaderProxy concurrency semaphore** — becomes `EdgeCapacity.current_queue_depth`.
- **`PeerRegistry`** — becomes `PeckingGraph.nodes` filtered by parent.
- **Capability Agent** — `PeckingGraph` methods replace all planned CA-1 through CA-5 phases.

### What gets extended
- **LeaderProxy** — still runs as the HTTP front-end, but routes requests through `PeckingGraph.route_request()` instead of forwarding to a single upstream.
- **Mother Maxim API** — `/v1/contribute` and `/v1/recall` become graph operations (cascade_up and cascade_down) instead of direct endpoint calls.
- **`maxim peer update`** — becomes `graph.cascade_down(domain=AUTHORITY, payload=UpdateCommand)` instead of a direct HTTP POST.

---

## Phase Plan

### Phase POG-0: Prep (weave into publication) — ~100 LOC

Items that position us for the graph without adding it yet. These are low-risk, additive changes.

**POG-0a: Add `NodeLoad` to `RuntimeCapabilities`** (~30 LOC)
- Extend `RuntimeCapabilities` with live load metrics: `gpu_util_pct`, `queue_depth`, `ram_pressure_pct`, `thermal_state`.
- These are already gathered by the heartbeat system — just expose them on the capabilities struct.
- Files: `src/maxim/runtime/capabilities.py`

**POG-0b: Add `parent_id` field to `AgentIdentity`** (~10 LOC)
- Optional field, defaults to `None`. Populated when graph is active.
- Broadcast in HEARTBEAT messages so peers know the topology.
- Files: `src/maxim/mesh/agent_identity.py`

**POG-0c: Unify gating types** (~60 LOC)
- Define `GateResult` and `EdgeCapacity` as shared types in `src/maxim/mesh/gate_types.py`.
- Both `MeshAdmissionControl` and `LeaderProxy` can start using `GateResult` for their return types without changing behavior.
- Files: new `src/maxim/mesh/gate_types.py`, minor type changes in `admission.py` and `leader_proxy.py`

### Phase POG-1: Graph Core — ~300 LOC

The data structures and local operations. No networking yet.

**POG-1a: `PeckingNode`, `PeckingRelation`, `PeckingGraph`** (~200 LOC)
- Core data model as described above.
- Pecking computation rules (compare capabilities, assign directions per domain).
- `route_request()`, `check_gate()`, `cascade_up()`, `cascade_down()` as local operations.
- Persistence: `~/.maxim/mesh/graph.json` via `atomic_write_json()`.
- Files: new `src/maxim/mesh/pecking_graph.py`

**POG-1b: Tests** (~100 LOC)
- Unit tests for pecking computation (GPU node pecks CPU node for compute, etc.)
- Routing tests (walk toward root, fall back to sibling)
- Gate tests (capacity, permission, trust)
- Cascade direction tests (authority flows down, knowledge flows up)
- DAG invariant tests (authority domain never has cycles)

### Phase POG-2: Registration + Discovery — ~250 LOC

Nodes join and leave the graph.

**POG-2a: Registration protocol** (~100 LOC)
- New mesh message types: `REGISTER_REQUEST`, `REGISTER_RESPONSE`, `DEREGISTER`
- On startup, a node sends `REGISTER_REQUEST` to its configured parent (from peer.yml or env var).
- Parent validates, computes pecking relations, adds to graph, responds with graph snapshot.
- On shutdown, node sends `DEREGISTER`. Parent re-parents orphaned grandchildren.
- Files: `src/maxim/mesh/pecking_graph.py`, `src/maxim/mesh/peer_channel.py`

**POG-2b: Heartbeat-driven recomputation** (~50 LOC)
- On HEARTBEAT, update `PeckingNode.load` and `PeckingNode.capabilities`.
- If capabilities changed significantly, call `recompute_pecking()` for affected edges.
- Files: `src/maxim/mesh/pecking_graph.py`

**POG-2c: mDNS discovery → graph registration** (~100 LOC)
- This IS mesh Phase 0a, but now discovery feeds into graph registration instead of a flat peer list.
- Discovered nodes auto-register under the discovering node with `trust_level="discovered"`.
- Files: new `src/maxim/mesh/discovery.py`

### Phase POG-3: Cascade Protocols — ~300 LOC

The graph comes alive with directional flows.

**POG-3a: Update cascade** (~100 LOC)
- `graph.cascade_down(domain=AUTHORITY, payload=UpdateCommand)` replaces `maxim peer update --all`.
- Each node applies update locally, reports status, then cascades to children.
- CLI: `maxim update --cascade` (from any node that pecks others for authority).
- Files: `src/maxim/mesh/pecking_graph.py`, `src/maxim/peer/cli.py`

**POG-3b: Knowledge cascade** (~100 LOC)
- Contributions flow up with enrichment at each intermediate node.
- Wisdom flows down with compounding trust discounts.
- Integrates with existing `ExperienceBroker` — the broker handles serialization, the graph handles routing.
- Files: `src/maxim/mesh/pecking_graph.py`, `src/maxim/mesh/knowledge.py`

**POG-3c: Compute routing** (~100 LOC)
- `graph.route_request(domain=COMPUTE)` replaces the deferred InferenceRouter.
- Integrates with `FunctionRouter` — `available_tiers` callable now queries the graph.
- Files: `src/maxim/mesh/pecking_graph.py`, `src/maxim/runtime/function_router.py`

### Phase POG-4: Mother Integration — ~250 LOC

Mother Maxim as graph root.

**POG-4a: Mother as root node** (~50 LOC)
- Mother registers as `PeckingNode(parent_id=None)` — the only node with no parent.
- All other nodes eventually trace their `parent_id` chain to Mother.
- Mother always pecks for AUTHORITY, KNOWLEDGE, and MEMORY.
- Files: `src/maxim/mother/runner.py`, `src/maxim/mesh/pecking_graph.py`

**POG-4b: Contribution cascade replaces direct API** (~100 LOC)
- `POST /v1/contribute` becomes: accept contribution, enrich locally, cascade up through graph.
- Mother's coalescence engine (M-4) receives pre-enriched contributions instead of raw ones.
- Each intermediate node's NAc evaluates the link, potentially boosting or flagging it.
- Files: `src/maxim/mother/api.py`, `src/maxim/mesh/pecking_graph.py`

**POG-4c: Wisdom propagation** (~100 LOC)
- Mother's consensus links cascade down through the graph.
- Each level applies its trust discount (compounding).
- Nodes can opt-out of wisdom reception (`permission_flags` on edge).
- Files: `src/maxim/mother/coalescence.py`, `src/maxim/mesh/pecking_graph.py`

---

## Example Topologies

### Current (2-node, leader + peer)
```
Leader-A [RTX 5080, large+small tiers]
  └── Peer-1 [Mac, medium+small tiers]

Pecking:
  Leader-A → Peer-1: authority=PECKS, compute=PECKS, memory=MUTUAL
  (Peer-1 has 192GB unified RAM — mutual for memory)
```

### Near-term (Mother + 2 nodes)
```
Mother Maxim [cloud/leader, large tier]
  └── Leader-A [RTX 5080, large+small tiers]
        └── Peer-1 [Mac, medium+small tiers]

Pecking:
  Mother → Leader-A: authority=PECKS, knowledge=PECKS, compute=MUTUAL
  Leader-A → Peer-1: authority=PECKS, compute=PECKS, memory=MUTUAL
```

### Future (federated)
```
Mother Maxim [root]
  ├── Leader-A [RTX 5080]
  │     ├── Peer-1 [Mac]
  │     └── Peer-2 [RPi, embodied]
  ├── Leader-B [cloud GPU]
  │     └── Peer-3 [laptop]
  └── Domain-Mother-Fantasy [specialized sub-Mother]
        ├── DM-Node-1
        └── DM-Node-2

Pecking:
  Mother → Domain-Mother-Fantasy: authority=PECKS, knowledge=PECKS
  Domain-Mother-Fantasy → DM-Node-1: authority=PECKS, knowledge=PECKS
  Peer-2 pecks Leader-A for embodiment (it has physical sensors)
  Leader-B pecks Leader-A for compute when A is overloaded (cross-branch routing)
```

---

## Publication Positioning (POG-0)

To position for the pecking order graph without blocking publication, we need these minimal prep items in v0.2.0:

1. **`NodeLoad` on RuntimeCapabilities** — additive, no API break.
2. **`parent_id` on AgentIdentity** — optional field, backward-compatible serialization.
3. **`GateResult` shared type** — new file, no changes to existing behavior.
4. **`SpatialContext` type** — new file, frozen dataclass for geographic location. Locks serialization format before publication. See Location Add-On section.
5. **Ensure `FunctionRouter.available_tiers` callable path works** — already designed, just verify it's exercised in tests.

These are all <100 LOC total and don't change any public API surface. They just ensure the internal types are ready for the graph to plug into post-publication.

See also: [Publication Refinement Plan — POG-0 items](#cross-references).

---

## Relationship to Other Plans

### Mother Maxim Plan
The pecking order graph **restructures** how Mother relates to other nodes:
- Mother MVP still ships as planned (Phase M-MVP in mother_maxim_plan.md).
- But `/v1/contribute` and `/v1/recall` become graph cascades instead of direct REST calls.
- Coalescence (M-4) benefits from intermediate enrichment — contributions arrive pre-validated by intermediate nodes.
- Federation (the `--mother --domain X` vision) becomes natural: domain Mothers are sub-roots in the graph.
- **POG-4 ships alongside or shortly after Mother MVP.** The graph gives Mother her place in the topology rather than being a standalone API.

### Mesh Plan (archived)
- Phase 0a (mDNS discovery) → absorbed into POG-2c.
- Phase 0b (InferenceRouter) → absorbed into POG-3c.
- Phases 1-7 (already shipped) → unchanged. AgentIdentity, ExperienceBroker, TaskDelegator, PeerChannel all continue to work. The graph adds a topology layer on top.

### Capability Agent (historical — design lived in an old `future_plans.md` that predates the current `docs/plans/` split)
- **Fully subsumed.** All 5 planned phases (CA-1 through CA-5) are replaced by `PeckingGraph` methods.
- `CapabilitySnapshot` → `PeckingNode.capabilities + .load`
- `check_model_availability()` → `graph.route_request(domain=COMPUTE, requirements={"model": X})`
- `gate_action()` → `graph.check_gate()`
- `on_peer_joined/left` → `graph.register() / deregister()`

### Multi-Node Admin (historical — same origin as Capability Agent above)
- **Fully subsumed.** Update cascade through the graph replaces the node registry + fan-out CLI.
- `maxim peer update --all` → `maxim update --cascade` (authority cascade from current node down).

---

## Stress-Test Campaign: The Kings' Duel

A dedicated DM campaign ([`scenarios/campaigns/kings_duel_v1.yaml`](../../../scenarios/campaigns/kings_duel_v1.yaml)) that exercises hierarchical social dynamics and cascading authority transfer — the exact patterns the pecking order graph is designed to handle.

**Scenario:** Medieval duel between English and French kings. The player is a herald (observer/diplomat) watching the hierarchy react to the outcome. 8 NPCs across a clear pecking order:

```
English side:              French side:
  King Edmund (monarch)      King Philippe (monarch)
    → Prince Aldric (heir)     → Prince Louis (heir)
    → Sir Godfrey (champion)   → Marshal Beaumont (knight)
    → Sgt. Thomas (soldier)
                 Father Bertrand (neutral priest)
```

**What it tests for the pecking order graph:**

| Bio-System | What the campaign exercises |
|---|---|
| **NAc** | Causal chain: king falls → heir panics → champion rages → soldiers waver. RPE spike at duel outcome. Marshal's hidden low restraint = surprise betrayal risk. |
| **Hippocampus** | Must recall pre-duel conversations (prince's fear, Godfrey's warning about the marshal) when counseling the new king. 5 recall targets across 6 encounters. |
| **ATL** | Concepts: honor, loyalty, succession, diplomacy, treaty. The agent must form abstract social concepts, not just track concrete events. |
| **Cerebellum** | Forward models during combat (predicting duel outcome from observed fighting styles). Also predicts court reactions based on NPC sensor states. |
| **Salience** | Novelty spike at king's death, decay during post-duel formalities. The agent should pay more attention to the marshal's hidden behavior than to the routine treaty reading. |

**Authority cascade stress test (the key scenario):** When King Edmund falls, the English hierarchy fractures in real time. The agent must:
1. Track the authority transfer (king → prince becomes the new decision-maker)
2. Observe morale cascade (champion's restraint drops, soldiers look to prince)
3. Identify the threat (marshal's hidden low restraint — he wants to break the truce)
4. Navigate the succession council where everyone has different motivations

This directly maps to POG behavior: when a root node (king) goes down, children (heir, champion, soldiers) must re-parent, capability-based pecking shifts (Godfrey pecks for military, Aldric pecks for authority), and the cascade of reactions tests whether the system tracks graph topology changes.

**Run with:** `maxim --sim scenarios/campaigns/kings_duel_v1.yaml`

---

## Add-On: Location-Aware Routing and Spatial Memory

### The Two Scales of Location

Location matters at two fundamentally different scales in this system, and they need different treatment:

| Scale | What it is | Who cares | Coordinates? | Example |
|-------|-----------|-----------|-------------|---------|
| **Geographic** | Where a node/agent is in the world | PeckingGraph routing, Mother coalescence, SCN timezone, knowledge relevance | Coarse (zone/region/label) | "us-west", "kitchen-pi", "london-office" |
| **Embodied** | Where a body part or object is relative to the agent | SEM/Cerebellum, motor programs, proprioception | Fine (x,y,z relative to body frame) | "arm at 45deg", "cup 30cm ahead" |

**Geographic location** belongs in the pecking order graph and hippocampus. It's about *which node* and *which memories* are relevant based on where you are in the world.

**Embodied location** belongs in SEM and Cerebellum. It's about *how to move* and *what's reachable*. This is a local concern — a robot's proprioceptive frame doesn't need to know it's in Portland to reach for a cup.

**The x,y,z question:** Full 3D coordinates are overengineering for the geographic scale (a zone label + optional lat/lon is enough) but potentially useful for embodiment. However, even for embodiment, the current SEM parent/child tree + sensor readings is probably sufficient until we're doing actual path planning. A robot knows "cup is on shelf.top, my arm can reach shelf.top" through the entity tree — it doesn't need `cup.position = (0.3, 0.1, 0.8)` yet. That's a hardware adapter concern (Phase 3, deferred). If/when we need it, it goes on `Entity.metadata["position"]` or a dedicated `SpatialFrame` on the entity, not in the memory system.

**Recommendation:** Geographic location gets first-class support (this add-on). Embodied x,y,z is deferred to the hardware adapter phase. The memory system carries geographic context, not coordinates.

---

### Geographic Location: Data Model

```python
@dataclass(frozen=True)
class SpatialContext:
    """Where something happened or where a node is.

    Designed for coarse geographic context, not fine-grained positioning.
    Mirrors TemporalSignature's approach: normalized, binnable, comparable.
    """

    # Hierarchical location label (most specific to least)
    label: str = ""              # "kitchen", "london-office", "server-room-3"
    zone: str = ""               # Coarse grouping: "local", "lan", "building", "city", "region", "global"
    region: str = ""             # Geographic region: "us-west", "eu-central", "home"

    # Optional coordinates (for when distance matters)
    lat: float | None = None
    lon: float | None = None

    # Containment hierarchy (like a path: building/floor/room)
    hierarchy: tuple[str, ...] = ()  # ("home", "kitchen") or ("london-office", "floor-3", "server-room")

    def distance_to(self, other: SpatialContext) -> float | None:
        """Haversine distance in km, or None if coordinates unavailable."""
        if self.lat is None or other.lat is None:
            return None
        # haversine formula
        ...

    def same_zone(self, other: SpatialContext) -> bool:
        """Are these in the same coarse zone?"""
        return bool(self.zone and self.zone == other.zone)

    def shares_hierarchy(self, other: SpatialContext, depth: int = 1) -> bool:
        """Do these share a common ancestor in the hierarchy?
        depth=1: same building. depth=2: same floor. etc."""
        return (self.hierarchy[:depth] == other.hierarchy[:depth]
                and len(self.hierarchy) >= depth)

    def to_dict(self) -> dict[str, Any]: ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpatialContext: ...
```

**Why `SpatialContext` mirrors `TemporalSignature`:** Just like time has nested cycles (hour within day within week), space has nested containment (room within building within city). The `hierarchy` tuple is the spatial equivalent of circadian/weekly/monthly phases — it lets you bin and compare at different granularities. And just like `TemporalSignature` is frozen and lightweight, `SpatialContext` should be too — it's stamped onto memories, not mutated.

---

### Integration 1: Hippocampus (Primary)

The biological hippocampus is literally the brain's spatial memory system — place cells and grid cells. This is the most natural and important integration point.

**What changes:**

1. **Add `location` field to `Perception`:**
```python
@dataclass
class Perception:
    observations: dict[str, Any] = field(default_factory=dict)
    # ... existing fields ...
    location: SpatialContext | None = None  # NEW — where this percept happened
```

This replaces the fragile `observations["location"]` string that `ConceptExtractor` currently sniffs for. Existing code that sets `observations["location"]` keeps working — the extractor checks the typed field first, falls back to the dict.

2. **Add `"location"` to `HippocampusConfig.indexed_keys`:**
```python
indexed_keys: frozenset[str] = frozenset({
    "goal", "tool", "object", "person", "success", "mode",
    "location",  # NEW
})
```

This means `recall(location="kitchen")` becomes a fast O(1) index lookup instead of a full scan. The `_context_index` stores entries like `"location:kitchen"`.

3. **Add `location` parameter to `recall()`:**
```python
def recall(
    self,
    query: str = "",
    *,
    k: int = 5,
    goal: str | None = None,
    mode: str | None = None,
    location: str | None = None,  # NEW — filter by location label
    # ... existing params ...
) -> list[EpisodicMemory]:
```

When `location` is provided, recall filters candidates through the location index before relevance scoring. This enables "what happened in the kitchen?" queries without scanning all memories.

4. **Location-weighted associative edges:**
The existing associative graph (`DependencyGraph`) computes edge weights as `0.6*perceptual + 0.25*goal + 0.15*temporal`. Add a spatial component:
```python
# Memories from the same location get a proximity boost
spatial_weight = 0.0
if mem_a.perception.location and mem_b.perception.location:
    if mem_a.perception.location.shares_hierarchy(mem_b.perception.location, depth=1):
        spatial_weight = 0.3  # Same building
    if mem_a.perception.location.label == mem_b.perception.location.label:
        spatial_weight = 0.5  # Same room

# Rebalanced: 0.45*perceptual + 0.20*goal + 0.10*temporal + 0.25*spatial
# (Only when both memories have location; otherwise original weights apply)
```

This means returning to the kitchen naturally activates memories of what happened there last time — exactly how human spatial memory works.

5. **`SpatialContext` on `EpisodicMemory.metadata`:**
The `metadata: dict[str, Any]` field already exists. For memories captured with location context, the full `SpatialContext` is stored in `metadata["spatial_context"]`. This preserves the hierarchy and coordinates without adding a new field to the core dataclass (keeping backward compatibility).

**LOC estimate:** ~80 LOC (Perception field + index key + recall param + edge weight component + metadata storage).

---

### Integration 2: ATL / ConceptExtractor (Light)

The concept extractor already sniffs for location strings. Upgrade it to produce richer spatial concepts.

**What changes:**

1. **`ConceptExtractor` reads `Perception.location` instead of `observations["location"]`:**
```python
# Current (fragile):
location = record.perception.observations.get("location")

# New (typed):
if record.perception.location:
    loc = record.perception.location
    concepts_found.append((loc.label, "location"))
    # Also extract hierarchy levels as location concepts
    for level in loc.hierarchy:
        concepts_found.append((level, "location"))
```

2. **Spatial relationship type in concept graph:**
Currently, `person+location` and `object+location` both map to `"RELATED_TO"`. Add `"LOCATED_AT"` and `"CONTAINS"`:
```python
if types == {"location", "object"}:
    return "LOCATED_AT"
if types == {"location", "location"}:
    return "CONTAINS"  # hierarchy: kitchen CONTAINS stove
```

This means ATL can answer "what's in the kitchen?" and "where is the cup?" through concept graph traversal.

**LOC estimate:** ~30 LOC (extractor upgrade + relationship types).

---

### Integration 3: NAc — Intentionally Skipped

Causal links should remain location-agnostic. "Threatening → hostility" applies everywhere. If location context matters for a specific link, it's already expressible in `event_context: dict` (the catch-all bag). Adding structured location to NAc would fragment the causal model — you'd get separate "threatening in kitchen → hostility" and "threatening in throne room → hostility" links instead of one strong generalized link.

The exception is Mother Maxim's coalescence: when merging causal links from different geographic regions, location metadata on the *contributing node* (from the pecking graph) tells Mother whether a pattern is universal or regional. But that's graph-level metadata, not NAc-level.

---

### Integration 4: SCN / Temporal System (Light)

**Timezone from location:**
```python
@dataclass(frozen=True)
class SpatialContext:
    # ... existing fields ...
    timezone: str = ""  # IANA timezone: "America/Los_Angeles", "Europe/London"
```

When a node has `SpatialContext.timezone`, SCN can use it for circadian phase computation instead of relying on system-local time or peer clock sync. This matters for Mother Maxim (she serves nodes in different timezones) and for distributed setups where a peer in London and a leader in Portland have different circadian rhythms.

**LOC estimate:** ~15 LOC (timezone field + SCN integration).

---

### Integration 5: PeckingGraph (Routing)

Already outlined in the main plan's `NodeLocation` discussion. Formalized here:

```python
@dataclass
class PeckingNode:
    # ... existing fields ...
    location: SpatialContext | None = None  # Geographic location of this node
```

**Routing preference by proximity:**
```python
def route_request(self, from_node, domain, requirements):
    candidates = self._find_capable_nodes(domain, requirements)
    # Sort by: capability match > same zone > lower latency > lower load
    candidates.sort(key=lambda n: (
        -self._capability_score(n, requirements),
        0 if n.location and from_node.location and n.location.same_zone(from_node.location) else 1,
        self.edges[(from_node.node_id, n.node_id)].latency_ms,
        n.load.gpu_util_pct if n.load else 0,
    ))
    return candidates[0] if candidates else None
```

**LOC estimate:** ~20 LOC (field + routing tiebreaker).

---

### Integration 6: Mother Maxim Coalescence (Regional Knowledge)

This is where location becomes genuinely powerful for the collective memory:

**Location-tagged contributions:**
When a node contributes memories up the graph, the contribution carries the node's `SpatialContext`. Mother can then distinguish:
- **Universal patterns** — observed across multiple regions ("threatening → hostility" everywhere)
- **Regional patterns** — observed only in specific contexts ("in the kitchen, reaching for stove → pain")
- **Local patterns** — single-location observations (weakest evidence for generalization)

**Coalescence rules (extend M-4):**
```python
# When merging causal links, boost confidence for cross-regional consensus
if link.observed_in_regions >= 3:
    consensus_boost = 1 + 0.15 * log(link.observed_in_regions)
    # A pattern seen in London, Portland, AND Tokyo is probably universal

# When a pattern is only seen in one region, tag it as regional
if link.observed_in_regions == 1:
    link.metadata["regional"] = True
    link.metadata["region"] = contributing_region
```

**Location-aware recall from Mother:**
`/v1/recall` can optionally filter by region — "what does Mother know about kitchen environments?" returns location-relevant shared memories first, universal ones second.

**LOC estimate:** ~40 LOC (contribution tagging + coalescence rules + recall filter).

---

### Integration 7: Embodiment — Local Spatial Context (deferred)

You're right that embodiment is its own beast. The SEM entity tree already handles spatial containment for body parts (`arm.elbow.wrist.gripper`) and environmental hierarchy (`room.shelf.cup`). This is fundamentally different from geographic location.

**What exists:** Parent/child entity trees. Sensors on entities report state (joint angles, distances). Cerebellum forward models predict outcomes of physical actions.

**What's missing but deferred:**
- `Entity.position: tuple[float, float, float] | None` — x,y,z in body-relative frame
- Spatial affordance filtering — "what can I reach from here?"
- Path planning through entity spatial graph
- Coordinate frame transforms (body frame ↔ world frame)

**Why defer:** All of this is only useful with actual hardware or a physics simulation. Virtual entities in DM campaigns don't need coordinates — "the cup is on the shelf" is expressed through the entity tree, and the agent interacts via affordances (named actions), not trajectories. When the hardware adapter lands (embodiment Phase 3), it brings real sensor data and real coordinate frames. That's the right time to add x,y,z.

**The bridge:** When embodiment does get coordinates, the `SpatialContext.label` on memories ("kitchen") connects to the `Entity` tree's environment node ("kitchen" entity with its children). Geographic memory and embodied space are linked through shared labels, not shared coordinate systems. The hippocampus knows "something happened in the kitchen" (geographic). The SEM knows "the kitchen contains a stove, shelf, and table" (embodied). The label is the join key.

---

### Summary: What Goes Where

| System | Location integration | Scale | LOC | When |
|--------|---------------------|-------|-----|------|
| **Hippocampus** | `Perception.location`, indexed recall, spatial edge weights | Geographic | ~80 | POG-1 or standalone |
| **ATL / ConceptExtractor** | Typed location concepts, LOCATED_AT/CONTAINS relations | Geographic | ~30 | POG-1 or standalone |
| **NAc** | None (intentionally) | — | 0 | — |
| **SCN** | Timezone from `SpatialContext.timezone` | Geographic | ~15 | POG-1 |
| **PeckingGraph** | `PeckingNode.location`, routing by proximity | Geographic | ~20 | POG-1 |
| **Mother coalescence** | Regional vs. universal knowledge tagging | Geographic | ~40 | POG-4 / M-4 |
| **EpisodicStore** | `query_by_location()` protocol method | Geographic | ~15 | M-1 (database) |
| **Embodiment (SEM)** | x,y,z on entities, coordinate frames | Embodied (local) | ~200+ | Hardware adapter phase |

**Total for geographic location (this add-on):** ~200 LOC, shipped alongside POG-1.
**Embodiment spatial:** deferred, separate initiative, ~200+ LOC when hardware adapter lands.

---

### POG-0 Prep for Location

One additional prep item for publication:

**POG-0d: Add `SpatialContext` type** (~40 LOC)
- Define `SpatialContext` in `src/maxim/mesh/spatial.py` (or `src/maxim/memory/spatial.py` since hippocampus uses it too).
- Frozen dataclass, `to_dict`/`from_dict`, `distance_to()`, `same_zone()`, `shares_hierarchy()`.
- No integration yet — just the type definition so it's available post-publication.
- Adding the type now means `Perception.location` serialization format is locked in v0.2.0.

---

## Open Questions

1. **Should siblings be able to peck each other?** Current design allows MUTUAL pecking between siblings (route by load/latency). Could also allow asymmetric sibling pecking when one has clearly better capabilities.

2. **How deep should the graph go?** Three levels (Mother → Leader → Peer) covers all current use cases. But the data model supports arbitrary depth. Should we cap at 4-5 to prevent routing overhead?

3. **Graph persistence format.** JSON via `atomic_write_json()` for now. But the graph is a global shared structure — if two nodes disagree on topology, who wins? Answer: the pecker wins. Authority domain parent's view is canonical.

4. **Offline nodes.** When a node goes offline, its children become orphaned. Current design: re-parent to grandparent. Alternative: children continue operating independently (solo mode) until parent returns. Probably: re-parent after a timeout (30s), revert if original parent comes back.

5. **Cross-branch routing cost.** Routing a compute request from Peer-1 → Leader-A → Mother → Leader-B is 3 hops. Direct Peer-1 → Leader-B would be 1 hop. Should the graph support shortcut edges for frequently-used cross-branch routes?
