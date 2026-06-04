# Mesh Perception Transport — peer-tunneled sensory percepts

**Status:** Shell plan, drafted 2026-06-02. 1.0 prep work scoped; 1.1 implementation sketched.
**Scope:** ~80-150 LOC of refactor + wire-format reservations in 1.0; ~400-600 LOC of transport + adapter + endpoint family in 1.1.
**Target versions:** 1.0 (prep + reservations only — no transport), 1.1 (full ship alongside Hivemind).
**Gates:** None as a 1.0 release gate. The 1.0 prep items gate themselves on the refactor-now-or-refactor-later test (see "Why land prep in 1.0" below).
**Driving use case:** Reachy Mini will ship an app form factor; running the app as a Maxim peer that tunnels vision + audio to a leader for cognition is the immediate consumer. Generalizes to any embodiment peer with sensors but limited compute.

**Depends on:**
- [`agents/percept_context.py`](../../src/maxim/agents/percept_context.py) — `PerceptContext.channel` already includes `"mesh"` as an allowed value (anticipated this transport)
- [`agents/bus.py`](../../src/maxim/agents/bus.py) — `Percept.to_dict / from_dict` already exists (wire format is ~80% in place)
- [`simulation/sources.py`](../../src/maxim/simulation/sources.py) — `PerceptSource` Protocol is the adapter shape a `RemotePerceptSource` plugs into
- [`mesh/message.py`](../../src/maxim/mesh/message.py) — `MeshMessage` envelope is transport-agnostic with `protocol_version` and `MeshMessageType` enum (already reserves `INFERENCE_REQUEST/RESPONSE` for non-LLM mesh traffic — same precedent)
- Plan 4 C1 ([`mesh_config.py`](../../src/maxim/peer/mesh_config.py)) — `mesh.yml::nodes` addresses perception-source peers by name
- Plan 4 C2 ([`drain_state.py`](../../src/maxim/peer/drain_state.py)) — `~/.maxim/util/` mutable-state layer + `filelock.FileLock` RMW pattern for any per-node percept inbox / sequence state

**Enables (future work that expects this):**
- Reachy Mini app (consumer #1) — peer runs on-device perception + STT + segmentation, ships event-shaped percepts to leader
- [`maxim_hivemind.md`](maxim_hivemind.md) — substrate-bundle exchange (1.1+) and perception transport are two distinct typed peer transports following the same playbook; the typed-transport-per-purpose pattern this plan establishes is the architectural invariant Hivemind needs
- Mineflayer / Minecraft adapter (named in `sources.py` as a canonical external adapter) — same Protocol contract, future networked variant
- Multi-Maxim training rigs — a research box that wants to fan multiple embodiment peers into one cognition node

---

## Front-gate scope pressure (Principle 3)

**Question:** does this need to be its own mechanism, or can it ride on existing infrastructure?

| Candidate | Why insufficient (or sufficient) |
|---|---|
| `_MaximPeerBackend` for transport | Wrong layer. The Plan 3 R2.5 invariant "exactly one HTTP call per call-site, no retry, no internal cooldown" is LLM-inference-specific and was earned via incident. Forcing perception payloads through that backend either dilutes the invariant or requires a parallel call path with `try: retry` (forbidden by the CI grep). Sibling typed backend is the right answer. |
| [`mesh_doc_transport.md`](deferred/mesh_doc_transport.md) (deferred C9) | **Could carry low-rate event percepts** (scene summaries, transcribed audio chunks, detection rosters) since those are small JSON docs by `(namespace, key)`. Real-time percept streams want lower latency + sequence semantics that the doc-transport KV-drop shape doesn't model well. Verdict: doc transport is a **fallback path** for non-time-critical percepts; a typed perception transport is needed for the real-time path. |
| `PerceptSource` Protocol + `Percept.to_dict/from_dict` | **Sufficient at the adapter layer.** A `RemotePerceptSource` implementing the existing 4-member Protocol is ~80 LOC of pure adapter code — no Protocol changes needed. This is the "rides on" half of the answer. |
| Existing `MeshMessage` + `MeshMessageType` enum | **Sufficient at the envelope layer.** Adding `PERCEPT_PUSH` (and possibly `PERCEPT_ACK`) to the enum is genuinely additive — same precedent as the reserved `INFERENCE_REQUEST/RESPONSE` slots. |

**Verdict:** **split** — adapter and envelope ride on existing infrastructure (`PerceptSource`, `MeshMessage`); transport layer is yes-needs-own (the parallel typed-backend slot to `_MaximPeerBackend`). The typed-transport-per-purpose pattern is the new architectural invariant; the per-transport implementation is small.

**Specific reason for the new mechanism:** the load-bearing "single HTTP call, no retry" rule on `_MaximPeerBackend` is LLM-failover-shaped (router handles retry across providers). A perception transport has different failure semantics (a dropped sensor frame is not a router-failover event; it's a missed observation that the agent loop tolerates by reading the next one). Mixing them under one backend re-introduces the silent-failure-mode class the Plan 3 invariant closed.

---

## Why land prep in 1.0 (vs. waiting for 1.1)

The user's instruction: *"build out the core infrastructure and refactor anything that we need to now before we fit the big 1.0 iteration and don't have to refactor later."*

Concretely, the test is: **what would be refactor-painful post-1.0?**

| Concern | Post-1.0 cost if deferred | 1.0 prep effort |
|---|---|---|
| `MeshMessage` wire envelope: adding new enum slots | **Low** — additive enum values are non-breaking by design (precedent: reserved `INFERENCE_REQUEST/RESPONSE` slots in `mesh/message.py`) | Reserve `PERCEPT_PUSH` slot. ~5 LOC. |
| `Percept.to_dict` wire format: completeness for remote sender | **Medium-high** — `to_dict` today says "omits large/internal fields" and drops `detections`, `embedding`, `explore_command`, `transcript_chunk_index`, `raw_transcript_text`, `maxim_runtime`. A remote perception peer probably WANTS to ship `detections` (it ran the on-device vision). Quietly extending `to_dict` post-1.0 changes what session-persisted percepts look like — a silent format-drift class of bug. | Decide and pin which fields are persisted-only vs. wire-tunneled. Either split `to_dict` / `to_wire_dict` or document the omission set under the [`_format_version`](../../src/maxim/utils/format_version.py) contract. ~20 LOC + test. |
| `PerceptSource` Protocol shape: confirm 4-member contract supports network-backed adapter | **High** — adding required Protocol members post-1.0 breaks every third-party `PerceptSource` implementation. | Verify the existing contract suffices for `RemotePerceptSource` (it does — see "Pin the contract" below). Document in [CLAUDE.md](../../CLAUDE.md) the network-adapter case so the next reader doesn't propose a breaking change. ~30 LOC of docstring + invariant. |
| `_MaximPeerBackend` naming + module location | **Low** — it's `_`-prefixed (internal), all 11 importers live under `src/maxim/`, and the file path `models/language/maxim_peer_backend.py` already conveys LLM-specificity. No rename needed. | None. Note in CLAUDE.md the typed-transport-per-purpose pattern so the 1.1 sibling lands at `models/perception/maxim_peer_perception.py` (or similar) by default. |
| Shared peer-transport plumbing (HTTP, auth, `X-Maxim-*` propagation, `for_url`, `health_check`) | **Medium** — if 1.1 perception transport duplicates 200 LOC of HTTP plumbing, the next sibling (Hivemind substrate bundle in 1.1+) duplicates it again. Three transports × 200 LOC duplicated = three places to fix when the auth header changes. | Audit `_MaximPeerBackend` for genuinely-shared plumbing. Extract to `models/peer_transport_base.py` if the diff is real; document the playbook (one-HTTP-call invariant, typed errors, no retry, `for_url` factory pattern, instance-level `_api_key_override`) as a CLAUDE.md architectural invariant. ~50-100 LOC of extraction OR pure documentation if extraction is premature. |

**Net 1.0 prep:** ~100-150 LOC + invariants. Three of the five items are pure documentation. The wire-format decision is the only one with real code.

The transport implementation, the endpoint family, the CLI, the Reachy app wiring — all of that lands in 1.1 alongside Hivemind, where the typed-transport-per-purpose pattern earns its second consumer and the abstraction is proven by parallel use rather than presumed by one.

---

## 1.0 prep work (the explicit list)

### Prep 1: Reserve `MeshMessageType` slots (~5 LOC)

Add to [`mesh/message.py`](../../src/maxim/mesh/message.py):

```python
# Perception transport (reserved — implementation in 1.1)
PERCEPT_PUSH = auto()       # Peer → leader: tunneled Percept
PERCEPT_ACK = auto()        # Leader → peer: receipt with sequence number
```

Additive only — no current consumer, no behavior change. Wire format frozen at 1.0 then.

### Prep 2: Pin `Percept` wire format (~20 LOC + test)

Decide:
- **(a)** Extend `Percept.to_dict` to include the fields a perception peer would tunnel (`detections`, `transcript_chunk_index`, `raw_transcript_text` if relevant) — and accept that session-persisted percepts grow accordingly.
- **(b)** Add a separate `Percept.to_wire_dict()` / `from_wire_dict()` pair distinct from the session-persistence pair, with explicit omission policy documented per format.

Default recommendation to validate in pre-implementation review: **(b)**, because session-persistence and wire-tunneling have different audiences — session persistence wants minimal disk footprint for replay; wire tunneling wants enough fidelity that the receiving leader can run substrate encoding without round-tripping back to the peer.

**Embedding is NEVER on the wire.** The leader owns the substrate (EC, ATL, LinguisticEncoder); the peer ships raw observations. This is also the bio-fidelity argument — the peer is "a sensor," not "a partial cognition."

Pin via `_format_version` contract (CC1) — bump the version on the wire-dict path when shape changes, treat session-dict and wire-dict as independent versioned formats.

**Regression guard:** test in `tests/unit/test_percept_wire_format.py` that pins the wire-dict field set + round-trips a representative `Percept` through `to_wire_dict / from_wire_dict` without loss.

### Prep 3: Pin `RemotePerceptSource` contract in CLAUDE.md (~30 LOC docstring + invariant)

Verify and document:
- `name: str` → `f"remote:{node_name}"` shape
- `next_percept() -> Percept | None` → polls a local inbox populated by the peer's `PERCEPT_PUSH` handler; returns None when inbox empty; **non-blocking, must NOT make a network call** (would re-introduce the per-tick HTTP-cost class of bug the Plan 3 invariant closed)
- `is_exhausted() -> bool` → False unless the peer has signaled end-of-stream (explicit shutdown, not transient network failure); transient unreachable handled by a separate health check at the transport layer, not by lying about exhaustion
- `capabilities: set[str]` → populated from peer's handshake (`{"vision", "audio", ...}`)
- Optional `has_pending()` → True iff local inbox has buffered percepts; used by the agent loop's idle-skip heuristic

Document in [CLAUDE.md](../../CLAUDE.md) `Architectural invariants` as a `[engineering]` entry — the network-backed PerceptSource contract is non-blocking-by-design, and any future revision that proposes a blocking variant must add a parallel Protocol (not modify this one).

**Regression guard:** the existing `PerceptSource` Protocol shape (`runtime_checkable`) + a unit test that registers a stub `RemotePerceptSource` implementation, verifies `isinstance(stub, PerceptSource)` returns True.

### Prep 4: Document typed-transport-per-purpose invariant in CLAUDE.md (~20 LOC)

New `[engineering]` invariant under `Architectural invariants`:

> **Peer transports are typed per purpose, not generic.** `_MaximPeerBackend` in [`models/language/maxim_peer_backend.py`](../../src/maxim/models/language/maxim_peer_backend.py) is the LLM-inference transport — the "exactly one HTTP call, no retry, typed `BackendError`" rule is its load-bearing invariant. Future peer transports (perception in 1.1, substrate bundle in 1.1+, any further sibling) live in their own files following the same playbook: single-purpose backend, typed exception hierarchy with `.fix_hint`, no internal retry, `for_url(api_key=k)` factory pattern with instance-level `_api_key_override` (no `os.environ` mutation), `health_check()` as the canonical probe entry point. Do NOT extend `_MaximPeerBackend` to carry non-LLM payloads — the failover semantics differ and the router-handles-retry invariant becomes ambiguous. Regression guard: file-path convention (one transport per file under `models/<modality>/`) + CI grep allow-list on each new transport's class name keeps cross-transport leakage out.

This is the load-bearing 1.0 prep — it pins the playbook before the second consumer arrives, so the second consumer (perception in 1.1) lands as a sibling file rather than a `_MaximPeerBackend` extension someone has to undo later.

### Prep 5 (optional): Extract shared HTTP plumbing IF the diff is real (~50-100 LOC)

Audit `_MaximPeerBackend` for genuinely-shared surface:
- `_get_api_key` / `_api_key_override` mechanism
- `_build_headers` (already in `utils/http.py` registry — likely already factored)
- `for_url` factory shape
- `health_check` skeleton
- Typed exception construction patterns

If the audit reveals 50+ LOC of cleanly-shared code, extract to `models/peer_transport_base.py` (or similar) as a pure utility module. If the audit reveals only 10-20 LOC of conceptually-shared but syntactically-divergent code, **skip this prep** and let the 1.1 perception transport duplicate small helpers — premature extraction is worse than honest duplication.

**Trigger to defer:** the audit comes back with a thin diff. Pure documentation (Prep 4) is sufficient if the code-sharing surface is small.

---

## 1.1 implementation (sketched)

### Transport layer

New file: `src/maxim/models/perception/maxim_peer_perception.py` (path mirrors `models/language/maxim_peer_backend.py`).

- Class `_MaximPercepTransport` (or similar — final name in pre-implementation review)
- One method per envelope verb: `push_percept(node, percept) -> ack_seq` and `pull_percepts(node, since_seq) -> list[Percept]`
- Typed exceptions mirroring `BackendError` hierarchy: `PercepTransportError` base + `PercepTimeout`, `PercepUnreachable`, `PercepAuthFailed`, `PercepSequenceGap`
- Single HTTP call per method, no retry, no internal cooldown (the load-bearing playbook)
- `for_url(url, api_key=k)` factory + instance `_api_key_override` (the C7 concurrency-safe pattern)
- `health_check()` re-uses the existing `_MaximPeerBackend.health_check` shape

### Adapter layer

New file: `src/maxim/agents/remote_percept_source.py` — implements `PerceptSource` Protocol.

- Holds a reference to the transport + a per-node local inbox (`deque[Percept]`, bounded)
- Background thread (one per source) consumes `PERCEPT_PUSH` envelopes from the transport, deserializes via `Percept.from_wire_dict`, drops on inbox-full with a structured-log warning
- `next_percept()` pops one Percept from inbox, returns None when empty — never blocks, never makes a network call
- Sequence-gap detection (peer sends monotonic seq numbers; gap → structured log + counter for ops visibility, no agent-loop interruption)

### Endpoint family

Leader proxy gains:
- `POST /v1/mesh/percepts/<node>` — peer pushes a percept envelope (body: wire-dict JSON)
- `GET /v1/mesh/percepts/<node>?since=<seq>` — leader-side debug pull (admin only)
- `GET /v1/mesh/percepts/health` — peer handshake / capability negotiation

CI grep allow-list on `/v1/mesh/percepts` literal — only `mesh_percept_core.py` (shared core, mirroring `install_core.py` pattern) + test file + `leader_proxy.py` (server handler) may reference it.

### Reachy app peer wiring

The Reachy app integrates as:
- Runs `maxim` in peer mode with role=peer + sensor-only profile (no LLM, no AUT)
- On-device: STT (whisper-tiny) + YOLO vision segmentation + drive sensor reads → produces `Percept` instances locally
- Each local Percept → `_MaximPercepTransport.push_percept(leader_node, percept)` against the leader URL from `peer.yml`
- Cradle of behavioral state (drives, body state) stays on the Reachy peer — only percepts cross the wire
- Leader's agent loop reads from `RemotePerceptSource` alongside any local sources

### CLI verbs

```bash
# Health-check the perception transport against a node
maxim peer --node <name> percepts health

# Tail recent percepts from a node (admin/debug)
maxim peer --node <name> percepts tail [--since <seq>] [--limit 50]

# Show capability handshake
maxim peer --node <name> percepts caps
```

### Data flow

```
Reachy app (peer)                Leader
─────────────────                ──────
  on-device STT                    ─── /v1/mesh/percepts/reachy ─── POST
  on-device vision                                                   │
  drive sensor reads                                                 ▼
        │                                                  inbox (deque)
        ▼                                                          │
  Percept(...)                                                     │
        │                                                          ▼
        ▼                                                  RemotePerceptSource
  to_wire_dict()                                            .next_percept()
        │                                                          │
        ▼                                                          ▼
  HTTP POST  ───────────────────────────────────────────────  agent_loop
                                                                  │
                                                                  ▼
                                                          PerceptionAgent
                                                                  │
                                                                  ▼
                                                          (existing pipeline:
                                                           bio-enrichment,
                                                           substrate encoding,
                                                           cognition)
```

---

## Open design questions (must be answered in 1.1 pre-implementation review)

### Q1 — Push vs. pull

- **(a) Push (peer POSTs to leader)** — peer controls cadence, leader's agent loop never blocks on network. Matches the existing admin-endpoint shape. Requires the leader to expose an inbound endpoint (and per-peer auth via cluster key).
- **(b) Pull (leader polls peer)** — leader controls cadence; peer hosts a `GET /v1/percepts/inbox` endpoint. Symmetric to `_MaximPeerBackend`'s leader-initiated flow. Latency cost on every tick.
- **(c) Hybrid (push + long-poll on the pull endpoint)** — best UX, real-time-ish, but introduces protocol complexity.

**Recommendation to validate:** (a). The user's framing ("Reachy app runs a peer and feeds visual/audio to a leader") is push semantics. The leader-blocks-on-network problem in (b) is real and would compound with the existing Plan 3 fail-slow class of bug.

### Q2 — Per-percept ack semantics

Does the leader ACK every percept? Drop on inbox-full silently? Some middle?

- **(a) No ack** — peer fire-and-forgets. Simplest. Loss is silent.
- **(b) Per-percept ack with sequence number** — peer can detect gaps and re-send (or just log them). Matches typical streaming-transport hygiene.
- **(c) Periodic ack (every N percepts)** — amortizes the ack cost.

**Recommendation to validate:** (b). Sequence-gap visibility is the minimum cost-effective ops surface; without it, missed percepts become an invisible class of bug. The leader can drop on inbox-full but the drop should be a counter + structured log, not silent.

### Q3 — Inbox bounds + drop policy

- **(a) Bounded deque, drop oldest** — agent loop sees freshest percepts.
- **(b) Bounded deque, drop newest** — preserves the historical trace.
- **(c) Unbounded, OOM the leader** — never.

**Recommendation to validate:** (a) with a configurable bound. Stale percepts are worse than missed-fresh percepts for a live agent loop. Drop counter visible via doctor + structured logs.

### Q4 — Wire-format vs. session-format split (the Prep 2 decision)

Already discussed — recommendation is `to_wire_dict / from_wire_dict` distinct from `to_dict / from_dict`. Pre-implementation review should validate against actual Reachy app payload shape before code lands.

### Q5 — How does the agent loop tolerate a remote source going unreachable?

- **(a) `RemotePerceptSource.next_percept()` returns None** — same as a local source with no fresh percept. Agent loop treats it identically. Health-check loop in the transport layer logs the unreachability, doctor surfaces it.
- **(b) Source flips to exhausted** — agent loop drops the source.
- **(c) Source raises** — agent loop catches and degrades.

**Recommendation to validate:** (a). The PerceptSource Protocol contract is "non-blocking, returns None when no percept available." Unreachability looks like "no percept available" to the agent loop; observability is the transport layer's job. Keeps the contract simple and the failure-mode handling co-located with the transport.

### Q6 — Streaming raw frames as a future expansion path

Explicitly out of scope for v1, but: does the wire envelope shape allow a future raw-frame variant without rework?

- **(a) Yes, if `Percept.metadata` can carry an opaque blob reference** — peer ships small frames inline, large frames via a separate blob endpoint. Compatible with v1 by design.
- **(b) No, raw frames need a separate transport (e.g., WebSocket on the leader proxy)** — then v1 needs to commit to an upgrade path.

**Recommendation to validate:** (a) with the v1 cap that inline payloads stay under 1MB (same as `mesh_doc_transport.md`). Large-frame streaming is a 1.2+ concern when (and if) the substrate-level vision encoder is load-bearing enough to need them.

---

## v1 scope cut (explicit non-goals for 1.1 ship)

- **No raw video / raw audio frames over the wire.** Peer does on-device STT + segmentation, ships event-shaped percepts. Frames stay local.
- **No bidirectional perception flow.** Leader → peer "look at this" is out of scope. Perception flows peer → leader only.
- **No multi-peer fan-in of the same modality.** Two Reachy peers reporting vision to one leader is allowed (different node names) but the leader doesn't reconcile them — agent loop sees both as independent `PerceptSource`s.
- **No per-percept ACL / authorization beyond cluster key.** Same v1 limitation as the doc-transport plan; C7 per-peer identity layers on top later.
- **No replay / persistence of remote percepts on the leader side beyond the existing session log.** Inbox is in-memory only; survives nothing.
- **No leader → peer percept echo / debug stream.** Operator debugging happens via `maxim peer --node <name> percepts tail` admin endpoint, not via a peer-side observability hook.

---

## Architectural invariants this plan establishes (1.0)

These land in CLAUDE.md as Prep 3 + Prep 4:

1. **[engineering] Network-backed `PerceptSource` implementations are non-blocking-by-design.** `next_percept()` MUST NOT make a synchronous network call — it reads from a local inbox populated by a separate transport-layer thread or async handler. Any future revision proposing a blocking variant adds a parallel Protocol; the existing Protocol does not change. Regression guard: the Protocol shape (`runtime_checkable`) + a test pinning `isinstance(stub, PerceptSource)` for a stub `RemotePerceptSource`.

2. **[engineering] Peer transports are typed per purpose, not generic.** See Prep 4 wording above. Regression guard: file-path convention (one transport per modality under `models/<modality>/`) + CI grep allow-list on each transport class.

3. **[engineering] `Percept` wire format is distinct from session-persistence format.** `Percept.to_wire_dict / from_wire_dict` is the cross-process transport contract; `to_dict / from_dict` is the session-persistence contract. Both are versioned under `_format_version`. Embedding is NEVER on the wire — the leader owns the substrate, the peer ships raw observations. Regression guard: `tests/unit/test_percept_wire_format.py` pins the wire-dict field set and round-trips a representative `Percept`.

---

## Proposed implementation sequence

### 1.0 prep (before 1.0 ship)

1. **Prep 1 + Prep 2 + Prep 3 + Prep 4** as a single PR — additive enum slots, wire-format split, two CLAUDE.md invariants, the Protocol-contract documentation. ~150 LOC, ~3 new tests. No new transport code. **One session.**
2. **Optional Prep 5** as a separate PR if the audit finds a real shared-plumbing surface. **Skip if the audit comes back thin.**
3. **Two-lens review round** (Executor + Architecture) per CLAUDE.md Principle on pre-merge review timing. The invariant additions to CLAUDE.md are the load-bearing part of this PR — the review should specifically validate that the typed-transport-per-purpose invariant covers the Hivemind substrate-bundle case as well, so 1.1 lands two siblings to `_MaximPeerBackend` rather than the wrong abstraction.

### 1.1 ship (alongside Hivemind)

4. **Pre-design review round** — answer Q1-Q6 above via a 3-lens review of this shell plan. Gate on cross-confirmed findings before any code.
5. **Commit 1:** Wire-format finalization + `Percept.to_wire_dict / from_wire_dict` if not already in 1.0 prep.
6. **Commit 2:** `models/perception/maxim_peer_perception.py` transport class with `push_percept` / `pull_percepts` / `health_check`. Mock-server unit tests.
7. **Commit 3:** Leader-side endpoint handlers in `leader_proxy.py`. Inbox storage. Sequence tracking. Integration tests.
8. **Commit 4:** `agents/remote_percept_source.py` adapter implementing `PerceptSource`. Background-thread inbox consumer. Health-check polling.
9. **Commit 5:** CLI verbs `maxim peer --node <name> percepts {health,tail,caps}`.
10. **Commit 6:** Reachy app peer-side integration — on-device STT + vision + drive sensors → `_MaximPercepTransport.push_percept` calls.
11. **Commit 7:** Docs — `cli-reference.md`, new `docs/user/perception_transport.md` walkthrough, CLAUDE.md updates if any invariant refinements surface during implementation.
12. **Three-lens pre-merge review round** — Executor + Architecture + Blast Radius lenses. Fold round before PR.

---

## Re-check triggers

Update this plan when:

- **Reachy app team firms up timeline** — if the app ships before 1.1 is ready, decide whether to compress the 1.1 work or ship a temporary Reachy-specific path with an explicit migration commitment.
- **Hivemind plan touches transport** — the typed-transport-per-purpose invariant is the load-bearing contract this plan and Hivemind share. If Hivemind's substrate-bundle exchange proposes anything that violates the playbook (retry inside the backend, generic envelope), revisit this plan's Prep 4 invariant.
- **A second perception consumer arrives** — Minecraft adapter, second robot platform, multi-Maxim training rig. Two consumers is the point at which the abstraction is proven by parallel use; the contract may need refinement.
- **`PerceptSource` Protocol gets a new optional duck-typed extension** — any new `has_pending` / `advance_step` sibling needs a `RemotePerceptSource` story before it ships.
- **`Percept` dataclass adds new fields** — wire-format vs. session-format omission policy needs to make an explicit choice for the new field.

---

## Related plans

- [`reactive_peer_mesh_roadmap.md`](reactive_peer_mesh_roadmap.md) — this plan slots in as Stage C10
- [`deferred/mesh_doc_transport.md`](deferred/mesh_doc_transport.md) — Stage C9 sibling, complementary transport (event docs vs. real-time percepts)
- [`maxim_hivemind.md`](maxim_hivemind.md) — 1.1+ substrate-bundle exchange, second consumer of the typed-transport-per-purpose pattern
- [`v1_refinement.md`](v1_refinement.md) — 1.0 plan; Prep items 1-4 (and optionally 5) slot into this scope
- [`grounded_language_acquisition.md`](grounded_language_acquisition.md) — substrate-primary AUT mode; the cognition layer that ultimately consumes the tunneled percepts
- [`archive/llm_path_fast_failover.md`](archive/llm_path_fast_failover.md) — Plan 3 R2.5, the load-bearing one-HTTP-call invariant on `_MaximPeerBackend` this plan deliberately mirrors rather than violates
