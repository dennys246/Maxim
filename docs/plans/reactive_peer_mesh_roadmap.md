# Reactive Peer Mesh — Roadmap

**Status:** Living roadmap, drafted 2026-04-15.
**Scope:** Tracks the full arc from "operator manually drives the mesh" to "fully reactive peer mesh." Cross-cuts several active and deferred plans rather than replacing them — this is the index that ties them together.
**Target versions:** 0.4 → 0.7 (mesh ship). 1.0 banner is cross-session learning, separate from mesh.
**Maintained alongside:**
- [archive/llm_path_operator_visibility.md](archive/llm_path_operator_visibility.md) — Plan 4 (ALL SHIPPED). The Plan 4 work feeds this roadmap; this doc is the index.
- [llm_path_refinement.md](llm_path_refinement.md) — Plan 1-3.6 ancestor.
- [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md) — Stage C5.
- [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — feeds C5.
- [node_security_simplification.md](node_security_simplification.md) — feeds C7.

---

## 1. Definition: what "fully functioning reactive peer mesh" means

A mesh is **reactive** when topology and runtime state changes propagate **automatically** to the components that need them, with no operator hand-holding. Concretely:

1. **Operator declares intent** (`mesh.yml`) and **runtime state changes** (drain, install, restart, model swap) propagate to the **router** without process restart.
2. **Router consults mesh + state** on every dispatch — drained nodes excluded, failed nodes backed off, healthy nodes load-balanced.
3. **Failures are observed** (doctor + admin API + structured logs) and **escalated** automatically (auto-drain on persistent failure, auto-recover on health-check pass).
4. **Operators can SEE** what the mesh is doing (live status, request traces, per-node metrics) without reading log files.
5. **The mesh is secure by default** — cluster keys are rotatable, peers can be authenticated by identity not just bearer token, request traces don't leak credentials.

By that definition we are at roughly 40%. The skeleton is solid (mesh.yml schema, drain state file, doctor checks, provider fallback), but several load-bearing wires are missing — see Stage C4 below.

---

## 2. What's shipped (the foundation)

| Layer | What's in main | Where | Plan |
|---|---|---|---|
| Routing fallback | Provider failover, typed `BackendError` dispatch, `dispatch_exhausted` aggregated WARN, recovery measured at 58.68s p99 | [router.py](../../src/maxim/models/language/router.py) | 3 + 3.5 + 4 A/B |
| Topology (declarative) | `mesh.yml` schema, FROZEN parser, `MeshConfig` dataclass with `__post_init__` validation | [mesh_config.py](../../src/maxim/peer/mesh_config.py) | 4 C1 |
| Topology (operator verbs) | `init-mesh`, `add-node`, `remove-node` — strict CI grep allow-list enforces single writer module | [mesh_setup.py](../../src/maxim/peer/mesh_setup.py) | 4 C3.1 + C3.2 |
| Mesh-aware install | `--node <name> install <extras>` composing drain → install → resume, shared `install_on_target` core, atomic `drain_node_if_absent` primitive, exit-code-3 post-install-resume-failure | [install_core.py](../../src/maxim/peer/install_core.py), [mesh_cli.py](../../src/maxim/peer/mesh_cli.py) | 4 C3.3 |
| State layer (drain) | `drain` / `resume` / `list-drained` + atomic `drain_node_if_absent`, `filelock` RMW, role-scoped, `atomic_write_secret` pattern | [drain_state.py](../../src/maxim/peer/drain_state.py) | 4 C2 + C3.3 |
| Probe + classification | `_MaximPeerBackend.health_check` + `for_url`, single shared `classify_probe_outcome` | [maxim_peer_backend.py](../../src/maxim/models/language/maxim_peer_backend.py), [probe_classify.py](../../src/maxim/peer/probe_classify.py) | 3 R2.5/R2.6 + 4 C1 |
| Observability (doctor) | `check_mesh_nodes` (with drain awareness), `check_vram_pressure`, agent_id binding, per-node `status` / `health` | [doctor/checks.py](../../src/maxim/doctor/checks.py) | 3.6 R5 + 4 A + 4 C1 |
| Observability (logs) | Structured JSONL via `MAXIM_LOG_FILE`, per-call `peer_backend_call` trace, `role_detected` event | [utils/http.py](../../src/maxim/utils/http.py) | 1 R1 + 2 R2a + 3 R2.5 |
| Bench harness | `maxim bench recovery-time` measures fail-slow → fail-fast deltas | [bench/](../../src/maxim/bench/) | 4 B |

---

## 3. The honest gap audit

A grep against `src/maxim/models/language/router.py` for any reference to `read_drained_nodes`, `drained_nodes`, or `peer.drain_state` returns **zero matches**. The router does not know that drain exists. An operator who runs `maxim peer --node <name> drain` today will see the node disappear from `list-nodes` and from `doctor`, but the next inference call from a co-resident agent will still hit it.

This is the **single biggest gap** between the current state and the definition in §1. It has its own stage (C4) and gates everything below it.

Two more honest gaps worth flagging up front:

- **`/v1/debug/vram` does not exist.** Plan 3.6 R5 deferred it explicitly. Until it ships (Stage C3.4), no remote node can answer "do you have headroom?" — which means capacity-aware routing (Stage C5) cannot start.
- **Cluster identity is one bearer token shared by all peers.** No per-peer keypair, no per-peer ACL, no rotation primitive. Acceptable for a research preview, blocking for any multi-tenant or untrusted-peer scenario.

---

## 4. Stages — in priority order

### Stage C3 finish (operator surface — small ships)

These complete the manage-the-mesh-by-hand surface. Each is a small ship.

- **C3.3 ✅ SHIPPED (PR #128, 2026-04-15):** `maxim peer --node <name> install <extras>` — mesh-aware install composing drain → install → resume around the shared `install_on_target` core in [install_core.py](../../src/maxim/peer/install_core.py). Cross-confirmed pre-merge review found + folded 17 items including the probe-cache URL mismatch (CC1) and drain TOCTOU (CC2). New `drain_node_if_absent` atomic primitive closes the TOCTOU window. Exit code 3 introduced for post-install-resume-failure distinguishability.
- **C3.4 ✅ SHIPPED (PR #142, 2026-04-17):** `GET /v1/debug/vram` admin endpoint. Returns live nvidia-smi ratio + projected model footprint from `project_vram_usage()` as JSON. 503 when nvidia-smi unavailable. Auth via bearer or localhost. Also lifted `_current_llama_server_n_ctx` to `leader_proxy.py` as canonical probe location, and fixed pre-existing `_is_debug_path`/`_route_debug` desync (deps + install-status bypassed auth gate). 2-lens pre-merge review, 11 new tests.
- **C3.5 ✅ SHIPPED (2026-04-17):** `maxim peer --node <name> update [--dry-run] [--force] [--branch <b>]` and `--node restart` — mesh-aware versions of the existing positional-URL verbs, composing drain → op → resume. HTTP wire-level logic extracted from `peer/cli.py` into shared `admin_core.py` (mirrors `install_core.py` pattern). CI grep allow-lists enforce single source of truth for `/v1/admin/update`, `/v1/admin/restart`. 2-lens pre-merge review found 1 cross-confirmed BLOCKING (dry-run bypassed self-guard) + folded 6 total findings. 42 new tests (22 mesh verb + 20 wire-level).
- **C3.6 ✅ SHIPPED (2026-04-17):** `maxim peer --node <name> llm <model>` — per-node model swap with drain → swap → resume composition. Key enabler for C5 capacity-aware routing (per-node model assignment). CI grep allow-list for `/v1/admin/llm-swap`. Shipped in the same PR as C3.5.

**Stage C3 operator surface COMPLETE.** All planned mesh management verbs shipped.

### Stage C4: Wire the router to drain state ✅ SHIPPED (PR #148, 2026-04-17)

`drain_constraint` callback injected into `LLMRouter`. `DrainConstraint` class in `peer/drain_routing.py` with mtime-cached drain file reads + URL→node mapping from `mesh.yml`. `dispatch_exhausted_all_drained` event when every candidate is drain-eliminated. Budget-blocked local fallback also respects drain. 2-lens review folded 5 findings. Plan doc: [router_drain_coupling.md](archive/router_drain_coupling.md). 27 tests.

### Stage C4.5: Auto-drain on persistent failure ✅ SHIPPED (PR #152, 2026-04-17)

Type-aware thresholds: permanent failures (auth, model_missing) auto-drain after 1, transient failures after 5 (configurable `MAXIM_AUTO_DRAIN_THRESHOLD`). `AutoDrainWriter` writes tagged entries (`# auto:<timestamp> reason:<type>`) via `atomic_write_text` under filelock. Pending buffer flushed outside `_inference_lock`. `_load_tagged_entries()` parser ready for C4.6 auto-undrain. 2-lens review folded 2 findings. Plan doc: [auto_drain_persistent_failure.md](archive/auto_drain_persistent_failure.md). 19 new tests (46 total).

### Stage C4.6: Auto-undrain via periodic health probe ✅ SHIPPED (2026-04-17)

Background daemon thread (`AutoUndrainProber`) probes auto-drained nodes every 90s (configurable `MAXIM_AUTO_UNDRAIN_PROBE_INTERVAL_S`, clamped [30, 600]) via `_MaximPeerBackend.health_check()`. On probe success, clears the auto-drain entry under filelock. **NEVER touches operator drains** (entries without `# auto:` tag are sticky). TOCTOU-safe: re-reads under lock before clearing — if an operator re-drained between probe and write, the operator drain is preserved. Singleton prober per process (review fold: `_build_local_backend` runs per lane, singleton prevents 3x probe threads). 2-lens review found 2 cross-confirmed findings (per-lane duplication + atexit accumulation) + 2 IMPORTANT (exception log level + empty cluster_key). All folded. 17 new tests (63 total in test_drain_routing.py).

**Self-healing loop complete:** C4.5 auto-drains on persistent failure → C4.6 auto-undrains on recovery. Operator drains are always sticky.

### Stage C5: Capacity-aware routing

Existing skeleton: [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md). Beyond drain (binary in/out), capacity-awareness means **picking among healthy nodes by load** — tok/s baseline, queue depth, current VRAM headroom, model availability.

**Where it pays off:**
- The `/v1/debug/vram` endpoint from C3.4 starts to pay back: the router becomes able to ask "which of my live nodes has the headroom?"
- The Plan 3.6 R5 invariant "VRAM is observability-not-routing" gets explicitly broken here, on purpose, with a real plan + review round behind the change.
- Per-node `last_seen_tok_s` baseline (currently absent) is the input signal for ranking among healthy peers.

**Estimated effort:** multi-session ship with its own pre-design review round. 0.5 → 0.6 territory.

### Stage C6: Admin API + dashboard surface

Today's admin endpoints are minimal: `/v1/admin/update`, `/v1/admin/install`, `/v1/debug/install-status`, `/v1/debug/deps`, the inference endpoints. Missing for full operator visibility:

- `/v1/admin/drain` + `/v1/admin/resume` (admin-API equivalents of the CLI verbs)
- `/v1/admin/mesh` (read-only `mesh.yml` dump for remote inspection)
- `/v1/debug/drain-state` (read drain state, for remote doctor)
- `/v1/debug/router-stats` (per-provider success rate, latency, last-error, current-backoff state)
- `/v1/debug/request-trace` (ring buffer of recent inference attempts, for post-mortems)

A web TUI or terminal dashboard on top of these is a nice-to-have, **NOT** load-bearing.

**Hard constraint from C2:** all C6 admin-API state writes go to `~/.maxim/util/`, never to `mesh.yml`. The `mesh.yml` declarative-vs-mutable-state split is non-negotiable. When in doubt, the admin endpoint writes to a state file, not topology. Adding a new mutable mesh surface means: (1) put it in `~/.maxim/util/`, (2) role-scope the filename, (3) wrap the RMW in `filelock.FileLock`, (4) `atomic_write_secret` for credentials, (5) validate against `mesh.yml`'s node set at write time.

**Estimated effort:** medium ship. Endpoints themselves are small, but auth + rate-limiting + trace ring buffer + permission model is a coherent design problem.

### Stage C7: Cluster security hardening

Tracked in [node_security_simplification.md](node_security_simplification.md), `feedback_strict_grep_caller_allowlist.md`, and the C2 invariants:

- **Cluster key rotation** (`maxim peer rotate-cluster-key`) — the canonical test of whether the strict CI grep allow-list rule needs to relax. The right answer is probably option (c) from `feedback_strict_grep_caller_allowlist.md`: split the secret into `~/.maxim/util/cluster_key.{role}` and have `mesh.yml::cluster_key` become a fallback. This is the spec-vs-status split applied to the secret.
- **Cluster-key consistency doctor check** (`check_cluster_key_consistency`) — **surfaced by C3.3 fold review (Blast Radius B2)**. Today `mesh.yml::cluster_key` and `peer.yml::api_key` can diverge silently if the operator rotates one without the other, because `init-mesh` copies once but the two files evolve independently after that. `maxim peer --node X install` uses `mesh.yml::cluster_key`; `maxim peer install` uses `peer.yml::api_key`. The symptom is "one install verb 401s, the other succeeds, against the same target." The doctor check should compare both values when both files are present and warn on mismatch. **Deferred from C3.3** (docs-only warning added to `cli-reference.md` + `mesh_debug.md`) because the full cluster-key rotation story should land before the consistency check so the check has a remediation path to point at.
- **Per-agent rate limiting** — read from `~/.maxim/util/rate_limits.{role}.json`. Already mentioned as a future C3 deferred item in C2 invariants. The `KeyedRateLimiter` primitive from Plan R0 already lives in [runtime/rate_limit.py](../../src/maxim/runtime/rate_limit.py).
- **Request trace ring buffer** — `~/.maxim/util/request_trace.{role}.jsonl` with size cap + rotation. Feeds C6's `/v1/debug/request-trace`.
- **Per-peer identity** — currently every peer has the same bearer token (the cluster key). Real identity (per-peer keypair, per-peer ACL) is a bigger ship.

**Estimated effort:** another multi-session ship, 0.6 → 0.7. Low priority until real multi-tenant use cases show up.

### Stage C8: Cross-version compatibility

`mesh.yml::protocol_version` exists as a field but isn't checked anywhere on probe responses. The intent (per C1) was to support graceful degradation when the mesh has mixed-version nodes. Concrete future work:

- Probe responses include the responder's protocol version
- Router downgrades capabilities when talking to older nodes
- `maxim doctor` warns on version skew across nodes
- `mesh.yml` schema bump migrations (today the parser is FROZEN; the next bump needs a clear migration path)

**Estimated effort:** small unless a real version skew incident forces it. Defer until needed.

### Stage C9: Mesh doc transport

Standardized small-document (`.md` / `.json`) exchange between mesh nodes. The missing primitive for **peer-to-peer coordination** — today's cross-node channels are inference traffic (LLM prompts), admin verbs (one-shot side effects), and structured logs (read-only operator surface). None of them let agent A on node X deposit a structured doc that agent B on node Y can read later. Plan detail in [mesh_doc_transport.md](mesh_doc_transport.md).

**The killer use case:** multi-agent coordination. When Maxim agents run on separate mesh nodes, they need a standardized channel for "share this context with your sibling." Doc-drop is the smallest useful unit. The immediate operator-facing win is multi-session Claude collaboration (parallel Claude sessions on the operator's leader + peer can exchange context via an inbox/outbox), but the long-term play is agent-to-agent coordination as the foundation for C4.5 auto-drain announcements, C7 cluster-key rotation broadcasts, and the Mother Maxim precursor.

**v1 shape (minimal):**
- Endpoint family: `PUT / GET / DELETE /v1/mesh/docs/<namespace>/<key>` + list endpoints
- Storage: `~/.maxim/util/mesh_docs/<namespace>/<key>.{json,md}` per the C2 state-layer invariant
- Shared core: `src/maxim/peer/mesh_doc_core.py` mirroring the C3.3 `install_core.py` pattern (single source of truth, CI grep allow-list)
- CLI: `maxim peer --node <name> docs put|get|ls|rm`
- 1 MB / doc cap, 100 MB / namespace cap, 24h default TTL with namespace overrides
- Authorization: shared cluster key (v1 limitation — C7 per-peer identity layers on top later)
- Delivery: pure pull (recipient polls); long-poll / webhook deferred to v2+

**Orthogonal to C4/C5.** Ships on its own timeline — doesn't block reactivity work and isn't blocked by it. Pre-design review round answers 5 open questions (secret-bearing policy, role-scoping, delivery semantics, namespace creation, authorization). See [mesh_doc_transport.md](mesh_doc_transport.md) §"Open design questions."

**Estimated effort:** ~3 sessions (design + implementation + review fold + docs). ~400-600 LOC + tests.

### Stage C10: Mesh perception transport

Peer-tunneled sensory percepts. Drives the Reachy Mini app form-factor: the app runs as a peer node, performs on-device STT + vision segmentation + drive sensor reads, and tunnels event-shaped `Percept` instances to a leader for cognition. Generalizes to any embodiment peer with sensors but limited compute (future Minecraft adapter, multi-Maxim training rigs). Plan detail in [mesh_perception_transport.md](mesh_perception_transport.md).

**Architectural framing:** the mesh today has exactly one cross-node transport (`_MaximPeerBackend` for LLM inference, with its load-bearing "one HTTP call, no retry" invariant). Perception traffic has different failure semantics (a dropped sensor frame is a missed observation, not a router-failover event) and wants a sibling typed backend rather than an extension. C10 establishes the **typed-transport-per-purpose** pattern as an architectural invariant — Hivemind substrate-bundle exchange (1.1+) is the second consumer that proves the abstraction.

**Split timeline (the unusual part):**
- **1.0 prep** — additive wire-format reservations, Protocol-contract documentation, and the typed-transport-per-purpose invariant land in CLAUDE.md before 1.0 ships. ~150 LOC + ~3 tests + 2 invariants. One session. The refactor-now-or-refactor-later test: post-1.0, adding required `PerceptSource` Protocol members would break third-party implementations, and silently extending `Percept.to_dict` would introduce a session-format-drift class of bug. Cheap to pin now, expensive to pin later.
- **1.1 ship** — actual `_MaximPercepTransport` typed backend, `RemotePerceptSource` adapter, `/v1/mesh/percepts/<node>` endpoint family, CLI verbs, Reachy app peer-side integration. ~400-600 LOC alongside Hivemind, which proves the typed-transport-per-purpose pattern with two siblings instead of one.

**v1 scope cut (1.1):** processed event-shaped percepts only — no raw video / raw audio frames. Peer does on-device perception; frames stay local. Raw-frame streaming is a 1.2+ concern when (and if) the substrate-level vision encoder is load-bearing enough to need them.

**Estimated effort:** 1.0 prep ~1 session (~150 LOC). 1.1 implementation ~3-4 sessions (~400-600 LOC + tests + Reachy app wiring + 3-lens pre-merge review).

---

## 5. Mapping to versions

| Version | Includes | Status |
|---|---|---|
| **0.4** (in flight) | Plan 4 C3.3 → C3.6 (operator verb surface) + C3.4 VRAM + C4/C4.5/C4.6 reactive drain | **C3.3-C3.6 SHIPPED**; **C4+C4.5+C4.6 SHIPPED** — reactive mesh complete |
| **0.5** | C4.6 auto-undrain + C5 capacity-aware routing + substrate P3a / P4 / B3-B5 | C4.6 design needed |
| **0.6** | C5 capacity-aware routing + C6 admin API + dashboard + **C9 mesh doc transport** | not started |
| **0.7+** | C7 security hardening + C8 cross-version compat | not started |
| **1.0** | Cross-session learning demonstration (banner) — separate from mesh. **C10 prep (1.0 slice)** lands here: additive wire-format reservations + CLAUDE.md invariants for the typed-transport-per-purpose pattern. ~150 LOC, no new transport. | C10 prep not started |
| **1.1** | **C10 ship (1.1 slice)** alongside Hivemind: `_MaximPercepTransport` typed backend + `RemotePerceptSource` adapter + endpoint family + Reachy app peer integration. ~400-600 LOC. | C10 ship not started |

**Note on C9 placement:** C9 is orthogonal to the reactivity work (C4/C4.5/C5) and doesn't share file territory with any other stage — it can slip earlier (0.5) if multi-agent coordination becomes a user-visible blocker, or later (0.7) if it stays a nice-to-have. 0.6 is the default slot because it pairs naturally with C6's admin API work on the same endpoint surface.

**Note on C10's split timeline:** C10 is the only stage in the roadmap that deliberately splits across versions. The 1.0 prep slice is refactor-hygiene (pin the Protocol contract, pin the wire format, document the typed-transport playbook) — cheap now, expensive post-1.0 because the prep items touch frozen surface (`PerceptSource` Protocol, `Percept` wire format under `_format_version`). The 1.1 ship slice waits for Hivemind because the typed-transport-per-purpose pattern is proven by two siblings (perception transport + substrate-bundle transport) rather than presumed by one.

---

## 6. Recommended sequencing

We have built the **bones** of a reactive peer mesh: declarative topology, mutable state layer, probe primitive, doctor visibility, provider fallback at the router. What's missing is the **nervous system** — the wiring from operator intent (drain) and from observability (failure rates, VRAM pressure) into runtime routing decisions. That wiring lands in **C4**, which is a real plan doc, not a half-day patch.

C3.3 → C3.6 are valuable operator polish but they don't change reactivity at the runtime layer.

**Two competing sequences:**

**Sequence A — finish operator surface first.** C3.3 → C3.4 → C3.5 → C3.6 → C4 → C5 → C6 → C7 → C8.
Pro: every C3 ship is small and low-risk. Operator gets steady incremental wins.
Con: reactivity (C4) is delayed by 4 sessions. Operators may grow accustomed to drain-as-UI-hint and be surprised when C4 changes the contract.

**Sequence B — pivot to C4 after C3.3.** C3.3 → C4 → C3.4 (now informs C5) → C5 → C3.5+C3.6 in parallel with C6 design.
Pro: reactivity gate clears earliest. C4 informs the C5 design. Operators learn drain-as-routing-constraint from day one.
Con: C3.5+C3.6 (the per-node update/restart/llm verbs) get delayed, which means operators still can't safely update a single node without operating directly on the leader URL.

**My recommendation: Sequence B**, with C3.4 (`/v1/debug/vram`) reordered to run in parallel with C4 because the two have zero file overlap (C3.4 is a leader-side endpoint addition, C4 is a peer-side router consumer change). C3.5+C3.6 land after C4 + C3.4 are both in.

---

## 7. Open architectural questions that need real answers before C4 starts

1. **Provider-name ↔ mesh-node-name mapping.** Is it lookup-by-URL? lookup-by-explicit-binding-in-mesh.yml? lookup-by-profile-name? The answer drives the whole C4 implementation.
2. **Drain cache invalidation.** mtime poll with a 1s window? inotify with poll fallback? pub-sub via a process-local channel? Each has a different blast radius.
3. **What happens when mesh.yml changes at runtime?** Today the answer is "nothing — restart the daemon." C4 may force a real answer because drain state and mesh.yml are now jointly consulted on every dispatch.
4. **Drain semantics in a single-node mesh.** Refusing to drain self is C2 invariant 3. But in a single-node mesh, every node IS self. Does drain become a no-op? An error? An auto-pause-router?
5. **Auto-drain failure modes.** If C4.5 ships, the router becomes able to evict providers without operator action. Does this need a "panic mode" where if N% of nodes are auto-drained the router holds them all in (because losing N% looks like a network partition, not a node failure)?

These are the things a C4 plan doc has to answer. None of them are blockers for C3.3 → C3.6.

---

## 8. Re-check triggers

Update this roadmap when:

- **A new C3.x sub-stage is identified** — most likely from operator feedback during real mesh use.
- **C4 plan doc is drafted** — link it from §4 Stage C4.
- **C9 plan doc activates** — when multi-agent coordination becomes a user-visible blocker OR when two C3.x+ features in a row would want the doc-transport primitive. See `mesh_doc_transport.md` §"Re-check triggers" for the activation criteria.
- **C10 1.0 prep lands** — update Stage C10 with the PR + commit hash and mark the prep slice shipped. Re-check before 1.1 implementation whether any prep invariant needs revision based on Hivemind's parallel substrate-bundle transport design.
- **Reachy app team firms up timeline** — C10's driving consumer. If the app ships before 1.1 is ready, decide whether to compress the 1.1 work or stand up a temporary Reachy-specific path with an explicit migration commitment back to the typed-transport playbook.
- **Any C5 / C6 / C7 / C9 architectural decision conflicts with a load-bearing invariant** — flag it explicitly here, don't let the conflict accumulate silently.
- **The "fully reactive" definition in §1 starts to feel wrong** — that's the load-bearing piece. If the definition shifts, every stage estimate shifts with it.
- **Versions ship and the table in §5 needs to slip** — common, expected, no apology needed; just keep it honest.

---

## 9. Related plans

- [archive/llm_path_operator_visibility.md](archive/llm_path_operator_visibility.md) — Plan 4 (ALL SHIPPED), which feeds C3.x in this roadmap.
- [llm_path_refinement.md](llm_path_refinement.md) — the Plan 1-3.6 ancestor chain.
- [llm_path_peer_failover.md](llm_path_peer_failover.md) — Plan 3.6 (VRAM spillover detection).
- [deferred/llm_mesh_capability_aware.md](deferred/llm_mesh_capability_aware.md) — the C5 skeleton.
- [deferred/llm_path_multi_peer_dispatch.md](deferred/llm_path_multi_peer_dispatch.md) — feeds C5.
- [node_security_simplification.md](node_security_simplification.md) — feeds C7.
- [mesh_doc_transport.md](mesh_doc_transport.md) — Stage C9 shell plan (mesh-to-mesh structured doc exchange).
- [mesh_perception_transport.md](mesh_perception_transport.md) — Stage C10 shell plan (peer-tunneled sensory percepts; split 1.0 prep / 1.1 ship). Driving consumer: Reachy Mini app peer.
- [maxim_hivemind.md](maxim_hivemind.md) — 1.1+ substrate-bundle exchange; second consumer of the typed-transport-per-purpose pattern C10 establishes in 1.0.
- [cross_platform_file_lock.md](cross_platform_file_lock.md) — shell plan to unify the two file-lock APIs (`utils/process_lock` and `filelock.FileLock`); blocks nothing, useful cleanup post-C4.
