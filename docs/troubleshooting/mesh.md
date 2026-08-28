# Agent Mesh — Post-Mortem Note (R0 deletion)

**This document is a historical note.** The "agent mesh" described here was a speculative subsystem that existed in the codebase from an older design phase but was never wired into production. It was deleted in R0 of the LLM Path Refinement plan (commit `e811787`, 2026-04-11) as part of a broader dead-code sweep. This note exists so that anyone following a stale link (commit message, plan doc, internal chat) understands what was removed, why, and where current peer troubleshooting lives.

## What was deleted

~1,250 LOC across seven modules, all under `src/maxim/mesh/`:

| Deleted module | What it claimed to do | Status at deletion |
|---|---|---|
| `peer_registry.py` | In-memory registry of known peer agents | Zero production imports |
| `peer_info.py` | Metadata about a known peer (trust level, capabilities) | Zero production imports |
| `peer_channel.py` | Persistent bi-directional message channel to a peer | Zero production imports |
| `task_delegation.py` | `TaskDelegator` — delegate goals to peers by capability | Zero production imports |
| `knowledge.py` | `ExperienceBroker` — broadcast causal links / motor programs | Zero production imports |
| `clock.py` | `PeerClockEstimator` — NTP-free offset estimation via heartbeat RTT | Zero production imports |
| `agent_identity.py` | `AgentIdentity` — tool list + trust level advertisement | Zero production imports |

All seven modules were swept up during the task→size lane refactor and never re-wired to the live agent loop. They were discoverable only by reading `mesh/__init__.py` and by this troubleshooting doc.

**Kept** (simulation-only, still referenced):
- `mesh/bus.py` — in-process pub-sub used by `simulation/` and `create.py`
- `mesh/identity.py` — stub used by simulation NPCs
- `mesh/message.py` — message dataclass used by simulation
- `mesh/naming.py` — agent naming helper used by simulation

**Cherry-picked before deletion:** `mesh/admission.py`'s rate-limiting logic (~150 LOC) was moved to `runtime/rate_limit.py` because Plan 4's admin API will consume it for per-agent rate limiting. Everything else was deleted clean.

## Why deleted

Three reasons, ordered by how much weight each carried:

1. **Zero production imports.** Every deleted module was discoverable only by the parallel `mesh/__init__.py` re-exports. `grep -rn "from maxim.mesh" src/maxim/` showed only `bus`, `identity`, `message`, `naming` in live use.
2. **Misleading to future work.** Plan 4 Stage C went on to introduce a new mesh abstraction — `mesh.yml` config + setup verbs (`init-mesh`, `add-node`, `remove-node`) + `drain`/`resume`/`list-drained` runtime state — shipped across PRs #112 (C1), #113 (C2), #118 (C3.1), and the C3.2 follow-up. Having a parallel dead mesh tree under `mesh/` would have confused the executor about which one to build on. The new surface lives in [`src/maxim/peer/mesh_setup.py`](../../src/maxim/peer/mesh_setup.py) and [`src/maxim/peer/mesh_config.py`](../../src/maxim/peer/mesh_config.py); operator runbook is [mesh_debug.md](mesh_debug.md).
3. **Audit cleanup aligned with a motivating incident.** The 2026-04-12 Cloudflare User-Agent incident (commit `8b52cbd`) surfaced a second class of dead code — scattered urllib call sites that Plan 1 R1 went on to consolidate. Deleting both together made the LLM path refinement's first commit a net negative-LOC "cleanup + consolidation" move.

## Where current peer troubleshooting lives

**Runbooks that replace pieces of the old mesh doc:**

- [peer_leader_connectivity.md](peer_leader_connectivity.md) — end-to-end network diagnosis for peer → leader calls (DNS, TCP, TLS, Cloudflare tunnel, upstream port routing). Replaces the "Peer Discovery" section above.
- [peer_diagnosis_runbook.md](peer_diagnosis_runbook.md) — step-by-step "my peer isn't talking to the leader, what do I check first" playbook. Replaces most of "Peer Discovery & Connectivity".
- [http_debugging.md](http_debugging.md) — **new in Plan 1 R1.** Covers the unified HTTP client, structured JSONL events, timeout tuning, Cloudflare diagnosis, pool exhaustion. Replaces the old "PeerChannel send queue full" and "retries exhausted" sections with a modern event-based debug surface.
- [leader_proxy_debug.md](leader_proxy_debug.md) — leader-side proxy runbook for admission control, request forwarding, GPU header injection. Replaces the "Admission Control" section that used to reference `mesh/admission.py`.
- [remote_update.md](remote_update.md) — `maxim peer update` / `maxim peer restart` troubleshooting.

- [mesh_debug.md](mesh_debug.md) — **new in Plan 4 Stage C.** Operator runbook for the `mesh.yml` declarative config + `init-mesh` / `add-node` / `remove-node` setup verbs + `drain` / `resume` / `list-drained` runtime state. The first place to look for "I added a node and it isn't routing" or "I drained a peer and it didn't come back" symptoms.

## What the old mesh doc got right

Not everything in the deleted mesh design was wrong — a few concepts survive and inform Plan 4:

- **Trust levels** (`verified` / `discovered` / `remote` / `unknown`) were a reasonable shape for differentiating LAN peers from remote tunnel peers. Plan 4 Stage C's `mesh.yml::cluster_key` collapsed this to a binary (valid cluster key = trusted, invalid = rejected), which is simpler but gives up the per-peer gradation. If operators later need gradation, the trust-level table is a starting point.
- **Per-peer rate limiting** (the `KeyedRateLimiter` in `admission.py`) was a solid primitive. It's been cherry-picked into `runtime/rate_limit.py` and Plan 4 will consume it for per-agent rate limiting at the router entry.
- **Knowledge transfer with trust-scoped confidence discount** (imported CausalLinks starting at lower confidence) is a good default for cross-agent learning. The substrate plan's cross-agent transfer work will revisit this pattern.

## Environment variables no longer recognized

The following env vars were documented in the old runbook and are no longer read by any code:

- `MAXIM_PEER_ENABLED` — referenced `mesh/__init__.py` startup; deleted.
- `MAXIM_PEER_ADVERTISE` — referenced `mesh/agent_identity.py`; deleted.

`MAXIM_LANE_TRACE` and `MAXIM_PEER_LOG_REQUESTS` still work — they were already part of the live `mesh_trace.py` module (not deleted).

## If you arrived here from a stale link

You probably clicked through from one of:

- A plan doc that references `PeerRegistry` or `PeerChannel` — those references are being updated piecemeal. See [tool_refinement_plan.md](../plans/deferred/tool_refinement_plan.md) which had a line-98 reference that was fixed in an earlier PR, and check whether the plan you're reading has been refreshed.
- An old git commit message mentioning `ExperienceBroker` or `TaskDelegator` — the commit is historical, the code is gone.
- A CLAUDE.md lesson entry — those should all be current. If you find a stale one, it's a bug.

The current cluster abstraction is:

- **Peer credentials:** `~/.config/maxim/peer.yml` (per-host: leader URL + API key + role-detection signal — left in place by every Stage C verb because `runtime/role.py` reads its existence)
- **Mesh topology:** `~/.config/maxim/mesh.yml` (declarative cluster topology — `cluster_key` + `self` + `nodes[]`; written only by `mesh_setup.py` via the strict CI grep allow-list)
- **Mutable runtime state:** `~/.maxim/util/drained_nodes.{role}.txt` (list of drained node names; serialized via `filelock.FileLock`)
- **Peer → leader routing:** `lane_backends.py::build_primary_router` + `models/language/router.py` (still the hot path)
- **Admission control:** `LeaderProxy` in `runtime/leader_proxy.py` (still the hot path)
- **Per-peer observability:** `mesh_trace.py` structured events, now emitted via `http.fetch_url` (Plan 1 R1)

See [../architecture/llm_routing.md](../architecture/llm_routing.md) for the authoritative layer diagram.
