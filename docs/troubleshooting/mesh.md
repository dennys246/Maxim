# Agent Mesh Troubleshooting

> **⚠ LEGACY — this document describes code that was deleted on 2026-04-12.**
>
> As of R0 of [llm_path_foundation.md](../plans/llm_path_foundation.md), the
> mesh modules referenced below (`PeerRegistry`, `PeerChannel`, `TaskDelegator`,
> `ExperienceBroker`, `MeshAdmissionControl`, `PeerClockEstimator`,
> `AgentIdentity`) have been deleted as dead code — they had zero production
> imports and were swept along during the task→size lane refactor without
> being re-wired.
>
> **For current peer routing troubleshooting**, see
> [peer_leader_connectivity.md](peer_leader_connectivity.md) and
> [peer_diagnosis_runbook.md](peer_diagnosis_runbook.md).
>
> **The replacement runbook** ([mesh_debug.md](mesh_debug.md)) will be
> written by Plan 4 (`llm_path_operator_visibility.md`) once the new
> `mesh.yml` + admin API ships. Until then, the content below is preserved
> for historical reference only. Do not follow these instructions as
> operational guidance — the classes they reference no longer exist.

## Peer Discovery & Connectivity

### Peer not found in registry

`PeerRegistry.get_peer()` returns None.

**Causes:**
- No peer config file (`~/.config/maxim/peer.yml`)
- Peer config has wrong URL or missing API key
- Peer was never registered (no `maxim peer connect` or manual registration)

**Fix:**
```bash
# Check peer config exists
maxim peer show

# Connect to a peer
maxim peer connect https://maxim.example.com/v1 --key sk-your-key

# Verify connectivity
maxim peer test https://maxim.example.com/v1
```

### PeerChannel send queue full (message dropped)

Log: `mesh send queue full, dropping message to <peer_id>`

**Cause:** The background send thread can't drain messages fast enough. Usually means the peer is unreachable and retries are consuming the queue.

**Fix:**
- Check peer is alive: `curl -s https://peer-url/v1/debug/ping`
- Check network: `maxim doctor`
- If persistent, the peer may be down — messages will be dropped until it recovers

### PeerChannel retries exhausted

Log: `failed to send GOAL_PROPOSAL to <peer_id> after 3 attempts`

**Cause:** Network failure, peer offline, or auth rejected (401).

**Fix:**
- Verify peer URL: `maxim peer show`
- Test connectivity: `maxim peer test <url>`
- Check API key: ensure the key matches the peer's `maxim tunnel key export`
- Check firewall / tunnel status: `maxim tunnel status`

---

## Admission Control

### Peer gated (429 responses)

Log: `mesh: gating peer <peer_id> for 30s — rate limit exceeded (61/60 in 60.0s)`

**Cause:** Peer sent too many messages in the rate-limit window. Escalating gates: 30s → 2min → 10min → 1hr.

**Fix:**
- If your peer is in a tight loop, fix the loop
- If legitimate high traffic, the peer needs higher trust level (verified=120/min vs unknown=20/min)
- Manual ungate: not yet exposed via CLI — requires code: `admission.ungate_peer("peer_id")`

### Burst detection triggered

Log: `mesh: gating peer <peer_id> for 30s — burst detected (21 msgs in 5.0s)`

**Cause:** 20+ messages in a 5-second window. Usually a buggy peer.

**Fix:** Same as rate-limit gating above. Check the peer's send loop.

### Trust level rates

| Trust level | Rate limit | How to achieve |
|------------|-----------|----------------|
| `verified` | 120 msg/min | Pre-shared key in peer config |
| `discovered` | 60 msg/min | mDNS auto-discovery (Phase 0a, future) |
| `remote` | 60 msg/min | Cloudflare tunnel auth |
| `unknown` | 20 msg/min | No auth / unsolicited contact |

---

## Knowledge Sharing

### Imported CausalLink not appearing in NAc

**Causes:**
- Deduplication: a link with the same event→outcome signature already exists locally
- No receiver registered: `ExperienceBroker` has no `CausalLinkReceiver`
- Wrong `knowledge_type` in payload (must be `"causal_link"`)

**Verify:**
```python
from maxim.mesh.knowledge import ExperienceBroker
broker = ExperienceBroker()
print(broker.receiver_types)  # Should include "causal_link"
```

### Motor program rejected on import

Log: `motor_program: rejecting <name> — entity <path> not in local spec`

**Cause:** The imported program references an entity path that doesn't exist in the local embodiment spec. This is expected — programs from a different robot body won't transfer unless the entity paths match.

**Fix:** This is by design. Only programs from homogeneous robots (same spec) will transfer. No action needed.

### Motor program rejected due to trust

Log: `motor_program: rejecting from <peer> (trust=unknown, need {'verified', 'discovered'})`

**Cause:** Motor programs require at least `"discovered"` trust level because they touch hardware.

**Fix:** Establish trust via pre-shared key (`maxim peer connect <url> --key <key>`) or mDNS discovery (future).

### Transfer discount seems too aggressive

Imported knowledge starts at reduced confidence:

| Trust | CausalLink discount | Reflection salience cap | Motor program |
|-------|--------------------|-----------------------|---------------|
| verified | 0.5 | 0.5 | 0.5 confidence, stats reset |
| discovered | 0.3 | 0.35 | 0.5 confidence, stats reset |
| remote | 0.3 | 0.35 | Rejected |
| unknown | 0.1 | 0.15 | Rejected |

This is intentional — local experience always dominates. Imported knowledge is a prior, not a replacement. As the local agent observes the same patterns, confidence grows organically.

---

## Task Delegation

### Delegation times out

`TaskDelegator.delegate()` returns `None`.

**Causes:**
- Peer is overloaded (MAX_DELEGATION_QUEUE = 5 exceeded)
- Peer doesn't have the required tool
- Network partition (peer unreachable)
- Peer's NAc predicts failure for the tool (confidence > 0.7)

**Diagnose:**
```python
# Check peer identity for tool availability
peer = registry.get_peer("peer_id")
print(peer.identity.available_tools if peer and peer.identity else "no identity")
```

### Delegation cycle detected

Log: `delegation cycle detected`

**Cause:** The goal has been delegated back to an agent that's already in the `delegation_chain`. Max depth is 2.

**Fix:** This is a safety mechanism. If it fires repeatedly, check why peers are bouncing goals back — likely none of them have the required tool or all predict failure.

### Delegation rejected: overloaded

Log: `overloaded (5 active)`

**Cause:** The receiving agent already has 5 concurrent delegations running.

**Fix:** Wait for existing delegations to complete, or try a different peer.

---

## Clock Synchronization

### Large clock offset between peers

`PeerClockEstimator` shows offset > 60s.

**Cause:** System clocks are out of sync. Common on machines without NTP configured.

**Fix:**
- Enable NTP on both machines: `sudo timedatectl set-ntp true` (Linux)
- The estimator will learn and correct the offset automatically via heartbeat RTT

### SCN temporal bins misaligned

Imported memories landing in wrong hour bins.

**Cause:** Clock skew > 30 minutes shifts memories into adjacent bins.

**Fix:**
- Use `SCN.register_external()` instead of `SCN.register()` for peer memories
- Ensure heartbeat exchange is active (clock estimator needs RTT data)
- Check: `estimator.get_estimate("peer_id")` — confidence should be > 0.1

### Clock estimate not converging

`PeerClockEstimate.confidence` stays low after many heartbeats.

**Cause:** High RTT variance (unstable network), or heartbeat payloads missing the sender's timestamp.

**Fix:**
- Check heartbeat payload includes `"timestamp": time.time()`
- Check network stability: high jitter degrades offset estimation
- Confidence reaches 1.0 after 10 sync points with stable RTT

---

## Protocol Version Issues

### "Unsupported mesh protocol version X"

`MeshMessage.from_dict()` raises `ValueError`.

**Cause:** The sending peer is running a newer version of Maxim with a breaking protocol change.

**Fix:**
- Update the local Maxim installation to match the peer
- Lower-version messages are always accepted (backward compat)
- Only higher-version messages are rejected

---

## Environment Variables

| Variable | Default | Purpose |
|---------|---------|---------|
| `MAXIM_PEER_ENABLED` | `0` | Enable mesh peer features (future: mDNS) |
| `MAXIM_PEER_ADVERTISE` | `0` | Broadcast identity on LAN (future: mDNS) |
| `MAXIM_LANE_TRACE` | `0` | Log per-request LLM traces (includes mesh requests) |
| `MAXIM_PEER_LOG_REQUESTS` | `0` | Structured JSON log per outbound peer call |

---

## Getting More Help

- Run `maxim doctor` for platform-aware diagnostics
- Run `maxim peer test <url>` to verify connectivity
- File issues at https://github.com/anthropics/maxim/issues
