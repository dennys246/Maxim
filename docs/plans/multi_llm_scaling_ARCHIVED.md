# Multi-LLM Scaling Plan — ARCHIVED (Complete)

> **Status:** All phases complete as of 2026-04-05. Remaining work (mDNS discovery + InferenceRouter) folded into [agent_mesh.md](agent_mesh.md) as Phases 0a-0b.

## What was delivered

| Phase | What | Status |
|-------|------|--------|
| 1-3 | LaneConfig, LaneModelConfig, LaneBackendManager with safety gates | Done |
| 4-5 | Remote LLM backend, Cloudflare tunnel docs | Done |
| 6 | LocalServerSpawner, leader mode detection, auto-spawn | Done |
| 7a | LeaderProxy — stdlib reverse proxy on :8099 (auth, logging, GPU metrics, debug endpoints) | Done |
| 7a-ext | Remote self-update — `maxim peer update` via LeaderProxy | Done |
| 7b | Admission control — concurrency semaphore + per-peer rate limiting | Done |
| 8 | LaneMetrics — per-lane p50/p99, failure rate, token throughput | Done |
| Heartbeat | System metrics collector + stall detection | Done |

## What moved to Agent Mesh

| Phase | What | Now |
|-------|------|-----|
| 7c | PeerRegistry + mDNS discovery | Agent Mesh Phase 0a |
| 7d | InferenceRouter (local → peer → remote fallback) | Agent Mesh Phase 0b |
| 7e | Multi-front input | Deferred to Phase 11 (no concrete use case) |

## Key files

- `runtime/leader_proxy.py` — LeaderProxy (auth, logging, GPU metrics, admin endpoints)
- `runtime/lane_backends.py` — LaneBackendManager, build_primary_router
- `runtime/rate_limiter.py` — TokenBucket + PeerRateLimiter
- `runtime/heartbeat.py` — HeartbeatMonitor
- `runtime/system_metrics.py` — GPU/CPU/RAM/disk/WiFi collector
- `runtime/leader_mode.py` — Role detection
- `runtime/local_server_spawner.py` — Auto-spawn llama-cpp-server
- `models/language/lane_metrics.py` — LaneMetrics + MetricsRegistry
- `models/language/mesh_trace.py` — Request-id propagation + trace logging
- `models/language/openai_backend.py` — OpenAI-compatible backend
- `peer/cli.py` — `maxim peer connect/show/forget/update`
- `tunnel/` — Cloudflare tunnel management
- `doctor/checks.py` — Platform-aware diagnostics

## Troubleshooting

- [docs/troubleshooting/peer_leader_connectivity.md](../troubleshooting/peer_leader_connectivity.md)
- [docs/troubleshooting/peer_diagnosis_runbook.md](../troubleshooting/peer_diagnosis_runbook.md)
- [docs/troubleshooting/remote_update.md](../troubleshooting/remote_update.md)
- [docs/troubleshooting/maxim_doctor.md](../troubleshooting/maxim_doctor.md)
