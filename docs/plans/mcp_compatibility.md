# MCP Compatibility — server, client, and schema interop

**Status:** Design exploration, post-1.0 (1.1 track). Stub.
**Target version:** 1.1+
**1.0 prerequisite:** [v1_refinement.md CC9](v1_refinement.md) — Tool schema dual-format support shipped in PR #204. Door is open.
**Concurrent with:** 1.0 stabilization.

---

## Goal

Make Maxim a first-class citizen in the MCP ecosystem: expose Maxim's tools to other MCP clients, consume MCP servers' tools as Maxim tools, share resources and prompts. Three usage patterns:

1. **Maxim as MCP server** — other agents (Claude Desktop, Cursor, custom clients) connect to a running Maxim instance and call its tools. The Maxim agent's bio-systems learn from outcomes; the calling agent gets the tools.
2. **Maxim as MCP client** — Maxim's tool registry includes tools from external MCP servers (filesystem, git, browser, custom domain tools). The bio-pipeline learns affordances of external tools the same way it learns local SEM affordances.
3. **Tool spec interop** — a tool authored in MCP format can be loaded into Maxim without rewriting; a tool authored in Maxim format can be exported to MCP without rewriting.

---

## What shipped in 1.0 (CC9 freeze hardening)

[v1_refinement.md CC9](v1_refinement.md) shipped in PR #204:

1. `Tool.to_json_schema()` method converts the custom `{"name": type}` format to JSONSchema 2020-12 (the wire format MCP uses).
2. `Tool.input_schema` accepts JSONSchema dict OR custom format (auto-detect at construction).
3. Internally normalize to existing format — no behavior change in 1.0.
4. Documented as "JSONSchema is canonical going forward; custom format is a convenience."

Without CC9, third parties writing tools against 1.0's custom format would break when MCP support lands. With CC9, MCP work is purely additive in 1.1+.

**Load-bearing CC9 invariants** (from CLAUDE.md):
- Format-sensitive consumers route through `to_json_schema()` rather than reading `input_schema` directly.
- The `_resolve_property_type` "unknown → dict" fallback is intentional today (in-process tool authors only); 1.1 should raise on unknown types when ingesting external schemas.
- Strict MCP clients (Claude Desktop in strict mode) reject extra parameters and treat `required` strictly. The legacy description-as-value pattern (`{NAME: "Optional ..."}`) emits `required: [NAME]` — a CC9 docstring already flags this; 1.1 may need a migration tool.

---

## What ships in 1.1+ (this plan)

### M1. MCP client — consume external MCP servers (~300 LOC)

Wire `mcp` Python SDK into the tool registry. New `MCPToolBridge` adapts MCP tools to Maxim's `Tool` ABC. Configuration via `mesh.yml` or a new `~/.maxim/mcp_servers.yaml`:

```yaml
mcp_servers:
  - name: filesystem
    command: ["mcp-server-filesystem", "/Users/me/projects"]
  - name: git
    command: ["mcp-server-git"]
```

On startup, Maxim spawns each server, fetches its tool list, registers each as a Maxim Tool. Tool calls forward to the MCP server; results map back to `ToolOutput`.

**Bio-pipeline integration:** MCP tool calls produce `ToolOutput.side_effects` the same way local tools do. Failure modes propagate to NAc. The agent learns to use external tools the same way it learns to use local ones.

**Strict-schema ingestion:** unlike in-process tools (where unknown types fall back to `dict`), externally-supplied JSONSchema crossing the trust boundary raises on unknown types. The `_resolve_property_type` strict-mode toggle exists in CC9; M1 wires it on.

### M2. MCP server — expose Maxim as an MCP server (~250 LOC)

`maxim mcp serve [--port N] [--agent <agent_id>]` runs Maxim's tool registry behind an MCP server. Other clients can:
- List Maxim's tools
- Call them
- Subscribe to prompts (the bio-enrichment context as an MCP resource)

**The bio-pipeline value:** the calling client gets tools; Maxim's bio-systems get learning signal from every tool call. A team using Claude Desktop with Maxim as an MCP server gets tools whose behavior is shaped by usage patterns over time.

**Authentication + sandboxing:** API key per client, scoped to the agent the client is connecting to. Audit logging per call. Reuse the existing leader/peer auth surface where possible.

### M3. Resource / prompt support (~150 LOC)

MCP supports `resources` (file-like content) and `prompts` (templated message generators). Map Maxim's:
- `resources` ← bio-enrichment sections (current memories, predictions, hippocampal recalls) as live MCP resources
- `prompts` ← Acting Coach + persona overlays as MCP prompts

This makes Maxim's introspection tools (`memory_recall`, `causal_links`, `examine`) accessible to MCP clients without bespoke API calls.

### M4. MCP-native tool authoring (~50 LOC)

Allow tool authors to write tools in pure MCP format (JSONSchema input/output, MCP-style errors) and have them work natively in Maxim. The CC9 dual-format support is the foundation; M4 is the documentation + examples + test fixtures + a migration tool for legacy description-as-value patterns that strict clients reject.

---

## Open questions

### Q1. Does the bio-pipeline learn from MCP-tool failures meaningfully?

A `filesystem.read_file` call that returns "file not found" is technically a failure but doesn't have the embodied consequence pattern Maxim's NAc is designed for. Should MCP-tool failures use a different reward signal (e.g., zero valence rather than negative), or should the existing `ToolOutput.side_effects["embodiment_failures"]` path apply?

Lean: small negative valence on tool failure (already the convention). Embodied attribution doesn't apply because there's no body sensor to write to. NAc still learns "filesystem.read_file with this path → failure" as causal information.

### Q2. Concurrency / lifecycle

MCP servers run as subprocesses. Lifecycle questions: when does Maxim spawn them? On `maxim run`? On first tool call? On demand? How are they shut down? What if one crashes mid-call?

Lean: spawn on first reference (lazy), keep alive for session duration, restart on crash with backoff.

### Q3. Tool name collisions

If `filesystem.read_file` and Maxim's local `read_file` both exist, which wins? Namespace by server name (`filesystem.read_file` vs `local.read_file`)? Per-call selection?

Lean: namespace by server name; default to local if unqualified. Document the precedence.

### Q4. Streaming MCP tools

MCP supports streaming tool results. Maxim's `Tool.execute()` returns a `ToolOutput` synchronously. Adapter needs to bridge streaming ↔ batch. Acceptable to buffer for v1 of MCP support; revisit if specific tools need streaming.

### Q5. MCP server mode + leader/peer mesh

How does MCP server mode interact with the existing leader/peer mesh? Is the MCP server per-agent or per-leader? Does each peer expose its own MCP server, or only the leader?

Lean: per-leader. Peers don't expose their tools externally; the leader is the cluster's MCP face.

### Q6. side_effects exposure

Tool side_effects (`embodiment_failures`, etc.) are Maxim-internal. Do they get surfaced over MCP, hidden, or routed to a separate channel? Strict MCP clients may fail on extra fields in tool results.

Lean: hidden by default. Add an opt-in flag (`?include_side_effects=true`) for clients that want the bio-pipeline visibility.

### Q7. Tool cancellation

CC11 (in [v1_refinement.md](v1_refinement.md) Section 7) adds `Tool.cancel()` to the Tool ABC. MCP supports notification-style cancellation. M1's `MCPToolBridge` should plumb cancellation both directions: when an MCP server's call is cancelled by the client, propagate to the Maxim tool's `cancel()`; when a Maxim agent's tool call is cancelled, propagate to the MCP server.

---

## Stages (TBD when full plan drafted)

### Stage P0 — `mcp` Python SDK integration (~50 LOC)

Add `mcp` as an optional extra. Wire into `pyproject.toml` extras + import-error handling.

### Stage P1 — MCP client mode (M1, ~300 LOC + tests)

Spawn external MCP servers, register their tools, route calls.

### Stage P2 — MCP server mode (M2, ~250 LOC + tests)

`maxim mcp serve` exposes Maxim tools over MCP stdio + HTTP.

### Stage P3 — Resources + prompts (M3, ~150 LOC)

Map bio-enrichment to MCP resources, Acting Coach to MCP prompts.

### Stage P4 — Migration + documentation (M4, ~50 LOC + docs)

Legacy description-as-value migration tool. Tool authoring docs.

---

## Risks and tradeoffs

**R1. MCP spec is evolving.** The 1.0 spec ships in early 2026. Field semantics, transport details, security model — still moving. Mitigation: track the upstream spec; don't ship MCP support until the spec is stable enough to commit to.

**R2. Subprocess lifecycle is gnarly cross-platform.** Spawning Node.js MCP servers from Python on Windows / Mac / Linux + handling crashes + log forwarding is more code than expected. Mitigation: start with in-process MCP servers (Python-native), add subprocess support after.

**R3. JSONSchema strict mode may break existing Maxim tools.** Some Maxim tools use the legacy description-as-value pattern that CC9 flagged. M1's strict ingestion would reject them. Mitigation: M4's migration tool fixes the legacy patterns; until M4 ships, Maxim tools exposed over MCP server mode (M2) use the lax-mode export.

**R4. Side-effects channel is Maxim-specific.** MCP doesn't have a "bio-pipeline signal" concept. Hiding it (Q6 lean) preserves compatibility but loses the differentiator for MCP clients who would want to know. Trade-off: visibility vs. compatibility — opt-in flag is the compromise.

**R5. Authentication sprawl.** Per-client API keys for MCP server mode + cluster keys for peer mesh + LLM provider keys = three auth surfaces. Risk of inconsistent enforcement. Mitigation: route MCP server auth through the existing leader auth surface; don't add a fourth.

---

## Why this is a stub

The 1.0 work was making the bridge possible (CC9 dual-format Tool schema). The 1.1 work is implementing both sides of the bridge. That's a meaningful design + testing effort that benefits from time, not haste — and it's outside the 1.0 freeze surface.

When MCP spec hits 1.0 stability and we have a concrete user need (Maxim user wants to use Claude Desktop with Maxim tools, or Maxim user wants to call filesystem-mcp from Maxim), this plan fleshes out into stage-by-stage implementation.

---

## Cross-references

- [v1_refinement.md](v1_refinement.md) CC9 — 1.0 prerequisite (dual-format Tool schema). Shipped PR #204.
- [v1_refinement.md](v1_refinement.md) CC11 — `Tool.cancel()` hook composes with MCP cancellation semantics.
- [v1_refinement.md](v1_refinement.md) Section 8 — 1.1 track index.
- [reactive_peer_mesh_roadmap.md](reactive_peer_mesh_roadmap.md) — sister concern: Maxim-to-Maxim communication. MCP is Maxim-to-other-agents.
- [minecraft_benchmark.md](minecraft_benchmark.md) — sister 1.1 work; Minecraft adapter could be exposed via MCP server mode for stream demos.
- Anthropic MCP spec: https://spec.modelcontextprotocol.io/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
