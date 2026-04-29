# MCP compatibility — server, client, and schema interop

**Status:** STUB (1.1+)
**Branch:** TBD
**Depends on:** CC9 (Tool schema dual-format support — SHIPPED in 1.0, see [v1_refinement.md](v1_refinement.md) Section 7)

---

## Motivation

Model Context Protocol ([spec.modelcontextprotocol.io](https://spec.modelcontextprotocol.io/)) is the emerging cross-vendor standard for tool servers, prompt servers, and resource servers. Maxim should both:

1. **Consume** MCP servers — let users plug third-party MCP tool servers into a Maxim agent.
2. **Expose** Maxim tools to MCP clients — let other harnesses (Claude Desktop, Cursor, custom agents) call Maxim-registered tools.

Both directions become tractable once `Tool.input_schema` accepts JSONSchema natively, which is what CC9 ships in 1.0.

## Prerequisites (already shipped in 1.0)

- **CC9: Dual-format `Tool.input_schema`** — accepts either the legacy custom format or JSONSchema 2020-12 (the wire format MCP uses). `Tool.to_json_schema()` is the canonical export.
- Format-sensitive consumers route through `to_json_schema()` rather than reading `input_schema` directly. The CLAUDE.md invariant pins this.

## 1.1 deliverables (TBD)

- **MCP client mode** — register an external MCP server URL via CLI / `pymaxim` API; its `tools/list` and `tools/call` plug into the Maxim tool registry alongside built-in tools.
- **MCP server mode** — `maxim serve --mcp` exposes the agent's registered tools to MCP clients over stdio (and optionally HTTP).
- **Resource + prompt support** — beyond tools, MCP also defines `resources` and `prompts`. Decide whether Maxim's percept layer / prompt assembler maps cleanly or needs adapters.
- **Authentication + sandboxing** — externally-supplied JSONSchema crossing the trust boundary needs a hardening pass. CC9's `_resolve_property_type` "unknown → dict" fallback is intentional today (in-process tool authors only); 1.1 should raise on unknown types when ingesting external schemas.

## Open questions

- How does MCP server mode interact with the existing leader/peer mesh? Is the MCP server per-agent or per-leader?
- Tool side_effects (`embodiment_failures`, etc.) are Maxim-internal — do they get surfaced over MCP, hidden, or routed to a separate channel?
- Strict MCP clients (Claude Desktop in strict mode) reject extra parameters and treat `required` strictly. The legacy description-as-value pattern (`{NAME: "Optional ..."}`) emits `required: [NAME]` — a CC9 docstring already flags this; 1.1 may need a migration tool.

## Why this is a stub

The 1.0 work was making the bridge possible (CC9). The 1.1 work is implementing both sides of the bridge. That's a meaningful design + testing effort that benefits from time, not haste — and it's outside the 1.0 freeze surface.