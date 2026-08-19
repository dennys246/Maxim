# AGENTS.md — provider-neutral entrypoint

> The canonical instruction corpus for this repository is [CLAUDE.md](CLAUDE.md).
> This file exists only for tools that auto-load `AGENTS.md`.

Before doing any work in this repository, read `CLAUDE.md` in full and follow its
required checks, safety rules, routing table, and subsystem-reading instructions.

Do not add project rules, commands, routing entries, or subsystem knowledge here.
Put cross-cutting guidance in `CLAUDE.md` and scoped guidance in `docs/agents/`.
CI enforces this pointer-only adapter byte-for-byte to prevent instruction drift.
