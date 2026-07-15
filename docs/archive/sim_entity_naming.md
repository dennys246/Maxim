# Simulation Entity Naming Plan

> **Status:** Not started. Small, optional log-readability enrichment.
>
> **Summary:** Give each `MaximAgent` instance (AUT, orchestrator) a display name, and thread it through the sim logger prefix so every line shows `{timestamp} [{entity_name}][{subsystem}] {message}`. Makes multi-agent sims readable. Scoped to AUT/orchestrator only — NPC display names are a DM runtime concern, not entity identity.

## Motivation

Today's sim logger output:

```
439.9s [Hippocampus] captured memory id=m_0421
441.2s [NAc] blocked action: delete_file (score=-0.8)
```

When a sim has only one AUT, fine. As soon as we have orchestrator + AUT both emitting bio-subsystem logs, or (future) multi-AUT party sims, it becomes impossible to tell whose Hippocampus fired. Prefixing entity names fixes this:

```
439.9s [Derek the Great][Hippocampus] captured memory id=m_0421
441.2s [Derek the Great][NAc] blocked action: delete_file (score=-0.8)
441.9s [Orchestrator][Analysis] AUT completed encounter in 3 turns
```

**Not in scope:** NPC display names. NPCs in DM campaigns are data in a registry, not `MaximAgent` instances with bio-subsystems. Their names render inside DM-composed stimulus text via the DM runtime — no logger integration needed.

## Design

**Entity identity attached at construction:**

```python
@dataclass
class EntityIdentity:
    name: str          # "Derek the Great", "Orchestrator"
    role: str          # "aut" | "orchestrator"
```

Attached to each `MaximAgent` instance at construction. Bio-subsystems log through a logger that reads the attached identity from the agent they belong to.

**Name sources:**
- AUT: CLI flag `--aut-name "Derek the Great"`, defaults to `"AUT"`
- Orchestrator: defaults to `"Orchestrator"`, or persona name (`"DungeonMaster"`) if set

**Log formatter extension:** prefix becomes `[{entity_name}][{subsystem}]`. Truncate entity name at 20 chars with ellipsis in rendered output (full name stays in structured log fields).

**Fallback:** entities with no identity attached fall back to class + short id (`[MaximAgent#a3f2][Hippocampus]`) so existing tests and un-migrated callsites keep working.

## Implementation (~120 LOC, single phase)

**New files:**
- `src/maxim/simulation/entity_identity.py` (~40) — `EntityIdentity` dataclass + identity-aware logger helper

**Modified:**
- Sim log formatter (wherever bio-subsystem log prefix is built) — extended to `[entity][subsystem]`
- `src/maxim/conscience/selfy.py` (`MaximAgent.__init__`) — accept `entity_identity` kwarg, store on instance
- `src/maxim/simulation/orchestrator.py` — construct orchestrator with identity; pass `--aut-name` through to AUT construction
- Bio-subsystem log callsites (`Hippocampus`, `NAc`, `SCN`, `ATL`, `AngularGyrus`, others) — use identity-aware logger rather than direct prefix
- CLI arg parser — add `--aut-name` flag

**New tests:**
- `tests/unit/test_entity_naming.py` (~50) — prefix formatting, truncation, fallback behavior

## Design Decisions

1. **Name is session-scoped, display-only** — no global registry, no identity key. Session UUIDs remain the real identity.
2. **Truncation at 20 chars** — keeps logs aligned; full name in structured fields
3. **Backward compatible** — no identity = fallback prefix, existing tests unaffected
4. **NPCs out of scope** — they're not MaximAgents; their names live in DM runtime stimulus text

## Risks

1. **Wide refactor across bio-subsystem log callsites** — mechanical but touches many files. Mitigation: centralize through one logger utility; convert callsites incrementally, leaning on the fallback prefix during migration.
2. **Test output churn** — tests that grep log lines may break on format change. Mitigation: run the full test suite and update expected prefixes in one pass.

## Ties to Other Plans

| Plan | Relationship |
|------|-------------|
| [DM MVP](dungeon_master_persona.md) | Not required — MVP has one AUT, existing format is fine |
| [DM Extensions](../plans/deferred/dungeon_master_extensions.md) | Soft win once DM campaigns produce heavy multi-entity log output |
| **Agent Mesh** (blocked) | Future consumer — multi-AUT mesh sims unreadable without this |
| **Realtime Refinement** (core done) | Optional consumer — refinement reports clearer with named AUT |

## When to Implement

**Optional.** Ship whenever log readability becomes a pain point. No plan hard-depends on this.

Likely trigger: first multi-entity sim (DM campaign with heavy NPC dialogue, or agent-mesh prototype).
