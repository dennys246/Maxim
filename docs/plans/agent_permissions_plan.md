# Agent Permissions Plan

> **Status:** Design complete, implementation ready.
> **Goal:** Per-agent authority levels, tool-scoped permissions, and SEM entity access control for multi-agent campaigns.
> **Estimated scope:** ~250 LOC core + ~100 LOC tests + ~50 LOC wiring
> **Depends on:** None. Foundation for Pecking Order Graph authority domain.
> **Blocks:** v1.0.0 publication (lightweight, high-impact feature).

---

## Why This Matters

Maxim simulates cognitive agents. Realistic scenarios require realistic authority dynamics — a king has more power than a soldier, a sysadmin has more access than an intern. Without permissions, every agent in a multi-agent campaign has identical capabilities, which undermines the simulation's value.

This is also the foundation for the Pecking Order Graph's authority domain. Building it now means POG has a concrete data model to build on.

---

## Data Model

### `AgentPermissions` (`src/maxim/agents/permissions.py`)

```python
@dataclass
class SEMAccessRule:
    """Access control for a specific SEM entity's sensors/modulators."""
    entity_pattern: str          # fnmatch glob: "server_room.*", "prod-db-*", "*"
    read_sensors: bool = True    # Can read sensor values
    actuate: bool = True         # Can trigger modulator affordances
    min_clearance: int = 0       # Entity requires this clearance level to interact

@dataclass
class AgentPermissions:
    """Per-agent permission bundle. Parsed from campaign YAML metadata."""

    # Authority (0-100). Higher = more power. Maps to POG authority score.
    authority: int = 50

    # Clearance level (0-5). Separate from authority — a warrior may be
    # powerful but lack technical clearance. Entities can require min clearance.
    clearance: int = 0

    # Tool access control
    tool_allow: frozenset[str] | None = None   # If set, ONLY these tools permitted
    tool_deny: frozenset[str] = frozenset()     # Always denied (overrides allow)

    # SEM entity access rules. Most-specific pattern wins (longest first).
    sem_rules: tuple[SEMAccessRule, ...] = ()

    # --- Methods ---

    def can_use_tool(self, tool_name: str) -> tuple[bool, str]:
        """Returns (allowed, reason). Reason is '' if allowed."""

    def can_read_entity(self, entity_path: str) -> bool:
        """Can this agent read sensors on the named entity?"""

    def can_actuate_entity(self, entity_path: str) -> bool:
        """Can this agent trigger modulators on the named entity?"""

    def can_command(self, target: "AgentPermissions") -> bool:
        """Can this agent issue commands to the target agent?"""
        return self.authority > target.authority

    def meets_clearance(self, required: int) -> bool:
        return self.clearance >= required

    @classmethod
    def from_dict(cls, data: dict) -> "AgentPermissions": ...

    @classmethod
    def permissive(cls) -> "AgentPermissions":
        """Default: full access. Used when no permissions specified."""
        return cls(authority=50, clearance=5)
```

### Design decisions

- **`authority` is 0-100 int**, not an enum. Supports fine-grained ordering (king=100, prince=80, knight=60, soldier=40). Maps directly to POG node authority score.
- **`clearance` is separate from `authority`** because a powerful warrior shouldn't automatically access a computer terminal.
- **`tool_deny` overrides `tool_allow`** — deny-list always wins for safety.
- **SEM rules use most-specific-match** (longest entity_pattern first), not first-match. This prevents the `"*": deny` footgun.
- **`from_dict` enables pure YAML configuration** — no code changes needed for new permission schemes.
- **`permissive()` default** ensures existing campaigns work without changes.

---

## Campaign YAML Schema

Permissions live in `metadata.permissions` on each NPC spec. No schema changes needed — NPC metadata is already a free-form dict.

### Fantasy example (Kings' Duel)

```yaml
npcs:
  english_king:
    entity_type: npc
    metadata:
      role: monarch
      permissions:
        authority: 100
        clearance: 5
        sem_rules:
          - entity_pattern: "*"
            read_sensors: true
            actuate: true

  english_prince:
    entity_type: npc
    metadata:
      role: heir
      permissions:
        authority: 80
        clearance: 4
        sem_rules:
          - entity_pattern: "english_*"
            actuate: true
          - entity_pattern: "french_*"
            actuate: false  # Can observe but not command French forces

  english_sergeant:
    entity_type: npc
    metadata:
      role: soldier
      permissions:
        authority: 40
        clearance: 2
        tool_deny: [command]  # Soldiers can't issue commands
```

### Cybersecurity example

```yaml
npcs:
  sysadmin:
    metadata:
      permissions:
        authority: 90
        clearance: 5
        sem_rules:
          - entity_pattern: "*"
            actuate: true

  intern:
    metadata:
      permissions:
        authority: 20
        clearance: 1
        tool_deny: [execute_command, write_file, delete_file]
        sem_rules:
          - entity_pattern: "staging_*"
            actuate: true
          - entity_pattern: "prod_*"
            read_sensors: true
            actuate: false  # Can see prod metrics, can't touch

  attacker:
    metadata:
      permissions:
        authority: 0
        clearance: 0
        sem_rules:
          - entity_pattern: "*"
            read_sensors: false
            actuate: false

world_objects:
  prod_database:
    entity_type: server
    metadata:
      min_clearance: 4
    sensors:
      cpu_load: {unit: percent, range: [0, 100], initial: 87}
```

---

## Wiring Points

### 1. Tool execution gating (`runtime/executor.py`)

Permissions flow per-call through an execution context, NOT stored on the Executor (which is shared across agents in a pool).

```python
def execute(self, action: dict, *, permissions: AgentPermissions | None = None) -> ToolOutput:
    tool_name = action.get("tool_name", "")
    if permissions is not None:
        allowed, reason = permissions.can_use_tool(tool_name)
        if not allowed:
            return ToolOutput(success=False, error=f"Permission denied: {reason}")
    # ... existing execution logic
```

### 2. Autonomy controller (`agents/autonomy.py`)

Add an optional permissions reference. The `can_execute_action` check runs permissions BEFORE safety constraints:

```python
def can_execute_action(self, action, confidence=None, *, permissions=None):
    # ... existing ALWAYS_ALLOWED_TOOLS check ...
    if permissions is not None:
        allowed, reason = permissions.can_use_tool(tool_name)
        if not allowed:
            return False, reason
    # ... existing safety/level checks ...
```

### 3. Agent factory (`runtime/agent_factory.py`)

Parse permissions from config metadata during agent creation:

```python
# In create_npc_agent() or similar:
perm_data = config.metadata.get("permissions")
permissions = AgentPermissions.from_dict(perm_data) if perm_data else AgentPermissions.permissive()
agent_instance.permissions = permissions
```

### 4. Campaign runner (`simulation/campaign_runner.py`)

When `run_dm_campaign()` instantiates SEM entities, also parse permissions and attach to agents.

### 5. DM runtime — authority transfer (`simulation/dm_runtime.py`)

Add `authority_transfer` to `on_choice` effects:

```yaml
on_choice:
  acknowledge_death:
    flags: [king_dead]
    authority_transfer:
      from: english_king
      to: english_prince
```

The runtime calls `target.permissions.authority = source.permissions.authority` and zeros the source. Emits a percept so other agents observe the transfer through their bio-stacks.

### 6. SEM tool gating

Auto-generated SEM tools (sense/actuate) check permissions:

```python
def execute(self, *, permissions: AgentPermissions | None = None, **kwargs):
    if permissions and not permissions.can_read_entity(self.entity_name):
        return ToolOutput(success=False, error=f"Access denied: cannot read {self.entity_name}")
```

### 7. Permission denial as percept

When a tool is denied, the denial flows back through the agent loop as a normal tool result (success=False). The agent's Hippocampus captures it, NAc learns "attempting X → denied", and future planning avoids denied actions. No special wiring needed — the existing bio-system integration handles this naturally.

---

## Dynamic Authority Transfer

Authority changes at runtime via direct mutation:

```python
def transfer_authority(pool, from_id, to_id, reason=""):
    source = pool.get_agent(from_id)
    target = pool.get_agent(to_id)
    target.permissions.authority = source.permissions.authority
    source.permissions.authority = 0
    # Emit as percept so agents observe the change
    pool.broadcast_percept(
        f"Authority transferred from {from_id} to {to_id}: {reason}",
        salience=1.0, novelty=1.0,
    )
```

Triggered by:
- `authority_transfer` in `on_choice` effects (campaign YAML)
- Entity HP reaching 0 (DM runtime death hook)
- Direct API call from orchestrator

---

## Mapping to Pecking Order Graph

| Permission System | POG Equivalent |
|---|---|
| `authority` (0-100) | `PeckingNode` authority score → pecking direction on AUTHORITY edges |
| `clearance` (0-5) | Node metadata, used for EMBODIMENT domain gating |
| `can_command(target)` | `graph.find_pecked(node_id, AUTHORITY)` |
| `tool_allow/deny` | Capability metadata on PeckingNode |
| `sem_rules` | `PeckingDomain.EMBODIMENT` — who can interact with what |
| `transfer_authority()` | `graph.recompute_pecking()` triggered on authority change |

When POG ships, `AgentPermissions` becomes the local view. The graph provides the relational view. The transition is additive — POG reads permissions, doesn't replace them.

---

## Implementation Sequence

| Step | What | LOC | Files |
|---|---|---|---|
| 1 | `AgentPermissions` + `SEMAccessRule` + tests | ~180 | `agents/permissions.py`, `tests/unit/test_permissions.py` |
| 2 | Executor per-call permissions check | ~10 | `runtime/executor.py` |
| 3 | AutonomyController permissions check | ~5 | `agents/autonomy.py` |
| 4 | AgentFactory permissions parsing | ~15 | `runtime/agent_factory.py` |
| 5 | Campaign runner wiring | ~15 | `simulation/campaign_runner.py` |
| 6 | DM runtime authority_transfer | ~20 | `simulation/dm_runtime.py` |
| 7 | SEM tool gating | ~20 | Where SEM tools are generated |
| 8 | Integration tests | ~60 | `tests/unit/test_permissions.py` |

Total: ~325 LOC. Can be done in one focused session.

---

## Open Questions

1. **Should denied actions count as "actions taken" for the consecutive-tool-cap?** Probably yes — prevents infinite retry loops on denied tools.

2. **Should authority transfer be reversible?** (e.g., the prince gives authority back if the king is healed). The current model supports it trivially — just mutate the int back. But should it be logged/tracked?

3. **Should the LLM prompt include permission context?** If the agent's LLM knows "you can't use execute_command", it won't waste a turn trying. This could be a few lines in the system prompt: "Your permissions: authority=20, tools=[speak, read_file, examine]. You do NOT have access to: execute_command, write_file."

4. **Multi-agent approval chains.** When Agent B (authority=40) proposes an action and Agent A (authority=80) could approve it — how does the approval flow? For v1.0, auto-approve based on authority. For POG, route through the graph.
