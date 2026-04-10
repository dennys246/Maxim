# Agent Permissions Plan

> **Status:** Design complete, implementation ready.
> **Goal:** Two-layer permission system — enforced (hard gates) + perceived (bio-stack-driven social norms) — for realistic multi-agent authority dynamics.
> **Estimated scope:** ~300 LOC core + ~120 LOC tests + ~50 LOC wiring
> **Depends on:** None. Foundation for Pecking Order Graph authority domain.
> **Blocks:** v1.0.0 publication (lightweight, high-impact feature).

---

## Core Insight: Enforced vs. Perceived Permissions

In the real world, permission operates on two levels:

1. **Enforced** — physical gates. The door is locked. The terminal requires a key. The modulator rejects the call. You **cannot** do this.
2. **Perceived** — social norms. You *could* sit on the king's throne, but you *don't* because you predict punishment. Permission is a learned expectation, not a physical barrier.

Maxim's bio-stack already models perceived permissions naturally:
- **NAc** learns "attempting X without authority → negative outcome" through causal observation
- **FearAgent** reviews actions against predicted harm
- **LLM system prompt** carries role context ("you are a soldier, you follow orders")
- **Hippocampus** remembers past denials and their consequences

What we need to BUILD is only the enforced layer. The perceived layer is the bio-stack doing its job — we just need to feed it the right context.

---

## Data Model

### `AgentPermissions` (`src/maxim/agents/permissions.py`)

This is the **enforced** layer only. Hard gates that the system physically blocks.

```python
@dataclass
class SEMAccessRule:
    """Enforced access control for a specific SEM entity."""
    entity_pattern: str          # fnmatch glob: "server_room.*", "prod-db-*", "*"
    read_sensors: bool = True    # Can read sensor values
    actuate: bool = True         # Can trigger modulator affordances
    min_clearance: int = 0       # Entity requires this clearance level

@dataclass
class AgentPermissions:
    """Enforced per-agent permissions. Hard gates — the system blocks the action."""

    # Clearance level (0-5). Physical access control.
    # A locked terminal requires clearance 4 regardless of authority.
    clearance: int = 0

    # Tool access control (hard gates)
    tool_allow: frozenset[str] | None = None   # If set, ONLY these tools permitted
    tool_deny: frozenset[str] = frozenset()     # Always denied (overrides allow)

    # SEM entity access rules. Most-specific pattern wins (longest first).
    sem_rules: tuple[SEMAccessRule, ...] = ()

    def can_use_tool(self, tool_name: str) -> tuple[bool, str]:
        """Hard gate: can this agent use this tool?"""

    def can_read_entity(self, entity_path: str) -> bool:
        """Hard gate: can this agent read sensors on this entity?"""

    def can_actuate_entity(self, entity_path: str) -> bool:
        """Hard gate: can this agent trigger modulators on this entity?"""

    def meets_clearance(self, required: int) -> bool:
        """Hard gate: does agent have sufficient clearance?"""
        return self.clearance >= required

    @classmethod
    def from_dict(cls, data: dict) -> "AgentPermissions": ...

    @classmethod
    def permissive(cls) -> "AgentPermissions":
        """Default: no enforced restrictions."""
        return cls(clearance=5)
```

### `PerceivedAuthority` — campaign YAML metadata (no code needed)

This is the **perceived** layer. It flows into the LLM system prompt and lets the bio-stack learn naturally.

```python
@dataclass
class PerceivedAuthority:
    """Social context that shapes agent behavior through learning, not hard gates."""

    # Authority score (0-100). Injected into LLM prompt as social context.
    # The agent *could* disobey — but NAc will learn consequences.
    authority: int = 50

    # Role expectations. Injected into LLM system prompt.
    # "Follow orders from knights and above. Do not issue commands."
    role_expectations: str = ""

    # Social hierarchy. Other agents the LLM knows to defer to or command.
    defers_to: list[str] = field(default_factory=list)    # ["english_king", "english_champion"]
    commands: list[str] = field(default_factory=list)      # ["militia_1", "militia_2"]

    @classmethod
    def from_dict(cls, data: dict) -> "PerceivedAuthority": ...
```

### Key distinction

| Aspect | Enforced (`AgentPermissions`) | Perceived (`PerceivedAuthority`) |
|---|---|---|
| Mechanism | Tool executor rejects call | NAc learns, FearAgent reviews, LLM avoids |
| Violation | Impossible — blocked by code | Possible — agent chooses not to (or learns not to) |
| Feedback | `ToolOutput(success=False, error="Access denied")` | Negative outcome → NAc RPE → future avoidance |
| Example | Locked door, missing SSH key | Social rank, role expectations |
| Where defined | `metadata.permissions.enforced` | `metadata.permissions.perceived` |
| Can be subverted? | No | Yes — that's the point (agent can rebel, face consequences) |

---

## Campaign YAML Schema

Both layers live in `metadata.permissions`:

### Fantasy example (Kings' Duel)

```yaml
npcs:
  english_king:
    entity_type: npc
    metadata:
      role: monarch
      persona_prompt: "King Edmund III. Proud, battle-hardened."
      permissions:
        enforced:
          clearance: 5
          sem_rules:
            - entity_pattern: "*"
              read_sensors: true
              actuate: true
        perceived:
          authority: 100
          role_expectations: "You are the king. You command all English forces."
          commands: [english_prince, english_champion, english_sergeant]

  english_sergeant:
    entity_type: npc
    metadata:
      role: soldier
      persona_prompt: "Sergeant Thomas. Veteran of three campaigns."
      permissions:
        enforced:
          clearance: 2
          tool_deny: [admin_console]  # Can't use command-level tools
          sem_rules:
            - entity_pattern: "royal_vault"
              actuate: false  # Vault is physically locked to non-royals
        perceived:
          authority: 40
          role_expectations: "Follow orders from knights and above. Protect the prince. Do not issue commands to nobility."
          defers_to: [english_king, english_champion]
```

### Cybersecurity example

```yaml
npcs:
  sysadmin:
    metadata:
      permissions:
        enforced:
          clearance: 5
          sem_rules:
            - entity_pattern: "*"
              actuate: true
        perceived:
          authority: 90
          role_expectations: "You are the security lead. Full system access. Responsible for incident response."

  intern:
    metadata:
      permissions:
        enforced:
          clearance: 1
          tool_deny: [execute_command, write_file, delete_file]
          sem_rules:
            - entity_pattern: "staging_*"
              actuate: true
            - entity_pattern: "prod_*"
              read_sensors: true
              actuate: false  # ENFORCED: can see prod metrics, physically can't touch
        perceived:
          authority: 20
          role_expectations: "You are a junior engineer. Ask the sysadmin before touching production. You should not run commands you don't understand."
          defers_to: [sysadmin]

  attacker:
    metadata:
      permissions:
        enforced:
          clearance: 0
          sem_rules:
            - entity_pattern: "*"
              read_sensors: false
              actuate: false
        perceived:
          authority: 0
          role_expectations: "You are outside the system. You must find vulnerabilities to gain access."

world_objects:
  prod_database:
    entity_type: server
    metadata:
      min_clearance: 4  # Enforced: only clearance >= 4 can interact
    sensors:
      cpu_load: {unit: percent, range: [0, 100], initial: 87}
```

### What makes this powerful

The **intern** example shows both layers working together:
- **Enforced**: Can't `execute_command` (tool_deny), can't actuate prod modulators (SEM rule). These are hard-blocked.
- **Perceived**: "Ask the sysadmin before touching production." The intern *could* try `read_file` on a prod config (not in tool_deny), but NAc will learn that doing so without asking leads to negative outcomes (sysadmin gets angry, access revoked in-story).

The **attacker** example shows pure enforcement — no perceived layer because the attacker doesn't respect social norms. They're hard-blocked by clearance and SEM rules.

---

## How Perceived Permissions Wire into Existing Systems

### LLM System Prompt (already exists — extend it)

The agent loop builds a system prompt for each LLM call. `PerceivedAuthority` data gets injected:

```
Your role: soldier (authority: 40/100)
You defer to: english_king, english_champion
Social expectations: Follow orders from knights and above. Do not issue commands to nobility.
```

This goes into the `ModeInfo` or `StructuredContext` that the LLM worker receives. The LLM then naturally avoids actions that violate these expectations — but it *can* choose to violate them, which creates interesting narrative.

### NAc Causal Learning (already exists — just observe)

When the agent violates perceived expectations and gets a negative response:
1. Agent (soldier) tries to give an order → NPC ignores the order
2. Outcome: failure → NAc observes `event: "command_npc", outcome: "ignored", valence: NEGATIVE`
3. Future: NAc predicts `command_npc → negative` → agent avoids commanding

No new code needed. The DM runtime's NPC response just needs to react appropriately to authority violations (already handled by the NPC's own persona_prompt).

### FearAgent (already exists — just feed context)

FearAgent reviews actions before execution. If the agent's perceived authority context is in the review prompt, FearAgent can flag "this action exceeds your perceived authority" as a risk. This is a ~5 line change to include perceived authority in the FearAgent's review context.

---

## Dynamic Authority Transfer

Authority changes affect BOTH layers:

```python
def transfer_authority(pool, from_id, to_id, reason=""):
    source = pool.get_agent(from_id)
    target = pool.get_agent(to_id)

    # Transfer enforced clearance (physical keys/access)
    if source.permissions and target.permissions:
        target.permissions.clearance = max(target.permissions.clearance, source.permissions.clearance)

    # Transfer perceived authority
    if source.perceived and target.perceived:
        target.perceived.authority = source.perceived.authority
        source.perceived.authority = 0

    # Broadcast so all agents observe the change through bio-stack
    pool.broadcast_percept(
        f"Authority transferred from {from_id} to {to_id}: {reason}",
        salience=1.0, novelty=1.0,
    )
```

Campaign YAML trigger:

```yaml
on_choice:
  acknowledge_death:
    flags: [king_dead]
    authority_transfer:
      from: english_king
      to: english_prince
      reason: "succession"
```

---

## Mapping to Pecking Order Graph

| Permission Layer | POG Mapping |
|---|---|
| `PerceivedAuthority.authority` (0-100) | `PeckingNode` authority score → pecking direction on AUTHORITY edges |
| `AgentPermissions.clearance` (0-5) | `PeckingDomain.EMBODIMENT` gating |
| `PerceivedAuthority.defers_to` | Graph edge: pecked_by in AUTHORITY domain |
| `PerceivedAuthority.commands` | Graph edge: pecks in AUTHORITY domain |
| `AgentPermissions.sem_rules` | `PeckingDomain.EMBODIMENT` — who can interact with what |
| `AgentPermissions.tool_allow/deny` | Capability metadata on PeckingNode |
| `transfer_authority()` | `PeckingGraph.recompute_pecking()` |

When POG ships:
- Phase 1: `AgentPermissions` is standalone (local enforcement)
- Phase 2: `PeckingGraph` is built from `PerceivedAuthority` scores (relational view)
- Phase 3: Graph position can dynamically adjust perceived authority

---

## Wiring Points

| Step | What | LOC | Files |
|---|---|---|---|
| 1 | `AgentPermissions` + `SEMAccessRule` + `PerceivedAuthority` + `from_dict` | ~150 | `agents/permissions.py` |
| 2 | Executor per-call permissions check | ~10 | `runtime/executor.py` |
| 3 | AutonomyController permissions check | ~5 | `agents/autonomy.py` |
| 4 | AgentFactory permissions parsing | ~15 | `runtime/agent_factory.py` |
| 5 | Perceived authority → LLM prompt injection | ~20 | `agents/llm_worker.py` or prompt builder |
| 6 | Campaign runner wiring | ~15 | `simulation/campaign_runner.py` |
| 7 | DM runtime `authority_transfer` | ~20 | `simulation/dm_runtime.py` |
| 8 | SEM tool gating | ~20 | SEM tool generation |
| 9 | FearAgent perceived authority context | ~5 | `agents/fear_agent.py` |
| 10 | Tests | ~120 | `tests/unit/test_permissions.py` |

Total: ~380 LOC.

---

## Open Questions

1. **Should the agent be told about enforced vs perceived?** If the LLM prompt says "You CANNOT use execute_command (enforced)" vs "You SHOULDN'T issue commands (social expectation)", the LLM can make smarter decisions. Recommend: yes, include both in context.

2. **Can perceived authority be negative?** A traitor or outcast might have negative social standing. The system supports authority=0 (no social power) but negative could model active hostility from others. Defer to POG — graph edges handle this better than a signed integer.

3. **Should NAc observation of permission denials be special-cased?** Currently denials flow through the normal tool result path. Should denials have higher salience so the agent learns faster? Probably yes — a 1-line salience boost in the capture path.

4. **Rebellion mechanic.** An agent with high NAc confidence in a positive outcome could choose to violate perceived authority. "I know the king said don't go into the vault, but I predict finding the cure will save him." This already works — NAc's prediction overrides the persona_prompt expectation. No code needed, but campaigns should test for it.

5. **SEM entities gaining/losing clearance requirements at runtime.** The hacker gains access to the server (reduces its min_clearance). This is just entity metadata mutation — `entity.metadata["min_clearance"] = 0`. Should be triggerable from campaign `on_choice` effects.
