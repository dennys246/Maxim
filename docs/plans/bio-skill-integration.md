# Bio-Skill Integration Plan (ATL Phase A7)

> **Status:** Not started. Depends on Skills & Protocols system and
> ATL phases A2-A5b (all implemented).

---

## Goal

Enable skills to be defined as markdown documents with bio system
references, executed as structured step sequences, and remembered as
concept memories that improve future skill selection.

---

## Prerequisites

- ConceptExtractor (A2): **done**
- ConceptGrounder (A3): **done**
- ConceptContextBuilder (A4): **done**
- PatternCompleter (A5b): **done**
- Skills & Protocols system: **done**

---

## Implementation Phases

### A7.1 Markdown Skill Definitions (~80 lines)

**New file:** `src/maxim/skills/definition_parser.py`
**New directory:** `src/maxim/skills/definitions/` (markdown skill files)

Skills defined as markdown with YAML frontmatter:

```markdown
---
name: slope_observation
type: bio_skill
bio: [hippocampus, atl, ips]
triggers: ["observe the slope", "watch the run"]
workspace_bounds:
  yaw: 30.0
  pitch: 20.0
---

## Steps

1. **recall_concepts** -- Query ATL for concepts related to current task.
2. **statistical_baseline** -- Ask IPS to compute motion variance.
3. **episodic_check** -- Recall hippocampus episodes with similar stats.
4. **activate_skill** -- Start RTSPStreamingSkill with workspace bounds.
```

Parser extracts `SkillDefinition` dataclass with name, bio_deps, triggers,
workspace_bounds, and structured `SkillStep` list.

### A7.2 BioSkill Executable Wrapper (~150 lines)

**New file:** `src/maxim/skills/bio_skill.py`

Wraps `SkillDefinition` into a `Skill` subclass. Each step maps to a
bio system operation:

| Step Name | Bio System Call |
|---|---|
| `recall_concepts` | `atl.recall_similar(query, top_k)` |
| `episodic_check` | `hippocampus.recall_associated(query)` |
| `statistical_baseline` | `ips.compute(data)` |
| `concept_ground` | `ConceptGrounder.ground(concept)` |
| `pattern_complete` | `PatternCompleter.complete(concept)` |
| `activate_skill` | Delegate to named Skill class |

Results from each step written to shared protocol context dict.
Raw markdown definition included in `context_for_llm()` for LLM understanding.

### A7.3 Skills as Memories (~40 lines)

**Extends:** `src/maxim/memory/concept_extractor.py`

When a BioSkill executes, create/update a concept for the skill itself:

```python
# In ConceptExtractor._process_item(), after existing extraction:
skill_name = memory.metadata.get("skill_name")
if skill_name:
    concept = self._atl.find_or_create(f"skill:{skill_name}", category="skill_execution")
    concept.add_ref("hippocampus", memory.id)
    # Link skill concept to co-occurring concepts
    for other in extracted_concepts:
        self._atl.define_relationship(concept.id, other.id, "EXECUTES_WITH", weight=0.5)
```

Enables learning: "slope_observation + high brightness --> better results
with tighter pitch constraints" via AG-grounded stats on skill concepts.

### A7.4 Concept-Driven Skill Prediction (~40 lines)

**Extends:** `src/maxim/memory/pattern_completer.py`

PatternCompleter finds past skill executions related to current context:

```python
skill_concepts = [c for c in matching_concepts if c.category == "skill_execution"]
for sc in skill_concepts:
    episodes = self._recall_by_ids("hippocampus", sc.memory_refs.get("hippocampus", {}))
    for ep in episodes:
        outcome = ep.metadata.get("skill_outcome")
        if outcome:
            predictions.append(PredictedOutcome(...))
```

### A7.5 Concept-Driven Skill Discovery (~30 lines)

**Extends:** `src/maxim/memory/concept_context.py`

ConceptContextBuilder includes relevant skill definitions in LLM context
based on concept matching:

```python
for defn in skill_definitions:
    relevance = sum(1 for trigger in defn.triggers
                    if any(c.name in trigger for c in matching_concepts))
    if relevance > 0:
        context.available_skills.append({
            "name": defn.name,
            "relevance": relevance,
            "past_success_rate": _get_skill_success_rate(defn.name),
        })
```

Gives the LLM SayCan-style affordance info: skills ranked by concept
relevance and past success rate.

---

## Implementation Order

| Step | What | Effort |
|---|---|---|
| A7.1 | SkillDefinition parser + directory loader | Small (~80 lines) |
| A7.2 | BioSkill executable wrapper | Medium (~150 lines) |
| A7.3 | ConceptExtractor skill-memory integration | Small (~40 lines) |
| A7.4 | PatternCompleter skill-prediction | Small (~40 lines) |
| A7.5 | ConceptContextBuilder skill-discovery | Small (~30 lines) |

**Total:** ~340 lines production + ~300 lines tests.
**Order:** A7.1 --> A7.2 --> A7.3 --> A7.4 --> A7.5

---

## Tests Needed

- Parser extracts SkillDefinition from markdown with YAML frontmatter
- BioSkill resolves bio dependencies and executes step sequence
- BioSkill step failure aborts with error result
- ConceptExtractor creates skill_execution concepts on BioSkill episodes
- PatternCompleter predicts outcomes from past skill executions
- ConceptContextBuilder ranks available skills by concept relevance
