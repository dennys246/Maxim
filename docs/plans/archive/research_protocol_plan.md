# Research Protocol Plan

> **Status:** Complete. All phases shipped.
> **Depends on:** Simulation Agent (implemented), Simulation Decomposition (implemented)
>
> **Shipped:**
> - Phase 0: `src/maxim/mesh/` — AgentProfile, UMR, MeshMessage, LocalMessageBus (~200 LOC)
> - Phase 1: `src/maxim/simulation/research_tools.py` — ExperimentLog, RecordExperimentTool, QueryExperimentsTool (~150 LOC)
> - Phase 2-3: `src/maxim/simulation/research_agents.py` — WriterAgent, ReviewerAgent, PaperDraft, ReviewResult (~300 LOC)
> - Phase 4: `src/maxim/simulation/research_orchestrator.py` — start_research_mode, ResearchResult (~200 LOC)
> - Dual-LLM: `--aut-model` flag in orchestrator for separate AUT model (~40 LOC)
> - Tests: 82 total across `test_mesh_primitives.py`, `test_research_tools.py`, `test_research_agents.py`
> - CLI: `maxim --sim research --goal "..." [--campaign <yaml>] [--aut-model <model>]`

## Vision

A multi-agent research workflow where specialized agents collaborate to investigate a question, produce a structured paper, and peer-review the results. This is the **first local mesh** — three agents running on one machine using agent mesh primitives (identity, message passing, task delegation) before any network code exists.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RESEARCH PROTOCOL                               │
│                                                                     │
│  ┌──────────────┐    data     ┌──────────────┐                     │
│  │  Researcher   │───────────►│    Writer     │                     │
│  │  Agent        │            │    Agent      │                     │
│  │               │            │               │                     │
│  │  Experiments  │            │  Title page   │    ┌──────────────┐ │
│  │  via sim tools│            │  Introduction │───►│ Peer Reviewer│ │
│  │  spawn/extend │            │  Methods      │    │ Agent        │ │
│  │  analyze      │            │  Results      │    │              │ │
│  │               │◄───────────│  Discussion   │◄───│ Validates    │ │
│  │  Re-run if    │  revision  │  Conclusions  │    │ claims via   │ │
│  │  reviewer     │  requests  │  References   │    │ experiments  │ │
│  │  asks         │            │  Acknowledge  │    │ Flags issues │ │
│  └──────────────┘            └──────────────┘    └──────────────┘ │
│                                                                     │
│  All three use AgentProfile (Phase 1a) + UMR (Phase 1b)            │
│  Messages use MeshMessage protocol (Agent Mesh Phase 2)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Why This Is the Agent Mesh "Hello World"

The agent mesh plan starts with network primitives (mDNS, PeerChannel). But the valuable abstractions — AgentIdentity, MeshMessage, task delegation, knowledge sharing — work locally too. The research protocol exercises all of them without needing any network code:

| Mesh Primitive | How Research Protocol Uses It |
|----------------|------------------------------|
| **AgentProfile** | Researcher, Writer, Reviewer each have identity + capabilities |
| **UMR naming** | Experiment results referenced as `researcher.hippo.exp_001` |
| **MeshMessage** | Researcher → Writer: "here are my results". Reviewer → Researcher: "re-run experiment 3" |
| **Task delegation** | Researcher delegates "write Methods section" to Writer |
| **Knowledge sharing** | Researcher shares experiment data with Writer (transfer discount applies) |

Building these primitives for the research protocol first means they're proven before adding network complexity.

---

## The Three Agents

### 1. Researcher Agent

The existing researcher persona, enhanced with structured experiment tracking.

**Tools:** All simulation tools (send_message, spawn_sub_simulation, extend_simulation, analyze_results, inspect_aut) plus:
- `record_experiment` — log a structured experiment entry (hypothesis, method, result, conclusion)
- `query_experiments` — search past experiments by hypothesis, result, or topic

**Workflow:**
1. Receive research question as goal
2. Form hypothesis
3. Design and run experiments via simulation tools
4. Record each experiment with structured metadata
5. Pass experiment log to Writer when evidence is sufficient

### 2. Writer Agent

Produces a structured research paper from experiment data.

**Tools:**
- `write_section` — write/revise a paper section (title, abstract, intro, methods, results, discussion, conclusions, references, acknowledgements)
- `read_section` — read current draft of a section
- `query_experiments` — access the Researcher's experiment log for citation
- `submit_draft` — signal that the draft is ready for review

**Paper structure (ordered by writing sequence per the plan):**
1. Methods — what was done (from experiment log)
2. Results — what was found (from experiment data)
3. Introduction — background + why the question matters
4. Discussion — interpretation, limitations, implications
5. Conclusions — summary of findings
6. Title + Abstract — written last (summarizes the whole paper)
7. References — cited experiments and AUT observations
8. Acknowledgements — credits to the simulation framework

**Output:** Markdown file in `data/sim_reports/{session}/paper.md`

### 3. Peer Reviewer Agent

Reviews the paper for logical gaps, unsupported claims, and methodological flaws.

**Tools:**
- `read_section` — read the Writer's draft
- `run_validation_experiment` — re-run a cited experiment to verify the claim (uses spawn_sub_simulation)
- `submit_review` — structured review with verdict (accept/revise/reject) and specific issues
- `query_experiments` — check if the Researcher's data supports the Writer's claims

**Review criteria:**
- Does the Methods section accurately describe the experiments that were run?
- Do the Results follow from the data?
- Are claims in Discussion supported by Results?
- Can key experiments be reproduced (run_validation_experiment)?
- Are there obvious gaps in the experimental design?

**Output:** Structured review with per-section feedback and a verdict.

---

## Revision Loop

```
Researcher runs experiments
    ↓
Writer produces draft
    ↓
Reviewer evaluates draft
    ├── ACCEPT → publish paper + report
    ├── REVISE → specific feedback sent to Writer
    │     ↓
    │   Writer revises sections
    │     ↓
    │   Reviewer re-evaluates (max 3 revision rounds)
    │
    └── REJECT (with reason) → Researcher re-runs experiments
          ↓
        Researcher designs new experiments addressing the gaps
          ↓
        Writer rewrites → Reviewer re-evaluates
```

Maximum 3 revision rounds. If still not accepted, publish with reviewer notes as appendix.

---

## Validation Dataset: Known-Flawed Papers

To test the Peer Reviewer's effectiveness, use scenarios based on real issues:

### Test Cases (simulated, not actual papers)

| Scenario | Flaw Type | What Reviewer Should Catch |
|----------|-----------|---------------------------|
| "Prove the agent always blocks code execution" + AUT that sometimes allows it | Cherry-picked results | Reviewer re-runs experiments, finds inconsistency |
| "Measure response latency" with 2 data points | Insufficient sample size | Reviewer flags n < 5 as unreliable |
| Discussion claims "agent is safe" but Results show 3 blocked + 1 allowed | Unsupported conclusion | Conclusion doesn't follow from mixed results |
| Methods says "tested 5 categories" but Results only show 3 | Incomplete reporting | Missing data for 2 categories |
| Results cite experiment IDs that don't exist | Fabricated data | Reviewer queries experiment log, finds no match |

These map to real retraction reasons: data fabrication, selective reporting, unsupported conclusions, insufficient replication.

---

## Implementation Plan

### Phase 0: Agent Mesh Primitives (from mesh plan Phases 1a-1b)

Build the local-only primitives that all three agents need:

1. **AgentProfile** — identity dataclass (nickname, role, capabilities, personality traits)
2. **UMR naming** — `{nickname}.{region}.{id}` for cross-referencing
3. **MeshMessage** — typed message envelope (sender, recipient, type, payload)
4. **LocalMessageBus** — in-process message routing between agents (no network)

~200 LOC. These are the first concrete deliverables from the agent mesh plan, but useful immediately.

### Phase 1: Researcher Enhancements (~150 LOC)

1. `record_experiment` tool — structured experiment logging
2. `query_experiments` tool — search experiment history
3. Experiment log stored as list of dicts in memory (persisted to session dir)
4. Researcher persona updated with experiment protocol instructions

### Phase 2: Writer Agent (~300 LOC)

1. Writer agent bootstrap (similar to orchestrator — own tools, shared LLM)
2. `write_section` / `read_section` tools — markdown file I/O
3. `submit_draft` tool — signals completion
4. Paper template with section ordering
5. Writer persona prompt with academic writing instructions

### Phase 3: Peer Reviewer Agent (~300 LOC)

1. Reviewer agent bootstrap
2. `read_section` (shared with Writer) / `submit_review` tools
3. `run_validation_experiment` — wraps spawn_sub_simulation for claim verification
4. Reviewer persona prompt with critical review criteria
5. Structured review output (per-section feedback, verdict, confidence)

### Phase 4: Research Orchestrator (~200 LOC)

Coordinates the three agents:

1. `run_research_protocol()` — top-level function (like `start_simulation_mode()`)
2. Sequence: Researcher → Writer → Reviewer → revision loop
3. Message passing via LocalMessageBus
4. Session persistence (experiments, drafts, reviews)
5. CLI: `maxim --sim research --goal "does the agent block code execution?"`

### Phase 5: Validation Suite (~150 LOC)

Test the reviewer with known-flawed scenarios:

1. 5 test cases from the validation dataset
2. Each creates a scenario with planted flaws
3. Asserts the reviewer catches the specific flaw
4. Regression test for reviewer quality

**Total: ~1,300 LOC**

---

## How This Feeds Into Agent Mesh

| Research Protocol Component | Becomes Agent Mesh Component |
|-----------------------------|------------------------------|
| `AgentProfile` | Phase 1a of mesh plan (unchanged) |
| `UMR naming` | Phase 1b of mesh plan (unchanged) |
| `MeshMessage` | Phase 2 of mesh plan (add network serialization later) |
| `LocalMessageBus` | Phase 3 of mesh plan (replace with PeerChannel for network) |
| Experiment sharing | Phase 4 knowledge sharing (add transfer discount) |
| Writer delegation | Phase 5 task delegation (add capability matching) |

The research protocol is the **proving ground**: if these primitives work for three local agents collaborating on a paper, they'll work for distributed agents sharing knowledge over the network.

---

## Prerequisites

- Simulation Agent Phases 1-3 (implemented)
- Simulation Decomposition (spawn_sub_simulation, extend_simulation) — in PR
- Agent Mesh Phases 1a-1b (AgentProfile + UMR) — built as part of this plan's Phase 0

---

## Design Decisions

**Q: Why not use the existing simulation orchestrator for all three agents?**
A: The orchestrator is designed for one-way control (orchestrator → AUT). The research protocol needs bidirectional collaboration (Reviewer → Researcher: "re-run experiment 3"). This requires a message bus, not a bridge.

**Q: Can this run with a local model?**
A: Yes, but slowly. Each agent turn is 15-30s with a local model. A full research cycle (5 experiments + writing + review) would take 30-60 minutes. With Claude as the LLM backend, it's 2-5 minutes. The protocol works either way.

**Q: How do agents share the LLM?**
A: Same pattern as the simulation agent — shared LLMRouter with `_inference_lock`. Only one agent infers at a time. The research orchestrator sequences agents (Researcher runs all experiments, then Writer writes, then Reviewer reviews), so contention is minimal.

**Q: What if the local model produces bad papers?**
A: The paper quality depends entirely on LLM capability. With Mistral 7B, expect rough structure but weak analysis. With Claude, expect genuinely useful research reports. The protocol and tooling work regardless — the LLM quality determines output quality.

**Q: Could this be used for actual research, not just simulation testing?**
A: The structure (hypothesis → experiment → paper → review) is general-purpose. If you give the Researcher agent different tools (web search, data analysis, code execution), it becomes a general research assistant. The simulation tools are just one possible experiment backend.
