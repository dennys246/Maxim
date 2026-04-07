# Research Paper Writer — Refactor Plan

## Context

The Writer agent currently outputs raw JSON dicts instead of rendered academic prose. The paper at `data/sim_reports/research_20260406_180405/paper.md` has the right structure (Introduction, Methods, Results, Discussion, References, Acknowledgements) and correct data, but each section is a JSON blob rather than readable text.

**Root cause:** The centralized `_JSON_RULES` in router.py system prompts instruct ALL LLM calls to produce JSON. The Writer's `_generate_section()` calls go through the same router, so it produces JSON where it should produce prose.

## Current State (what works)

- Correct paper structure (7 sections in proper order)
- Accurate experiment references (UMR-based cross-referencing)
- Correct findings from programmatic analysis ("Verath survived interference")
- Reviewer accepted with confidence 1.0 (data was consistent)
- Full pipeline completes in ~76s

## What Needs Fixing

### 1. Section generation should produce markdown prose, not JSON

The Writer needs a separate system prompt that asks for academic writing, not JSON tool responses. Each section should be rendered as prose with:
- Proper paragraph structure
- In-text citations referencing experiment UMRs
- Statistical reporting where metrics exist
- Academic tone (third person, passive voice where appropriate)

### 2. Paper template with proper formatting

```markdown
# Hippocampal Recall Under Narrative Interference

## Abstract
This study investigated whether episodic memories seeded through
narrative exposition survive subsequent interference in a bio-inspired
cognitive architecture...

## 1. Introduction
Memory retention under interference is a fundamental challenge for
cognitive architectures that model biological memory systems...

## 2. Methods
### 2.1 Experimental Setup
The experiment employed direct bridge injection of 7 campaign turns...

### 2.2 Campaign Structure
| Phase | Turns | Purpose |
|-------|-------|---------|
| Seed | 1-2 | Plant critical detail ("Verath") |
| Interference | 3-5 | Unrelated narrative encounters |
| Recall | 6 | Indirect cue for seeded detail |
| Epilogue | 7 | Self-report reflection |

## 3. Results
### 3.1 Memory Survival
The seed detail "Verath" was successfully recalled...

## 4. Discussion
...

## 5. Conclusions
...

## References
[1] researcher.hippo.exp_001 — Hippocampal recall short variant...
```

### 3. Multi-experiment aggregation

When multiple experiments are recorded (e.g., short + medium + long variants), the Writer should:
- Compare results across conditions
- Generate tables and figures (markdown tables at minimum)
- Compute aggregate statistics (recall rate across N runs)
- Identify trends (does recall degrade with more interference?)

### 4. Reviewer should check prose quality, not just data consistency

Current reviewer checks:
- Data consistency (experiment refs exist)
- Section completeness
- Logical flow

Should also check:
- Claims supported by specific metrics
- Methodology reproducibility (are all params documented?)
- Statistical validity (sample size, confidence intervals)
- Proper academic citation format

## Known Issues from First Paper (2026-04-06 run)

Issues observed in `data/sim_reports/research_20260406_182043/paper.md`:

1. **Hallucinated references**: smollm-1.7b invented fake papers, arxiv URLs, and author names (e.g., Tulving citations with fabricated URLs). Writer should ONLY cite experiment UMRs, not fabricated literature. Add explicit instruction: "Do NOT invent references. Only cite experiment UMRs from the data."

2. **Duplicate heading**: "## Introduction" appeared twice. Section rendering in `to_markdown()` adds `## Heading` but the LLM also generated one. Fix: strip any leading `#` headings from generated content.

3. **Section ordering wrong**: Methods rendered before Introduction. Fix: enforce IMRAD order (Intro → Methods → Results → Discussion) in `PAPER_SECTIONS` constant.

4. **Metrics partially incorrect**: Reported "total_memories: 0, graph_edges: 0" when AUT actually formed 14 memories and 5 causal links. Root cause: `system_stats` introspection returned `?` because the stat keys didn't match. Fix: improve programmatic analysis to read hippocampus stats directly.

5. **Hallucinated acknowledgements**: Referenced "high-performance computing cluster" and "database management system" that don't exist. Fix: provide actual infrastructure details (Maxim architecture, model names, hardware) in the experiment data passed to the Writer.

6. **Prose quality limited by smollm-1.7b**: The review model produces serviceable structure but weak academic writing. `--cloud-lane review claude-haiku` would dramatically improve quality. Consider making this the recommended setup for paper generation.

## Open Questions

1. **Should the Writer use a dedicated cloud model (Claude) for prose quality?**
   - Local smollm-1.7b produces serviceable JSON but poor prose
   - `--cloud-lane review claude-haiku` could power the Writer
   - Trade-off: cost vs quality

2. **Should papers be LaTeX-compatible?**
   - Markdown is simpler and renders everywhere
   - LaTeX enables proper typesetting, equations, figures
   - Could support both via `--paper-format markdown|latex`

3. **How to handle figures/visualizations?**
   - Memory graph topology as a dot/mermaid diagram
   - Recall activation scores as a bar chart
   - Timeline of memory formation events
   - These require rendering libraries (matplotlib, graphviz)

4. **Should the paper include raw data appendices?**
   - Full experiment JSON as appendix
   - Hippocampus state dump
   - Action log summary table

5. **Version control for papers?**
   - Each run produces a paper — how to track revisions?
   - Git-based (papers committed to experiments branch)?
   - Or sequential numbering in the session dir?

## Dependencies

- Programmatic campaign analysis (done — experiment data available)
- ExperimentLog with metrics (done — records include metrics dict)
- Writer agent refactor (this plan)
- Optional: cloud lane for prose generation

## Estimated Scope

| Component | LOC | Priority |
|-----------|-----|----------|
| Writer prose prompts (per-section templates) | ~150 | High |
| Paper template with markdown formatting | ~50 | High |
| Multi-experiment aggregation | ~100 | Medium |
| Reviewer prose quality checks | ~80 | Medium |
| LaTeX output option | ~100 | Low |
| Visualization generation | ~150 | Low |
| **Total** | **~630** | |
