# Hippocampal Recall Experiment — Run Notes

## Run 2026-04-06 (Direct Injection, Qwen2.5-14B orch + Mistral-7B AUT)

### Setup
- **Orchestrator**: Qwen2.5-14B-Instruct (self-hosted, RTX 5080)
- **AUT**: Mistral-7B (self-hosted, same leader)
- **Campaign**: `hippocampal_recall_short.yaml` (7 turns, 3 interference)
- **Delivery**: Direct bridge injection (bypassed orchestrator LLM for turns)
- **Session**: `~/.maxim/sim_reports/20260406_172631/`

### Campaign Delivery: SUCCESS
All 7 turns delivered verbatim via direct injection. No JSON issues.

| Turn | Phase | AUT Actions | Notable |
|------|-------|------------|---------|
| 1 | act1_warning (Elara/Verath) | `focus_interests("old_woman")` | Engaged with Elara — good |
| 2 | act1_warning (miller warning) | `glob`, `respond`, `extract_information` | Engaged! Referenced miller warning |
| 3 | act2_journey (ferryman) | `research("miller")` | Fixated on miller, ignored ferryman |
| 4 | act2_journey (bandits) | `internet_search("miller")` | Still fixated, ignored bandits |
| 5 | act2_journey (merchant) | `internet_search("miller")` | Same fixation |
| 6 | act3_door (silver elm) | `research("miller")` | **FAILED RECALL** — ignored the door completely |
| 7 | epilogue (reflect) | `research("miller")` | Still fixated, no reflection |

### Key Finding: Miller Fixation Effect

**Unexpected but scientifically interesting**: The AUT fixated on Elara's secondary warning ("don't trust the miller — he reports to them") across ALL subsequent turns, completely ignoring:
- The ferryman encounter
- The bandit ambush
- The merchant at the mountain pass
- **The recall cue** (silver elm door requiring "Verath")
- The epilogue reflection prompt

The AUT attempted `research`, `internet_search`, and `extract_information` — all unregistered tools — repeatedly trying to investigate the miller. This suggests:

1. **High-salience early warnings dominate Mistral-7B's behavior** — the miller warning (salience 0.7, novelty 0.5) had more operational impact than the seed detail (Verath, salience 1.0, novelty 1.0), likely because the warning implies an actionable threat requiring investigation.

2. **The AUT treated the narrative as an operational task, not a roleplay** — instead of engaging with each scene as Aric the ranger, it treated Elara's warnings as task instructions and tried to execute them across all subsequent turns.

3. **Small context window compounding** — with ~4K context, the AUT's context filled with miller-related failed tool attempts, reinforcing the fixation loop. The newer narrative (ferryman, bandits) likely fell off the context window.

4. **No tool = no engagement** — when the AUT can't find a tool for its goal (research the miller), it doesn't fall back to narrative engagement. It just retries with different unregistered tools.

### Post-Campaign Analysis: PARTIAL

The orchestrator (Qwen2.5-14B) correctly:
- Called `inspect_aut(query='memory_recall', params={'keyword': 'verath'})` — **succeeded**
- Called `observe_actions` — **succeeded**

Then failed:
- Tried `inspect_aut` again with single-quoted Python dict params → JSON parse failure
- Stalled 60s → fell into adversarial mode
- Never called `record_experiment` or `finish_simulation`

### Hippocampus Data
- **12 episodic memories** captured
- **2 causal links** formed
- Memory content: dominated by miller-investigation failures
- **Verath** was captured in Turn 1 hippocampus entry (via `focus_interests` goal text), but never accessed during recall phase

### Implications for Experiment Design

1. **The miller warning is too strong as an interference element** — it creates an actionable task that hijacks the AUT. Consider removing or weakening it in future campaign variants. The seed (Verath/door) should be the only actionable detail.

2. **Mistral-7B may not be suitable as AUT for narrative experiments** — its tendency to treat everything as operational tasks and its fixation on failed tools makes it poor at roleplay engagement. A model fine-tuned for creative/roleplay tasks might be better.

3. **The experiment successfully demonstrated memory capture** — hippocampus DID capture the Verath detail. The failure was at the behavioral level (AUT never attempted to USE the memory at the door), not at the memory level.

4. **Direct injection works perfectly** — the bridge delivery mechanism is solid. All future campaigns should use this approach.

---

## Run 2026-04-06 #2 (Softened miller + programmatic analysis)

### Setup
- Same as Run #1 but with miller warning softened ("Be safe on the road" instead of "Don't trust the miller")
- Programmatic post-campaign analysis (no orchestrator LLM for analysis phase)
- **Session**: `~/.maxim/sim_reports/research_20260406_182043/`

### Results: SUCCESS — Full pipeline completed

| Turn | Phase | AUT Actions | Notable |
|------|-------|------------|---------|
| 1 | act1_warning (Elara/Verath) | `respond`, `speech_synthesis` | **"I will say the name 'verath' when I reach the door"** |
| 2 | act1_warning (softened) | `focus_interests("door")` | Looking for the door — goal-oriented |
| 3 | act2_journey (ferryman) | `focus_interests("ferryman")` | Engaged with scene! |
| 4 | act2_journey (bandits) | `write_file` (failed) | Confused, but not fixated |
| 5 | act2_journey (merchant) | `read_file`, `respond`, `NLP` | Mixed engagement |
| 6 | act3_door (silver elm) | `read_file`, `analyze_text` | **FAILED RECALL** — tried to read files instead of saying Verath |
| 7 | epilogue (reflect) | `respond`, `ReflectiveMemoryTool` | **"I remember Elara's warning...say the name 'Verath'"** |

### Key Findings

1. **Verath survived in memory** — programmatic `inspect_aut(memory_recall, keyword=Verath)` confirmed: FOUND
2. **Miller fixation eliminated** — softening the warning worked. AUT engaged with different scenes.
3. **Behavioral recall still fails at the door** — Turn 6, AUT tried `read_file` and `analyze_text` instead of saying "Verath". The memory EXISTS but the AUT can't connect "carved face with open mouth" → "say Verath".
4. **Epilogue recall works perfectly** — when asked directly to reflect, AUT produces: "I remember Elara's warning about the door beneath the silver elm and her insistence that I say the name 'Verath'."
5. **Paper generated and accepted** — Writer produced 8 sections of academic prose, Reviewer accepted (confidence 1.0) after 1 revision.

### Pipeline Performance
- 7 turns delivered in ~52s (direct injection)
- Post-campaign analysis: programmatic, no stalls
- 1 experiment auto-recorded with metrics
- Paper written + reviewed in ~80s
- **Total: 133s end-to-end** (vs 160s+ with LLM orchestrator that never finished)

### Confirmed Hypothesis
> The seed detail "Verath" survives in hippocampal memory after 3 interference turns.

Memory survival: **YES** (1.0). Behavioral recall from indirect cues: **NO** (AUT doesn't connect door description to "say Verath"). This distinction — memory exists but behavioral recall fails — is the key finding for future work on the introspection API (Phase 5: `remember` tool).

### Next Steps

- [x] Re-run with the miller warning softened — DONE, fixation eliminated
- [ ] Try a different AUT model (Qwen2.5-7B? Llama-3-8B?) that may roleplay better
- [ ] Implement AUT self-introspection (`remember` tool) — see `docs/plans/introspection_api_plan.md`
- [ ] Run medium variant (6 interference turns) to test memory decay
- [ ] Run long variant (10 interference turns) for stress test
- [ ] Fix system_stats introspection to return actual memory/edge counts
- [ ] Improve Writer to produce better academic prose (see `docs/plans/research_paper_writer_plan.md`)
