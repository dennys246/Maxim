# Hippocampal Recall Experiment — Run Notes

## Run 2026-04-06 (Direct Injection, Qwen2.5-14B orch + Mistral-7B AUT)

### Setup
- **Orchestrator**: Qwen2.5-14B-Instruct (self-hosted, RTX 5080)
- **AUT**: Mistral-7B (self-hosted, same leader)
- **Campaign**: `hippocampal_recall_short.yaml` (7 turns, 3 interference)
- **Delivery**: Direct bridge injection (bypassed orchestrator LLM for turns)
- **Session**: `data/sim_reports/20260406_172631/`

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

### Next Steps

- [ ] Re-run with the miller warning removed or softened (modify Turn 2)
- [ ] Try a different AUT model (Qwen2.5-7B? Llama-3-8B?) that may roleplay better
- [ ] Add a `respond`-based engagement check after each turn (does the AUT say anything about the scene?)
- [ ] Inspect the hippocampus state file to verify Verath memory content
- [ ] Check if spreading activation from "silver elm + door" reaches the Verath memory
