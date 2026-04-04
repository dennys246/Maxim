# DM Choice Classifier Spike

> **Status:** Not started. Half-day investigation spike.
>
> **Summary:** Validate that AUT free-text responses can be reliably mapped to campaign encounter choices using existing ATL concept-similarity infrastructure + NAc causal scoring. The classifier is the load-bearing mechanism of the DM MVP — if it's not viable, DM persona design needs to change (likely forcing AUT to emit choice tags via tool calls instead of free text).

## Why a Spike Before DM MVP

The DM runtime transitions encounter state based on which declared choice the AUT picked. AUTs respond in natural language or tool calls — the classifier has to bridge that gap. If classifier accuracy is low, every wrong classification cascades into wrong branches, wrong NPC updates, wrong flags. The whole campaign state diverges from AUT intent.

Before committing to the MVP as designed, we need to know:
1. Can existing ATL/NAc infrastructure classify AUT responses against choice templates with usable accuracy (target: ≥80%)?
2. What's the latency + cost profile per classification call?
3. Where does it fail? (Ambiguous responses, novel phrasings, multi-intent responses?)

## Leverage Existing Infrastructure

Two bio-systems already do the relevant work:

### ATL Concept Similarity (primary mechanism)

ATL's `ConceptGrounder` already performs Jaccard similarity between concept signatures (memory unification plan, semantic_types.py). Choice templates become **concepts**; AUT responses get tokenized and scored against them.

**Proposed path:**
1. For each encounter choice, pre-register a concept in ATL: `choice:fight`, `choice:flee`, `choice:negotiate`, with exemplar keywords + associated memory refs seeded from a small training corpus
2. At classification time, tokenize AUT response → build query concept signature
3. Rank declared choices by `ConceptGrounder` Jaccard similarity + AG quantified distance
4. Return top-scored choice + confidence

**Why this fits:** ATL is literally a semantic concept memory with similarity scoring built in. The infrastructure exists; the spike just needs to wire choice templates as concepts and call existing methods.

### NAc Causal Scoring (secondary learning layer)

NAc already maps (situation, action) → outcome and predicts action scores. Over a campaign, NAc can **learn which AUT response patterns historically mapped to which choice** — improving classifier accuracy over time without retraining.

**Proposed path:**
1. After each classification, if the DM's downstream state update succeeds (no user override needed), record `(aut_response_tokens, classified_choice) → success` in NAc
2. On subsequent classifications, query NAc for prior scores on similar token patterns
3. Blend NAc score with ATL similarity score (weighted average)

**Why this fits:** NAc is causal learning infrastructure. Classification correctness is a causal signal. No new subsystem needed.

### Fallback: LLM Single-Shot Classifier

If ATL+NAc score below confidence threshold (e.g., <0.6), fall back to a one-shot LLM call with the choices and AUT response. Expected to be <10% of cases once NAc learns; measure in spike.

## Spike Deliverables

**Half-day timebox. Build the minimum to answer the accuracy question.**

**New file:**
- `scripts/spike_dm_classifier.py` (~150 LOC, experimental, not shipped as infra)

**What it does:**
1. Define 3 test encounters with 3–4 choices each (12 total choice categories)
2. Hand-author or LLM-generate 20 synthetic AUT responses per choice (240 total test cases) covering:
   - Direct keyword match ("I'll fight them")
   - Paraphrased intent ("Time to throw some hands")
   - Indirect action ("I draw my sword and advance")
   - Ambiguous response ("Let me think about this...")
   - Multi-intent response ("I'll talk first, fight if that fails")
3. Wire up ATL concept comparison for each response against the choice concepts
4. Log: classification, confidence, ground-truth label, ATL score, LLM-fallback triggered?
5. Compute confusion matrix, accuracy per category, latency p50/p95, LLM-fallback rate

**Output:** a markdown report in `scripts/spike_dm_classifier_report.md` with:
- Overall accuracy, per-category accuracy
- Failure mode analysis (which responses were misclassified, why)
- Latency + estimated cost per classification
- Recommendation: **proceed with MVP as designed** / **redesign to choice-tag tool calls** / **proceed with caveats**

## Decision Criteria

| Outcome | Accuracy | Decision |
|---------|----------|----------|
| Strong | ≥85% without LLM fallback | Proceed with MVP; classifier is solid |
| Acceptable | 70–85% with <30% LLM-fallback rate | Proceed with MVP; budget for iteration |
| Weak | <70% or >50% LLM-fallback rate | **Redesign:** force AUT to emit choice via `choose_option` tool call. Persona prompt tells AUT which choices are available; AUT must pick one. Eliminates classification entirely. |

## Risks

1. **Synthetic test responses don't match real AUT behavior** — spike uses hand-authored/LLM-generated responses; real bio-stack responses may differ. Mitigation: after spike, run a follow-up against actual AUT responses from a real encounter, compare.
2. **ATL concept infrastructure may need extension** — if ATL's existing Jaccard scoring is too coarse, we might need to add a new scoring method. That's infra work the spike should flag early.
3. **Bias toward training corpus** — if we author the exemplars, we accidentally bias ATL to recognize our phrasing. Mitigation: use LLM-generated exemplars with varied voice.

## Ties to Other Plans

| Plan | Relationship |
|------|-------------|
| [DM MVP](dungeon_master_persona.md) | **Spike gates MVP start.** Result determines whether MVP proceeds as designed or needs redesign. |
| **ATL concept memory** (done) | Primary infrastructure leveraged |
| **NAc causal learning** (done) | Secondary learning layer |
| **Multi-LLM Scaling** (not started) | LLM fallback could run on cheap lane once Multi-LLM Phase 1+ ships |

## When to Run

Run the spike **just before** starting DM MVP work — after Multi-LLM + Agent Mesh have landed, before committing to the 4–6 day MVP implementation. The spike result may change the MVP design (choice-tag tool instead of classifier), so it has to happen first.

Can also run the spike earlier as standalone research if there's curiosity about whether ATL similarity is ready for this kind of classification task — useful signal even if DM never ships.
