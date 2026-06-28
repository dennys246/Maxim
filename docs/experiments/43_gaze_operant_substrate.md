# Exp 43 — Operant gaze & substrate generalization (Maxim's "first job" feasibility)

**Status:** sim feasibility study, COMPLETE (2026-06-28). No production code changed.
**Probes:** [`scripts/gaze_substrate/`](../../scripts/gaze_substrate/) (5 standalone scripts, run against the real `NAc` / `EntorhinalCortex`, no LLM).
**Origin:** scoping whether to embody Maxim through a Reachy Mini camera driven by the
[ShredderSegmenter](https://github.com/) action-sports platform — a server alert ("skier present")
→ the robot orients its vision toward people. Constraint: leverage the architecture **without
changing either repo**; the glue is a third package riding the public `PerceptSource`/`ActionSink`
protocols (CC8) + Shredder's existing `/agent` + MediaMTX APIs.

The studies below isolate **what, if anything, the bio-substrate buys** over a plain table/dict for
this attention task — because if a 50-line tracker + a dict would do, Maxim is a fancy `if`.

---

## Method (shared)

A minimal gaze world on the real machinery: perceptual state → NAc cluster id, gaze action →
NAc "tool". Learning rides entirely on the production API:

```python
nac.update_cluster_reward(agent_id, state, f"tool:{action}", reward)   # credit (state,action)
nac.recommend_action(agent_id, available_tools=..., current_cluster_id=state, ...)  # select
```

Every study keeps three arms as a **bug detector**: `contingent` (real), `yoked` (same reward
credited to a *random* (state,action) — Skinner superstition control), `none` (no credit). If
`yoked`/`none` ever show learning, the world is leaking and the result is void.

---

## Probe 1 — operant redirection ([`1_operant_redirection.py`](../../scripts/gaze_substrate/1_operant_redirection.py))

**Q:** does reward redirect random gaze toward a subject, or is any change superstition?
**Result (foveation rate, final window):** contingent **0.88**, yoked 0.08, none 0.09.
Yoked dropped *below* chance. **Decisive: real operant redirection, not superstition.** Pure
substrate (no LLM) by construction.

## Probe 2 — search (Layer 0) + multi-subject coverage ([`2_search_and_coverage.py`](../../scripts/gaze_substrate/2_search_and_coverage.py))

Finite FOV (out-of-view = `empty_<last-seen-side>` state), finer bins, probabilistic reward,
multiple subjects + habituation.
- **Search learns:** in-FOV acquisition 0.80→0.83 (contingent) vs 0.40 (none) — directional
  re-acquisition from empty states.
- **Cross-session (powered, 16 trained→eval pairs, short eval):** loaded-NAc beats fresh-cold by
  **+0.281 ± 0.055 SE** foveation. Learning survives `dump()/load_state()` and skips the learning
  phase. (Probe 1's noisy +0.107 single-pair was underpowered, not weak.)
- **Anti-fixation (persistent/lockable subjects):** habituation OFF dwells 0.77 on one subject
  (covers 1.8/3); ON spreads (dwell 0.50, covers 2.7/3) — **but coverage costs foveation
  (0.68→0.41).** A product knob, not a free win. (Transient subjects confound this — the world
  forces coverage regardless; fixation only manifests with lockable subjects.)

## Probe 3 — gaze-geometry generalization ([`3_geometry_generalization.py`](../../scripts/gaze_substrate/3_geometry_generalization.py))

Encode *bearing* as a place-cell population code (width σ); train NEAR offsets, probe held-out FAR.
**Result:** inverted-U over σ; peak σ=8 → FAR directedness 0.51 (>chance 0.435), and **direction-
correct *given a transfer fired* = 0.84**, while synthetic dict transfers 0%. Too sharp → no
transfer; too broad → left/right merge, discrimination collapses. **Real but modest** — on this
low-dimensional axis a dict is nearly competitive.

## Probe 4 — visual-category transfer to NOVEL instances ([`4_visual_category_transfer.py`](../../scripts/gaze_substrate/4_visual_category_transfer.py))

The axis where the substrate is **provably** load-bearing. Each entity has an appearance vector +
hidden type (person/distractor); the agent learns orient-vs-ignore (orienting to people pays).
Freeze, test on **never-seen individuals**.

| | seen | NOVEL | nodes |
|---|---|---|---|
| dict (exact appearance) | 1.00 | **−0.005 (chance)** | — |
| EC, tight cluster (eps=0.15) | 0.99 | **0.938** | ~4 |
| EC, diffuse (eps=1.0) | 0.81 | 0.091 | ~12 |

**A dict can't recognize a novel face — new key, no value. EC recognizes a never-seen person as a
person and transfers the learned "orient" (0.94).** Bound: transfer only holds while the encoder
*clusters* the category (nodes explode 4→12 as within-category spread grows). The substrate's
generalization is exactly as good as the embedding geometry it's handed.

## Probe 5 — real-encoder validation ([`5_real_encoder_validation.py`](../../scripts/gaze_substrate/5_real_encoder_validation.py))

Does the encoder Maxim *actually* uses (`all-mpnet-base-v2`, via `_get_encoder`) give geometry tight
enough for Probe 4's transfer? Ski-domain phrases, people vs inanimate mountain objects.
- within-people **0.436** > across **0.175** (gap +0.26).
- **RAW embeddings @ default 0.44 → 0.917 novel discrimination** (dict = 0). Centering unnecessary.

**The real encoder supports the transfer out of the box** — because EC does nearest-neighbour-
above-threshold (a novel skier's nearest trained node is a skier), not mean-cosine clustering, so
it is robust to sentence-embedding anisotropy. **CAVEAT:** this is a *text* proxy. The robot needs a
**vision** encoder on real images; that validation is the hardware prerequisite (below), not done here.

---

## Verdict (ranked by evidence strength)

1. **Operant redirection** — decisive (0.88 vs 0.08; yoked rules out superstition).
2. **Cross-session persistence** — strong (+0.281 ± 0.055 SE).
3. **Visual-category generalization to novel instances** — decisive (0.94 synthetic / 0.92 real
   text encoder) **and the only result that strictly requires the substrate.** Maps directly onto the
   ShredderSegmenter job: track *novel* skiers, not memorized ones.
4. **Gaze-geometry generalization** — modest (dict competitive in low-dim).

1 and 2 a dict could also do (credit machinery / persistence). 3 is the load-bearing substrate win,
on the visual-content axis. **Honest hardware pitch: "a camera that learns *what's worth looking at*
and generalizes it to subjects it has never seen,"** not "persisted dict + servo."

---

## Hardware integration design (gated — do NOT start until prerequisites pass)

**Topology (no changes to Maxim or ShredderSegmenter):** a third glue package.
- **`PerceptSource` (CC8):** non-blocking `next_percept()` reads a local inbox filled by a transport
  thread that pulls frames from Shredder's MediaMTX stream, runs Shredder's pose/detection engine,
  and emits detected entities as `(bearing, visual_feature_vector, confidence)`. Leader owns the
  substrate — embeddings are derived on-leader, never on the wire (`Percept` wire-format invariant).
- **State:** `visual_feature_vector` → EC `"vision"` node = the "what" (category); `bearing` → the
  "where" (motor). Layer-0 entropic search drive for out-of-FOV; Layer-1 homeostatic gaze-centering.
- **`ActionSink` / gaze tool:** pan/tilt → Reachy SDK (or a sim camera effector behind the same
  interface, so sim-trained policy transfers).
- **Reward layers:** Layer-1 dense **potential-difference** gaze-centering (use the `γΦ(s')−Φ(s)`
  form — policy-invariant, no reward-hacking) emanating from substrate-recognized entities (do NOT
  hand-code "person" — route recognition through EC); Layer-0 entropic search; Layer-2 Shredder
  live pose-confidence; Layer-3 review approve/reject as persisted cross-session `reward_bias`.

**Prerequisites (each is cheap and sim/desk-side — clear them before any robot work):**
- **P1 — vision-encoder validation:** re-run Probes 4/5 with a real **vision** encoder
  (CLIP/ResNet-class) on real images of skiers vs mountain objects. Confirm within>across +
  novel-instance transfer hold for *images*, not just text. This is the make-or-break; the whole
  substrate pitch rests on it.
- **P2 — reward-field shaping:** implement Layer-1 as potential-difference shaping over
  substrate-recognized entities; confirm it fixes the far-bin coverage hole without reward-hacking.
- **P3 — anti-fixation tuning:** set the habituation/coverage-vs-foveation trade for realistic
  multi-subject scenes.

---

## Running the probes

```bash
# editable install already exposes `maxim`; each probe is standalone, no LLM, seconds to run
python scripts/gaze_substrate/1_operant_redirection.py
python scripts/gaze_substrate/4_visual_category_transfer.py
python scripts/gaze_substrate/5_real_encoder_validation.py   # downloads all-mpnet-base-v2 on first run
```
