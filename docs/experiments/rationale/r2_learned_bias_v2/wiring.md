# R2 learned-bias v2 — WIRING-lens design review

**Verdict: DO-NOT-BUILD as written — one blocking flaw (the round-robin schedule re-creates the exact
v1 causal-saturation forced-null), plus six SHOULD-FIX seam issues.** The good news: the encoding /
credit seams v2 inherits from the v1 harness are *correct* (real consumers, real `record_outcome`,
`_encode_current_clusters` at both book and probe time), so this is a schedule + framing problem, not
a plumbing rebuild. But two design assumptions — "balanced round-robin" and "eat's baseline is
~1/(K+1)" — are false against the *real deterministic argmax consumer*, and if built as written the
run yields a forced null that looks exactly like a real null (the wiring doc's own warning).

Charter: does the design ride the real consumers + real credit path (D43), with the right
encoding/seams, and no hand-composed shortcut that passes while the loop fails?

## What I verified (code, this repo, main unless noted)

- **Real credit path is inherited correctly.** The v1 harness (commit `f4f3e26f`, `scripts/survival_world/r2_learned_bias.py`)
  routes through `build_minecraft_aut` → `executor.execute` → `read_learning_side_effects` →
  `record_outcome`, and encodes clusters with the production `_encode_current_clusters` at BOTH
  training time and probe time (`_marginal_probe`). This is D43-clean — no hand-composed credit. v2
  "evolves" this, so the seam is sound *if* v2 keeps it (see SHOULD-FIX 6).
- **The two learning channels behave as `docs/wiring/substrate-learning-channels.md` records.** In
  `tool_dispatch.record_outcome`: the causal link (`nac.observe`, `event_signature="tool:X"`) books
  UNCONDITIONALLY on every success and is state-blind; the cluster reward bias is state-conditioned,
  gated, and routed to the **interoception** cluster only. `drive_credit_withheld=True` (the NO-CREDIT
  ablation) suppresses the cluster term but leaves the causal link intact — correct.
- **`recommend_action` is deterministic argmax with a name-sort tie-break.**
  `best_tool = max(scores, key=lambda t: (scores[t], t))` (`nac.py`). On a score tie the
  **lexicographically largest tool name wins** — for the roster `minecraft_player_{eat,mine_block,move_to,turn}`,
  that is `turn` (t > m > e), i.e. a *competitor*, never `eat`. There is no stochasticity anywhere in
  the selector.
- **`food` is DUAL-modality on `minecraft_player`** (`_data/components/bodies/minecraft_player.yaml`):
  declared `modality: world` (1 of 16 world sensors) AND carries an entropic drive, so it feeds the
  interoception channel via `_read_drive_states` (which returns raw `food`/`health` PLUS derived
  `hunger`/`threat`). The credited cluster is the INTEROCEPTION one; the world cluster is inert for eat
  bias under relief-only.
- **The drive gate exists and is config-gated.** `NACConfig.drive_gate_enabled` (default False,
  wired from `config.json::sim.drive_gate_enabled`); when a drive > `drive_gate_threshold` (0.5) it
  HARD-narrows selection to drive-relevant tools ({eat} here). v1's `FROZEN_CONFIG` does NOT pin it.
- **The v1 harness that v2 "reuses" is NOT on `main`** — `f4f3e26f` is unmerged (`git merge-base
  --is-ancestor f4f3e26f main` → false). `scripts/survival_world/` on main holds only
  `setup_world.py` + `break3_smoke.py`.

## The algebraic reduction that reframes everything (read this first)

Under the deterministic argmax, `isolated_effect` collapses to a familiar quantity. Per seed, with
`A_state,arm = 1[eat wins at that probe]`:

    isolated_effect = gap_LEARNING − gap_NO-CREDIT
                    = (A_11,L − A_18,L) − (A_11,NC − A_18,NC)
                    = (A_11,L − A_11,NC) − (A_18,L − A_18,NC)

At **food 18** the LEARNING and NO-CREDIT arms are IDENTICAL: neither carries cluster bias on the
food-18 cluster (LEARNING booked bias only on the food≤4/food-11 cluster `fd0ae83c`; NO-CREDIT booked
none), and same-seed training executes eat/competitors identically, so the causal links and prior
match. Therefore **A_18,L = A_18,NC**, the second term is 0, and:

    isolated_effect  ≡  A_11,L − A_11,NC   =   the v1 MARGINAL flip at food 11, over a competitor roster.

**Consequence.** The satiated arm and the food-18 probe contribute *nothing* to the primary; the
whole "state-contingency gap" is, for this consumer, the v1 marginal-flip-at-food-11 measured with K
trained competitors in the roster. That is a perfectly good experiment — it directly attacks the v1
saturation finding — but the design should say so, because (a) it tells you exactly what must be true
for a positive (SHOULD-FIX 1), and (b) it exposes that the food-18 machinery can only *harm* the
result, never help (SHOULD-FIX 3).

---

## DO-NOT-BUILD

### D1. The round-robin schedule `eat, c1, eat, c2, …` re-creates the v1 causal saturation → forced null

**Flaw.** The stated schedule trains eat every *other* episode → eat gets ~50% of episodes while each
competitor gets ~50%/K. Causal-link confidence is monotone in success count (Rescorla-Wagner toward
1.0), so eat's causal score ends up STRICTLY above every competitor's. From the reduction,
`isolated_effect = A_11,L − A_11,NC`; a positive requires **A_11,NC = 0**, i.e. in the NO-CREDIT arm
(no cluster bias) eat must NOT win at food 11. But if eat's causal strictly dominates, eat wins the
food-11 argmax with no cluster bias at all → **A_11,NC = 1 → isolated_effect ≤ 0 for every seed and
every K.**

**Consequence.** A structural, forced null — the exact failure `substrate-learning-channels.md`
warns about ("a single dominant causal link saturates the behavioural signal… you'll get a forced
null"), re-entered through the training schedule instead of the roster. It is indistinguishable from
"the drive-relief credit doesn't work," so the run cannot answer its own question. Because the probe
is a binary argmax, even an epsilon causal edge to eat is decisive — this is a knife-edge, not a soft
bias.

**Fix.** (a) Equalize **per-tool success counts**: cycle `eat, c1, c2, …, cK` (each tool once per
cycle) so eat and every competitor reach the same causal count — then at food 11 the no-bias case
ties and the name-sort tie-break hands the pick to a competitor (`turn`), giving the +1 cluster bias
real headroom. (b) Add a **causal-parity / headroom validity gate** (mirror of v1's anti-vacuity
gates): refuse the seed unless, in the NO-CREDIT arm, the food-11 pick is a *competitor* (A_11,NC=0)
AND each competitor's `tool:X` causal confidence is within an epsilon of eat's. Without this gate a
silent imbalance produces a null that no reader can distinguish from a real one — the very thing this
design review exists to prevent.

---

## SHOULD-FIX

### S1. `drive_relief_only=True` must be passed for EVERY `record_outcome` — v1's harness does NOT set it

**Flaw.** The prereg lists "relief-only enforced" as a carried-over guard, but the v1 harness's
`record_outcome` call passes only `drive_credit_withheld=dwh`; it never sets `drive_relief_only`.
That was harmless in v1 (single trained tool, eat always relieved). In v2 the competitors execute and
record too, and with `drive_relief_only` unset a *successful competitor with no drive effect* falls
through to the **tool-success floor** (`tool_dispatch.record_outcome`, `cluster_reward = 1.0; source
= "tool_success"`) and books +1 to the **interoception** cluster — the same cluster eat's relief
credit lives on. Competitors would accrue cluster bias, and worse, that bias tips the probe toward
*them*, corrupting both the balance premise and the primary.

**Consequence.** "Eat is the only state-conditioned tool" silently becomes false; the isolated effect
is measured against a polluted baseline. Directly the `substrate-learning-channels.md` "floor also
books to the interoception cluster" trap.

**Fix.** Pass `drive_relief_only=True` on every v2 `record_outcome` (eat and competitors). Add an
anti-vacuity gate asserting each competitor's interoception cluster bias for `tool:X` stayed 0 across
training. Do not simply "reuse v1's call" — this kwarg is the difference.

### S2. The dose-response premise assumes a stochastic selector; the real consumer is deterministic argmax

**Flaw.** The titration rationale — "eat's causal baseline is ~1/(K+1), so headroom grows with K" —
is a softmax/frequency intuition. `recommend_action` is deterministic argmax (verified; the prereg
itself relies on this for "K_probe=1 binary picks"). Under argmax with a fixed schedule, eat's
"baseline" is 0 or 1 depending on the causal-tie / tie-break, **independent of K**: once one competitor
(`turn`) beats eat on a tie, adding more competitors does not lower eat's baseline further. There is
no `1/(K+1)` quantity in this instrument.

**Consequence.** The predicted dose-response curve is flat, so the titration cannot demonstrate the
"scales with choice-space" claim it is built for. It won't *false-pass* (decision rule 2 accepts
flat-positive at K*), but the second headline question is unanswerable by this apparatus — the extra
K machinery buys nothing scientifically while multiplying compute (arms × K × N).

**Fix.** Either (a) re-derive the dose-response prediction for a deterministic argmax selector and
state honestly what varies with K (likely: nothing, absent stochasticity) and drop the titration to a
single well-powered K*; or (b) if a graded P(eat) curve is genuinely wanted, introduce a stochastic
pick (temperature over `scores`) as a *pre-registered instrument change* — but that is a new consumer
and needs its own wiring/confounding pass. Do not ship a titration whose mechanism the selector
cannot produce.

### S3. Keep food-18 only as a leakage NEGATIVE-CONTROL, with a pre-freeze "different cluster" disclosure + runtime assert

**Flaw.** Per the reduction, the food-18 probe contributes to the primary only through the term
`A_18,L − A_18,NC`, which is 0 **iff food-18 is a different interoception cluster than the training
cluster `fd0ae83c`**. If EC lumps food-18 into `fd0ae83c` (a clustering fact governed by the pinned
`ec_pattern_complete_threshold`), then A_18,L picks up the cluster bias, gap_LEARNING shrinks, and the
isolated effect is *depressed* — a false null. So the satiated probe can only null the result, never
strengthen it.

**Consequence.** A load-bearing wiring fact ("food-18 is a distinct cluster") is assumed but, unlike
the food-11-shares fact, is neither disclosed nor asserted. A drifted threshold that merges 11 and 18
manufactures a null.

**Fix.** Treat food-18 as an explicit negative control (its purpose: detect cluster-bias leakage),
and add BOTH a pre-freeze disclosure ("food-18 encodes interoception cluster X ≠ `fd0ae83c`") AND a
per-seed runtime assert (the encoded food-18 interoception cluster id ≠ the trained/credited cluster
id). Mirror the existing food-11-shares assert. If food-18 is dropped in favor of stating the primary
as the marginal-over-competitors (S from the reduction), this collapses to just the food-11-shares +
food-18-differs disclosures.

### S4. Pin `drive_gate_enabled=False` in the frozen apparatus (v1's FROZEN_CONFIG omits it)

**Flaw.** With competitors now in the roster, the drive gate is decisive where it was irrelevant in
v1. If `drive_gate_enabled` is True (it is read from ambient `config.json::sim`) and the derived
hunger at food 11 exceeds `drive_gate_threshold` (0.5), the gate HARD-narrows selection to the
drive-relevant subset ({eat}) — forcing eat in BOTH arms at the hungry probe → A_11,NC = 1 → forced
null. v1's `FROZEN_CONFIG` / `_config_fingerprint` do not include this field, so a stable-but-wrong
ambient config would pass the frozen-apparatus assert and silently kill the effect.

**Fix.** Add `drive_gate_enabled` (expected False) to `FROZEN_CONFIG` and the fingerprint assert. Also
re-confirm at freeze, WITH the competitor roster present, that the derived hunger at food 11 is ≤
`drive_gate_threshold` (the `>` is strict, so exactly-0.5 is safe) and that the NO-CREDIT food-11 pick
is a competitor — the headroom disclosure, not just the cold-prior disclosure.

### S5. Each competitor must book a POSITIVE, comparable causal link — guard it (mine_block is the risk)

**Flaw.** The design asserts competitors are "always-executable, always-successful" but only v1's
eat-side anti-vacuity gate is inherited. `mine_block` needs a mineable block at the supplied
coordinates every episode; if it fails, `record_outcome` books a NEGATIVE causal link (`learn_valence
= NEGATIVE` on `not success`) → its score goes negative → it cannot balance eat → causal
re-saturates toward eat → the D1 null returns through a different door. `move_to`/`turn` are safer but
still need verified success.

**Fix.** Add a per-competitor anti-vacuity gate: assert each competitor booked a POSITIVE `tool:X`
causal link whose confidence is within epsilon of eat's, per seed; refuse the seed otherwise. This is
the balance premise made mechanical rather than assumed. (Environment lens owns whether the world
affords each competitor; the wiring requirement is that the harness *verifies the causal booking*, not
just that the tool "ran".)

### S6. Pin the encode seam to `_encode_current_clusters` — do NOT crib exp56's `encode_clusters`

**Flaw.** The prereg says v2 reuses "exp56 machinery" and "the v1 harness." exp56's
`common.py::encode_clusters` hand-rolls the interoception encode as `encode_sensors(modality=
"interoception", sensors={"d1": d1}, …)` — correct for the exp56 *bench body*, WRONG for
`minecraft_player`, whose interoception vector is `_read_drive_states` (raw food/health + derived
hunger/threat). A builder starting from exp56 would encode the wrong cluster, and training-book vs
probe-read cluster ids would silently diverge → credit never applies → false null. v1 already does
this correctly (`_encode_current_clusters` at both sites); v2 must not regress it.

**Fix.** State explicitly in the prereg that v2 uses `runtime.agent_loop._encode_current_clusters`
for BOTH the training-time `clusters=` argument to `record_outcome` and the probe-time
`current_clusters=` argument to `recommend_action`, and that `_set_probe_state` pins `health` (= 20,
as v1's `probe_health`) so only food varies at the probe. Do not import exp56's `encode_clusters`.

---

## NIT

- **N1. Land the v1 harness on `main` before claiming reuse.** `f4f3e26f` is unmerged; "reuses
  `scripts/survival_world/r2_learned_bias.py` machinery" references a file not on `main`. Either merge
  it first (it carries all the inherited guards) or accept that every v1 guard is a fresh build, not a
  reuse — and re-review it as new code.
- **N2. Low per-seed variance may leave the permutation test underpowered.** The selector is
  deterministic and cluster credit is SIGN-only (±1, not magnitude), so most seed-to-seed variation is
  squeezed out; per-seed gaps may be near-identical, giving the one-sided permutation test little
  power (or a degenerate p). Name, at freeze, what legitimately varies across seeds (drain end-food
  jitter, world-state at encode) and confirm it actually moves the interoception cluster / the argmax —
  otherwise N is measuring the same number 20 times.
- **N3. Confirm `fd0ae83c` is the INTEROCEPTION cluster.** Because `food` is dual-modality, the
  prereg's shared/different cluster ids must be the interoception ones (where relief routes), not the
  world-channel ones (which carry zero eat bias under relief-only). State the modality alongside the
  id in the pre-freeze disclosure.
- **N4. Satiated arm executes eat-while-full.** The prereg has the SATIATED arm run eat while satiated
  ("eat gives no relief"); at food ≥ satisfaction the bridge `eat` may mechanically fail → a NEGATIVE
  eat causal link unique to that arm. Since the satiated arm is not in the primary (see reduction),
  this is cosmetic — but it is another reason the satiated arm earns its keep only as an explicit
  control, not as scaffolding.

## Bottom line

The seams are right and inherited cleanly; the blocking problems are a training *schedule* that
recreates the v1 forced null (D1) and a titration *rationale* that the deterministic selector cannot
produce (S2). Fold D1 + S1 + S4 + S5 (the balance/headroom cluster) and the design measures the real
composition — which, honestly stated, is the v1 marginal-flip-at-food-11 over a balanced competitor
roster. Decide whether the K titration survives S2 before spending arms × K × N compute on a curve the
instrument may render flat.
