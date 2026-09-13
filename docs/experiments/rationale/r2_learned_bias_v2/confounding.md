# Confounding lens — R2 learned-bias v2 pre-registration

**VERDICT: DO-NOT-BUILD as framed.** The primary does not isolate the *claimed* cause. The
"state-contingency" (eat-when-hungry-not-satiated) the claim advertises is **structurally inert** in
the primary: because the satiated probe (food 18) is a *different EC cluster* than the training
cluster, the cluster bias is 0 there in **both** arms, so the satiated term cancels identically and
`isolated_effect(K)` collapses to the v1 single-state marginal cluster probe at food 11 — now with
competitors and a `K` axis. That is a legitimate experiment, but it measures **cluster-contingent
competitive selection**, not **hunger-contingent** selection. Reframe the claim (drop the
satiety-contrast pretense) or add a genuine dissociation probe before building; several SHOULD-FIXes
below also gate a trustworthy null.

Verified against: `nac.py::recommend_action` (cluster-bias lookup is exact-key over the *encoded*
cluster set, line ~2080 — a non-training cluster returns 0.0; causal/prior/cluster all add directly
to one argmax score), `substrate-learning-channels.md` (the two-channel finding), the v1 prereg +
its pre-data amendment, and `bio-memory.md` (R2 break-1/break-2 invariants, cluster keying).

---

## DO-NOT-BUILD

### D1. The satiated probe is structurally inert → the "state-contingency gap" is an illusion; the primary is a single-state marginal probe, and the claim it is sold as is untestable in this substrate.

**The flaw.** Score in `recommend_action` is a single argmax over `{eat} ∪ competitors`:

```
score_eat(state)   = prior_eat(state) + causal_eat + cluster_bias(cluster(state), eat)
score_comp_j(state)= causal_cj                         (non-corrective ⇒ no prior; relief-only ⇒ no cluster bias)
```

`cluster_bias(cluster(state), eat)` is an **exact-key** lookup over the *encoded* cluster set
(confirmed at `nac.py:2080`). The bias formed on the food≤4 training cluster, which **food 11 shares**
and **food 18 does not**. So:

- At the satiated probe (food 18): cluster_bias = 0 in **LEARNING and NO-CREDIT alike** ⇒
  `P_L(18) = P_NC(18)`.
- Expand the primary:
  `isolated(K) = [P_L(11)−P_L(18)] − [P_NC(11)−P_NC(18)] = [P_L(11)−P_NC(11)] − [P_L(18)−P_NC(18)]`.
  The second bracket is **0 by construction**. So `isolated(K) ≈ P_L(11) − P_NC(11)` — the marginal
  cluster flip at **food 11 alone**. The satiated probe contributes nothing to the isolation.

**Why it matters (the claim vs. what's measured).** The headline claim is *state-contingent*
selection: "eat when hungry, **not** when satiated." A behavioural demonstration of that would need
the *same* cluster bias to be **consulted at both states but win only at the hungry one**. It cannot
happen here: in this substrate **cluster ≡ drive state** (food value → derived hunger → EC cluster),
so the bias is only ever *consulted* (non-zero) at its own — hungry — cluster. State-contingency is
therefore inherited **by construction** from cluster-keying, not an emergent selection property, and
is untestable with this metric. What the primary *does* legitimately test is: does the drive-relief
cluster bias produce a **competitive flip** at the trained cluster (beat K equally-causal-linked
alternatives), and does that flip scale with `K`.

**Failure scenario.** The run yields a clean positive `isolated(K*)` and the outcome reports
"state-contingent competitive selection demonstrated." A reader (or a later meta-analysis) notes the
positive is *identical* to what a purely cluster-keyed recall with **no hunger semantics** would
produce — any credit keyed to the cluster active at reward time gives this result whether or not it
"knows about" the drive. The satiety contrast was decorative. This is a false/overclaimed result of
exactly the retraction-risk shape the design review exists to stop (cf. the v1 causal-link and
floor-into-cluster confounds).

**Fix (pick one, before freeze):**
1. **Reframe the claim** to "the drive-relief cluster credit produces **cluster-contingent
   competitive selection that scales with K**," drop `P(eat|satiated)` from the *primary*, and make
   the primary explicitly `P_L(11) − P_NC(11)` at each `K` (the marginal flip under competition). Keep
   the satiated probe only as a reported sanity check (it should read ~0 in both arms — if it doesn't,
   the substrate is bleeding bias across clusters and *that's* a finding). This is the honest,
   buildable version.
2. **OR add a real dissociation probe** that separates cluster-membership from satiety — e.g. a
   *hungry* state in a *different* cluster (does the flip fail to transfer? = the generalization
   limit, already acknowledged) **and**, if any state can be constructed that shares food-11's cluster
   while being satiety-labelled otherwise, probe it. If no such state exists (likely — cluster is a
   function of the drive value), that impossibility must be stated plainly in the prereg as the reason
   strong state-contingency is not claimed.

Do not build-and-report the state-contingency framing without one of these.

---

## SHOULD-FIX

### S1. No run-time headroom/saturation gate on NO-CREDIT — the v1 causal-link-saturation trap is a *pilot hope*, not a confirmatory refusal.

**The flaw.** The whole isolation depends on eat's **causal link not already winning** at food 11
against the competitors. If it does (`P_NC(11)=1`), then `isolated = P_L(11)−P_NC(11) = 0` — a
**forced null**, which is precisely the v1 failure (`substrate-learning-channels.md`: the state-blind
causal link "gets there first"). The design's only guard is (a) balanced round-robin training and (b)
the raw-P(eat) secondary — but the secondary is **not in the decision rule**, and the pilot only
*locates* the effect. The confirmatory run has no gate that refuses/flags a saturated baseline.

**Failure scenario.** Balanced training under-equalizes the causal links at the confirmatory `N`
(more variance than the `N≈5` pilot showed); `P_NC(11)` sits near 1.0 at `K*`; `isolated≈0`; the run
ships **PREMISE-NULL**. That null is uninformative (credit-doesn't-work vs. no-headroom are
indistinguishable) but reads as evidence against the claim.

**Fix.** Add a pre-registered **validity gate** mirroring v1's "instrument still holds": at each `K`,
refuse or flag the cell if `P_NC(eat|hungry, K)` exceeds a pre-registered ceiling (e.g. > 0.8) —
headroom must exist *at run time*, not just in the pilot. Classify a null accompanied by saturation as
**REFUSED-UNVERIFIED**, never PREMISE-NULL. This makes S1 and the "headroom didn't grow" ambiguity
(the charter's non-claim-null risk) a mechanical refusal rather than a post-hoc judgement.

### S2. Training must be a *scripted fixed round-robin*, or eat-success counts (hence causal-link strength) diverge across arms and contaminate `isolated_effect`.

**The flaw.** `isolated = gap_L − gap_NC` cancels the causal link **only if eat's causal link is equal
in both arms**, i.e. only if the number of successful eats during training is identical. If training
selection is *policy-driven* (via `recommend_action`), then in LEARNING the growing cluster bias makes
the agent eat *more* (or pick competitors less) than in NO-CREDIT → eat's causal link ends up
**stronger in LEARNING** → part of the measured `isolated_effect` is a causal-link asymmetry, **not**
the cluster credit. The prereg implies a fixed schedule ("`eat, c1, eat, c2, …`") but does not state
that training bypasses the policy.

**Failure scenario.** Training uses the AUT's own selection "for realism"; LEARNING accrues 1.3× the
eat-successes of NO-CREDIT; a positive `isolated_effect` is booked as the credit when a chunk of it is
the extra causal link. False positive.

**Fix.** Pre-register that training action selection is a **scripted round-robin** (not
`recommend_action`), identical across all arms by construction, and **assert eat-success counts (and
competitor-success counts) are equal across arms per seed** as a validity gate. The cluster credit is
still the only cross-arm difference only if the causal substrate is byte-matched.

### S3. Dose-response rule 2 ("non-decreasing with K") is untestable-as-written and wrongly conjoined with rule 1.

**The flaw.** Two problems. (a) At a pilot-informed (small) `N` with **binary per-seed gaps**, the
`isolated(K)` point estimates carry sampling noise; a strict "does not decrease" point-estimate
criterion will fail on any noise dip and pass on any noise bump — it is not a well-defined test and
has no tolerance or trend statistic. (b) The rule ANDs "isolation at K*" with "non-decreasing," yet
the gloss says "flat-positive still HELD at K\*" and "decreasing → report as such" — so it is unclear
whether a decreasing curve makes the whole thing NULL or merely "held but not scaling." A genuine
isolation with flat scaling is a real partial result, not a null.

**Failure scenario.** A true, replicable isolation at `K*` with a 0.02 noise dip from K=2→K=3 fails
rule 2 → the whole premise reported NULL. Or the reverse: three noisy points happen to ascend and
"dose-response confirmed" is claimed on an order-statistic artifact of competitor causal links (the
`1/(K+1)` headroom grows mechanically with `K` regardless of the credit).

**Fix.** Split into **two independently-reported claims**: **Claim A (isolation)** = rule 1 at `K*`
(gating); **Claim B (scaling)** = a pre-registered **monotone-trend test** across the `K` set
(Jonckheere–Terpstra, or a pre-registered slope with CI), reported with its own p, **not** ANDed into
the isolation verdict. State explicitly that a positive-A / flat-B result is "isolated but not
choice-space-scaling," a real outcome.

### S4. Permutation test — specify it as *paired* (sign-flip per seed), not an unpaired group shuffle.

**The flaw.** Seeds are shared across arms (v1: "same seeds across arms"; each cell a fresh substrate
seeded identically). The natural statistic is the **per-seed** `isolated(seed,K) =
gap_L(seed) − gap_NC(seed)`, tested against 0. The correct null is a **paired** permutation
(sign-flip the per-seed differences, i.e. permute the arm label *within* each seed), not shuffling two
pooled groups. An unpaired shuffle discards the pairing, mis-estimates the null variance, and loses
power. The prereg says only "one-sided permutation on per-seed gaps," which is ambiguous.

**Fix.** Pre-register the paired sign-flip permutation on per-seed `isolated` values (and confirm
seeds are genuinely paired, i.e. same RNG seed → comparable init, despite "fresh substrate per cell").
If seeds are *not* paired across arms, say so and use the unpaired test deliberately — but the current
"same seeds" language implies pairing.

### S5. `K` must be interleaved with seed, or wall-clock drift aliases directly onto the dose-response axis.

**The flaw.** The new axis of this experiment is `K`. The prereg pins "arms interleaved within seed"
(v1 fix) but says "the outer `K` loop" — if all K=1 cells run, then all K=2, etc., any live-world /
time drift is **confounded with K**, manufacturing (or masking) a monotone `isolated(K)` trend — the
exact quantity Claim B rests on.

**Fix.** Interleave `K` with seed too (`for seed: for K: for arm:`) or randomize `K` order within each
seed. Pre-register the loop order.

---

## NIT

### N1. The SATIATED (training) arm is orphaned from the primary.
v1's decision rule used SATIATED as rule 3 (drift/repetition control). v2's rule uses only LEARNING
and NO-CREDIT; SATIATED does no work in the primary. Either drop it to save compute, or state its
reporting role explicitly — it does control one real thing NO-CREDIT does not: whether the *training
process itself* (competitor cycling, or a leaked tool-success floor booking bias at the hungry cluster
even under relief-only) moves the probe with no relief at all. If kept, name that role.

### N2. Carry the v1 credit-source audit into *competitor* training.
The tool-success **floor** also books cluster bias (`substrate-learning-channels.md` trap). Relief-only
enforcement suppresses it for eat, but verify **competitors** (which succeed every round-robin step)
don't accrue floor cluster-bias *at food-11's cluster* while trained hungry — that would lift
competitors at the probe and *suppress* eat's flip (a false null). Pre-register the per-episode
`drive_relief`-vs-`floor` source read for competitors, not just eat.

### N3. Confirm the argmax tie-break isn't a roster-order artifact.
`P(eat)` is a deterministic binary pick over `{eat} ∪ competitors`. With balanced training, eat and
competitors can have near-tied causal scores; if the argmax resolves ties by tool order/name, `P(eat)`
becomes a function of roster ordering rather than the score, and the cluster bias's job (breaking the
tie) is confounded with the tie-break rule. Pre-register that scores are separated enough that the
pick reflects the cluster bias, and that ties (if any) resolve in a way that does not favour `eat` a
priori.

---

## What could still produce a NULL for a non-claim reason (and whether the design distinguishes it)

| Non-claim null cause | Distinguished today? | Gap |
|---|---|---|
| NO-CREDIT causal link saturates at food 11 (v1 trap) | Pilot only, not confirmatory | **S1** — needs a run-time headroom gate |
| Headroom didn't grow with K | raw-P(eat) secondary, but not gating | **S1** — fold into the validity gate |
| Cluster bias formed but didn't transfer to the probe cluster | Acknowledged (generalization limit) + magnitude secondary | OK — honest null |
| Under-powered `N` | Pilot sets `N`; permutation reported | S4 (paired test) improves power |
| Causal-link asymmetry across arms from policy-driven training | **No** | **S2** — scripted round-robin + count-match |

A positive, conversely, is protected against prior (cancels — verified: innate, arm-independent) and
causal link (cancels *within-arm* as a state-blind term, provided S2 holds so it's equal *across*
arms). The one alternative explanation a positive does **not** rule out is D1: cluster-keyed recall
with no genuine hunger-semantics, which the current framing mislabels as state-contingency.
