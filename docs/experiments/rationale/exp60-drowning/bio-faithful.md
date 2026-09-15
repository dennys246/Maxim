# Exp 60 — BIO-FAITHFUL lens (four-lens design review)

**Reviewer lens:** does the design test the Wire-4 situation-fear mechanism's *real job as
it is actually wired*, not a caricature?
**Date:** 2026-09-15 · **Target:** `docs/experiments/exp60_drowning_avoidance_prereg.md` (DRAFT)
**Verdict:** the mechanism fires and its read path is live, but the design rests on a
**drive that does not exist**, and the resulting `drive:health`-only path reintroduces the
Exp-58 dead-read failure at a *different seam* (conditioning-moment cluster ≠ recall-moment
cluster). Two DO-NOT-BUILDs before a harness.

---

## What I verified in the code (facts, not opinion)

- **`oxygen` carries NO drive.** `minecraft_player.yaml::entity.sensors.oxygen` is a plain
  `modality: world` sensor (range [0,40], initial 20). Only `health` and `food` declare a
  `drive:` block. No body in `src/maxim/_data/components/bodies/*.yaml` declares an
  oxygen/air/breath/asphyxia drive.
- **Drive pain's failure_mode is `drive:{drive_name}`** (`body.py::_publish_drive_pain`,
  `context["failure_mode"] = f"drive:{drive_name}"`; `evaluate_failures` emits
  `drive:{ds_name}:discomfort`). Since oxygen has no drive, **oxygen depletion publishes NO
  pain**. Drowning only reaches the pain path as `drive:health`, and only once health breaches
  `comfort_band` (health < 14 hp).
- **`drive:health` is ALREADY in the allowlist** —
  `NACConfig.cluster_fear_failure_modes = frozenset({"drive:health"})` (nac.py:399). The write
  gate is `record_cluster_fear` (nac.py:3136).
- **`health` is `modality: world`** (yaml:30) AND drive-bearing → the world-channel encode at
  pain-time includes low health as well as low oxygen.
- **The Wire-4 READ path is live and NOT the Exp-58 dead path.** `agent_loop.py:1556-1562`
  reads `anticipatory_threat_need` and maxes it into `drives["threat"]`;
  `_DRIVE_TOOL_AFFINITIES["threat"]` (nac.py:574) includes `flee`, and `minecraft_player.yaml`
  exposes the param-free `flee` affordance. The Exp-58 "threat matched zero affordances" defect
  is fixed here.
- **encode is hoisted above the pain tick** (`agent_loop.py:1453` ORDER comment; note at
  1524-1530) — pain published this tick books fear on THIS tick's noted clusters. Co-activation
  at pain-time holds by construction.
- **Pain intensity at breach:** homeostatic `severity = |current-20| - 6`, `intensity =
  min(1, severity·0.5)`. First non-zero at 13 hp (0.5), saturates at 12 hp (1.0). Clears the
  subscriber's 0.3 threshold. `cluster_fear_alpha=0.5` → ~2 publishes saturate `max_cluster_fear=1.0`.
- **No per-tick decay** on `_cluster_fear` (only the slow wall-`decay` sweep, nac.py:3818).

---

## Findings, ranked

### DO-NOT-BUILD 1 — the design rests on an oxygen/air drive that does not exist

**Issue.** The prereg (lines 38-39, 52) treats the drowning failure mode as `drive:oxygen`
"(or the existing air/drowning drive)" and calls adding it to the allowlist "a config addition
to verify." There is no oxygen drive and there is no `drive:oxygen` failure mode — it will
never publish. Drowning reaches Wire-4 only as `drive:health`, which is already allowlisted, so
**the prereg's one concrete config action is both wrong and unnecessary.** More importantly the
doc has not actually decided *what fires the pain*, and that decision changes the entire claim.

**Consequence.** As written, "learned drowning-avoidance driven by fear booked onto the
underwater cluster" is really "learned fear of whatever world-cluster was active the last time
`drive:health` breached." Drowning is one instance among fall/mob/lava/starvation damage —
none of them distinguishable at the failure-mode axis. The headline over-specifies the
mechanism.

**Minimal fix (pick one, name it in the prereg):**
- **(a) Give `oxygen` an air-hunger drive** (homeostatic or entropic-down with a
  `comfort_band`/`deprivation_threshold`), so asphyxia publishes `drive:oxygen` pain *as oxygen
  depletes, before health damage*. This is the **bio-faithful** choice — the air-hunger /
  hypercapnia alarm is one of the strongest aversive drives and fires *before* tissue damage,
  which is exactly the anticipatory signal the claim wants. It also yields a drowning-specific
  failure mode (add `"drive:oxygen"` to `cluster_fear_failure_modes` — *that* is the real config
  line). Cost: a new drive = a mechanism addition; front-gate it in the plan's motivation per
  the design-time scope-pressure rule (existing infra can't do X because Y). Recommended.
- **(b) Explicitly ride on `drive:health`** and rewrite the claim as generic situation-fear
  keyed by cluster, accepting the late-firing / generic-damage semantics below. Then the
  apparatus MUST make drowning the sole `drive:health` source (see SHOULD-FIX 4).

Until this fork is resolved the prereg is not buildable — it currently assumes a wire that
isn't there.

### DO-NOT-BUILD 2 — conditioning-moment cluster ≠ recall-moment cluster (Exp-58 redux at a new seam)

**Issue.** Under option (b), fear is booked *late* — at the first health breach (~13-12 hp),
which is ~4 drowning-damage ticks *after* oxygen already hit 0 (~19s after submersion). At that
instant the world cluster encodes **oxygen≈0 AND health≈12**. But the DV requires anticipation
to fire *early* on a fresh submersion, when the world cluster encodes **oxygen descending, health
= 20 (neutral)**. `anticipatory_threat_need` (nac.py:3165) reads the deepest fear across the
*currently* active clusters; if the early-submersion cluster is a *different EC cluster* from the
one fear was booked onto, it reads **0.0 → no anticipation → the mechanism looks dead**, exactly
Exp 58's outcome, just relocated from the failure-mode axis to the conditioning-vs-recall time
axis.

**Consequence.** The instrument preflight the prereg proposes ("safe-surface cluster ≠
underwater cluster") does not test the load-bearing thing. You can pass that preflight and still
measure a null, because the fear lives on the wrong cluster relative to where the read happens.

**Mitigating fact (measure, don't assume):** at 12 hp, health's A4 gain weight is only
`(|0.3-0.5|·2)³ = 0.064` while oxygen at 0 is `1.0`. So the pain-time world cluster may be
oxygen-dominated and *may* coincide with the early-submersion cluster. This is plausible but
must be MEASURED, not argued.

**Minimal fix.** Add to the preflight (using the L11 Slice-1 live-geometry probe, not a range
argument): verify **cluster(oxygen-low, health-full) == cluster(oxygen-low, health-low)** — the
recall-moment cluster equals the conditioning-moment cluster. Option (a) largely dissolves this
finding, because air-hunger pain fires while health is still full, so fear is booked on the same
oxygen-loud/health-full cluster that recurs at re-submersion.

### SHOULD-FIX 3 — "oxygen swings full-range therefore it separates" repeats the L11 error

**Issue.** The viability argument (lines 28-33) rests on oxygen's full-range gain swing (0→1.0)
implying a distinct cluster. L11 Slice-1 (commit a4e90961, `diluted_present`) proved precisely
that a single sensor's full swing can be diluted below the 0.85 cosine gate by the *constant
mass of the other world sensors* — and `minecraft_player` now carries **17 world sensors**, 15 of
them resting at neutral during a clean submersion. One loud sensor among 16 constant ones is the
exact geometry L11 found *does not* separate.

**Consequence.** The "instrument-check expected to PASS" may fail the same way dark=danger did.
Confidence in separation is currently an assertion, and the verify-the-instrument lesson has
already been paid twice.

**Minimal fix.** Make the separation preflight the actual L11 Slice-1 live re-encode
(`scripts/survival_world/l11_geometry_probe.py` methodology): capture live surfaced vs submerged
`_read_world_states` vectors, run them through the shipped `SensorEncoder` + frozen-centroid EC,
and confirm distinct cluster IDs *and* cos < 0.85 — as a gate, not an expectation.

### SHOULD-FIX 4 — generic-health-fear contamination (only if option (b))

**Issue.** `drive:health` fear books onto whatever world cluster is co-active at *any* health
drop. If the apparatus lets the agent take fall damage entering the pool, mob damage, or any
other injury during conditioning, the "underwater fear" is a mixture.

**Consequence.** Confounds the learned-vs-innate isolation the confounding lens cares about, and
falsifies the claim's wording.

**Minimal fix.** Guarantee drowning is the SOLE `drive:health` source during conditioning
(no-mob, no-fall entry, peaceful difficulty), and word the claim as cluster-keyed situation fear,
not drowning-specific fear. (Under option (a) this finding is moot — `drive:oxygen` is
drowning-specific by construction.)

### NIT 5 — the read is cluster-agnostic while the write is world-only

`anticipatory_threat_need` takes `min` over *all* active clusters' fear
(`clusters.values()`), but `create_pain_cluster_fear_subscriber` only ever writes the `world`
cluster (pain_bus.py:610). Harmless today (no other cluster carries fear), but the read's breadth
is wider than the write's — worth a one-line note so a future intero-cluster fear writer doesn't
silently leak into the threat need.

### NIT 6 — "surface" is not the wired flight action (partly wiring lens)

The prereg frames the corrective act as "surfacing." The wired action is `threat → flee`, and
`minecraft_player`'s `flee` is "retreat to the spawn anchor" (a horizontal fixed-action pattern),
not "swim up for air." If the spawn anchor is not adjacent to air, flee can path *through* more
water and drown mid-flee, or restore no oxygen. Bio-faithfully the mechanism's flight ≠
surfacing; the design must ensure retreat-to-anchor actually restores oxygen (or add a real
vertical `surface` affordance and map `threat` to it). Flagged here because it decides whether
the mechanism's real job is even measurable; the wiring lens should own the execution detail.

---

## Bottom line

The mechanism is genuinely reused and its read path is live (not the Exp-58 dead path). But the
design as drafted cannot be built: it names a nonexistent oxygen drive (DNB-1), and the
`drive:health` fallback relocates Exp-58's null to the conditioning-vs-recall cluster seam
(DNB-2). The **bio-faithful resolution is option (a): give oxygen an air-hunger drive** — it
fires the alarm before tissue damage (true to asphyxia physiology), books fear on the same
oxygen-loud cluster that recurs at re-submersion (dissolving DNB-2), and yields a
drowning-specific failure mode (dissolving SHOULD-FIX 4). Whichever fork, the separation
preflight must be the live L11 geometry probe, not the range argument (SHOULD-FIX 3).
