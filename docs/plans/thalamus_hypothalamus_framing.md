# Thalamus & hypothalamus — the organizing frame for percept ingress + drive integration

**Status:** Framing note (2026-07-17). Not a build plan — an **organizing model** that explains the
four-facet percept-testbed audit ([percept_testbed_audit.md](percept_testbed_audit.md)) and names
where the missing pieces belong. Companion to the audit; the audit says *what's there and what's
missing*, this says *what those pieces are, biologically, and how they should extend.*

**Discipline up front (this project's own rule):** bio-inspired naming is load-bearing for the mental
model, but *"calling a module Hippocampus doesn't validate it does what a hippocampus does."* New
mechanisms enter as `[engineering]` and graduate to `[behavioral]` only when an experiment earns them.
So this frame is an **architecture-organizing principle**, not a license to build a `Thalamus`/
`Hypothalamus` class hierarchy ahead of need. The thalamic relay stays `[engineering]` until the audio
experiment earns its first slice.

---

## The one-paragraph thesis

Percepts and drives enter cognition through **two organs, not one**. Exteroceptive sensing
(vision, audio, external touch) is relayed and gated by a **thalamus** on its way to cortex (the LLM)
and the substrate. Interoceptive/homeostatic state (temperature, hunger, energy, circadian) is
integrated by a **hypothalamus** into motivational/drive state. The audit's central "problem" — two
separate ingress paths (percept path vs body/drive path) — is **not a bug to unify away; it is correct
neuroanatomy.** The two organs are distinct *and* they communicate (drive state modulates sensory
gain — you attend to what you need). Both already exist in Maxim, fragmented and partly named; the
work is to **recognize, unify, and complete** them, not to invent them.

---

## What's already here (fragmented, and explicitly named)

### The thalamus already exists three times — text/vision-only, relay-to-LLM-only
- **`ThalamicGate`** ([default_network/gate.py](../../src/maxim/default_network/gate.py)) — "gates
  sensory information to cortex… only significant percepts are escalated for LLM deliberation." This is
  the **thalamic reticular nucleus** (the inhibitory gate), but only the escalation direction, and
  vision/DN-centric.
- **`BioEnrichmentPipeline`** ([integration/bio_enrichment.py](../../src/maxim/integration/bio_enrichment.py))
  — its module docstring reads verbatim: *"The thalamic relay for text."*
- **`exec_agent.py`** ([agents/exec_agent.py](../../src/maxim/agents/exec_agent.py)) — "the thalamus →
  PFC path: salient stimuli get automatic [enrichment]… the thalamus routes salient stimuli to PFC."

**The missing part is exactly what the audit found missing:** a **modality-preserving multiplexing
relay** that *all* sensory channels pass through — gated (TRN) and gain-controlled per channel — and
routed to *both* the LLM and the substrate. Today percepts collapse to text one hop past the source
(`sim_adapter.next_observation` flatten), there is no multiplexer, and per-channel gain is discarded.
That missing relay **is the body of the thalamus.** Naming it tells us the fix is **one unifying
relay, not N ad-hoc channel handlers** — the difference between an architecture and a config bolt-on.

### The hypothalamus already exists too — and is partly correctly named
- **Homeostatic/entropic drives** ([embodiment/sem.py](../../src/maxim/embodiment/sem.py)) —
  temperature, hunger, energy: textbook hypothalamic homeostasis. Extended purely by **data** (a
  `HomeostaticDriveSpec`/`EntropicDriveSpec` in the SEM body YAML).
- **Drive-pain integration** — `evaluate_failures` → `_publish_drive_pain` → PainBus → NAc: interoceptive
  breach → motivational signal.
- **The `SCN`** ([time/scn.py](../../src/maxim/time/scn.py)) — the *suprachiasmatic nucleus*, literally a
  hypothalamic structure (the master circadian clock).

So "build a hypothalamus" is mostly **recognize + organize** the drives + SCN + drive-pain as one
system. It's already the extensible, data-driven thing.

---

## Why ablate/scale-for-hypotheses IS thalamic modulation (the convergence)

In neuroscience, measuring a sensory channel's contribution to behavior *is* a thalamic-nucleus lesion
or gain modulation. So the operator goal — "make the sim perfect for testing percept/behavior
hypotheses; ablate and scale interactions" — is not served by a manifest sitting *beside* the
pipeline. **The testbed and the organ are the same object:** per-channel `enabled` = TRN gates the
channel off; per-channel `gain` = thalamic relay amplification (tonic/burst); `modality` = which
nucleus. This is why the frame is diagnostic, not decorative — it collapses "build a testbed" and
"build the thalamus" into one build.

---

## The dual-ingress is neuroanatomy — and the audio case proves it

The audit worried: two ingress paths, should we unify them? **No** — they're two organs:
- **Thalamus:** exteroceptive → cortex(LLM)/substrate (the Percept path).
- **Hypothalamus:** interoceptive/homeostatic → motivational state (the body/drive path).

They **communicate** (hypothalamic drive state modulates thalamic gain). The audio-azimuth
double-representation the earlier reviews fretted over is exactly this dual, and it's bio-plausible,
not a double-count to eliminate:
- **signed EC cluster** = *where the sound is* — exteroceptive spatial (thalamic "where");
- **sign-folded centeredness drive** = *how off-center / uncomfortable I am* — motivational magnitude.

**Honesty caveat:** the hypothalamus mapping is cleanest for the *genuine* homeostatic drives
(temperature/hunger/energy). The centeredness "drive" is really an **orienting/collicular** signal
*wearing homeostatic-drive clothing* — hypothalamic *machinery* used for a collicular *signal* (the
"borrowing the homeostatic shape" note already recorded in `reachy_mini.yaml`). Keep that straight so
the frame stays honest rather than over-fitted.

---

## The three-axis picture (this frame does NOT collide with placement)

Percept handling has **three orthogonal axes** — the same capability-vs-placement orthogonality the
LLM lanes earned:

| Axis | Question | Owner |
|---|---|---|
| **Placement** | *Where* does each pipeline stage run (sensor / GPU leader / substrate owner)? | [perception_pipeline_placement.md](perception_pipeline_placement.md) (active, 1.1) |
| **Thalamic relay / gating** | *Which* channels pass, *how much* (gate/gain), routed to *where in cognition* (LLM vs substrate)? | this frame |
| **Hypothalamic drive** | *What* homeostatic/motivational state do interoceptive signals integrate to? | the drive system (already here) |

So the M0 question is answered: the channel gate/gain surface is **not a collision with, nor an
extension of, the perception-placement plan — it is a sibling third axis.** Placement = where;
thalamus = what/how-much; hypothalamus = motivational state.

---

## Extensibility: data + protocol seams, NOT a deep class hierarchy

The goal is "flexibly extend each." The extensibility this codebase has *proven* — and which is *more*
flexible than an inheritance tree — is **declaration + protocol seams + governed frozen-dataclass
types**, wired through builders. The bio-systems here (Hippocampus, NAc, SCN, EC) are concrete classes
wired via `build_bio_stack`, extended by *declaring*, not subclassing. Thalamus/hypothalamus follow the
same shape:

- **Hypothalamus extends by DATA.** A new drive = a new drive-spec in the SEM body YAML. Already works,
  no subclass. (The perception plan bans the tempting alternative in writing: *"do NOT introduce a new
  AxisSpec type… ride the body YAML."*)
- **Thalamus extends by PROTOCOL + tag.** A new sensory channel = a `PerceptSource` implementation + a
  `SensoryModality` tag + (if it reaches the substrate) an encoding route. Adding a channel = registering
  one, not editing a class tree.
- **Any new *types* follow the CC3 frozen-dataclass discipline** — frozen + `extra` hatch, or
  shape-frozen with a documented reason. Open to additive fields, closed to silent shape drift.
- **The hierarchy ends up shallow-concrete + data, not deep-abstract.** A channel or drive can be added
  without touching the relay code at all.

**The trap this explicitly avoids:** `Thalamus(ABC)` → `AudioNucleus(ThalamicNucleus)`,
`VisionNucleus(...)`, `TouchNucleus(...)`. That is inheritance modeling *data*, and it bakes the first
channel's (audio's) assumptions into the base class — the exact "don't abstract from N=1" mistake this
project has scars from. Nuclei are declarations; channels are data + a protocol impl.

---

## Consequence: "build the CS analogy now" is smaller than it sounds

- **Hypothalamus:** already the extensible data-driven thing (drives + SCN + drive-pain). Mostly
  *recognize + organize*, not build.
- **Thalamus:** the genuinely new build is the **relay/gating body** — the unifying multiplexer that
  consumes channel declarations, **preserves modality (fixes the flatten)**, applies per-channel
  gate/gain, and routes to LLM *and* substrate. Built **incrementally, earned by the audio channel**,
  not as a framework up front.

This *strengthens* the audit's "don't build a side manifest" conclusion: per-channel gate/gain belongs
**in the thalamus** (unify the three existing fragments into a proper relay), not in a parallel config
surface. What looked like a "percept-channel manifest" is really the thalamus's declarative channel set.

---

## Open design fork (decide at the first design pass — do NOT assume)

**Does the escalation-only `ThalamicGate` GROW into the full relay** — becoming the TRN/gating nucleus
of a larger thalamus that also feeds the substrate — **or is the unified relay a new structure that
subsumes all three fragments** (`ThalamicGate` + `BioEnrichmentPipeline` + the exec_agent path) and
demotes the gate to one role inside it? This is a genuine fork with blast-radius implications (the DN
owns `ThalamicGate` today; a thalamus that also feeds the substrate crosses into memory_hub/agent_loop
territory). Flagged for the first design pass; not to be assumed either way here.

Secondary: the thalamus relays to **both** cortex(LLM) and substrate; today the three fragments only
relay to the LLM/PFC, while the substrate path (`SensorEncoder`→EC) is separate. Unifying means the
thalamus feeds both — a real routing decision, mode-aware (llm-primary vs substrate-primary consume
different outputs; see the audit's mode-split facet).

---

## The earning experiment

**"Does the audio-orient percept change behavior, and does scaling its salience/pain matter?"** — the
ablate-and-scale question already on the path (Layer 2 feeds azimuth; Track 2 is the reflex). It
exercises the hardest case (the straddling exteroceptive/interoceptive channel, the mode-split, the
sign-folding landmine), so it validates the frame against the sharpest test instead of the abstract.
Build the minimum thalamic-relay slice that makes *this* experiment clean (plus the audit's M2
active-config record + M3 per-channel telemetry so the impact is *attributable*); the general relay
falls out of serving it.

---

## Related

- [percept_testbed_audit.md](percept_testbed_audit.md) — the four-facet audit this frames.
- [perception_pipeline_placement.md](perception_pipeline_placement.md) — the orthogonal *placement* axis (active, 1.1).
- [embodiment_runtime_wiring.md](embodiment_runtime_wiring.md) / [hybrid_substrate_reflex_runtime.md](hybrid_substrate_reflex_runtime.md) — Track 1 (body wired) / Track 2 (the reflex); the runtime this lands in.
- [substrate_native_orienting.md](substrate_native_orienting.md) — the azimuth "two learning signals" (signed EC state + folded drive reward) the dual-organ split explains.
