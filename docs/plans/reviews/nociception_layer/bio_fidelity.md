# Bio-fidelity lens — nociception_layer.md (+ step 1 diff)

**Verdict: ADOPT WITH FIXES.** A per-signal kind is faithful, and so is consumer-declared
consumption. Pain is not one signal in the brain. Four things need fixing: two labels in the taxonomy
misclassify, the fix for F1 drops a learning signal that biology keeps, and "one adapted value for
every consumer" merges pathways the nervous system keeps separate.

## DO-NOT-ADOPT

None.

## SHOULD-FIX

**S1. F1's fix should turn anticipation into a prediction error, not just delete it (step 4).**
Paying out `-anticipated` as reward is a real defect, and it is worse than double counting. The
intensity from `PerceivedPain` comes from NAc's own learned confidence. Feeding it back as a negative
outcome makes the prediction confirm itself, with no path to extinction: the more NAc predicts pain,
the more it is punished, so the more it predicts. This matches the maladaptive loop in fear
incubation, where avoidance prevents disconfirmation. In temporal-difference terms, a prediction
enters learning as a *change* in value (γV(s′) − V(s)), never as the reward r.

Dropping ANTICIPATORY from reward altogether goes too far. Two-factor theory of avoidance (Mowrer):
Pavlovian fear attaches to the cue, and the avoidance response is instrumentally reinforced by fear
*reduction*. That reduction is the relief, or safety signal, when an expected aversive outcome does
not arrive, a positive prediction error. A rising predicted threat also works as conditioned
punishment of the act that led into it.

Recommendation: in step 4, (a) never pay the prediction level as reward, which is the plan's current
minimum and correct, and (b) state which signal carries anticipation instead. The option is the
signed change in anticipated pain across an action: a rise punishes approach, and a fall (relief)
reinforces the avoiding act. If (b) stays unbuilt, record it as a known gap and explain why Exp 60's
avoidance does not need it (that avoidance rides Wire-4 cluster fear, not this path). Otherwise
"drop it" can read as "anticipation has no role in learning".

**S2. `drive:health` is not uniformly nociception.** Health is a damage *proxy*; biology has no
health transducer. When health falls from a hit or a fall, calling it nociceptive is fair. When it
falls from drowning or starvation, it is not: hypoxic tissue injury is famously close to painless,
and what feels aversive in drowning is air hunger (the CO2/respiratory drive), which the plan
classifies as DRIVE. So in the drowning case the memory "pain" label lands on the painless channel
and misses the aversive one. Recommendation: label the health exception an *engineering proxy for
injury*, not nociception. Where the game reports why damage happened, carry that cause on the signal
so a later revision can split them. Do not change Exp 60–62's unconditioned stimulus.

**S3. Excluding air hunger from pain is defensible, but the plan should say what keeps it in
memory.** Dyspnea shares the affective-motivational circuitry of pain: insula, anterior cingulate
and amygdala. Its unpleasantness can be separated from its intensity, as with pain. The brain encodes
suffocation episodes strongly, but through arousal-driven modulation of consolidation by the amygdala
and noradrenaline, not through nociceptors. The design keeps this *if* drive pressure really reaches
the encoding tag during a breach, and it appears to (relevance-gated `corrective_need_intensity`).
Recommendation: make that the stated rationale. Decision 4 is "the `pain` field is nociception", not
"aversive interoception does not strengthen memory". Pin the claim with a check that a drowning
capture's tag is raised by pressure.

**S4. "Consumers say what they take" should also cover *which stage* of the signal they read (steps
3 and 5).** The nervous system runs parallel pain pathways:

- the lateral, sensory-discriminative path (to VPL and S1/S2), which says where and how strong;
- the medial, affective-motivational path (to the medial thalamus, ACC and insula), which carries
  unpleasantness;
- the spinoparabrachial path to the amygdala, the unconditioned-stimulus route for fear conditioning;
- spinal withdrawal reflexes, which act before any perception.

Adaptation is also layered. Peripheral sensitization happens at the nociceptor terminal, central
sensitization (wind-up) in the dorsal horn, and top-down modulation runs through the PAG and RVM,
which covers expectation, placebo and fear-induced analgesia. It can change unpleasantness separately
from intensity. So principle 3, "memory, NAc, fear and reflexes see one adapted value", is less
faithful than the kind split it sits beside. Recommendation: step 5 keeps the peripheral/spinal gain
(habituation, sensitization) at the producer, and it can remain one number. Anything driven by
expectation or context is a separate modulation that a consumer declares, the way it declares kinds.
Reflexes then read the spinal-stage value, and fear and memory read the affective one. reflex_layering
3a already says expectation's sign is not fixed, and this is where that belongs.

**S5. `SAFETY_VIOLATION` is classified NOCICEPTIVE, but its own enum comment says "FearAgent-detected
threat".** A detected threat is anticipatory, not tissue damage. It has no producer today, so this is
latent. Reclassify it as ANTICIPATORY, or document it as a physical-limit breach, before anything
publishes it.

## NIT

- **FRUSTRATION** is faithful as a separate class: Amsel's frustrative nonreward says that when an
  expected reward is omitted, the result is aversive and arousing, and it is learned from. But its
  biological magnitude is the negative reward-prediction error (the dopamine dip at omission), not a
  fixed intensity. The memory plan already routes it through |RPE|, so say in the docstring that
  FRUSTRATION's `intensity` is not its learning signal.
- **COGNITIVE_OVERLOAD** fits better under effort cost or fatigue (ACC effort valuation) than under
  frustration.
- **EXHAUSTION** overlaps DRIVE. Biological fatigue is interoceptive and homeostatic, and bodies
  already declare an entropic `fatigue` drive that classifies as DRIVE. Say in the docstring that
  EXHAUSTION means a *compute/energy budget* (a non-biological resource), so there are not two kinds
  for one physiology.
- **MOVEMENT_FAILURE** ("commanded but didn't reach target") is a sensorimotor prediction error, the
  cerebellum's territory. It is only nociceptive when the cause is obstruction or strain, which
  SUSTAINED_STRAIN already covers. Worth a note.
- Minecraft's air bar tracks O2. Real air hunger is driven mainly by CO2: hypoxia without hypercapnia
  produces little dyspnea. This is a game-native (D1) constraint, not a defect, but record it for any
  "air hunger" claim.

## Verified fine

- Separating nociception from homeostatic drive signals matches the split between interoception and
  exteroceptive nociception. Keeping the drive signal as `extra["drive_pain"]` loses nothing, since
  it is not discarded.
- Fear's unconditioned stimulus `{drive:health, drive:oxygen}` is faithful. The parabrachial alarm
  hub, which relays the unconditioned stimulus to the amygdala, converges nociceptive and
  interoceptive threats, and CO2/suffocation is a potent innate fear stimulus (the suffocation
  false-alarm account of panic). Including a DRIVE in fear while excluding it from `pain` is exactly
  the consumer-specific rule the plan argues for.
- Keeping ANTICIPATORY out of the memory `pain` field is correct as an engineering guard against a
  self-reinforcing loop. Fear does enhance encoding in biology, but through arousal, which the tag
  scores separately.
- Two-directional adaptation that depends on damage (habituation vs sensitization), with a floor
  above zero and recovery on the experience clock, is faithful. So is stopping the reflex from
  adapting the stimulus.
- `classify_pain` raising on an unclassified type is the right tier-1 behaviour.
