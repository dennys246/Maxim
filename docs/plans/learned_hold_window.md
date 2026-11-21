# Learned hold window — tuning when the substrate waits for deliberation (v2 of the hybrid's hold)

> **PROPOSED 2026-11-12 — plan stub only, no code. Split out of [hybrid_arbitration.md](hybrid_arbitration.md)
> §A.1b by owner decision on 2026-11-12.** It starts only after the hybrid's v1 hold (frozen per world) has produced
> evidence. It enters as its own `[engineering]` mechanism, with a **learned** behaviour tier and its own experiment.

## What it is

In v1 the hybrid's hold *h* is fixed: frozen per world in the prereg, and scaled by conflict. Under v2 the agent
tunes *h* itself, so the hold adapts per environment and per SEM component. This answers the owner's request
(2026-11-12) to "evolve a system for fine tuning" the pre-emption window.

## The rule to start from

The rule comes from Keramati, Dezfouli & Piray (2011, *PLoS Comput Biol* 7(5):e1002055) and from Pezzulo, Rigoli &
Chersi (2013, *Front Psychol* 4:92). Deliberation is worth its time when the value of information beats the cost of
waiting, where that cost is deliberation time × the average reward rate. The two inputs map onto Maxim like this:
- **Value of information:** estimated from the substrate's uncertainty, meaning its margin and its recent RPE.
- **Waiting cost:** the world's hazard and reward rate, the backend's action duration, and the observed LLM latency.

In those models, the agent chooses between acting and deliberating. In the hybrid, it acts *and* deliberates. So v2
learns *how long to hold*, not whether to deliberate.

## Not the base to reuse

`runtime/gating.py::AdaptiveThresholdController` is homeostatic. It steers toward a fixed escalation rate, has no
reward term, and runs only in DN, so it does not fit as the starting point.

## Gates

- The hybrid's v1 portfolio member has run, and its hold time and outcomes are reported per decision.
- A four-lens design review is run before any harness.
- **The safety floor still applies:** no learned *h* may delay a danger-selected or protective action
  (hybrid_arbitration §A.1d).
