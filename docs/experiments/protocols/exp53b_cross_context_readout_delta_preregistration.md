# Exp 53b — cross-context readout, re-run with the robot's own step size (pre-registration)

**Status:** PRE-REGISTERED 2026-08-26, frozen before any 53b record; same robot session as
Exp 53. **Inherits everything from
[exp53_cross_context_readout_preregistration.md](exp53_cross_context_readout_preregistration.md)**
(agents, targets as amended, Gate I, Gate T, stop rules, S1–S8, amendments 1–2) **with
exactly one declared change.**

## Why a re-run (the Exp 53 APPARATUS finding, disclosed in full)

Exp 53's Phase 2 primary block returned the pre-registered **APPARATUS** verdict: taught
delivered directedness 0.75 / 0.75 / 0.75 with the chosen direction correct on **36/36**
gated trials (satiated 0.00, no action 36/36; no_feed 0.50, side-blind `turn_right`).
Every one of the nine taught misses was the **−0.2 target** — delivered at −0.12…−0.13,
where the declared δ = 0.55 rad step (≈ +0.30 az) lands at +0.17…+0.19, so
`|after| > |before|` (by target: −0.3 → 9/9, −0.2 → 0/9, +0.5 → 9/9, +0.6 → 9/9). The
sign-rule agreement with the delivered measure was 27/36 = 0.75 < 0.80 and three seeds
sat at exactly 0.75, so both apparatus clauses fired. Exploratory placements: −0.6 →
9/9 toward; +0.2 → 9/9 wrong-way, as amendment 1 predicted. The learned *direction*
transferred; the step size I declared did not fit the small target.

## The one change

**δ = 0.30 rad** — the `bodies/reachy_mini` body's own `turn_left` / `turn_right`
`self_effect: head_yaw` (the production magnitude, ≈ 0.17 az delivered at the H1 gain),
in place of 0.55 rad. Not tuned to the data: it is the step the robot's own body declares.
At the azimuths Exp 53 actually delivered (−0.13, −0.22, +0.56, +0.67) a 0.17-az step
toward centre satisfies `|after| < |before| − 0.05` at every gated target.

Nothing else moves: agents (taught 42/43/44; satiated + no_feed 42/43/44; taught 48
exploratory — the SAME files, SHA-verified, now committed under
`docs/experiments/data/53_agents/`), targets {−0.3, −0.2, +0.5, +0.6} + exploratory
{−0.6, +0.2}, Gate I, Gate T (LEARNED-LIVE ≥ 0.70, margins ≥ 0.20, sign-rule agreement
≥ 0.80, seed spread below ceiling), explore 0.0 primary / 1.5 secondary, the speech-gate
floor 0.50 / 30 s, δ frozen after the first record, run once, stop rule I.

Phase 1 is re-run (cheap, no motion) so the instrument check is recorded under the
same δ; Phase 2 refuses to run without its PASS record in the 53b file.

## Outcome tree

| result | verdict | `1.1.0` |
|---|---|---|
| Gate I pass, Gate T pass | **cross-context transfer EARNED** (Exp 53 recorded beside it as the apparatus finding that motivated δ) | ships with the claim + video with controls |
| Gate I pass, Gate T fail | FAIL recorded | ships sim-only claim, fail named |
| APPARATUS again | second hardware finding; **no third run in this session** — a D-number and a separate plan | ships sim-only claim |
| Gate I fail | instrument stop | ships as-is |

**Stop rule (additional):** 53b runs once. An APPARATUS verdict from a different cause
than Exp 53's is a new finding, not a second retune; the session ends.

## Runbook

```bash
export PYTHONPATH="$PWD/src"
python scripts/orient_backbone/exp53_cross_context_readout.py run --host 10.6.0.63 --yes --delta 0.30 \
    --manifest docs/experiments/data/53_agents_manifest.json --phase 1 \
    --out docs/experiments/data/53b_cross_context_readout.jsonl
python scripts/orient_backbone/exp53_cross_context_readout.py run --host 10.6.0.63 --yes --delta 0.30 \
    --manifest docs/experiments/data/53_agents_manifest.json --phase 2 --condition primary \
    --out docs/experiments/data/53b_cross_context_readout.jsonl
python scripts/orient_backbone/exp53_cross_context_readout.py verdict \
    --records docs/experiments/data/53b_cross_context_readout.jsonl
```

## Sign-off

1. Read and accepted with the one change — ☑ 2026-08-26 (operator: "let's do it")
2. Exp 53's APPARATUS record committed unmodified beside this — ☑
