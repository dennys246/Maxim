# pymaxim 1.1.4 — "The world seam"

**Released 2026-09-04.** `pip install --upgrade pymaxim`

1.1.3 was about mechanisms that existed and could not be reached. 1.1.4 builds the first seam
where Maxim's substrate meets a world it does not control: a Minecraft server, real hostiles,
real timing — and it ships the encoding change that lets a body-scale sensor set register events
at all. Infrastructure only, **no behavioral claim**; the claims come in 1.2, against this
apparatus.

## The world seam itself

- **Minecraft bridge** (`scripts/minecraft_bridge/` + `simulation/minecraft.py`): NDJSON over
  TCP, protocol frozen in the module docstring. The honesty contract is the Reachy one:
  a refusal is a *failure*, a timeout is *UNKNOWN* — reported as success with neutral valence,
  never as a lie in either direction.
- **`bodies/minecraft_player.yaml`** — 16 `modality: world` sensors with ranges **re-centered so
  resting values sit at the encoding's neutral point**. This is not cosmetic: rest-at-extreme
  ranges were measured structurally blind pre-freeze (event cosine 0.926 vs 0.747 re-centered).
  The range declaration is part of the design under test.
- **The `world` ModalityChannel** + selection-dynamics re-baseline; inert-until-declared for
  every existing body (swept in CI).
- **Two-AUT-one-world harness** (`simulation/minecraft_harness.py`): one server, two substrate
  agents, a liveness-gated smoke verdict (`verdict_is_green` requires live feed writes — a
  review round *demonstrated* the gate passing with the pumps never started, so the gate now
  refuses a dead feed and CI carries the negative control).

## A4, shipped the measured way

The roadmap's original encoding prescription (channels + scaled threshold) was measured **worst**
before a line of it shipped — the bake-off exists so a roadmap sentence cannot instruct anyone to
rebuild a known barrier. What shipped instead: the **nonlinear gain (A4)** at the **unchanged
0.85 threshold**, **world-only** — at N=6 A4's stability collapses (0.62), so interoception and
audio stay byte-identical and Exp 53b never re-staled. The scan-cost prerequisite was measured
*first* (verdict: index-prerequisite), so A4 rides a vectorized exact scan (0.31 ms p95 at a
10k-node store) that is decision-equivalent to the reference scan by test.

## The re-measure — the verdict the release hinged on

The L11 dilution limit's re-measure ran as a **frozen, lint-governed pre-registration**
(protocol merged to main before the first data timestamp; verdict computed by the protocol's own
decision function). On 10 minutes of live world data at N=16:

| | A0 (the shipped limit) | A4 |
|---|---|---|
| separation | **0.0 — fully blind** | **0.0566** |
| clusters | 1 | 3 |
| stability | 1.0 (vacuous) | 0.9984 |

**Verdict: `mitigation-confirmed`, NOT `retired-eligible`.** A4 restores real but weak
separation; 0.057 is nowhere near the 0.70 retirement bar. L11 stays ACTIVE with A4 as partial
mitigation — the honest outcome, pre-committed before the data existed.

## Console hardening, same window

- **Bearer auth, always on, fail-closed** (#613): every `/api/*`, `/docs`, `/openapi.json` and
  `/ws` requires the console token; `GET /api/hello` is the one tokenless surface; browser `/ws`
  carries the token as a subprotocol. Contract 0.4.0.
- **Reachy device handoff** (#616, decision A9): on a robot with no terminal, the sign-in URL
  rides Pollen's own `custom_app_url` dashboard link as a `/#token=` fragment
  (*errata, same day: the vendor daemon reads `custom_app_url` by regex over `main.py` at
  list time, so the link cannot carry a boot-minted token on-device — superseded by
  amendment A9.1, spoken-code pairing, in the next release*);
  `build_app(extra_trusted_origins=…)` admits the LAN bind to the trust guard. The two-lens
  review earned its keep: both lenses independently found that sandbox mode composed with the
  new parameter into a tokenless LAN console — refused at build time before merge.

## Defects, counted

Six found and fixed en route (D68, D76, D77, D78, two pre-commit); D79 filed and OPEN
(structural fix owed); D51's scope corrected twice — the unindexed scan, not `LSHIndex`, was the
prerequisite; D67 resolved. Every PR went through a two-lens pre-merge round with folds landed on
the merge target.
