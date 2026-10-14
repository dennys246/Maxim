# B1 seed 99 diagnostic (2026-10-13): the liveness ladder before the freeze

The prereg's §Liveness ladder requires one armed diagnostic run before any data. This is that run, on the rig at
`b17b6f21` (main), with the leader down (proxy `absent`). It is seed 99, never pooled into a verdict (rule 12), and
written to a non-gated path. Raw rows: [diagnostic_seed99.jsonl](diagnostic_seed99.jsonl), campaign
`diag-797bcd6291ca`, one attempt per arm, no refusals.

An earlier attempt the same day, campaign `diag-e814874b0f2c`, was refused by the rig check on every arm-seed. The
`maxim-leader` proxy was up and reported no lane metrics. That attempt is not kept; it led to the v5 amendment of
rule 11.

## Per arm

| Arm | First post placement | Post P(surface) | Pre P(surface) | Executed actions to first surfacing |
|---|---|---|---|---|
| FEAR | surfaced, 2.98 s (`flee` fails, then `escape_water`, as in Exp 60 Run 2) | 1.0 | 0.0 | 0 in training (propose-only) |
| RANDOM | censored | 0.17 | 0.17 | n/a |
| Q_EXPLORE | surfaced, 2.55 s | 1.0 | 0.67 | 29 |
| Q_EXPLORE_PRIOR | surfaced, 2.30 s | 1.0 | 1.0 | 16 |
| Q_EXPLORE_RAW | surfaced, 2.59 s | 1.0 | 0.33 | 40 |

- **FEAR:** live G2 passed (water fear −1.0, shore 0.0).
- **Shore roam:** FEAR's post-probe roam made 14 calls, the habit finding; the floors' roams are at chance by
  construction (rule 16).

## The checks the ladder names

- **Identity:** the policy's picks, the telemetry's proposals and the executor's calls are identical in all 12
  placement windows of every floor arm.
- **RANDOM** executed 26 distinct tools.
- **Q updates:** one update per credited in-water pick, with non-zero costs.

  | Arm | Updates | In-water picks | Non-zero reward |
  |---|---|---|---|
  | Q_EXPLORE | 31 | 33 | 21 |
  | Q_EXPLORE_PRIOR | 24 | 24 | 13 |
  | Q_EXPLORE_RAW | 46 | 51 | 38 |

  Every drop is a shore pick (`made_out_of_water`), never a pick that failed to execute. So the next-pass credit
  fix holds on the live loop.
- **Hygiene:** no policy violations; one tool set of 26 tools per arm. `turn`, `place_block` and `mine_block`
  failed on `params={}` before the bridge.
- **FEAR's shore-start check (rule 17):** its first post escape began at 1.58 s, after its first in-water read at
  0.40 s.
- **Rig check:** proxy `absent`, `tick query` parsed to TPS 20.0, no problems.

## What it settled for the run

- **Escape timing:** a successful `escape_water` call takes a median of 1.83 s (1.6–1.9 s). Surfacing lands
  2.3–2.6 s after the teleport. The toy model (`learner_sim.py`) assumed about 2.6 s.
- **Water keys:** the live EC did not split the water state. Q_EXPLORE saw one water key; RAW saw two, its oxygen
  bins.
- **Where the Q arms learn:** in the greedy pre-probe, as the confounding review predicted. Training then
  surfaced on all 10 episodes in every Q arm.
