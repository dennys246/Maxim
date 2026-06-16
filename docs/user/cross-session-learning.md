# Cross-Session Learning

## Overview

Maxim's headline differentiator over a stateless LLM is that it **carries experience across sessions without fine-tuning.** When an agent touches a hazard and feels pain, that experience is written to disk as causal links and reward biases. The next time you run the agent — a fresh process, a new session — it recalls the association and surfaces it to the LLM as experience-grounded context. No gradient updates, no retraining: the experience lives in the bio-system substrate (Hippocampus episodes, NAc causal links and reward biases, EC/ATL concept nodes) that persists between runs.

> **What "learning" means here (read this first).** The substrate **persistence** is real, measurable, and the earned 1.0 claim: memories and causal links carry across sessions and the resumed agent demonstrably *recalls* the prior experience. Whether that recalled context then **changes what the agent does** is a separate question — and the honest 1.0 answer is that a strong LLM's prior often dominates the carried signal (shown across four frontier models in the [1.0 release findings](../announcements/maxim_1_0_release.md)). So this guide shows you how to *make the carried experience visible and measurable* (the `Pain:`-count delta, the restored-substrate logs, `maxim roy diff`); treat any behavioral shift you observe as the LLM choosing to act on that context, not as a guarantee.

This guide walks through the loop end to end:

1. Install the dependencies that make recall high-quality.
2. Run a goal-specific simulation where the agent records that a hazard causes pain.
3. Inspect what was recorded using `maxim roy diff`.
4. Resume into a second session and measure what carried over.

## Step 0 — Install for full memory quality

Cross-session learning works on a bare install, but the memory and substrate systems quietly fall back to a bag-of-words hash embedding when `sentence-transformers` is absent. That fallback severely degrades concept matching, which is exactly what cross-session recall depends on. For the real thing, install the `semantic` extra:

```bash
pip install 'pymaxim[all,semantic]'
python -m spacy download en_core_web_sm
```

> **Note:** `[all]` does **not** include `[semantic]` (it pulls in `sentence-transformers` + `torch` + `spacy`, which are large). Install both extras together as shown above, or memory features silently use the lower-quality fallback.

You also need an LLM backend. A local model is fine for this walkthrough:

```bash
pip install 'pymaxim[llm-llama,llm-server]'
```

## Step 1 — Run a session where the agent learns

The substrate path that records concept-level reward biases is opt-in. Enable it with `MAXIM_SUBSTRATE_PATH=1`. We use the bundled `weapons/rusty_sword` embodiment fixture, which exposes affordances that publish pain through the SEM cascade when the agent handles the hazard:

```bash
MAXIM_SUBSTRATE_PATH=1 maxim --sim "learn that gripping the rusty blade causes pain" \
  --embodiment weapons/rusty_sword \
  --interactive false \
  --sim-max-turns 12
```

Notes:

- `--interactive false` is required for scripted / non-TTY runs (interactive mode is on by default in a terminal and its raw stdin reader conflicts with automation).
- Keep the goal specific. `"learn that gripping the rusty blade causes pain"` converges far faster than `"test combat"`.
- `--sim-max-turns` caps the run; 10–15 turns is plenty to form the association.

When the run finishes, Maxim prints a report and the saved session directory:

```text
SIMULATION REPORT — sim_20260606_141203_a1b2
  Goal: learn that gripping the rusty blade causes pain
  Persona: default
  Model: mistral-7b (llama-cpp)
  Duration: 84.2s | Turns: 12
  Finish: goal_reached
  Actions: 18 (2 blocked)
  AUT Memories: 14 | Causal Links: 6
  Bio events: ▲ Learn: 9 | Pain: 3
  ...

  Session: ~/.maxim/sim_reports/sim_20260606_141203_a1b2
```

The `AUT Memories`, `Causal Links`, and `Bio events` lines are your first signal that learning happened. `Pain: 3` means the pain cascade fired three times; `▲ Learn: 9` counts substrate-weight changes and tier promotions. Note the **session id** (`sim_20260606_141203_a1b2` here) — you'll need it for the diff.

### What gets written to disk

Each session directory under `~/.maxim/sim_reports/{session_id}/` contains:

| File | Contents |
|---|---|
| `report.json` | The full machine-readable report (the same fields shown above). |
| `actions.jsonl` | Per-turn action log. |
| `aut_hippocampus.json` | Episodic memories formed during the session. |
| `aut_nac.json` | Causal links and reward biases (the cross-session learning payload). |
| `aut_ec.json` | EC substrate nodes (written when the substrate path is enabled). |
| `aut_atl.json` | ATL semantic concepts (written when the substrate path is enabled). |

`aut_nac.json` is the file that carries the learned associations forward. `aut_hippocampus.json` carries the episodic record.

## Step 2 — Run a baseline session for comparison

To *prove* learning rather than assume it, run a second session as a control. The cleanest comparison is a "before" run (a short, neutral session that hasn't touched the hazard) versus the "after" run from Step 1. Run a neutral baseline first:

```bash
MAXIM_SUBSTRATE_PATH=1 maxim --sim "look around and describe the scene" \
  --embodiment weapons/rusty_sword \
  --interactive false \
  --sim-max-turns 6
```

Note its session id too (e.g. `sim_20260606_140510_z9y8`). Because the agent never gripped the blade here, its NAc should hold little or no pain-associated reward bias for the hazard — which is exactly what makes it a good baseline.

## Step 3 — Inspect what was learned with `maxim roy diff`

`maxim roy diff` compares the substrate of two session directories and reports where they diverge across NAc, EC, Hippocampus, and ATL. Pass two session ids (resolved against `~/.maxim/sim_reports/`) or two paths:

```bash
maxim roy diff sim_20260606_140510_z9y8 sim_20260606_141203_a1b2
```

Example output:

```text
Substrate divergence: .../sim_20260606_140510_z9y8 ↔ .../sim_20260606_141203_a1b2

NAc diff:
  reward_bias L2: 0.4187  (4 keys differ)
    grip:rusty_sword: Δ -0.3120
    touch:rusty_sword: Δ -0.1450
  goal_reward_bias L2: 0.0000  (0 keys differ)
  cluster_reward_bias L2: 0.2965  (3 keys differ)
    rusty_sword|grip|tool: Δ -0.2410
  percept_valences: not persisted (reserved for 1.1)
  causal links: a=1  b=6  Δ=-5

EC diff:
  substrate nodes: a=8  b=19  Δ=-11
  modality histogram: text: a=8 b=19 (Δ -11)

Hippocampus diff:
  episodes: a=4  b=14  Δ=-10
  memories: a=4  b=14  Δ=-10
  valence:  mean a=-0.012  mean b=-0.184  KS=0.412  p=0.031
  salience: mean a=+0.090  mean b=+0.301  KS=0.355  p=0.058

ATL diff:
  concepts: a=2  b=9  Δ=+... 
  name overlap: shared=2  only_in_a=0  only_in_b=7  jaccard=0.222
```

For machine-readable output, add `--json`:

```bash
maxim roy diff sim_20260606_140510_z9y8 sim_20260606_141203_a1b2 --json
```

### Reading the signals honestly

Not every number proves learning. Here is what each one actually means:

- **`reward_bias L2`** (NAc) is the most broadly reliable signal. A nonzero L2 with **negative deltas on the hazard's keys** (e.g. `grip:rusty_sword: Δ -0.3120`) means the agent has learned to associate that action with a bad outcome. This is the headline confirmation — it is always present regardless of whether the substrate path was enabled.
- **`causal links`** count delta. The "after" session having more links (`Δ=-5`, meaning side `b` has 5 more than side `a`) shows the agent built new cause→effect structure during the hazard run.
- **`cluster_reward_bias L2`** is the substrate-native learning signal — concept-level reward bias keyed by EC cluster. A **nonzero `cluster_reward_bias_l2`** with negative deltas is the strongest evidence that the *concept* "rusty sword" (not just the literal tool string) acquired a learned aversion that can transfer to paraphrases. **Caveat:** this row only populates when the substrate path is active (`MAXIM_SUBSTRATE_PATH=1`) and the EC cluster pipeline ran. If you see `cluster_reward_bias: not persisted (pre-G4 snapshot)`, the run didn't have the substrate path enabled — fall back to `reward_bias L2` as your confirmation.
- **Hippocampus `valence` mean** shifting negative (`mean b=-0.184` vs `mean a=-0.012`) with a low KS p-value reflects that the hazard session accumulated more negatively-valenced episodes.
- Raw **count deltas** (episodes, memories, EC nodes) confirm activity but, on their own, only show that *more happened* — not that the right thing was learned. Pair them with the reward-bias direction.

The single most honest one-line claim of cross-session learning is: **a nonzero `reward_bias L2` (or `cluster_reward_bias L2`) with negative deltas on the hazard's action keys.**

## Step 4 — Measure what carried over

Persistence is only interesting if you can see it. Resume the learned session into a fresh run with `--resume-sim`, which restores `aut_hippocampus.json` and `aut_nac.json` from the prior session before the new one starts:

```bash
MAXIM_SUBSTRATE_PATH=1 maxim --sim "decide whether to grip the rusty blade again" \
  --embodiment weapons/rusty_sword \
  --resume-sim sim_20260606_141203_a1b2 \
  --interactive false \
  --sim-max-turns 8
```

On restore you'll see a log line confirming the carried-over substrate:

```text
Restored AUT hippocampus from .../aut_hippocampus.json (14 memories)
Restored AUT NAc from .../aut_nac.json (6 links)
```

With the learned aversion in place, the resumed agent recalls that gripping the blade hurt — the restored substrate gives the LLM that context. **If the LLM acts on it**, you'll see it hesitate, examine instead of grip, or choose a safer affordance, and the `Pain:` count in the resumed report drops below the original learning run. That delta — same scenario, fewer self-inflicted pain events — is the carried experience made visible.

Whether the delta appears depends on the model: a strong LLM's prior can override the carried aversion (the [1.0 release findings](../announcements/maxim_1_0_release.md) measured exactly this across four frontier models). The reliable, earned signal is the *recall* itself — the restored memories and causal links, and what `maxim roy diff` shows carried over — not a guaranteed behavioral change. Treat the `Pain:`-count delta as a measurement, not a promise.

## Troubleshooting

- **`AUT Memories: 0` / `Causal Links: 0` in the report.** The substrate path was likely off, or the `[semantic]` extra is missing and the hash-embedding fallback failed to form concepts. Re-run with `MAXIM_SUBSTRATE_PATH=1` and confirm `pip show sentence-transformers` succeeds.
- **`cluster_reward_bias: not persisted (pre-G4 snapshot)` in the diff.** The session was run without `MAXIM_SUBSTRATE_PATH=1`. Use `reward_bias L2` for confirmation instead, or re-run with the substrate path enabled.
- **`maxim roy diff` says `session_a not found`.** Pass a full session id (the directory name under `~/.maxim/sim_reports/`) or an explicit path. The id printed in the report after `Session:` is the one to use.
- **Nothing carried over after `--resume-sim`.** Confirm the restore log lines appeared. If the prior session formed zero causal links, there's nothing to carry forward — go back to Step 1 and verify the learning run actually logged `Pain:` events. (Note: a restored substrate with *no* behavioral change is an expected outcome, not a bug — the LLM prior may dominate the carried signal. Use `maxim roy diff` to confirm the substrate carried; that's the earned claim, independent of behavior.)

## Related

- [Memory](memory-user-guide.md) — what gets remembered and the tier lifecycle
- [Concept Decomposition](concept-decomposition.md) — finer-grained concepts for better cross-session matching
- [Simulation Guide](simulation.md) — running and recording simulated scenarios
- [Configuration](configuration.md) — `MAXIM_SUBSTRATE_PATH` and other environment variables
