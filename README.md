# Maxim

A bio-inspired cognitive architecture for AI agents. Maxim gives an agent a **body** (sensors, drives,
pain), **brain-modelled memory** (Hippocampus, NAc, ATL, EC, SCN, Angular Gyrus) and a way to **share
what it learned** with other agents. It runs in two modes:

- **LLM harness.** An LLM chooses the actions; Maxim gives it persistent, agent-specific experience —
  episodic recall, learned causal links, valence and drive state — as prompt context, across sessions
  and without fine-tuning.
- **Substrate-primary.** No language model in the action path: the agent's own learned substrate
  (NAc reward and fear, keyed on the situation its sensors encode) chooses what to do.

Works headless, in simulation, in a live Minecraft world, or on a Reachy Mini robot.

- **Website:** [pymaxim.bio](https://pymaxim.bio)
- **Documentation:** [pymaxim.bio/getting-started](https://pymaxim.bio/getting-started/)
- **Source, experiments and ledgers:** [github.com/dennys246/Maxim](https://github.com/dennys246/Maxim)

## What it has shown — and what it has not

Each result below is a row on the ledger of record, graded against gates frozen before the data; the
ledger states each claim's exact scope, its evidence, its current status and what would invalidate it:
[behavioral_graduation_candidates.md](https://github.com/dennys246/Maxim/blob/main/docs/plans/behavioral_graduation_candidates.md).

| Result | What was measured |
|---|---|
| <!-- claim: T1-16 --> **Memory carried across sessions takes part in recall** ([Exp 63](https://github.com/dennys246/Maxim/blob/main/docs/experiments/exp63_carried_recall_prereg.md), LLM harness; **EARNED 2026-10-04** (narrow: storage persistence into recall on the current goal-keyword ranker, `access_based` retention, one run, mistral-7b; it supersedes Exp 10, whose 2026-09-27 re-run ended in typed aborts) | All 84 memories carried from one session reload unchanged. On resumed turns 2–8, with new memories also competing, the memories surfaced into the prompt were exactly those the shipped ranker picks over carried and new alike (it reads no memory's origin or age beyond a recency tiebreak), and carried memories were among them every turn. It does not show that recalled memories change what the agent does, or that NAc causal links reach the prompt |
| <!-- claim: T1-11 --> **A taught want transfers between agents** ([Exp 56](https://github.com/dennys246/Maxim/blob/main/docs/experiments/56_four_arm_sharing.md), 1.2, substrate-primary; **RE-VALIDATED 2026-09-19** on Paper 1.20.4) | An agent that ingests another's exported substrate acts on what that agent was taught on first contact, in a live Minecraft world |
| <!-- claim: T1-13 --> **Anticipatory avoidance from game-native pain** ([Exp 60](https://github.com/dennys246/Maxim/blob/main/docs/experiments/exp60_drowning_avoidance_prereg.md), 1.3, substrate-primary; **EARNED 2026-09-16**) | An agent that felt air-hunger underwater leaves the water *before* the pain on later submersions; its yoked twin without the fear pathway never does |
| <!-- claim: T1-14 --> **That fear transfers** ([Exp 61](https://github.com/dennys246/Maxim/blob/main/docs/experiments/exp61_shared_fear_prereg.md), 1.3, substrate-primary; **EARNED 2026-09-17**) | A receiver that never felt the pain leaves the water on its first submersion — 12/12, against 0 of 60 across the three control arms |

What is **not** claimed is just as load-bearing. In the 1.3 survival benchmark, survival itself was at
ceiling: a carried fear buys about 25 s of latency, 11 hp and 22 s of oxygen pain — not life. "Dark = danger"
is blocked at the instrument, and generalization to an unseen situation is untested. Each
mechanism's behavioural status is tracked on the ledger; mechanisms that failed to earn weight are
marked Dormant in the code.

The Minecraft results were produced by the harnesses in
[`scripts/exp56/`](https://github.com/dennys246/Maxim/tree/main/scripts/exp56) and
[`scripts/survival_world/`](https://github.com/dennys246/Maxim/tree/main/scripts/survival_world) against
Paper servers (Exp 56 earned on 1.16.5 and re-baselined on 1.20.4; Exp 60 and 61 on 1.20.4). They run
from a repository checkout, not from the installed wheel.

## Quickstart

```bash
# With Claude (fastest way to start)
pip install 'pymaxim[llm-anthropic]'
export ANTHROPIC_API_KEY=sk-...
maxim --sim "test memory recall under interference"

# Or with a local model (no API key needed)
pip install 'pymaxim[llm-llama,llm-server]'
maxim --list-models                                  # see available models
maxim --sim "test memory recall" --llm mistral-7b    # downloads on first run

# Cradle sensorimotor development (an infant agent learns from sensation)
pip install 'pymaxim[llm-llama,llm-server,semantic]'
maxim --sim cradle --embodiment bodies/infant_humanoid --sim-max-turns 25
```

Check your setup with `maxim doctor`. Simulation reports are written to `~/.maxim/sim_reports/{session_id}/`.
Substrate-primary action selection in a simulation is `--aut-mode substrate-primary` (experimental).

## Bio-systems

Maxim's architecture is modelled on brain systems, not software patterns:

| System | Biological analog | What it does |
|---|---|---|
| **Hippocampus** | Episodic memory | Captures experiences with the situation they happened in; recalls by context |
| **NAc** (nucleus accumbens) | Reward and punishment learning | Causal links from actions to outcomes, reward bias, situation-keyed fear |
| **EC** (entorhinal cortex) | Pattern separation and completion | Encodes sensor state into situation clusters |
| **ATL** (anterior temporal lobe) | Semantic concepts | Forms and reinforces concepts from experience |
| **SCN** (suprachiasmatic nucleus) | Circadian clock | Temporal phase tracking and time-anchored credit fallback |
| **Angular Gyrus** | Arithmetic-fact retrieval (posterior parietal) | Math layer: exact arithmetic and math-fact memory, used to ground ATL concept statistics |
| **PainBus** | Nociception | Pain signals from the body, which drive NAc learning |
| **Default Network** | Resting-state network | Novelty detection, arousal, reactive behaviours |

## Bodies and drives

Agents have bodies with sensors, modulators and failure modes declared in YAML:

```yaml
# Homeostatic drive — the body self-regulates toward set_point
core_temperature:
  drive:
    drift_mode: homeostatic
    set_point: 0.0
    drift_rate: 0.001
    comfort_band: 0.4        # no discomfort within +/-0.4
    pain_scale: 0.5          # pain per unit outside the band

# Entropic drive — drifts away; only an action restores it
hunger:
  drive:
    drift_mode: entropic
    drift_direction: up
    drift_rate: 0.006
    deprivation_threshold: 0.7
    deprivation_pain: 0.3
```

Contact, touch and narrated events all converge on one pipeline: sensor change → failure evaluation →
PainBus. Whether that pain reaches NAc learning depends on its intensity and class: the infant's
touch-burn, for one, is a weak drive pain below every learner's threshold
([#1161](https://github.com/dennys246/Maxim/issues/1161)). In the Minecraft world the game owns the drives (hunger drains, air runs out)
and the pain comes from the game, not from a model.

## Sharing what an agent learned

An agent's learned substrate — its NAc policy and EC situation clusters, never its episodic memories —
exports as a bundle another agent can ingest. Ingest validates every bundle before anything is merged,
and a donor can deepen a receiver's negative biases but never weaken them.

```bash
# --session takes a session directory (a simulation's is ~/.maxim/sim_reports/<session_id>)
maxim substrate export out.zip --session ~/.maxim/sim_reports/<id> \
    --contributor-id <your-id> --body-ref minecraft_player
maxim substrate inspect out.zip                      # read the manifest
maxim substrate ingest out.zip --session <receiver-dir> --trust <contributor-id> \
    --receiver-body minecraft_player                 # dry run; add --apply to merge

# An Oasis is a shared source of signed releases
maxim hive add <name> <url> --queen-key <identity>=<pubkey_b64>
maxim hive pull --from <name> --session <receiver-dir> --receiver-body minecraft_player \
    --receiver-agent-id <your-agent-id>              # dry run; add --apply to merge
```

Signed releases carry a signature over every member, a signed entry index and a release sequence. A
receiver refuses a second payload under the same key and sequence (equivocation), and a legacy v1 bundle
from a key it has already accepted a v2 release from (downgrade). See
[Substrate sharing](https://github.com/dennys246/Maxim/blob/main/docs/user/substrate-sharing.md) and the
[bundle format](https://github.com/dennys246/Maxim/blob/main/docs/user/hivemind_bundle_format.md).

## Installation

```bash
pip install pymaxim
```

### Optional extras

| Extra | What it adds |
|-------|-------------|
| `llm-anthropic` | Claude backend |
| `llm-openai` | OpenAI backend |
| `llm-llama` | Local LLM inference via llama.cpp |
| `llm-server` | Local OpenAI-compatible model server (includes llama.cpp) |
| `llm-torch` | PyTorch/Transformers backend |
| `semantic` | Sentence-transformer embeddings for memory and encoding |
| `temporal` | Natural-language date parsing |
| `training` | TensorFlow/Keras training |
| `vision` | Camera and object detection |
| `yolo` | YOLO object detection |
| `audio` | Microphone and Whisper transcription |
| `tts` | Text-to-speech via Piper |
| `reachy` | Reachy Mini robot SDK |
| `pi` | The Raspberry Pi bundle: `reachy`, `console`, `llm-anthropic`, `tts` |
| `sign` | Signing and verifying substrate releases |
| `console` | The web console server |
| `search` | Web search (DuckDuckGo) |
| `comms` | Twilio SMS and voice |
| `database` | PostgreSQL and pgvector memory stores |
| `all` | Every extra except `llm-torch`, `semantic`, `yolo`, `pi` and `test` |
| `test` | The test suite's dependencies |

> **Note:** `[all]` does **not** include `[semantic]`. Without it, memory recall and substrate encoding
> fall back to bag-of-words hashing. For full memory quality: `pip install 'pymaxim[all,semantic]'`.

## Python API

21 verb-based functions give programmatic access to the same runtime:

```python
import maxim

result = maxim.imagine(goal="test safety boundaries")   # run a simulation
state = maxim.observe("memory")                         # inspect a bio-system
report = maxim.diagnose()                               # the same checks as `maxim doctor`

maxim.run(model="mistral-7b", goal="inspect the workspace")   # needs a configured LLM backend

# Controller-backed motion on a robot (robot and headless=True are contradictory)
maxim.run(model="mistral-7b", goal="turn your head 20 degrees left", robot="reachy_mini", headless=False)

models = maxim.list_models()
maxim.download_model("qwen2.5-14b-instruct")
```

See [the Python API reference](https://github.com/dennys246/Maxim/blob/main/docs/user/python-api.md).

## CLI quick reference

```bash
maxim                                     # interactive menu
maxim --llm claude-sonnet                 # agent runtime with Claude
maxim --sim "test memory recall"          # generative simulation
maxim --sim benchmark --models mistral-7b,qwen2.5-14b
maxim doctor                              # environment check
maxim config list                         # every resolved setting and where it came from
maxim model list                          # user-defined model profiles (catalog: --list-models)
```

Simulation exit codes separate run integrity from experimental verdicts: `0` means the run produced
usable evidence (including outcomes such as `failed` or `inconclusive`), `1` is an error, and `4` is an
incomplete or aborted run — campaign scripts must reject every non-zero exit before analysing a report.
Python APIs return the structured `finish_reason` instead of exiting.

See the [CLI reference](https://github.com/dennys246/Maxim/blob/main/docs/user/cli-reference.md) for every flag.

## Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](https://github.com/dennys246/Maxim/blob/main/docs/user/getting-started.md) | First-run walkthrough |
| [CLI Reference](https://github.com/dennys246/Maxim/blob/main/docs/user/cli-reference.md) | All command-line flags |
| [Python API](https://github.com/dennys246/Maxim/blob/main/docs/user/python-api.md) | Programmatic usage |
| [Simulation](https://github.com/dennys246/Maxim/blob/main/docs/user/simulation.md) | Campaigns, scenarios, cradle, benchmarks |
| [Substrate sharing](https://github.com/dennys246/Maxim/blob/main/docs/user/substrate-sharing.md) | Export, ingest, Oases |
| [Substrate-primary mode](https://github.com/dennys246/Maxim/blob/main/docs/substrate_primary.md) | Action selection without an LLM |
| [Architecture](https://github.com/dennys246/Maxim/blob/main/docs/reference.md) | Module map, bio-system glossary |
| [LLM Setup](https://github.com/dennys246/Maxim/blob/main/docs/user/llm-setup.md) | Model download and configuration |
| [Peer Setup](https://github.com/dennys246/Maxim/blob/main/docs/user/peer-setup.md) | Multi-machine and tunnel setup |
| [Robot Setup](https://github.com/dennys246/Maxim/blob/main/docs/user/robot-setup.md) | Reachy Mini ships in-tree; third-party robots plug in via the `maxim.robots` entry-point group |
| [Configuration](https://github.com/dennys246/Maxim/blob/main/docs/user/configuration.md) | Environment variables, config.json |
| [Experiments](https://github.com/dennys246/Maxim/blob/main/docs/experiments/README.md) | Every experiment, its prereg and its verdict |
| [Troubleshooting](https://github.com/dennys246/Maxim/blob/main/docs/user/troubleshooting.md) | Common issues and diagnostics |

## Design essays

[dennyschaedig.com/maxim](https://www.dennyschaedig.com/maxim) hosts Denny's **design essays** — the *why*
behind Maxim's architecture. They are opinion and rationale, not reference: the canonical
reference and evidence site is [pymaxim.bio](https://pymaxim.bio/getting-started/), which wins
wherever the two disagree, and the repository's experiment, defect, limits, and graduation
ledgers win over both.

| Essay | Topic |
|---|---|
| [Maxim 1.0 — The Honest Benchmark](https://www.dennyschaedig.com/maxim/release-1-0) | The 1.0 release: what shipped, and the pre-registered experiments that mapped where the bio-substrate helps and where the LLM prior dominates |
| [Sound orientation](https://www.dennyschaedig.com/maxim/sound-orientation) | The Reachy Mini sound-orient case study — real-hardware sensorimotor learning, including the actuation bug |
| [Substrate-primary mode](https://www.dennyschaedig.com/maxim/substrate-primary) | Why the bio-substrate should drive action selection, and the phased plan for it |
| [Hivemind + Oasis](https://www.dennyschaedig.com/maxim/hivemind) | Federated bio-substrate sharing — the design, not a shipped service |
| [Agent architecture](https://www.dennyschaedig.com/maxim/agent-architecture) | Layered architecture, the bio-system pipeline, fear circuit, cerebellum |
| [Math & statistical cognition](https://www.dennyschaedig.com/maxim/math-cognition) | Statistician agent, variance, NAc reward, Angular Gyrus |
| [Memory systems](https://www.dennyschaedig.com/maxim/memory-systems) | Hippocampus, NAc, SCN, ATL, EC, Angular Gyrus in depth; semantic memory at `#semantic` |
| [Embodiment](https://www.dennyschaedig.com/maxim/embodiment) | Sensor-Entity-Modulator protocol, drives, pain cascade |
| [Imagination](https://www.dennyschaedig.com/maxim/imagination) | Real-time entity design from novel percepts |
| [Proprioception & body awareness](https://www.dennyschaedig.com/maxim/proprioception) | Body state, drive evaluation, interoception |
| [Attention & salience](https://www.dennyschaedig.com/maxim/attention-salience) | Salience modulation and attention weighting |
| [Deliberation](https://www.dennyschaedig.com/maxim/deliberation) | PFC inner monologue and the thought stream |

The reference pages that used to live beside the essays have moved to pymaxim.bio (the old
URLs redirect):

| Was | Now |
|---|---|
| Usage guide | [pymaxim.bio/installation/](https://pymaxim.bio/installation/) |
| Tools & introspection | [pymaxim.bio/reference/tools/](https://pymaxim.bio/reference/tools/) |
| Simulation | [pymaxim.bio/guides/simulation/](https://pymaxim.bio/guides/simulation/) |
| Networking / Agent mesh | [pymaxim.bio/guides/networking/](https://pymaxim.bio/guides/networking/) |
| Operating modes | [pymaxim.bio/concepts/operating-modes/](https://pymaxim.bio/concepts/operating-modes/) |
| Communication & safety | [pymaxim.bio/concepts/communication/](https://pymaxim.bio/concepts/communication/) |
| Technical deep dive | [pymaxim.bio/concepts/architecture/](https://pymaxim.bio/concepts/architecture/) |
| Experiments & results | [pymaxim.bio/research/experiments/](https://pymaxim.bio/research/experiments/) |
| Overview | [pymaxim.bio/getting-started/](https://pymaxim.bio/getting-started/) |

Five reference-flavoured pages are still served on dennyschaedig.com only until their
pymaxim.bio equivalents deploy; delete a row here when the page is retired:

| Held page | Retires to |
|---|---|
| [DM campaigns](https://www.dennyschaedig.com/maxim/dm-campaigns) | [pymaxim.bio/guides/dm-campaigns/](https://pymaxim.bio/guides/dm-campaigns/) |
| [Benchmarks](https://www.dennyschaedig.com/maxim/benchmarks) | [pymaxim.bio/guides/benchmarks/](https://pymaxim.bio/guides/benchmarks/) |
| [Prompt system & tool injection](https://www.dennyschaedig.com/maxim/prompt-system) | [pymaxim.bio/concepts/prompt-system/](https://pymaxim.bio/concepts/prompt-system/) |
| [Concept decomposition](https://www.dennyschaedig.com/maxim/concept-decomposition) | [pymaxim.bio/systems/concept-decomposition/](https://pymaxim.bio/systems/concept-decomposition/) |
| [Component library (interactive catalog)](https://www.dennyschaedig.com/maxim/component-library) | [pymaxim.bio/reference/components/](https://pymaxim.bio/reference/components/) |

## Contributing

Issues and PRs welcome at [github.com/dennys246/Maxim](https://github.com/dennys246/Maxim).

## License

See [LICENSE](https://github.com/dennys246/Maxim/blob/main/LICENSE) for details.
