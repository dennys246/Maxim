# Maxim User Guide

Documentation for users of Maxim (pymaxim) — a bio-inspired cognitive architecture with adaptive planning, biological memory systems, and local/cloud LLM inference. Works headless, with simulation, with DM campaigns, or connected to robots.

## Start Here

- [Getting Started](getting-started.md) — Installation, first run, prerequisites
- **`maxim`** (no args) — launches a Rich interactive menu with campaign discovery, recent sessions, and quick-start options
- [Python API](python-api.md) — All API verbs with examples

## Core Guides

- [Modes Guide](modes-guide.md) — Choosing the right operating mode
- [CLI Reference](cli-reference.md) — All command-line flags
- [Configuration](configuration.md) — Environment variables, config files, data paths (`~/.maxim/`)
- [Tools Reference](tools.md) — What the agent can do (40+ tools + custom tool registration)

## Simulation & Campaigns

- [Simulation Guide](simulation.md) — Running and recording simulated scenarios
- [Writing Scenarios](writing-scenarios.md) — Authoring YAML test scenarios
- [DM Campaigns](dm-campaigns.md) — D&D-style campaigns with NPC agents, party mode, encounter templates
- [Benchmarks](benchmarks.md) — Multi-model comparison testing

## Subsystems

- [LLM Setup](llm-setup.md) — Local models, 8 cloud providers (Anthropic, OpenAI, Gemini, Groq, Together, Fireworks, Mistral, DeepSeek)
- [Peer Setup](peer-setup.md) — Connecting to a remote leader via Cloudflare tunnel
- [Vision & Audio](vision-audio.md) — Camera, Whisper, VAD, voice commands
- [Memory](memory-user-guide.md) — What persists, lifecycle, recall improvements
- [Concept Decomposition](concept-decomposition.md) — Breaking text into substrate-encodable concept chunks for richer cross-session associations
- [Deliberation](deliberation.md) — PFC inner-monologue cycle: thinking, bio-memory consultation, and reasoning before acting
- [Asset Foundry](asset-foundry.md) — Autonomous LLM-driven pipeline to generate, validate, and score SEM components
- [Substrate & Hivemind](substrate-sharing.md) — Exporting and sharing learned substrate state between Maxim instances
- [Cross-Session Learning](cross-session-learning.md) — How affordance and causal knowledge accumulate across sessions
- [Safety](safety.md) — Autonomy levels, FearAgent, pain detection, harm prediction
- [Component Library](component-library.md) — 73 SEM components across 7 genre-gated categories
- [API Quickstart](api-quickstart.md) — Quick examples for the Python API

## Setup & Support

- [Robot Setup](robot-setup.md) — Reachy Mini connection, daemon, diagnostics
- [Troubleshooting](troubleshooting.md) — Common issues and fixes
- [Upgrading](upgrading.md) — Version-to-version upgrade contract (state files, warnings, manual actions)

## Extending Maxim

- [Extension API](extension_api.md) — Stable extension points: robots, tools, backends, percept sources, action sinks, bio-system bridges
- [Tool side_effects Registry](tool_side_effects.md) — Append-only registry of well-known `ToolOutput.side_effects` keys
- [Stable API](stable_api.md) — 1.0 stability contract on the public verbs (async wrappability, CWD assumptions, etc.)

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

## Developer Documentation

For internal architecture and design decisions, see:

- [Architecture](../../ARCHITECTURE.md)
- [Design Decisions](../../DECISIONS.md)
- [Contributing](../../CONTRIBUTING.md) — Code style, testing, PR process
- [Internal Docs](../index.md)
