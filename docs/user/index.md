# Maxim User Guide

Documentation for users of Maxim (pymaxim) — a bio-inspired cognitive architecture with adaptive planning, biological memory systems, and local/cloud LLM inference. Works headless, with simulation, with DM campaigns, or connected to robots.

## Start Here

- [Getting Started](getting-started.md) — Installation, first run, prerequisites
- **`maxim`** (no args) — launches a Rich interactive menu with campaign discovery, recent sessions, and quick-start options
- [Python API](python-api.md) — All 17 API verbs with examples

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

## Website Guides

Long-form topic guides are published at **https://dennyschaedig.com/maxim**. These are the authoritative prose walkthroughs for each major system — deeper than the reference pages above, lighter than the source code.

| Guide | URL slug | What it covers |
|---|---|---|
| Overview | `maxim-overview` | What Maxim is, why it exists, design philosophy |
| Agent Architecture | `maxim-agent-architecture` | 5-agent pipeline, bio-system wiring, agent loop |
| Memory Systems | `maxim-memory-systems` | Hippocampus, tier progression, consolidation |
| Semantic Memory | `maxim-semantic-memory` | EC, ATL, substrate encoding, cross-session persistence |
| Attention & Salience | `maxim-attention-salience` | Salience scoring, working memory, gating |
| Prompt System | `maxim-prompt-system` | PromptBuilder, priority tiers, tool injection |
| Embodiment | `maxim-embodiment` | SEM drives, sensors, modulators, pain cascade |
| Body Awareness | `maxim-proprioception` | Interoception, sensor readings, reflex system |
| Deliberation | `maxim-deliberation` | PFC inner-monologue cycle (long-form) |
| Concept Decomposition | `maxim-concept-decomposition` | Noun-chunk extraction, spaCy pipeline (long-form) |
| Imagination | `maxim-imagination` | Real-time entity design, ImaginationTrigger pipeline |
| Component Library | `maxim-component-library` | 73 SEM seed components, genre gating, YAML schema |
| Operating Modes | `maxim-operating-modes` | Awake/sleep, planning/supervised/autonomous |
| Simulation | `maxim-simulation` | Percept simulation, generative campaigns, fixtures |
| DM Campaigns | `maxim-dm-campaigns` | D&D-style narrative campaigns |
| Benchmarks | `maxim-benchmarks` | Multi-model scoring, benchmark harness |
| Tools & Introspection | `maxim-tools` | Tool catalog, custom tools, introspection API |
| Multi-LLM Networking | `maxim-networking` | Peer/leader mesh, Cloudflare tunnel, lane routing |
| Agent Mesh | `maxim-agent-mesh` | Peer identity, discovery, and transport layer |
| Communication & Safety | `maxim-communication` | Communication gateway, preemption circuit, execution tracking |
| Hivemind + Oasis | `maxim-hivemind` | Substrate sharing, Oasis aggregation, P2P protocol |
| Substrate-Primary Mode | `maxim-substrate-primary` | Running without an LLM, substrate-driven decisions |
| Math & Statistical Cognition | `maxim-math-cognition` | Statistician agent, variance tracking, causal inference |
| Experiments & Results | `maxim-experiments` | Key Roy experiment outcomes and what they proved |
| Technical Deep Dive | `maxim-technical-deepdive` | Architecture internals for contributors |
| Usage Guide | `maxim-usage-guide` | End-to-end usage walkthrough |
| Roadmap | `maxim-roadmap` | Planned features, 1.0 gates, post-1.0 directions |

## Developer Documentation

For internal architecture and design decisions, see:

- [Architecture](../../ARCHITECTURE.md)
- [Design Decisions](../../DECISIONS.md)
- [Contributing](../../CONTRIBUTING.md) — Code style, testing, PR process
- [Internal Docs](../index.md)
