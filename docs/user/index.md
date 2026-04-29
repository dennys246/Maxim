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
- [Safety](safety.md) — Autonomy levels, FearAgent, pain detection, harm prediction
- [Component Library](component-library.md) — 55 SEM components across 7 genre-gated categories
- [API Quickstart](api-quickstart.md) — Quick examples for the Python API

## Setup & Support

- [Robot Setup](robot-setup.md) — Reachy Mini connection, daemon, diagnostics
- [Troubleshooting](troubleshooting.md) — Common issues and fixes
- [Upgrading](upgrading.md) — Version-to-version upgrade contract (state files, warnings, manual actions)

## Extending Maxim

- [Extension API](extension_api.md) — Stable extension points: robots, tools, backends, percept sources, action sinks, bio-system bridges
- [Tool side_effects Registry](tool_side_effects.md) — Append-only registry of well-known `ToolOutput.side_effects` keys
- [Stable API](stable_api.md) — 1.0 stability contract on the public verbs (async wrappability, CWD assumptions, etc.)

## Developer Documentation

For internal architecture and design decisions, see:

- [Architecture](../../ARCHITECTURE.md)
- [Design Decisions](../../DECISIONS.md)
- [Contributing](../../CONTRIBUTING.md) — Code style, testing, PR process
- [Internal Docs](../index.md)
