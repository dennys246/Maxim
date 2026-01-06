# Architecture (Maxim)

Maxim orchestrates Reachy Mini data capture (camera + mic), perception/inference, optional learning, and motor control.

## High-Level Flow
Capture (Reachy) → Writers (video/audio) → Inference (vision/audio) → Control (motor) → Persist (models/history)

## Threading / Process Model
- Video capture thread pulls frames from Reachy and feeds:
  - a bounded “save” queue for the video writer (records everything; blocks when backpressured)
  - a “latest” queue for observation/control (keeps the loop responsive)
- Video writer thread writes a single MP4 for the run.
- Audio capture thread pulls samples and feeds a bounded “save” queue.
- Audio writer thread appends to a single WAV and (optionally) cuts chunk WAVs for transcription.
- Transcription runs in a separate process consuming chunk paths and appending JSONL transcripts.
- Motor commands are funneled through a single executor thread to avoid unsafe concurrent SDK calls.

## Agentic Architecture (Enforcement Rules)

These are **hard architectural rules** for the agentic subsystems in this repo. Violations are architectural bugs and should be caught in docs, code review, and (where possible) CI.

### Layer Ownership (Exclusive)

Paths refer to the `src/maxim/` package layout.

- `src/maxim/agents/`: owns goals, role-specific reasoning, intent generation; must **not** execute tools, mutate state, or inspect environments.
- `src/maxim/planning/`: owns plan generation/refinement; must **not** execute actions, select final actions, or mutate state.
- `src/maxim/planning/decision_engine.py`: owns action selection/arbitration/control flow; must **not** generate plans, execute tools, store memory, or inspect environment internals.
- `src/maxim/planning/policy.py`: owns constraints/guardrails/safety rules; must **not** perform planning, execution, or goal reasoning.
- `src/maxim/tools/`: owns side effects (I/O, network, filesystem, APIs); must **not** do control flow, reasoning, or decision making.
- `src/maxim/environment/`: owns observation of the world; must **not** perform side effects or execute tools.
- `src/maxim/memory/`: owns storage/retrieval/compression/forgetting; must **not** do decision making or action selection.
- `src/maxim/state/` (reserved): owns authoritative runtime truth; must **not** contain long-term storage logic or planning.
- `src/maxim/runtime/`: owns agentic orchestration/main execution loop; must **not** do domain reasoning.
- `src/maxim/conscience/`: owns robot orchestration/main loop (Reachy capture/inference/control); must **not** do agentic decision making.

### Absolute Separation Rules
- Agents never call tools directly.
- Environments never cause side effects.
- Memory never selects actions.
- Planning never mutates state.
- Action selection happens in one place only.
- State is the single source of truth.
- No component may bypass state.

### One-Way Dependency Graph

Dependencies must flow strictly downward (reverse imports are forbidden):

Agents → Planning → Decision Engine → Runtime → Executor → Tools → Environment → State → Memory

### Testability Rule

Each layer must be independently mockable:
- Tools can be no-op or simulated.
- Environments can be simulated.
- Memory can be in-memory.
- Agents can run without side effects.

If a component cannot be tested in isolation, the architecture is violated.

## Key Modules
- `src/maxim/cli.py`: primary CLI entrypoint (`maxim` console script).
- `scripts/main.py`: legacy checkout entrypoint (delegates to `maxim.cli`).
- `src/configs/`: version-controlled config templates and notes.
- `src/maxim/conscience/selfy.py`: `Maxim` orchestrator (capture loop, lifecycle, logging, key responses).
- `src/maxim/agents/`: agent interfaces + implementations (reasoning/intent, no side effects).
- `src/maxim/planning/`: planning + policy + decision engine (agentic action selection).
- `src/maxim/tools/`: tool implementations (side effects).
- `src/maxim/environment/`: environment interfaces/implementations (observations, no side effects).
- `src/maxim/memory/`: memory interfaces/implementations (storage/retrieval, no decisions).
- `src/maxim/evaluation/`: lightweight evaluators/metrics for tools, plans, and agent intents.
- `src/maxim/runtime/`: agentic runtime loop + bootstrap wiring (decision engine → executor → tools).
- `src/maxim/inference/`: observation/control functions (vision target selection, motor control, etc.).
- `src/maxim/models/vision/`: perception models (YOLO segmentation/pose).
- `src/maxim/models/movement/`: MotorCortex model (ConvNeXt-Tiny head-movement prediction).
- `src/maxim/models/audio/`: Whisper wrapper (transcription backend).
- `src/maxim/models/language/`: optional local LLM routing (transcript → agentic action).
- `src/maxim/data/`: camera/audio utilities and file outputs.
- `src/maxim/utils/`: config, logging, plotting, filesystem helpers (and reusable small helpers).

## Output Layout (Default)
- `data/videos/`: `reachy_video_<YYYY-MM-DD_HHMMSS>.mp4`
- `data/audio/`: `reachy_audio_<YYYY-MM-DD_HHMMSS>.wav` and optional `audio/chunks/*.wav`
- `data/transcript/`: `reachy_transcript_<YYYY-MM-DD_HHMMSS>.jsonl`
- `data/training/`: `motor_training_set.jsonl` (trainable samples + user marks)
- `data/agents/<AGENT_NAME>/runtime/`: `state_<run_id>.json` (agentic runtime state snapshots)
- `data/models/MotorCortex/`: MotorCortex checkpoint + training artifacts
- `data/util/llm.json`: optional local LLM config (disabled by default)

## Invariants
- Control loop must not perform heavy disk I/O.
- Recording uses backpressure (bounded queues) rather than intentional dropping when “record everything” is requested.
- Public import paths should remain stable, or be preserved via re-exports when refactoring.
- Reusable helpers should live at module scope (prefer `src/maxim/utils/`) instead of being defined inside hot-loop functions.
