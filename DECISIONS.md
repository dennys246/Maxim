# Design Decisions (Maxim)

This file tracks decisions that affect public behavior, repo structure, and long-term maintenance.

## 2026-01-02: Queue-based capture + writer pipeline
Reason:
- Avoid blocking perception/motor control on disk I/O.
- Enable “record everything” semantics by applying backpressure (blocking queues) instead of dropping samples.

Tradeoffs:
- More moving parts (threads/process + shutdown signaling).
- When disk/CPU can’t keep up, capture blocks and effective FPS may decrease.

## 2026-01-02: Single-run artifacts (MP4 + WAV + JSONL transcript)
Reason:
- A single `videos/*.mp4` is more efficient and simpler than thousands of PNGs.
- A single `audio/*.wav` preserves a continuous audio stream; JSONL allows streaming transcript append.

Tradeoffs:
- Requires codecs/backends for MP4 writing (environment dependent).
- Large files require log/cleanup discipline.

## 2026-01-04: Store transcripts under `data/transcript/`
Decision:
- JSONL transcripts are written under `data/transcript/` (previously `data/text/`).

Reason:
- Avoid confusion with generic “text” outputs and make transcripts easier to locate.

## 2026-01-02: Whisper transcription runs in a separate process
Reason:
- Whisper inference is heavy and should not stall the control loop.
- Process isolation avoids GIL contention and keeps the rest of the system responsive.

Tradeoffs:
- Whisper dependency/model availability may be missing; transcription must degrade gracefully.
- Requires chunking audio and coordinating handoff via a queue.

## 2026-01-02: Optional audio pipeline via CLI flags
Decision:
- `--audio True/False` controls audio capture/transcription.
- `--audio_len <seconds>` controls chunk size for efficient streaming transcription.

Reason:
- Some runs are vision-only; audio should be skippable.
- Chunking balances latency (short chunks) vs throughput (long chunks).

## 2026-01-02: `--mode sleep` skips `wake_up()`
Decision:
- `--mode sleep` records/transcribes audio without running the camera/ML loop and does not call `ReachyMini.wake_up()`.

Reason:
- Support “leave motors asleep” debugging and audio-only dataset capture.

Tradeoffs:
- The run won’t auto-stop based on frame epochs; it runs until interrupted.

## 2026-01-02: Default mode is `passive-interaction`
Decision:
- Default `--mode` is `passive-interaction`.
- `Maxim(mode=...)` defaults to `passive-interaction`.

Reason:
- Safer default behavior (no MotorCortex training) while still tracking targets.
- Reduces surprise ML/compute costs when running `python scripts/main.py` with no flags.

Tradeoffs:
- Users who want the previous behavior must pass `--mode live` (or `--mode train`).

## 2026-01-02: Per-run logs saved under `data/logs/`
Decision:
- Each CLI run writes logs to `data/logs/reachy_log_<run_id>.log`.

Reason:
- Makes runs debuggable after the fact without copying terminal output.
- Keeps artifacts grouped per session alongside video/audio/transcripts.

Tradeoffs:
- Produces additional files; users may need periodic cleanup.

## 2026-01-02: Inference code lives under `src/maxim/inference/`
Reason:
- Keep “runtime inference/control” separate from “robot orchestration” (`src/maxim/conscience/`) and “model definitions” (`src/maxim/models/`).

Tradeoffs:
- Requires stable re-export modules to preserve import paths during refactors.

## 2026-01-02: Vision via YOLOv8 (segmentation + pose)
Reason:
- Fast, general-purpose perception for “person/object of interest” detection.
- Pose keypoints enable eye/face target refinement when available.

Tradeoffs:
- Heavier runtime dependency; performance depends on hardware.
- Model weights and backends vary by environment.

## 2026-01-02: MotorCortex uses ConvNeXt-Tiny backbone
Decision:
- MotorCortex predicts head movement deltas: `[x, y, z, roll, pitch, yaw, duration]`.

Reason:
- Strong image feature extractor that trains well for regression with minimal custom code.

Tradeoffs:
- Requires TensorFlow/Keras for training/inference in this repo’s implementation.

## 2026-01-02: Add `maxim` CLI entrypoint
Decision:
- `pip install -e .` installs a `maxim` console script (entrypoint: `maxim.cli:main`).
- The importable package is `maxim` (code lives under `src/maxim/`); `src.*` imports are removed.
- `python scripts/main.py` remains supported as a compatibility entrypoint.

Reason:
- Reduce friction for new users (no need to remember the module/file path).
- Avoid confusion from a top-level package named `src`.

## 2026-01-02: JSON-configured key responses
Decision:
- Maxim loads `data/util/key_responses.json` on startup and listens for terminal key presses while running (override via `$MAXIM_KEY_RESPONSES`).

Reason:
- Allow quick, extensible runtime actions (e.g., recenter vision) without impacting the control loop.

## 2026-01-03: Store motion presets under `data/`
Decision:
- Default motion actions load from `data/motion/default_actions.json`.

Reason:
- Keep editable JSON configs separate from code and easy to find.

## 2026-01-04: Store head poses under `data/motion/default_poses.json`
Decision:
- Default head poses (including the `centered` pose used by the `c` key) load from `data/motion/default_poses.json`.

Reason:
- Allow robot-specific calibration of “centered” without changing code.

## 2026-01-03: Store trained models under `data/models/`
Decision:
- Default model artifacts (MotorCortex checkpoints/history, YOLO weights) live under `data/models/`.

Reason:
- Keep model artifacts separate from run outputs under `data/`.

## 2026-01-03: Extract reusable helpers from nested defs
Decision:
- Avoid defining reusable helper functions inside other functions/methods.
- Put cross-cutting helpers under `src/maxim/utils/` (or at module scope) and import them where needed.

Reason:
- Improve reuse and reduce duplicated logic while keeping runtime loops readable.

## 2026-01-04: Keep Python code under the `maxim` namespace
Decision:
- Importable code lives under `src/maxim/` (packaged as `maxim*`).
- Avoid creating new top-level packages under `src/` (e.g., `src/agents/`) unless `pyproject.toml` explicitly includes them.

Reason:
- Ensures `pip install -e .` installs everything needed for imports and avoids collisions with overly-generic package names.
