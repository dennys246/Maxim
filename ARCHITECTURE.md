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

## Key Modules
- `src/maxim/cli.py`: primary CLI entrypoint (`maxim` console script).
- `scripts/main.py`: legacy checkout entrypoint (delegates to `maxim.cli`).
- `src/configs/`: version-controlled config templates and notes.
- `src/maxim/conscience/selfy.py`: `Maxim` orchestrator (capture loop, lifecycle, logging, key responses).
- `src/maxim/inference/`: observation/control functions (vision target selection, motor control, etc.).
- `src/maxim/models/vision/`: perception models (YOLO segmentation/pose).
- `src/maxim/models/movement/`: MotorCortex model (ConvNeXt-Tiny head-movement prediction).
- `src/maxim/models/audio/`: Whisper wrapper (transcription backend).
- `src/maxim/data/`: camera/audio utilities and file outputs.
- `src/maxim/utils/`: config, logging, plotting, filesystem helpers (and reusable small helpers).

## Output Layout (Default)
- `data/videos/`: `reachy_video_<YYYY-MM-DD_HHMMSS>.mp4`
- `data/audio/`: `reachy_audio_<YYYY-MM-DD_HHMMSS>.wav` and optional `audio/chunks/*.wav`
- `data/transcript/`: `reachy_transcript_<YYYY-MM-DD_HHMMSS>.jsonl`
- `data/models/MotorCortex/`: MotorCortex checkpoint + training artifacts

## Invariants
- Control loop must not perform heavy disk I/O.
- Recording uses backpressure (bounded queues) rather than intentional dropping when “record everything” is requested.
- Public import paths should remain stable, or be preserved via re-exports when refactoring.
- Reusable helpers should live at module scope (prefer `src/maxim/utils/`) instead of being defined inside hot-loop functions.
