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
- `main.py`: CLI entrypoint and flags (`--mode`, `--verbosity`, `--audio`, `--audio_len`).
- `src/conscience/selfy.py`: `Maxim` orchestrator (capture loop, lifecycle, logging).
- `src/inference/`: observation/control functions (vision target selection, motor control, etc.).
- `src/models/vision/`: perception models (YOLO segmentation/pose).
- `src/models/movement/`: MotorCortex model (ConvNeXt-Tiny head-movement prediction).
- `src/models/audio/`: Whisper wrapper (transcription backend).
- `src/data/`: camera/audio utilities and file outputs.
- `src/utils/`: config, logging, plotting, filesystem helpers.

## Output Layout (Default)
- `experiments/maxim/videos/`: `reachy_video_<YYYY-MM-DD_HHMMSS>.mp4`
- `experiments/maxim/audio/`: `reachy_audio_<YYYY-MM-DD_HHMMSS>.wav` and optional `audio/chunks/*.wav`
- `experiments/maxim/text/`: `reachy_transcript_<YYYY-MM-DD_HHMMSS>.jsonl`
- `experiments/models/MotorCortex/`: model checkpoint + training artifacts

## Invariants
- Control loop must not perform heavy disk I/O.
- Recording uses backpressure (bounded queues) rather than intentional dropping when “record everything” is requested.
- Public import paths should remain stable, or be preserved via re-exports when refactoring.
