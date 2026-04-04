# Vision & Audio

This guide covers Maxim's visual perception and audio transcription systems -- the two primary sensory inputs for the Reachy Mini robot.

---

## Vision

### How It Works

Maxim uses the Reachy Mini's head camera for visual perception. Each frame is processed through a multi-stage pipeline:

1. **Object detection** -- Identifies and localizes objects, people, and body parts in the camera frame.
2. **Pose estimation** -- Detects human body poses (skeleton tracking) for any people in view.
3. **Tracking** -- Maintains object identity across frames using IoU-based tracking, so the same person or object keeps a consistent ID.
4. **Salience** -- Ranks detected objects by novelty and relevance to the robot's current goals.
5. **Attention** -- Directs the robot's gaze toward the most salient target.

### Vision Engines

Maxim supports two vision backends. The default engine is fully open-source under Apache 2.0.

| Engine | Models | License | Install |
|--------|--------|---------|---------|
| **RTM** (default) | RTMDet-m + RTMPose-m | Apache 2.0 | Included in base install |
| **YOLO** (optional) | YOLOv8 | AGPL-3.0 | `pip install -e '.[yolo]'` |

Switch engines with the `--segmentation-model` flag:

```bash
maxim --segmentation-model rtm
maxim --segmentation-model yolo
```

### Downloading Vision Models

Before using vision, download the required ONNX models:

```bash
python -m maxim.models.download --vision
```

This downloads RTMDet-m (detection) and RTMPose-m (pose estimation) models. The YOLO engine downloads its own models automatically on first use.

### What the Robot Can See

- **People** -- with full pose estimation (skeleton keypoints)
- **Common objects** -- COCO categories including cups, bottles, chairs, books, and more
- **Hands and faces** -- detected as part of pose estimation
- **Movement and spatial relationships** -- tracked across frames

### Novelty Tracking

The salience system tracks what the robot has seen before versus what is new:

- New objects receive higher salience scores and attract attention.
- Familiar objects decay in novelty over time.
- Discovery of a new object triggers an attention shift, causing the robot to look toward it.

This means the robot naturally pays attention to changes in its environment -- a new person entering the room, an object being placed on a table, or unexpected movement.

### Limitations

- **Forward-looking only** -- the Reachy Mini has a single head camera with no rear or side vision.
- **Detection range** -- depends on object size and lighting conditions.
- **No depth perception** -- detection produces 2D bounding boxes only; the robot cannot judge distance from vision alone.
- **Frame rate** -- depends on hardware. A GPU accelerates inference significantly; CPU-only setups will be slower.

---

## Audio

### How It Works

Maxim uses faster-whisper for real-time audio transcription. The pipeline runs continuously while audio is enabled:

1. **Recording** -- Captures audio from the microphone in chunks (default: 5-second segments).
2. **VAD** -- Voice Activity Detection filters out silence before sending audio to the transcription model, saving compute.
3. **Transcription** -- The Whisper model converts detected speech to text.
4. **Intent detection** -- Transcribed text is checked for wake words and known commands.

### Whisper Models

Choose a model based on your hardware and accuracy needs:

| Model | Params | VRAM (int8) | Speed | Accuracy |
|-------|--------|-------------|-------|----------|
| `tiny` | 39M | 75 MB | 32x | Low |
| `base` | 74M | 150 MB | 16x | Medium-low |
| `small` | 244M | 500 MB | 6x | Medium |
| `medium` | 769M | 1.5 GB | 2x | High |
| `large-v3` | 1550M | 3 GB | 1x | Highest |
| `distil-large-v3` | 756M | 800 MB | 6x | High (recommended) |

English-only variants (`.en` suffix, e.g. `small.en`) are slightly more accurate for English-only use cases.

**Recommendation:** Start with `distil-large-v3` for the best balance of speed and accuracy. Drop to `small` or `base` if you are running on limited hardware.

### Configuration

Edit `data/util/whisper.json` to configure the transcription engine:

```json
{
  "model": "distil-large-v3",
  "device": "auto",
  "compute_type": "int8",
  "language": "en",
  "vad_filter": true,
  "vad_threshold": 0.25
}
```

- **model** -- Which Whisper model to load (see table above).
- **device** -- `"auto"` selects GPU if available, otherwise CPU. Can also be `"cpu"` or `"cuda"`.
- **compute_type** -- `"int8"` for fastest inference, `"float16"` for GPU, `"float32"` for maximum compatibility.
- **language** -- Language code (e.g. `"en"`). Set to `null` for auto-detection.
- **vad_filter** -- Enable Voice Activity Detection to skip silent segments.
- **vad_threshold** -- Confidence threshold for VAD (0.0 to 1.0).

### VAD Tuning

If speech is being missed or silence is being transcribed, adjust these VAD parameters:

- **Lower `vad_threshold`** (toward 0.0) to catch quieter speech.
- **Reduce `vad_min_speech_duration_ms`** to catch shorter utterances.
- **Increase `vad_speech_pad_ms`** to include more context around detected speech boundaries.

A threshold of `0.25` works well in most environments. Noisy rooms may need a higher value; quiet rooms can go lower.

### CLI Flags

```bash
maxim --audio True          # Enable audio (default)
maxim --audio False         # Disable audio
maxim --audio_len 5.0       # Chunk length in seconds
```

### Voice Commands

The robot responds to voice commands prefixed with a wake word. Recognized wake words are **"Maxim"** and **"Reachy"**.

| Command | Action |
|---------|--------|
| "Maxim" | Wake up / activate agentic mode |
| "Maxim shutdown" | Clean shutdown |
| "Maxim sleep" | Switch to sleep mode |
| "Maxim observe" | Switch strategy to observe |
| "center" | Center the robot's gaze (available in agentic mode) |

Custom voice commands can be added by editing `data/util/phrase_responses.json`.

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Whisper segfaults | Run with float32: `MAXIM_WHISPER_COMPUTE_TYPE=float32 maxim` |
| Speech not detected | Lower `vad_threshold` in `data/util/whisper.json` |
| Too much noise transcribed | Raise `vad_threshold` in `data/util/whisper.json` |
| Slow transcription | Use a smaller model or set `compute_type` to `"int8"` |
| No microphone detected | Ensure the audio device is accessible and that `--audio True` is set |

### Output Files

Audio and transcription data are saved to disk:

- `data/audio/` -- Raw WAV recordings of each audio chunk.
- `data/transcript/` -- JSONL transcripts with timestamps for each detected utterance.
