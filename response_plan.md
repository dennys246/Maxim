# Response Output System Implementation Plan

## Overview

Implement a unified response output system that:
1. Sends simple responses directly to the CLI
2. Saves complex outputs to sandbox files and notifies the user of the location
3. (Optional) Converts LLM text responses to speech via TTS

---

## Part 1: CLI Output for Simple Responses

### Goal
Display short, conversational LLM responses directly in the CLI interface.

### Implementation

#### 1.1 Create Response Output Utility
**New file:** `src/maxim/utils/response_output.py`

```python
class ResponseOutput:
    """Manages LLM response output to CLI and files."""

    SIMPLE_THRESHOLD = 500  # Characters (confirmed) - responses under this go to CLI

    def __init__(self, sandbox_path: Path, logger: logging.Logger):
        self.sandbox_path = sandbox_path
        self.logger = logger

    def output_response(self, text: str, response_type: str = "general") -> str | None:
        """Route response to CLI or file based on complexity."""
        if self._is_simple(text):
            self._output_to_cli(text)
            return None
        else:
            return self._output_to_file(text, response_type)

    def _is_simple(self, text: str) -> bool:
        """Determine if response is simple enough for CLI."""
        # Simple if: short, no code blocks, no tables, no lists > 5 items
        ...

    def _output_to_cli(self, text: str) -> None:
        """Print response to CLI with formatting."""
        print(f"\n[Maxim]: {text}\n")

    def _output_to_file(self, text: str, response_type: str) -> str:
        """Save to sandbox and return path."""
        ...
```

#### 1.2 Integrate with Agent Loop
**Modify:** `src/maxim/runtime/agent_loop.py`

After LLM proposal is processed, check if there's a response to display:
- Add `ResponseOutput` instance to loop initialization
- After tool execution, if result contains displayable text, route through `ResponseOutput`

#### 1.3 Modify CLI Listener
**Modify:** `src/maxim/conscience/selfy.py`

- After phrase matching, if LLM generates a response, call `ResponseOutput.output_response()`
- Maintain the `"maxim> "` prompt flow

### Files to Modify
- `src/maxim/utils/response_output.py` (NEW)
- `src/maxim/runtime/agent_loop.py` (lines ~550-600)
- `src/maxim/conscience/selfy.py` (lines ~787-797)
- `src/maxim/utils/__init__.py` (export)

---

## Part 2: Complex Output to Sandbox Files

### Goal
Save lengthy responses, code, data, and structured content to sandbox files with user notification.

### Implementation

#### 2.1 Extend ResponseOutput Class

```python
class ResponseOutput:
    # ... existing code ...

    def _output_to_file(self, text: str, response_type: str) -> str:
        """Save complex output to sandbox and notify user."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine file extension based on content
        ext = self._detect_extension(text, response_type)
        filename = f"response_{timestamp}.{ext}"

        output_path = self.sandbox_path / "outputs" / "responses" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)

        # Notify user via CLI
        relative_path = output_path.relative_to(self.sandbox_path.parent)
        print(f"\n[Maxim]: Response saved to: {relative_path}\n")

        return str(output_path)

    def _detect_extension(self, text: str, response_type: str) -> str:
        """Detect appropriate file extension."""
        if "```python" in text or response_type == "code":
            return "py"
        elif "```json" in text or response_type == "data":
            return "json"
        elif "```" in text:
            return "md"  # Markdown for mixed content
        else:
            return "txt"
```

#### 2.2 Add Response Tool for LLM
**New tool:** `src/maxim/tools/response.py`

Allow LLM to explicitly choose output format:

```python
class RespondTool(BaseTool):
    """Tool for LLM to send responses to user."""

    name = "respond"
    description = "Send a response to the user. Use for answering questions or providing information."

    def run(self, message: str, output_type: str = "auto") -> ToolResult:
        """
        Args:
            message: The response text to send
            output_type: "cli" (force CLI), "file" (force file), or "auto" (decide based on length)
        """
        ...
```

#### 2.3 Update LLM Prompt Template
**Modify:** `src/maxim/agents/llm_worker.py`

Add the `respond` tool to available tools and update system prompt to instruct LLM:
- Use `respond` tool for user-facing responses
- Set `output_type="file"` for code, data, or long explanations

### Files to Modify
- `src/maxim/utils/response_output.py` (extend)
- `src/maxim/tools/response.py` (NEW)
- `src/maxim/tools/__init__.py` (export)
- `src/maxim/agents/llm_worker.py` (prompt template, lines ~330-400)
- `src/maxim/runtime/bootstrap.py` (register tool)

---

## Part 3: Audio Response via TTS (Text-to-Speech)

### Goal
Convert LLM text responses to speech and play through Reachy's speaker.

### Recommended TTS Options

| Model | Size | Quality | Speed | Offline |
|-------|------|---------|-------|---------|
| **Piper TTS** | ~50-100MB | Good | Fast | Yes |
| Coqui TTS (XTTS v2) | ~1.5GB | High | Medium | Yes |
| edge-tts | N/A | Good | Fast | No (Azure) |
| OpenAI TTS | N/A | Excellent | Fast | No (API) |

**Selected:** **Piper TTS** (confirmed)
- Lightweight, fast, fully offline
- Multiple voice models available
- Easy installation via pip

### Implementation

#### 3.1 Create TTS Module
**New file:** `src/maxim/models/audio/tts.py`

```python
class TTSEngine:
    """Text-to-speech synthesis using Piper."""

    def __init__(
        self,
        model_path: str = "data/models/tts/en_US-lessac-medium.onnx",
        sample_rate: int = 22050,
    ):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self._model = None

    def _load_model(self) -> None:
        """Lazy-load TTS model."""
        import piper
        self._model = piper.PiperVoice.load(self.model_path)

    def synthesize(self, text: str) -> np.ndarray:
        """Convert text to audio samples."""
        if self._model is None:
            self._load_model()

        # Generate audio
        audio = self._model.synthesize(text)
        return audio

    def synthesize_to_file(self, text: str, output_path: Path) -> Path:
        """Synthesize and save to WAV file."""
        audio = self.synthesize(text)
        # Save using scipy or soundfile
        ...
        return output_path
```

#### 3.2 Create Speak Tool (Enhanced)
**Modify:** `src/maxim/tools/reachy.py` or create `src/maxim/tools/speak.py`

```python
class SpeakTool(BaseTool):
    """Speak text aloud via TTS."""

    name = "speak"
    description = "Convert text to speech and play through speaker."

    def __init__(self, tts_engine: TTSEngine, speaker_fn: Callable):
        self.tts = tts_engine
        self.speaker = speaker_fn

    def run(self, text: str, save_audio: bool = False) -> ToolResult:
        """
        Args:
            text: Text to speak
            save_audio: Also save audio file to sandbox
        """
        # Synthesize
        audio_samples = self.tts.synthesize(text)

        # Resample if needed for Reachy speaker
        from maxim.utils.audio import resample_audio
        audio_resampled = resample_audio(audio_samples, self.tts.sample_rate, 16000)

        # Play
        self.speaker(audio_resampled)

        # Optionally save
        if save_audio:
            ...

        return ToolResult(success=True, output={"spoken": text})
```

#### 3.3 Integrate with ResponseOutput
**Modify:** `src/maxim/utils/response_output.py`

```python
class ResponseOutput:
    def __init__(self, ..., tts_engine: TTSEngine | None = None, speaker_fn: Callable | None = None):
        self.tts = tts_engine
        self.speaker = speaker_fn
        self.audio_enabled = tts_engine is not None and speaker_fn is not None

    def output_response(self, text: str, speak: bool = False, ...) -> str | None:
        """Route response with optional audio."""
        result = ...  # existing logic

        if speak and self.audio_enabled:
            self._speak_response(text)

        return result

    def _speak_response(self, text: str) -> None:
        """Synthesize and play response."""
        # Truncate very long text for speech
        speech_text = text[:1000] if len(text) > 1000 else text
        audio = self.tts.synthesize(speech_text)
        self.speaker(audio)
```

#### 3.4 Configuration
**Modify:** `src/maxim/cli.py`

Add CLI flags:
```python
parser.add_argument("--tts", action="store_true", help="Enable text-to-speech responses")
parser.add_argument("--tts-model", type=str, default="en_US-lessac-medium", help="TTS voice model")
```

### Files to Create/Modify
- `src/maxim/models/audio/tts.py` (NEW)
- `src/maxim/tools/speak.py` (NEW or modify reachy.py)
- `src/maxim/utils/response_output.py` (extend)
- `src/maxim/cli.py` (add flags)
- `src/maxim/runtime/bootstrap.py` (initialize TTS)
- `requirements.txt` or `pyproject.toml` (add piper-tts dependency)

---

## Implementation Order

### Phase 1: CLI Output (Core)
1. Create `response_output.py` with basic CLI output
2. Create `respond` tool
3. Integrate with agent loop
4. Test with simple queries

### Phase 2: File Output
1. Extend `ResponseOutput` with file saving
2. Add content type detection
3. Update LLM prompt to use respond tool appropriately
4. Test with code/data outputs

### Phase 3: TTS (Optional)
1. Add Piper TTS dependency
2. Create `tts.py` module
3. Create/modify speak tool
4. Integrate with ResponseOutput
5. Add CLI flags
6. Test end-to-end

---

## Verification Plan

### CLI Output Tests
1. Start in agentic mode: `python -m maxim --mode agentic`
2. Type a simple question: "What time is it?"
3. Verify response appears inline after `[Maxim]:`

### File Output Tests
1. Ask for code: "Write a Python function to sort a list"
2. Verify file saved to `sandbox/outputs/responses/`
3. Verify CLI shows path notification

### TTS Tests (if implemented)
1. Start with `--tts` flag
2. Type a question
3. Verify audio plays through speaker
4. Verify CLI output still appears

---

## Dependencies

### Required (Phase 1-2)
- None (uses existing stdlib)

### Optional (Phase 3 - TTS)
```
piper-tts>=1.2.0
onnxruntime>=1.15.0
```

Model download (~100MB):
```bash
# Download Piper voice model
mkdir -p data/models/tts
wget -O data/models/tts/en_US-lessac-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
```

---

## Architecture Diagram

```
User Input (CLI/Voice)
        |
        v
  [Phrase Matching / LLM Worker]
        |
        v
  [LLM Proposal with "respond" tool]
        |
        v
  [ResponseOutput.output_response()]
        |
        +---> Simple? ---> CLI print()
        |
        +---> Complex? --> Sandbox file + CLI notification
        |
        +---> speak=True? --> TTS Engine --> Speaker
```
