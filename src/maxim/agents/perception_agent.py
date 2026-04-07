"""PerceptionAgent: Pure observation and classification.

Transforms raw inputs into structured Percepts without decision-making.

Phase 3: Supports direct frame input from CaptureManager in addition to
JSONL-based observation polling.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from maxim.agents.base import Agent
from maxim.agents.bus import AgentBus, Percept
from maxim.utils.structured_logging import log_structured

if TYPE_CHECKING:
    from maxim.runtime.capture import CapturedFrame, CaptureManager


class PerceptionAgent(Agent):
    """
    Pure perception: observe and classify, no decisions.

    Responsibilities:
    - Normalize transcript text
    - Detect "maxim" keyword and hard overrides
    - Compute salience and novelty for vision
    - Watch for file changes
    - Publish structured Percept to bus

    Phase 3: Can receive frames directly from CaptureManager for
    lower-latency perception without JSONL intermediary.
    """

    agent_name = "perception"

    KEYWORD_ALIASES = {"maximum": "maxim", "maximums": "maxim", "maxims": "maxim"}

    def __init__(
        self,
        bus: AgentBus,
        *,
        name: str | None = None,
        enabled: bool = True,
        interests: set[int] | None = None,
        novelty_window: int = 10,
        capture_manager: CaptureManager | None = None,
    ) -> None:
        super().__init__(name=name, enabled=enabled)
        self._bus = bus
        self._interests = interests or set()
        self._recent_percepts: deque[Percept] = deque(maxlen=novelty_window)
        self._capture_manager = capture_manager

        # Thread pool for non-blocking percept publishing
        # This prevents the segmentation thread from being blocked by bus handlers
        self._publish_executor: ThreadPoolExecutor | None = None
        if capture_manager is not None:
            self._publish_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="percept_pub")

        # Register for segmented frame callbacks if capture manager provided
        # IMPORTANT: Use on_segmented_frame (not on_frame) to get detections
        # The DefaultNetwork needs detections for reactive head movements
        if self._capture_manager is not None:
            self._capture_manager.on_segmented_frame(self._on_captured_frame)

        # State
        self._last_transcript_path: str | None = None
        self._last_transcript_mtime: float | None = None
        self._last_chunk_index: int | None = None

    @staticmethod
    def _normalize_transcript_text(text: str) -> str:
        """Normalize transcript for keyword matching."""
        raw = str(text or "").strip().lower()
        if not raw:
            return ""
        cleaned = re.sub(r"[^\w\s]", " ", raw, flags=re.UNICODE)
        normalized = " ".join(cleaned.split())
        if not normalized:
            return ""
        tokens = [t for t in normalized.split() if t and t != "s"]
        for idx, token in enumerate(tokens):
            if token in PerceptionAgent.KEYWORD_ALIASES:
                tokens[idx] = PerceptionAgent.KEYWORD_ALIASES[token]
        return " ".join(tokens)

    def _detect_hard_override(self, tokens: list[str]) -> str | None:
        """Detect hard override commands."""
        if "sleep" in tokens:
            return "request_sleep"
        if "observe" in tokens:
            return "request_observe"
        if "shutdown" in tokens or ("shut" in tokens and "down" in tokens):
            return "request_shutdown"
        # Exploration commands are handled separately via _parse_explore_command
        return None

    def _parse_explore_command(self, raw_text: str | None) -> dict[str, Any] | None:
        """Parse exploration commands from raw transcript text.

        Detects patterns like:
        - "maxim explore please" -> general exploration
        - "maxim explore [focus] please" -> focused exploration
        - "maxim stop exploring please" -> stop exploration

        Returns:
            dict with keys:
                - "focus": str | None - the exploration focus
                - "stop": bool - True if stop command
                - "source": str - always "voice_command"
            or None if no explore command detected.
        """
        if not raw_text:
            return None

        try:
            from maxim.modes.exploration import parse_explore_command

            return parse_explore_command(raw_text)
        except ImportError:
            return None

    def _compute_salience(self, detections: list[dict]) -> float:
        """Compute salience from vision detections."""
        if not detections:
            return 0.0
        max_sal = 0.0
        for det in detections:
            class_id = det.get("class_id", -1)
            conf = float(det.get("conf", 0.0))
            bbox = det.get("bbox_xyxy", [0, 0, 0, 0])
            w = abs(bbox[2] - bbox[0]) / 1920.0 if len(bbox) >= 4 else 0
            h = abs(bbox[3] - bbox[1]) / 1080.0 if len(bbox) >= 4 else 0
            area = w * h
            interest_mult = 2.0 if class_id in self._interests else 1.0
            sal = conf * (0.3 + 0.7 * area) * interest_mult
            max_sal = max(max_sal, sal)
        return min(1.0, max_sal)

    def _compute_novelty(self, detections: list[dict]) -> float:
        """Compute novelty via Jaccard similarity with recent percepts."""
        if not self._recent_percepts:
            return 1.0
        current_tracks = {d.get("track_id") for d in detections if d.get("track_id")}
        overlaps = []
        for past in self._recent_percepts:
            past_tracks = {d.get("track_id") for d in past.detections if d.get("track_id")}
            if not past_tracks and not current_tracks:
                overlaps.append(1.0)
            elif not past_tracks or not current_tracks:
                overlaps.append(0.0)
            else:
                jaccard = len(current_tracks & past_tracks) / len(current_tracks | past_tracks)
                overlaps.append(jaccard)
        return 1.0 - (sum(overlaps) / len(overlaps))

    def _check_transcript_file(self, obs: dict[str, Any]) -> tuple[bool, str | None]:
        """Check if transcript file changed."""
        latest_transcript = obs.get("latest_transcript")
        repo_root = obs.get("repo_root")
        if not isinstance(latest_transcript, str) or not latest_transcript.strip():
            return False, None
        path = latest_transcript.strip()
        abs_path = path
        try:
            if not os.path.isabs(abs_path) and isinstance(repo_root, str):
                abs_path = os.path.join(repo_root.strip(), abs_path)
        except Exception:
            abs_path = path
        try:
            mtime = float(os.path.getmtime(abs_path))
        except Exception:
            mtime = None
        changed = self._last_transcript_path != path or (
            mtime is not None and (self._last_transcript_mtime is None or mtime > self._last_transcript_mtime)
        )
        if changed:
            self._last_transcript_path = path
            self._last_transcript_mtime = mtime
        return changed, path

    def process_observation(self, obs: dict[str, Any]) -> Percept:
        """Transform raw observation into Percept."""
        vision_event = obs.get("latest_vision_event") or {}
        transcript_rec = obs.get("latest_transcript_record") or {}
        maxim_runtime = obs.get("maxim_runtime")
        cli_input = obs.get("cli_input")

        detections = vision_event.get("detections", []) if isinstance(vision_event, dict) else []

        # Transcript processing
        transcript = None
        chunk_index = None
        raw_text = None
        normalized_tokens: list[str] = []
        has_maxim = False
        hard_override = None

        if cli_input:
            raw_text = str(cli_input).strip()
            transcript = self._normalize_transcript_text(raw_text)
            normalized_tokens = transcript.split() if transcript else []
            has_maxim = True
            hard_override = self._detect_hard_override(normalized_tokens)
        elif isinstance(transcript_rec, dict) and transcript_rec.get("text") is not None:
            raw_text = str(transcript_rec.get("text", "") or "").strip()
            transcript = self._normalize_transcript_text(raw_text)
            normalized_tokens = transcript.split() if transcript else []
            try:
                ci = transcript_rec.get("chunk_index")
                chunk_index = int(ci) if ci is not None else None
            except (ValueError, TypeError):
                chunk_index = None

            # Check for maxim keyword
            is_new_chunk = chunk_index is not None and chunk_index != self._last_chunk_index
            if is_new_chunk and "maxim" in normalized_tokens:
                has_maxim = True
                hard_override = self._detect_hard_override(normalized_tokens)
                self._last_chunk_index = chunk_index

        # Compute signals
        salience = self._compute_salience(detections)
        novelty = self._compute_novelty(detections)

        # Check for exploration commands
        explore_command = self._parse_explore_command(raw_text)

        # Boost salience for voice commands
        if has_maxim:
            salience = max(salience, 0.9)

        # Boost salience even more for exploration commands
        if explore_command:
            salience = 1.0

        # Determine source
        source = "idle"
        if cli_input:
            source = "cli"
        elif transcript and chunk_index != self._last_chunk_index:
            source = "transcript"
        elif detections:
            source = "vision"

        percept = Percept(
            timestamp=time.time(),
            source=source,
            detections=detections,
            transcript_chunk=transcript,
            transcript_chunk_index=chunk_index,
            cli_input=cli_input,
            salience=salience,
            novelty=novelty,
            has_voice_command=bool(transcript and normalized_tokens),
            has_maxim_keyword=has_maxim,
            hard_override=hard_override,
            explore_command=explore_command,
            raw_transcript_text=raw_text,
            maxim_runtime=maxim_runtime,
        )

        self._recent_percepts.append(percept)

        # Log to abstraction stream (only significant percepts)
        if has_maxim or salience > 0.5 or explore_command:
            log_data = {
                "source": source,
                "salience": round(salience, 2),
                "novelty": round(novelty, 2),
                "has_maxim": has_maxim,
            }
            if explore_command:
                log_data["explore_command"] = {
                    "focus": explore_command.get("focus"),
                    "stop": explore_command.get("stop", False),
                }
            log_structured(
                self.log,
                logging.INFO,
                "percept",
                log_data,
            )

        return percept

    def propose_intent(self, state: Any, memory: Any, **kwargs: Any) -> dict[str, Any] | None:
        """Process observation and publish percept.

        NOTE: If CaptureManager is active, the callback path (_on_captured_frame)
        handles Percept publishing. This method only publishes when the callback
        path is not active to avoid duplicate Percepts.
        """
        # Skip if CaptureManager callback is handling perception
        # The callback path publishes Percepts with actual detections
        if self._capture_manager is not None:
            return None

        obs = getattr(state, "data", None) if hasattr(state, "data") else None
        if not isinstance(obs, dict):
            obs = {}

        # Check file changes
        file_changed, file_path = self._check_transcript_file(obs)

        percept = self.process_observation(obs)
        if file_changed:
            percept.file_changed = file_path

        # Publish for downstream agents (only when no CaptureManager)
        self._bus.publish(percept)

        return None  # PerceptionAgent doesn't propose intents directly

    def update_interests(self, add: list[int] | None = None, remove: list[int] | None = None) -> None:
        """Update object class interests for salience boosting.

        Interest classes get a 2x salience multiplier. All COCO classes are
        always detected; this controls prioritization, not visibility.
        """
        if add:
            self._interests.update(add)
        if remove:
            self._interests.difference_update(remove)

    def _on_captured_frame(self, captured: CapturedFrame) -> None:
        """Handle segmented frame from CaptureManager (Phase 3).

        This receives frames AFTER YOLO segmentation completes, so detections
        are populated. The Percept is published to the bus for DefaultNetwork
        to enable reactive head movements.
        """
        # Build percept from captured frame
        detections = captured.detections

        salience = self._compute_salience(detections)
        novelty = self._compute_novelty(detections)

        source = "vision" if detections else "idle"

        percept = Percept(
            timestamp=captured.timestamp,
            source=source,
            detections=detections,
            transcript_chunk=None,
            transcript_chunk_index=None,
            cli_input=None,
            salience=salience,
            novelty=novelty,
            has_voice_command=False,
            has_maxim_keyword=False,
            hard_override=None,
            raw_transcript_text=None,
            maxim_runtime=None,
        )

        self._recent_percepts.append(percept)

        # Log significant percepts
        if salience > 0.5:
            log_structured(
                self.log,
                logging.INFO,
                "percept",
                {
                    "source": "capture_direct",
                    "salience": round(salience, 2),
                    "novelty": round(novelty, 2),
                    "frame_index": captured.frame_index,
                },
            )

        # Publish for downstream agents (async to avoid blocking segmentation thread)
        if self._publish_executor is not None:
            self._publish_executor.submit(self._bus.publish, percept)
        else:
            self._bus.publish(percept)

    def process_captured_frame(self, captured: CapturedFrame) -> Percept:
        """Process a captured frame into a Percept (public API).

        Use this for manual frame processing outside the callback flow.
        """
        detections = captured.detections
        salience = self._compute_salience(detections)
        novelty = self._compute_novelty(detections)
        source = "vision" if detections else "idle"

        percept = Percept(
            timestamp=captured.timestamp,
            source=source,
            detections=detections,
            transcript_chunk=None,
            transcript_chunk_index=None,
            cli_input=None,
            salience=salience,
            novelty=novelty,
            has_voice_command=False,
            has_maxim_keyword=False,
            hard_override=None,
            raw_transcript_text=None,
            maxim_runtime=None,
        )

        self._recent_percepts.append(percept)
        return percept

    def on_stop(self, **kwargs: Any) -> None:
        """Clean up on stop."""
        self._recent_percepts.clear()

        # Shutdown publish executor
        if self._publish_executor is not None:
            self._publish_executor.shutdown(wait=False)
            self._publish_executor = None
