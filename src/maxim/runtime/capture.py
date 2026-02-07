"""Unified capture manager for frame and audio data.

Phase 3: Centralizes all media capture for the agentic runtime.
Provides direct frame/audio access without JSONL intermediary.
"""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from maxim.conscience.selfy import Maxim

logger = logging.getLogger(__name__)


@dataclass
class CapturedFrame:
    """A captured video frame with metadata."""

    frame: np.ndarray
    timestamp: float
    frame_index: int
    detections: list[dict[str, Any]] = field(default_factory=list)
    segmented: bool = False
    pending_segmentation: bool = False  # True if queued for async segmentation


@dataclass
class CapturedAudio:
    """A captured audio sample with metadata."""

    samples: np.ndarray
    timestamp: float
    sample_rate: int


class CaptureManager:
    """Manages frame and audio capture for the agentic runtime.

    Provides:
    - Direct frame access (no JSONL polling)
    - Optional YOLO segmentation on frames
    - Audio sample streaming
    - Callbacks for new data
    """

    def __init__(
        self,
        maxim: Maxim | None = None,
        *,
        frame_queue_size: int = 2,
        audio_queue_size: int = 64,
        target_fps: float = 10.0,
        enable_segmentation: bool = True,
    ) -> None:
        """Initialize capture manager.

        Args:
            maxim: Maxim instance for hardware access (optional for testing)
            frame_queue_size: Max frames to buffer
            audio_queue_size: Max audio samples to buffer
            target_fps: Target frame capture rate
            enable_segmentation: Whether to run YOLO on frames
        """
        self._maxim = maxim
        self._target_fps = target_fps
        self._enable_segmentation = enable_segmentation

        # Queues for captured data
        self._frame_queue: queue.Queue[CapturedFrame] = queue.Queue(maxsize=frame_queue_size)
        self._audio_queue: queue.Queue[CapturedAudio] = queue.Queue(maxsize=audio_queue_size)

        # Latest data (for polling access)
        self._latest_frame: CapturedFrame | None = None
        self._latest_audio: CapturedAudio | None = None
        self._frame_lock = threading.Lock()
        self._audio_lock = threading.Lock()

        # Callbacks
        self._frame_callbacks: list[Callable[[CapturedFrame], None]] = []
        self._audio_callbacks: list[Callable[[CapturedAudio], None]] = []
        self._segmented_frame_callbacks: list[Callable[[CapturedFrame], None]] = []

        # Control
        self._stop_event = threading.Event()
        self._frame_thread: threading.Thread | None = None
        self._audio_thread: threading.Thread | None = None
        self._segmentation_thread: threading.Thread | None = None
        self._frame_index = 0

        # Async segmentation queue (decouple from capture for smooth frame pull)
        # Size 8 provides better buffering when YOLO or callbacks are slow
        # Increased from 4 to prevent frame drops during YOLO processing spikes
        self._segmentation_queue: queue.Queue[CapturedFrame] = queue.Queue(maxsize=8)

        # Thread pool for timeout-wrapped hardware calls
        self._hw_executor: concurrent.futures.ThreadPoolExecutor | None = None
        # Timeout for hardware calls (prevents indefinite blocking)
        self._hw_timeout = 2.0

        # Stats
        self._frames_captured = 0
        self._audio_samples_captured = 0
        self._start_time: float | None = None

    def start(self) -> None:
        """Start capture threads."""
        if self._maxim is None:
            logger.warning("CaptureManager started without Maxim instance - no hardware capture")
            return

        self._stop_event.clear()
        self._start_time = time.time()
        self._frame_index = 0

        # Create thread pool for hardware calls with timeout
        self._hw_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="HWCapture"
        )

        # Start frame capture thread
        self._frame_thread = threading.Thread(
            target=self._frame_capture_loop,
            name="agentic.capture.frame",
            daemon=True,
        )
        self._frame_thread.start()

        # Start async segmentation thread (decoupled from capture to prevent buffer underrun)
        if self._enable_segmentation:
            self._segmentation_thread = threading.Thread(
                target=self._segmentation_loop,
                name="agentic.capture.segmentation",
                daemon=True,
            )
            self._segmentation_thread.start()

        # Start audio capture thread if audio enabled
        if getattr(self._maxim, "audio", False):
            self._audio_thread = threading.Thread(
                target=self._audio_capture_loop,
                name="agentic.capture.audio",
                daemon=True,
            )
            self._audio_thread.start()

        logger.info("CaptureManager started (fps=%.1f, segmentation=%s, async=%s)",
                    self._target_fps, self._enable_segmentation, self._enable_segmentation)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop capture threads."""
        self._stop_event.set()

        # Shutdown hardware executor first to unblock any pending hardware calls
        if self._hw_executor is not None:
            try:
                self._hw_executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                # Python < 3.9 doesn't have cancel_futures
                self._hw_executor.shutdown(wait=False)
            self._hw_executor = None

        if self._frame_thread is not None:
            self._frame_thread.join(timeout=timeout)
            if self._frame_thread.is_alive():
                logger.warning("Frame capture thread did not stop in time")
            self._frame_thread = None

        if self._segmentation_thread is not None:
            # Unblock segmentation queue
            try:
                self._segmentation_queue.put_nowait(None)  # type: ignore
            except queue.Full:
                pass
            self._segmentation_thread.join(timeout=timeout)
            if self._segmentation_thread.is_alive():
                logger.warning("Segmentation thread did not stop in time")
            self._segmentation_thread = None

        if self._audio_thread is not None:
            self._audio_thread.join(timeout=timeout)
            if self._audio_thread.is_alive():
                logger.warning("Audio capture thread did not stop in time")
            self._audio_thread = None

        logger.info(
            "CaptureManager stopped (frames=%d, audio_samples=%d)",
            self._frames_captured,
            self._audio_samples_captured,
        )

    def _frame_capture_loop(self) -> None:
        """Frame capture worker.

        IMPORTANT: This loop must pull frames quickly to prevent WebRTC buffer overflow.
        Segmentation runs in a separate thread to avoid blocking frame capture.
        """
        min_period = 1.0 / self._target_fps
        last_capture = 0.0

        while not self._stop_event.is_set():
            now = time.time()
            if now - last_capture < min_period:
                time.sleep(0.001)
                continue

            frame = self._capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            last_capture = now
            self._frame_index += 1
            self._frames_captured += 1

            # Create frame object (segmentation runs async)
            captured = CapturedFrame(
                frame=frame,
                timestamp=now,
                frame_index=self._frame_index,
                detections=[],  # Will be filled by segmentation thread
                segmented=False,
                pending_segmentation=self._enable_segmentation,
            )

            # Update latest
            with self._frame_lock:
                self._latest_frame = captured

            # Queue for async segmentation (non-blocking, drop oldest if full)
            if self._enable_segmentation:
                try:
                    self._segmentation_queue.put_nowait(captured)
                except queue.Full:
                    # Drop oldest frame to keep up
                    try:
                        self._segmentation_queue.get_nowait()
                        self._segmentation_queue.put_nowait(captured)
                    except queue.Empty:
                        pass

            # Put in output queue (non-blocking, drop oldest if full)
            try:
                self._frame_queue.put_nowait(captured)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                    self._frame_queue.put_nowait(captured)
                except queue.Empty:
                    pass

            # Notify callbacks (with unsegmented frame - they'll get updates via latest)
            for callback in self._frame_callbacks:
                try:
                    callback(captured)
                except Exception as e:
                    logger.error("Frame callback error: %s", e)

    def _segmentation_loop(self) -> None:
        """Async segmentation worker.

        Runs YOLO in a separate thread to avoid blocking frame capture.
        Updates the latest frame with detection results.
        """
        while not self._stop_event.is_set():
            try:
                # Block waiting for frame (with timeout for stop check)
                try:
                    captured = self._segmentation_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if captured is None:  # Poison pill
                    break

                # Run segmentation
                detections = self._segment_frame(captured.frame)

                # Update the captured frame with detections
                captured.detections = detections
                captured.segmented = True
                captured.pending_segmentation = False

                # Update latest frame with segmented version
                with self._frame_lock:
                    # Only update if this is still the latest or newer
                    if self._latest_frame is None or captured.frame_index >= self._latest_frame.frame_index:
                        self._latest_frame = captured

                # Notify segmentation-complete callbacks (for DefaultNetwork/PerceptionAgent)
                for callback in self._segmented_frame_callbacks:
                    try:
                        callback(captured)
                    except Exception as e:
                        logger.error("Segmented frame callback error: %s", e)

            except Exception as e:
                logger.debug("Segmentation loop error: %s", e)

    def _audio_capture_loop(self) -> None:
        """Audio capture worker."""
        while not self._stop_event.is_set():
            audio = self._capture_audio()
            if audio is None:
                time.sleep(0.01)
                continue

            self._audio_samples_captured += 1

            # Update latest
            with self._audio_lock:
                self._latest_audio = audio

            # Put in queue (non-blocking)
            try:
                self._audio_queue.put_nowait(audio)
            except queue.Full:
                try:
                    self._audio_queue.get_nowait()
                    self._audio_queue.put_nowait(audio)
                except queue.Empty:
                    pass

            # Notify callbacks
            for callback in self._audio_callbacks:
                try:
                    callback(audio)
                except Exception as e:
                    logger.error("Audio callback error: %s", e)

    def _capture_frame(self) -> np.ndarray | None:
        """Capture a frame from the camera with timeout protection."""
        if self._maxim is None or self._stop_event.is_set():
            return None

        executor = self._hw_executor
        if executor is None:
            return None

        def _do_capture() -> np.ndarray | None:
            mini = getattr(self._maxim, "mini", None)
            if mini is None:
                return None

            media_lock = getattr(self._maxim, "_media_lock", None)
            if media_lock is not None:
                # Use timeout on lock acquisition
                if not media_lock.acquire(timeout=self._hw_timeout):
                    return None
                try:
                    frame = mini.media.get_frame()
                finally:
                    media_lock.release()
            else:
                frame = mini.media.get_frame()

            if frame is None or (hasattr(frame, "size") and frame.size == 0):
                return None

            return np.asarray(frame)

        try:
            future = executor.submit(_do_capture)
            # Poll with short timeout to check stop_event
            timeout_remaining = self._hw_timeout
            poll_interval = 0.1
            while timeout_remaining > 0:
                if self._stop_event.is_set():
                    future.cancel()
                    return None
                try:
                    return future.result(timeout=min(poll_interval, timeout_remaining))
                except concurrent.futures.TimeoutError:
                    timeout_remaining -= poll_interval
                    continue
            # Timeout exceeded
            future.cancel()
            return None
        except concurrent.futures.CancelledError:
            return None
        except Exception as e:
            logger.debug("Frame capture failed: %s", e)
            return None

    def _capture_audio(self) -> CapturedAudio | None:
        """Capture an audio sample with timeout protection."""
        if self._maxim is None or self._stop_event.is_set():
            return None

        executor = self._hw_executor
        if executor is None:
            return None

        def _do_capture() -> CapturedAudio | None:
            mini = getattr(self._maxim, "mini", None)
            if mini is None:
                return None

            media_lock = getattr(self._maxim, "_media_lock", None)
            if media_lock is not None:
                # Use timeout on lock acquisition
                if not media_lock.acquire(timeout=self._hw_timeout):
                    return None
                try:
                    sample = mini.media.get_audio_sample()
                    sample_rate = mini.media.get_output_audio_samplerate()
                finally:
                    media_lock.release()
            else:
                sample = mini.media.get_audio_sample()
                sample_rate = mini.media.get_output_audio_samplerate()

            if sample is None or len(sample) == 0:
                return None

            return CapturedAudio(
                samples=np.asarray(sample),
                timestamp=time.time(),
                sample_rate=int(sample_rate),
            )

        try:
            future = executor.submit(_do_capture)
            # Poll with short timeout to check stop_event
            timeout_remaining = self._hw_timeout
            poll_interval = 0.1
            while timeout_remaining > 0:
                if self._stop_event.is_set():
                    future.cancel()
                    return None
                try:
                    return future.result(timeout=min(poll_interval, timeout_remaining))
                except concurrent.futures.TimeoutError:
                    timeout_remaining -= poll_interval
                    continue
            # Timeout exceeded
            future.cancel()
            return None
        except concurrent.futures.CancelledError:
            return None
        except Exception as e:
            logger.debug("Audio capture failed: %s", e)
            return None

    # Target coordinate system for behaviors (matches look_at_image expectations)
    _TARGET_WIDTH = 640.0
    _TARGET_HEIGHT = 480.0

    def _segment_frame(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run YOLO segmentation on frame.

        Returns detections with bounding boxes scaled to 640x480 coordinate system,
        regardless of the camera's native resolution.
        """
        if self._maxim is None:
            return []

        segmenter = getattr(self._maxim, "segmenter", None)
        if segmenter is None:
            return []

        try:
            observations = segmenter.segment_photos(
                [frame],
                interests=list(getattr(self._maxim, "interests", []) or []),
            )

            detections = []
            names = getattr(getattr(segmenter, "model", None), "names", None)

            # Get actual frame dimensions for coordinate scaling
            frame_height, frame_width = frame.shape[:2]
            scale_x = self._TARGET_WIDTH / frame_width
            scale_y = self._TARGET_HEIGHT / frame_height

            # DEBUG: Log camera resolution (only occasionally to avoid spam)
            import time
            if not hasattr(self, "_last_res_log_time") or (time.time() - self._last_res_log_time) > 30:
                self._last_res_log_time = time.time()
                self._log.warning(
                    "CAPTURE_DEBUG: camera_resolution=%dx%d, target=%dx%d, scale_x=%.3f, scale_y=%.3f",
                    frame_width, frame_height,
                    int(self._TARGET_WIDTH), int(self._TARGET_HEIGHT),
                    scale_x, scale_y
                )

            for obs in observations or []:
                if not isinstance(obs, (list, tuple)) or len(obs) < 8:
                    continue

                try:
                    track_id = int(obs[0]) if obs[0] is not None else None
                except Exception:
                    track_id = None

                try:
                    cls_id = int(obs[7]) if obs[7] is not None else None
                except Exception:
                    cls_id = None

                try:
                    conf = float(obs[6]) if obs[6] is not None else 0.0
                except Exception:
                    conf = 0.0

                label = None
                try:
                    if isinstance(names, dict) and cls_id in names:
                        label = names.get(cls_id)
                    elif isinstance(names, (list, tuple)) and cls_id is not None:
                        if 0 <= cls_id < len(names):
                            label = names[cls_id]
                except Exception:
                    pass

                # Scale bounding box from camera resolution to 640x480
                x1 = float(obs[2]) * scale_x
                y1 = float(obs[3]) * scale_y
                x2 = float(obs[4]) * scale_x
                y2 = float(obs[5]) * scale_y

                detections.append({
                    "track_id": track_id,
                    "class_id": cls_id,
                    "label": label,
                    "conf": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                })

            return detections
        except Exception as e:
            logger.debug("Segmentation failed: %s", e)
            return []

    # Public API

    def get_latest_frame(self) -> CapturedFrame | None:
        """Get the most recent captured frame."""
        with self._frame_lock:
            return self._latest_frame

    def get_latest_audio(self) -> CapturedAudio | None:
        """Get the most recent captured audio."""
        with self._audio_lock:
            return self._latest_audio

    def get_frame(self, timeout: float = 0.1) -> CapturedFrame | None:
        """Get next frame from queue (blocking)."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_audio(self, timeout: float = 0.1) -> CapturedAudio | None:
        """Get next audio sample from queue (blocking)."""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def on_frame(self, callback: Callable[[CapturedFrame], None]) -> None:
        """Register callback for new frames (fires before segmentation)."""
        self._frame_callbacks.append(callback)

    def on_segmented_frame(self, callback: Callable[[CapturedFrame], None]) -> None:
        """Register callback for segmented frames (fires after YOLO completes).

        Use this callback instead of on_frame() when you need detection data.
        The DefaultNetwork and PerceptionAgent should use this for reactive behaviors.
        """
        self._segmented_frame_callbacks.append(callback)

    def on_audio(self, callback: Callable[[CapturedAudio], None]) -> None:
        """Register callback for new audio."""
        self._audio_callbacks.append(callback)

    @property
    def is_running(self) -> bool:
        """Check if capture is running."""
        return not self._stop_event.is_set() and (
            (self._frame_thread is not None and self._frame_thread.is_alive()) or
            (self._audio_thread is not None and self._audio_thread.is_alive())
        )

    @property
    def stats(self) -> dict[str, Any]:
        """Get capture statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        return {
            "frames_captured": self._frames_captured,
            "audio_samples_captured": self._audio_samples_captured,
            "elapsed_seconds": elapsed,
            "effective_fps": self._frames_captured / elapsed if elapsed > 0 else 0,
            "is_running": self.is_running,
        }

    def remove_frame_callback(self, callback: Callable[[CapturedFrame], None]) -> None:
        """Remove a frame callback."""
        try:
            self._frame_callbacks.remove(callback)
        except ValueError:
            pass

    def remove_audio_callback(self, callback: Callable[[CapturedAudio], None]) -> None:
        """Remove an audio callback."""
        try:
            self._audio_callbacks.remove(callback)
        except ValueError:
            pass

    def remove_segmented_frame_callback(self, callback: Callable[[CapturedFrame], None]) -> None:
        """Remove a segmented frame callback."""
        try:
            self._segmented_frame_callbacks.remove(callback)
        except ValueError:
            pass
