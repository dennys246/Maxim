from __future__ import annotations

import logging
import math
import os
import signal
import threading
import time
import weakref
from typing import Any, Callable, Optional

import numpy as np
from scipy.signal import resample, resample_poly

logger = logging.getLogger(__name__)


def _to_device_float32(samples: np.ndarray) -> np.ndarray:
    """Normalize samples to float32 in [-1.0, 1.0] for a float32 audio sink.

    The single conversion used by BOTH the local playback path and the Reachy
    device path. Piper TTS returns ``int16`` while ``reachy_mini``'s
    ``push_audio_sample`` is typed ``NDArray[float32]`` and wraps ``.tobytes()``
    with no conversion — pushing int16 straight through reinterprets the bytes as
    float32 (noise at half duration). This is that missing conversion, in one place.

    Strict about scale (#1435, pulse's R6): float input must already be finite and in
    [-1, 1]. A float array still on the int16 scale (what ``resample_poly`` returns for
    int16 input) used to pass through and hard-clip on the device; it is now refused,
    never guessed. Integer input other than int16 is refused for the same reason.

    Raises:
        ValueError: non-finite or out-of-range float input, or an unsupported dtype.
    """
    arr = np.asarray(samples)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    if np.issubdtype(arr.dtype, np.floating):
        out = arr.astype(np.float32)
        if out.size and not np.all(np.isfinite(out)):
            raise ValueError("audio samples contain NaN or inf")
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 1.0 + 1e-3:
            raise ValueError(f"float audio must be in [-1, 1]; peak is {peak:.1f} (int16 scale?)")
        return out
    raise ValueError(f"unsupported audio dtype {arr.dtype}; expected int16 or float in [-1, 1]")


# ─────────────────────────────────────────────────────────────────────────────
# Local Audio Playback
# ─────────────────────────────────────────────────────────────────────────────

# Global state for local audio playback
_local_audio_available: bool | None = None
_local_audio_lock = threading.Lock()


def _check_local_audio() -> bool:
    """Whether local audio playback can actually play: sounddevice imports AND an output device exists.

    Truthful for #1437 (R1): on Linux, ``import sounddevice`` raises ``OSError`` (not ``ImportError``)
    when the PortAudio library is missing, which used to escape this check, and a machine with
    sounddevice but no output device (a headless box) used to report available. Cached per process.
    """
    global _local_audio_available
    if _local_audio_available is not None:
        return _local_audio_available

    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:  # OSError: the PortAudio system library is missing
        _local_audio_available = False
        logger.debug("Local audio not available (sounddevice: %s)", e)
        return False
    try:
        sd.query_devices(kind="output")
    except (sd.PortAudioError, ValueError) as e:  # no default output device
        _local_audio_available = False
        logger.debug("Local audio not available (no output device: %s)", e)
        return False
    _local_audio_available = True
    logger.debug("Local audio playback available (sounddevice)")
    return True


def play_audio_local(
    samples: np.ndarray,
    sample_rate: int = 16000,
    blocking: bool = False,
) -> bool:
    """Play audio through local speakers (Mac/PC).

    Args:
        samples: Audio samples as numpy array (int16 or float).
        sample_rate: Sample rate in Hz (default 16000).
        blocking: If True, wait for playback to complete.

    Returns:
        True if playback started successfully, False otherwise.
    """
    if not _check_local_audio():
        logger.warning(
            "Local audio not available (sounddevice, the PortAudio library, or an output device is missing). "
            "Install with: pip install 'pymaxim[tts]'; on Linux also the PortAudio library (apt install libportaudio2)."
        )
        return False

    if samples is None or len(samples) == 0:
        return False

    try:
        import sounddevice as sd

        # Convert to float32 for sounddevice (expects -1.0 to 1.0)
        audio_float = _to_device_float32(samples)

        # Play audio
        with _local_audio_lock:
            sd.play(audio_float, samplerate=sample_rate)
            if blocking:
                sd.wait()

        return True

    except Exception as e:
        logger.warning("Local audio playback failed: %s", e)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Reachy Mini speaker: the player's lifecycle and the sink (#1435)
# ─────────────────────────────────────────────────────────────────────────────
#
# The SDK's ``push_audio_sample`` returns None whether or not anything plays: before
# ``start_playing()`` the LOCAL pipeline drops the buffer, with ``media.audio is None``
# it drops after a log line, and an unconnected WebRTC send chain drops silently. No
# public SDK query says whether the player is started, so the lifecycle owner RECORDS
# it here and the sink reads the record (docs/plans/speaker_sink_1435.md, D1). The
# record is keyed on the MediaManager object: ``acquire_media()`` and a reconnect
# create a new manager, which has no record; ``close()``/``release_media()`` keep the
# same object with ``audio = None``, which the sink also checks, and maxim's close paths
# call ``stop_device_player``. The SDK's ``ReachyMini.cancel_move()`` stops the player
# behind the record's back: maxim never calls it, and must not without
# ``stop_device_player``. The device plan's step 3 (``AudioStream`` as the port, PLAYS)
# absorbs this record (docs/plans/embodied_device.md).

# media manager -> the device's output rate, read when the player was started
_PLAYERS: "weakref.WeakKeyDictionary[Any, int]" = weakref.WeakKeyDictionary()
# managers already warned about: one WARNING per manager until its next start, so a
# shutdown or a long WebRTC session cannot flood
_WARNED: "weakref.WeakSet[Any]" = weakref.WeakSet()
# Guards the record across "check -> push" (F14). Never call out (on_failure, logging
# handlers aside) while holding it: the runtime's media lock is taken inside callbacks.
_PLAYERS_LOCK = threading.RLock()

_LOCAL, _WEBRTC, _NO_MEDIA = "local", "webrtc", "no_media"


def _backend_of(media: Any) -> str | None:
    """The media manager's backend as the SDK's ``MediaBackend`` value, or None if it has none."""
    backend = getattr(media, "backend", None)
    value = getattr(backend, "value", backend)
    return value if isinstance(value, str) else None


def _webrtc_send_chain_ready(media: Any) -> bool:
    """Whether a WebRTC media manager's audio send chain has been BUILT (owner decision 2026-10-24).

    The ONE read of SDK private state in this module: ``GstWebRTCClient`` sets
    ``_appsrc`` only when the peer connection adds the send chain, and its
    ``push_audio_sample`` silently returns while it is None. No public query exists.
    Known gap: the SDK (1.8.3) never clears ``_appsrc``, so after the peer drops
    mid-session this stays True and a push is a silent drop reported as played; the
    device plan's step 3 (PLAYS) owns closing it. (For LOCAL, ``_appsrc`` exists before
    ``start_playing``, which is why the record, not this, is LOCAL's truth.) The
    attribute's shape is pinned by a contract test against the installed SDK
    (tests/unit/test_reachy_speaker_sink.py); it skips in CI until the R2 reachy cell
    exists (mechanization backlog M69) and runs wherever the SDK is installed.
    """
    return getattr(getattr(media, "audio", None), "_appsrc", None) is not None


def _warn_once(media: Any, log: logging.Logger, msg: str, *args: Any) -> None:
    with _PLAYERS_LOCK:
        try:
            first = media not in _WARNED
            _WARNED.add(media)
        except TypeError:  # not weak-referenceable: warn every time rather than never
            first = True
    if first:
        log.warning(msg, *args)


def start_device_player(mini: Any, *, log: logging.Logger | None = None) -> bool:
    """Start the Reachy Mini's speaker player once, with the head wobbler disabled first.

    Call this from whoever holds the SDK handle, once at media start: the embodied
    runtime does (``MediaLoopMixin``), and a bare-SDK embedder (pulse's announcer
    bootstrap) must, before handing ``mini`` to :func:`make_device_speak_sink`, and
    call :func:`stop_device_player` last. The sink never starts the player.

    Order (owner decision 2026-10-24): ``mini.disable_wobbling()`` FIRST, which turns
    off both the SDK-side wobbler and the daemon-side one (driven by incoming WebRTC
    audio), so ``start_playing()`` can never start a wobbler that turns speech into head
    motion (docs/agents/embodiment.md §1). If the handle has no ``disable_wobbling`` or
    it raises, the player is not started. A repeat call re-issues the disable and does
    not start the player a second time.

    Works for the LOCAL backend and for WebRTC (whose ``start_playing`` is a no-op: the
    send chain starts on connection, which the sink checks per push).

    Returns True when the player is started and recorded; False (with a WARNING) when
    audio is not initialized, the backend is neither LOCAL nor WebRTC, the wobbler
    cannot be disabled, or the device reports no output rate. Never raises.
    """
    log = log or logger
    media = getattr(mini, "media", None) if mini is not None else None
    if media is None or getattr(media, "audio", None) is None:
        log.warning("Reachy speaker not started: the SDK's audio system is not initialized")
        return False
    backend = _backend_of(media)
    if backend not in (_LOCAL, _WEBRTC):
        log.warning("Reachy speaker not started: the %s media backend has no speaker player", backend)
        return False
    disable = getattr(mini, "disable_wobbling", None)
    if not callable(disable):
        log.warning(
            "Reachy speaker not started: this SDK handle has no disable_wobbling(), so speech could move the head"
        )
        return False
    try:
        disable()
    except Exception as e:
        log.warning("Reachy speaker not started: disable_wobbling() failed (%s), so speech could move the head", e)
        return False
    try:
        rate = int(media.get_output_audio_samplerate())
    except Exception as e:
        log.warning("Reachy speaker not started: could not read the output sample rate (%s)", e)
        return False
    if rate <= 0:
        log.warning("Reachy speaker not started: the device reports no output sample rate (%d)", rate)
        return False
    try:
        weakref.ref(media)  # the record is weak-keyed (and hashed): probe BEFORE starting, never after
        hash(media)
    except TypeError:
        log.warning("Reachy speaker not started: %r cannot hold a player record", type(media).__name__)
        return False
    with _PLAYERS_LOCK:
        if media not in _PLAYERS:
            try:
                media.start_playing()
            except Exception as e:
                log.warning("Reachy speaker not started: start_playing() failed (%s)", e)
                return False
        _PLAYERS[media] = rate
        _WARNED.discard(media)
    log.info("Reachy speaker started (%s, %d Hz, head wobbler off)", backend, rate)
    return True


def stop_device_player(mini: Any, *, log: logging.Logger | None = None) -> None:
    """Forget the started player and stop it. Call it last, and before closing media.

    Safe on a player that was never started and on a closed manager. Never raises.
    """
    log = log or logger
    media = getattr(mini, "media", None) if mini is not None else None
    if media is None:
        return
    with _PLAYERS_LOCK:
        try:
            started = _PLAYERS.pop(media, None) is not None
        except TypeError:  # never recordable, so never started here
            return
    if started and getattr(media, "audio", None) is not None:
        try:
            media.stop_playing()
        except Exception as e:
            log.warning("Failed to stop the Reachy speaker: %s", e)


def _push_to_started_player(
    media: Any,
    samples: np.ndarray,
    sample_rate: int,
    log: logging.Logger,
    on_failure: Callable[[Exception], None] | None,
) -> bool:
    """The one device push: normalize the scale, resample to the recorded device rate, push.

    Refuses (False + one WARNING per manager) when no player was started for ``media``
    or its audio was closed. The conversion runs outside the record lock; the lock is
    held across the record re-check and the push (non-blocking in the SDK), so
    ``stop_device_player`` cannot slip between them, and released before
    ``on_failure`` runs (it may take the runtime's media lock to reconnect).
    """
    with _PLAYERS_LOCK:
        try:
            rate = _PLAYERS.get(media)
        except TypeError:  # not weak-referenceable, so start_device_player never recorded it
            rate = None
    if rate is None or getattr(media, "audio", None) is None:
        _warn_once(
            media,
            log,
            "Reachy speaker refused: the player is not started (call "
            "maxim.utils.audio.start_device_player(mini) once at media start); nothing played",
        )
        return False
    try:
        rate_in = 0 if isinstance(sample_rate, bool) else int(sample_rate)
    except (TypeError, ValueError, OverflowError):
        rate_in = 0
    if rate_in <= 0:
        log.warning("Reachy speaker refused: the samples' rate %r is not a rate", sample_rate)
        return False
    try:
        pcm = _to_device_float32(samples)
        if rate_in != rate:
            pcm = np.clip(resample_audio(pcm, rate_in, rate), -1.0, 1.0).astype(np.float32)
    except ValueError as e:
        log.warning("Reachy speaker refused: %s", e)
        return False
    error: Exception | None = None
    with _PLAYERS_LOCK:
        # stop_device_player, a close or a restart at another rate during the conversion
        if _PLAYERS.get(media) != rate or getattr(media, "audio", None) is None:
            log.warning("Reachy speaker refused: the player stopped before the push; nothing played")
            return False
        try:
            media.push_audio_sample(pcm)
        except Exception as e:
            error = e
    if error is not None:
        log.warning("Failed to play audio on Reachy: %s", error)
        if on_failure is not None:
            on_failure(error)
        return False
    return True


def push_to_started_player(
    mini: Any, samples: np.ndarray, *, sample_rate: int, log: logging.Logger | None = None
) -> bool:
    """Push samples to a started Reachy player, resampling from ``sample_rate``; no local fallback.

    The device half of :func:`push_audio_to_device`, for the ``AudioStream.push_sample``
    contract (a stream is the device or nothing). Refuses, False with a WARNING, exactly
    as the sink does. Never raises.
    """
    log = log or logger
    if mini is None or samples is None or np.ndim(samples) == 0 or len(samples) == 0:
        return False
    media = getattr(mini, "media", None)
    if media is None:
        return False
    return _push_to_started_player(media, samples, sample_rate, log, None)


def push_audio_to_device(
    mini: Any,
    samples: np.ndarray,
    sample_rate: int,
    *,
    log: logging.Logger | None = None,
    on_failure: Callable[[Exception], None] | None = None,
) -> bool:
    """Play audio on a Reachy Mini's speaker, or say truthfully that it did not.

    The single device-speak implementation, needing ONLY the SDK handle ``mini`` — no
    embodied runtime, no agent. Used by :meth:`MediaLoopMixin.speak` and by an embedder
    that holds a bare ``reachy_mini`` before any agent exists (e.g. the spoken-code
    pairing announcer, which must speak before the SetupWizard runs). The player must
    have been started by :func:`start_device_player`.

    Decision table (owner decisions 2026-10-24, docs/plans/speaker_sink_1435.md):

    - ``MAXIM_TTS_LOCAL`` set, or ``mini is None`` → local speakers;
    - LOCAL backend, player started → normalized to float32 in [-1, 1], resampled from
      ``sample_rate`` to the device's rate, pushed; True;
    - LOCAL, player not started or audio closed → False with a WARNING; nothing plays,
      and there is no hop to local speakers (that was the silent lie of #1435);
    - WebRTC (off-robot), player started and its send chain built → pushed to the robot
      exactly as LOCAL (known gap: the SDK never clears the chain after a peer drop, so
      a push then is a silent drop); started but no chain yet, or never started → local
      speakers, with one WARNING;
    - ``no_media`` → local speakers, with one WARNING;
    - any other backend (an SDK rename, a handle without ``backend``) → False with a
      WARNING; never a silent re-route.

    Never raises.

    Args:
        mini: SDK handle exposing ``media`` (a reachy_mini ``MediaManager``).
        samples: audio samples (int16 from Piper, or float in [-1, 1]).
        sample_rate: the samples' rate; the device path resamples from it. Required: a
            silent 16 kHz default replayed #1435 for any caller that forgot it.
        log: logger for diagnostics (defaults to this module's).
        on_failure: optional callback for a device push error (e.g. the mixin's
            connection-failure note); the exception is passed to it.
    """
    log = log or logger
    if samples is None or np.ndim(samples) == 0 or len(samples) == 0:
        return False

    prefer_local = os.environ.get("MAXIM_TTS_LOCAL", "").lower() in ("1", "true", "yes")

    if not prefer_local and mini is not None:
        media = getattr(mini, "media", None)
        backend = _backend_of(media)
        if backend == _LOCAL:
            return _push_to_started_player(media, samples, sample_rate, log, on_failure)
        if backend == _WEBRTC:
            with _PLAYERS_LOCK:
                try:
                    started = _PLAYERS.get(media) is not None
                except TypeError:
                    started = False
            if started and _webrtc_send_chain_ready(media):
                return _push_to_started_player(media, samples, sample_rate, log, on_failure)
            _warn_once(
                media,
                log,
                "Reachy speaker unavailable over WebRTC (%s); playing on local speakers",
                "send chain not built yet" if started else "player not started by start_device_player",
            )
        elif backend == _NO_MEDIA:
            _warn_once(media, log, "Reachy has no media backend (no_media); playing on local speakers")
        else:
            log.warning("Reachy speaker refused: unknown media backend %r; nothing played", backend)
            return False

    try:
        success = play_audio_local(samples, sample_rate=sample_rate, blocking=False)
        if success:
            log.debug("Playing audio through local speakers")
        return success
    except Exception as e:
        log.warning("Local audio playback failed: %s", e)
        return False


def make_device_speak_sink(mini: Any) -> Callable[..., bool]:
    """Bind a Reachy Mini handle into a ``speak(samples, *, sample_rate) -> bool`` sink.

    Construction is inert — it only closes over ``mini`` and touches no audio device
    until called — so it is safe to build offline / at import time. Intended for a
    bare-SDK embedder that owns the handle's lifecycle (#1435)::

        start_device_player(mini)            # once: wobbler off, then the player
        announcer = make_pairing_announcer(tts, make_device_speak_sink(mini))
        ...
        stop_device_player(mini)             # last, before closing media

    Without the start the sink refuses (False + WARNING) rather than report a silent drop.
    """

    def speak(samples: np.ndarray, *, sample_rate: int) -> bool:
        return push_audio_to_device(mini, samples, sample_rate)

    return speak


def run_in_own_process_group(target: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Child-process entry: leave the terminal's process group, then run ``target(*args, **kwargs)``.

    For the robot's spawned transcriber (#1436): in the terminal's foreground group a Ctrl-C
    reached the child first, it exited before the parent noticed the stop, and the audio writer
    reported "Voice input is OFF" on every ordinary shutdown. In its own group only the parent's
    teardown stops it (the shutdown file, then SIGTERM, then SIGKILL), or the parent's death: leaving
    the group also stops the terminal's SIGHUP from reaching it, so a watchdog thread SIGTERMs the
    child once its parent is gone (a hard-killed parent never runs its teardown, and an orphaned
    transcriber would keep its model, and GPU memory, loaded). A module-level function, so a spawn
    context can pickle it; it lives here, not beside its caller, so the child does not import the
    embodied runtime. No-op grouping where ``os.setpgrp`` does not exist.
    """
    setpgrp = getattr(os, "setpgrp", None)
    if setpgrp is not None:
        setpgrp()
    import multiprocessing

    # The spawning parent's pid, not getppid(): if the parent died while this child was still
    # importing, getppid() would already name the reaper and the watchdog would never fire.
    spawner = multiprocessing.parent_process()
    parent = spawner.pid if spawner is not None and spawner.pid else os.getppid()

    def _exit_when_orphaned() -> None:
        while os.getppid() == parent:
            time.sleep(1.0)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=_exit_when_orphaned, name="transcriber-parent-watch", daemon=True).start()
    return target(*args, **kwargs)


def stop_local_audio() -> None:
    """Stop any currently playing local audio."""
    if not _check_local_audio():
        return

    try:
        import sounddevice as sd

        sd.stop()
    except Exception:
        pass


class LocalSpeaker:
    """Speaker that plays audio through local computer speakers.

    Can be used as a drop-in replacement for Reachy speaker when
    running remotely or for testing.
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        """Initialize local speaker.

        Args:
            sample_rate: Expected sample rate of audio.
        """
        self.sample_rate = sample_rate
        self._available = _check_local_audio()

    @property
    def is_available(self) -> bool:
        """Check if local audio is available."""
        return self._available

    def __call__(self, samples: np.ndarray, sample_rate: int | None = None) -> bool:
        """Play audio samples (the ``speaker_fn(samples, sample_rate=...)`` sink shape).

        Args:
            samples: Audio samples to play.
            sample_rate: The samples' rate; defaults to this speaker's ``sample_rate``.

        Returns:
            True if playback started successfully.
        """
        rate = self.sample_rate if sample_rate is None else int(sample_rate)
        return play_audio_local(samples, rate, blocking=False)

    def play(self, samples: np.ndarray, blocking: bool = False) -> bool:
        """Play audio samples.

        Args:
            samples: Audio samples to play.
            blocking: If True, wait for playback to complete.

        Returns:
            True if playback started successfully.
        """
        return play_audio_local(samples, self.sample_rate, blocking=blocking)

    def stop(self) -> None:
        """Stop any currently playing audio."""
        stop_local_audio()


# ─────────────────────────────────────────────────────────────────────────────
# Audio Format Conversion
# ─────────────────────────────────────────────────────────────────────────────


def to_int16(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.int16:
        return np.ascontiguousarray(arr)
    if np.issubdtype(arr.dtype, np.floating):
        clipped = np.clip(arr, -1.0, 1.0)
        return np.ascontiguousarray((clipped * 32767.0).astype(np.int16))
    return np.ascontiguousarray(np.clip(arr, -32768, 32767).astype(np.int16))


def resample_audio(sample: np.ndarray, input_rate: Optional[int], output_rate: Optional[int]) -> np.ndarray:
    if not input_rate or not output_rate or int(input_rate) == int(output_rate):
        return sample

    try:
        gcd = math.gcd(int(input_rate), int(output_rate))
        up = int(output_rate) // gcd
        down = int(input_rate) // gcd
        return resample_poly(sample, up, down, axis=0)
    except Exception:
        num_sample = int(int(output_rate) * len(sample) / int(input_rate))
        return resample(sample, num_sample)
