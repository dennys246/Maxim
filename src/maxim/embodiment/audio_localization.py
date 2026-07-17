"""Audio sound-localization front-end (azimuth from onboard DoA).

The stock Reachy Mini's 4-mic array sits behind a Seeed XVF3800 DSP chip
that exposes only a 2-channel processed stream — there is **no
sample-aligned 4-mic raw access**, so a custom ITD/TDOA front-end is not
feasible on that hardware. Instead the chip computes Direction-of-Arrival
on-chip and the SDK exposes it; we consume that. See
``docs/embodiment/reachy_mini/audio_localization.md`` for the full hardware analysis and
``docs/plans/perception_pipeline_placement.md`` for where this sits in the
perception pipeline (commit 4 — the DoA-consumption front-end).

This module is **hardware-agnostic**: :class:`AzimuthDoASource` takes an
injected ``doa_reader`` callable, so the localization → percept logic is
fully unit-testable without a robot. :func:`make_reachy_doa_reader` is the
thin (on-device-verified) adapter that wires a live Reachy Mini's
``get_DoA()`` into that callable.

Design note — **pull-per-tick, no background DSP thread**: the chip
computes DoA continuously, so ``next_percept()`` samples the *current*
estimate when the agent decides ("localize-at-decision-time"). There is no
high-rate raw stream to decouple from here, so no background thread / RMW
stash is needed (unlike a true ITD front-end, which would require one).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

from maxim.agents.bus import Percept
from maxim.agents.percept_factory import make_audio_percept

logger = logging.getLogger(__name__)

# A DoA reader returns (doa_radians, is_speech_detected), or None when no
# reading is available yet (e.g. the media stream hasn't produced a value).
DoAReading = tuple[float, bool]
DoAReader = Callable[[], "DoAReading | None"]


def doa_to_azimuth(doa_radians: float) -> float:
    """Normalize an onboard DoA angle to a centered azimuth in ``[-1, 1]``.

    Convention (XVF3800, per Reachy Mini docs): ``0`` = left,
    ``π/2`` = front/back, ``π`` = right. Mapping ``(doa − π/2)/(π/2)``
    yields ``-1`` = left, ``0`` = centered (front), ``+1`` = right — which
    is exactly the "centeredness" set-point shape the homeostatic drive
    wants (centered = 0). Clamped to ``[-1, 1]`` defensively.

    NOTE (hardware limitation): a linear mic array has **front/back
    ambiguity** — a source directly behind reads the same as one in front
    (both ≈ π/2 → 0). Left/right discrimination is clean; front/back is
    not recoverable. See ``docs/embodiment/reachy_mini/audio_localization.md``.
    """
    az = (doa_radians - math.pi / 2.0) / (math.pi / 2.0)
    if az < -1.0:
        return -1.0
    if az > 1.0:
        return 1.0
    return az


class AzimuthDoASource:
    """``PerceptSource`` that emits an audio percept from onboard DoA.

    Conforms to the :class:`maxim.simulation.sources.PerceptSource`
    Protocol (``name`` / ``next_percept`` / ``is_exhausted`` /
    ``capabilities``). ``next_percept()`` calls the injected ``doa_reader``,
    **gates on the hardware's ``is_speech_detected`` flag** (the device's
    own "is there a sound to localize?" signal — this directly addresses
    the transient-sound problem: we only emit when a sound is actually
    present), normalizes the angle, and returns an ``"audio"`` Percept
    carrying ``metadata["azimuth"]``. Returns ``None`` on no reading / no
    speech — non-blocking, pull-per-tick.
    """

    def __init__(
        self,
        doa_reader: DoAReader,
        *,
        name: str = "reachy:audio-doa",
        agent_id: str | None = None,
    ) -> None:
        self._reader = doa_reader
        self._name = name
        self._agent_id = agent_id

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> set[str]:
        return {"audio"}

    def is_exhausted(self) -> bool:
        # Live hardware source — never exhausted.
        return False

    def next_percept(self) -> Percept | None:
        try:
            reading = self._reader()
        except Exception:
            logger.debug("AzimuthDoASource: doa_reader raised", exc_info=True)
            return None
        if reading is None:
            return None
        doa_radians, is_speech_detected = reading
        if not is_speech_detected:
            # No sound to localize this tick — don't fabricate a direction.
            return None
        azimuth = doa_to_azimuth(doa_radians)
        return make_audio_percept(azimuth, source=self._name, agent_id=self._agent_id)


def make_reachy_rest_doa_reader(
    host: str,
    *,
    port: int = 8000,
    timeout: float = 2.0,
    fetch: "Callable[[str, float], DoAReading | None] | None" = None,
) -> DoAReader:
    """Return a :data:`DoAReader` that reads DoA over the daemon's REST endpoint.

    THE OFF-ROBOT PATH. ``make_reachy_doa_reader`` reads ``mini.media.get_DoA()``,
    which is **local-USB in SDK >= 1.5** — it only works ONBOARD. When Maxim runs
    on a laptop / peer talking to the robot over the network (the usual topology),
    the client-side call returns nothing; the daemon reads the XVF3800 and serves
    the value at ``GET /api/state/doa``. Convention is unchanged from the onboard
    path (0=left, pi/2=front, pi=right), so :func:`doa_to_azimuth` applies as-is.

    Lifted from the Step-1 bring-up script (where it was duplicated inline) into
    the library so :class:`AzimuthDoASource` has a real off-robot reader.

    Network calls go through ``maxim.utils.http`` (the CI-enforced single HTTP
    surface — raw ``urllib`` is blocked outside ``utils/http.py``), matching
    ``ReachyMiniController._daemon_status``. The ``fetch`` parameter is a seam for
    tests: inject a fake to exercise the reader with NO network and NO robot (the
    same dependency-gate-inside pattern that makes ``make_reachy_doa_reader``
    CI-testable). ``None`` is a live reading this tick (no reading / no speech
    yet), never fabricated.
    """
    if fetch is not None:
        _fetch = fetch
    else:

        def _fetch(url: str, t: float) -> "DoAReading | None":
            from maxim.utils import http as maxim_http

            try:
                resp = maxim_http.fetch_url(url, timeout=t)
                data = resp.json() if hasattr(resp, "json") else None
            except Exception:
                logger.debug("REST DoA fetch failed", exc_info=True)
                return None
            if not data:
                return None
            return (float(data["angle"]), bool(data.get("speech_detected", False)))

    url = f"http://{host}:{port}/api/state/doa"

    def _read() -> DoAReading | None:
        return _fetch(url, timeout)

    return _read


def make_reachy_doa_reader(mini: object | None = None) -> DoAReader:
    """Return a :data:`DoAReader` wrapping a Reachy Mini's onboard DoA.

    If ``mini`` is ``None``, constructs a ``ReachyMini`` (which requires the
    optional ``reachy`` extra, ``pip install pymaxim[reachy]``) and starts
    its media stream so ``get_DoA()`` produces values. If a ``mini`` is
    **injected**, no ``reachy_mini`` import happens at all — the optional
    dependency is only needed to build the SDK object ourselves, so an
    injected object (a real one, or a test fake) works without the extra
    installed. This is why the adapter glue is CI-testable: the dep gate
    lives inside the ``mini is None`` branch, not at the top.

    .. warning::
        The exact ``get_DoA()`` lifecycle/return shape is taken from the
        Reachy Mini SDK docs and is **verified on-device**, not in CI. The
        hardware-agnostic core (:func:`doa_to_azimuth`,
        :class:`AzimuthDoASource`) carries the unit coverage; this adapter
        is the thin glue to confirm against a physical unit.
    """
    if mini is None:
        from maxim.utils.optional_deps import require_optional_dependency

        require_optional_dependency("reachy_mini", feature="Reachy Mini audio sound-localization")
        from reachy_mini import ReachyMini  # type: ignore[import-not-found]

        mini = ReachyMini()

    media = mini.media  # type: ignore[attr-defined]
    # Ensure the media stream is producing values for get_DoA().
    start = getattr(media, "start_recording", None)
    if callable(start):
        start()

    def _read() -> DoAReading | None:
        result = media.get_DoA()
        if result is None:
            return None
        doa_radians, is_speech_detected = result
        return (float(doa_radians), bool(is_speech_detected))

    return _read


def build_reachy_audio_orienting_source(
    *,
    connection_mode: str = "network",
    host: str | None = None,
    mini: object | None = None,
    agent_id: str | None = None,
    name: str = "reachy:audio-doa",
) -> "AzimuthDoASource | None":
    """Assemble a runtime-ready audio-orient :class:`AzimuthDoASource` for a Reachy.

    This is the ONE place that chooses the DoA transport, so the runtime wiring
    (Landing 1 step 3) doesn't carry that decision inline:

    * ``mini`` injected  -> onboard local-USB reader (``make_reachy_doa_reader``);
      the caller already holds an SDK object, so no network hop.
    * ``connection_mode`` in {"network", "auto"} with a ``host`` -> the REST reader
      (``make_reachy_rest_doa_reader``): the daemon serves DoA over the network,
      which is the off-robot topology (laptop/peer talking to the robot).
    * neither -> ``None``. **A missing transport is not an error and never a stub**
      — the caller simply gets no audio-orient percepts, exactly as a robot without
      a mic would (the capability-driven principle: absent capability => absent
      source, no dead config).

    Returns ``None`` rather than raising so the runtime can treat "no audio source"
    as a normal, capability-driven outcome — the same way vision is gated on
    ``has_vision``. Enabling/disabling the feature is the CALLER's decision (a
    config flag at the wiring layer, per the config-over-env-vars standard); this
    builder only assembles what the transport allows.

    Testable with no robot and no network: inject ``mini`` (a fake), or rely on the
    REST reader's own ``fetch`` seam downstream.
    """
    if mini is not None:
        reader = make_reachy_doa_reader(mini)
    elif connection_mode in ("network", "auto") and host:
        reader = make_reachy_rest_doa_reader(host)
    else:
        logger.debug(
            "no DoA transport (connection_mode=%r, host=%r, mini=%s) — no audio-orient source",
            connection_mode,
            host,
            "set" if mini is not None else "None",
        )
        return None

    return AzimuthDoASource(reader, name=name, agent_id=agent_id)
