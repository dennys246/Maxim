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
        salience: float = 0.5,
        novelty: float = 0.3,
    ) -> None:
        self._reader = doa_reader
        self._name = name
        self._agent_id = agent_id
        # Attention weight the emitted percept carries. The factory defaults
        # (0.5 / 0.3) sit AT or BELOW every `> 0.5` attention/escalation gate in
        # the pipeline (perception_agent, exec_agent proposal + attention,
        # context_pool novelty), so a default DoA percept is passively perceived
        # but never proactively attended. These are the experiment knobs for
        # "does scaling audio salience/novelty change behavior?" — raise them to
        # push the sound above the attention thresholds.
        self._salience = salience
        self._novelty = novelty

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> set[str]:
        return {"audio"}

    def is_exhausted(self) -> bool:
        # Live hardware source — never exhausted.
        return False

    def has_pending(self) -> bool:
        """Always True — an ambient live sensor must be sampled every tick.

        DoA has no queued backlog; the reader yields a reading (or None) per
        tick, so the agent loop's idle-sleep must NOT skip the tick or the
        sensor is starved (it would never emit). ``CompositePerceptSource``
        treats a missing ``has_pending`` as True already; declaring it makes the
        ambient-sample semantics explicit rather than accidental (S2). Audio is
        a sim-only channel (orchestrator, behind ``MAXIM_SIM_AUDIO_ORIENT``), so
        this never affects the production loop.
        """
        return True

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
        return make_audio_percept(
            azimuth,
            source=self._name,
            agent_id=self._agent_id,
            salience=self._salience,
            novelty=self._novelty,
        )


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


# ─────────────────────────────────────────────────────────────────────────────
# Thalamic-relay consumption + sim wiring (thalamus_relay_design_pass.md stage 4)
#
# These make an audio/DoA percept *recognized in the agentic loop*:
#   - ``format_audio_orientation`` renders a passive azimuth observation that the
#     loop folds into the auto-sense (passive-perception) prompt channel;
#   - ``build_audio_composite`` attaches an ``AzimuthDoASource`` to an existing
#     percept source via ``CompositePerceptSource`` (the first-slice multiplexer);
#   - ``default_sim_doa_reader`` is a deterministic synthetic reader so the path
#     is demonstrable offline (no hardware), standing in for the live onboard-DoA
#     feed until Layer 2 / the motor repair.
# ─────────────────────────────────────────────────────────────────────────────


def format_audio_orientation(percept: object) -> str:
    """Render a passive azimuth observation from an audio/DoA percept.

    Returns ``""`` for ``None``, non-audio percepts, or a percept without an
    azimuth — so callers can unconditionally fold the result into the passive-
    perception (auto-sense) channel. The azimuth convention matches
    :func:`make_audio_percept`: ``-1`` = left, ``0`` = centered, ``+1`` = right.
    """
    if percept is None:
        return ""
    meta = getattr(percept, "metadata", None) or {}
    if "azimuth" not in meta:
        return ""
    # Belt + suspenders: only render for a SOUND-modality percept, so an
    # ``azimuth`` key riding some unrelated percept's metadata is ignored.
    sensory = getattr(percept, "sensory", None)
    modality = getattr(sensory, "modality", None)
    if modality is not None and getattr(modality, "value", modality) != "sound":
        return ""
    try:
        az = float(meta["azimuth"])
    except (TypeError, ValueError):
        return ""
    az = max(-1.0, min(1.0, az))
    if abs(az) <= 0.1:
        return "You hear a sound directly ahead of you (centered, azimuth 0.00)."
    side = "left" if az < 0 else "right"
    magnitude = "slightly" if abs(az) <= 0.5 else "well"
    return f"You hear a sound {magnitude} to your {side} (azimuth {az:+.2f})."


def audio_attention_profile(salience: float, novelty: float) -> "dict[str, float | bool]":
    """Which pipeline attention/escalation gates a percept at (salience, novelty)
    would pass — the per-run trace that quantifies the salience A/B.

    Every gate below is a strict ``>`` comparison in the pipeline; the reference
    is the consuming call site. Emitted as the ``data`` of each ``audio-orient``
    sim-log record so an ablation can ask "did the hot arm's percepts actually
    clear the gates the baseline's didn't, and did behavior follow?" without
    re-deriving the thresholds by hand. The default DoA weights (0.5 / 0.3) pass
    NONE of these — the sound is perceived (via §1.16 auto-sense) but never
    proactively attended.
    """
    s = float(salience)
    n = float(novelty)
    return {
        "salience": s,
        "novelty": n,
        # perception_agent.py:261/357 — salience > 0.5 marks a percept salient
        "passes_salience_gate": s > 0.5,
        # context_pool.py:268 + exec_agent.py:1737 — novelty > 0.5
        "passes_novelty_gate": n > 0.5,
        # exec_agent.py:981/1709 — BOTH > 0.5 triggers a proposal / attention
        "passes_proposal_gate": s > 0.5 and n > 0.5,
        # memory_agent.py:734 — novelty > 0.7 stores as a high-novelty memory
        "passes_high_novelty_memory": n > 0.7,
    }


def should_emit_orientation(
    prev_az: "float | None",
    new_az: float,
    *,
    threshold: float = 0.15,
) -> bool:
    """Change-gate for the §1.16 audio-orient prompt line.

    Returns True when the direction is worth re-announcing: the first sound
    (``prev_az is None``) or an azimuth that moved at least ``threshold`` from
    the last announced one. Suppresses re-emitting an identical "sound to your
    left" line on every tick — pure prompt noise that also burns context. Kept
    minimal (a delta gate, not a refractory timer) so it stays deterministic and
    loop-clock-free; a timed refractory is the coordinator's job if ever needed.
    """
    if prev_az is None:
        return True
    try:
        return abs(float(new_az) - float(prev_az)) >= threshold
    except (TypeError, ValueError):
        return True


def build_audio_composite(
    base_source: object,
    doa_reader: DoAReader,
    *,
    name: str = "reachy:audio-doa",
    agent_id: str | None = None,
    composite_name: str = "sim+audio",
    salience: float = 0.5,
    novelty: float = 0.3,
) -> object:
    """Multiplex ``base_source`` with an ``AzimuthDoASource(doa_reader)``.

    Returns a :class:`~maxim.simulation.composite_source.CompositePerceptSource`
    whose audio child is **ambient** (perpetual-live) so it never blocks the
    base (scripted/interactive) source from terminating the sim. ``salience`` /
    ``novelty`` set the attention weight of the emitted audio percepts (the
    "does scaling audio salience matter" experiment knob). The composite import
    is lazy to avoid a module-level embodiment→simulation dependency.
    """
    from maxim.simulation.composite_source import CompositePerceptSource

    audio = AzimuthDoASource(doa_reader, name=name, agent_id=agent_id, salience=salience, novelty=novelty)
    return CompositePerceptSource([base_source, audio], ambient=[audio], name=composite_name)


def sim_audio_salience_novelty() -> "tuple[float, float]":
    """Resolve the sim audio-orient (salience, novelty) from env, for the
    scaling experiment. ``MAXIM_SIM_AUDIO_SALIENCE`` / ``MAXIM_SIM_AUDIO_NOVELTY``
    override the sub-threshold factory defaults (0.5 / 0.3); malformed or unset
    values fall back to the defaults. Clamped to ``[0, 1]``."""
    import os

    def _resolve(var: str, default: float) -> float:
        raw = os.environ.get(var)
        if not raw:
            return default
        try:
            return max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            logger.warning("%s=%r is not a float — using default %.2f", var, raw, default)
            return default

    return _resolve("MAXIM_SIM_AUDIO_SALIENCE", 0.5), _resolve("MAXIM_SIM_AUDIO_NOVELTY", 0.3)


def default_sim_doa_reader(
    *,
    period: int = 40,
    max_events: "int | None" = None,
    angles_rad: "list[float] | None" = None,
) -> DoAReader:
    """Deterministic synthetic DoA reader for offline sims (no hardware).

    Emits ``(angle, is_speech=True)`` once every ``period`` calls, cycling
    through ``angles_rad``; ``(0.0, False)`` otherwise. A stand-in for the live
    onboard-DoA feed (:func:`make_reachy_rest_doa_reader`) or a future
    sim-narrative-driven angle source — it exists to prove the recognition path
    end-to-end offline, NOT as a permanent input. Deterministic (no RNG) so
    experiment runs are reproducible.

    Cadence (calibrated from the first live run, 2026-07-18): ``period`` counts
    reader *calls*, and the agent loop calls it at its tick rate (~2 Hz), so the
    default ``period=40`` yields a sound event roughly every ~20 s — an
    occasional EVENT, not the every-2 s ambient hum the first demo produced
    (which swamped the agent and confounded any behavioral read). ``max_events``
    caps the total number of sound events (``None`` = unbounded); pass a small
    value for a controlled stimulus set that goes silent afterward.
    """
    seq = (
        list(angles_rad)
        if angles_rad is not None
        else [
            math.radians(30),  # left of front  (doa 30° → az < 0)
            math.radians(150),  # right of front (doa 150° → az > 0)
            math.radians(75),  # near-centered
        ]
    )
    counter = {"n": 0, "i": 0, "emitted": 0}

    def _reader() -> "DoAReading | None":
        counter["n"] += 1
        if max_events is not None and counter["emitted"] >= max_events:
            return (0.0, False)
        if period > 0 and counter["n"] % period == 0 and seq:
            angle = seq[counter["i"] % len(seq)]
            counter["i"] += 1
            counter["emitted"] += 1
            return (angle, True)
        return (0.0, False)

    return _reader


def audio_orient_enabled() -> bool:
    """Whether the offline audio-orient channel is opted in for this sim.

    Reads ``MAXIM_SIM_AUDIO_ORIENT`` via the canonical truthy parser. Default
    OFF → the orchestrator wiring is byte-identical. Experiment/harness toggle
    (env, not config) — attaching a live DoA reader is a separate wiring
    decision. Paired with an autouse conftest scrub per the hot-path rule.
    """
    import os

    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    return annotation_disabled_via_env(os.environ.get("MAXIM_SIM_AUDIO_ORIENT"))
