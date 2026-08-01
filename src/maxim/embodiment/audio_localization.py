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
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

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


def gated_azimuth(
    reader: DoAReader,
    *,
    k: int = 3,
    timeout_s: float = 5.0,
    poll_s: float = 0.15,
    stop_event: "object | None" = None,
) -> float | None:
    """Median of ``k`` speech-gated azimuth samples, or ``None`` on timeout.

    Promoted from ``scripts/orient_backbone/live_common.py`` (Stage 1d of
    live_audio_orient_wiring.md) — the measured DoA characterization
    justifies library status: the speech gate fires only 23-100% (median
    50%) per utterance, so raw single reads are too noisy for credit.

    Gates on the hardware's ``is_speech_detected`` flag — a transient clap
    or silence never fabricates a direction. Median-of-``k`` smooths
    single-read DoA noise so ``potential_diff`` credits the turn, not the
    measurement jitter.

    BLOCKS for up to ``timeout_s`` (sleeps between polls) — call it from a
    dedicated poll thread (the Stage-2 feed), NEVER from
    ``PerceptSource.next_percept()`` or any agent-loop tick path (the
    non-blocking contract, ``sources.py``). ``stop_event`` (an object with
    ``is_set()``/``wait(t)``, i.e. a ``threading.Event``) makes the sampling
    window interruptible so runtime teardown never waits out ``timeout_s``;
    returns ``None`` immediately when set.
    """
    samples: list[float] = []
    deadline = time.time() + timeout_s
    while len(samples) < k and time.time() < deadline:
        if stop_event is not None and stop_event.is_set():  # type: ignore[attr-defined]
            return None
        try:
            reading = reader()
        except Exception:
            logger.debug("gated_azimuth: doa_reader raised", exc_info=True)
            reading = None
        if reading is not None:
            doa_rad, is_speech = reading
            if is_speech:
                samples.append(doa_to_azimuth(float(doa_rad)))
        if stop_event is not None:
            stop_event.wait(poll_s)  # type: ignore[attr-defined]
        else:
            time.sleep(poll_s)
    if not samples:
        return None
    samples.sort()
    return samples[len(samples) // 2]


class DoAFeed:
    """Live DoA → body-sensor feed (live_audio_orient_wiring.md Stage 2).

    Decouples the DoA transport's synchronous read (the REST reader does a
    blocking GET) from every consumer: ``run()`` — executed in a thread the
    WIRING layer owns (CaptureManager pattern: shared ``stop_event``,
    registered in the runtime teardown sweep) — repeatedly samples
    :func:`gated_azimuth`, and on each FRESH speech-gated reading:

    1. caches ``(azimuth, timestamp)`` under a lock (consumers read
       :attr:`latest` — never the transport);
    2. world-sets the body's ``azimuth`` sensor via
       :func:`world_set_azimuth` (the drive/EC/body_state lane); and
    3. hands ``make_audio_percept(az)`` to the optional ``percept_sink``
       (the Stage-3 modality-preserving percept lane — e.g.
       ``NullSimulationAdapter``'s side-channel slot).

    Silence writes NOTHING: a stale direction is never re-written, so a
    world-set sensor with ``drift_rate: 0`` holds the last real measurement
    between utterances (the ``reachy_mini.yaml`` contract).

    This is deliberately NOT a ``PerceptSource`` — attaching one on the
    live path would flip ``is_sim_mode`` across its 12 consumer sites.
    It is also not a background DSP thread inside ``AzimuthDoASource``
    (whose pull-per-tick contract is unchanged); it is the wiring-layer
    poll thread the ``PerceptSource`` non-blocking invariant demands.
    """

    def __init__(
        self,
        reader: DoAReader,
        embodiment: object,
        *,
        stop_event: threading.Event,
        percept_sink: "Callable[[Percept], None] | None" = None,
        agent_id: str | None = None,
        source_name: str = "reachy:audio-doa",
        k: int = 3,
        sample_timeout_s: float = 2.0,
        sample_poll_s: float = 0.15,
        salience: float = 0.5,
        novelty: float = 0.3,
    ) -> None:
        self._reader = reader
        self._embodiment = embodiment
        self._stop_event = stop_event
        self._percept_sink = percept_sink
        self._agent_id = agent_id
        self._source_name = source_name
        self._k = k
        self._sample_timeout_s = sample_timeout_s
        self._sample_poll_s = sample_poll_s
        # Attention weights the emitted percepts carry — same experiment
        # knobs as AzimuthDoASource: the defaults (0.5/0.3) sit AT or BELOW
        # every > 0.5 escalation gate, so a live sound is passively
        # perceived but never escalates to a submission until an operator
        # raises them (per-run decision, mirroring the sim knobs).
        self._salience = salience
        self._novelty = novelty
        self._lock = threading.Lock()
        self._latest: "tuple[float, float] | None" = None  # (azimuth, monotonic ts)
        # Claim the sensor as live-world-owned (pre-merge review fold): the
        # modeled self_effect on ``azimuth`` must not write/credit while a
        # real measurement stream owns the sensor — a modeled shift the next
        # reading reverts would book relief for actuation that never
        # happened, repeatably (the phantom credit mill). See
        # ModulatorAffordanceTool.execute + Embodiment.live_world_set_sensors.
        owned = getattr(embodiment, "live_world_set_sensors", None)
        if owned is not None:
            owned.add("azimuth")

    @property
    def latest(self) -> "tuple[float, float] | None":
        """Most recent gated ``(azimuth, timestamp)``, or None before first speech."""
        with self._lock:
            return self._latest

    def run(self) -> None:
        """Poll loop — run inside a dedicated thread until ``stop_event`` sets.

        ``gated_azimuth`` paces the loop internally (~6.7 Hz sampling; the
        chip converges in 0.23 s, faster would be waste) and returns
        ``None`` per silent window, so quiet rooms cost no writes.
        """
        warned_once = False
        while not self._stop_event.is_set():
            # Per-iteration guard (pre-merge review fold): an unexpected
            # raise must not silently kill the thread and leave audio dark
            # for the rest of the session. First failure logs a WARNING;
            # repeats stay at DEBUG. gated_azimuth paces the loop, so a
            # persistent failure cannot spin.
            try:
                az = gated_azimuth(
                    self._reader,
                    k=self._k,
                    timeout_s=self._sample_timeout_s,
                    poll_s=self._sample_poll_s,
                    stop_event=self._stop_event,
                )
                if self._stop_event.is_set():
                    return
                if az is None:
                    continue
                with self._lock:
                    self._latest = (az, time.monotonic())
                # Concurrent-writer note: this plain set races the loop
                # thread's RMW sensor writes on OTHER keys only — the
                # live_world_set_sensors filter removes ``azimuth`` from
                # modeled self_effect application, so this feed is the
                # sensor's single writer; a lost update is impossible.
                if not world_set_azimuth(self._embodiment, az):
                    logger.debug("DoAFeed: body has no azimuth sensor — value cached only")
                if self._percept_sink is not None:
                    try:
                        self._percept_sink(
                            make_audio_percept(
                                az,
                                source=self._source_name,
                                agent_id=self._agent_id,
                                salience=self._salience,
                                novelty=self._novelty,
                            )
                        )
                    except Exception:
                        logger.debug("DoAFeed: percept_sink raised", exc_info=True)
            except Exception:
                if not warned_once:
                    logger.warning("DoAFeed: iteration raised; feed continues", exc_info=True)
                    warned_once = True
                else:
                    logger.debug("DoAFeed: iteration raised again", exc_info=True)


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
# Thalamic-relay consumption helpers (thalamus_relay_design_pass.md stage 4)
#
# The runtime loop's §1.16 consumes an audio/DoA percept through these:
#   - ``format_audio_orientation`` renders a passive azimuth observation the loop
#     folds into the auto-sense (passive-perception) prompt channel;
#   - ``audio_attention_profile`` reports which attention gates the percept clears
#     (the salience-A/B trace);
#   - ``should_emit_orientation`` is the change-gate (skip an unchanged direction).
# The sim-side wiring (synthetic reader, env knobs, composite-attach) lives in
# ``simulation/audio_orient_wiring.py`` — kept out of this hardware-agnostic
# front-end per the pre-merge Architecture review.
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


@dataclass(frozen=True)
class OrientingProfile:
    """Per-entity sound-reactivity + orient-limit config — extend by DATA.

    Declared in a body's YAML under a top-level ``orienting:`` key (which
    ``_parse_entity`` flows into ``entity.metadata``); absent → these defaults.
    Lets an ``elderly_humanoid`` raise its thresholds (reduced sensitivity) or a
    limited robot declare how far it can physically orient — all without
    per-body code. The three tiers (thalamus_hypothalamus_framing three-tier
    gate): below ``escalate_above`` the sound is ignored; above it the agent
    CHOOSES to attend (reaches the LLM); above BOTH reflex thresholds it triggers
    an AUTOMATIC orient (superior-colliculus startle, bypasses the LLM).

    Runtime-ephemeral config (resolved from the body each use; never persisted or
    wire-crossing), so out of the CC3 frozen-dataclass audit scope — new fields
    append with defaults.
    """

    escalate_above: float = 0.5  # perception: deliberative tier (reaches the LLM)
    reflex_salience_above: float = 0.9  # perception: reflex tier — loud
    reflex_novelty_above: float = 0.9  # perception: reflex tier — sudden
    max_orient_azimuth: float = 0.0  # motor: closest-to-center it can orient (0 = full reach)


_DEFAULT_ORIENTING = OrientingProfile()


def resolve_orienting_profile(embodiment: object) -> OrientingProfile:
    """Resolve a body's :class:`OrientingProfile` from its declared ``orienting:``
    metadata, else the default. Data-driven — no per-body code."""
    root = getattr(embodiment, "root", None)
    meta = getattr(root, "metadata", None) or {}
    cfg = meta.get("orienting")
    if not isinstance(cfg, dict):
        return _DEFAULT_ORIENTING
    d = _DEFAULT_ORIENTING

    def _f(key: str, default: float) -> float:
        try:
            return float(cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    return OrientingProfile(
        escalate_above=_f("escalate_above", d.escalate_above),
        reflex_salience_above=_f("reflex_salience_above", d.reflex_salience_above),
        reflex_novelty_above=_f("reflex_novelty_above", d.reflex_novelty_above),
        max_orient_azimuth=max(0.0, min(1.0, _f("max_orient_azimuth", d.max_orient_azimuth))),
    )


def is_audio_escalation(salience: float, profile: OrientingProfile = _DEFAULT_ORIENTING) -> bool:
    """Deliberative tier: is the sound salient enough to reach the LLM (the agent
    can then CHOOSE to listen / orient)?"""
    return float(salience) > profile.escalate_above


def is_orienting_reflex(salience: float, novelty: float, profile: OrientingProfile = _DEFAULT_ORIENTING) -> bool:
    """Reflex tier: loud AND sudden enough to trigger an AUTOMATIC orient.

    The top tier — a bang, not a background hum — triggers a reflexive turn
    toward the sound (superior-colliculus startle), bypassing cortical
    deliberation. Keyed on BOTH salience (loud) and novelty (sudden): a steady
    loud hum is not startling. Thresholds are per-entity (``profile``) so
    sensitivity is data-driven, not a hard 0.9 across the board.
    """
    return float(salience) > profile.reflex_salience_above and float(novelty) > profile.reflex_novelty_above


def reflex_oriented_azimuth(azimuth: float, profile: OrientingProfile = _DEFAULT_ORIENTING) -> float:
    """Where the body ends up after a reflexive orient toward ``azimuth``.

    Turns to face the sound (azimuth → 0), but clamped to the body's physical
    reach: a body with ``max_orient_azimuth > 0`` cannot get closer than that,
    so a sound farther out leaves a residual offset (it turned as far as it
    physically can). Data-driven physical limit — no per-body code."""
    reach = profile.max_orient_azimuth
    az = float(azimuth)
    if reach > 0.0 and abs(az) > reach:
        return reach if az > 0 else -reach
    return 0.0


def world_set_azimuth(embodiment: object, azimuth: float) -> bool:
    """World-set the body's ``azimuth`` root sensor from a DoA percept.

    Dimensionality note: orientation here is deliberately **1-D (horizontal
    azimuth)** — reachy's XVF3800 gives azimuth only (no elevation, and
    front/back ambiguous). Extension to 2-D (elevation/altitude, e.g. a
    spherical mic array or vision-driven ``head_pitch``) is a clean, mechanical
    generalization — axis-parameterize this + ``reflex_oriented_azimuth`` +
    ``OrientingProfile.max_orient_azimuth`` + the ``metadata["azimuth"]`` reads
    in agent_loop §1.16, add ``turn_up``/``turn_down`` affordances, and pick the
    2-D magnitude model. NOT done now (one axis = N=1; would force guessing the
    magnitude model). Full plan + the one open design decision:
    docs/plans/productive_orienting_affordance.md § "Dimensionality".


    Capability-driven: ANY body that declares an ``azimuth`` root sensor gets
    sound localization — no per-body code. Writes the clamped value into the
    root entity's ``vital_metrics`` (what ``SpecSensor.read()`` returns), so the
    ``listen`` affordance's sensor read-back surfaces the current direction and
    the agent can act on it. Returns True if the body has the sensor and it was
    written; False (fail-soft) for a body without sound localization — it is
    simply unaffected. Mirrors the real robot, where live DoA world-sets the
    same sensor (Track 2 Layer 2); the sim write is the offline dry-run.

    One-line delegate to :func:`world_set_axis` — the axis-generic form this
    docstring's dimensionality note asks for (Stage 2 of
    live_audio_orient_wiring.md picked this parameterization).
    """
    return world_set_axis(embodiment, "azimuth", azimuth, default_range=(-1.0, 1.0))


def world_set_axis(
    embodiment: object,
    sensor_name: str,
    value: float,
    *,
    default_range: "tuple[float, float] | None" = None,
) -> bool:
    """World-set any declared root sensor from an exteroceptive measurement.

    The axis-generic helper behind :func:`world_set_azimuth` (Stage 2,
    live_audio_orient_wiring.md): a future elevation axis is one body-YAML
    sensor + one call here — NO new type, NO axis config schema (the
    perception_pipeline_placement.md:127 guardrail: if an ``AxisSpec`` type
    starts to appear, stop and use the body YAML).

    Clamps to the sensor's declared ``reading_schema["range"]`` when one is
    resolvable, else to ``default_range`` when given, else writes the raw
    float (a world-set write must not invent a range the body didn't
    declare). Returns True if the body declares the sensor and it was
    written; False fail-soft otherwise.
    """
    root = getattr(embodiment, "root", None)
    if root is None:
        return False
    sensors = getattr(root, "sensors", None) or {}
    if sensor_name not in sensors:
        return False
    vm = getattr(root, "vital_metrics", None)
    if vm is None:
        return False
    v = float(value)
    lo_hi: "tuple[float, float] | None" = None
    schema = getattr(sensors[sensor_name], "reading_schema", None)
    rng = schema.get("range") if isinstance(schema, dict) else None
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        try:
            lo, hi = float(rng[0]), float(rng[1])
            if lo <= hi:
                lo_hi = (lo, hi)
        except (TypeError, ValueError):
            lo_hi = None
    if lo_hi is None:
        lo_hi = default_range
    if lo_hi is not None:
        v = max(lo_hi[0], min(lo_hi[1], v))
    vm[sensor_name] = v
    return True


def audio_attention_profile(salience: float, novelty: float) -> "dict[str, float | bool]":
    """Which pipeline attention/escalation gates a percept at (salience, novelty)
    would pass — the per-run trace that quantifies the salience A/B.

    Every gate below is a strict ``>`` comparison in the pipeline. Emitted as the
    ``data`` of each ``audio-orient`` sim-log record so an ablation can ask "did
    the hot arm's percepts actually clear the gates the baseline's didn't, and
    did behavior follow?" without re-deriving the thresholds by hand. The default
    DoA weights (0.5 / 0.3) pass NONE of these — the sound is perceived (via
    §1.16 auto-sense) but never proactively attended.

    The module:line references in the comments below are **illustrative, not
    authoritative** — the upstream gates are bare inline literals (nothing to
    import), so re-verify against source if this diagnostic ever disagrees with
    observed behavior.
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
