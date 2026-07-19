"""Sim-side wiring for the audio-orient channel (thalamic-relay stage 4).

Keeps the sim/harness concerns — the synthetic DoA reader, the env-driven
enable + salience/novelty knobs, and the composite-attach factory — OUT of the
hardware-agnostic front-end (``embodiment/audio_localization.py``), which should
stay pure localization + rendering (per the pre-merge Architecture review). This
module freely imports from both ``embodiment`` (the front-end) and ``simulation``
(the multiplexer), which is the natural dependency direction.

These are experiment/harness toggles (env, not config.json) — attaching a live
DoA reader is a separate wiring decision; the ``MAXIM_SIM_AUDIO_*`` vars are
scrubbed by the autouse conftest fixture per the hot-path rule.
"""

from __future__ import annotations

import logging
import math
import os

from maxim.embodiment.audio_localization import AzimuthDoASource, DoAReader, DoAReading

logger = logging.getLogger(__name__)


def audio_orient_enabled() -> bool:
    """Whether the offline audio-orient channel is opted in for this sim.

    Reads ``MAXIM_SIM_AUDIO_ORIENT`` via the canonical truthy parser. Default
    OFF → the orchestrator wiring is byte-identical.
    """
    from maxim.prompts.cluster_bias_annotation import annotation_disabled_via_env

    return annotation_disabled_via_env(os.environ.get("MAXIM_SIM_AUDIO_ORIENT"))


def sim_audio_salience_novelty() -> "tuple[float, float]":
    """Resolve the sim audio-orient (salience, novelty) from env, for the
    scaling experiment. ``MAXIM_SIM_AUDIO_SALIENCE`` / ``MAXIM_SIM_AUDIO_NOVELTY``
    override the sub-threshold factory defaults (0.5 / 0.3); malformed or unset
    values fall back to the defaults. Clamped to ``[0, 1]``.

    Note: with the review's escalation-on-salience fix, a salience of exactly
    the default 0.5 is *sub-threshold* (the attention gates are strict ``> 0.5``)
    so the default channel is perceived-but-not-escalated — raise salience above
    0.5 for the sound to reach the LLM.
    """

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
    onboard-DoA feed (``make_reachy_rest_doa_reader``) or a future
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
    "does scaling audio salience matter" experiment knob).
    """
    from maxim.simulation.composite_source import CompositePerceptSource

    audio = AzimuthDoASource(doa_reader, name=name, agent_id=agent_id, salience=salience, novelty=novelty)
    return CompositePerceptSource([base_source, audio], ambient=[audio], name=composite_name)
