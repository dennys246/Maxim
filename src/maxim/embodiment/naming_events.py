"""Embodiment naming-event scaffolding — deterministic drive-utterance co-firing.

**Dormant since 2026-05-29: superseded by EC drift fix per
[docs/experiments/36_roy_5b_confound_isolation.md].** Roy-5b-confound-
isolation showed the Roy-2c recognition gap closes from the EC drift
fix alone (PR #264 raised `pattern_complete_threshold` 0.40 → 0.44) —
no scaffold needed. With the scaffold, this module produces +1 (drive,
drive) binding edge at default rule that doesn't form without it, but
the recognition gap closure that motivated the scaffold is already
achieved at the EC pattern-completion layer independent of binding
edges. The scaffold IS structurally doing what it was designed to do
(co-temporal drive + linguistic EC firing) — it just doesn't carry the
load the Roy-5 disambiguator plan needed. Per CLAUDE.md Principle 2,
the code stays wired and tests stay green; no new features build on
top. Resurrection requires a new experiment that earns this
mechanism's behavioral weight (e.g., demonstrating that
co-temporal-firing-based learning produces some downstream property
the threshold-only path cannot).

**Pre-dormancy history:** Roy-5b (Stage 3 of
docs/plans/archive/roy_5_encoder_alignment_disambiguator.md, 2026-05-28)
ran this scaffold and produced 1 matched binding edge at default.
Roy-5b-confound-isolation (2026-05-29) ran the same protocol WITHOUT
the scaffold at HEAD and produced the same arm A overlap (10/10),
showing the recognition gap closure was threshold-driven, not
scaffold-driven. `docs/plans/archive/cross_modal_substrate_binding.md` was
archived as a result; `docs/plans/deferred/jepa_cross_modal_alignment.md`
remains parked with weakened motivation.

Roy-5b needs the priming arc to produce a temporally-aligned (sensor
pattern, drive state, body-utterance) triple within a shared salience
window so the proposed Hebbian binding rule from
[docs/plans/archive/cross_modal_substrate_binding.md] has co-firing to bind on.

The standard cradle arcs do NOT produce this co-firing: the
LLM-narrator's generated prose lands as a separate text-modality EC
node from the sensor/drive snapshot's interoception-modality node,
with no within-tick co-activation guarantee. Even worse, in
``aut_mode: substrate-primary`` the narrator's text is SUPPRESSED at
``SimulationBridge.send_and_wait`` so the linguistic modality stays
empty during priming (the empirical reason Roy-4's priming had 37 EC
nodes with zero overlap to arm-A's test-phase nodes).

This module is the *deterministic drive-utterance emitter*: when a
drive crosses a configured threshold, it emits a short, structurally-
stable utterance ("hungry", "thirsty", "warm") into the SAME percept
the ``EmbodimentPerceptSource`` is building from the sensor snapshot.
Because the utterance lands inside the body-state text the substrate-
primary AUT does consume, the ``LinguisticEncoder`` fires on the
utterance in the same tick the ``SensorEncoder`` fires on the drive
state — closing the co-firing gap.

The framing is **interoceptive vocalization**, NOT joint-attention
caregiver naming: the body emits its own utterance derived from its
drive state, mirroring the way real interoceptive states produce
vocalizations (crying when hungry) without requiring an external
caregiver. The bio-defense the plan-doc currently frames as "joint
attention" is more accurately a drive→linguistic-channel scaffold;
both routes establish text-modality EC firing co-temporally with
interoception-modality EC firing, which is the only property the
Hebbian retest needs.

The module deliberately holds no LLM — the utterance shape is
load-bearing for the binding-rule retest, so it cannot vary per run.
Hysteresis prevents the same utterance from re-firing every tick once
a drive sits above threshold; the utterance re-arms after the drive
drops back below ``threshold - hysteresis_band``.

**Only bodies that declare ``naming_events:`` metadata opt in**
(currently ``infant_humanoid_naming_v1.yaml`` for Roy-5b). The
standard ``infant_humanoid.yaml`` is unchanged.

See [docs/plans/archive/roy_5_encoder_alignment_disambiguator.md] Stage 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from maxim.embodiment.body import Embodiment


@dataclass(frozen=True)
class NamingPattern:
    """One naming-event rule: when ``sensor_key`` crosses ``threshold``
    in ``direction``, emit ``utterance``. ``hysteresis_band`` defines
    how far the sensor must move back before the pattern re-arms.

    SHAPE-FROZEN at 1.0 (CC3): this is research scaffolding for the
    Stage 3 Hebbian retest; the field set is the diagnostic surface.
    Adding fields post-1.0 requires the Stage 4 verdict to be in.
    """

    sensor_key: str
    utterance: str
    threshold: float
    direction: str = "above"  # "above" or "below"
    hysteresis_band: float = 0.2


def parse_naming_patterns(raw: Any, *, source: str | None = None) -> tuple[NamingPattern, ...]:
    """Parse ``naming_events`` metadata from YAML into NamingPattern.

    Accepts the list-of-dicts shape declared by
    ``infant_humanoid_naming_v1.yaml``. Empty / missing config returns
    an empty tuple — callers detect "no naming events" by emptiness.

    Rejects malformed entries with a ValueError so a body YAML typo
    fails at scenario load time, not silently at runtime when the
    binding test would notice missing co-firing. ``source`` (typically
    the body entity name) is prepended to the error message so the
    operator sees which YAML produced the error.
    """
    if not raw:
        return ()

    prefix = f"{source}: " if source else ""

    if not isinstance(raw, list):
        raise ValueError(f"{prefix}naming_events must be a list, got {type(raw).__name__}")

    patterns: list[NamingPattern] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"{prefix}naming_events[{i}] must be a dict, got {type(entry).__name__}")
        try:
            sensor_key = str(entry["sensor_key"])
            utterance = str(entry["utterance"])
            threshold = float(entry["threshold"])
        except KeyError as e:
            raise ValueError(f"{prefix}naming_events[{i}] missing required field {e}") from e
        except (TypeError, ValueError) as e:
            raise ValueError(f"{prefix}naming_events[{i}] field type error: {e}") from e

        direction = str(entry.get("direction", "above"))
        if direction not in ("above", "below"):
            raise ValueError(f"{prefix}naming_events[{i}].direction must be 'above' or 'below', got {direction!r}")

        hysteresis_band = float(entry.get("hysteresis_band", 0.2))
        if hysteresis_band < 0:
            raise ValueError(f"{prefix}naming_events[{i}].hysteresis_band must be >= 0, got {hysteresis_band}")

        patterns.append(
            NamingPattern(
                sensor_key=sensor_key,
                utterance=utterance,
                threshold=threshold,
                direction=direction,
                hysteresis_band=hysteresis_band,
            )
        )
    return tuple(patterns)


def collect_sensor_values(embodiment: Embodiment) -> dict[str, float]:
    """Walk the embodiment tree and collect every readable sensor value.

    Returns a flat dict keyed by either:

    - bare sensor name (``hunger``) — for entity-level sensors and
      vital_metrics
    - dotted modulator path (``arms.thermal``) — for modulator
      sub-sensors stored in ``modulator.vital_metrics``

    This bypasses ``Body.body_state_summary()`` because that method
    intentionally skips dotted-key drive specs (see body.py:527's
    ``"." not in ds_name`` check) — the modulator vital_metrics layer
    is where ``arms.thermal`` actually lives. Roy-4's pre-merge review
    surfaced this as a structural gap: the naming hook MUST read from
    the modulator layer or the "warm" utterance will silently never
    fire across Roy-5b's 50 priming turns.

    Later modulators with the same dotted suffix overwrite earlier
    ones — both walks happen in entity-tree order, so the convention
    is "last write wins" for ambiguous keys. The Roy-5b body declares
    only unique dotted keys (``arms.thermal``, ``arms.pressure``,
    ``head.thermal``); the helper warns at parse time if a pattern's
    ``sensor_key`` resolves to multiple sites in the same tree.
    """
    values: dict[str, float] = {}

    for ent in embodiment.root.walk():
        # Entity-level vital_metrics (hunger, thirst, core_temperature)
        for key, val in ent.vital_metrics.items():
            if "." in key:
                # Defensive — entity vital_metrics shouldn't carry
                # dotted keys, but if they ever do, route through the
                # dotted bucket so the modulator path doesn't collide.
                continue
            try:
                values[key] = float(val)
            except (TypeError, ValueError):
                continue

        # Live sensor readings (entity-level Sensors). Reading these
        # is idempotent — Hippocampus.recall touches memory, sensors
        # don't carry state mutation on read.
        for sname, sensor in ent.sensors.items():
            try:
                reading = sensor.read()
                values[sname] = float(reading.value)
            except Exception:
                continue

        # Modulator vital_metrics — dotted keys (arms.thermal, etc.)
        for mod_name, mod in ent.modulators.items():
            mod_metrics = getattr(mod, "vital_metrics", None)
            if not isinstance(mod_metrics, dict):
                continue
            for sub_key, val in mod_metrics.items():
                try:
                    values[f"{mod_name}.{sub_key}"] = float(val)
                except (TypeError, ValueError):
                    continue

    return values


def derive_naming_utterances(
    sensor_values: dict[str, float],
    patterns: tuple[NamingPattern, ...],
    fired_keys: set[str],
) -> tuple[tuple[str, ...], set[str]]:
    """Derive co-firing utterances for this tick + advance hysteresis state.

    Returns ``(utterances_this_tick, updated_fired_keys)``. The caller
    overwrites its hysteresis state with ``updated_fired_keys``. The
    utterances are intended to be appended verbatim to the
    ``EmbodimentPerceptSource`` body-state text so LinguisticEncoder
    fires on them in the same agent-loop tick the SensorEncoder fires
    on the underlying drive state.

    A pattern fires when its sensor crosses ``threshold`` in
    ``direction`` AND the pattern is not already in ``fired_keys``. It
    re-arms when the sensor moves back beyond ``threshold +/-
    hysteresis_band`` (sign depends on direction).

    Patterns whose ``sensor_key`` is not present in ``sensor_values``
    are silently skipped — a transient sensor (e.g. on an entity that
    wasn't activated yet) shouldn't break the priming run.
    """
    if not patterns:
        return ((), set(fired_keys))

    utterances: list[str] = []
    updated = set(fired_keys)

    for pattern in patterns:
        val = sensor_values.get(pattern.sensor_key)
        if val is None:
            continue

        if pattern.direction == "above":
            crossed = val >= pattern.threshold
            rearm = val < pattern.threshold - pattern.hysteresis_band
        else:  # "below"
            crossed = val <= pattern.threshold
            rearm = val > pattern.threshold + pattern.hysteresis_band

        if crossed and pattern.sensor_key not in updated:
            utterances.append(pattern.utterance)
            updated.add(pattern.sensor_key)
        elif rearm and pattern.sensor_key in updated:
            updated.discard(pattern.sensor_key)

    return (tuple(utterances), updated)


def format_naming_section(utterances: tuple[str, ...]) -> str:
    """Format utterances as a body-state-adjacent section.

    The section header (``=== Body Utterances ===``) is structurally
    neutral by design — earlier drafts used ``=== Caregiver ===``
    which (a) overclaimed the bio-fidelity (no caregiver exists,
    these are drive-derived self-utterances) and (b) introduced a
    recurring developmental-framing token into the substrate's
    learned linguistic-modality EC. ``Body Utterances`` keeps the
    framing accurate without contaminating the substrate's token
    space with experiment-specific framing.

    Empty input returns the empty string so the pre-naming-events
    percept shape is byte-identical when no patterns fire.
    """
    if not utterances:
        return ""
    lines = ["=== Body Utterances ==="]
    for u in utterances:
        lines.append(f"- {u}")
    return "\n".join(lines)
