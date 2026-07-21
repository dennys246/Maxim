"""P1: range-aware ``_normalize_value`` so signed sensors don't fold.

The legacy range-BLIND map collapses a ``[-1, 1]`` sensor: center ``0.0`` aliases
hard-left ``-1.0`` (both -> 0.0), and the fold at 0 collides opposite-signed
values (``-0.5`` and ``+0.25`` both -> 0.25). For a signed azimuth that means a
left sound and a right sound can share one EC cluster, so the orient policy can't
condition on direction (the full-range right-side 0.729 in the motor-credit
probe). Supplying the sensor's ``(lo, hi)`` range maps it monotonically.

These tests pin BOTH: (1) the legacy no-range path stays byte-identical (so
callers that don't pass ranges — every pre-P1 caller — are unaffected), and
(2) the range-aware path resolves direction.
"""

from __future__ import annotations

from maxim.similarity.encoder import _normalize_value, _sensor_embed


def _cos(a, b):
    import math

    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


# ── _normalize_value ─────────────────────────────────────────────────────────


def test_legacy_no_range_is_byte_identical_including_the_fold():
    # The pre-P1 behavior, pinned so a future edit can't silently change it.
    assert _normalize_value(-1.0) == 0.0
    assert _normalize_value(0.0) == 0.0  # center aliases hard-left (the bug)
    assert _normalize_value(-0.5) == 0.25
    assert _normalize_value(0.25) == 0.25  # collides with -0.5 (the fold)
    assert _normalize_value(0.5) == 0.5
    assert _normalize_value(1.0) == 1.0


def test_range_aware_signed_is_monotonic_no_fold():
    r = (-1.0, 1.0)
    vals = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.7, 1.0]
    mapped = [_normalize_value(v, r) for v in vals]
    # strictly increasing → no collisions, no fold
    assert all(b > a for a, b in zip(mapped, mapped[1:])), mapped
    assert _normalize_value(0.0, r) == 0.5  # center is the midpoint, not 0.0
    assert _normalize_value(-1.0, r) == 0.0 and _normalize_value(1.0, r) == 1.0
    # the legacy collision is resolved
    assert _normalize_value(-0.5, r) != _normalize_value(0.25, r)


def test_range_zero_one_is_identity():
    for v in (0.0, 0.3, 0.5, 0.8, 1.0):
        assert _normalize_value(v, (0.0, 1.0)) == v


def test_range_aware_clips_out_of_range():
    r = (-1.0, 1.0)
    assert _normalize_value(-5.0, r) == 0.0
    assert _normalize_value(5.0, r) == 1.0


def test_degenerate_range_clamps():
    assert _normalize_value(0.5, (1.0, 1.0)) == 0.5
    assert _normalize_value(-3.0, (2.0, 2.0)) == 0.0


# ── _sensor_embed: left vs right must separate with a range, collide without ──


def test_left_and_right_azimuth_separate_only_with_range():
    left = {"azimuth": -0.5}
    right = {"azimuth": 0.25}  # legacy: both normalize to 0.25 -> identical embed
    # Legacy (no range): the fold makes these two embeddings identical (cos ~= 1).
    assert _cos(_sensor_embed(left), _sensor_embed(right)) > 0.9999
    # Range-aware: distinct normalized values -> distinct embeddings.
    ranges = {"azimuth": (-1.0, 1.0)}
    cos_ranged = _cos(_sensor_embed(left, ranges), _sensor_embed(right, ranges))
    assert cos_ranged < 0.99, cos_ranged


def test_center_distinct_from_hard_left_only_with_range():
    center = {"azimuth": 0.0}
    hard_left = {"azimuth": -1.0}
    # Legacy: both -> 0.0 -> identical embedding (center indistinguishable from left).
    assert _cos(_sensor_embed(center), _sensor_embed(hard_left)) > 0.9999
    ranges = {"azimuth": (-1.0, 1.0)}
    assert _cos(_sensor_embed(center, ranges), _sensor_embed(hard_left, ranges)) < 0.99


def test_zero_one_sensor_embedding_unchanged_by_supplying_its_range():
    # Supplying (0,1) for a [0,1] sensor must not change its embedding.
    s = {"hunger": 0.8}
    assert _sensor_embed(s) == _sensor_embed(s, {"hunger": (0.0, 1.0)})


# ── P1 review fold: two-walk drift guard ─────────────────────────────────────


def test_read_drive_ranges_covers_every_signed_drive():
    """_read_drive_ranges and _read_drive_states walk the same drive_specs; if a
    future edit makes a (signed) drive appear in one walk but not the other, that
    sensor silently RE-FOLDS. Pin agreement on the real reachy + infant bodies:
    every drive the value-walk emits (minus the derived, sensorless `cold` need)
    must have a range, and every declared signed sensor must be covered."""
    from maxim.embodiment.body import Embodiment
    from maxim.embodiment.component_registry import ComponentRegistry
    from maxim.runtime.agent_loop import _read_drive_ranges, _read_drive_states

    class _Exec:
        def __init__(self, emb):
            self.embodiment = emb

    for body_ref in ("bodies/infant_humanoid_chilled", "bodies/reachy_mini"):
        ex = _Exec(Embodiment(root=ComponentRegistry().instantiate(body_ref)))
        drives = _read_drive_states(ex)
        ranges = _read_drive_ranges(ex)
        # Every drive that gets a VALUE must also get a RANGE, or it silently
        # re-folds — except the derived, sensorless "cold" need (setdefault'd in
        # _read_drive_states only when there is no real cold drive; it has no
        # sensor/range and correctly falls back to the legacy [0,1] map).
        missing = set(drives) - set(ranges) - {"cold"}
        assert not missing, f"{body_ref}: drives with a value but no range (would fold): {missing}"
        # and there IS at least one signed drive covered (else the fix is inert here)
        assert any(lo < 0 for lo, _ in ranges.values()), f"{body_ref}: no signed drive got a range"


def test_read_drive_ranges_skips_malformed_range_without_disabling():
    """A malformed range (non-iterable scalar / non-numeric bounds) must NOT
    raise — it is evaluated inside the encode_sensors try/except, so a raise would
    silently disable ALL substrate encoding for the agent every tick. The bad
    sensor is skipped (falls back to the legacy map); good ones still get ranges."""
    from maxim.runtime.agent_loop import _read_drive_ranges

    class _Sensor:
        def __init__(self, rng):
            self.reading_schema = {"range": rng}

    class _Ent:
        def __init__(self, sensors, specs):
            self.sensors = sensors
            self.drive_specs = specs
            self.modulators = {}

        def walk(self):
            return [self]

    class _Emb:
        def __init__(self, root):
            self.root = root

    class _Exec:
        def __init__(self, emb):
            self.embodiment = emb

    root = _Ent(
        sensors={"good": _Sensor([-1, 1]), "bad_scalar": _Sensor(5), "bad_str": _Sensor(["a", "b"])},
        specs={"good": object(), "bad_scalar": object(), "bad_str": object()},
    )
    ranges = _read_drive_ranges(_Exec(_Emb(root)))  # must not raise
    assert ranges == {"good": (-1.0, 1.0)}  # malformed skipped, good kept
